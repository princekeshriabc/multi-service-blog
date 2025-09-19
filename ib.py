from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import re
import pandas as pd


# -----------------------------------------------------------
# Data models (reuse if already defined in your project)
# -----------------------------------------------------------

@dataclass
class HeaderInfo:
    header_row_indices: List[int]                  # 0-based indices of header rows
    header_valid: bool = True                      # your detector’s validity flag
    header_names: Optional[List[str]] = None       # as-is (not normalized)
    debug: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TableInput:
    df: pd.DataFrame
    page_index: int                                # 1-based page number (or logical order)
    table_index_on_page: int = 1
    header_info: Optional[HeaderInfo] = None
    # merged_cells: list of dicts with keys row_index, column_index, row_span, column_span (1-based indices)
    merged_cells: Optional[List[Dict[str, Any]]] = None


# =========================
# Enums and dataclasses for Pattern B
# =========================

class TablePattern(str, Enum):
    B = "B"  # Continuation (headers on first page only)

@dataclass
class PatternBConfig:
    # Grid regularity thresholds (same idea as Pattern A)
    max_row_colcount_deviation_ratio: float = 0.2
    min_data_rows_for_analysis: int = 5

    # Body merges not allowed for B (like A)
    max_body_merged_anchors: int = 0

    # Continuation markers (optional heuristic)
    continuation_tokens: List[str] = field(default_factory=lambda: [
        "continued", "carried forward", "carry forward", "carry fwd", "to be continued"
    ])
    # Page marker tokens often seen around footer/header text
    page_marker_tokens: List[str] = field(default_factory=lambda: ["page", "of"])

    # How many top/bottom rows to scan for markers
    scan_top_rows_for_markers: int = 2
    scan_bottom_rows_for_markers: int = 2

    # B requires a reference page with header
    require_reference_header: bool = True

    # Columns consistency with reference (strict)
    require_same_mode_cols_as_reference: bool = True

    # Allow partial headers on continuation pages (as per edge cases)
    allow_partial_headers_on_continuation: bool = True
    partial_header_max_columns: int = 2  # e.g., small “Description | Amount” partial header

@dataclass
class TableBFeatures:
    is_reference_page: bool
    has_header: bool
    header_rows: List[int]                # 0-based (as detected)
    header_names_raw: List[str]           # as-is (not normalized)
    header_cols_count: int
    data_start_row: int                   # 0-based first data row
    data_rows_count: int
    data_mode_cols: int
    row_colcount_variance_ratio: float    # fraction of data rows deviating from mode
    body_merged_anchors: int
    continuation_markers_found: bool
    reasons: List[str] = field(default_factory=list)

@dataclass
class PageBPatternResult:
    page_index: int
    table_index_on_page: int
    is_type_b: bool
    confidence: float
    reasons: List[str]
    features: TableBFeatures

@dataclass
class DocumentBResult:
    is_type_b_document: bool
    confidence: float
    page_results: List[PageBPatternResult]
    reference_page_index: Optional[int]
    reference_mode_cols: Optional[int]
    reasons: List[str]


# =========================
# Utilities
# =========================

def _squeeze(s: str) -> str:
    return " ".join((s or "").split())

def _nonempty_count(row: List[Any]) -> int:
    return sum(1 for x in row if _squeeze(str(x)) != "")

def _mode(values: List[int]) -> int:
    if not values:
        return 0
    from collections import Counter
    return Counter(values).most_common(1)[0][0]

def _extract_header_names_raw(df: pd.DataFrame, header_rows: List[int], header_names: Optional[List[str]]) -> List[str]:
    if header_names:
        return [str(x) for x in header_names]
    if header_rows:
        last_idx = header_rows[-1]
        if 0 <= last_idx < len(df):
            return [_squeeze(str(x)) for x in df.iloc[last_idx].tolist()]
    return []

def _compute_data_start_row(header_rows: List[int]) -> int:
    return (max(header_rows) + 1) if header_rows else 0

def _count_body_merged_anchors(merged_cells: Optional[List[Dict[str, Any]]], data_start_row_0based: int) -> int:
    if not merged_cells:
        return 0
    count = 0
    for m in merged_cells:
        r = int(m.get("row_index") or m.get("RowIndex") or 1)
        rs = int(m.get("row_span") or m.get("RowSpan") or 1)
        cs = int(m.get("column_span") or m.get("ColumnSpan") or 1)
        r0 = r - 1
        if r0 >= data_start_row_0based and (rs > 1 or cs > 1):
            count += 1
    return count

def _analyze_data_region(df: pd.DataFrame, data_start_row: int) -> Tuple[int, int, float, List[int]]:
    """
    Returns:
        data_rows_count, data_mode_cols, row_colcount_variance_ratio, per_row_nonempty_counts
    """
    if df.empty or data_start_row >= len(df):
        return 0, 0, 0.0, []

    nonempty_counts: List[int] = []
    for i in range(data_start_row, len(df)):
        row = df.iloc[i].tolist()
        nonempty_counts.append(_nonempty_count(row))

    analyzed = [c for c in nonempty_counts if c > 0]
    data_rows_count = len(analyzed)
    if data_rows_count == 0:
        return 0, 0, 0.0, nonempty_counts

    m = _mode(analyzed)
    deviations = sum(1 for c in analyzed if c != m)
    variance_ratio = deviations / data_rows_count
    return data_rows_count, m, variance_ratio, nonempty_counts

def _scan_for_continuation_markers(df: pd.DataFrame, cfg: PatternBConfig) -> bool:
    tokens = set([t.lower() for t in (cfg.continuation_tokens + cfg.page_marker_tokens)])
    rows_to_scan = []

    # Top N rows
    for i in range(min(cfg.scan_top_rows_for_markers, len(df))):
        rows_to_scan.append(df.iloc[i].tolist())
    # Bottom N rows
    for i in range(max(0, len(df) - cfg.scan_bottom_rows_for_markers), len(df)):
        rows_to_scan.append(df.iloc[i].tolist())

    for r in rows_to_scan:
        text = _squeeze(" ".join([str(x) for x in r])).lower()
        if not text:
            continue
        for t in tokens:
            if t in text:
                return True
    return False


# =========================
# Feature extraction for Pattern B
# =========================

def extract_table_b_features(
    table: TableInput,
    cfg: PatternBConfig,
    reference_page_index: Optional[int],
    reference_mode_cols: Optional[int]
) -> TableBFeatures:
    df = table.df
    hi = table.header_info or HeaderInfo(header_row_indices=[], header_valid=False, header_names=None)

    has_header = bool(hi.header_row_indices) and hi.header_valid
    header_rows = hi.header_row_indices if has_header else []
    header_names_raw = _extract_header_names_raw(df, header_rows, hi.header_names if has_header else None)
    header_cols_count = len(header_names_raw) if has_header else 0

    data_start_row = _compute_data_start_row(header_rows)
    data_rows_count, data_mode_cols, variance_ratio, _ = _analyze_data_region(df, data_start_row)
    body_merged_anchors = _count_body_merged_anchors(table.merged_cells, data_start_row)
    continuation_markers_found = _scan_for_continuation_markers(df, cfg)

    is_reference_page = (reference_page_index is not None and table.page_index == reference_page_index)

    reasons: List[str] = []
    if is_reference_page:
        reasons.append("This page is the reference (first valid header page).")
    if has_header:
        reasons.append(f"Header present: rows={header_rows}.")
    else:
        reasons.append("No header present on this page.")

    reasons.append(f"Data rows: {data_rows_count}, mode non-empty cols: {data_mode_cols}, variance ratio: {variance_ratio:.2f}.")
    reasons.append(f"Body merged anchors: {body_merged_anchors}.")
    if continuation_markers_found:
        reasons.append("Continuation markers found near top/bottom.")

    return TableBFeatures(
        is_reference_page=is_reference_page,
        has_header=has_header,
        header_rows=header_rows,
        header_names_raw=header_names_raw,
        header_cols_count=header_cols_count,
        data_start_row=data_start_row,
        data_rows_count=data_rows_count,
        data_mode_cols=data_mode_cols,
        row_colcount_variance_ratio=variance_ratio,
        body_merged_anchors=body_merged_anchors,
        continuation_markers_found=continuation_markers_found,
        reasons=reasons,
    )


# =========================
# Page-level Pattern B
# =========================

def classify_page_pattern_b(
    table: TableInput,
    cfg: Optional[PatternBConfig],
    reference_page_index: Optional[int],
    reference_mode_cols: Optional[int]
) -> PageBPatternResult:
    cfg = cfg or PatternBConfig()
    feats = extract_table_b_features(table, cfg, reference_page_index, reference_mode_cols)

    reasons = list(feats.reasons)

    # Can only be a continuation if it's NOT the reference page
    not_reference = (reference_page_index is not None and table.page_index != reference_page_index)

    # Header condition
    header_condition = False
    if not feats.has_header:
        header_condition = True  # typical Pattern B
        reasons.append("Header condition OK: no header on continuation page.")
    elif cfg.allow_partial_headers_on_continuation and (feats.header_cols_count <= cfg.partial_header_max_columns):
        header_condition = True
        reasons.append(f"Header condition OK: partial header allowed (cols={feats.header_cols_count}).")
    else:
        reasons.append("Header condition failed (full header present).")

    # Grid regularity
    grid_regular = feats.row_colcount_variance_ratio <= cfg.max_row_colcount_deviation_ratio
    if grid_regular:
        reasons.append("Grid regularity OK.")
    else:
        reasons.append("Grid irregular (too many rows deviate from mode).")

    # Body merges
    no_body_merges = feats.body_merged_anchors <= cfg.max_body_merged_anchors
    if no_body_merges:
        reasons.append("No body merges (OK).")
    else:
        reasons.append("Found body merges (not allowed for Type B).")

    # Column count alignment with reference
    cols_match_reference = True
    if cfg.require_same_mode_cols_as_reference:
        if reference_mode_cols is None or reference_mode_cols == 0:
            cols_match_reference = False
            reasons.append("No reference mode column count available; cannot verify columns match.")
        else:
            cols_match_reference = (feats.data_mode_cols == reference_mode_cols)
            if cols_match_reference:
                reasons.append(f"Data mode cols match reference ({reference_mode_cols}).")
            else:
                reasons.append(f"Data mode cols ({feats.data_mode_cols}) differ from reference ({reference_mode_cols}).")

    # Enough data rows
    enough_rows = feats.data_rows_count >= cfg.min_data_rows_for_analysis
    if enough_rows:
        reasons.append("Data rows sufficient.")
    else:
        reasons.append("Few data rows for robust decision.")

    # Final decision for page-level B
    is_type_b = (
        not_reference and
        header_condition and
        grid_regular and
        no_body_merges and
        cols_match_reference and
        enough_rows
    )

    # Confidence: base from strong signals
    confidence = 0.5
    if not_reference:
        confidence += 0.1
    if header_condition:
        confidence += 0.2
    if grid_regular:
        confidence += 0.1
    if no_body_merges:
        confidence += 0.05
    if cols_match_reference:
        confidence += 0.1
    if feats.continuation_markers_found:
        confidence += 0.05
    if not enough_rows:
        confidence -= 0.1

    confidence = max(0.0, min(1.0, confidence))
    if is_type_b:
        reasons.insert(0, "Classified as Type B (continuation page).")
    else:
        reasons.insert(0, "Not Type B.")

    return PageBPatternResult(
        page_index=table.page_index,
        table_index_on_page=table.table_index_on_page,
        is_type_b=is_type_b,
        confidence=confidence,
        reasons=reasons,
        features=feats,
    )


# =========================
# Document-level Pattern B
# =========================

def classify_document_pattern_b(
    tables: List[TableInput],
    cfg: Optional[PatternBConfig] = None
) -> DocumentBResult:
    """
    Document is considered Pattern B if:
      - There is a valid reference page (first page with header).
      - Reference page acts as schema origin.
      - One or more subsequent pages classify as Type B (continuations).
    """
    cfg = cfg or PatternBConfig()

    if not tables:
        return DocumentBResult(
            is_type_b_document=False,
            confidence=0.0,
            page_results=[],
            reference_page_index=None,
            reference_mode_cols=None,
            reasons=["No tables provided."],
        )

    # Find reference page: the first page with a valid header
    reference_page_index: Optional[int] = None
    reference_mode_cols: Optional[int] = None

    for t in tables:
        hi = t.header_info or HeaderInfo(header_row_indices=[], header_valid=False, header_names=None)
        if hi.header_valid and hi.header_row_indices:
            data_start = (max(hi.header_row_indices) + 1) if hi.header_row_indices else 0
            data_rows_count, data_mode_cols, _, _ = _analyze_data_region(t.df, data_start)
            reference_page_index = t.page_index
            reference_mode_cols = data_mode_cols
            break

    reasons: List[str] = []
    if reference_page_index is not None:
        reasons.append(f"Found reference page at page_index={reference_page_index} with mode cols={reference_mode_cols}.")
    else:
        reasons.append("No reference page with header found.")
        if cfg.require_reference_header:
            return DocumentBResult(
                is_type_b_document=False,
                confidence=0.0,
                page_results=[],
                reference_page_index=None,
                reference_mode_cols=None,
                reasons=reasons + ["Reference header is required for Pattern B document."],
            )

    # Page-level B classification using the reference
    page_results: List[PageBPatternResult] = []
    for t in tables:
        pr = classify_page_pattern_b(
            t, cfg, reference_page_index=reference_page_index, reference_mode_cols=reference_mode_cols
        )
        page_results.append(pr)

    # Decide document-level Pattern B:
    continuation_pages_b = [pr for pr in page_results if pr.is_type_b]
    has_b_pages = len(continuation_pages_b) > 0
    is_type_b_document = (reference_page_index is not None) and has_b_pages

    if is_type_b_document:
        reasons.insert(0, "Document classified as Type B (continuation pages detected).")
    else:
        reasons.insert(0, "Document not classified as Type B.")

    # Confidence: based on fraction of continuation pages (after reference) that qualify as B
    total_after_ref = sum(1 for t in tables if reference_page_index is not None and t.page_index != reference_page_index)
    frac_b = (len(continuation_pages_b) / total_after_ref) if total_after_ref > 0 else 0.0

    # Scale confidence: average of B-page confidences adjusted by coverage
    if continuation_pages_b:
        avg_b_conf = sum(pr.confidence for pr in continuation_pages_b) / len(continuation_pages_b)
    else:
        avg_b_conf = 0.0
    doc_confidence = max(0.0, min(1.0, 0.5 * avg_b_conf + 0.5 * frac_b))

    return DocumentBResult(
        is_type_b_document=is_type_b_document,
        confidence=doc_confidence,
        page_results=page_results,
        reference_page_index=reference_page_index,
        reference_mode_cols=reference_mode_cols,
        reasons=reasons,
    )


# You provide a list of TableInput objects (per table/page) with:
# df: pandas DataFrame containing the table (header=None when reading CSVs)
# header_info: your detected header rows/validity (no normalization)
# merged_cells: optional list of merged cells with row_index/column_index/row_span/column_span (1-based)
# The code finds a reference page (first valid header), then classifies later pages as B if they look like continuations.

# How to use

# Build TableInput objects with your DataFrames, header detection results, and merged-cells list.
# Call classify_document_pattern_b(tables).
# Example

# After reading CSVs as df (header=None) and running your header detector:

tables = [TableInput(df=df1, page_index=1, header_info=HeaderInfo([0], True)),
TableInput(df=df2, page_index=2, header_info=HeaderInfo([], False))]

result = classify_document_pattern_b(tables)

print(result.is_type_b_document, round(result.confidence, 3))

for pr in result.page_results:

    print(pr.page_index, pr.is_type_b, round(pr.confidence, 3))
    print(result.reasons)



# IF every page has header
#     IF headers are identical → Type A
#     ELSE → Type C
# ELSE IF only first page has header
#     IF subsequent rows align with same column count → Type B
#     ELSE → Type D
# ELSE (mix of some pages with headers, some without)
#     → Type E

def detect_pattern(textract_json):
    headers = extract_headers_per_page(textract_json)
    first_header = headers[0]
    
    header_flags = []
    for h in headers:
        if h: header_flags.append(1)
        else: header_flags.append(0)

    # Case 1: All pages have headers
    if all(header_flags):
        if all(similar(h, first_header) for h in headers):
            return "Type A"
        else:
            return "Type C"

    # Case 2: Only first page header
    if header_flags[0] == 1 and all(f == 0 for f in header_flags[1:]):
        if check_column_alignment(headers, first_header):
            return "Type B"
        else:
            return "Type D"

    # Case 3: Mixed headers
    return "Type E"


def normalize_tables(textract_json, ib_type):
    if ib_type == "Type A":
        return merge_drop_repeat_headers(textract_json)
    elif ib_type == "Type B":
        return propagate_first_header(textract_json)
    elif ib_type == "Type C":
        return unify_with_header_mapping(textract_json)
    elif ib_type == "Type D":
        return align_with_positions(textract_json)
    elif ib_type == "Type E":
        return normalize_pagewise(textract_json)


from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import re
import pandas as pd


# -----------------------------------------------------------
# Data models (reuse/import these if already defined)
# -----------------------------------------------------------

@dataclass
class HeaderInfo:
    header_row_indices: List[int]                  # 0-based indices of header rows
    header_valid: bool = True                      # your detector’s validity flag
    header_names: Optional[List[str]] = None       # as-is (not normalized)
    debug: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TableInput:
    df: pd.DataFrame
    page_index: int                                # 1-based page number (or logical order)
    table_index_on_page: int = 1
    header_info: Optional[HeaderInfo] = None
    # merged_cells: list of dicts with keys row_index, column_index, row_span, column_span (1-based indices)
    merged_cells: Optional[List[Dict[str, Any]]] = None


# =========================
# Enums and dataclasses for Pattern C
# =========================

class TablePattern(str, Enum):
    C = "C"  # Inconsistent headers across pages

@dataclass
class PatternCConfig:
    # Header inconsistency thresholds (no normalization; only trim/lower for compare)
    header_jaccard_min_for_consistency: float = 0.8
    header_order_min_for_consistency: float = 0.8

    # Grid regularity thresholds (structured table)
    max_row_colcount_deviation_ratio: float = 0.25
    min_data_rows_for_analysis: int = 5

    # Allow a few merges in body but keep it mostly structured
    max_body_merged_anchors_for_structured: int = 2

@dataclass
class TableCFeatures:
    is_reference_page: bool
    has_header: bool
    header_rows: List[int]                # 0-based
    header_names_raw: List[str]           # as-is (not normalized)
    header_cols_count: int

    # Data/grid
    data_start_row: int
    data_rows_count: int
    data_mode_cols: int
    row_colcount_variance_ratio: float
    body_merged_anchors: int
    grid_regular: bool

    # Header comparisons to reference (if present)
    jaccard_similarity_to_ref: Optional[float]
    order_similarity_to_ref: Optional[float]
    column_count_match_ref: Optional[bool]

    reasons: List[str] = field(default_factory=list)

@dataclass
class PageCPatternResult:
    page_index: int
    table_index_on_page: int
    is_type_c: bool
    confidence: float
    reasons: List[str]
    features: TableCFeatures

@dataclass
class DocumentCResult:
    is_type_c_document: bool
    confidence: float
    page_results: List[PageCPatternResult]
    reference_page_index: Optional[int]
    reference_headers: Optional[List[str]]
    header_similarity_by_page: List[Tuple[int, Optional[float], Optional[float], Optional[bool]]]  # (page, jaccard, order, colcount_match)
    reasons: List[str]


# =========================
# Utilities
# =========================

def _squeeze(s: str) -> str:
    return " ".join((s or "").split())

def _nonempty_count(row: List[Any]) -> int:
    return sum(1 for x in row if _squeeze(str(x)) != "")

def _mode(values: List[int]) -> int:
    if not values:
        return 0
    from collections import Counter
    return Counter(values).most_common(1)[0][0]

def _extract_header_names_raw(df: pd.DataFrame, header_rows: List[int], header_names: Optional[List[str]]) -> List[str]:
    if header_names:
        return [str(x) for x in header_names]
    if header_rows:
        last_idx = header_rows[-1]
        if 0 <= last_idx < len(df):
            return [_squeeze(str(x)) for x in df.iloc[last_idx].tolist()]
    return []

def _compute_data_start_row(header_rows: List[int]) -> int:
    return (max(header_rows) + 1) if header_rows else 0

def _count_body_merged_anchors(merged_cells: Optional[List[Dict[str, Any]]], data_start_row_0based: int) -> int:
    if not merged_cells:
        return 0
    count = 0
    for m in merged_cells:
        r = int(m.get("row_index") or m.get("RowIndex") or 1)
        rs = int(m.get("row_span") or m.get("RowSpan") or 1)
        cs = int(m.get("column_span") or m.get("ColumnSpan") or 1)
        r0 = r - 1
        if r0 >= data_start_row_0based and (rs > 1 or cs > 1):
            count += 1
    return count

def _analyze_data_region(df: pd.DataFrame, data_start_row: int) -> Tuple[int, int, float, List[int]]:
    """
    Returns:
        data_rows_count, data_mode_cols, row_colcount_variance_ratio, per_row_nonempty_counts
    """
    if df.empty or data_start_row >= len(df):
        return 0, 0, 0.0, []

    nonempty_counts: List[int] = []
    for i in range(data_start_row, len(df)):
        row = df.iloc[i].tolist()
        nonempty_counts.append(_nonempty_count(row))

    analyzed = [c for c in nonempty_counts if c > 0]
    data_rows_count = len(analyzed)
    if data_rows_count == 0:
        return 0, 0, 0.0, nonempty_counts

    m = _mode(analyzed)
    deviations = sum(1 for c in analyzed if c != m)
    variance_ratio = deviations / data_rows_count
    return data_rows_count, m, variance_ratio, nonempty_counts

def _normalize_for_compare(headers: List[str]) -> List[str]:
    # Only trim + lowercase for comparison (no synonym mapping)
    return [_squeeze(h).lower() for h in headers if _squeeze(h) != ""]

def _header_jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def _header_order_similarity(a: List[str], b: List[str]) -> float:
    # Position-based similarity: exact token match in the same position over max length
    n = max(len(a), len(b))
    if n == 0:
        return 1.0
    matches = 0
    for i in range(min(len(a), len(b))):
        if a[i] == b[i]:
            matches += 1
    return matches / n


# =========================
# Feature extraction
# =========================

def extract_table_c_features(
    table: TableInput,
    cfg: PatternCConfig,
    reference_page_index: Optional[int],
    reference_headers_raw: Optional[List[str]]
) -> TableCFeatures:
    df = table.df
    hi = table.header_info or HeaderInfo(header_row_indices=[], header_valid=False, header_names=None)

    has_header = bool(hi.header_row_indices) and hi.header_valid
    header_rows = hi.header_row_indices if has_header else []
    header_names_raw = _extract_header_names_raw(df, header_rows, hi.header_names if has_header else None)
    header_cols_count = len(header_names_raw) if has_header else 0

    data_start_row = _compute_data_start_row(header_rows)
    data_rows_count, data_mode_cols, variance_ratio, _ = _analyze_data_region(df, data_start_row)
    body_merged_anchors = _count_body_merged_anchors(table.merged_cells, data_start_row)
    grid_regular = variance_ratio <= cfg.max_row_colcount_deviation_ratio

    is_reference_page = (reference_page_index is not None and table.page_index == reference_page_index)

    # Header comparisons to reference
    jaccard = None
    order_sim = None
    colcount_match = None
    if has_header and reference_headers_raw is not None:
        ref_norm = _normalize_for_compare(reference_headers_raw)
        cur_norm = _normalize_for_compare(header_names_raw)
        jaccard = _header_jaccard(ref_norm, cur_norm)
        order_sim = _header_order_similarity(ref_norm, cur_norm)
        colcount_match = (len(header_names_raw) == len(reference_headers_raw))

    reasons: List[str] = []
    if is_reference_page:
        reasons.append("This page is the reference (first valid header page).")
    if has_header:
        reasons.append(f"Header present: rows={header_rows}.")
    else:
        reasons.append("No header present on this page.")

    reasons.append(f"Data rows: {data_rows_count}, mode non-empty cols: {data_mode_cols}, variance ratio: {variance_ratio:.2f}.")
    reasons.append(f"Body merged anchors: {body_merged_anchors}.")
    if grid_regular:
        reasons.append("Grid regularity OK.")
    else:
        reasons.append("Grid irregular.")

    if has_header and reference_headers_raw is not None:
        reasons.append(f"Header vs ref: jaccard={jaccard:.2f} order={order_sim:.2f} colcount_match={colcount_match}.")

    return TableCFeatures(
        is_reference_page=is_reference_page,
        has_header=has_header,
        header_rows=header_rows,
        header_names_raw=header_names_raw,
        header_cols_count=header_cols_count,
        data_start_row=data_start_row,
        data_rows_count=data_rows_count,
        data_mode_cols=data_mode_cols,
        row_colcount_variance_ratio=variance_ratio,
        body_merged_anchors=body_merged_anchors,
        grid_regular=grid_regular,
        jaccard_similarity_to_ref=jaccard,
        order_similarity_to_ref=order_sim,
        column_count_match_ref=colcount_match,
        reasons=reasons,
    )


# =========================
# Page-level Pattern C
# =========================

def classify_page_pattern_c(
    table: TableInput,
    cfg: Optional[PatternCConfig],
    reference_page_index: Optional[int],
    reference_headers_raw: Optional[List[str]]
) -> PageCPatternResult:
    cfg = cfg or PatternCConfig()
    feats = extract_table_c_features(table, cfg, reference_page_index, reference_headers_raw)

    reasons = list(feats.reasons)

    # Must have a header to be considered for C
    if not feats.has_header:
        reasons.insert(0, "Not Type C (no header; might be Type B or other).")
        return PageCPatternResult(
            page_index=table.page_index,
            table_index_on_page=table.table_index_on_page,
            is_type_c=False,
            confidence=0.2,
            reasons=reasons,
            features=feats,
        )

    # Reference page itself is not "inconsistent" by definition
    if feats.is_reference_page:
        reasons.insert(0, "Not Type C (reference page).")
        return PageCPatternResult(
            page_index=table.page_index,
            table_index_on_page=table.table_index_on_page,
            is_type_c=False,
            confidence=0.2,
            reasons=reasons,
            features=feats,
        )

    # If no reference headers available, we cannot assert inconsistency vs ref
    if reference_headers_raw is None:
        reasons.insert(0, "Not Type C (no reference headers available).")
        return PageCPatternResult(
            page_index=table.page_index,
            table_index_on_page=table.table_index_on_page,
            is_type_c=False,
            confidence=0.2,
            reasons=reasons,
            features=feats,
        )

    # Grid should be regular (structured table) to qualify as C (not D)
    structured_enough = feats.grid_regular and (feats.body_merged_anchors <= cfg.max_body_merged_anchors_for_structured)
    if not structured_enough:
        reasons.insert(0, "Not Type C (table not sufficiently structured; may be Type D).")
        return PageCPatternResult(
            page_index=table.page_index,
            table_index_on_page=table.table_index_on_page,
            is_type_c=False,
            confidence=0.3,
            reasons=reasons,
            features=feats,
        )

    # Inconsistency conditions vs reference
    inconsistent = False
    inconsistency_strength = 0.0

    if feats.column_count_match_ref is not None and not feats.column_count_match_ref:
        inconsistent = True
        inconsistency_strength += 0.5
        reasons.append("Header column count differs from reference (strong inconsistency).")

    if feats.jaccard_similarity_to_ref is not None and feats.jaccard_similarity_to_ref < cfg.header_jaccard_min_for_consistency:
        inconsistent = True
        gap = cfg.header_jaccard_min_for_consistency - feats.jaccard_similarity_to_ref
        inconsistency_strength += min(0.4, max(0.1, gap))
        reasons.append(f"Header set differs from reference (jaccard={feats.jaccard_similarity_to_ref:.2f} < {cfg.header_jaccard_min_for_consistency}).")

    if feats.order_similarity_to_ref is not None and feats.order_similarity_to_ref < cfg.header_order_min_for_consistency:
        inconsistent = True
        gap = cfg.header_order_min_for_consistency - feats.order_similarity_to_ref
        inconsistency_strength += min(0.3, max(0.05, gap))
        reasons.append(f"Header order differs from reference (order_sim={feats.order_similarity_to_ref:.2f} < {cfg.header_order_min_for_consistency}).")

    is_type_c = inconsistent

    # Confidence
    confidence = 0.4
    if inconsistent:
        confidence += 0.3  # base for inconsistency
        confidence += min(0.3, inconsistency_strength)  # scaled by how different
    if feats.data_rows_count >= cfg.min_data_rows_for_analysis:
        confidence += 0.05
    if feats.grid_regular:
        confidence += 0.05
    confidence = max(0.0, min(1.0, confidence))

    if is_type_c:
        reasons.insert(0, "Classified as Type C (inconsistent headers across pages).")
    else:
        reasons.insert(0, "Not Type C.")

    return PageCPatternResult(
        page_index=table.page_index,
        table_index_on_page=table.table_index_on_page,
        is_type_c=is_type_c,
        confidence=confidence,
        reasons=reasons,
        features=feats,
    )


# =========================
# Document-level Pattern C
# =========================

def classify_document_pattern_c(
    tables: List[TableInput],
    cfg: Optional[PatternCConfig] = None
) -> DocumentCResult:
    """
    Document is considered Pattern C if:
      - A reference header page exists.
      - At least one other page with a header is inconsistent with the reference (names, count, or order),
        while remaining structured (not noisy/corrupted).
    """
    cfg = cfg or PatternCConfig()

    if not tables:
        return DocumentCResult(
            is_type_c_document=False,
            confidence=0.0,
            page_results=[],
            reference_page_index=None,
            reference_headers=None,
            header_similarity_by_page=[],
            reasons=["No tables provided."],
        )

    # Find reference: first page with a valid header
    reference_page_index: Optional[int] = None
    reference_headers: Optional[List[str]] = None

    for t in tables:
        hi = t.header_info or HeaderInfo(header_row_indices=[], header_valid=False, header_names=None)
        if hi.header_valid and hi.header_row_indices:
            # Build reference headers from last header row or provided names
            reference_headers = _extract_header_names_raw(t.df, hi.header_row_indices, hi.header_names)
            reference_page_index = t.page_index
            break

    reasons: List[str] = []
    if reference_page_index is None:
        reasons.append("No reference header page found; cannot determine inconsistency.")
        return DocumentCResult(
            is_type_c_document=False,
            confidence=0.0,
            page_results=[],
            reference_page_index=None,
            reference_headers=None,
            header_similarity_by_page=[],
            reasons=reasons,
        )

    reasons.append(f"Reference header chosen from page_index={reference_page_index}.")

    # Page-level C classification
    page_results: List[PageCPatternResult] = []
    sim_list: List[Tuple[int, Optional[float], Optional[float], Optional[bool]]] = []

    for t in tables:
        pr = classify_page_pattern_c(
            t, cfg, reference_page_index=reference_page_index, reference_headers_raw=reference_headers
        )
        page_results.append(pr)
        sim_list.append((
            pr.page_index,
            pr.features.jaccard_similarity_to_ref,
            pr.features.order_similarity_to_ref,
            pr.features.column_count_match_ref
        ))

    # Document-level decision: at least one page flagged as C
    c_pages = [pr for pr in page_results if pr.is_type_c]
    is_type_c_document = len(c_pages) > 0

    if is_type_c_document:
        reasons.insert(0, "Document classified as Type C (inconsistent headers across pages).")
    else:
        reasons.insert(0, "Document not classified as Type C.")

    # Confidence: based on fraction of header pages (excluding reference) that are inconsistent
    header_pages = [pr for pr in page_results if pr.features.has_header and not pr.features.is_reference_page]
    denom = len(header_pages) if header_pages else 1
    frac_inconsistent = len(c_pages) / denom if denom > 0 else 0.0
    avg_c_conf = sum(pr.confidence for pr in c_pages) / len(c_pages) if c_pages else 0.0
    doc_confidence = max(0.0, min(1.0, 0.5 * avg_c_conf + 0.5 * frac_inconsistent))

    return DocumentCResult(
        is_type_c_document=is_type_c_document,
        confidence=doc_confidence,
        page_results=page_results,
        reference_page_index=reference_page_index,
        reference_headers=reference_headers,
        header_similarity_by_page=sim_list,
        reasons=reasons,
    )


# What it does

# Loads a Textract JSON (synchronous or asynchronous AnalyzeDocument TABLES output).
# Uses TRP to iterate pages → tables → rows → cells.
# Flags merged anchors where RowSpan > 1 or ColumnSpan > 1.
# Returns a mapping keyed by (page_index, table_index_on_page) → list of merged cell dicts:
# row_index, column_index, row_span, column_span, text, confidence, covers (list of covered positions).
# Optional: exports a CSV summary for auditing.

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Tuple, Optional

# pip install textract-trp
from trp import Document  # TRP v1


# -----------------------------
# Data models
# -----------------------------

@dataclass
class MergedCell:
    # 1-based indices aligned with Textract
    row_index: int
    column_index: int
    row_span: int
    column_span: int
    text: str = ""
    confidence: Optional[float] = None
    # All 1-based (row, col) grid positions covered by this merged cell
    covers: List[Tuple[int, int]] = field(default_factory=list)

def _covers_positions(r: int, c: int, rs: int, cs: int) -> List[Tuple[int, int]]:
    return [(rr, cc) for rr in range(r, r + rs) for cc in range(c, c + cs)]


# -----------------------------
# I/O helpers
# -----------------------------

def load_textract_json_any(path: str) -> Dict[str, Any]:
    """
    Load a Textract JSON. If 'path' is a directory, merges all *.json files' Blocks.
    Returns a dict with a single 'Blocks' list.
    """
    if os.path.isdir(path):
        blocks: List[Dict[str, Any]] = []
        for fname in sorted(os.listdir(path)):
            if not fname.lower().endswith(".json"):
                continue
            with open(os.path.join(path, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
                blocks.extend(data.get("Blocks", []))
        return {"Blocks": blocks}
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "Blocks" in data:
            return data
        elif isinstance(data, list):
            return {"Blocks": data}
        else:
            raise ValueError("Input does not look like a Textract response (missing 'Blocks').")


# -----------------------------
# Core extraction (TRP v1)
# -----------------------------

def extract_merged_cells_trp(textract_json: Dict[str, Any]) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """
    Parse Textract JSON via TRP and collect merged cells for each table.

    Returns:
        merged_by_table: dict keyed by (page_index, table_index_on_page) -> List[dict]
        Each dict item has:
            {
              "row_index": int, "column_index": int,
              "row_span": int, "column_span": int,
              "text": str, "confidence": float|None,
              "covers": List[Tuple[int,int]]   # 1-based (row, col) covered by the span
            }
    Notes:
        - TRP exposes cell.rowIndex, cell.columnIndex, cell.rowSpan, cell.columnSpan, cell.text, cell.confidence
        - A "merged cell" anchor is any cell with rowSpan>1 or columnSpan>1
        - All indices are 1-based to match Textract and your pattern detectors
    """
    doc = Document(textract_json)

    merged_by_table: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}

    for page_number, page in enumerate(doc.pages, start=1):
        for table_idx, table in enumerate(page.tables, start=1):
            key = (page_number, table_idx)
            merged_list: List[Dict[str, Any]] = []

            for row in table.rows:
                for cell in row.cells:
                    # TRP properties are already 1-based indices
                    r = int(getattr(cell, "rowIndex", 1) or 1)
                    c = int(getattr(cell, "columnIndex", 1) or 1)
                    rs = int(getattr(cell, "rowSpan", 1) or 1)
                    cs = int(getattr(cell, "columnSpan", 1) or 1)

                    if rs > 1 or cs > 1:
                        mc = MergedCell(
                            row_index=r,
                            column_index=c,
                            row_span=rs,
                            column_span=cs,
                            text=getattr(cell, "text", "") or "",
                            confidence=getattr(cell, "confidence", None),
                            covers=_covers_positions(r, c, rs, cs),
                        )
                        merged_list.append(asdict(mc))

            merged_by_table[key] = merged_list

    return merged_by_table


# -----------------------------
# Optional: CSV export for auditing
# -----------------------------

def export_merged_cells_summary_csv(
    merged_by_table: Dict[Tuple[int, int], List[Dict[str, Any]]],
    out_csv_path: str
) -> None:
    """
    Write a summary CSV with one row per merged cell.
    Columns: page_index, table_index_on_page, anchor_row, anchor_col, row_span, col_span, text, confidence, covers
    """
    import csv
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "page_index", "table_index_on_page",
            "anchor_row", "anchor_col", "row_span", "col_span",
            "text", "confidence", "covers"
        ])
        for (page_idx, tbl_idx), merged_list in merged_by_table.items():
            for m in merged_list:
                covers_str = ";".join([f"{r}:{c}" for (r, c) in m.get("covers", [])])
                w.writerow([
                    page_idx, tbl_idx,
                    m.get("row_index"), m.get("column_index"),
                    m.get("row_span"), m.get("column_span"),
                    m.get("text", ""), m.get("confidence", ""),
                    covers_str
                ])


# -----------------------------
# Glue: attach merged cells to your TableInput list
# -----------------------------

def attach_merged_cells_to_tables(
    tables: List[Dict[str, Any]],
    merged_by_table: Dict[Tuple[int, int], List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    If your tables are kept as dicts with 'page_index' and 'table_index_on_page',
    attach the merged_cells list to each. If you use TableInput dataclass, adapt accordingly.
    """
    for t in tables:
        key = (t["page_index"], t.get("table_index_on_page", 1))
        t["merged_cells"] = merged_by_table.get(key, [])
    return tables


# -----------------------------
# Example usage (CLI)
# -----------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract merged cells from AWS Textract JSON using TRP.")
    parser.add_argument("--input", required=True, help="Path to Textract JSON file or directory containing JSON shards.")
    parser.add_argument("--out-csv", default=None, help="Optional path to write merged_cells_summary.csv")
    parser.add_argument("--out-json", default=None, help="Optional path to write merged_cells.json (nested by page/table)")
    args = parser.parse_args()

    # Load Textract response (single or merged)
    data = load_textract_json_any(args.input)

    # Extract merged cells per table
    merged_by_table = extract_merged_cells_trp(data)

    # Print summary
    total_tables = len(merged_by_table)
    total_merged = sum(len(v) for v in merged_by_table.values())
    print(f"Detected {total_tables} table(s). Total merged cells: {total_merged}")
    for (page_idx, tbl_idx), lst in sorted(merged_by_table.items()):
        print(f"- Page {page_idx} Table {tbl_idx}: {len(lst)} merged cell(s)")

    # Optional CSV export
    if args.out_csv:
        export_merged_cells_summary_csv(merged_by_table, args.out_csv)
        print(f"Saved merged-cells summary CSV to: {args.out_csv}")

    # Optional JSON export (nested by page -> table)
    if args.out_json:
        nested: Dict[str, Dict[str, Any]] = {}
        for (p, t), lst in merged_by_table.items():
            nested.setdefault(str(p), {})[str(t)] = lst
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(nested, f, ensure_ascii=False, indent=2)
        print(f"Saved merged-cells JSON to: {args.out_json}")
