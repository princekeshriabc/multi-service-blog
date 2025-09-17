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

How to use

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
