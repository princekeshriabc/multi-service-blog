# 🔹 1. Source Input

# AWS Textract JSON = our raw input.

# This JSON represents the entire PDF (all pages).

# Inside it, there is a "Blocks" list with objects like "PAGE", "TABLE", "CELL", "WORD" etc.

# 🔹 2. From JSON → Working Input for Detection

# We will parse the JSON once, and extract all tables across all pages.

# Each table will be converted into a DataFrame (rows × columns).

# So if the PDF has 10 tables (spread across 10 pages, or multiple per page), we will end up with a list of DataFrames.

# Example:

# tables = [
#     DataFrame1,  # Page 1, Table 1
#     DataFrame2,  # Page 2, Table 1
#     DataFrame3,  # Page 2, Table 2
#     ...
# ]

# 🔹 3. Input to Detection Code

# Instead of giving detection function one table at a time,
# 👉 we will give the detection code the entire list of DataFrames (representing the whole PDF).

# Something like:

# detect_pattern(list_of_dataframes)

# 🔹 4. Why this design?

# Consistency across whole PDF:
# If even one table breaks the condition of Pattern A, then the whole PDF is not Pattern A.

# Scalability:
# Each DataFrame is independent, but we only decide the final pattern after checking all tables together.

# Industrial optimization:

# Parsing Textract JSON once → less compute.

# Using Pandas DataFrames → efficient column/row checks.

# Detection functions only get clean, tabular inputs.

# Easy to extend when we add Patterns B, C, D, E.

# AWS Lambda readiness:

# Input = Textract JSON (from S3 trigger).

# Code extracts tables → list of DataFrames.

# Detection function runs on this list.

# Output = final pattern classification (one per PDF).


# Perfect 👍 Let’s do a mini end-to-end example for your case:

# A PDF (via Textract JSON) with 5 tables across pages.

# We’ll simulate a small Textract-like JSON (not full AWS output, just enough for testing).

# Code will:

# Parse JSON.

# Extract all tables.

# Convert them into Pandas DataFrames.

# Show you the list of tables.

# 🔹 Example Code


import pandas as pd

# --- Simulated Textract JSON with 5 small tables ---
# (Real Textract is more complex, but this mimics the structure for demo)
sample_json = {
    "Blocks": [
        # Page 1, Table 1
        {"BlockType": "TABLE", "Id": "t1", "Relationships": [{"Ids": ["c1", "c2", "c3", "c4"]}]},
        {"BlockType": "CELL", "Id": "c1", "RowIndex": 1, "ColumnIndex": 1, "Text": "Item"},
        {"BlockType": "CELL", "Id": "c2", "RowIndex": 1, "ColumnIndex": 2, "Text": "Price"},
        {"BlockType": "CELL", "Id": "c3", "RowIndex": 2, "ColumnIndex": 1, "Text": "Apple"},
        {"BlockType": "CELL", "Id": "c4", "RowIndex": 2, "ColumnIndex": 2, "Text": "10"},

        # Page 1, Table 2
        {"BlockType": "TABLE", "Id": "t2", "Relationships": [{"Ids": ["c5", "c6", "c7", "c8"]}]},
        {"BlockType": "CELL", "Id": "c5", "RowIndex": 1, "ColumnIndex": 1, "Text": "Name"},
        {"BlockType": "CELL", "Id": "c6", "RowIndex": 1, "ColumnIndex": 2, "Text": "Qty"},
        {"BlockType": "CELL", "Id": "c7", "RowIndex": 2, "ColumnIndex": 1, "Text": "Banana"},
        {"BlockType": "CELL", "Id": "c8", "RowIndex": 2, "ColumnIndex": 2, "Text": "5"},

        # Page 2, Table 1
        {"BlockType": "TABLE", "Id": "t3", "Relationships": [{"Ids": ["c9", "c10", "c11", "c12"]}]},
        {"BlockType": "CELL", "Id": "c9", "RowIndex": 1, "ColumnIndex": 1, "Text": "Product"},
        {"BlockType": "CELL", "Id": "c10", "RowIndex": 1, "ColumnIndex": 2, "Text": "Amount"},
        {"BlockType": "CELL", "Id": "c11", "RowIndex": 2, "ColumnIndex": 1, "Text": "Milk"},
        {"BlockType": "CELL", "Id": "c12", "RowIndex": 2, "ColumnIndex": 2, "Text": "20"},

        # Page 3, Table 1
        {"BlockType": "TABLE", "Id": "t4", "Relationships": [{"Ids": ["c13", "c14", "c15", "c16"]}]},
        {"BlockType": "CELL", "Id": "c13", "RowIndex": 1, "ColumnIndex": 1, "Text": "Service"},
        {"BlockType": "CELL", "Id": "c14", "RowIndex": 1, "ColumnIndex": 2, "Text": "Charge"},
        {"BlockType": "CELL", "Id": "c15", "RowIndex": 2, "ColumnIndex": 1, "Text": "Delivery"},
        {"BlockType": "CELL", "Id": "c16", "RowIndex": 2, "ColumnIndex": 2, "Text": "30"},

        # Page 3, Table 2
        {"BlockType": "TABLE", "Id": "t5", "Relationships": [{"Ids": ["c17", "c18", "c19", "c20"]}]},
        {"BlockType": "CELL", "Id": "c17", "RowIndex": 1, "ColumnIndex": 1, "Text": "Desc"},
        {"BlockType": "CELL", "Id": "c18", "RowIndex": 1, "ColumnIndex": 2, "Text": "Value"},
        {"BlockType": "CELL", "Id": "c19", "RowIndex": 2, "ColumnIndex": 1, "Text": "Tax"},
        {"BlockType": "CELL", "Id": "c20", "RowIndex": 2, "ColumnIndex": 2, "Text": "15"},
    ]
}

def extract_tables(textract_json):
    blocks = textract_json["Blocks"]

    # Index all cells by ID
    cell_map = {b["Id"]: b for b in blocks if b["BlockType"] == "CELL"}

    tables = []
    for block in blocks:
        if block["BlockType"] == "TABLE":
            rows = {}
            for rel in block.get("Relationships", []):
                for cid in rel.get("Ids", []):
                    cell = cell_map[cid]
                    r, c = cell["RowIndex"], cell["ColumnIndex"]
                    rows.setdefault(r, {})[c] = cell.get("Text", "")
            # convert dict -> dataframe
            max_cols = max(len(row) for row in rows.values())
            data = [[row.get(c, "") for c in range(1, max_cols+1)] for row in rows.values()]
            df = pd.DataFrame(data)
            tables.append(df)

    return tables

# --- Run ---
tables = extract_tables(sample_json)

print("Extracted tables (total):", len(tables))
for i, df in enumerate(tables, 1):
    print(f"\nTable {i}:\n", df)


# 🔹 Output (for our fake JSON)

# Extracted tables (total): 5

# Table 1:
#         0      1
# 0    Item  Price
# 1   Apple     10

# Table 2:
#        0    1
# 0   Name  Qty
# 1 Banana    5

# Table 3:
#         0       1
# 0  Product  Amount
# 1     Milk      20

# Table 4:
#       0       1
# 0 Service  Charge
# 1 Delivery     30

# Table 5:
#       0      1
# 0   Desc  Value
# 1    Tax     15


# Nice — below is a robust, industry-minded single-file implementation that detects Pattern A for an entire PDF (given its tables). I thought a lot about accuracy, edge-cases, and production-readiness:

# Accepts tables as list-of-lists (rows × cols) or as pandas.DataFrame objects (auto-converted).

# Applies normalization, robust header detection, numeric/type consistency checks, totals filtering, grouping for split tables, and a clear scoring function.

# Operates in strict mode by default: the PDF is labeled Pattern A only if every candidate item table matches the Pattern A threshold.

# Clean dataclasses, type hints, logging, and a small unit-test scaffold with 5 example tables.

# Copy the file, run it in Python 3.8+, and experiment with thresholds in Config


"""
pattern_a_detector.py
Detect "Pattern A" for a whole PDF (list of tables).
Author: ChatGPT (crafted for production use)
"""

from __future__ import annotations
import re
import math
import logging
from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional, Tuple, Iterable

# Optional import (we accept DataFrame inputs if pandas is available)
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

# ---------------------------
# Configure logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PatternA")

# ---------------------------
# Configurable thresholds & weights
# ---------------------------
class Config:
    TABLE_SCORE_THRESHOLD: float = 0.85  # table must score >= this to be counted as matching Pattern A
    LOW_CONFIDENCE_THRESHOLD: float = 0.60  # used if confidences provided (not mandatory)
    # Weights to combine component scores -> final table_score
    WEIGHT_HEADER = 0.40
    WEIGHT_UNIFORMITY = 0.30
    WEIGHT_TYPE = 0.20
    WEIGHT_CONF = 0.10
    # Fraction used to detect numeric vs text columns
    NUMERIC_FRAC_EXPECTED = 0.80
    # Max header rows to consider as header candidate (top N rows)
    MAX_HEADER_CANDIDATES: int = 3

# ---------------------------
# Canonical header token sets (expand as needed)
# ---------------------------
HEADER_TOKENS = {
    "item", "description", "desc", "product", "name",
    "qty", "quantity", "pcs", "nos", "unit",
    "rate", "price", "amount", "value", "total", "subtotal", "charge", "cost", "mrp",
    "amount", "amt"
}

NUMERIC_HEADER_TOKENS = {
    "qty", "quantity", "rate", "price", "amount", "value", "charge", "cost", "mrp", "amt"
}

TOTAL_KEYWORDS = {"total", "subtotal", "grand total", "amount due", "balance due", "net total"}

# ---------------------------
# Utilities
# ---------------------------
def normalize_text(s: Any) -> str:
    """Normalize string text for token matching."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\xa0", " ").replace("\u200b", "").strip()
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def tokenize(s: str) -> List[str]:
    """Lowercase word tokens (alphanumeric)."""
    s = normalize_text(s).lower()
    tokens = re.findall(r"[a-z0-9]+", s)
    return tokens

_NUMERIC_RE = re.compile(r"^[\-\+]?[\d,]*\.?\d+$")
def looks_like_number(s: str) -> bool:
    """Loose numeric detection after stripping currency symbols and commas."""
    if s is None:
        return False
    t = normalize_text(s)
    t = re.sub(r"[₹$€£,]", "", t)          # remove common currency symbols & commas
    t = t.replace("%", "")
    t = t.strip()
    if t == "":
        return False
    # Accept numbers like "10", "1.23", "-5.00"
    return bool(_NUMERIC_RE.match(t))

# ---------------------------
# Data structure
# ---------------------------
@dataclass
class TableFrame:
    """
    Represents a reconstructed table (rows x columns).
    df: list of rows; each row is list[str] (strings). it's allowed to have empty strings.
    meta: optional metadata dictionary (page number, table_id, optional confidences)
    """
    df: List[List[str]]
    page: Optional[int] = None
    table_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return len(self.df)

    @property
    def n_cols(self) -> int:
        if not self.df:
            return 0
        return max(len(r) for r in self.df)

    def normalized_df(self) -> List[List[str]]:
        """Return normalized (trimmed) df where all rows have length == n_cols (pad with '')."""
        nc = self.n_cols
        out = []
        for r in self.df:
            newr = [normalize_text(c) for c in r] + [""] * (nc - len(r))
            out.append(newr)
        return out

    def avg_confidence(self) -> Optional[float]:
        """
        Optional: If meta contains per-cell or per-table confidences, return avg.
        Format accepted:
          meta['cell_confidences'] = [[...], [...]] matching df shape
          or meta['avg_confidence'] = float
        """
        if "avg_confidence" in self.meta:
            try:
                return float(self.meta["avg_confidence"])
            except Exception:
                pass
        if "cell_confidences" in self.meta:
            try:
                arr = self.meta["cell_confidences"]
                flat = [v for row in arr for v in row if v is not None]
                if not flat:
                    return None
                return float(sum(flat) / len(flat))
            except Exception:
                return None
        return None

# ---------------------------
# Core detection helpers
# ---------------------------
def convert_table_input(table: Any, page: Optional[int] = None, table_id: Optional[str] = None) -> TableFrame:
    """
    Accept either:
      - TableFrame -> return as-is
      - pandas.DataFrame -> convert rows to list-of-lists (stringified)
      - list[list] -> use directly
    """
    if isinstance(table, TableFrame):
        return table
    if pd is not None and isinstance(table, pd.DataFrame):
        # convert columns preserving column order
        rows: List[List[str]] = []
        for _, row in table.iterrows():
            rows.append([normalize_text(x) for x in row.tolist()])
        return TableFrame(df=rows, page=page, table_id=table_id)
    # assume iterable of rows
    if isinstance(table, list):
        rows = []
        for r in table:
            rows.append([normalize_text(x) for x in r])
        return TableFrame(df=rows, page=page, table_id=table_id)
    raise ValueError("Unsupported table input type. Provide TableFrame, pandas.DataFrame, or list[list].")

def header_candidate_indices(tf: TableFrame) -> List[int]:
    """Return top-k candidate header indices (0-based)."""
    k = min(Config.MAX_HEADER_CANDIDATES, tf.n_rows)
    return list(range(k))

def detect_header_row(tf: TableFrame) -> Tuple[Optional[int], Dict[str, float]]:
    """
    Try to detect the header row index and return component scores as debug info.
    Returns (best_idx_or_None, debug_scores)
    debug_scores = { 'idx': int, 'header_token_fraction': float, 'numeric_ratio': float, 'header_score': float }
    """
    df = tf.normalized_df()
    if tf.n_rows == 0:
        return None, {}

    best_idx = None
    best_score = -1.0
    best_debug = {}
    for idx in header_candidate_indices(tf):
        row = df[idx]
        tokens_per_cell = [tokenize(cell) for cell in row]
        # fraction of header cells that include at least one header token
        header_cells_with_token = 0
        total_cells = max(1, len(row))
        for tks in tokens_per_cell:
            if any(tok in HEADER_TOKENS for tok in tks):
                header_cells_with_token += 1
        header_token_fraction = header_cells_with_token / total_cells

        # numeric ratio in header row (should be low)
        numeric_cells = sum(1 for cell in row if looks_like_number(cell))
        numeric_ratio = numeric_cells / total_cells

        # compute a combined header score (higher is better)
        # prefer rows with header tokens and low numeric presence
        header_score = 0.7 * header_token_fraction + 0.3 * (1.0 - numeric_ratio)

        if header_score > best_score:
            best_score = header_score
            best_idx = idx
            best_debug = {
                "idx": idx,
                "header_token_fraction": header_token_fraction,
                "numeric_ratio": numeric_ratio,
                "header_score": header_score,
            }
    return best_idx, best_debug

def is_totals_row(row: List[str]) -> bool:
    """Rudimentary totals row detection by token match in any cell."""
    for cell in row:
        t = normalize_text(cell).lower()
        if not t:
            continue
        # if contains words like 'total', 'subtotal', 'grand total'
        for kw in TOTAL_KEYWORDS:
            if kw in t:
                return True
    return False

def detect_column_numeric_fracs(df_rows: List[List[str]], header_idx: Optional[int] = None) -> List[float]:
    """
    Return list of fractions per column: fraction of data rows that look numeric in that column.
    Excludes header rows and trailing totals rows when evident.
    """
    if not df_rows:
        return []

    ncols = max(len(r) for r in df_rows)
    # normalize rows to equal length
    rows = [ [normalize_text(c) for c in r] + [""] * (ncols - len(r)) for r in df_rows ]

    # identify data rows by skipping header & trailing totals rows
    data_rows = []
    for i, r in enumerate(rows):
        if header_idx is not None and i == header_idx:
            continue
        # skip rows that look empty
        if all(not normalize_text(c) for c in r):
            continue
        data_rows.append(r)

    # If final row is totals-like, drop it from data rows
    if data_rows and is_totals_row(data_rows[-1]):
        data_rows = data_rows[:-1]

    if not data_rows:
        # return zeros (no data rows)
        return [0.0] * ncols

    fracs = []
    for c in range(ncols):
        col_cells = [row[c] for row in data_rows]
        numeric_count = sum(1 for v in col_cells if looks_like_number(v))
        fracs.append(numeric_count / len(col_cells))
    return fracs

def is_totals_only_table(tf: TableFrame) -> bool:
    """
    Detect a totals-only or summary table (exclude these from candidate item tables).
    Heuristics:
      - 1-2 columns
      - header or first column contains 'total'/'subtotal' keywords
      - many numeric cells and small distinct row count
    """
    df = tf.normalized_df()
    if tf.n_cols <= 1:
        return True
    # Check header tokens
    header_idx, _ = detect_header_row(tf)
    if header_idx is not None:
        header_row = df[header_idx]
        header_text = " ".join(header_row).lower()
        if any(kw in header_text for kw in TOTAL_KEYWORDS):
            return True
    # If most cells are numeric and n_rows is small, assume totals-only
    flat = [normalize_text(c) for row in df for c in row]
    nonempty = [c for c in flat if c]
    if not nonempty:
        return True
    numeric_frac = sum(1 for c in nonempty if looks_like_number(c)) / len(nonempty)
    if numeric_frac > 0.85 and tf.n_rows <= 4:
        return True
    return False

# ---------------------------
# Pattern A detection per table
# ---------------------------
def detect_pattern_A_table(tf: TableFrame) -> Dict[str, Any]:
    """
    Detect how well this single table matches Pattern A.
    Returns a dict with component scores and final table_score (0..1).
    """
    df = tf.normalized_df()
    header_idx, header_debug = detect_header_row(tf)

    # uniformity: fraction of data rows that have a high count of non-empty cells
    ncols = tf.n_cols
    rows = df
    # Build data rows excluding header and trailing totals
    data_rows = []
    for i, r in enumerate(rows):
        if header_idx is not None and i == header_idx:
            continue
        if all(not normalize_text(c) for c in r):
            continue
        data_rows.append(r)
    if data_rows and is_totals_row(data_rows[-1]):
        data_rows = data_rows[:-1]

    # uniformity metric: how many data rows have >= (ncols - 1) non-empty cells
    if not data_rows:
        uniformity_score = 0.0
    else:
        counts = [sum(1 for cell in r if normalize_text(cell) != "") for r in data_rows]
        acceptable = sum(1 for c in counts if c >= max(1, ncols - 1))
        uniformity_score = acceptable / len(counts)

    # column numeric fractions (excl. header)
    col_numeric_fracs = detect_column_numeric_fracs(df, header_idx=header_idx)

    # expected numeric columns from header tokens (if header exist)
    expected_numeric_mask = []
    if header_idx is not None:
        header_row = rows[header_idx]
        for cell in header_row:
            tks = tokenize(cell)
            if any(tok in NUMERIC_HEADER_TOKENS for tok in tks):
                expected_numeric_mask.append(True)
            else:
                expected_numeric_mask.append(False)
    else:
        # no header detected: infer numeric expectation by column numeric frac
        expected_numeric_mask = [fr >= Config.NUMERIC_FRAC_EXPECTED for fr in col_numeric_fracs]

    # type consistency: for numeric-expected columns we want high numeric_frac; for text-expected low numeric_frac
    if col_numeric_fracs:
        per_col_scores = []
        for i, fr in enumerate(col_numeric_fracs):
            expect_num = expected_numeric_mask[i] if i < len(expected_numeric_mask) else False
            if expect_num:
                # score proportional to closeness to 1.0
                col_score = min(1.0, fr / Config.NUMERIC_FRAC_EXPECTED)
            else:
                # prefer small numeric fraction (text columns)
                col_score = 1.0 - min(1.0, fr / 0.5)  # if fr=0.2 -> 1-0.4=0.6; if fr>0.5 -> 0
                col_score = max(0.0, col_score)
            per_col_scores.append(col_score)
        type_consistency_score = sum(per_col_scores) / len(per_col_scores)
    else:
        type_consistency_score = 0.0

    # header_score from header_debug (if header exists)
    header_score = header_debug.get("header_score", 0.0) if header_debug else 0.0

    # confidence factor (optional)
    avg_conf = tf.avg_confidence()
    confidence_factor = avg_conf if avg_conf is not None else 0.95

    # small penalty if many merged/spanned cells are suspected (we don't have row/col spans here,
    # but a heuristic: if many empty cells exist in header or rows, penalize)
    empty_cells = sum(1 for r in rows for c in r if not normalize_text(c))
    total_cells = sum(len(r) for r in rows)
    empty_frac = (empty_cells / total_cells) if total_cells > 0 else 0.0
    span_penalty = max(0.0, min(0.25, empty_frac * 0.5))  # small penalty up to 0.25

    # combine scores using config weights
    final_score = (
        Config.WEIGHT_HEADER * header_score
        + Config.WEIGHT_UNIFORMITY * uniformity_score
        + Config.WEIGHT_TYPE * type_consistency_score
        + Config.WEIGHT_CONF * confidence_factor
    )
    final_score = final_score * (1.0 - span_penalty)  # apply span penalty multiplicatively

    # Clip to [0,1]
    final_score = max(0.0, min(1.0, final_score))

    return {
        "table_id": tf.table_id,
        "page": tf.page,
        "n_rows": tf.n_rows,
        "n_cols": tf.n_cols,
        "header_idx": header_idx,
        "header_debug": header_debug,
        "uniformity_score": uniformity_score,
        "type_consistency_score": type_consistency_score,
        "confidence_factor": confidence_factor,
        "span_penalty": span_penalty,
        "table_score": final_score,
    }

# ---------------------------
# Document-level decision (strict: ALL candidate tables must match)
# ---------------------------
def detect_pdf_pattern_A(tables: Iterable[Any], strict: bool = True) -> Dict[str, Any]:
    """
    Given an iterable of tables (TableFrame|pandas.DataFrame|list[list[str]]), decide if the entire PDF
    is Pattern A under strict rules:
      - First, convert inputs to TableFrames
      - Exclude totals-only and obviously-non-item tables
      - For each remaining candidate table, compute table_score
      - In strict mode: ALL candidate tables must have table_score >= Config.TABLE_SCORE_THRESHOLD
      - If any fail, return 'Mixed' with diagnostics
    Returns a dictionary with verdict and per-table diagnostics.
    """
    # Normalize inputs
    table_frames: List[TableFrame] = []
    for i, t in enumerate(tables):
        tf = convert_table_input(t)
        # set id/page if missing for traceability
        if tf.table_id is None:
            tf.table_id = f"t{i}"
        if tf.page is None:
            tf.page = i  # best effort
        table_frames.append(tf)

    # Filter candidate item tables
    candidates: List[TableFrame] = []
    excluded: List[Tuple[TableFrame, str]] = []
    for tf in table_frames:
        if tf.n_rows <= 1 or tf.n_cols <= 1:
            excluded.append((tf, "too_small"))
            continue
        if is_totals_only_table(tf):
            excluded.append((tf, "totals_only"))
            continue
        # Heuristic: require at least one numeric column or at least one header token in header candidate
        header_idx, hdr_dbg = detect_header_row(tf)
        col_fracs = detect_column_numeric_fracs(tf.normalized_df(), header_idx=header_idx)
        has_numeric_col = any(fr >= 0.3 for fr in col_fracs)  # loose check
        header_token_frac = hdr_dbg.get("header_token_fraction", 0.0) if hdr_dbg else 0.0
        if not (has_numeric_col or header_token_frac >= 0.25):
            # Not candidate item table (likely metadata)
            excluded.append((tf, "non_item"))
            continue
        candidates.append(tf)

    if not candidates:
        return {
            "doc_pattern": None,
            "reason": "no_candidate_item_tables",
            "excluded_count": len(excluded),
            "excluded": [{"table_id": e.table_id, "reason": r} for e, r in excluded],
        }

    # Evaluate each candidate table
    table_results = []
    for tf in candidates:
        res = detect_pattern_A_table(tf)
        table_results.append(res)
        # early-exit in strict mode if a candidate table fails the threshold (but do a quick double-check)
        if strict and res["table_score"] < Config.TABLE_SCORE_THRESHOLD:
            logger.info("Early exit: table %s (page %s) failed threshold with score %.3f",
                        res["table_id"], res["page"], res["table_score"])
            return {
                "doc_pattern": "Mixed",
                "reason": "table_failed_threshold",
                "failed_table": res,
                "table_results": table_results,
                "excluded_count": len(excluded),
                "excluded": [{"table_id": e.table_id, "reason": r} for e, r in excluded],
            }

    # If we reach here, all candidates passed (strict) or we evaluated all (non-strict)
    # Compute document confidence as weighted mean by number of data rows
    weights = []
    scores = []
    for tf, r in zip(candidates, table_results):
        # weight by data rows count
        data_rows_count = max(1, tf.n_rows - (1 if r.get("header_idx") is not None else 0))
        weights.append(data_rows_count)
        scores.append(r["table_score"])
    total_weight = sum(weights) if weights else 1
    doc_confidence = sum(s * w for s, w in zip(scores, weights)) / total_weight

    return {
        "doc_pattern": "A",
        "doc_confidence": doc_confidence,
        "table_results": table_results,
        "excluded_count": len(excluded),
        "excluded": [{"table_id": e.table_id, "reason": r} for e, r in excluded],
    }

# ---------------------------
# Small unit-test scaffold with 5 tables (example)
# ---------------------------
if __name__ == "__main__":
    # Example: 5 tables across a PDF (clean, matching Pattern A)
    sample_tables = [
        # Table 1
        [
            ["Item", "Qty", "Rate", "Amount"],
            ["Apple", "2", "10", "20"],
            ["Banana", "1", "5", "5"],
        ],
        # Table 2
        [
            ["Name", "Quantity", "Price"],
            ["Sugar", "3", "60"],
            ["Salt", "2", "20"],
        ],
        # Table 3
        [
            ["Product", "Amount"],
            ["Milk", "20"],
            ["Butter", "40"],
        ],
        # Table 4
        [
            ["Service", "Charge"],
            ["Delivery", "30"],
        ],
        # Table 5
        [
            ["Desc", "Value"],
            ["Tax", "15"],
        ],
    ]

    # Convert raw lists to TableFrames (page/table_id auto-assigned)
    table_frames = [convert_table_input(t, page=i, table_id=f"table_{i+1}") for i, t in enumerate(sample_tables)]

    result = detect_pdf_pattern_A(table_frames, strict=True)

    import json
    print(json.dumps(result, indent=2))

# How it works — short explanation (why this is accurate)

# Normalization & token matching: header detection uses a token set with many synonyms (item, description, qty, price, amount, charge, etc.) to handle vendor differences and OCR variations.

# Header + structural signals: header token presence and low numeric ratio in header strongly indicate a header. Structural uniformity (rows with most columns filled) indicates a clean itemized table.

# Type consistency: we check per-column numeric fractions and compare against expected numeric header tokens (qty/price/amount). Pattern A expects consistent numeric columns.

# Totals handling: possible totals row at the end is excluded from structural checks so summary rows don't break the table scoring.

# Combined scoring: header (40%), uniformity (30%), type consistency (20%), confidence (10%) — weighted to reflect which signals are most important for Pattern A.

# Strict PDF decision: The PDF is Pattern A only if every candidate item table (after exclusions) scores above TABLE_SCORE_THRESHOLD (default 0.85). Early-exit is implemented to minimize wasted work.

#     How to use

# If you have Textract JSON → first extract tables into list[list[str]] using the extractor you already have (or the earlier example I gave). Then pass that list to detect_pdf_pattern_A(...).

# If you prefer pandas DataFrame, the detector accepts them (if pandas is installed).

# Tune Config.TABLE_SCORE_THRESHOLD and other constants for your dataset.
