# What you get in memory

# A list called tables. One entry per TABLE block found by Textract. Each entry is a dict with:
# page: Page number the table is on.
# confidence: Textract’s confidence for the table block (if present).
# bbox and polygon: The table’s bounding box and polygon (normalized 0–1 coordinates).
# matrix: A 2D list of strings representing the table as rows and columns.
# Merged cells: the text is placed in the top-left anchor cell; the other merged positions are left empty.
# cells: A list of cell dicts with rich metadata:
# row_index, column_index: 1-based indices.
# row_span, column_span: spans for merged cells.
# text: concatenated WORDs (and [x]/[ ] for checkboxes).
# confidence: Textract’s confidence for the cell.
# bbox, polygon: cell geometry (normalized 0–1).
# What gets written to disk

# One CSV per table using matrix only:
# File name: page_{page}table{i}.csv in the out_dir you pass.
# Each CSV row equals one table row; each column equals one table column.
# Quick example of what tables[0] might look like

# tables[0]["page"] -> 1
# tables[0]["matrix"] -> [
# ["Date of Service", "CPT", "Units", "Charge", "Patient Resp"],
# ["01/12/2024", "99213", "1", "
# 150.00
# "
# ,
# "
# 150.00","30.00"],
# ...
# ]
# tables[0]["cells"][0] -> {
# "row_index": 1,
# "column_index": 1,
# "row_span": 1,
# "column_span": 1,
# "text": "Date of Service",
# "confidence": 98.3,
# "bbox": {"Left": 0.12, "Top": 0.18, "Width": 0.15, "Height": 0.02},
# "polygon": [...]
# }
# Notes

# Coordinates (bbox/polygon) are relative to the page (0–1). Convert to pixels/points by multiplying by page width/height if needed.
# Checkboxes in cells appear as “[x]” or “[ ]”.
# If Textract finds no tables, tables = [] and no CSVs are written.
# The CSVs only include text; if you need geometry/confidence exported too, you can dump tables to a JSON file or build a custom CSV writer for the cells list.
# How to see it quickly

# After running:
# print(len(tables)) # number of tables
# print(tables[0]["page"], len(tables[0]["matrix"]), len(tables[0]["matrix"][0])) # page, row count, column count
# print(tables[0]["cells"][0]) # first cell’s metadata and text

import json
from typing import List, Dict, Any, Tuple, Optional
import csv
import os

def load_textract_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_block_map(blocks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {b["Id"]: b for b in blocks}

def get_text_from_block(block: Dict[str, Any], block_map: Dict[str, Dict[str, Any]]) -> str:
    # Concatenate WORDs and handle checkboxes (SELECTION_ELEMENT)
    texts = []
    for rel in block.get("Relationships", []):
        if rel["Type"] == "CHILD":
            for cid in rel["Ids"]:
                child = block_map.get(cid)
                if not child:
                    continue
                if child["BlockType"] == "WORD":
                    texts.append(child["Text"])
                elif child["BlockType"] == "SELECTION_ELEMENT":
                    status = child.get("SelectionStatus", "NOT_SELECTED")
                    texts.append("[x]" if status == "SELECTED" else "[ ]")
    return " ".join(texts).strip()

def extract_table_cells(
    table_block: Dict[str, Any],
    block_map: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    cell_ids = []
    for rel in table_block.get("Relationships", []):
        if rel["Type"] == "CHILD":
            cell_ids.extend(rel["Ids"])
    cells = []
    for cid in cell_ids:
        cell = block_map.get(cid)
        if not cell or cell["BlockType"] not in ("CELL", "MERGED_CELL"):
            continue
        cell_text = get_text_from_block(cell, block_map)
        cells.append({
            "row_index": cell.get("RowIndex", 1),
            "column_index": cell.get("ColumnIndex", 1),
            "row_span": cell.get("RowSpan", 1),
            "column_span": cell.get("ColumnSpan", 1),
            "text": cell_text,
            "confidence": cell.get("Confidence"),
            "bbox": cell.get("Geometry", {}).get("BoundingBox"),
            "polygon": cell.get("Geometry", {}).get("Polygon"),
        })
    return cells

def build_table_matrix(cells: List[Dict[str, Any]]) -> List[List[str]]:
    if not cells:
        return []
    max_row = 0
    max_col = 0
    for c in cells:
        max_row = max(max_row, c["row_index"] + c["row_span"] - 1)
        max_col = max(max_col, c["column_index"] + c["column_span"] - 1)
    matrix = [["" for _ in range(max_col)] for _ in range(max_row)]
    # Place text; for merged cells, put text in the top-left anchored cell
    for c in cells:
        r0 = c["row_index"] - 1
        c0 = c["column_index"] - 1
        for dr in range(c["row_span"]):
            for dc in range(c["column_span"]):
                rr = r0 + dr
                cc = c0 + dc
                if dr == 0 and dc == 0:
                    matrix[rr][cc] = c["text"]
                else:
                    # Optional: mark merged positions or leave empty
                    if not matrix[rr][cc]:
                        matrix[rr][cc] = ""
    return matrix

def extract_tables(textract_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    blocks = textract_json.get("Blocks", [])
    block_map = build_block_map(blocks)
    tables = []
    for b in blocks:
        if b["BlockType"] == "TABLE":
            cells = extract_table_cells(b, block_map)
            matrix = build_table_matrix(cells)
            tables.append({
                "page": b.get("Page"),
                "confidence": b.get("Confidence"),
                "bbox": b.get("Geometry", {}).get("BoundingBox"),
                "polygon": b.get("Geometry", {}).get("Polygon"),
                "cells": cells,
                "matrix": matrix,
            })
    return tables

def save_tables_to_csv(tables: List[Dict[str, Any]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for i, t in enumerate(tables, start=1):
        page = t.get("page", "unknown")
        path = os.path.join(out_dir, f"page_{page}_table_{i}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for row in t["matrix"]:
                writer.writerow(row)

# Example usage:
# data = load_textract_json("textract_output.json")
# tables = extract_tables(data)
# save_tables_to_csv(tables, "./tables_out")
# Now you also have rich metadata in tables[i]["cells"].


