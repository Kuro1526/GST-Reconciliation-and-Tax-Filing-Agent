#!/usr/bin/env python3
"""
Enhanced Invoice Extractor (single-file) - improved version

Features:
- Uses pdfplumber.extract_tables() for born-digital PDFs.
- Uses OpenCV + pytesseract.image_to_data to rebuild tables from scanned pages.
- Adaptive Tesseract PSM/OEM choices (CLI options).
- Multi-language support (e.g., -l eng+hin).
- Parallel processing via ProcessPoolExecutor (--workers).
- Incremental persistence to SQLite with deduplication (unique constraints).
- Optionally export to Excel after run.
- Debug raw OCR outputs per-file.

Dependencies:
    pip install pandas pdfplumber pytesseract pillow opencv-python openpyxl tqdm numpy
    (Install tesseract engine on system: brew install tesseract (mac) / apt install tesseract-ocr (linux) / Windows installer)

Usage example:
    python extractor_improved.py -i ./inputs -o ./invoice_data.db --lang eng+hin --psm 6 --oem 3 --workers 4 --min-confidence 60 --debug --export-excel ./invoice_data.xlsx
"""

import argparse
import hashlib
import logging
import os
import re
import sqlite3
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pdfplumber
from PIL import Image
import pytesseract
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("invoice_extractor")

# ------------------- Dataclasses -------------------
@dataclass
class InvoiceRecord:
    file_name: str
    file_path: str
    extraction_date: str
    invoice_from: str = ''
    invoice_to: str = ''
    address: str = ''
    invoice_number: str = ''
    salesperson: str = ''
    table_data_count: int = 0
    raw_text_preview: str = ''
    file_hash: str = ''
    avg_confidence: Optional[float] = None
    low_confidence_flag: bool = False

# ------------------- Utility helpers -------------------
def compute_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def build_tess_config(languages: str, psm: int, oem: int) -> str:
    return f'--oem {oem} --psm {psm} -l {languages}'

def detect_script_hint(text: str) -> Optional[str]:
    """Naive script hint - returns a language code if script strongly detected."""
    if re.search(r'[\u0900-\u097F]', text):  # Devanagari
        return 'hin'
    if re.search(r'[\u0400-\u04FF]', text):  # Cyrillic
        return 'rus'
    if re.search(r'[\u0600-\u06FF]', text):  # Arabic
        return 'ara'
    return None

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ------------------- Table reconstruction helpers -------------------
def cluster_words_to_rows(data_df: pd.DataFrame, x_tolerance: int = 10) -> List[str]:
    """
    Given pytesseract image_to_data output (dataframe), cluster words into lines
    using line_num, block_num etc. Then sort by left coordinate and join.
    """
    df = data_df.copy()
    if df.empty:
        return []
    # Filter out empty text
    df = df[df['text'].notna() & df['text'].astype(str).str.strip().astype(bool)]
    if df.empty:
        return []

    # Group by line-based attributes if available
    if {'page_num','block_num','par_num','line_num'}.issubset(df.columns):
        grouped = df.groupby(['page_num','block_num','par_num','line_num'])
        rows = []
        for _, g in grouped:
            g_sorted = g.sort_values('left')
            rows.append(' '.join(g_sorted['text'].astype(str).tolist()))
        return rows

    # Fallback: cluster by y coordinate
    df['y_center'] = df['top'] + df['height'] / 2
    df = df.sort_values('y_center')
    rows = []
    current_row = []
    current_y = None
    threshold = 8  # pixels
    for _, row in df.iterrows():
        if current_y is None:
            current_y = row['y_center']
            current_row = [row]
        elif abs(row['y_center'] - current_y) <= threshold:
            current_row.append(row)
            current_y = (current_y + row['y_center']) / 2
        else:
            # flush
            current_row_sorted = sorted(current_row, key=lambda r: r['left'])
            rows.append(' '.join([str(r['text']) for r in current_row_sorted]))
            current_row = [row]
            current_y = row['y_center']
    if current_row:
        current_row_sorted = sorted(current_row, key=lambda r: r['left'])
        rows.append(' '.join([str(r['text']) for r in current_row_sorted]))
    return rows

def detect_table_grid_and_extract_rows(img_np: np.ndarray, pil_image: Image.Image, tess_config: str, min_cell_area=1000) -> Tuple[List[Dict[str, Any]], float]:
    """
    Attempt to detect table gridlines using OpenCV and rebuild table rows.
    Returns (rows_as_dicts, avg_confidence)
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    # binarize
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 9)

    # detect horizontal and vertical lines
    horizontal = th.copy()
    vertical = th.copy()
    cols = horizontal.shape[1]
    rows = horizontal.shape[0]
    horizontal_size = max(1, cols // 30)
    vertical_size = max(1, rows // 30)
    h_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    v_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    horizontal = cv2.erode(horizontal, h_structure)
    horizontal = cv2.dilate(horizontal, h_structure)
    vertical = cv2.erode(vertical, v_structure)
    vertical = cv2.dilate(vertical, v_structure)

    grid = cv2.bitwise_and(horizontal, vertical)
    # find contours on grid to identify cell regions
    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cells = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > min_cell_area:
            cells.append((x, y, w, h))
    if not cells:
        # fallback: use OCR to cluster by Y (no grid)
        data = pytesseract.image_to_data(pil_image, config=tess_config, output_type='data.frame')
        avg_conf = float(data.conf.replace(-1, np.nan).dropna().astype(float).mean()) if 'conf' in data else 100.0
        rows_text = cluster_words_to_rows(data)
        row_dicts = []
        for i, rt in enumerate(rows_text):
            row_dicts.append({
                'row_number': i+1,
                'raw_text': rt
            })
        return row_dicts, avg_conf

    # sort cells top-to-bottom, left-to-right and group into rows by similar y
    cells_sorted = sorted(cells, key=lambda c: (c[1], c[0]))
    # group by y
    grouped_rows = []
    current_row = []
    current_y = None
    y_tol = 10
    for (x, y, w, h) in cells_sorted:
        if current_y is None:
            current_y = y
            current_row = [(x,y,w,h)]
        elif abs(y - current_y) <= y_tol:
            current_row.append((x,y,w,h))
            current_y = (current_y + y) // 2
        else:
            grouped_rows.append(current_row)
            current_row = [(x,y,w,h)]
            current_y = y
    if current_row:
        grouped_rows.append(current_row)

    # For each detected cell, run OCR inside the cell and build columns
    row_dicts = []
    confs = []
    for r_idx, row_cells in enumerate(grouped_rows):
        # sort cells by x
        row_cells_sorted = sorted(row_cells, key=lambda c: c[0])
        cell_texts = []
        for c in row_cells_sorted:
            x, y, w, h = c
            # pad slightly
            pad = 2
            x0 = max(0, x-pad); y0 = max(0, y-pad); x1 = min(img_np.shape[1], x+w+pad); y1 = min(img_np.shape[0], y+h+pad)
            crop = img_np[y0:y1, x0:x1]
            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            data = pytesseract.image_to_data(pil_crop, config=tess_config, output_type='data.frame')
            text = ''
            conf = []
            if not data.empty:
                data = data[data['text'].notna() & data['text'].str.strip().astype(bool)]
                if not data.empty:
                    text = ' '.join(data.sort_values('left')['text'].astype(str).tolist())
                    conf_vals = pd.to_numeric(data['conf'], errors='coerce').dropna()
                    if not conf_vals.empty:
                        conf = conf_vals.tolist()
            if conf:
                confs.extend(conf)
            cell_texts.append(text)
        # build a raw_text line
        raw_text = ' | '.join([t for t in cell_texts if t.strip()])
        row_dicts.append({
            'row_number': r_idx + 1,
            'raw_text': raw_text,
            'cols': cell_texts
        })
    avg_conf = float(np.nan) if not confs else float(np.mean(confs))
    return row_dicts, avg_conf

# ------------------- Parsing helpers -------------------
def find_value_by_keywords(text: str, keyword_list: List[str]) -> str:
    text_lower = text.lower()
    # try single-line captures
    for keyword in keyword_list:
        pattern = rf'\b{re.escape(keyword)}\b\s*[:\-]?\s*(.+)'
        m = re.search(pattern, text_lower, flags=re.IGNORECASE)
        if m:
            value = m.group(1).split('\n')[0].strip()
            return value
    # fallback line-based
    for line in text.splitlines():
        low = line.lower()
        for keyword in keyword_list:
            idx = low.find(keyword)
            if idx != -1:
                after = line[idx + len(keyword):].strip(' :\t-')
                if after:
                    return after
    return ''

def extract_address(text: str) -> str:
    addr = find_value_by_keywords(text, ['address', 'street', 'city', 'state', 'zip', 'postal', 'country'])
    if addr:
        return addr
    patterns = [
        r'\d{1,5}\s+[\w\s\.,-]+(?:street|st|avenue|ave|road|rd|lane|ln|boulevard|blvd)\b',
        r'P\.?O\.?\s*Box\s*\d+',
        r'[\w\s]+,\s*[A-Za-z]{2,}\s*\d{5,6}'
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return m.group(0).strip()
    return ''

def clean_invoice_number(raw: str) -> str:
    if not raw:
        return ''
    raw = raw.strip()
    raw = re.sub(r'^(invoice|inv|bill|bill no|invoice no|invoice#|inv\.)[:\s\-]*', '', raw, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', '_', raw)
    cleaned = re.sub(r'[^A-Za-z0-9\-\_#]', '', cleaned)
    cleaned = cleaned[:120]
    return cleaned if cleaned else raw

def parse_table_row_simple(row_text: str) -> List[str]:
    row_text = row_text.strip()
    tokens = re.split(r'\s{2,}|\t|\s\|\s', row_text)
    if len(tokens) >= 3:
        return [t.strip() for t in tokens if t.strip()]
    parts = row_text.split()
    nums = []
    non_nums = []
    for part in reversed(parts):
        if re.match(r'^[\d\.,\-\(\)]+$', part):
            nums.insert(0, part)
        else:
            non_nums = parts[:len(parts) - len(nums)]
            break
    desc = ' '.join(non_nums).strip()
    return [desc] + nums

def extract_table_data_from_text(text: str, invoice_number: str, file_name: str) -> List[Dict[str, Any]]:
    lines = [re.sub(r'\s+', ' ', ln).strip() for ln in text.splitlines() if ln.strip()]
    table_data = []
    in_table = False
    headers = []
    table_keywords = ['description','item','quantity','qty','rate','price','amount','total','unit price','subtotal']
    for i, line in enumerate(lines):
        low = line.lower()
        if any(h in low for h in table_keywords):
            in_table = True
            headers = parse_table_row_simple(line)
            continue
        if in_table:
            if re.search(r'^(total|subtotal|grand total|balance due)\b', low):
                in_table = False
                continue
            if re.search(r'\d', line):
                cols = parse_table_row_simple(line)
                if cols:
                    row = {
                        'invoice_number': invoice_number,
                        'file_name': file_name,
                        'row_number': len(table_data) + 1,
                        'raw_text': line
                    }
                    for j, val in enumerate(cols):
                        row[f'col_{j+1}'] = val
                    table_data.append(row)
    return table_data

# ------------------- Worker function -------------------
def process_single_file(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function executed in separate process.
    args dict contains: file_path, tesseract_cmd, languages, psm, oem, debug, debug_out, min_confidence
    Returns a serializable dict with keys:
      - success: bool
      - record: InvoiceRecord as dict (if success)
      - table_rows: list of dicts
      - error: optional error message
    """
    file_path = Path(args['file_path'])
    try:
        # configure tesseract binary if provided
        if args.get('tesseract_cmd'):
            pytesseract.pytesseract.tesseract_cmd = args['tesseract_cmd']

        tess_config = build_tess_config(args['languages'], args['psm'], args['oem'])

        # compute file hash
        file_hash = compute_file_hash(file_path)

        raw_text_accum = []
        avg_confidences = []

        # Try PDFs and images and txt
        ext = file_path.suffix.lower()
        extracted_text = ''
        table_rows_all: List[Dict[str, Any]] = []

        # PDF path
        if ext == '.pdf':
            try:
                with pdfplumber.open(str(file_path)) as pdf:
                    # Attempt to extract tables first for each page
                    for pageno, page in enumerate(pdf.pages, start=1):
                        # extract textual content
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            extracted_text += '\n' + page_text
                        # try table extraction if available
                        try:
                            tables = page.extract_tables()
                            if tables:
                                for t in tables:
                                    # convert table rows to raw lines and then to dicts
                                    header = t[0] if t and len(t[0]) > 0 else None
                                    for r_idx, row in enumerate(t[1:] if header else t):
                                        # build raw_text from columns
                                        row_text = ' | '.join([str(c).strip() for c in row if c is not None])
                                        tr = {
                                            'invoice_number': '',
                                            'file_name': file_path.name,
                                            'row_number': len(table_rows_all) + 1,
                                            'raw_text': row_text
                                        }
                                        for j, c in enumerate(row):
                                            tr[f'col_{j+1}'] = str(c).strip() if c is not None else ''
                                        table_rows_all.append(tr)
                        except Exception:
                            # ignore table extraction errors and fallback later
                            pass

                    # If we found some extracted_text already, good; otherwise we may rasterize pages for OCR
                    if not extracted_text.strip() or not table_rows_all:
                        # Rasterize pages and OCR
                        for pageno, page in enumerate(pdf.pages, start=1):
                            try:
                                pil_img = page.to_image(resolution=300).original
                                # run OCR on page
                                page_text = pytesseract.image_to_string(pil_img, config=tess_config)
                                extracted_text += '\n' + page_text
                                # Use image_to_data to help table detection if no table rows yet
                                data = pytesseract.image_to_data(pil_img, config=tess_config, output_type='data.frame')
                                if not table_rows_all:
                                    rows_text = cluster_words_to_rows(data)
                                    # basic table extraction from rows_text heuristics
                                    for rt in rows_text:
                                        if re.search(r'\d', rt):
                                            table_rows_all.extend(extract_table_data_from_text(rt, '', file_path.name))
                                # attempt grid detection if still empty
                                if not table_rows_all:
                                    img_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                                    rows_detected, avg_conf = detect_table_grid_and_extract_rows(img_np, pil_img, tess_config)
                                    avg_confidences.append(avg_conf if avg_conf is not None else 0.0)
                                    for r in rows_detected:
                                        tr = {
                                            'invoice_number': '',
                                            'file_name': file_path.name,
                                            'row_number': len(table_rows_all) + 1,
                                            'raw_text': r.get('raw_text', '')
                                        }
                                        # add any detected cols
                                        if 'cols' in r:
                                            for j, c in enumerate(r['cols']):
                                                tr[f'col_{j+1}'] = c
                                        table_rows_all.append(tr)
                            except Exception:
                                continue
            except Exception as e:
                # can't open PDF, fallback to OCR on file bytes with PIL if possible
                logger.exception("Failed to open PDF %s: %s", file_path, e)

        elif ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            # process image file
            try:
                pil_img = Image.open(str(file_path)).convert('RGB')
                img_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                # Preprocess image to improve OCR
                gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
                processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY, 31, 2)
                # run OCR
                page_text = pytesseract.image_to_string(Image.fromarray(processed), config=tess_config)
                extracted_text += '\n' + page_text
                data = pytesseract.image_to_data(Image.fromarray(processed), config=tess_config, output_type='data.frame')
                # cluster rows
                rows_text = cluster_words_to_rows(data)
                for rt in rows_text:
                    if re.search(r'\d', rt):
                        table_rows_all.extend(extract_table_data_from_text(rt, '', file_path.name))
                # grid detection
                rows_detected, avg_conf = detect_table_grid_and_extract_rows(img_np, pil_img, tess_config)
                avg_confidences.append(avg_conf if avg_conf is not None else 0.0)
                for r in rows_detected:
                    tr = {
                        'invoice_number': '',
                        'file_name': file_path.name,
                        'row_number': len(table_rows_all) + 1,
                        'raw_text': r.get('raw_text', '')
                    }
                    if 'cols' in r:
                        for j, c in enumerate(r['cols']):
                            tr[f'col_{j+1}'] = c
                    table_rows_all.append(tr)
            except Exception as e:
                logger.exception("Failed to process image %s: %s", file_path, e)
                extracted_text += ''
        elif ext == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    extracted_text = f.read()
                    table_rows_all = extract_table_data_from_text(extracted_text, '', file_path.name)
            except Exception:
                with open(file_path, 'r', encoding='latin-1') as f:
                    extracted_text = f.read()
                    table_rows_all = extract_table_data_from_text(extracted_text, '', file_path.name)
        else:
            # unsupported
            return {'success': False, 'error': f'Unsupported extension {ext}', 'file': str(file_path)}

        if not extracted_text.strip():
            # nothing extracted
            return {'success': False, 'error': 'No text extracted', 'file': str(file_path)}

        # Save debug text if requested
        if args.get('debug') and args.get('debug_out'):
            try:
                outdir = Path(args['debug_out'])
                ensure_dir(outdir)
                with open(outdir / (file_path.stem + '.txt'), 'w', encoding='utf-8') as rf:
                    rf.write(extracted_text)
            except Exception:
                pass

        # Normalize and parse invoice fields
        invoice_from = find_value_by_keywords(extracted_text, ['from', 'seller', 'vendor', 'supplier', 'issued by', 'bill from'])
        invoice_to = find_value_by_keywords(extracted_text, ['to', 'bill to', 'customer', 'client', 'recipient', 'ship to'])
        address = extract_address(extracted_text)
        invoice_number_raw = find_value_by_keywords(extracted_text, ['invoice no', 'invoice number', 'invoice#', 'inv no', 'bill no', 'bill number', 'invoice id'])
        invoice_number = clean_invoice_number(invoice_number_raw)
        if not invoice_number:
            m = re.search(r'\bINV[-\s_]*\d+\b', extracted_text, flags=re.IGNORECASE)
            if m:
                invoice_number = clean_invoice_number(m.group(0))
            else:
                invoice_number = f'INV_{file_path.stem}'

        salesperson = find_value_by_keywords(extracted_text, ['salesperson', 'sales rep', 'account manager', 'representative'])

        # If table_rows_all empty, try to extract heuristically from full text
        if not table_rows_all:
            table_rows_all = extract_table_data_from_text(extracted_text, invoice_number, file_path.name)

        # Fill invoice_number field for each table row
        for tr in table_rows_all:
            tr['invoice_number'] = invoice_number
            tr['file_name'] = file_path.name

        avg_conf = None
        if avg_confidences:
            # average non-nan confidences
            vals = [v for v in avg_confidences if v is not None and not np.isnan(v)]
            avg_conf = float(np.mean(vals)) if vals else None

        # If no avg_conf computed from image OCR, try word-level extraction confidence
        if avg_conf is None:
            try:
                data_all = pytesseract.image_to_data(Image.fromarray(cv2.cvtColor(np.array(Image.open(str(file_path)).convert('RGB')), cv2.COLOR_RGB2BGR)), config=tess_config, output_type='data.frame') if ext in ['.jpg','.jpeg','.png','.tiff','.bmp'] else None
                if data_all is not None and 'conf' in data_all:
                    conf_vals = pd.to_numeric(data_all['conf'], errors='coerce').dropna()
                    if not conf_vals.empty:
                        avg_conf = float(conf_vals.mean())
            except Exception:
                pass

        low_conf_flag = False
        min_conf = args.get('min_confidence')
        if min_conf is not None and avg_conf is not None:
            low_conf_flag = avg_conf < float(min_conf)

        record = InvoiceRecord(
            file_name=file_path.name,
            file_path=str(file_path.resolve()),
            extraction_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            invoice_from=invoice_from,
            invoice_to=invoice_to,
            address=address,
            invoice_number=invoice_number,
            salesperson=salesperson,
            table_data_count=len(table_rows_all),
            raw_text_preview=(extracted_text[:500] if extracted_text else ''),
            file_hash=file_hash,
            avg_confidence=float(avg_conf) if avg_conf is not None else None,
            low_confidence_flag=low_conf_flag
        )

        return {
            'success': True,
            'record': asdict(record),
            'table_rows': table_rows_all,
            'file': str(file_path)
        }

    except Exception as e:
        logger.exception("Unhandled error processing file %s: %s", file_path, e)
        return {'success': False, 'error': str(e), 'file': str(file_path)}

# ------------------- Persistence (SQLite) -------------------
def init_sqlite_db(db_path: Path):
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS Invoices (
            file_name TEXT,
            file_path TEXT,
            extraction_date TEXT,
            invoice_from TEXT,
            invoice_to TEXT,
            address TEXT,
            invoice_number TEXT,
            salesperson TEXT,
            table_data_count INTEGER,
            raw_text_preview TEXT,
            file_hash TEXT UNIQUE,
            avg_confidence REAL,
            low_confidence_flag INTEGER
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS Table_Data (
            invoice_number TEXT,
            file_name TEXT,
            row_number INTEGER,
            raw_text TEXT,
            -- flexible columns for parsed columns
            -- col_* may be present
            PRIMARY KEY (invoice_number, file_name, row_number)
        )
    ''')
    con.commit()
    return con

def insert_invoice_and_rows(con: sqlite3.Connection, record: Dict[str, Any], rows: List[Dict[str, Any]]):
    cur = con.cursor()
    # Insert invoice with INSERT OR IGNORE based on UNIQUE file_hash
    cur.execute('''
        INSERT OR IGNORE INTO Invoices (
            file_name, file_path, extraction_date, invoice_from, invoice_to, address,
            invoice_number, salesperson, table_data_count, raw_text_preview, file_hash,
            avg_confidence, low_confidence_flag
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        record.get('file_name'),
        record.get('file_path'),
        record.get('extraction_date'),
        record.get('invoice_from'),
        record.get('invoice_to'),
        record.get('address'),
        record.get('invoice_number'),
        record.get('salesperson'),
        record.get('table_data_count'),
        record.get('raw_text_preview'),
        record.get('file_hash'),
        record.get('avg_confidence'),
        1 if record.get('low_confidence_flag') else 0
    ))
    # Insert table rows, using INSERT OR IGNORE to skip duplicates
    for r in rows:
        invoice_number = r.get('invoice_number', '')
        file_name = r.get('file_name', '')
        row_number = int(r.get('row_number', 0)) if r.get('row_number') is not None else None
        raw_text = r.get('raw_text', '')
        # Prepare column data keys like col_1, col_2...
        extra_cols = {k: v for k,v in r.items() if k.startswith('col_')}
        # Build a merged raw_text that includes all cols if cols present
        if not raw_text and extra_cols:
            raw_text = ' | '.join([str(v) for k,v in sorted(extra_cols.items()) if v])
        # Insert while ignoring duplicates
        try:
            cur.execute('''
                INSERT OR IGNORE INTO Table_Data (invoice_number, file_name, row_number, raw_text)
                VALUES (?, ?, ?, ?)
            ''', (invoice_number, file_name, row_number, raw_text))
        except Exception:
            # If PRIMARY KEY constraints different, just ignore
            pass
    con.commit()

# ------------------- CLI / main -------------------
def main():
    parser = argparse.ArgumentParser(description='Improved Invoice extraction with table detection, parallelism, and SQLite persistence.')
    parser.add_argument('--input', '-i', default=None, help='Input folder containing invoices (default: ./inputs)')
    parser.add_argument('--output', '-o', default=None, help='Output sqlite db file (default: ./invoice_data.db)')
    parser.add_argument('--tesseract', default=None, help='Path to tesseract executable (if not in PATH)')
    parser.add_argument('--lang', default='eng', help='OCR languages for tesseract (e.g., eng+hin)')
    parser.add_argument('--psm', type=int, default=6, help='Tesseract PSM (page segmentation mode)')
    parser.add_argument('--oem', type=int, default=3, help='Tesseract OEM (engine mode)')
    parser.add_argument('--workers', type=int, default=2, help='Number of parallel worker processes')
    parser.add_argument('--min-confidence', type=float, default=None, help='Minimum average OCR confidence; flag results below this for review')
    parser.add_argument('--debug', action='store_true', help='Save raw OCR text for each file into debug_texts/')
    parser.add_argument('--debug-out', default='./debug_texts', help='Custom folder for debug text outputs')
    parser.add_argument('--force', action='store_true', help='Force reprocessing even if file_hash already in DB')
    parser.add_argument('--export-excel', default=None, help='Optional path to export results to Excel at end (xlsx)')
    args = parser.parse_args()

    input_folder = Path(args.input) if args.input else Path('./inputs')
    output_db = Path(args.output) if args.output else Path('./invoice_data.db')
    debug_out = Path(args.debug_out)

    print(f"Using input folder: {input_folder.resolve()}")
    print(f"Using SQLite DB:   {output_db.resolve()}")
    if args.debug:
        print(f"Debug OCR text will be written to: {debug_out.resolve()}")
    print(f"Workers: {args.workers} | Tess lang: {args.lang} | PSM: {args.psm} | OEM: {args.oem}")

    if not input_folder.exists() or not input_folder.is_dir():
        print(f"Input folder '{input_folder}' does not exist or is not a folder.")
        return

    # Gather supported files
    exts = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.txt']
    files = []
    for ext in exts:
        files.extend(sorted(input_folder.glob(f'*{ext}')))
        files.extend(sorted(input_folder.glob(f'*{ext.upper()}')))
    files = sorted(set(files))
    if not files:
        print(f"No supported invoice files found in {input_folder}.")
        return

    # Init DB
    con = init_sqlite_db(output_db)

    # If force is not set, load existing file_hashes to skip
    existing_hashes = set()
    if not args.force:
        try:
            df_existing = pd.read_sql_query("SELECT file_hash FROM Invoices", con)
            existing_hashes = set(df_existing['file_hash'].astype(str).fillna(''))
        except Exception:
            existing_hashes = set()

    # Build worker arg list
    worker_args = []
    for f in files:
        fh = compute_file_hash(f)
        if not args.force and fh in existing_hashes:
            logger.info("Skipping already-processed file (hash match): %s", f.name)
            continue
        worker_args.append({
            'file_path': str(f),
            'tesseract_cmd': args.tesseract,
            'languages': args.lang,
            'psm': args.psm,
            'oem': args.oem,
            'debug': args.debug,
            'debug_out': str(debug_out) if args.debug else None,
            'min_confidence': args.min_confidence
        })

    if not worker_args:
        print("No new files to process.")
        # still export if requested
        if args.export_excel:
            try:
                df_inv = pd.read_sql_query("SELECT * FROM Invoices", con)
                df_tbl = pd.read_sql_query("SELECT * FROM Table_Data", con)
                with pd.ExcelWriter(args.export_excel, engine='openpyxl') as writer:
                    df_inv.to_excel(writer, sheet_name='Invoices', index=False)
                    df_tbl.to_excel(writer, sheet_name='Table_Data', index=False)
                print(f"Exported existing DB to Excel: {args.export_excel}")
            except Exception as e:
                print("Failed to export to Excel:", e)
        return

    results = []
    # Use process pool to parallelize
    max_workers = max(1, min(args.workers, (os.cpu_count() or 1)))
    print(f"Processing {len(worker_args)} files with {max_workers} workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(process_single_file, wa): wa['file_path'] for wa in worker_args}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Files"):
            try:
                res = fut.result()
            except Exception as e:
                logger.exception("Worker failed: %s", e)
                continue
            if res.get('success'):
                rec = res.get('record')
                rows = res.get('table_rows', [])
                try:
                    insert_invoice_and_rows(con, rec, rows)
                except Exception:
                    logger.exception("Failed to insert results into DB for file %s", res.get('file'))
            else:
                logger.warning("File %s processing failed: %s", res.get('file'), res.get('error'))

    # Optional Excel export
    if args.export_excel:
        try:
            df_inv = pd.read_sql_query("SELECT * FROM Invoices", con)
            df_tbl = pd.read_sql_query("SELECT * FROM Table_Data", con)
            with pd.ExcelWriter(args.export_excel, engine='openpyxl') as writer:
                df_inv.to_excel(writer, sheet_name='Invoices', index=False)
                df_tbl.to_excel(writer, sheet_name='Table_Data', index=False)
            print(f"Exported DB to Excel: {args.export_excel}")
        except Exception as e:
            logger.exception("Failed to export DB to Excel: %s", e)

    con.close()
    print("Processing complete.")

if __name__ == '__main__':
    main()
