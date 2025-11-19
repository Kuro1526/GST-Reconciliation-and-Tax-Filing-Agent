#!/usr/bin/env python3
"""
extractor_all_in_one.py  (migrated & fixed)

Features:
- Improved extractor (pdfplumber + tesseract + OpenCV)
- Parallel processing via ProcessPoolExecutor
- Persistent SQLite DB with automatic schema migration for older DBs
- Cleaning and postprocessing steps (Items, Totals, Verification)
- Optional JSON export (fixed NaN->None handling)
- Debug outputs supported
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
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

# ------------------ Logging ------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("extractor_all_in_one")

# ------------------ Dataclasses ------------------
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

# ------------------ Utility ------------------
def compute_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def build_tess_config(languages: str, psm: int, oem: int) -> str:
    return f'--oem {oem} --psm {psm} -l {languages}'

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ------------------ DB migration helpers ------------------
def get_table_columns(con: sqlite3.Connection, table: str) -> List[str]:
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]  # name is second field
    return cols

def migrate_db_schema(db_path: Path):
    """
    Adds missing columns to Invoices and creates Table_Data if missing.
    Non-destructive: only adds new columns.
    """
    need_close = False
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        # If Invoices table doesn't exist, we'll create it using init_sqlite_db later.
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Invoices'")
        if not cur.fetchone():
            logger.info("Invoices table not found - will be created by init_sqlite_db.")
            return con  # return open connection to be used by init
        # fetch existing columns
        cols = get_table_columns(con, 'Invoices')
        # desired columns with their SQL definitions (only those we might add)
        desired = {
            'file_name': 'TEXT',
            'file_path': 'TEXT',
            'extraction_date': 'TEXT',
            'invoice_from': 'TEXT',
            'invoice_to': 'TEXT',
            'address': 'TEXT',
            'invoice_number': 'TEXT',
            'salesperson': 'TEXT',
            'table_data_count': 'INTEGER',
            'raw_text_preview': 'TEXT',
            'file_hash': 'TEXT UNIQUE',
            'avg_confidence': 'REAL',
            'low_confidence_flag': 'INTEGER'
        }
        # Add missing columns
        for col, coltype in desired.items():
            if col not in cols:
                # SQLite ALTER TABLE ADD COLUMN supports simple addition
                try:
                    logger.info("Adding missing column to Invoices: %s %s", col, coltype)
                    # For unique constraint we can't add UNIQUE via ALTER easily; just add column and rely on INSERT OR IGNORE or handle separately.
                    # So don't include UNIQUE in ALTER.
                    base_type = coltype.replace(' UNIQUE','')
                    cur.execute(f'ALTER TABLE Invoices ADD COLUMN {col} {base_type}')
                except Exception as e:
                    logger.exception("Failed to add column %s: %s", col, e)
        # Ensure Table_Data exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Table_Data'")
        if not cur.fetchone():
            logger.info("Table_Data not found - creating.")
            cur.execute('''
                CREATE TABLE IF NOT EXISTS Table_Data (
                    invoice_number TEXT,
                    file_name TEXT,
                    row_number INTEGER,
                    raw_text TEXT,
                    PRIMARY KEY (invoice_number, file_name, row_number)
                )
            ''')
        con.commit()
        return con
    except Exception:
        logger.exception("Migration failed; closing connection.")
        con.close()
        raise

def init_sqlite_db(db_path: Path) -> sqlite3.Connection:
    """
    Initialize DB if not present, and return connection.
    We call migrate_db_schema which will add missing columns if needed.
    """
    if not db_path.exists():
        # create file and full schema
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
                PRIMARY KEY (invoice_number, file_name, row_number)
            )
        ''')
        con.commit()
        return con
    else:
        # open and migrate
        con = migrate_db_schema(db_path)
        return con

# ------------------ OCR / Table reconstruction helpers ------------------
def cluster_words_to_rows(data_df: pd.DataFrame, x_tolerance: int = 10) -> List[str]:
    df = data_df.copy()
    if df.empty:
        return []
    df = df[df['text'].notna() & df['text'].astype(str).str.strip().astype(bool)]
    if df.empty:
        return []
    if {'page_num','block_num','par_num','line_num'}.issubset(df.columns):
        grouped = df.groupby(['page_num','block_num','par_num','line_num'])
        rows = []
        for _, g in grouped:
            g_sorted = g.sort_values('left')
            rows.append(' '.join(g_sorted['text'].astype(str).tolist()))
        return rows
    df['y_center'] = df['top'] + df['height'] / 2
    df = df.sort_values('y_center')
    rows = []
    current_row = []
    current_y = None
    threshold = 8
    for _, row in df.iterrows():
        if current_y is None:
            current_y = row['y_center']
            current_row = [row]
        elif abs(row['y_center'] - current_y) <= threshold:
            current_row.append(row)
            current_y = (current_y + row['y_center']) / 2
        else:
            current_row_sorted = sorted(current_row, key=lambda r: r['left'])
            rows.append(' '.join([str(r['text']) for r in current_row_sorted]))
            current_row = [row]
            current_y = row['y_center']
    if current_row:
        current_row_sorted = sorted(current_row, key=lambda r: r['left'])
        rows.append(' '.join([str(r['text']) for r in current_row_sorted]))
    return rows

def detect_table_grid_and_extract_rows(img_np: np.ndarray, pil_image: Image.Image, tess_config: str, min_cell_area=1000):
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 9)
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
    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cells = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > min_cell_area:
            cells.append((x, y, w, h))
    if not cells:
        data = pytesseract.image_to_data(pil_image, config=tess_config, output_type='data.frame')
        avg_conf = float(data.conf.replace(-1, np.nan).dropna().astype(float).mean()) if 'conf' in data else 100.0
        rows_text = cluster_words_to_rows(data)
        row_dicts = []
        for i, rt in enumerate(rows_text):
            row_dicts.append({'row_number': i+1, 'raw_text': rt})
        return row_dicts, avg_conf
    cells_sorted = sorted(cells, key=lambda c: (c[1], c[0]))
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
    row_dicts = []
    confs = []
    for r_idx, row_cells in enumerate(grouped_rows):
        row_cells_sorted = sorted(row_cells, key=lambda c: c[0])
        cell_texts = []
        for c in row_cells_sorted:
            x, y, w, h = c
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
        raw_text = ' | '.join([t for t in cell_texts if t.strip()])
        row_dicts.append({'row_number': r_idx + 1, 'raw_text': raw_text, 'cols': cell_texts})
    avg_conf = float(np.nan) if not confs else float(np.mean(confs))
    return row_dicts, avg_conf

# ------------------ Text parsing helpers ------------------
def find_value_by_keywords(text: str, keyword_list: List[str]) -> str:
    text_lower = text.lower()
    for keyword in keyword_list:
        pattern = rf'\b{re.escape(keyword)}\b\s*[:\-]?\s*(.+)'
        m = re.search(pattern, text_lower, flags=re.IGNORECASE)
        if m:
            value = m.group(1).split('\n')[0].strip()
            return value
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
                    row = {'invoice_number': invoice_number,'file_name': file_name,'row_number': len(table_data) + 1,'raw_text': line}
                    for j, val in enumerate(cols):
                        row[f'col_{j+1}'] = val
                    table_data.append(row)
    return table_data

# ------------------ Worker: processes single file ------------------
def process_single_file(args: Dict[str, Any]) -> Dict[str, Any]:
    file_path = Path(args['file_path'])
    try:
        if args.get('tesseract_cmd'):
            pytesseract.pytesseract.tesseract_cmd = args['tesseract_cmd']
        tess_config = build_tess_config(args['languages'], args['psm'], args['oem'])
        file_hash = compute_file_hash(file_path)
        extracted_text = ''
        table_rows_all: List[Dict[str, Any]] = []
        avg_confidences = []

        ext = file_path.suffix.lower()
        if ext == '.pdf':
            try:
                with pdfplumber.open(str(file_path)) as pdf:
                    for pageno, page in enumerate(pdf.pages, start=1):
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            extracted_text += '\n' + page_text
                        try:
                            tables = page.extract_tables()
                            if tables:
                                for t in tables:
                                    header = t[0] if t and len(t[0]) > 0 else None
                                    for r in (t[1:] if header else t):
                                        row_text = ' | '.join([str(c).strip() for c in r if c is not None])
                                        tr = {'invoice_number': '', 'file_name': file_path.name, 'row_number': len(table_rows_all) + 1, 'raw_text': row_text}
                                        for j, c in enumerate(r):
                                            tr[f'col_{j+1}'] = str(c).strip() if c is not None else ''
                                        table_rows_all.append(tr)
                        except Exception:
                            pass
                    if not extracted_text.strip() or not table_rows_all:
                        for pageno, page in enumerate(pdf.pages, start=1):
                            try:
                                pil_img = page.to_image(resolution=300).original
                                page_text = pytesseract.image_to_string(pil_img, config=tess_config)
                                extracted_text += '\n' + page_text
                                data = pytesseract.image_to_data(pil_img, config=tess_config, output_type='data.frame')
                                if not table_rows_all:
                                    rows_text = cluster_words_to_rows(data)
                                    for rt in rows_text:
                                        if re.search(r'\d', rt):
                                            table_rows_all.extend(extract_table_data_from_text(rt, '', file_path.name))
                                if not table_rows_all:
                                    img_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                                    rows_detected, avg_conf = detect_table_grid_and_extract_rows(img_np, pil_img, tess_config)
                                    if avg_conf is not None:
                                        avg_confidences.append(avg_conf)
                                    for r in rows_detected:
                                        tr = {'invoice_number': '', 'file_name': file_path.name, 'row_number': len(table_rows_all) + 1, 'raw_text': r.get('raw_text', '')}
                                        if 'cols' in r:
                                            for j, c in enumerate(r['cols']):
                                                tr[f'col_{j+1}'] = c
                                        table_rows_all.append(tr)
                            except Exception:
                                continue
            except Exception as e:
                logger.exception("Failed to open PDF %s: %s", file_path, e)

        elif ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            try:
                pil_img = Image.open(str(file_path)).convert('RGB')
                img_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
                processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
                page_text = pytesseract.image_to_string(Image.fromarray(processed), config=tess_config)
                extracted_text += '\n' + page_text
                data = pytesseract.image_to_data(Image.fromarray(processed), config=tess_config, output_type='data.frame')
                rows_text = cluster_words_to_rows(data)
                for rt in rows_text:
                    if re.search(r'\d', rt):
                        table_rows_all.extend(extract_table_data_from_text(rt, '', file_path.name))
                rows_detected, avg_conf = detect_table_grid_and_extract_rows(img_np, pil_img, tess_config)
                if avg_conf is not None:
                    avg_confidences.append(avg_conf)
                for r in rows_detected:
                    tr = {'invoice_number': '', 'file_name': file_path.name, 'row_number': len(table_rows_all) + 1, 'raw_text': r.get('raw_text', '')}
                    if 'cols' in r:
                        for j, c in enumerate(r['cols']):
                            tr[f'col_{j+1}'] = c
                    table_rows_all.append(tr)
            except Exception as e:
                logger.exception("Failed to process image %s: %s", file_path, e)
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
            return {'success': False, 'error': f'Unsupported extension {ext}', 'file': str(file_path)}

        if not extracted_text.strip():
            return {'success': False, 'error': 'No text extracted', 'file': str(file_path)}

        if args.get('debug') and args.get('debug_out'):
            try:
                outdir = Path(args['debug_out'])
                ensure_dir(outdir)
                with open(outdir / (file_path.stem + '.txt'), 'w', encoding='utf-8') as rf:
                    rf.write(extracted_text)
            except Exception:
                pass

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
        if not table_rows_all:
            table_rows_all = extract_table_data_from_text(extracted_text, invoice_number, file_path.name)
        for tr in table_rows_all:
            tr['invoice_number'] = invoice_number
            tr['file_name'] = file_path.name
        avg_conf = None
        if avg_confidences:
            vals = [v for v in avg_confidences if v is not None and not np.isnan(v)]
            avg_conf = float(np.mean(vals)) if vals else None
        if avg_conf is None and ext in ['.jpg','.jpeg','.png','.tiff','.bmp']:
            try:
                data_all = pytesseract.image_to_data(Image.fromarray(cv2.cvtColor(np.array(Image.open(str(file_path)).convert('RGB')), cv2.COLOR_RGB2BGR)), config=tess_config, output_type='data.frame')
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
        return {'success': True, 'record': asdict(record), 'table_rows': table_rows_all, 'file': str(file_path)}
    except Exception as e:
        logger.exception("Unhandled error processing file %s: %s", file_path, e)
        return {'success': False, 'error': str(e), 'file': str(file_path)}

# ------------------ DB insertion (robust) ------------------
def insert_invoice_and_rows(con: sqlite3.Connection, record: Dict[str, Any], rows: List[Dict[str, Any]]):
    """
    Insert invoice record and rows into DB. This function adapts to existing table columns.
    """
    cur = con.cursor()
    # get actual columns in Invoices table
    cur.execute("PRAGMA table_info(Invoices)")
    info = cur.fetchall()
    col_names = [row[1] for row in info]  # name field
    # build insert column list and values according to available columns
    insert_cols = []
    insert_vals = []
    mapping = {
        'file_name': record.get('file_name'),
        'file_path': record.get('file_path'),
        'extraction_date': record.get('extraction_date'),
        'invoice_from': record.get('invoice_from'),
        'invoice_to': record.get('invoice_to'),
        'address': record.get('address'),
        'invoice_number': record.get('invoice_number'),
        'salesperson': record.get('salesperson'),
        'table_data_count': record.get('table_data_count'),
        'raw_text_preview': record.get('raw_text_preview'),
        'file_hash': record.get('file_hash'),
        'avg_confidence': record.get('avg_confidence'),
        'low_confidence_flag': 1 if record.get('low_confidence_flag') else 0
    }
    for k, v in mapping.items():
        if k in col_names:
            insert_cols.append(k)
            insert_vals.append(v)
    if insert_cols:
        placeholders = ','.join('?' for _ in insert_cols)
        cols_sql = ','.join(insert_cols)
        sql = f'INSERT OR IGNORE INTO Invoices ({cols_sql}) VALUES ({placeholders})'
        try:
            cur.execute(sql, tuple(insert_vals))
        except Exception:
            logger.exception("Failed to execute invoice insert SQL: %s - values: %s", sql, insert_vals)
    # Insert table rows (Table_Data)
    # Ensure Table_Data has columns invoice_number, file_name, row_number, raw_text
    cur.execute("PRAGMA table_info(Table_Data)")
    td_info = cur.fetchall()
    td_cols = [r[1] for r in td_info]
    for r in rows:
        invoice_number = r.get('invoice_number', '')
        file_name = r.get('file_name', '')
        row_number = r.get('row_number', None)
        raw_text = r.get('raw_text', '')
        # fallback: if raw_text missing but col_* present, join them
        extra_cols = {k:v for k,v in r.items() if k.startswith('col_')}
        if not raw_text and extra_cols:
            raw_text = ' | '.join([str(v) for k,v in sorted(extra_cols.items()) if v])
        insert_cols_td = []
        insert_vals_td = []
        for k, v in [('invoice_number', invoice_number), ('file_name', file_name), ('row_number', row_number), ('raw_text', raw_text)]:
            if k in td_cols:
                insert_cols_td.append(k)
                insert_vals_td.append(v)
        if insert_cols_td:
            placeholders = ','.join('?' for _ in insert_cols_td)
            cols_sql = ','.join(insert_cols_td)
            sql_td = f'INSERT OR IGNORE INTO Table_Data ({cols_sql}) VALUES ({placeholders})'
            try:
                cur.execute(sql_td, tuple(insert_vals_td))
            except Exception:
                logger.exception("Failed to insert table row for invoice %s row %s", invoice_number, row_number)
    con.commit()

# ------------------ Cleaning step (DB -> cleaned_invoice_table.xlsx) ------------------
def cleaning_step_from_db(db_path: Path, cleaned_xlsx_path: Path) -> bool:
    if not db_path.exists():
        logger.error("DB not found: %s", db_path)
        return False
    con = sqlite3.connect(str(db_path))
    try:
        df_tbl = pd.read_sql_query("SELECT * FROM Table_Data", con)
    except Exception as e:
        logger.exception("Failed to read Table_Data from DB: %s", e)
        con.close()
        return False
    con.close()
    if df_tbl.empty:
        logger.warning("No Table_Data rows found in DB.")

    def looks_like_date(s):
        return bool(re.match(r'^\d{1,2}\.\d{1,2}\.\d{2,4}$', str(s).strip()))

    def extract_values(text):
        text = str(text).replace("€", "").replace(",", "").strip()
        parts = [p.strip() for p in text.split("|")]
        if any(looks_like_date(p) for p in parts):
            return None, None, None
        m = re.search(r'Qty\.?\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)', text, re.I)
        if m:
            return float(m.group(1)), float(m.group(2)), float(m.group(3))
        numeric_tokens = []
        for p in parts:
            if re.match(r'^\d+(\.\d+)?$', p):
                numeric_tokens.append(p)
        if len(numeric_tokens) == 3:
            try:
                return float(numeric_tokens[0]), float(numeric_tokens[1]), float(numeric_tokens[2])
            except:
                return None, None, None
        last_num = re.findall(r'(\d+(?:\.\d+)?)$', text)
        if last_num:
            try:
                return None, None, float(last_num[0])
            except:
                return None, None, None
        return None, None, None

    clean_rows = []
    for _, row in df_tbl.iterrows():
        raw = str(row.get("raw_text", "")).strip()
        inv = row.get("invoice_number")
        if any(x in raw.lower() for x in ["subtotal", "total", "vat", "gross", "balance", "due"]):
            continue
        if not raw.strip():
            continue
        description = raw.split("\n")[0]
        qty, rate, amount = extract_values(raw)
        clean_rows.append({"invoice_number": inv, "description": description, "qty": qty, "rate": rate, "amount": amount})
    clean_df = pd.DataFrame(clean_rows)
    try:
        ensure_dir(Path(cleaned_xlsx_path).parent)
        clean_df.to_excel(str(cleaned_xlsx_path), index=False)
        logger.info("Cleaned table saved to %s", cleaned_xlsx_path)
        return True
    except Exception as e:
        logger.exception("Failed to write cleaned xlsx: %s", e)
        return False

# ------------------ Postprocess / verification + JSON export ------------------
def postprocess_cleaned(input_cleaned_xlsx: Path, output_xlsx: Path, export_json_path: Optional[Path] = None) -> bool:
    if not input_cleaned_xlsx.exists():
        logger.error("Cleaned input not found: %s", input_cleaned_xlsx)
        return False
    df = pd.read_excel(str(input_cleaned_xlsx))
    expected_cols = ["invoice_number","description","qty","rate","amount"]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None
    SUMMARY_KEYWORDS = ["subtotal","total due","total","sales tax","vat","tax","grand total","balance due","amount due","gst"]
    def is_summary_description(desc):
        if not isinstance(desc, str):
            return False
        low = desc.strip().lower()
        for k in SUMMARY_KEYWORDS:
            if k in low:
                return True
        if re.search(r'\b(total|subtotal|vat|tax|sales tax|amount due|balance due|grand total)\b', low):
            return True
        return False
    def parse_amount_from_string(s):
        if not isinstance(s, str):
            return None
        s_clean = s.replace("€","").replace("$","").replace("£","").replace(",","." )
        tokens = re.findall(r'(-?\d+(?:\.\d+)?)', s_clean)
        if not tokens:
            return None
        try:
            return float(tokens[-1])
        except:
            return None
    def normalize_qty(q):
        if pd.isna(q):
            return None
        if isinstance(q,(int,float)) and float(q).is_integer():
            return int(q)
        if isinstance(q,str):
            q2 = q.replace(',','.').strip()
            if re.match(r'^\d+(\.\d+)?$', q2):
                f = float(q2)
                if float(f).is_integer():
                    return int(f)
                return f
        try:
            return int(q)
        except:
            try:
                return float(q)
            except:
                return None
    def normalize_rate_amount(val):
        if pd.isna(val):
            return None
        if isinstance(val,(int,float)):
            return float(val)
        s = str(val).strip()
        s = s.replace("€","").replace("$","").replace("£","")
        if '.' in s and ',' in s:
            if s.rfind(',') > s.rfind('.'):
                s = s.replace('.','').replace(',','.')
            else:
                s = s.replace(',','')
        else:
            s = s.replace(',','')
        s = s.strip()
        if re.match(r'^-?\d+(\.\d+)?$', s):
            try:
                return float(s)
            except:
                return None
        m = re.findall(r'(-?\d+(?:\.\d+)?)', s)
        if m:
            try:
                return float(m[-1])
            except:
                return None
        return None

    items = []
    totals = []
    for _, row in df.iterrows():
        inv = row.get('invoice_number')
        desc = row.get('description','')
        qty = row.get('qty', None)
        rate = row.get('rate', None)
        amt = row.get('amount', None)
        if is_summary_description(str(desc)):
            parsed_amount = normalize_rate_amount(amt)
            if parsed_amount is None:
                parsed_amount = parse_amount_from_string(str(desc))
            typ = "other"
            low = str(desc).strip().lower()
            if "subtotal" in low:
                typ = "subtotal"
            elif "tax" in low or "vat" in low or "gst" in low:
                typ = "tax"
            elif "total" in low or "amount due" in low or "grand total" in low or "balance due" in low:
                typ = "total"
            totals.append({"invoice_number": inv, "type": typ, "raw": desc, "parsed_amount": parsed_amount})
            continue
        normalized_qty = normalize_qty(qty)
        normalized_rate = normalize_rate_amount(rate)
        normalized_amount = normalize_rate_amount(amt)
        if normalized_amount is None:
            parsed_from_desc = parse_amount_from_string(str(desc))
            if parsed_from_desc is not None:
                normalized_amount = parsed_from_desc
        items.append({"invoice_number": inv, "description": desc, "qty": normalized_qty, "rate": normalized_rate, "amount": normalized_amount})

    df_items = pd.DataFrame(items)
    df_totals = pd.DataFrame(totals)

    verification_rows = []
    invoices = sorted(set(df_items['invoice_number'].dropna().unique().tolist() + df_totals['invoice_number'].dropna().unique().tolist()))
    for inv in invoices:
        inv_items = df_items[df_items['invoice_number'] == inv]
        sum_items = inv_items['amount'].dropna().astype(float).sum() if not inv_items.empty else 0.0
        inv_totals = df_totals[df_totals['invoice_number'] == inv]
        subtotal = None; tax = None; total = None
        for _, trow in inv_totals.iterrows():
            ttype = trow.get('type')
            val = trow.get('parsed_amount')
            if pd.isna(val):
                continue
            if ttype == 'subtotal' and subtotal is None:
                subtotal = float(val)
            elif ttype == 'tax' and tax is None:
                tax = float(val)
            elif ttype == 'total' and total is None:
                total = float(val)
            else:
                if subtotal is None:
                    subtotal = float(val)
        if subtotal is None and total is not None and tax is not None:
            subtotal = total - tax
        if total is None and subtotal is not None and tax is not None:
            total = subtotal + tax
        if subtotal is None:
            subtotal = float(sum_items)
        if total is None:
            total = subtotal + (tax if tax is not None else 0.0)
        diff = float(sum_items) - float(subtotal)
        mismatch = abs(diff) > 0.5
        verification_rows.append({"invoice_number": inv, "items_sum": float(sum_items), "parsed_subtotal": float(subtotal) if subtotal is not None else None, "parsed_tax": float(tax) if tax is not None else None, "parsed_total": float(total) if total is not None else None, "difference_items_vs_subtotal": float(diff), "mismatch_flag": bool(mismatch)})

    df_ver = pd.DataFrame(verification_rows)

    try:
        ensure_dir(Path(output_xlsx).parent)
        with pd.ExcelWriter(str(output_xlsx), engine="openpyxl") as writer:
            df_items.to_excel(writer, sheet_name="Items", index=False)
            df_totals.to_excel(writer, sheet_name="Totals", index=False)
            df_ver.to_excel(writer, sheet_name="Verification", index=False)
        logger.info("Postprocessed output written to: %s", output_xlsx)
    except Exception as e:
        logger.exception("Failed to write postprocessed excel: %s", e)
        return False

    # JSON export: convert NaN -> None properly using where
    if export_json_path:
        try:
            items_list = df_items.where(pd.notnull(df_items), None).to_dict(orient='records')
            totals_list = df_totals.where(pd.notnull(df_totals), None).to_dict(orient='records')
            ver_list = df_ver.where(pd.notnull(df_ver), None).to_dict(orient='records')
            by_invoice = {}
            invoices_set = sorted(set([r.get('invoice_number') for r in items_list if r.get('invoice_number')] + [r.get('invoice_number') for r in totals_list if r.get('invoice_number')]))
            for inv in invoices_set:
                inv_items = [r for r in items_list if r.get('invoice_number') == inv]
                inv_totals = [r for r in totals_list if r.get('invoice_number') == inv]
                inv_ver = next((r for r in ver_list if r.get('invoice_number') == inv), None)
                by_invoice[inv] = {"items": inv_items, "totals": inv_totals, "verification": inv_ver}
            combined = {"items": items_list, "totals": totals_list, "verification": ver_list, "by_invoice": by_invoice}
            ensure_dir(Path(export_json_path).parent)
            with open(str(export_json_path), 'w', encoding='utf-8') as jf:
                json.dump(combined, jf, ensure_ascii=False, indent=2)
            logger.info("Exported JSON to: %s", export_json_path)
        except Exception:
            logger.exception("Failed to export JSON to %s", export_json_path)
            return False

    return True

# ------------------ CLI / Main ------------------
def main():
    parser = argparse.ArgumentParser(description='Combined extractor -> cleaner -> postprocess pipeline (with DB migration).')
    parser.add_argument('--input', '-i', default='./inputs', help='Input folder (defaults ./inputs)')
    parser.add_argument('--db', '-o', default='./invoice_data.db', help='SQLite DB file (defaults ./invoice_data.db)')
    parser.add_argument('--tesseract', default=None, help='Path to tesseract executable')
    parser.add_argument('--lang', default='eng', help='Tesseract languages (eng+hin)')
    parser.add_argument('--psm', type=int, default=6, help='Tesseract PSM')
    parser.add_argument('--oem', type=int, default=3, help='Tesseract OEM')
    parser.add_argument('--workers', type=int, default=2, help='Parallel workers')
    parser.add_argument('--min-confidence', type=float, default=None, help='Min avg OCR confidence to flag low_confidence_flag')
    parser.add_argument('--debug', action='store_true', help='Save raw OCR text')
    parser.add_argument('--debug-out', default='./debug_texts', help='Debug output folder')
    parser.add_argument('--force', action='store_true', help='Force reprocessing even if file_hash exists in DB')
    parser.add_argument('--export-excel', default=None, help='Optional Excel export of DB (Invoices & Table_Data)')
    parser.add_argument('--skip-extract', action='store_true', help='Skip extraction stage (assume DB exists)')
    parser.add_argument('--skip-clean', action='store_true', help='Skip cleaning stage (assume cleaned_invoice_table.xlsx exists)')
    parser.add_argument('--skip-post', action='store_true', help='Skip postprocess stage')
    parser.add_argument('--cleaned-xlsx', default='./cleaned_invoice_table.xlsx', help='Path for cleaned invoice table xlsx')
    parser.add_argument('--final-xlsx', default='./cleaned_final.xlsx', help='Path for final postprocessed xlsx')
    parser.add_argument('--export-json', default=None, help='Optional path to export combined JSON (e.g., ./out.json)')
    args = parser.parse_args()

    input_folder = Path(args.input).expanduser().resolve()
    db_path = Path(args.db).expanduser().resolve()
    debug_out = Path(args.debug_out).expanduser().resolve()
    cleaned_xlsx = Path(args.cleaned_xlsx).expanduser().resolve()
    final_xlsx = Path(args.final_xlsx).expanduser().resolve()
    export_json_path = Path(args.export_json).expanduser().resolve() if args.export_json else None

    print(f"Input folder: {input_folder}")
    print(f"DB file:      {db_path}")
    print(f"Workers: {args.workers} | Tess: {args.lang} | PSM: {args.psm} | OEM: {args.oem}")

    # Stage 1: extraction
    if not args.skip_extract:
        if not input_folder.exists() or not input_folder.is_dir():
            logger.error("Input folder not found: %s", input_folder)
            return
        # Gather files
        exts = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.txt']
        files = []
        for ext in exts:
            files.extend(sorted(input_folder.glob(f'*{ext}')))
            files.extend(sorted(input_folder.glob(f'*{ext.upper()}')))
        files = sorted(set(files))
        if not files:
            logger.warning("No supported files found in input folder.")
        else:
            # initialize DB (and migrate if needed)
            con = init_sqlite_db(db_path)
            existing_hashes = set()
            if not args.force:
                try:
                    df_existing = pd.read_sql_query("SELECT file_hash FROM Invoices", con)
                    existing_hashes = set(df_existing['file_hash'].astype(str).fillna(''))
                except Exception:
                    existing_hashes = set()
            worker_args = []
            for f in files:
                fh = compute_file_hash(f)
                if not args.force and fh in existing_hashes:
                    logger.info("Skipping already processed file (hash match): %s", f.name)
                    continue
                worker_args.append({'file_path': str(f), 'tesseract_cmd': args.tesseract, 'languages': args.lang, 'psm': args.psm, 'oem': args.oem, 'debug': args.debug, 'debug_out': str(debug_out) if args.debug else None, 'min_confidence': args.min_confidence})
            if worker_args:
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
            else:
                logger.info("No new files to process.")
            # Optional Excel export of DB contents
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
    else:
        logger.info("Skipping extraction stage (--skip-extract)")

    # Stage 2: cleaning step
    if args.skip_clean:
        logger.info("Skipping cleaning stage (--skip-clean)")
    else:
        ok = cleaning_step_from_db(db_path, cleaned_xlsx)
        if not ok:
            logger.error("Cleaning step failed. Aborting further postprocessing.")
            return

    # Stage 3: postprocess
    if args.skip_post:
        logger.info("Skipping postprocess stage (--skip-post)")
    else:
        ok2 = postprocess_cleaned(cleaned_xlsx, final_xlsx, export_json_path)
        if not ok2:
            logger.error("Postprocess step failed.")
            return

    print("All requested steps complete.")

if __name__ == '__main__':
    main()
