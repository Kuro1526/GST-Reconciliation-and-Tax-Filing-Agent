#!/usr/bin/env python3
"""
Enhanced Invoice Extractor (single-file)
- Usage:
    python extractor.py                 # defaults: input ../inputs , output ../invoice_data.xlsx
    python extractor.py --input ... --output ... --tesseract "/path/to/tesseract" --lang eng --debug

Dependencies:
    pip install pandas pdfplumber pytesseract pillow opencv-python openpyxl
    (Install tesseract engine on system: brew install tesseract (mac) / apt install tesseract-ocr (linux) / installer for Windows)
"""
import os
import re
import sys
import argparse
import logging
import hashlib
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import pdfplumber
from PIL import Image
import pytesseract
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


class InvoiceExtractor:
    def __init__(self, tesseract_cmd: Optional[str] = None, languages: str = 'eng', debug: bool = False, debug_out: Optional[Path] = None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        self.tesseract_config = f'-l {languages} --psm 6'  # tune as necessary
        self.debug = debug
        self.debug_out = debug_out

        # Centralized keyword sets (lowercase)
        self.keywords = {
            'invoice_from': ['from', 'seller', 'vendor', 'supplier', 'issued by', 'bill from'],
            'invoice_to': ['to', 'bill to', 'customer', 'client', 'recipient', 'ship to'],
            'address': ['address', 'street', 'city', 'state', 'zip', 'postal', 'country'],
            'invoice_number': ['invoice no', 'invoice number', 'invoice#', 'inv no', 'bill no', 'bill number', 'invoice id'],
            'salesperson': ['salesperson', 'sales rep', 'account manager', 'representative'],
            'table_headers': ['description', 'item', 'quantity', 'qty', 'rate', 'price', 'amount', 'total', 'unit price', 'subtotal']
        }

    # --------------------- File helpers ---------------------
    def get_supported_files(self, folder_path: str, extensions: Optional[List[str]] = None) -> List[Path]:
        if extensions is None:
            extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.txt']

        folder = Path(folder_path)
        if not folder.exists():
            logger.info("Input folder '%s' does not exist. Creating it...", folder_path)
            folder.mkdir(parents=True, exist_ok=True)
            return []

        files: List[Path] = []
        for ext in extensions:
            files.extend(folder.glob(f'*{ext}'))
            files.extend(folder.glob(f'*{ext.upper()}'))

        files_sorted = sorted(files)
        logger.info('Found %d supported files in %s', len(files_sorted), folder_path)
        return files_sorted

    def compute_file_hash(self, file_path: Path) -> str:
        h = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()

    # --------------------- Text extraction ---------------------
    def preprocess_image_for_ocr(self, img: np.ndarray) -> np.ndarray:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Use adaptive threshold for varied lighting
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, 2)
        # Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        return clean

    def extract_text_from_image(self, image_path: Path) -> str:
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                logger.warning('Could not read image: %s', image_path)
                return ''

            processed = self.preprocess_image_for_ocr(img)

            # Write to a secure temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_name = tmp.name
                cv2.imwrite(tmp_name, processed)

            try:
                text = pytesseract.image_to_string(Image.open(tmp_name), config=self.tesseract_config)
            finally:
                try:
                    os.remove(tmp_name)
                except OSError:
                    pass

            return text
        except Exception as e:
            logger.exception('Error processing image %s: %s', image_path, e)
            return ''

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        text_parts: List[str] = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_parts.append(page_text)
                    else:
                        # If text extraction fails, rasterize page and OCR it
                        try:
                            pil_image = page.to_image(resolution=200).original
                            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                                pil_image.save(tmp.name)
                                tmp_name = tmp.name

                            page_ocr = pytesseract.image_to_string(Image.open(tmp_name), config=self.tesseract_config)
                            text_parts.append(page_ocr)

                            try:
                                os.remove(tmp_name)
                            except OSError:
                                pass
                        except Exception:
                            logger.debug('Failed raster OCR on page %d of %s', i + 1, pdf_path)
                            continue
        except Exception as e:
            logger.exception('Error reading PDF %s: %s', pdf_path, e)

        return '\n'.join(text_parts)

    def extract_text(self, file_path) -> str:
        """Extract text from various file types. Robustly accepts str or Path."""
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()

        # Debugging helper (temporary) — remove if verbose
        logger.debug("extract_text called for: %s -> suffix: '%s'", file_path, file_ext)

        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            return self.extract_text_from_image(file_path)
        elif file_ext == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
        else:
            print(f"Unsupported file type: {file_ext}")
            return ""

    # --------------------- Parsing helpers ---------------------
    def find_value_by_keywords(self, text: str, keyword_list: List[str]) -> str:
        text_lower = text.lower()

        # Try flexible regex: keyword possibly followed by punctuation and then capture rest of the line
        for keyword in keyword_list:
            pattern = rf'\b{re.escape(keyword)}\b\s*[:\-]?\s*(.+)'
            m = re.search(pattern, text_lower, flags=re.IGNORECASE)
            if m:
                value = m.group(1).split('\n')[0].strip()
                return value

        # fallback: find a line containing the keyword and return content after it
        for line in text.splitlines():
            low = line.lower()
            for keyword in keyword_list:
                idx = low.find(keyword)
                if idx != -1:
                    after = line[idx + len(keyword):].strip(' :\t-')
                    if after:
                        return after
        return ''

    def extract_address(self, text: str) -> str:
        addr = self.find_value_by_keywords(text, self.keywords['address'])
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

    def normalize_number(self, s: str) -> str:
        if s is None:
            return ''
        return re.sub(r'[£$₹, \s]', '', s)

    def parse_table_row(self, row_text: str) -> List[str]:
        row_text = row_text.strip()
        tokens = re.split(r'\s{2,}|\t', row_text)
        if len(tokens) >= 3:
            return [t.strip() for t in tokens if t.strip()]

        parts = row_text.split()
        nums = []
        non_nums = []
        for part in reversed(parts):
            if re.match(r'^[\d.,]+$', part):
                nums.insert(0, part)
            else:
                non_nums = parts[:len(parts) - len(nums)]
                break
        desc = ' '.join(non_nums).strip()
        return [desc] + nums

    def extract_table_data(self, text: str, invoice_number: str, file_name: str) -> List[Dict[str, Any]]:
        lines = [re.sub(r'\s+', ' ', ln).strip() for ln in text.splitlines() if ln.strip()]
        table_data = []
        in_table = False
        headers = []

        for i, line in enumerate(lines):
            low = line.lower()
            if any(h in low for h in self.keywords['table_headers']):
                in_table = True
                headers = self.parse_table_row(line)
                continue

            if in_table:
                if re.search(r'^(total|subtotal|grand total|balance due)\b', low):
                    in_table = False
                    continue

                if re.search(r'\d', line):
                    cols = self.parse_table_row(line)
                    if cols:
                        row = {
                            'invoice_number': invoice_number,
                            'file_name': file_name,
                            'row_number': len(table_data) + 1,
                            'raw_text': line,
                        }
                        for j, val in enumerate(cols):
                            row[f'col_{j+1}'] = val
                        table_data.append(row)
        return table_data

    # --------------------- Invoice number cleaning ---------------------
    def clean_invoice_number(self, raw: str) -> str:
        """Basic cleaning to produce a tidy invoice identifier from OCR output."""
        if not raw:
            return ''
        # take up to first line or token that looks like an invoice number
        raw = raw.strip()
        # remove common prefixes/labels that might be included
        raw = re.sub(r'^(invoice|inv|bill|bill no|invoice no|invoice#)[:\s\-]*', '', raw, flags=re.IGNORECASE)
        # keep only alnum, dash, underscore, and hash; replace sequences of whitespace with single underscore
        cleaned = re.sub(r'\s+', '_', raw)
        cleaned = re.sub(r'[^A-Za-z0-9\-\_#]', '', cleaned)
        # truncate to reasonable length
        cleaned = cleaned[:80]
        return cleaned if cleaned else raw

    # --------------------- Main extraction ---------------------
    def extract_invoice_data(self, file_path: Path) -> Optional[Tuple[InvoiceRecord, List[Dict[str, Any]]]]:
        text = self.extract_text(file_path)
        if not text or not text.strip():
            logger.warning('No text extracted from %s', file_path)
            return None

        # Save debug raw text if requested
        if self.debug and self.debug_out:
            try:
                self.debug_out.mkdir(parents=True, exist_ok=True)
                rawfile = self.debug_out / (Path(file_path).stem + ".txt")
                with open(rawfile, 'w', encoding='utf-8') as rf:
                    rf.write(text)
            except Exception:
                logger.exception('Failed to save debug text for %s', file_path)

        invoice_from = self.find_value_by_keywords(text, self.keywords['invoice_from'])
        invoice_to = self.find_value_by_keywords(text, self.keywords['invoice_to'])
        address = self.extract_address(text)
        invoice_number_raw = self.find_value_by_keywords(text, self.keywords['invoice_number'])
        invoice_number = self.clean_invoice_number(invoice_number_raw)

        # fallback: if still empty, try to find a token like INV or digits
        if not invoice_number:
            m = re.search(r'\bINV[-\s_]*\d+\b', text, flags=re.IGNORECASE)
            if m:
                invoice_number = self.clean_invoice_number(m.group(0))
            else:
                # pick filename-based fallback
                invoice_number = f'INV_{Path(file_path).stem}'

        salesperson = self.find_value_by_keywords(text, self.keywords['salesperson'])

        table_data = self.extract_table_data(text, invoice_number, Path(file_path).name)

        record = InvoiceRecord(
            file_name=Path(file_path).name,
            file_path=str(file_path),
            extraction_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            invoice_from=invoice_from,
            invoice_to=invoice_to,
            address=address,
            invoice_number=invoice_number,
            salesperson=salesperson,
            table_data_count=len(table_data),
            raw_text_preview=text[:500],
            file_hash=self.compute_file_hash(Path(file_path))
        )

        return record, table_data

    # --------------------- Persistence ---------------------
    def save_to_excel(self, invoice_records: List[InvoiceRecord], table_data_list: List[List[Dict[str, Any]]], excel_file: str = 'invoice_data.xlsx') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Save invoices and table rows to an Excel file with de-duplication.
        Backwards compatible: if existing Excel doesn't have 'file_hash', fall back to (file_name, invoice_number) dedupe.
        """
        # New dataframes
        df_invoices_new = pd.DataFrame([asdict(r) for r in invoice_records])
        all_table_rows = []
        for rows in table_data_list:
            all_table_rows.extend(rows)
        df_tables_new = pd.DataFrame(all_table_rows)

        # If file exists, load existing sheets safely
        if os.path.exists(excel_file):
            with pd.ExcelFile(excel_file) as xls:
                existing_invoices = pd.read_excel(xls, 'Invoices') if 'Invoices' in xls.sheet_names else pd.DataFrame()
                existing_tables = pd.read_excel(xls, 'Table_Data') if 'Table_Data' in xls.sheet_names else pd.DataFrame()
        else:
            existing_invoices = pd.DataFrame()
            existing_tables = pd.DataFrame()

        # --- Deduplicate invoices ---
        # Preferred de-dup key: file_hash (if present). Fallback: (file_name, invoice_number)
        if not existing_invoices.empty and 'file_hash' in existing_invoices.columns:
            existing_hashes = set(existing_invoices['file_hash'].astype(str).fillna(''))
            # Keep only new invoice records whose file_hash is not in existing hashes
            df_invoices_to_add = df_invoices_new[~df_invoices_new['file_hash'].astype(str).fillna('').isin(existing_hashes)]
        elif not existing_invoices.empty:
            # Fallback: use (file_name, invoice_number) pair
            existing_pairs = set(zip(existing_invoices['file_name'].astype(str), existing_invoices['invoice_number'].astype(str)))
            df_invoices_new['pair_key'] = list(zip(df_invoices_new['file_name'].astype(str), df_invoices_new['invoice_number'].astype(str)))
            df_invoices_to_add = df_invoices_new[~df_invoices_new['pair_key'].isin(existing_pairs)].drop(columns=['pair_key'])
        else:
            df_invoices_to_add = df_invoices_new

        # --- Deduplicate table rows ---
        if existing_tables.empty:
            df_tables_to_add = df_tables_new
        else:
            # Build a set of existing table row keys (invoice_number, file_name, row_number)
            existing_tables = existing_tables.copy()
            # If 'row_number' isn't present, create it via index (best-effort)
            if 'row_number' not in existing_tables.columns:
                existing_tables['row_number'] = range(1, len(existing_tables) + 1)

            existing_keys = set(zip(
                existing_tables['invoice_number'].astype(str),
                existing_tables['file_name'].astype(str),
                existing_tables['row_number'].astype(int)
            ))

            if df_tables_new.empty:
                df_tables_to_add = pd.DataFrame()
            else:
                df_tables_new = df_tables_new.copy()
                # Ensure row_number exists and is integer
                if 'row_number' not in df_tables_new.columns:
                    df_tables_new['row_number'] = range(1, len(df_tables_new) + 1)
                else:
                    try:
                        df_tables_new['row_number'] = df_tables_new['row_number'].astype(int)
                    except Exception:
                        df_tables_new['row_number'] = range(1, len(df_tables_new) + 1)

                # Build keys and filter
                df_tables_new['__key'] = list(zip(df_tables_new['invoice_number'].astype(str), df_tables_new['file_name'].astype(str), df_tables_new['row_number'].astype(int)))
                df_tables_to_add = df_tables_new[~df_tables_new['__key'].isin(existing_keys)].drop(columns=['__key'])

        # --- Combine dataframes ---
        if existing_invoices.empty:
            df_invoices_combined = df_invoices_new
        else:
            df_invoices_combined = pd.concat([existing_invoices, df_invoices_to_add], ignore_index=True)

        if existing_tables.empty:
            df_tables_combined = df_tables_new
        else:
            df_tables_combined = pd.concat([existing_tables, df_tables_to_add], ignore_index=True)

        # --- Ensure file_hash column exists going forward ---
        if 'file_hash' not in df_invoices_combined.columns:
            df_invoices_combined['file_hash'] = df_invoices_combined.get('file_hash', '')

        # Save to Excel
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df_invoices_combined.to_excel(writer, sheet_name='Invoices', index=False)
            if not df_tables_combined.empty:
                df_tables_combined.to_excel(writer, sheet_name='Table_Data', index=False)

        logger.info('Saved %d invoices and %d table rows to %s', len(df_invoices_combined), len(df_tables_combined) if not df_tables_combined.empty else 0, excel_file)
        return df_invoices_combined, df_tables_combined


# --------------------- CLI entrypoint ---------------------
def main():
    # compute defaults relative to this script file location
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir.parent / "inputs"   # ../inputs relative to extractor/
    default_output = script_dir.parent / "invoice_data.xlsx"  # ../invoice_data.xlsx
    default_debug_dir = script_dir.parent / "debug_texts"

    parser = argparse.ArgumentParser(description='Invoice extraction utility')
    parser.add_argument('--input', '-i', default=None,
                        help=f'Input folder containing invoices (default: {default_input})')
    parser.add_argument('--output', '-o', default=None,
                        help=f'Output excel file (default: {default_output})')
    parser.add_argument('--tesseract', default=None, help='Path to tesseract executable (if not in PATH)')
    parser.add_argument('--lang', default='eng', help='OCR languages for tesseract')
    parser.add_argument('--debug', action='store_true', help='Save raw OCR text for each file into debug_texts/')
    parser.add_argument('--debug-out', default=None, help='Custom folder for debug text outputs')

    args = parser.parse_args()

    # resolve final paths
    input_folder = Path(args.input) if args.input else default_input
    output_file = Path(args.output) if args.output else default_output
    debug_out = Path(args.debug_out).expanduser().resolve() if args.debug_out else default_debug_dir

    # expand user and resolve
    input_folder = input_folder.expanduser().resolve()
    output_file = output_file.expanduser().resolve()

    # Print chosen paths for clarity
    print(f"Using input folder: {input_folder}")
    print(f"Using output file:  {output_file}")
    if args.debug:
        print(f"Debug OCR text will be written to: {debug_out}")

    # Create extractor instance
    extractor = InvoiceExtractor(tesseract_cmd=args.tesseract, languages=args.lang, debug=args.debug, debug_out=debug_out if args.debug else None)

    # Ensure input exists
    if not input_folder.exists() or not input_folder.is_dir():
        print(f"Input folder '{input_folder}' does not exist or is not a folder.")
        print("Please create the folder or pass a correct --input path.")
        return

    # Gather files
    files = extractor.get_supported_files(str(input_folder))

    if not files:
        print(f"No supported invoice files found in {input_folder}.")
        return

    records = []
    all_table_data = []

    for f in files:
        print(f"Processing: {f.name}")
        res = extractor.extract_invoice_data(f)
        if res:
            record, tables = res
            records.append(record)
            all_table_data.append(tables)
            print(f"  -> Extracted invoice_number={record.invoice_number} table_rows={len(tables)}")
        else:
            print(f"  -> Failed to extract from {f.name}")

    if records:
        extractor.save_to_excel(records, all_table_data, excel_file=str(output_file))
        print(f"Saved results to: {output_file}")
    else:
        print("No invoice data was extracted.")


if __name__ == '__main__':
    main()
