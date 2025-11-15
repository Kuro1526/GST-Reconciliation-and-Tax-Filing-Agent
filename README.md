# Billing Extraction Agent (prototype)

This repository contains a minimal prototype agent that scans a path (local directory), extracts text from PDFs/images/DOCX, parses common billing/invoice fields with simple rules, and writes a JSON summary.

## Quick Start (Single Command)

```powershell
conda activate invextract
python .\pipeline.py -i .\inputs -o .\invoice_data.db --lang eng+hin --workers 2
```

This runs the **complete end-to-end pipeline** in one command:
1. **Extract** invoices from PDFs (OCR + pdfplumber) → `invoice_data.db`
2. **Clean** & parse extracted data → `cleaned_invoice_table.xlsx`
3. **Reconcile** & verify totals → `cleaned_final.xlsx` ✓

**Output:** `cleaned_final.xlsx` with three sheets:
- **Items**: Line items (qty, rate, amount)
- **Totals**: Summary rows (subtotal, tax, total)
- **Verification**: Reconciliation & discrepancy checks

## Requirements

- Python 3.8+
- For OCR on scanned PDFs/images: install `tesseract` on your system and the Python packages below.
- For PDF OCR conversion, `pdf2image` requires `poppler` installed on your system.

Python requirements (see `requirements.txt`). On Windows, install Tesseract and Poppler separately; their locations may need to be added to PATH.

## Installation

### 1. Create Conda Environment

```powershell
conda create -n invextract python=3.8 -y
conda activate invextract
```

### 2. Install Python Packages

```powershell
pip install -r requirements.txt
```

Or use conda:

```powershell
conda install -c conda-forge -y pytesseract pillow opencv numpy pandas openpyxl tqdm pypdf python-dateutil tzdata
conda install -c conda-forge -y camelot-py ghostscript poppler
pip install pypdfium2 pdfplumber tabula-py langdetect
```

### 3. Verify Installation

```powershell
python -c "import pytesseract, pdfplumber, camelot, pandas, cv2; print('Packages OK')"
```

## Advanced Options

```powershell
python .\pipeline.py -i ./inputs -o ./invoice_data.db \
  --lang eng+hin \        # Languages (English + Hindi)
  --psm 6 \               # Tesseract PSM mode
  --oem 3 \               # Tesseract OEM mode
  --workers 2 \           # Parallel workers
  --debug                 # Enable debug output
```

## Legacy: Running Stages Separately

If you need to run individual stages (not recommended):

```powershell
# Stage 1: Extract
python .\extractor\extractor_improved.py -i .\inputs -o .\invoice_data.db --lang eng+hin --psm 6 --oem 3 --workers 2 --export-excel .\invoice_data.xlsx

# Stage 2: Clean
python .\clean_table_data.py

# Stage 3: Post-process
python .\postprocess_cleaned.py
```

## Notes

- This is a starting point: connectors (DB/SMB/S3), stronger parsing (ML/layout) and tests should be added next.
- The pipeline deduplicates invoices based on file hash (MD5).
- Reconciliation checks verify that item totals match parsed summary rows.