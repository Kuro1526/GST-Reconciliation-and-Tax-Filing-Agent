# Billing Extraction Agent (prototype)

This repository contains a minimal prototype agent that scans a path (local directory), extracts text from PDFs/images/DOCX, parses common billing/invoice fields with simple rules, and writes a JSON summary.

Requirements
- Python 3.8+
- For OCR on scanned PDFs/images: install `tesseract` on your system and the Python packages below.
- For PDF OCR conversion, `pdf2image` requires `poppler` installed on your system.

Python requirements (see `requirements.txt`). On Windows, install Tesseract and Poppler separately; their locations may need to be added to PATH.

Quick start (PowerShell)
```powershell
python extracter.py --input inputs --output outputs
```

Notes
- This is a starting point: connectors (DB/SMB/S3), stronger parsing (ML/layout) and tests should be added next.





conda activate invextract
conda install -c conda-forge -y pytesseract pillow opencv numpy pandas openpyxl tqdm pypdf python-dateutil tzdata
conda install -c conda-forge -y camelot-py ghostscript poppler
pip install pypdfium2 pdfplumber tabula-py langdetect
python -c "import pytesseract, pdfplumber, camelot, pandas, cv2; print('Packages OK')"
