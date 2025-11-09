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
