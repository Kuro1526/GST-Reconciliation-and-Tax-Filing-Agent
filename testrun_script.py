# =============================================
# 📘 CLASSIFICATION PIPELINE – GOOGLE COLAB NOTEBOOK
# =============================================

# ============================================================
# 1) INSTALL DEPENDENCIES
# ============================================================
!apt-get update -y
!apt-get install -y tesseract-ocr libmagic1

!pip install pdfminer.six pillow pytesseract python-magic filetype docx2txt pandas openpyxl scikit-learn joblib

print("Dependencies installed successfully!")

# ============================================================
# 2) UPLOAD YOUR SCRIPT (classify_docs_formats.py)
# ============================================================
from google.colab import files
print("Upload your classifier script: classify_docs_formats.py")
uploaded_script = files.upload()

# Save uploaded script
import io
import os

for name, data in uploaded_script.items():
    with open(name, "wb") as f:
        f.write(data)
print("Classifier script saved.")

# ============================================================
# 3) UPLOAD INVOICE FILES INTO /content/input
# ============================================================
print("Upload your invoice files (PDF, JPG, JSON, DOCX, etc.)")
uploaded = files.upload()

os.makedirs("input", exist_ok=True)
for name, data in uploaded.items():
    with open(f"input/{name}", "wb") as f:
        f.write(data)

print("Uploaded invoices:")
!ls -l input

# ============================================================
# 4) OPTIONAL: UPLOAD TEMPLATE FILES (format_A.txt, etc.)
# ============================================================
os.makedirs("templates/formats", exist_ok=True)
print("If you have template files (format_A.txt etc.), upload them now.")
print("If not, skip this cell.")

uploaded_templates = files.upload()

for name, data in uploaded_templates.items():
    with open(f"templates/formats/{name}", "wb") as f:
        f.write(data)

print("Templates saved:")
!ls -l templates/formats

# ============================================================
# 5) RUN THE CLASSIFIER
# ============================================================
!python classify_docs_formats.py \
    --input "input" \
    --output "out" \
    --templates "templates/formats" \
    --db "out/results.db" \
    --report "out/report.json"

print("\nClassification Completed!")

# ============================================================
# 6) DISPLAY OUTPUT FOLDER TREE
# ============================================================
print("\n📂 Output Folder Structure:")
!find out -maxdepth 4 -type d

# ============================================================
# 7) SHOW TOP 20 DB ENTRIES
# ============================================================
import sqlite3, pandas as pd

conn = sqlite3.connect("out/results.db")
df = pd.read_sql_query("SELECT id, filename, detected_format, document_type, invoice_format, invoice_format_source, invoice_format_score FROM documents ORDER BY id DESC LIMIT 20;", conn)
conn.close()

print("\n📄 Top 20 Classified Documents:")
df

# ============================================================
# 8) SHOW FULL REPORT.JSON
# ============================================================
import json

with open("out/report.json") as f:
    report = json.load(f)

print("\n📘 Report Summary:")
print("Processed files:", report["count"])
report["results"][:3]  # preview first 3 entries

# ============================================================
# 9) DOWNLOAD RESULTS (DB + OUTPUT ZIP)
# ============================================================
!zip -r out_results.zip out

from google.colab import files
files.download("out_results.zip")
files.download("out/results.db")
files.download("out/report.json")

print("All downloads ready!")
