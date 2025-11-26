#!/usr/bin/env python3
"""
classify_docs_formats.py

Two-stage classifier:
  1) Document type (invoice / purchase / credit_note / others)
  2) If invoice -> invoice format (format_A / format_B / ...)

Buckets files by file-format and stores metadata + extracted_text into SQLite.

Usage:
  python classify_docs_formats.py --input ./input --output ./out --templates ./templates/formats --db ./out/results.db --report ./out/report.json
"""
import argparse, logging, os, shutil, sqlite3, json as _json
from pathlib import Path
from datetime import datetime, timezone

# text extraction
from pdfminer.high_level import extract_text as extract_pdf_text
from PIL import Image
import filetype
try:
    import pytesseract
    TESS = True
except Exception:
    TESS = False
try:
    import docx2txt
    DOCX = True
except Exception:
    DOCX = False
import pandas as pd

# template similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------------
# Config: keywords & signatures
# -------------------------
DOCUMENT_KEYWORD_RULES = {
    "gst_sales_invoice": ["tax invoice", "invoice no", "gstin", "totinv", "total invoice value", "totinvval", "grand total", "invoice date"],
    "gst_purchase_invoice": ["purchase invoice", "supplier", "gstin", "purchase", "supplier gstin"],
    "credit_note": ["credit note", "credit memo"],
    "debit_note": ["debit note"],
    "bank_statement": ["account number", "statement", "balance", "ifsc"],
    "receipt": ["receipt", "amount received"],
    "others": []
}

INVOICE_FORMAT_SIGNATURES = {
    "format_A": ["invoice no", "gstin", "taxable value", "hsn", "total", "totinvval", "invoice date"],
    "format_B": ["bill to", "grand total", "invoice date", "invoice number", "gstin"],
    # add more formats as needed
}

SIGNATURE_MATCH_THRESHOLD = 0.35
TEMPLATE_SIMILARITY_THRESHOLD = 0.45

# -------------------------
# file format detection & extractors
# -------------------------
def detect_file_format(path: Path):
    try:
        kind = filetype.guess(str(path))
        if kind:
            if 'pdf' in (kind.mime or "") or kind.extension == 'pdf':
                return "pdf"
            if (kind.mime or "").startswith("image/"):
                return "image"
            if kind.extension in ("xls","xlsx","xlsm"):
                return "excel"
            if kind.extension in ("doc","docx"):
                return "docx"
            if kind.extension == "json":
                return "json"
    except Exception:
        pass
    ext = path.suffix.lower()
    if ext == ".pdf": return "pdf"
    if ext in [".png",".jpg",".jpeg",".tiff",".bmp"]: return "image"
    if ext in [".xls",".xlsx",".csv"]: return "excel"
    if ext in [".doc",".docx"]: return "docx"
    if ext == ".txt": return "text"
    if ext == ".json": return "json"
    return "unknown"

def extract_text_from_pdf(path: Path):
    try:
        return extract_pdf_text(str(path)) or ""
    except Exception as e:
        logging.warning("PDF extract error: %s", e)
        return ""

def extract_text_from_image(path: Path):
    if not TESS:
        logging.warning("Tesseract not available; skipping OCR for %s", path)
        return ""
    try:
        # basic preprocessing
        img = Image.open(str(path)).convert("L")
        try:
            from PIL import ImageFilter, ImageOps
            img = ImageOps.autocontrast(img, cutoff=1)
            img = img.filter(ImageFilter.MedianFilter(size=3))
            img = img.point(lambda p: 255 if p > 160 else 0)
        except Exception:
            pass
        text = pytesseract.image_to_string(img, config="--psm 6")
        return text or ""
    except Exception as e:
        logging.warning("OCR failed for %s: %s", path, e)
        return ""

def extract_text_from_docx(path: Path):
    if not DOCX:
        logging.warning("docx2txt not installed; skipping %s", path)
        return ""
    try:
        return docx2txt.process(str(path)) or ""
    except Exception as e:
        logging.warning("DOCX extract failed: %s", e)
        return ""

def extract_text_from_excel(path: Path, max_chars=2000):
    try:
        df = pd.read_excel(str(path), sheet_name=0, dtype=str, engine="openpyxl")
        sample = " ".join(df.fillna("").astype(str).stack().astype(str).tolist()[:500])
        return sample[:max_chars]
    except Exception as e:
        logging.warning("Excel extract failed: %s", e)
        return ""

def extract_text_from_textfile(path: Path):
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def extract_text_from_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = _json.load(f)
        # flatten text
        texts = []
        def flatten(o):
            if o is None: return
            if isinstance(o, dict):
                for k,v in o.items():
                    texts.append(str(k))
                    flatten(v)
            elif isinstance(o, list):
                for el in o:
                    flatten(el)
            else:
                texts.append(str(o))
        flatten(j)
        flat_text = " ".join(texts)[:30000]
        return flat_text
    except Exception as e:
        logging.warning("JSON extract failed %s: %s", path, e)
        return ""

def extract_text(path: Path, fmt: str):
    if fmt == "pdf": return extract_text_from_pdf(path)
    if fmt == "image": return extract_text_from_image(path)
    if fmt == "docx": return extract_text_from_docx(path)
    if fmt == "excel": return extract_text_from_excel(path)
    if fmt == "text": return extract_text_from_textfile(path)
    if fmt == "json": return extract_text_from_json(path)
    return ""

# -------------------------
# simple keyword doc-type classifier
# -------------------------
def classify_document_type(text: str):
    if not text or text.strip()=="":
        return "others", 0.0
    t = text.lower()
    best, best_score = "others", 0.0
    for dt, keywords in DOCUMENT_KEYWORD_RULES.items():
        if not keywords: continue
        matches = sum(1 for k in keywords if k in t)
        score = matches / max(1, len(keywords))
        if score > best_score:
            best, best_score = dt, score
    if best_score < 0.10:
        return "others", best_score
    return best, best_score

# -------------------------
# invoice format classifier: signature + tfidf fallback
# -------------------------
class InvoiceFormatClassifier:
    def __init__(self, templates_dir: Path = None, model_store: Path = None):
        self.templates_dir = Path(templates_dir) if templates_dir else None
        self.vectorizer = None
        self.template_names = []
        self.template_texts = []
        self.template_vectors = None
        self.model_store = Path(model_store) if model_store else None
        if self.templates_dir and self.templates_dir.exists():
            self._load_templates()

    def _load_templates(self):
        names, texts = [], []
        for f in sorted(self.templates_dir.glob("*.txt")):
            names.append(f.stem)
            texts.append(f.read_text(encoding="utf-8", errors="ignore"))
        if texts:
            self.template_names = names
            self.template_texts = texts
            self.vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=3000)
            self.template_vectors = self.vectorizer.fit_transform(texts)
            if self.model_store:
                self.model_store.mkdir(parents=True, exist_ok=True)
                joblib.dump(self.vectorizer, self.model_store / "tfidf_vectorizer.joblib")
                joblib.dump(self.template_names, self.model_store / "template_names.joblib")

    def classify(self, text: str):
        if not text or text.strip()=="":
            return None, None, 0.0
        t = text.lower()
        # signature match
        for fmt, keys in INVOICE_FORMAT_SIGNATURES.items():
            if not keys: continue
            matches = sum(1 for k in keys if k in t)
            score = matches / max(1, len(keys))
            if score >= SIGNATURE_MATCH_THRESHOLD:
                return fmt, "signature", float(score)
        # template similarity
        if self.template_vectors is not None and self.vectorizer is not None:
            vec = self.vectorizer.transform([text])
            sims = cosine_similarity(vec, self.template_vectors)[0]
            best_idx = int(sims.argmax())
            best_score = float(sims[best_idx])
            if best_score >= TEMPLATE_SIMILARITY_THRESHOLD:
                return self.template_names[best_idx], "template", best_score
        return None, "none", 0.0

# -------------------------
# DB helpers
# -------------------------
def init_db(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY,
        filename TEXT, original_path TEXT, s3_uri TEXT,
        detected_format TEXT, document_type TEXT, doc_type_score REAL,
        invoice_format TEXT, invoice_format_source TEXT, invoice_format_score REAL,
        extracted_text TEXT, size_bytes INTEGER, processed_at TEXT
    );
    """)
    conn.commit()
    return conn

def store_result(conn, row):
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO documents (filename, original_path, s3_uri, detected_format, document_type, doc_type_score,
                           invoice_format, invoice_format_source, invoice_format_score, extracted_text, size_bytes, processed_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row.get("filename"), row.get("original_path"), row.get("s3_uri"),
        row.get("detected_format"), row.get("document_type"), row.get("doc_type_score"),
        row.get("invoice_format"), row.get("invoice_format_source"), row.get("invoice_format_score"),
        row.get("extracted_text"), row.get("size_bytes"), row.get("processed_at")
    ))
    conn.commit()

# -------------------------
# Main processing
# -------------------------
def process_batch(input_folder: Path, output_root: Path, templates_dir: Path, db_path: Path, report_path: Path, s3_prefix: str = None):
    db_conn = init_db(db_path)
    classifier = InvoiceFormatClassifier(templates_dir=templates_dir, model_store=output_root / "model_store")

    results = []
    for root, dirs, files in os.walk(input_folder):
        for fname in files:
            fpath = Path(root) / fname
            try:
                fmt = detect_file_format(fpath)
                # extract text (JSON flattened if json)
                text = extract_text(fpath, fmt)
                doc_type, doc_score = classify_document_type(text)

                invoice_fmt = None
                invoice_source = None
                invoice_score = 0.0
                if doc_type in ("gst_sales_invoice", "gst_purchase_invoice", "invoice", "receipt"):
                    invoice_fmt, invoice_source, invoice_score = classifier.classify(text)

                processed_at = datetime.now(timezone.utc).isoformat()
                size_bytes = fpath.stat().st_size if fpath.exists() else 0

                s3_uri = None
                if s3_prefix:
                    rel = fpath.relative_to(input_folder)
                    s3_uri = str(Path(s3_prefix) / rel).replace("\\","/")

                # create output path: out/<doc_type>/<invoice_format_or_no_format>/<file_format>/
                group1 = doc_type
                group2 = invoice_fmt if invoice_fmt else "no_format"
                dest_dir = Path(output_root) / group1 / group2 / fmt
                dest_dir.mkdir(parents=True, exist_ok=True)
                copied_to = dest_dir / fpath.name
                try:
                    shutil.copy2(str(fpath), str(copied_to))
                except Exception as e:
                    logging.warning("Could not copy file: %s", e)

                row = {
                    "filename": fpath.name,
                    "original_path": str(fpath),
                    "s3_uri": s3_uri,
                    "detected_format": fmt,
                    "document_type": doc_type,
                    "doc_type_score": doc_score,
                    "invoice_format": invoice_fmt,
                    "invoice_format_source": invoice_source,
                    "invoice_format_score": invoice_score,
                    "extracted_text": text,
                    "size_bytes": size_bytes,
                    "processed_at": processed_at
                }

                store_result(db_conn, row)
                results.append(row)
                logging.info("Processed %s -> doc:%s (%.2f) format:%s (%.2f) [%s]", fpath.name, doc_type, doc_score, invoice_fmt, invoice_score, invoice_source)

            except Exception as e:
                logging.exception("Failed processing %s: %s", fpath, e)

    # write report
    report = {"processed_at": datetime.now(timezone.utc).isoformat(), "count": len(results), "results": results}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        _json.dump(report, fh, indent=2, ensure_ascii=False)

    db_conn.close()
    logging.info("Done. Processed %d files. Report: %s DB: %s", len(results), report_path, db_path)
    return results

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Two-stage classifier + format bucketing (JSON parsed as text)")
    parser.add_argument("--input", "-i", required=True, help="Input folder")
    parser.add_argument("--output", "-o", required=True, help="Output root folder")
    parser.add_argument("--templates", "-t", required=False, help="Folder with invoice template .txt files", default=None)
    parser.add_argument("--db", "-d", default="out/results.db", help="SQLite DB path")
    parser.add_argument("--report", "-r", default="out/report.json", help="JSON report path")
    parser.add_argument("--s3-prefix", default=None, help="Optional S3 prefix to populate s3_uri fields")
    args = parser.parse_args()

    input_folder = Path(args.input)
    output_root = Path(args.output)
    templates_dir = Path(args.templates) if args.templates else None
    db_path = Path(args.db)
    report_path = Path(args.report)

    if not input_folder.exists():
        logging.error("Input folder not found: %s", input_folder)
        return

    output_root.mkdir(parents=True, exist_ok=True)
    process_batch(input_folder, output_root, templates_dir, db_path, report_path, args.s3_prefix)

if __name__ == "__main__":
    main()
