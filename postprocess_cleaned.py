#!/usr/bin/env python3
"""
Post-process cleaned invoice table:
- Splits item rows vs summary rows (Subtotal, VAT, Sales Tax, Total, etc.)
- Normalizes types (qty, rate, amount)
- Computes per-invoice sums and verifies against parsed totals
- Outputs cleaned_final.xlsx with sheets: Items, Totals, Verification

Run:
    python postprocess_cleaned.py
"""

import pandas as pd
import re
from pathlib import Path

INPUT_XLSX = "cleaned_invoice_table.xlsx"
OUTPUT_XLSX = "cleaned_final.xlsx"

SUMMARY_KEYWORDS = [
    "subtotal", "total due", "total", "sales tax", "sales tax", "vat", "tax",
    "grand total", "balance due", "amount due", "gst", "sales tax", "sales tax |"
]

def is_summary_description(desc: str) -> bool:
    if not isinstance(desc, str):
        return False
    low = desc.strip().lower()
    # common words
    for k in SUMMARY_KEYWORDS:
        if k in low:
            return True
    # patterns like "Sales Tax | 255.39" or "Total | 381,12 €"
    if re.search(r'\b(total|subtotal|vat|tax|sales tax|amount due|balance due|grand total)\b', low):
        return True
    return False

def parse_amount_from_string(s: str):
    """
    Extract the last numeric token as amount. Handles commas and dots.
    Returns float or None.
    """
    if not isinstance(s, str):
        return None
    s_clean = s.replace("€", "").replace("$", "").replace("£", "").replace(",", ".")
    # find last number-like token (allow grouping like 1.234,56 -> tricky; we replaced comma->dot above)
    # prefer tokens with digits and optional decimal
    tokens = re.findall(r'(-?\d+(?:\.\d+)?)', s_clean)
    if not tokens:
        return None
    # choose last token
    try:
        val = float(tokens[-1])
        return val
    except:
        return None

def normalize_qty(q):
    if pd.isna(q):
        return None
    if isinstance(q, (int, float)) and float(q).is_integer():
        return int(q)
    # strip formatting
    if isinstance(q, str):
        q2 = q.replace(',', '.').strip()
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
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    s = s.replace("€", "").replace("$", "").replace("£", "")
    # unify comma decimal separators e.g. "1.234,56" -> "1234.56" is not handled perfectly — assume commas as thousands only in most cases
    # Heuristic: if both '.' and ',' present and comma after dot, treat comma as decimal (European)
    if '.' in s and ',' in s:
        # if last comma comes after last dot => comma is decimal sep
        if s.rfind(',') > s.rfind('.'):
            s = s.replace('.', '').replace(',', '.')
        else:
            s = s.replace(',', '')
    else:
        s = s.replace(',', '')
    s = s.strip()
    if re.match(r'^-?\d+(\.\d+)?$', s):
        try:
            return float(s)
        except:
            return None
    # fallback: extract last numeric substring
    m = re.findall(r'(-?\d+(?:\.\d+)?)', s)
    if m:
        try:
            return float(m[-1])
        except:
            return None
    return None

def main():
    in_path = Path(INPUT_XLSX)
    if not in_path.exists():
        print(f"Error: {INPUT_XLSX} not found. Run the cleaning step first.")
        return

    df = pd.read_excel(in_path, sheet_name=0)  # expect first sheet
    # Ensure columns exist
    expected_cols = ["invoice_number", "description", "qty", "rate", "amount"]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None

    items = []
    totals = []  # will store parsed summary lines per invoice (could be multiple types)
    # For each invoice, collect summary lines
    for _, row in df.iterrows():
        inv = row.get("invoice_number")
        desc = row.get("description", "")
        qty = row.get("qty", None)
        rate = row.get("rate", None)
        amt = row.get("amount", None)

        # If description looks like a summary / tax / total row OR the description contains 'sales tax' etc.
        if is_summary_description(str(desc)):
            # parse amount from desc OR from amount column
            parsed_amount = None
            # prefer the amount column if numeric
            parsed_amount = normalize_rate_amount(amt)
            if parsed_amount is None:
                parsed_amount = parse_amount_from_string(str(desc))
            # determine type from desc string
            typ = "other"
            low = str(desc).strip().lower()
            if "subtotal" in low:
                typ = "subtotal"
            elif "tax" in low or "vat" in low or "gst" in low:
                typ = "tax"
            elif "total" in low or "amount due" in low or "grand total" in low or "balance due" in low:
                typ = "total"
            elif "sales tax" in low:
                typ = "tax"
            else:
                # also try to pick up lines like "Sales Tax | 255.39"
                if re.search(r'\b(sales tax|sales tax \|)\b', low):
                    typ = "tax"

            totals.append({
                "invoice_number": inv,
                "type": typ,
                "raw": desc,
                "parsed_amount": parsed_amount
            })
            continue

        # otherwise treat as item row
        normalized_qty = normalize_qty(qty)
        normalized_rate = normalize_rate_amount(rate)
        normalized_amount = normalize_rate_amount(amt)

        # If amount is missing but in description there's a trailing number, capture it
        if normalized_amount is None:
            # sometimes description may contain amount in same string (e.g., "1. Skid-Steer € 1000")
            parsed_from_desc = parse_amount_from_string(str(desc))
            if parsed_from_desc is not None:
                normalized_amount = parsed_from_desc

        items.append({
            "invoice_number": inv,
            "description": desc,
            "qty": normalized_qty,
            "rate": normalized_rate,
            "amount": normalized_amount
        })

    # Build DataFrames
    df_items = pd.DataFrame(items)
    df_totals = pd.DataFrame(totals)

    # If no totals found for invoice, we may try to compute subtotal from items
    # Now compute verification table: per-invoice sums and compare
    verification_rows = []
    invoices = sorted(set(df_items['invoice_number'].dropna().unique().tolist() + df_totals['invoice_number'].dropna().unique().tolist()))
    for inv in invoices:
        inv_items = df_items[df_items['invoice_number'] == inv]
        sum_items = inv_items['amount'].dropna().astype(float).sum() if not inv_items.empty else 0.0

        # find parsed subtotal, tax, total if present
        inv_totals = df_totals[df_totals['invoice_number'] == inv]
        subtotal = None
        tax = None
        total = None
        # prefer typed totals
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
                # collect first numeric if no type
                if subtotal is None:
                    subtotal = float(val)

        # Heuristics: if subtotal missing but total present and tax present, subtotal = total - tax
        if subtotal is None and total is not None and tax is not None:
            subtotal = total - tax

        # If total missing but subtotal present and tax present:
        if total is None and subtotal is not None and tax is not None:
            total = subtotal + tax

        # If subtotal still None, set to sum_items
        if subtotal is None:
            subtotal = float(sum_items)

        # If total still None, total = subtotal + tax if tax exists else subtotal
        if total is None:
            total = subtotal + (tax if tax is not None else 0.0)

        # Compare computed sum_items vs subtotal (allow small tolerance e.g., 0.5 currency units)
        diff = float(sum_items) - float(subtotal)
        mismatch = abs(diff) > 0.5  # tolerance; tune if needed

        verification_rows.append({
            "invoice_number": inv,
            "items_sum": float(sum_items),
            "parsed_subtotal": float(subtotal) if subtotal is not None else None,
            "parsed_tax": float(tax) if tax is not None else None,
            "parsed_total": float(total) if total is not None else None,
            "difference_items_vs_subtotal": float(diff),
            "mismatch_flag": bool(mismatch)
        })

    df_ver = pd.DataFrame(verification_rows)

    # Save to Excel
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df_items.to_excel(writer, sheet_name="Items", index=False)
        df_totals.to_excel(writer, sheet_name="Totals", index=False)
        df_ver.to_excel(writer, sheet_name="Verification", index=False)

    # Print summary
    total_invoices = len(invoices)
    mismatches = df_ver['mismatch_flag'].sum() if not df_ver.empty else 0
    print(f"Processed {len(df_items)} item rows across {total_invoices} invoices.")
    print(f"Found {len(df_totals)} summary rows (subtotal/tax/total).")
    print(f"Verification: {mismatches} invoices flagged with mismatch (|items_sum - subtotal| > 0.5).")
    print(f"Output written to: {OUTPUT_XLSX}")

if __name__ == "__main__":
    main()
