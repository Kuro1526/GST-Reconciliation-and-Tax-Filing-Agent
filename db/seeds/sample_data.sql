-- Sample data for end-to-end test
-- This script uses psql's \gset to capture RETURNING values as psql variables.

-- 1) Taxpayer
INSERT INTO taxpayer (gstin, business_name, pan, state_code, contact_email)
VALUES ('29ABCDE1234F2Z5', 'ACME Pvt Ltd', 'ABCDE1234F', '29', 'acct@acme.example')
ON CONFLICT (gstin) DO NOTHING;

-- 2) Insert a sample invoice and capture invoice_id
INSERT INTO invoice (gstin, source, invoice_no, invoice_date, type, supplier_gstin, supplier_name, taxable_value, cgst_amount, sgst_amount, igst_amount, total_amount, raw_payload)
VALUES (
  '29ABCDE1234F2Z5',
  'CSV',
  'INV-1001',
  '2025-11-01',
  'purchase',
  '27SUPP0001A1Z2',
  'Supplier One Pvt Ltd',
  10000.00,
  900.00,
  900.00,
  0.00,
  11800.00,
  jsonb_build_object('source_file','sample.csv','row',1)
)
RETURNING invoice_id;
\gset

-- 3) Insert a sample GSTR-2B row and capture gstr2b_id
INSERT INTO gstr2b_row (gstin, supplier_gstin, supplier_name, invoice_no, invoice_date, taxable_value, cgst_amount, sgst_amount, igst_amount, gstr_unique_ref, raw_payload)
VALUES (
  '29ABCDE1234F2Z5',
  '27SUPP0001A1Z2',
  'Supplier One Pvt Ltd',
  'INV-1001',
  '2025-11-01',
  10000.00,
  900.00,
  900.00,
  0.00,
  'GSTRREF-0001',
  jsonb_build_object('source','GSTR-2B','row',1)
)
RETURNING gstr2b_id;
\gset

-- 4) Create reconciliation linking invoice <-> gstr2b
INSERT INTO reconciliation (gstin, fiscal_period, invoice_id, gstr2b_id, match_status, match_score, matched_fields, mismatch_reason, itc_eligible, flagged_by, review_required)
VALUES (
  '29ABCDE1234F2Z5',
  '2025-11',
  :'invoice_id',
  :'gstr2b_id',
  'matched',
  100,
  'invoice_no,gstin,taxable_value',
  NULL,
  'eligible',
  'system',
  false
);

-- 5) Create a tax computation snapshot and capture computation_id
INSERT INTO tax_computation (gstin, fiscal_period, snapshot_time, total_output_cgst, total_output_sgst, total_output_igst, total_itc_cgst, total_itc_sgst, total_itc_igst, ineligible_itc, adjustments, net_payable_cgst, net_payable_sgst, net_payable_igst, rounding, created_by)
VALUES (
  '29ABCDE1234F2Z5',
  '2025-11',
  now(),
  5000.00,
  5000.00,
  0.00,
  900.00,
  900.00,
  0.00,
  0.00,
  jsonb_build_array(jsonb_build_object('type','recon-ref','recon_id', (SELECT recon_id FROM reconciliation WHERE invoice_id = :'invoice_id' LIMIT 1), 'note','sample')),
  4100.00,
  4100.00,
  0.00,
  0.00,
  'system'
)
RETURNING computation_id;
\gset

-- 6) Create a filing (GSTR-3B) referencing the computation and capture filing_id
INSERT INTO filing (gstin, fiscal_period, form_type, status, computation_id, payload, requires_payment, total_tax_liability, total_tax_paid, created_by)
VALUES (
  '29ABCDE1234F2Z5',
  '2025-11',
  'GSTR-3B',
  'ready',
  :'computation_id',
  jsonb_build_object('summary', jsonb_build_object('total_tax',8200),'notes','auto-generated sample'),
  true,
  8200.00,
  0.00,
  'system'
)
RETURNING filing_id;
\gset

-- 7) Insert a payment linked to the filing
INSERT INTO payment (gstin, fiscal_period, filing_id, challan_no, cin, amount, cgst_amount, sgst_amount, igst_amount, payment_mode, bank_ref, payment_date, raw_challan)
VALUES (
  '29ABCDE1234F2Z5',
  '2025-11',
  :'filing_id',
  'CPIN-202511-0001',
  'CIN-202511-0001',
  8200.00,
  4100.00,
  4100.00,
  0.00,
  'NETBANKING',
  'BANKREF1234',
  '2025-11-10',
  jsonb_build_object('challan_pdf','s3://bucket/challan-202511-0001.pdf')
);

-- 8) Update filing.total_tax_paid from payment (simple demonstration)
UPDATE filing f
SET total_tax_paid = COALESCE((
  SELECT SUM(amount) FROM payment p WHERE p.filing_id = f.filing_id
),0)
WHERE f.filing_id = :'filing_id';

-- 9) Insert a sample audit log entry
INSERT INTO audit_log (gstin, actor, action, target_table, target_id, details)
VALUES (
  '29ABCDE1234F2Z5',
  'system',
  'seed_data_loaded',
  'filing',
  :'filing_id',
  jsonb_build_object('note','seed script inserted sample rows')
);


