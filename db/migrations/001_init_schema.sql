-- Useful extensions
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS pgcrypto;    -- provides gen_random_uuid()


-- ========== TAXPAYER ==========
CREATE TABLE IF NOT EXISTS taxpayer (
  gstin           VARCHAR(15) PRIMARY KEY,
  business_name   TEXT,
  pan             VARCHAR(10),
  state_code      VARCHAR(2),
  contact_email   TEXT,
  created_at      TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at      TIMESTAMP WITH TIME ZONE DEFAULT now()
);


-- ========== INVOICE ==========
CREATE TABLE IF NOT EXISTS invoice (
  invoice_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  gstin                VARCHAR(15) NOT NULL REFERENCES taxpayer(gstin) ON DELETE RESTRICT,
--   ingestion_batch_id   UUID,
  invoice_no           TEXT,
--   invoice_no_normalized TEXT GENERATED ALWAYS AS (
--                           lower(regexp_replace(coalesce(invoice_no,''), '\s+|[^a-z0-9]', '', 'gi'))
--                         ) STORED,
  invoice_date         DATE,
  type                 TEXT CHECK (type IN ('sales','purchase')) DEFAULT 'purchase',
  supplier_gstin       VARCHAR(15),
  supplier_name        TEXT,
  taxable_value        NUMERIC(18,2),
  cgst_amount          NUMERIC(18,2),
  sgst_amount          NUMERIC(18,2),
  igst_amount          NUMERIC(18,2),
  total_amount         NUMERIC(18,2),
  rcm_applicable       BOOLEAN DEFAULT false,
  blocked_credit       BOOLEAN DEFAULT false,
  line_items           JSONB,
  source               TEXT,
  raw_payload          JSONB,
--   canonical_payload    JSONB,
--   parse_confidence     NUMERIC(5,4),
--   needs_review         BOOLEAN DEFAULT false,
  uploaded_by          TEXT,
  uploaded_at          TIMESTAMP WITH TIME ZONE DEFAULT now(),
  validated            BOOLEAN DEFAULT false,
  validation_notes     TEXT,
  created_at           TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at           TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Indexes for invoice
CREATE INDEX IF NOT EXISTS idx_invoice_gstin_date ON invoice (gstin, invoice_date);
-- CREATE INDEX IF NOT EXISTS idx_invoice_gstin_no ON invoice (gstin, invoice_no_normalized);
-- CREATE INDEX IF NOT EXISTS idx_invoice_needs_review ON invoice (needs_review) WHERE needs_review;
-- CREATE INDEX IF NOT EXISTS idx_invoice_canonical_gin ON invoice USING GIN (canonical_payload);
CREATE INDEX IF NOT EXISTS idx_invoice_lineitems_gin ON invoice USING GIN (line_items);
-- CREATE INDEX IF NOT EXISTS idx_invoice_supplier_trgm ON invoice USING gin (supplier_name gin_trgm_ops);
-- CREATE INDEX IF NOT EXISTS idx_invoice_invoiceno_trgm ON invoice USING gin (invoice_no gin_trgm_ops);


-- ========== GSTR2B ROW ==========
CREATE TABLE IF NOT EXISTS gstr2b_row (
  gstr2b_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  gstin             VARCHAR(15) NOT NULL REFERENCES taxpayer(gstin) ON DELETE RESTRICT,
--   ingestion_batch_id UUID,
  supplier_gstin    VARCHAR(15),
  supplier_name     TEXT,
  invoice_no        TEXT,
--   invoice_no_normalized TEXT GENERATED ALWAYS AS (
--                           lower(regexp_replace(coalesce(invoice_no,''), '\s+|[^a-z0-9]', '', 'gi'))
--                         ) STORED,
  invoice_date      DATE,
  taxable_value     NUMERIC(18,2),
  cgst_amount       NUMERIC(18,2),
  sgst_amount       NUMERIC(18,2),
  igst_amount       NUMERIC(18,2),
  gstr_unique_ref   TEXT,
  raw_payload       JSONB,
  fetched_on        TIMESTAMP WITH TIME ZONE DEFAULT now(),
  created_at        TIMESTAMP WITH TIME ZONE DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_gstr2b_gstin_period ON gstr2b_row (gstin, invoice_date);
-- CREATE INDEX IF NOT EXISTS idx_gstr2b_invoiceno_norm ON gstr2b_row (gstin, invoice_no_normalized);
-- CREATE INDEX IF NOT EXISTS idx_gstr2b_supplier_trgm ON gstr2b_row USING gin (supplier_name gin_trgm_ops);


-- ========== RECONCILIATION ==========
CREATE TABLE IF NOT EXISTS reconciliation (
  recon_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  gstin             VARCHAR(15) NOT NULL REFERENCES taxpayer(gstin) ON DELETE CASCADE,
  fiscal_period     TEXT NOT NULL,
  invoice_id        UUID REFERENCES invoice(invoice_id) ON DELETE SET NULL,
  gstr2b_id         UUID REFERENCES gstr2b_row(gstr2b_id) ON DELETE SET NULL,
  match_status      TEXT CHECK (match_status IN ('matched','partial','mismatch','missing_in_2b','missing_in_books')) NOT NULL DEFAULT 'mismatch',
  match_score       NUMERIC(5,2),
  matched_fields    TEXT,
  mismatch_reason   TEXT,
  itc_eligible      TEXT CHECK (itc_eligible IN ('eligible','partially_held','blocked')) DEFAULT 'eligible',
  flagged_by        TEXT,
  review_required   BOOLEAN DEFAULT false,
  reviewed          BOOLEAN DEFAULT false,
  review_notes      TEXT,
  created_at        TIMESTAMP WITH TIME ZONE DEFAULT now(),
  resolved_at       TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_recon_gstin_period ON reconciliation (gstin, fiscal_period);
CREATE INDEX IF NOT EXISTS idx_recon_invoice ON reconciliation (invoice_id);
CREATE INDEX IF NOT EXISTS idx_recon_gstr2b ON reconciliation (gstr2b_id);
CREATE INDEX IF NOT EXISTS idx_recon_needs_review ON reconciliation (review_required) WHERE review_required;


-- ========== TAX_COMPUTATION ==========
CREATE TABLE IF NOT EXISTS tax_computation (
  computation_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  gstin                 VARCHAR(15) NOT NULL REFERENCES taxpayer(gstin) ON DELETE CASCADE,
  fiscal_period         TEXT NOT NULL,
  snapshot_time         TIMESTAMP WITH TIME ZONE DEFAULT now(),
  total_output_cgst     NUMERIC(18,2) DEFAULT 0,
  total_output_sgst     NUMERIC(18,2) DEFAULT 0,
  total_output_igst     NUMERIC(18,2) DEFAULT 0,
  total_itc_cgst        NUMERIC(18,2) DEFAULT 0,
  total_itc_sgst        NUMERIC(18,2) DEFAULT 0,
  total_itc_igst        NUMERIC(18,2) DEFAULT 0,
  ineligible_itc        NUMERIC(18,2) DEFAULT 0,
  adjustments           JSONB,
  net_payable_cgst      NUMERIC(18,2) DEFAULT 0,
  net_payable_sgst      NUMERIC(18,2) DEFAULT 0,
  net_payable_igst      NUMERIC(18,2) DEFAULT 0,
  rounding              NUMERIC(10,2),
  created_by            TEXT,
  created_at            TIMESTAMP WITH TIME ZONE DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_comp_gstin_period ON tax_computation (gstin, fiscal_period);


-- ========== FILING ==========
CREATE TABLE IF NOT EXISTS filing (
  filing_id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  gstin                     VARCHAR(15) NOT NULL REFERENCES taxpayer(gstin) ON DELETE CASCADE,
  fiscal_period             TEXT NOT NULL,
  form_type                 TEXT NOT NULL,
  status                    TEXT CHECK (status IN ('draft','ready','submitted','failed','filed')) DEFAULT 'draft',
  computation_id            UUID REFERENCES tax_computation(computation_id) ON DELETE SET NULL,
  payload                   JSONB,
  submission_response       JSONB,
  requires_payment          BOOLEAN DEFAULT false,
  total_tax_liability       NUMERIC(18,2),
  total_tax_paid            NUMERIC(18,2) DEFAULT 0,
  arn                       TEXT,
  ack_no                    TEXT,
  created_by                TEXT,
  created_at                TIMESTAMP WITH TIME ZONE DEFAULT now(),
  submitted_at              TIMESTAMP WITH TIME ZONE,
  acknowledged_at           TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_filing_gstin_period ON filing (gstin, fiscal_period);
CREATE INDEX IF NOT EXISTS idx_filing_status ON filing (status);

-- ========== PAYMENT ==========
CREATE TABLE IF NOT EXISTS payment (
  payment_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  gstin              VARCHAR(15) NOT NULL REFERENCES taxpayer(gstin) ON DELETE CASCADE,
  fiscal_period      TEXT,
  filing_id          UUID REFERENCES filing(filing_id) ON DELETE SET NULL,
  challan_no         TEXT,
  cin                TEXT,
  amount             NUMERIC(18,2) NOT NULL,
  cgst_amount        NUMERIC(18,2),
  sgst_amount        NUMERIC(18,2),
  igst_amount        NUMERIC(18,2),
  payment_mode       TEXT,
  bank_ref           TEXT,
  payment_date       DATE,
  raw_challan        JSONB,
  created_at         TIMESTAMP WITH TIME ZONE DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_payment_gstin_date ON payment (gstin, payment_date);
CREATE INDEX IF NOT EXISTS idx_payment_filing ON payment (filing_id);


-- ========== AUDIT LOG ==========
CREATE TABLE IF NOT EXISTS audit_log (
  log_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  gstin         VARCHAR(15),
  actor         TEXT,
  action        TEXT,
  target_table  TEXT,
  target_id     TEXT,
  details       JSONB,
  created_at    TIMESTAMP WITH TIME ZONE DEFAULT now(),
  verified_by   TEXT
);




-- ========== triggers: update updated_at ==========
CREATE OR REPLACE FUNCTION touch_updated_at()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_invoice_updated_at ON invoice;
CREATE TRIGGER trg_invoice_updated_at
BEFORE UPDATE ON invoice
FOR EACH ROW
EXECUTE PROCEDURE touch_updated_at();

