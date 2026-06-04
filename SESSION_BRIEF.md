# Session Brief
**Project:** Pretty Fly Universal Neural Estimator
**Hackathon:** Wayflyer × Fin AI — Build the Future of E-commerce (3–5 Jun 2026, London)
**Dataset:** 24 months of data for fictional London streetwear brand Pretty Fly
**Validation:** `validate.py data/` — 20/20 rules pass, 0 failed, data untouched

---

## What Was Built

A single neural network CLI that predicts **any variable in the dataset from all others**.
Pick a target column, train, get a metric + ranked feature importances + example predictions.

```bash
python nn/estimator.py --target damaged_in_transit --epochs 30 --plot --save_model models/damaged_in_transit
python nn/estimator.py --predict '{"price": 129.17, "collection": "Autumn 25"}' --load_model models/damaged_in_transit
python nn/evaluate.py --all --epochs 30 --plot
```

---

## File Map

```
pretty_fly_data_pack/
├── nn/
│   ├── data_builder.py   579 lines — joins all 21 tables → 69,956-row × 73-col feature matrix
│   ├── model.py          229 lines — PrettyFlyNet + training loop
│   ├── estimator.py      379 lines — CLI entry point
│   └── evaluate.py       442 lines — full metric suites + 6-panel charts
├── data/                 21 files (20 CSV + 1 JSON) — all loaded
├── requirements_nn.txt   — torch, pandas, numpy, scikit-learn, tqdm, matplotlib
├── TODO.md               — full phase log + pending work (Phases 7–9)
├── REPORT.md             — evaluation results + system guide
├── README_nn.md          — usage examples (colourful, up to date)
└── baby_guide.md         — plain-English explanations (gitignored, local only)
```

---

## Architecture

```
19 categorical cols → nn.Embedding(vocab+1, dim=8)  ┐
54 numeric cols     → StandardScaled float32        ┘ concat
                          ↓
              BatchNorm1d → Dense(256) → ReLU → Dropout(0.3)
                          → Dense(128) → ReLU → Dropout(0.2)
                          → Dense(64)  → ReLU
                          ↓
         binary: Linear(64,1)+Sigmoid → BCELoss    → AUC
      regression: Linear(64,1)        → MSELoss    → RMSE
  classification: Linear(64,n)        → CrossEntropy → Accuracy
```

**Training:** Adam lr=1e-3 · ReduceLROnPlateau(factor=0.5, patience=3) · early stopping(patience=5, min_delta=1e-5) · 80/20 split (70/30 if <1000 rows) · seed=42 everywhere

---

## Data Pipeline

**All 21 files loaded** (base = `line_items`, 69,956 rows → 73 columns):

```
line_items → orders → variants → products → customers
           → po_line_items → purchase_orders → suppliers
           → inventory_movements
           → discount_codes
           → refunds
           → support_tickets → support_messages.json
           → addresses
           → email_events
           → google_ads_daily
           → meta_ads_daily
```

**Previously unused files now integrated (Phase 8):**

| File | New Features |
|------|-------------|
| `discount_codes.csv` | `discount_type` (cat), `discount_value` |
| `purchase_orders.csv` + `suppliers.csv` | `supplier_country` (cat), `lead_time_days`, `delivery_delay_days` |
| `inventory_movements.csv` | `variant_latest_stock`, `variant_return_rate`, `variant_restock_count` |
| `email_events.csv` | `email_open_count`, `email_click_count`, `email_campaign_count`, `days_since_last_email` |
| `addresses.csv` | `city` (cat), `postcode_district` (cat) — PII stripped, area-level only |
| `support_messages.json` | `msg_count`, `customer_msg_count`, `avg_customer_msg_length`, `n_escalations`, `response_time_first_seconds` |

**Skipped (no useful per-order join):** `bank_transactions.csv`, `email_campaigns.csv`, `product_collections.csv`, `collections.csv`

**Engineered features:** `discount_pct`, `gross_margin_est`, `order_month/dayofweek/hour`, `is_discounted`, `total_ad_spend`, `total_ad_conversions`, `price_components_sum`, `damaged_in_transit`, `size_issue`

**Leakage protection (`LEAKAGE_MAP`):** 7 entries — drops columns that encode the target
**Row filtering (`FILTER_MAP`):** 7 entries — restricts to meaningful rows (e.g. `has_refund==1` for damage targets)
**Inference:** `load_and_predict()` auto-computes all engineered features from raw inputs

---

## Current Evaluation Results (30 epochs, CUDA)

| Target | Task | Metric | Notes |
|--------|------|--------|-------|
| `has_refund` | binary | AUC **0.844** | Leakage-free; +0.01 from Phase 8 features |
| `total_price` | regression | RMSE **£4.37** | 96% better than naive; `price_components_sum` fix |
| `product_type` | classification | acc **1.000** | Perfect — weight+size+margin separate categories |
| `resolved_by` | classification | acc **1.000** | Perfect — resolution_time dominates |
| `resolution_time_minutes` | regression | RMSE **323 min** | Ticket rows only (1,705) — honest |
| `satisfaction_rating` | regression | RMSE **≈1.17** | Dataset limitation — synthetic msgs have zero variance in escalations/response_time |
| `damaged_in_transit` | binary | AUC **0.829** | +0.03 from Phase 8; `variant_return_rate` new signal |
| `size_issue` | binary | AUC **~0.75** | Refund rows only (10,025) |

---

## All Valid Targets (73 columns)

Everything in the feature matrix is a valid `--target`. Notable ones:

| Target | Task | What it answers |
|--------|------|-----------------|
| `has_refund` | binary | Will this order be returned? |
| `damaged_in_transit` | binary | Was the parcel damaged in transit? |
| `size_issue` | binary | Was it a size fit problem? |
| `total_price` | regression | What will this order cost? |
| `gross_margin_est` | regression | What margin does this order carry? |
| `landed_cost_per_unit_gbp` | regression | What did this variant cost us? |
| `delivery_delay_days` | regression | Was this supplier delivery late? |
| `variant_return_rate` | regression | What fraction of this variant gets returned? |
| `variant_latest_stock` | regression | Current stock level of this variant? |
| `satisfaction_rating` | regression | How happy will this customer be? |
| `resolution_time_minutes` | regression | How long will support take? |
| `product_type` | classification | What kind of product is this? |
| `resolved_by` | classification | Bot or human support? |
| `option1_value` | classification | What size is this? |
| `option2_value` | classification | What colour is this? |
| `supplier_country` | classification | Which supplier made this? |
| `city` | classification | What city is this customer in? |

---

## Commit History

| Hash | What |
|------|------|
| `8673a15` | Initial dataset commit |
| `1f03011` | Phase 1: `data_builder.py` |
| `e63e301` | Phase 2: `model.py` |
| `a32b55a` | Phase 3: `estimator.py` |
| `03ecdc5` | Phase 4: validation — 20/20 rules pass |
| `de2bf7d` | Phase 5: `--save_model`, `--predict`, `--plot` |
| `e1e4152` | Phase 6: `evaluate.py`, `REPORT.md` |
| `1f3278d` | Fix B1–B3 — leakage, zero-inflation, bar scaling |
| `f156bc9` | Fix M1–M4 — early stopping, LR scheduler, sparse split |
| `e5de42d` | Add `.gitignore` |
| `f46d26e` | Remove tracked artifacts |
| `d9dc544` | Rewrite `README_nn.md` |
| `30367bd` | Add `damaged_in_transit`, `size_issue`; fix leakage ordering bug |
| `831f704` | Add Phases 7/8/9 to TODO |
| `1564e1c` | Fix `total_price`: `price_components_sum` + auto-compute at inference |
| `c914576` | Update SESSION_BRIEF.md |
| `59e6066` | Phase 8: all 21 files integrated — 53→73 columns |

---

## Remaining Work (TODO Phases 7–9)

| Phase | Item | Effort | Status |
|-------|------|--------|--------|
| 7 | `--subset "product_type=Hoodie"` flag — train on any data slice | 20 min | ⬜ |
| 9 | LLM recommendation layer on `--predict` via Claude API (`--recommend` flag) | 90 min | ⬜ |
| — | `U1` — warn on unknown keys in `--predict` JSON | 10 min | ⬜ |
| — | `U3` — build feature table once in `evaluate.py --all` | 5 min | ⬜ |
| — | `F3` — simplify `gross_margin_est` fallback (dead code) | 5 min | ⬜ |

**Phase 8 complete** — all 21 data files loaded, 19 new features added, 20/20 validation rules still pass.
**`satisfaction_rating` note:** Synthetic dataset has zero variance in `n_escalations` (all 0) and `response_time_first_seconds` (all 120s). Message features don't improve it — fundamental data limitation, not a code issue.

---

## How to Run

```bash
# Install
pip install -r requirements_nn.txt

# List all 73 valid targets
python nn/estimator.py --list_targets

# Train + save + plot
python nn/estimator.py --target total_price --epochs 50 --save_model models/total_price --plot

# Predict from saved model (partial row OK — engineered features auto-computed)
python nn/estimator.py \
  --predict '{"price": 129.17, "product_type": "Sweatpants", "collection": "Autumn 25"}' \
  --load_model models/total_price

# Full evaluation suite (6 key targets, charts)
python nn/evaluate.py --all --epochs 30 --plot

# Single target full evaluation
python nn/evaluate.py --target damaged_in_transit --epochs 30 --plot

# Validate data integrity (20 rules)
python validate.py data/
```
