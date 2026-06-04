# Session Brief
**Project:** Pretty Fly Universal Neural Estimator
**Hackathon:** Wayflyer × Fin AI — Build the Future of E-commerce (3–5 Jun 2026, London)
**Dataset:** 24 months of data for fictional London streetwear brand Pretty Fly

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
│   ├── data_builder.py   366 lines — joins 10 tables → 69,956-row feature matrix
│   ├── model.py          229 lines — PrettyFlyNet + training loop
│   ├── estimator.py      379 lines — CLI entry point
│   └── evaluate.py       442 lines — full metric suites + 6-panel charts
├── requirements_nn.txt   — torch, pandas, numpy, scikit-learn, tqdm, matplotlib
├── TODO.md               — full phase log + pending work (Phases 7–9)
├── REPORT.md             — evaluation results + system guide
├── README_nn.md          — usage examples (colourful, up to date)
└── baby_guide.md         — plain-English explanations (gitignored, local only)
```

---

## Architecture

```
15 categorical cols → nn.Embedding(vocab+1, dim=8)  ┐
38 numeric cols     → StandardScaled float32        ┘ concat
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

**Join chain** (base = `line_items`, 69,956 rows):
```
line_items → orders → variants → products → customers
           → po_line_items (landed cost per unit)
           → refunds (has_refund flag, reason, amount)
           → support_tickets (category, resolved_by, satisfaction_rating)
           → google_ads_daily (aggregated by campaign+date)
           → meta_ads_daily  (aggregated by campaign+date)
```

**Engineered features:** `discount_pct`, `gross_margin_est`, `order_month/dayofweek/hour`, `is_discounted`, `total_ad_spend`, `total_ad_conversions`, `price_components_sum`, `damaged_in_transit`, `size_issue`

**Leakage protection (`LEAKAGE_MAP`):** 6 entries — drops columns that algebraically encode the target (e.g. `refund_reason` dropped when predicting `damaged_in_transit`)

**Row filtering (`FILTER_MAP`):** 3 entries — restricts training to meaningful rows (e.g. `has_refund==1` for damage/size targets, `has_ticket==1` for resolution time)

**Inference:** `load_and_predict()` auto-computes all engineered features from raw input fields — user never needs to supply `price_components_sum`, `discount_pct`, etc.

---

## Current Evaluation Results (30 epochs, CUDA)

| Target | Task | Metric | Notes |
|--------|------|--------|-------|
| `has_refund` | binary | AUC **0.83** | Leakage-free — refund_reason/financial_status excluded |
| `total_price` | regression | RMSE **£4.37** | 96% better than predict-mean; price_components_sum fix applied |
| `product_type` | classification | acc **0.9999** | +46 pp vs 54% baseline |
| `resolved_by` | classification | acc **0.9993** | +2 pp vs 98% baseline |
| `resolution_time_minutes` | regression | RMSE **323 min** | Ticket rows only (1,705) — honest |
| `satisfaction_rating` | regression | RMSE **1.17** | ≈ naive — needs Phase 8 sentiment features |
| `damaged_in_transit` | binary | AUC **0.80** | New target — 10,025 refund rows |
| `size_issue` | binary | AUC **~0.75** | New target — size complaint returns |

---

## All Valid Targets (55 columns)

Everything in the feature matrix is a valid `--target`. Notable ones:

| Target | Task | What it answers |
|--------|------|-----------------|
| `has_refund` | binary | Will this order be returned? |
| `damaged_in_transit` | binary | Was the parcel damaged? |
| `size_issue` | binary | Was it a sizing problem? |
| `total_price` | regression | What will this order cost? |
| `gross_margin_est` | regression | What margin does this order carry? |
| `landed_cost_per_unit_gbp` | regression | What did this variant cost us? |
| `satisfaction_rating` | regression | How happy will this customer be? |
| `resolution_time_minutes` | regression | How long will support take? |
| `product_type` | classification | What kind of product is this? |
| `resolved_by` | classification | Bot or human support? |
| `option1_value` | classification | What size is this? |
| `option2_value` | classification | What colour is this? |

---

## Commit History

| Hash | What |
|------|------|
| `8673a15` | Initial dataset commit |
| `1f03011` | Phase 1: `data_builder.py` |
| `e63e301` | Phase 2: `model.py` |
| `a32b55a` | Phase 3: `estimator.py` |
| `03ecdc5` | Phase 4: validation complete |
| `de2bf7d` | Phase 5: `--save_model`, `--predict`, `--plot` |
| `e1e4152` | Phase 6: `evaluate.py`, `REPORT.md` |
| `1f3278d` | Fix B1–B3 — leakage, zero-inflation, bar scaling |
| `f156bc9` | Fix M1–M4 — early stopping, LR scheduler, sparse split |
| `e5de42d` | Add `.gitignore` |
| `f46d26e` | Remove tracked artifacts |
| `d9dc544` | Rewrite `README_nn.md` |
| `30367bd` | Add `damaged_in_transit`, `size_issue`; fix leakage ordering bug |
| `831f704` | Add Phase 7/8/9 to TODO |
| `1564e1c` | Fix `total_price`: `price_components_sum` + auto-compute at inference |

---

## Remaining Work (TODO Phases 7–9)

| Phase | Item | Effort | Status |
|-------|------|--------|--------|
| 7 | `--subset "product_type=Hoodie"` flag — train on any data slice | 20 min | ⬜ |
| 8 | Sentiment features from `support_messages.json` — fixes `satisfaction_rating` | 60 min | ⬜ |
| 9 | LLM recommendation layer on `--predict` via Claude API (`--recommend` flag) | 90 min | ⬜ |
| — | `U1` — warn on unknown keys in `--predict` JSON | 10 min | ⬜ |
| — | `U3` — build feature table once in `evaluate.py --all` | 5 min | ⬜ |
| — | `F3` — simplify `gross_margin_est` fallback (dead code) | 5 min | ⬜ |

---

## How to Run

```bash
# Install
pip install -r requirements_nn.txt

# List all valid targets
python nn/estimator.py --list_targets

# Train + save + plot
python nn/estimator.py --target total_price --epochs 50 --save_model models/total_price --plot

# Predict from saved model (partial row OK — missing fields default to 0/unknown)
python nn/estimator.py \
  --predict '{"price": 129.17, "product_type": "Sweatpants", "collection": "Autumn 25"}' \
  --load_model models/total_price

# Full evaluation suite (6 key targets, charts)
python nn/evaluate.py --all --epochs 30 --plot

# Single target full evaluation
python nn/evaluate.py --target damaged_in_transit --epochs 30 --plot
```
