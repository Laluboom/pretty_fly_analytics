# Session Brief
**Project:** Pretty Fly Universal Neural Estimator
**Hackathon:** Wayflyer × Fin AI — Build the Future of E-commerce (3–5 Jun 2026, London)
**Dataset:** 24 months of data for fictional London streetwear brand Pretty Fly

---

## What Was Built

A single neural network CLI that predicts **any variable in the dataset from all others**.
Pick a target column, train, get a metric + ranked feature importances + example predictions.

```bash
python nn/estimator.py --target has_refund --epochs 30 --plot --save_model models/has_refund
python nn/estimator.py --predict '{"price": 85.0, "collection": "Core"}' --load_model models/has_refund
python nn/evaluate.py --all --epochs 30 --plot
```

---

## File Map

```
pretty_fly_data_pack/
├── nn/
│   ├── data_builder.py   353 lines — joins 10 tables → 69,956-row feature matrix
│   ├── model.py          229 lines — PrettyFlyNet + training loop
│   ├── estimator.py      364 lines — CLI entry point
│   └── evaluate.py       442 lines — full metric suites + 6-panel charts
├── requirements_nn.txt   — torch, pandas, numpy, scikit-learn, tqdm, matplotlib
├── PRD_neural_estimator.md
├── TODO.md               — full phase log with completion briefs
├── REPORT.md             — evaluation results + system guide
└── README_nn.md          — usage examples
```

---

## Commits (12 total this session)

| Hash | What |
|------|------|
| `8673a15` | Initial dataset commit |
| `2422bb4` | PRD + TODO + plan docs |
| `1f03011` | Phase 1: `data_builder.py` — 10-table join, feature engineering, encoding |
| `e63e301` | Phase 2: `model.py` — PrettyFlyNet, training loop, early stopping |
| `a32b55a` | Phase 3: `estimator.py` — CLI, permutation importance, results printer |
| `03ecdc5` | Phase 4: all 5 validation tests pass, 20/20 data rules clean |
| `de2bf7d` | Phase 5: `--save_model`, `--predict`, `--plot`, `README_nn.md` |
| `e1e4152` | Phase 6: `evaluate.py`, `REPORT.md`, 6 evaluation charts |
| `8b0f605` | Improvements audit — 10 grounded items in TODO |
| `1f3278d` | Fix B1 B2 M2 B3 — leakage, zero-inflation, reproducibility, bar scaling |
| `f156bc9` | Fix M1 M3 M4 — early stopping min_delta, LR scheduler, sparse val split |
| `1cc3289` | TODO updated with all fix briefs |

---

## Architecture

```
15 categorical cols → nn.Embedding(vocab+1, dim=8)  ┐
35 numeric cols     → StandardScaled float32        ┘ concat
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

## Key Evaluation Results (30 epochs, CUDA)

| Target | Task | Metric | vs Naive |
|--------|------|--------|---------|
| `has_refund` | binary | AUC **0.66** | real signal (was 1.0 — leakage fixed) |
| `total_price` | regression | RMSE **6.4**, R²=**0.997** | 94% better than predict-mean |
| `product_type` | classification | acc **0.9999** | +46 pp vs 54% baseline |
| `resolved_by` | classification | acc **0.9993** | +2 pp vs 98% baseline |
| `resolution_time_minutes` | regression | RMSE **323 min** | honest — ticket rows only (was fake 50 min) |
| `satisfaction_rating` | regression | RMSE **1.17** | ≈ naive — needs sentiment features |

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

**Engineered features:** `discount_pct`, `gross_margin_est`, `order_month/dayofweek/hour`, `is_discounted`, `total_ad_spend`, `total_ad_conversions`

---

## Bugs Fixed This Session

| ID | Fix | Impact |
|----|-----|--------|
| B1 | `LEAKAGE_MAP` — drops `financial_status`/`refund_reason`/`refund_amount` when target is `has_refund` | AUC 1.0→0.66 (honest) |
| B2 | `FILTER_MAP` — filters `resolution_time_minutes` to `has_ticket==1` rows (1,705 not 69,956) | Removes 97.6% structural zeros |
| B3 | Relative bar scaling `int(20 * score / top_score)` in importance chart | Features no longer all show equal-width bars |
| M1 | `min_delta=1e-5` in early stopping | Near-perfect models stop at epoch 8, not 50 |
| M2 | `torch.manual_seed(seed)` at start of `train_model()` | Bit-exact reproducibility across runs |
| M3 | `ReduceLROnPlateau` after Adam | LR halves on plateaus; `lr=X ↓` shown in epoch line |
| M4 | `test_size=0.3` when `len(y) < 1000` | `satisfaction_rating` val set: 107→162 rows |

---

## Remaining Improvements (in TODO)

| ID | Item | Effort |
|----|------|--------|
| F1 | Load `support_messages.json` → add msg_count, escalation count, response time features → fixes satisfaction_rating | 60 min |
| F3 | Simplify `gross_margin_est` fallback (0% missing, formula wrong anyway) | 5 min |
| U1 | Warn on unknown keys in `--predict` JSON input | 10 min |
| U3 | Build feature table once in `evaluate.py --all` (currently rebuilt 6×) | 5 min |

---

## Train/Test Split — How Accuracy Is Measured

Every `train_model()` call splits data **before training**:
- `test_size=0.2` (or 0.3 for sparse targets) with `random_state=42`
- Stratified for binary/classification targets
- **Model never sees val rows during training**
- Final metric (AUC / accuracy / RMSE) is computed only on the held-out val set
- `val_idx` is returned and used for permutation importance too

The reported metrics are all out-of-sample.

---

## How to Run

```bash
# Install
pip install -r requirements_nn.txt

# List all 51 valid targets
python nn/estimator.py --list_targets

# Train + evaluate + save
python nn/estimator.py --target total_price --epochs 50 --save_model models/total_price --plot

# Full evaluation suite with charts (6 key targets)
python nn/evaluate.py --all --epochs 30 --plot

# Single target with full metrics
python nn/evaluate.py --target satisfaction_rating --epochs 30 --plot

# Predict from saved model (partial row OK — missing fields default to 0/unknown)
python nn/estimator.py \
  --predict '{"price": 85.0, "collection": "Core", "option1_value": "M"}' \
  --load_model models/product_type
```
