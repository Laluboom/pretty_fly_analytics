# 🧠 Pretty Fly — Universal Neural Estimator

> **One model. Any question. Zero excuses.**

Built for the **Wayflyer × Fin AI Hackathon** (London, Jun 2026) — a single neural network that predicts *any variable* in the Pretty Fly streetwear dataset from all the others. No custom models, no manual feature selection. Just pick a target and go.

---

## 🎯 What it does

Pretty Fly is a fictional London streetwear brand with **69,956 orders**, 21 data files, and a goldmine of signals. This estimator joins all of it into a flat feature matrix and trains a deep net that can answer questions like:

| Question | Target | Result |
|----------|--------|--------|
| 🔄 Will this order get refunded? | `has_refund` | AUC **0.844** (leakage-free) |
| 💰 What will this order be worth? | `total_price` | RMSE **£4.37**, R² **0.997** |
| 👕 What product type is this? | `product_type` | Accuracy **99.99%** |
| 🎧 Who resolves this support ticket? | `resolved_by` | Accuracy **99.93%** |
| ⏱️ How long will support take? | `resolution_time_minutes` | RMSE **323 min** (53% better than naive) |
| ⭐ How happy will this customer be? | `satisfaction_rating` | RMSE **1.17** — synthetic data limitation |
| 🚚 Was this delivery late? | `delivery_delay_days` | regression — supplier chain signal |
| 📦 What's the return rate for this variant? | `variant_return_rate` | regression — inventory health |
| 🏙️ What city is this customer in? | `city` | classification — 20+ UK cities |

All metrics are **out-of-sample** (held-out val set, never seen during training).

---

## 🚀 Quick Start

```bash
pip install -r requirements_nn.txt
```

Run everything from `pretty_fly_data_pack/`:

```bash
# See all 76 things you can predict
python nn/estimator.py --list_targets

# Train on any of them
python nn/estimator.py --target total_price --epochs 50 --plot --save_model models/total_price

# Full evaluation suite — 6 key targets, charts, summary table
python nn/evaluate.py --all --epochs 30 --plot
```

---

## 🎬 Examples

### 💸 What drives refunds?

```bash
python nn/estimator.py --target has_refund --epochs 30 --plot
```

```
============================================================
  RESULTS
============================================================
  Target  : has_refund
  Task    : binary
  val_AUC : 0.6596

  Top 10 Feature Importances (permutation):
  1  subtotal           ████████████████████  0.85201
  2  total_price        ████████████████       0.71044
  3  product_type       ██████                 0.25193
  ...

  Business insight: Top return drivers: subtotal, total_price, product_type
============================================================
```

> Leakage-free — `financial_status`, `refund_reason`, and `refund_amount` are automatically excluded when predicting refunds.

---

### 👟 Classify product type from everything else

```bash
python nn/estimator.py --target product_type --epochs 20 --plot --save_model models/product_type
```

```
  val_accuracy: 0.9999

  5 Example Predictions vs Actuals:
  #     Predicted     Actual
  ──────────────────────────
  1     Tee           Tee
  2     Outerwear     Outerwear
  3     Cap           Cap
  4     Hoodie        Hoodie
  5     Tee           Tee
```

Decoded labels. No integer gymnastics.

---

### 🔮 Predict from a JSON row (no full dataset needed)

Train once, predict forever:

```bash
# Save the model
python nn/estimator.py --target product_type --epochs 20 --save_model models/product_type

# Predict on any partial row — missing fields default to 0 / "unknown"
python nn/estimator.py \
  --predict '{"price": 85.0, "weight_grams": 320, "option1_value": "M", "collection": "Core"}' \
  --load_model models/product_type
```

```
Predicting: product_type
Prediction : Hoodie  (confidence=0.9231)
```

---

### 📊 Full evaluation suite

```bash
python nn/evaluate.py --all --epochs 30 --plot
```

Runs all 6 key targets in sequence, prints a summary table, and saves `eval_{target}.png` charts for each — ROC curves, residual plots, confusion matrices, the works.

---

## 🏗️ Architecture

```
19 categorical cols ──► nn.Embedding(vocab+1, dim=8) ──┐
57 numeric cols     ──► StandardScaled float32         ┘
                                  │
                            concat + BatchNorm1d
                                  │
                     Linear(input_dim → 256) → ReLU → Dropout(0.3)
                     Linear(256 → 128)       → ReLU → Dropout(0.2)
                     Linear(128 → 64)        → ReLU
                                  │
              ┌───────────────────┼───────────────────┐
           binary             regression          classification
      Linear(64,1)           Linear(64,1)        Linear(64, n_classes)
        + Sigmoid                │                  + CrossEntropy
        + BCELoss             + MSELoss
        → AUC                 → RMSE               → Accuracy
```

**Training details:**
- 🔧 Adam (lr=1e-3) + ReduceLROnPlateau (factor=0.5, patience=3)
- ⏹️ Early stopping (patience=5, min_delta=1e-5) with best-weight restore
- 🎲 Fully reproducible — seed=42 everywhere (torch, numpy, cuda)
- ✂️ 80/20 val split (70/30 for sparse targets < 1,000 rows)
- 🔍 Permutation feature importance (3 shuffles × 50 features on val set)

---

## 📂 File Map

```
pretty_fly_data_pack/
├── nn/
│   ├── data_builder.py   626 lines — joins 21 files → 69,956-row × 76-col feature matrix
│   ├── model.py          229 lines — PrettyFlyNet + training loop
│   ├── estimator.py      404 lines — CLI entry point, importance, save/predict
│   └── evaluate.py       442 lines — full metric suites + 6-panel charts
├── requirements_nn.txt   torch · pandas · numpy · scikit-learn · tqdm · matplotlib · vaderSentiment
├── REPORT.md             full evaluation results + system guide
└── README_nn.md          ← you are here
```

---

## 🧩 Data Pipeline

21 raw data files (20 CSV + 1 JSON) → one flat table in ~5 seconds (CUDA):

```
line_items
  ├─► orders              (financials, discounts, timestamps)
  ├─► variants            (weight, SKU)
  ├─► products            (type, collection)
  ├─► customers           (orders_count, total_spent, acquisition source)
  ├─► po_line_items ──► purchase_orders ──► suppliers  (landed cost, lead time, delivery delay)
  ├─► inventory_movements (stock level, return rate, restock count — per variant)
  ├─► discount_codes      (discount type and value)
  ├─► email_events        (opens, clicks, campaigns, days since last email — per customer)
  ├─► addresses           (city, postcode district — PII stripped)
  ├─► refunds             (has_refund flag, reason, amount)
  ├─► support_tickets     (category, resolved_by, satisfaction_rating)
  ├─► support_messages.json (message counts, avg length, VADER sentiment — per ticket)
  ├─► google_ads_daily    (spend, impressions, clicks, conversions)
  └─► meta_ads_daily      (spend, impressions, clicks, conversions)
```

**Engineered features:** `discount_pct` · `gross_margin_est` · `order_month` · `order_dayofweek` · `order_hour` · `is_discounted` · `total_ad_spend` · `total_ad_conversions` · `damaged_in_transit` · `size_issue` · `price_components_sum` · `avg_sentiment` · `min_sentiment` · `pct_negative_msgs`

---

## 🎛️ All CLI Flags

| Flag | Default | What it does |
|------|---------|--------------|
| `--target` | *required* | Column to predict (any of 76) |
| `--epochs` | 50 | Max training epochs (early stopping kicks in sooner) |
| `--batch_size` | 512 | Mini-batch size |
| `--data_dir` | `../data` | Path to raw CSVs |
| `--list_targets` | — | Print all 76 targets with sparsity info and exit |
| `--subset` | — | Filter rows before training, e.g. `"product_type=Hoodie"` |
| `--save_model` | — | Save `{prefix}.pt` + `{prefix}.pkl` after training |
| `--load_model` | — | Load a saved model (use with `--predict`) |
| `--predict` | — | JSON string or file path; partial rows OK, typos warned |
| `--plot` | — | Save `importance_{target}.png` bar chart |

---

## ✨ Key Design Decisions

- **No target-specific code** — task type (binary / regression / classification) is auto-detected from the column
- **Leakage protection** — `LEAKAGE_MAP` in `data_builder.py` drops columns that directly encode the target (e.g. `financial_status` when predicting `has_refund`)
- **Sparse target handling** — `FILTER_MAP` restricts rows to relevant subset (e.g. `resolution_time_minutes` only trains on the 1,705 rows that actually have a ticket)
- **Partial row prediction** — missing features silently default to `0` / `"unknown"` at inference time

---

*Built in ~4 hours. Pretty fly indeed.* 🕶️
