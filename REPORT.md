# Pretty Fly — Neural Estimator: Evaluation Report & System Guide

**Generated from live evaluation runs | 30 epochs | CUDA | Pretty Fly dataset (Jun 2024 – May 2026)**

---

## Contents
1. [What This System Does](#1-what-this-system-does)
2. [How It Works — End to End](#2-how-it-works--end-to-end)
3. [Evaluation Results — All 6 Targets](#3-evaluation-results--all-6-targets)
4. [Feature Importance Findings](#4-feature-importance-findings)
5. [Worked Examples](#5-worked-examples)
6. [Caveats and Known Limitations](#6-caveats-and-known-limitations)
7. [Quick Reference](#7-quick-reference)

---

## 1. What This System Does

PrettyFlyNet is a **universal neural estimator**: given any column in the Pretty Fly dataset as a prediction target, it trains a feed-forward neural network using all other columns as inputs and outputs:

- A validation metric (AUC / accuracy / RMSE depending on target type)
- A ranked list of the top features driving that prediction
- Example predictions vs actuals
- A one-line business interpretation

The same architecture handles binary classification, multi-class classification, and regression — the output head and loss function are auto-selected based on the target column's data type.

---

## 2. How It Works — End to End

### Step 1: Data Pipeline (`nn/data_builder.py`)

Ten CSV tables are loaded and joined into a single flat feature matrix.

**Join chain (base = `line_items.csv`, 69,956 rows):**

```
line_items (69,956 rows)
  ├── LEFT JOIN orders          on order_id
  ├── LEFT JOIN variants        on variant_id       → size, colour, price, weight
  ├── LEFT JOIN products        on product_id       → type, gender, collection
  ├── LEFT JOIN customers       on customer_id      → LTV, acquisition, country
  ├── LEFT JOIN po_line_items   on variant_id       → landed cost per unit
  ├── LEFT JOIN refunds         on order_id         → has_refund flag, reason, amount
  ├── LEFT JOIN support_tickets on order_id         → ticket category, resolved_by, rating
  ├── LEFT JOIN google_ads      on utm_campaign+date → daily ad spend/impressions/clicks
  └── LEFT JOIN meta_ads        on utm_campaign+date → same for Meta
```

**Result:** 69,956 rows × 51 columns (50 features + 1 target per run)

**Engineered features added on top of raw columns:**

| Feature | Formula |
|---------|---------|
| `discount_pct` | `total_discounts / subtotal`, clipped to [0, 1] |
| `gross_margin_est` | `(price - landed_cost) / price` |
| `order_month` | Month of order (1–12) |
| `order_dayofweek` | Day of week (0=Mon, 6=Sun) |
| `order_hour` | Hour of order (0–23) |
| `is_discounted` | 1 if discount code applied, else 0 |
| `total_ad_spend` | `google_spend + meta_spend` on that campaign-day |
| `total_ad_conversions` | `google_conversions + meta_conversions` |

**Encoding:**
- 15 categorical columns → `LabelEncoder` (integer codes)
- 35 numeric columns → `StandardScaler` (zero mean, unit variance)

**Sparse targets** (e.g. `satisfaction_rating` — only 539 non-null rows) are handled by dropping null rows before training. The same architecture runs on 539 or 69,956 rows.

---

### Step 2: Target Detection

`prepare_for_target(df, target_col)` auto-detects task type:

| Condition | Task type | Loss | Metric |
|-----------|-----------|------|--------|
| 2 unique values or bool | `binary` | BCELoss | AUC |
| object dtype or int with 2 < unique ≤ 20 | `classification` | CrossEntropyLoss | Accuracy |
| continuous numeric | `regression` | MSELoss | RMSE |

Classification targets are label-encoded and decoded back to readable names in output.

---

### Step 3: Network Architecture (`nn/model.py`)

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT                                                      │
│  15 categorical columns → nn.Embedding(vocab+1, dim=8) each │
│  35 numeric columns     → StandardScaled float32            │
└──────────────────┬──────────────────────────────────────────┘
                   │ Concatenate all embeddings + numerics
                   ▼
           BatchNorm1d(input_dim)
                   │
           Linear(input_dim → 256) → ReLU → Dropout(0.3)
                   │
           Linear(256 → 128) → ReLU → Dropout(0.2)
                   │
           Linear(128 → 64) → ReLU
                   │
    ┌──────────────┼──────────────┐
    ▼              ▼              ▼
 binary         regression   classification
 Linear(64,1)   Linear(64,1) Linear(64, n_classes)
 + Sigmoid       (raw)        (logits → softmax)
 → BCELoss      → MSELoss    → CrossEntropyLoss
 → AUC          → RMSE       → Accuracy
```

**Parameters:** ~80,000–85,000 (varies slightly with target, as target column is excluded from features)

**Training config:**
- Optimiser: Adam (lr=1e-3)
- Train/val split: 80% / 20% (stratified for binary and classification)
- Early stopping: patience=5 on validation loss, restores best weights
- Device: CUDA (auto-detected), falls back to CPU

---

### Step 4: Feature Importance (`nn/estimator.py`)

Permutation importance: for each of the 50 feature columns, the column is shuffled 3 times on the validation set. The average increase in loss is the importance score.

```
importance(feature_i) = mean over 3 shuffles of:
    loss(model, shuffled feature_i) − loss(model, original)
```

- A high score means the model heavily relies on that feature — removing it hurts a lot.
- A score near 0 means the model doesn't use that feature for this target.
- Cat and numeric features are shuffled in their respective tensors (`X_cat[:,i]` vs `X_num[:,j]`).

---

## 3. Evaluation Results — All 6 Targets

All results from 30-epoch runs on CUDA. Evaluation charts saved as `eval_{target}.png`.

---

### 3.1 `has_refund` — Binary Classification

> **Business question:** Which orders will be refunded?

| Metric | Value |
|--------|-------|
| AUC | **1.0000** |
| Average Precision | 1.0000 |
| Accuracy | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1 | 1.0000 |
| Positive rate (actual) | 14.33% |

**Confusion matrix:**
```
              Pred 0   Pred 1
  Actual 0    11,987        0
  Actual 1         0    2,005
```

**Note on perfect scores:** The dataset encodes `financial_status = "partially_refunded"` for all refunded orders — this column directly labels the target. The model correctly identifies this as the dominant feature (Δ loss = 2.11). In a production setting you would exclude leakage columns; here the target is a flag engineered from the same data, so perfect performance is structurally expected.

---

### 3.2 `satisfaction_rating` — Regression (Sparse)

> **Business question:** What predicts whether a customer gives a high or low rating?

| Metric | Value | Context |
|--------|-------|---------|
| RMSE | **1.1716** | Rating scale is 1–5 |
| MAE | 0.9851 | Avg error < 1 star |
| R² | −0.0069 | Near-zero predictive power |
| Median Abs Error | 0.8200 | Half predictions within 0.82 stars |
| Within ±20% | 41.67% | |
| Naive RMSE (predict mean) | 1.1675 | |
| **Improvement over naive** | **−0.3%** | Not better than predict-mean |

**Interpretation:** With only 539 training rows and satisfaction driven largely by subjective factors not in the data (product quality, delivery experience, personal mood), the model effectively learns to predict the mean. The feature importances are noisy across runs — no single feature dominates. This is the hardest target in the dataset; **more data or richer features (e.g. message sentiment from `support_messages.json`) would be needed to improve it.**

---

### 3.3 `total_price` — Regression

> **Business question:** What determines the value of a line item order?

| Metric | Value | Context |
|--------|-------|---------|
| RMSE | **6.42** | Orders range £12–£690 |
| MAE | 4.44 | Average error ~£4.44 |
| R² | **0.9968** | Explains 99.7% of variance |
| Median Abs Error | 3.20 | Half predictions within £3.20 |
| Within ±10% | **95.5%** | |
| Within ±20% | 99.5% | |
| Naive RMSE (predict mean) | 113.32 | |
| **Improvement over naive** | **94.3%** | |

**Interpretation:** Near-perfect prediction. `subtotal` (Δ loss = 14,450) is overwhelmingly dominant — `total_price = subtotal + shipping + tax - discounts` is nearly an algebraic identity given the features available. This confirms data consistency (validates the reconciliation). The more interesting signal comes from `discount_pct` at rank 4 — discount rate has measurable independent influence on final price.

---

### 3.4 `product_type` — Multi-class Classification (6 classes)

> **Business question:** Can we identify what kind of product an order is for from all other signals?

| Metric | Value |
|--------|-------|
| Accuracy | **0.9999** |
| Macro F1 | 0.9998 |
| Weighted F1 | 0.9999 |
| Baseline (majority class: Tee) | 53.64% |
| **Improvement over baseline** | **+46.4 pp** |

**Per-class breakdown (validation set):**
```
             Precision  Recall    F1   Support
Cap            1.000    1.000   1.000    1,203
Hoodie         1.000    1.000   1.000    2,761
Outerwear      1.000    1.000   1.000      431
Sweatpants     1.000    1.000   1.000    1,265
Tee            1.000    1.000   1.000    7,506
Trainer        1.000    1.000   1.000      826
```

One Hoodie was misclassified as Outerwear. All other 13,991 predictions correct.

**Top features:** `gross_margin_est` (Δ loss = 1.574), `option1_value` (size — trainers have UK sizes, apparel has S/M/L), `weight_grams`. Products are physically distinct enough that these signals perfectly separate them.

---

### 3.5 `resolved_by` — Multi-class Classification (3 classes: bot / human / none)

> **Business question:** Can we predict whether a ticket will be escalated to a human?

| Metric | Value |
|--------|-------|
| Accuracy | **0.9993** |
| Macro F1 | 0.9801 |
| Weighted F1 | 0.9993 |
| Baseline (majority: "none" = no ticket) | 97.56% |
| **Improvement over baseline** | **+2.4 pp** |

**Per-class breakdown:**
```
          Precision  Recall    F1   Support
bot          0.934    1.000   0.966      142
human        1.000    0.950   0.974      199
none         1.000    1.000   1.000   13,651
```

**Confusion matrix:**
```
         bot   human   none
bot      142       0      0
human     10     189      0
none       0       0  13,651
```

10 human tickets were predicted as bot. Bot prediction is perfect (142/142).

**Top features:** `resolution_time_minutes` (Δ loss = 0.385) is far and away the strongest predictor — bots resolve instantly while humans take longer. `support_channel` and `ticket_category` are secondary drivers.

**Business insight:** The 10 misclassified tickets (human predicted as bot) are likely complex tickets that look like bot-resolvable categories but required human escalation due to nuance in the conversation. These are exactly the tickets worth reviewing.

---

### 3.6 `resolution_time_minutes` — Regression

> **Business question:** How long will this ticket take to resolve?

| Metric | Value | Context |
|--------|-------|---------|
| RMSE | **50.30** | Minutes |
| MAE | 5.18 | Average error ~5 min |
| R² | **0.7775** | Explains 77.8% of variance |
| Median Abs Error | **0.045** | Median prediction within 3 seconds |
| Within ±10% | 77.7% | |
| Within ±20% | 93.0% | |
| Naive RMSE (predict mean) | 106.65 | |
| **Improvement over naive** | **52.8%** | |

The RMSE of 50 minutes is high due to a long tail of very complex tickets (max error: 941 min). The **median error of 0.045 minutes** (3 seconds) tells the real story — most predictions are extremely accurate. The high RMSE is pulled by a small number of outlier tickets that took many hours.

**Top features:** `resolved_by` (Δ loss = 8,186) — bots take seconds, humans take minutes. `ticket_category` and `support_channel` are secondary.

---

## 4. Feature Importance Findings

Summary of what drives each target, with business interpretation:

| Target | #1 Feature | #2 Feature | #3 Feature | Business Takeaway |
|--------|-----------|-----------|-----------|-------------------|
| `has_refund` | `financial_status` | `refund_reason` | `refund_amount` | Status/reason directly encode the label (expected) |
| `satisfaction_rating` | `support_channel` | `financial_status` | `refund_reason` | Channel and refund experience shape ratings most |
| `total_price` | `subtotal` | `total_shipping` | `total_tax` | Algebraic identity — strong internal consistency |
| `product_type` | `gross_margin_est` | `option1_value` | `variant_price` | Margin and size type perfectly separate product categories |
| `resolved_by` | `resolution_time_minutes` | `support_channel` | `ticket_category` | Resolution time is the clearest signal of human vs bot |
| `resolution_time_minutes` | `resolved_by` | `ticket_category` | `support_channel` | Bot/human distinction dominates resolution time |

**Cross-target patterns worth noting:**
- `support_channel` appears in the top 3 for both `satisfaction_rating` and `resolved_by` — Instagram DM tickets behave differently from email/chat
- `financial_status` (partially_refunded vs paid) leaks into satisfaction and other targets because refund experience co-varies with customer sentiment
- `ticket_category` and `resolved_by` are mutually predictive — they form a tight cluster of support signals

---

## 5. Worked Examples

### Example A: Predicting total_price on real orders

```
#         Predicted    Actual
1             60.41     55.94   (err: £+4.47)
2            129.11    125.00   (err: £+4.11)
3             41.01     39.94   (err: £+1.07)
4            121.45    112.50   (err: £+8.95)
5            182.82    187.99   (err: £-5.17)
```
R² = 0.9968 — prices predicted to within ~£4.44 on average across 14,000 validation orders.

---

### Example B: Classifying product_type

```
#     Predicted     Actual
1           Tee        Tee   ✓
2     Outerwear  Outerwear   ✓
3           Tee        Tee   ✓
4           Cap        Cap   ✓
5        Hoodie     Hoodie   ✓
```
Only 1 error in 13,992 val predictions (1 Hoodie classified as Outerwear).

---

### Example C: Predicting from a saved model (CLI)

```bash
# Train and save
python nn/estimator.py --target product_type --epochs 20 --save_model models/product_type

# Predict on a new partial row
python nn/estimator.py \
  --predict '{"price": 85.0, "weight_grams": 320, "option1_value": "M"}' \
  --load_model models/product_type
```

**Output:**
```
Predicting: product_type
Input features used: 14 categorical + 36 numeric
Prediction : Hoodie  (confidence=0.9231)
```

Missing features default to `0` (numeric) or `"unknown"` (categorical).

---

### Example D: Running a full evaluation with plots

```bash
python nn/evaluate.py --target total_price --epochs 30 --plot
```

Produces:
- Full metrics table (RMSE, MAE, R², within-±10%, residual distribution)
- `eval_total_price.png` — 6-panel chart: feature importance, predicted vs actual scatter, residual histogram, residual vs actual, absolute error distribution

---

## 6. Caveats and Known Limitations

### 6.1 Data leakage in `has_refund`
`financial_status = "partially_refunded"` directly encodes whether a refund occurred. The model achieves AUC=1.0 trivially by learning this. For a genuine refund *predictor* (useful before a refund happens), remove `financial_status`, `refund_reason`, and `refund_amount` from features.

### 6.2 `satisfaction_rating` has insufficient data
539 rows is too few for a 50-feature model to learn meaningful patterns. The model performs at parity with predict-mean (R² ≈ 0). Useful directions:
- Add sentiment features from `support_messages.json` (message tone, length, escalation count)
- Reduce feature count for this target to avoid overfitting

### 6.3 Permutation importance is post-hoc
Importance scores measure influence on validation loss, not causal relationships. Two correlated features will split importance between them. `subtotal` dominates `total_price` not because the others are uninformative, but because `subtotal` alone explains most variance.

### 6.4 All targets train from scratch per run
There is no shared pre-training. Each `--target` invocation fits a fresh model. A future improvement would be to pre-train a shared trunk on a reconstruction objective and fine-tune per target.

### 6.5 Ads data is aggregated
`google_ads_daily` and `meta_ads_daily` are joined at campaign-day granularity and averaged across all line items in that campaign on that day. This means ad signal is shared across orders in the same campaign/day bucket — it is order-correlated, not order-specific.

---

## 7. Quick Reference

### Run evaluation

```bash
# Single target with all metrics + plots
python nn/evaluate.py --target satisfaction_rating --epochs 30 --plot

# All 6 key targets at once
python nn/evaluate.py --all --epochs 30 --plot
```

### Run estimator

```bash
# List all 51 valid targets
python nn/estimator.py --list_targets

# Train and get results
python nn/estimator.py --target total_price --epochs 50 --plot --save_model models/total_price

# Predict from saved model (partial row OK)
python nn/estimator.py \
  --predict '{"price": 149.0, "collection": "SS25 Drop", "acquisition_source": "facebook/paid_social/Prospecting_Mens_UK"}' \
  --load_model models/total_price
```

### Summary table (30-epoch results)

| Target | Task | Metric | Score | vs Naive |
|--------|------|--------|-------|---------|
| `has_refund` | binary | AUC | **1.0000** | >> 0.857 majority |
| `satisfaction_rating` | regression | RMSE | 1.172 | ≈ naive 1.168 |
| `total_price` | regression | RMSE | **6.42** | 94% better than naive 113.3 |
| `product_type` | classification | Accuracy | **0.9999** | +46 pp vs 54% baseline |
| `resolved_by` | classification | Accuracy | **0.9993** | +2 pp vs 98% baseline |
| `resolution_time_minutes` | regression | RMSE | 50.30 | 53% better than naive 106.7 |

### Output files

| File | Generated by |
|------|-------------|
| `eval_{target}.png` | `nn/evaluate.py --plot` — 6-panel evaluation chart |
| `importance_{target}.png` | `nn/estimator.py --plot` — importance bar chart only |
| `models/{target}.pt` | `nn/estimator.py --save_model` — model weights |
| `models/{target}.pkl` | `nn/estimator.py --save_model` — feature metadata |
