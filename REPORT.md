# Pretty Fly — Neural Estimator: Evaluation Report & System Guide

**Live evaluation | 30 epochs | CUDA | Pretty Fly dataset (Jun 2024 – May 2026)**

---

## Contents
1. [What This System Does](#1-what-this-system-does)
2. [How It Works — End to End](#2-how-it-works--end-to-end)
3. [Evaluation Results — Core 6 Targets](#3-evaluation-results--core-6-targets)
4. [Feature Importance Findings](#4-feature-importance-findings)
5. [Worked Examples](#5-worked-examples)
6. [Caveats and Known Limitations](#6-caveats-and-known-limitations)
7. [Quick Reference](#7-quick-reference)

---

## 1. What This System Does

PrettyFlyNet is a **universal neural estimator**: given any of the 76 columns in the Pretty Fly dataset as a prediction target, it trains a feed-forward neural network using all other columns as inputs and outputs:

- A validation metric (AUC / accuracy / RMSE depending on target type)
- A ranked list of the top features driving that prediction
- Example predictions vs actuals
- A one-line business interpretation

The same architecture handles binary classification, multi-class classification, and regression — the output head and loss function are auto-selected based on the target column's data type.

---

## 2. How It Works — End to End

### Step 1: Data Pipeline (`nn/data_builder.py`, 626 lines)

21 data files (20 CSV + 1 JSON) are loaded and joined into a single flat feature matrix.

**Join chain (base = `line_items.csv`, 69,956 rows):**

```
line_items (69,956 rows)
  ├── LEFT JOIN orders               on order_id
  ├── LEFT JOIN variants             on variant_id       → size, colour, price, weight
  ├── LEFT JOIN products             on product_id       → type, gender, collection
  ├── LEFT JOIN customers            on customer_id      → LTV, acquisition, country
  ├── LEFT JOIN po_line_items        on variant_id       → landed cost per unit
  │     └── LEFT JOIN purchase_orders on po_id
  │           └── LEFT JOIN suppliers  on supplier_id    → country, lead time, delivery delay
  ├── LEFT JOIN inventory_movements  on variant_id       → stock level, return rate, restock count
  ├── LEFT JOIN discount_codes       on discount_code    → discount type and value
  ├── LEFT JOIN email_events         on customer_id      → opens, clicks, campaigns, recency
  ├── LEFT JOIN addresses            on customer_id      → city, postcode district (PII stripped)
  ├── LEFT JOIN refunds              on order_id         → has_refund flag, reason, amount
  ├── LEFT JOIN support_tickets      on order_id         → category, resolved_by, rating
  ├── LEFT JOIN support_messages.json on ticket_id       → message counts, length, VADER sentiment
  ├── LEFT JOIN google_ads_daily     on utm_campaign+date → spend, impressions, clicks, conversions
  └── LEFT JOIN meta_ads_daily       on utm_campaign+date → same for Meta
```

**Result:** 69,956 rows × 76 columns (75 features + 1 target per run)

**Engineered features added on top of raw columns:**

| Feature | Formula / Source |
|---------|-----------------|
| `discount_pct` | `total_discounts / subtotal`, clipped to [0, 1] |
| `gross_margin_est` | `(price - landed_cost) / price` |
| `order_month` | Month of order (1–12) |
| `order_dayofweek` | Day of week (0=Mon, 6=Sun) |
| `order_hour` | Hour of order (0–23) |
| `is_discounted` | 1 if discount code applied, else 0 |
| `total_ad_spend` | `google_spend + meta_spend` on that campaign-day |
| `total_ad_conversions` | `google_conversions + meta_conversions` |
| `price_components_sum` | `subtotal + shipping + tax - discounts` |
| `damaged_in_transit` | 1 if refund reason indicates transit damage |
| `size_issue` | 1 if refund reason indicates a sizing problem |
| `avg_sentiment` | Mean VADER compound score of customer messages in ticket |
| `min_sentiment` | Lowest (most negative) single customer message score |
| `pct_negative_msgs` | Fraction of customer messages with compound < −0.05 |

**Encoding:**
- 19 categorical columns → `LabelEncoder` (integer codes) → `nn.Embedding(vocab+1, dim=8)`
- 57 numeric columns → `StandardScaler` (zero mean, unit variance)

**Data quality controls:**
- `LEAKAGE_MAP`: 6 entries — drops columns that directly encode the target before training
- `FILTER_MAP`: 10 entries — restricts to meaningful rows per target (e.g. `resolution_time_minutes` trains only on the 1,705 rows that have a ticket)

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

### Step 3: Network Architecture (`nn/model.py`, 229 lines)

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT                                                          │
│  19 categorical columns → nn.Embedding(vocab+1, dim=8) each    │
│  57 numeric columns     → StandardScaled float32               │
└──────────────────┬──────────────────────────────────────────────┘
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

**Training config:**
- Optimiser: Adam (lr=1e-3) + ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-6)
- Train/val split: 80/20 stratified (70/30 for sparse targets < 1,000 rows)
- Early stopping: patience=5, min_delta=1e-5, restores best weights
- Reproducibility: seed=42 everywhere (torch, numpy, cuda)
- Device: CUDA (auto-detected), falls back to CPU

---

### Step 4: Feature Importance (`nn/estimator.py`, 538 lines)

Permutation importance on the held-out validation set:

```
importance(feature_i) = mean over 3 shuffles of:
    loss(model, shuffled feature_i) − loss(model, original)
```

- High score → model relies heavily on this feature; removing it hurts
- Near 0 → feature is not used for this target
- Cat and numeric features shuffled in their respective tensors

---

## 3. Evaluation Results — Core 6 Targets

All results from 30-epoch runs on CUDA. Evaluation charts saved as `eval_{target}.png`.
Full suite of 11 key targets available via `python nn/evaluate.py --all`.

---

### 3.1 `has_refund` — Binary Classification (⚠️ PENDING RE-RUN — see note)

> **Business question:** Which orders are likely to be refunded?

> **⚠️ Not leakage-free.** `size_issue` and `damaged_in_transit` are pure functions of
> `refund_reason` (dropped) but were missing from `LEAKAGE_MAP["has_refund"]`, so every
> number below was trained with those two columns still present. `LEAKAGE_MAP` has now
> been fixed (`nn/data_builder.py`), but re-running `python nn/evaluate.py --target
> has_refund --epochs 30` to get the honest numbers requires ML dependencies
> (`torch`, `scikit-learn`, …) that could not be installed in the environment this fix
> was made in — no network/package-install access. The figures below are the **stale,
> leaky** numbers and must be replaced with a real re-run before being cited again.

| Metric | Value |
|--------|-------|
| AUC | **0.8436** |
| Average Precision | 0.7251 |
| Accuracy | 0.9345 |
| Precision | 0.9936 |
| Recall | 0.5461 |
| F1 | 0.7049 |
| Positive rate (actual) | 14.33% |

**Confusion matrix:**
```
              Pred 0   Pred 1
  Actual 0    11,980        7
  Actual 1       910    1,095
```

**Top features:** `size_issue` (Δ=0.794), `damaged_in_transit` (Δ=0.278), `product_type`, `variant_return_rate`

**Interpretation:** Leakage-safe — `financial_status`, `refund_reason`, and `refund_amount` are dropped before training. The dominant signal is `size_issue` (an engineered flag from Phase 8) followed by `damaged_in_transit`. High precision (99.4%) means almost no false alarms; lower recall (54.6%) means some refunds are missed, which is expected without leakage columns. The 910 false negatives are genuine refunds the model couldn't predict from product/order features alone.

---

### 3.2 `satisfaction_rating` — Regression (Sparse)

> **Business question:** What predicts whether a customer gives a high or low rating?

| Metric | Value | Context |
|--------|-------|---------|
| RMSE | **1.2125** | Rating scale is 1–5 |
| MAE | 0.9994 | Avg error ~1 star |
| R² | −0.0643 | Near-zero predictive power |
| Median Abs Error | 0.9111 | |
| Within ±20% | 44.4% | |
| Naive RMSE (predict mean) | 1.1753 | |
| **Improvement over naive** | **−3.2%** | Not beating predict-mean |

**Top features:** `city`, `ticket_category`, `gender_segment_affinity`, `resolved_by`, `utm_source`

**Interpretation:** Only 539 training rows — too sparse for the model to find real signal. The VADER sentiment features (`avg_sentiment`, `min_sentiment`) do not appear in the top 10 because the synthetic message transcripts lack genuine emotional variance. This is a fundamental dataset limitation, not a code issue. The model learns to predict near the mean (~3.7) for most inputs.

---

### 3.3 `total_price` — Regression

> **Business question:** What determines the value of a line item order?

| Metric | Value | Context |
|--------|-------|---------|
| RMSE | **£4.86** | Orders range £12–£690 |
| MAE | £3.57 | Average error ~£3.57 |
| R² | **0.9982** | Explains 99.8% of variance |
| Median Abs Error | £2.90 | Half predictions within £2.90 |
| Within ±10% | **98.2%** | |
| Within ±20% | 99.7% | |
| Naive RMSE (predict mean) | £113.32 | |
| **Improvement over naive** | **95.7%** | |

**Top features:** `subtotal` (Δ=4,683), `price_components_sum` (Δ=3,669), `total_shipping` (Δ=502), `discount_pct` (Δ=329), `discount_type` (Δ=133)

**Interpretation:** Near-perfect prediction — `total_price` is almost an algebraic identity of `subtotal + shipping + tax - discounts`. The engineered `price_components_sum` feature sits at rank 2, confirming the relationship. More interesting is `discount_type` at rank 7: the type of discount code (percentage vs fixed amount) has independent influence on final price beyond the raw discount values.

---

### 3.4 `product_type` — Multi-class Classification (6 classes) (⚠️ PENDING RE-RUN — see note)

> **Business question:** Can we identify product category from all other signals?

> **⚠️ Not leakage-free.** `weight_grams` has exactly 6 distinct values across all 645
> variants, each mapping to exactly one product type — a literal lookup table.
> `supplier_country`, `gross_margin_est`, `variant_price`, `price`, and `lead_time_days`
> are the same leak in disguise: all are product- or variant-level attributes broadcast
> onto every line item of that product, so the model is joining, not predicting. None of
> these were in `LEAKAGE_MAP["product_type"]`. `LEAKAGE_MAP` has now been fixed
> (`nn/data_builder.py`), but re-running `python nn/evaluate.py --target product_type
> --epochs 30` requires ML dependencies (`torch`, `scikit-learn`, …) not installed in
> this environment. The figures and "top features" below are the **stale, leaky**
> numbers and must not be cited as the product-type mechanism.

| Metric | Value |
|--------|-------|
| Accuracy | **1.0000** |
| Macro F1 | 1.0000 |
| Weighted F1 | 1.0000 |
| Baseline (majority class: Tee) | 53.64% |
| **Improvement over baseline** | **+46.4 pp** |

**Per-class breakdown (validation set — 13,992 rows, zero errors):**
```
             Precision  Recall    F1   Support
Cap            1.000    1.000   1.000    1,203
Hoodie         1.000    1.000   1.000    2,761
Outerwear      1.000    1.000   1.000      431
Sweatpants     1.000    1.000   1.000    1,265
Tee            1.000    1.000   1.000    7,506
Trainer        1.000    1.000   1.000      826
```

**Top features:** `supplier_country` (Δ=1.892), `gross_margin_est` (Δ=1.005), `variant_price`, `price`, `lead_time_days`, `weight_grams`

**Interpretation (⚠️ wrong — see note above):** The credited mechanism (supplier origin + margin) is itself a co-leak, not the real driver. `weight_grams` alone is a perfect lookup; everything else in the top-6 is just another broadcast product/variant attribute riding along. "Can we identify product category from all other signals" is trivially yes when those signals include the product's own weight.

---

### 3.5 `resolved_by` — Multi-class Classification (3 classes) (⚠️ PENDING RE-RUN — see note)

> **Business question:** Will this ticket be resolved by a bot, a human agent, or not at all?

> **⚠️ Not leakage-free, and scored against the wrong baseline.** `resolution_time_minutes`
> is a perfect lookup for `resolved_by` (bot tickets resolve in [5,29] min, human in
> [60,1439] min — zero overlap, verified this run: n=504/700). `msg_count` and
> `customer_msg_count` are the same leak in disguise (transcript shapes {0,3,5}/{0,2,3}
> map 1:1 onto {none,bot,human}). None of these were in `LEAKAGE_MAP["resolved_by"]`
> (that key did not exist at all). Separately, `resolved_by` was missing from
> `FILTER_MAP`, so it is scored against a 97.56% "none" baseline instead of the real
> bot-vs-human split — that's why "+2.4pp" reads as unremarkable instead of alarming.
> Both are now fixed in `nn/data_builder.py`. The claim below that `support_channel`
> "almost perfectly determines who handles them" is also wrong: cross-tabbed this run,
> majority-vote accuracy of `support_channel` → `resolved_by` is 0.5814 — exactly the
> naive baseline (700/1204). `support_channel` only topped the permutation ranking
> because it is `"none"` on 97.56% of rows, so shuffling it destroys *has_ticket*
> detection, not handler prediction. Re-running `python nn/evaluate.py --target
> resolved_by --epochs 30` requires ML dependencies not installed in this environment.
> The figures below are **stale and leaky**.

| Metric | Value |
|--------|-------|
| Accuracy | **1.0000** |
| Macro F1 | 1.0000 |
| Weighted F1 | 1.0000 |
| Baseline (majority: "none" = no ticket) | 97.56% |
| **Improvement over baseline** | **+2.4 pp** |

**Per-class breakdown (13,992 rows, zero errors):**
```
          Precision  Recall    F1   Support
bot          1.000    1.000   1.000      142
human        1.000    1.000   1.000      199
none         1.000    1.000   1.000   13,651
```

**Top features:** `support_channel` (Δ=0.120), `resolution_time_minutes` (Δ=0.111), `ticket_category` (Δ=0.027), `msg_count` (Δ=0.010), `pct_negative_msgs` (Δ=0.0002)

**Interpretation (⚠️ wrong — see note above):** Not a real business insight. `support_channel`'s top rank measures the model's ability to detect whether a ticket exists at all (via the 97.56% "none" rate), not who handles it. The real signal, `resolution_time_minutes`/`msg_count`/`customer_msg_count`, is a one-column lookup and will be dropped from features once this target is re-run.

---

### 3.6 `resolution_time_minutes` — Regression (⚠️ PENDING RE-RUN — see note)

> **Business question:** How long will this support ticket take to resolve?

> **⚠️ Not leakage-free.** There was no `LEAKAGE_MAP["resolution_time_minutes"]` key, so
> `resolved_by` (plus `msg_count`/`customer_msg_count`) survived as a feature. Measured
> this run: predicting just the group mean of `resolved_by` (bot 16.9 min, human 756.5
> min) gives RMSE **304.1** against a naive-mean RMSE of **475.0** — the model's 315
> below is *worse* than that one-column lookup. The entire "33% better than naive"
> claim is that column, and nothing else; §4's framing of this as a "mutually
> predictive" finding is wrong — it's the same leak as §3.5, not an insight.
> `LEAKAGE_MAP` has now been fixed (`nn/data_builder.py`). Re-running `python
> nn/evaluate.py --target resolution_time_minutes --epochs 30` requires ML dependencies
> not installed in this environment. The figures below are **stale and leaky**.

Trained on ticket rows only (1,705 rows — `FILTER_MAP` excludes non-ticket orders).

| Metric | Value | Context |
|--------|-------|---------|
| RMSE | **315 min** | Range: [5, 1,439] min |
| MAE | 245 min | |
| R² | **0.5546** | Explains 55.5% of variance |
| Median Abs Error | 169 min | |
| Within ±10% | 8.5% | Long-tail distribution |
| Naive RMSE (predict mean) | 472.6 min | |
| **Improvement over naive** | **33.3%** | |

**Top features:** `resolved_by` (Δ=136,908), `msg_count` (Δ=4,509), `customer_msg_count` (Δ=2,485), `postcode_district` (Δ=2,277), `avg_sentiment` (Δ=2,235), `ticket_category` (Δ=1,354)

**Interpretation (⚠️ wrong — see note above):** `resolved_by` does not "overwhelmingly dominate" a real regression — it's a one-column lookup for the target's two sub-ranges, and the model's RMSE is worse than that lookup alone. The `avg_sentiment` signal at rank 5 may still be real, but it can't be assessed until this is re-run leakage-free.

---

## 4. Feature Importance Findings

| Target | #1 Feature | #2 Feature | #3 Feature | Business Takeaway |
|--------|-----------|-----------|-----------|-------------------|
| `has_refund` | `size_issue` | `damaged_in_transit` | `product_type` | Fit and transit damage are the primary refund drivers |
| `satisfaction_rating` | `city` | `ticket_category` | `gender_segment_affinity` | No dominant signal — dataset too sparse |
| `total_price` | `subtotal` | `price_components_sum` | `total_shipping` | Algebraic identity — confirms data consistency |
| `product_type` ⚠️ | `supplier_country` | `gross_margin_est` | `variant_price` | STALE — all 3 are product/variant-level leaks (see §3.4); real target is `weight_grams`, now dropped |
| `resolved_by` ⚠️ | `support_channel` | `resolution_time_minutes` | `ticket_category` | STALE — `support_channel`'s rank is a `has_ticket`-detection artifact, not a handler signal; `resolution_time_minutes` is a leak, now dropped (see §3.5) |
| `resolution_time_minutes` ⚠️ | `resolved_by` | `msg_count` | `customer_msg_count` | STALE — all 3 are the same `resolved_by` leak, now dropped (see §3.6) |

**Cross-target patterns:**
- ⚠️ `resolved_by` and `resolution_time_minutes` being "mutually predictive" was a leakage bug, not a finding — `LEAKAGE_MAP` had no entry for either, so each stayed in the other's feature set. Both are fixed in `nn/data_builder.py`; see §3.5/§3.6.
- Phase 8 supplier features (`supplier_country`, `lead_time_days`) appear in product-type and refund importance — ⚠️ for `product_type` this is a leak (see §3.4), not supply-chain signal
- `avg_sentiment` (VADER) shows up in `resolution_time_minutes` top 5 — angrier tickets take longer to resolve, but this can't be assessed until the target is re-run without `resolved_by`/`msg_count`/`customer_msg_count`
- `size_issue` and `damaged_in_transit` dominate `has_refund`, confirming that reason-based engineered flags capture the refund mechanism better than raw financials

---

## 5. Worked Examples

### Example A: Predicting `total_price` on real orders

```
#         Predicted    Actual     Error
1             60.41     55.94    +£4.47
2            129.11    125.00    +£4.11
3             41.01     39.94    +£1.07
4            121.45    112.50    +£8.95
5            182.82    187.99    −£5.17
```
R² = 0.9982 — prices predicted to within ~£3.57 on average across 14,000 validation orders.

---

### Example B: Classifying `product_type`

```
#     Predicted     Actual
1           Tee        Tee   ✓
2     Outerwear  Outerwear   ✓
3           Cap        Cap   ✓
4        Hoodie     Hoodie   ✓
5   Sweatpants  Sweatpants   ✓
```
Zero errors in 13,992 val predictions.

---

### Example C: Predicting from a saved model with subset filtering

```bash
# Train on Hoodies only and save
python nn/estimator.py \
  --target has_refund --epochs 30 \
  --subset "product_type=Hoodie" \
  --save_model models/has_refund_hoodies

# Predict from saved model — typo warns, partial rows fill to defaults
python nn/estimator.py \
  --predict '{"pric": 85.0, "collection": "Core"}' \
  --load_model models/has_refund_hoodies
```

```
Warning: unknown input keys (ignored): ['pric']
Predicting: has_refund
Input features used: 19 categorical + 56 numeric
Prediction : 0  (probability=0.1234)
```

---

### Example D: Full evaluation with plots

```bash
python nn/evaluate.py --target total_price --epochs 30 --plot
```

Produces:
- Full metrics table (RMSE, MAE, R², within-±10%, residual distribution)
- `eval_total_price.png` — 6-panel chart: feature importance, predicted vs actual scatter, residual histogram, residual vs actual, absolute error distribution

---

## 6. Caveats and Known Limitations

### 6.1 `satisfaction_rating` requires richer data
539 rows with low-variance synthetic messages is insufficient. VADER sentiment, message counts, and escalation flags are all in the model but the synthetic transcripts don't contain genuine emotional variance. R² ≈ −0.06 (at naive baseline). Real-world message data with genuine sentiment variation would unlock this target.

### 6.2 Permutation importance is post-hoc
Importance scores measure influence on validation loss, not causal relationships. Correlated features split importance between them. `subtotal` dominates `total_price` not because other components are uninformative, but because it alone captures most variance.

### 6.3 All targets train from scratch per run
Each `--target` invocation fits a fresh model. There is no shared pre-training or transfer. A future improvement would be to pre-train a shared trunk on a reconstruction objective and fine-tune per target.

### 6.4 Ads data is aggregated at campaign-day level
`google_ads_daily` and `meta_ads_daily` are joined on `utm_campaign + date`. Multiple orders on the same campaign-day share identical ad features — it is order-correlated, not order-specific. Per-click attribution data would be needed to improve this.

### 6.5 `--subset` is applied before `FILTER_MAP`
When combining `--subset` and a target covered by `FILTER_MAP` (e.g. `resolution_time_minutes`), the subset filter runs first, then `FILTER_MAP` further restricts rows. Row counts in the log reflect the post-FILTER_MAP state.

---

## 7. Quick Reference

### Run estimator

```bash
# List all 76 valid targets
python nn/estimator.py --list_targets

# Train and get results
python nn/estimator.py --target total_price --epochs 50 --plot --save_model models/total_price

# Train on a data slice
python nn/estimator.py --target has_refund --epochs 30 --subset "product_type=Hoodie"

# Predict from saved model (partial row OK — typos warned)
python nn/estimator.py \
  --predict '{"price": 149.0, "collection": "Core", "product_type": "Hoodie"}' \
  --load_model models/total_price

# Predict + LLM business recommendation (requires OPENROUTER_API_KEY)
python nn/estimator.py \
  --predict '{"price": 149.0, "product_type": "Hoodie"}' \
  --load_model models/has_refund \
  --recommend

# Train on a data slice
python nn/estimator.py --target has_refund --epochs 30 --subset "product_type=Hoodie"
```

### Run evaluation

```bash
# Single target — full metrics + plots
python nn/evaluate.py --target has_refund --epochs 30 --plot

# All 11 key targets at once
python nn/evaluate.py --all --epochs 30 --plot
```

### Validate data integrity

```bash
python validate.py data/
# Expected: 20 passed, 0 failed, 0 skipped
```

### Summary table (30-epoch live results, CUDA)

| Target | Task | Metric | Score | vs Naive |
|--------|------|--------|-------|---------|
| `has_refund` | binary | AUC | **0.844** ⚠️ | STALE — leaked via `size_issue`/`damaged_in_transit`, needs re-run (see §3.1) |
| `satisfaction_rating` | regression | RMSE | 1.213 | ≈ naive 1.175 — data limitation |
| `total_price` | regression | RMSE | **£4.86** | 96% better than naive £113 |
| `product_type` | classification | Accuracy | **1.000** ⚠️ | STALE — leaked via `weight_grams`/`supplier_country`/etc, needs re-run (see §3.4) |
| `resolved_by` | classification | Accuracy | **1.000** ⚠️ | STALE — leaked via `resolution_time_minutes`/`msg_count`/etc, needs re-run (see §3.5) |
| `resolution_time_minutes` | regression | RMSE | 315 min ⚠️ | STALE — leaked via `resolved_by`/`msg_count`/etc, needs re-run (see §3.6) |
| `damaged_in_transit` | binary | AUC | **0.829** | refund rows only; `variant_return_rate` top signal |
| `size_issue` | binary | AUC | ~0.75 | refund rows only (10,025) |
| `delivery_delay_days` | regression | RMSE | — | supplier chain; run `--target delivery_delay_days` |
| `variant_return_rate` | regression | RMSE | — | inventory health signal |
| `city` | classification | Accuracy | — | `postcode_district` is a one-column lookup for this target, now dropped via `LEAKAGE_MAP` — not yet run |

### Output files

| File | Generated by |
|------|-------------|
| `eval_{target}.png` | `nn/evaluate.py --plot` — 6-panel evaluation chart |
| `importance_{target}.png` | `nn/estimator.py --plot` — importance bar chart only |
| `models/{target}.pt` | `nn/estimator.py --save_model` — model weights |
| `models/{target}.pkl` | `nn/estimator.py --save_model` — feature metadata |
