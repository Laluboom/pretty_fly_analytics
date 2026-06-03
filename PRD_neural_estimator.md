# PRD: Pretty Fly Universal Neural Estimator

## Context

Pretty Fly's team can't easily answer operational questions like "what will refund rate be if we discount 20%?" or "which customer attributes predict high LTV?" because answering each requires a custom model or manual analysis. This tool provides a single neural network that accepts any variable in the dataset as the prediction target and uses all other available variables as inputs — a universal estimator for any business question that can be framed as "given everything else, what is X?"

---

## Problem Statement

The dataset has ~30 meaningful numeric/categorical variables spread across 20 tables. Deriving estimates across them today requires writing bespoke SQL/pandas queries per question. There is no single interface to ask "what drives conversions?" or "predict satisfaction rating from order and product features."

---

## Goal

Build a Python-based neural network pipeline that:
1. Merges the relevant tables into a flat feature matrix
2. Accepts a user-specified **target variable** (any numeric or categorical column)
3. Trains a feed-forward network to predict that variable from all others
4. Outputs the prediction, feature importances, and a brief natural-language summary

---

## Scope

### In scope
- Single-file or small-module Python implementation
- Support for **numeric regression** targets and **categorical classification** targets
- A curated flat feature table joining the most signal-rich tables
- CLI interface: `python estimator.py --target <column_name>`
- Feature importance via gradient-based saliency or permutation importance
- Brief printed summary of result + top 5 driving features

### Out of scope
- Real-time inference API
- Model persistence / serving
- Hyperparameter tuning UI
- Time-series forecasting (this is tabular, not sequential)

---

## Data Architecture

### Master feature table (assembled at runtime)

Join path:
```
orders (base)
  → line_items        (via order_id)       → variants (via variant_id) → products (via product_id)
  → customers         (via customer_id)
  → refunds           (via order_id, flag: has_refund + refund_reason)
  → support_tickets   (via related_order_id, flag: has_ticket + category)
  → google_ads_daily  (via utm_campaign + date)
  → meta_ads_daily    (via utm_campaign + date)
```

Resulting flat table: ~one row per **line item** (69,956 rows), with ~40 features.

### Feature groups

| Group | Features |
|-------|---------|
| Product | `product_type`, `gender_segment`, `collection`, `price`, `weight_grams` |
| Order | `subtotal`, `total_discounts`, `total_shipping`, `total_tax`, `total_price`, `discount_pct` |
| Customer | `orders_count`, `total_spent`, `acquisition_source`, `default_country`, `gender_segment_affinity` |
| Time | `order_month`, `order_dayofweek`, `order_hour` (from `created_at`) |
| Marketing | `utm_source`, `utm_medium`, `utm_campaign`, `ad_spend_gbp`, `ad_impressions`, `ad_clicks`, `ad_conversions` |
| Refund | `has_refund` (bool), `refund_reason` (categorical) |
| Support | `has_ticket` (bool), `ticket_category`, `resolved_by`, `satisfaction_rating` |
| Inventory | `inventory_quantity` (at time of sale), `landed_cost_per_unit_gbp` |

### Target variable candidates (most useful)

| Variable | Type | Business question |
|----------|------|------------------|
| `has_refund` | binary | What drives returns? |
| `satisfaction_rating` | numeric (1–5) | What makes customers happy? |
| `total_price` | numeric | What predicts order value? |
| `ad_conversions` | numeric | What drives ad performance? |
| `resolved_by` | categorical (bot/human) | Which tickets need humans? |
| `quantity` | numeric | What drives units per order? |

---

## Neural Network Architecture

```
Input layer (N features after encoding)
    ↓
Embedding layers for categoricals (dim=8 each)
    ↓
Concatenation + BatchNorm
    ↓
Dense(256) → ReLU → Dropout(0.3)
    ↓
Dense(128) → ReLU → Dropout(0.2)
    ↓
Dense(64)  → ReLU
    ↓
Output head:
  - Regression target  → Dense(1), MSE loss
  - Binary target      → Dense(1) + Sigmoid, BCE loss
  - Multi-class target → Dense(n_classes) + Softmax, CE loss
```

Framework: **PyTorch** (or PyTorch Lightning for brevity).

---

## Implementation Plan

### Step 1 — `data_builder.py`
- Load and join all tables into the master feature table
- Engineer derived features (discount_pct, order_month, etc.)
- Encode categoricals (label encode or embedding index)
- Normalise numerics (StandardScaler)
- Output: `X` (features), `y` (target), `feature_names`

### Step 2 — `model.py`
- Define `PrettyFlyNet(nn.Module)` with embedding + MLP structure
- Auto-detect output head type from target dtype
- Train/val split (80/20), early stopping on val loss

### Step 3 — `estimator.py` (entry point)
- CLI: `python estimator.py --target satisfaction_rating --epochs 50`
- Calls `data_builder`, trains `PrettyFlyNet`, prints results
- Outputs: val loss/accuracy, top-5 feature importances (permutation), example predictions

### Step 4 — `requirements_nn.txt`
- `torch`, `pandas`, `scikit-learn`, `numpy`

---

## File Structure

```
pretty_fly_data_pack/
├── nn/
│   ├── data_builder.py     ← feature engineering + joins
│   ├── model.py            ← PrettyFlyNet definition + training loop
│   └── estimator.py        ← CLI entry point
├── requirements_nn.txt
└── PRD_neural_estimator.md ← this file
```

---

## Verification

1. `python nn/estimator.py --target has_refund` — should train, print val AUC > 0.6
2. `python nn/estimator.py --target satisfaction_rating` — val MAE printed
3. `python nn/estimator.py --target total_price` — val RMSE printed
4. Feature importance output should name plausible drivers (e.g. `product_type`, `discount_pct` for refund target)
5. `validate.py data/` should still pass all 20 checks (data unchanged)

---

## Success Criteria

- Any of the 6 target candidates above runs end-to-end without error
- Val metric is better than a naive baseline (mean prediction / majority class)
- Feature importance output is legible and business-interpretable
- Total runtime under 5 minutes on CPU for 50 epochs

---

## Architecture Decision: One Universal Model vs Multiple Specialised Models

**Decision: one universal model.**

### Why universal wins for this dataset

- **Cross-table signal is the core value.** Predicting refund probability is more accurate when the model sees `product_type` + `discount_pct` + `acquisition_source` + `ad_channel` together — not just order-level features in isolation. A siloed model misses those interactions entirely.
- **The join is the hard work.** Once `data_builder.py` produces the flat feature table, training one model vs five is trivial. Multiple models would multiply build time and maintenance burden without improving results.
- **70k rows is sufficient to generalise across all targets.** The dataset is large enough that one model can learn shared representations across product, customer, order, and marketing dimensions simultaneously.
- **Shared embeddings learn richer representations.** Categorical embeddings (e.g. for `utm_campaign`, `product_type`, `collection`) trained across all targets capture richer semantics than embeddings trained on a single narrow objective.
- **Single interface, lower cognitive load.** One CLI, one model file, one feature pipeline — easier to demo, easier to explain, easier to iterate on during a hackathon.
- **Transfer of signal across sparse targets.** Targets like `satisfaction_rating` (only ~1,200 non-null rows) benefit from a model whose lower layers have already learned strong feature representations from 70k rows of other signals. The shared backbone acts as implicit pre-training.
- **Feature importance is comparable across targets.** With one model you can directly compare which features drive refunds vs which drive satisfaction vs which drive order value — without reconciling different feature sets across models.
- **Simpler hyperparameter surface.** One architecture to tune, one learning rate, one dropout schedule — not five separate experiments that need to be kept in sync.

### The one real concern — and how it is handled

Some targets are sparse: `satisfaction_rating` and `resolved_by` only have values for the ~1,200 support ticket rows. The fix is not a separate model — `get_X_y()` drops rows where the target is null before training. Each target automatically trains on its relevant non-null subset. Same architecture, different slice.

### When multiple models would be the right call

- If granularities were fundamentally incompatible (e.g. predicting daily ad spend lives at campaign-day level, not line-item level — a genuinely different row shape that cannot be aggregated down without losing meaning).
- If the dataset had millions of rows and training speed per domain became a bottleneck.
- If targets had wildly different feature sets with near-zero overlap, making the shared backbone learn nothing useful.

None of those conditions apply here. The ads data is the closest edge case — it is aggregated from campaign-day granularity down to order level in `data_builder.py`. This is a deliberate trade-off noted in the implementation.

### Reference: data sparsity by target

| Target | Non-null rows | % of total |
|--------|--------------|-----------|
| `total_price` | ~69,956 | 100% |
| `has_refund` | ~69,956 | 100% (engineered flag) |
| `quantity` | ~69,956 | 100% |
| `product_type` | ~69,956 | 100% |
| `satisfaction_rating` | ~1,200 | ~1.7% |
| `resolved_by` | ~1,200 | ~1.7% |
