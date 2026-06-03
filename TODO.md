# TODO: Pretty Fly Universal Neural Estimator
**Target: 4-hour build | Entry point: `python nn/estimator.py --target <col>`**

---

## Phase 0 — Setup ✅ DONE
- [x] Create `nn/` directory inside `pretty_fly_data_pack/`
- [x] Create `requirements_nn.txt` with: `torch`, `pandas`, `numpy`, `scikit-learn`, `tqdm`
- [x] Run `pip install -r requirements_nn.txt` and confirm imports work
- [x] Confirm `data/` path is accessible from `nn/` (use `../data/` relative paths or pass as arg)

> **What was done:**
> - Created `pretty_fly_data_pack/nn/` directory (empty, ready for Phase 1 files)
> - Created `requirements_nn.txt` at project root with 5 dependencies: `torch`, `pandas`, `numpy`, `scikit-learn`, `tqdm`
> - Installed all dependencies via `pip3 install --break-system-packages`
> - Verified with `python3 -c "import torch; import pandas; ..."` — all imports OK
> - Git repo was already initialised (`origin/main`); all new files (`PRD_neural_estimator.md`, `TODO.md`, `requirements_nn.txt`) are tracked

---

## Phase 1 — `nn/data_builder.py`
> Builds the master flat feature table from raw CSVs. Output: `(X, y, feature_meta)`.

### 1.1 Load raw tables
- [ ] Load `data/orders.csv` — parse `created_at` as datetime
  - Keep: `order_id`, `customer_id`, `created_at`, `subtotal`, `total_discounts`, `total_shipping`, `total_tax`, `total_price`, `utm_source`, `utm_medium`, `utm_campaign`, `discount_code`, `financial_status`
- [ ] Load `data/line_items.csv`
  - Keep: `line_item_id`, `order_id`, `variant_id`, `product_id`, `quantity`, `price`, `total_discount`
- [ ] Load `data/variants.csv`
  - Keep: `variant_id`, `product_id`, `price` (as `variant_price`), `weight_grams`, `inventory_quantity`, `option1_value` (Size), `option2_value` (Colour)
- [ ] Load `data/products.csv`
  - Keep: `product_id`, `product_type`, `gender_segment`, `collection`
- [ ] Load `data/customers.csv`
  - Keep: `customer_id`, `orders_count`, `total_spent`, `acquisition_source`, `default_country`, `gender_segment_affinity`, `accepts_marketing`
- [ ] Load `data/refunds.csv`
  - Keep: `order_id`, `amount` (as `refund_amount`), `reason` (as `refund_reason`)
  - Parse `refund_line_items` JSON column → extract list of variant_ids
  - Create `has_refund = True` flag per order
- [ ] Load `data/support_tickets.csv`
  - Keep: `related_order_id` (rename → `order_id`), `category` (as `ticket_category`), `resolved_by`, `resolution_time_minutes`, `satisfaction_rating`, `channel` (as `support_channel`)
  - Create `has_ticket = True` flag per order
- [ ] Load `data/po_line_items.csv`
  - Keep: `variant_id`, `landed_cost_per_unit_gbp`
- [ ] Load `data/google_ads_daily.csv`
  - Aggregate by `campaign_name` + `date`: sum `spend_gbp`, `impressions`, `clicks`, `conversions`, `conversion_value_gbp`
  - Rename to `google_spend`, `google_impressions`, `google_clicks`, `google_conversions`
- [ ] Load `data/meta_ads_daily.csv`
  - Aggregate by `campaign_name` + `date`: sum same fields
  - Rename to `meta_spend`, `meta_impressions`, `meta_clicks`, `meta_conversions`

### 1.2 Build join chain (base = `line_items`)
- [ ] `line_items` LEFT JOIN `orders` on `order_id`
- [ ] LEFT JOIN `variants` on `variant_id` → adds `weight_grams`, `inventory_quantity`, `option1_value`, `option2_value`
- [ ] LEFT JOIN `products` on `product_id` → adds `product_type`, `gender_segment`, `collection`
- [ ] LEFT JOIN `customers` on `customer_id` → adds `orders_count`, `total_spent`, `acquisition_source`, `default_country`, `gender_segment_affinity`, `accepts_marketing`
- [ ] LEFT JOIN `po_line_items` on `variant_id` → adds `landed_cost_per_unit_gbp`
- [ ] LEFT JOIN `refunds` on `order_id` → adds `has_refund`, `refund_amount`, `refund_reason` (fill NaN → `has_refund=False`, `refund_reason='none'`)
- [ ] LEFT JOIN `support_tickets` on `order_id` → adds `has_ticket`, `ticket_category`, `resolved_by`, `resolution_time_minutes`, `satisfaction_rating`, `support_channel` (fill NaN → `has_ticket=False`, etc.)
- [ ] Build `order_date` column (date only from `created_at`) for ad join
- [ ] LEFT JOIN `google_ads_daily` on `utm_campaign == campaign_name` AND `order_date == date` → adds `google_*` columns
- [ ] LEFT JOIN `meta_ads_daily` on `utm_campaign == campaign_name` AND `order_date == date` → adds `meta_*` columns

### 1.3 Feature engineering
- [ ] `discount_pct = total_discounts / subtotal` (clip to [0, 1], fill NaN → 0)
- [ ] `gross_margin_est = (price - landed_cost_per_unit_gbp) / price` (fill NaN → median)
- [ ] `order_month` = `created_at.dt.month` (int 1–12)
- [ ] `order_dayofweek` = `created_at.dt.dayofweek` (int 0–6)
- [ ] `order_hour` = `created_at.dt.hour` (int 0–23)
- [ ] `is_discounted` = 1 if `discount_code` is not null else 0
- [ ] `total_ad_spend` = `google_spend + meta_spend` (fill NaN → 0)
- [ ] `total_ad_conversions` = `google_conversions + meta_conversions` (fill NaN → 0)
- [ ] Drop raw join keys and columns not useful as features: `order_id`, `line_item_id`, `variant_id`, `product_id`, `customer_id`, `created_at`, `order_date`, `discount_code`, `utm_campaign`

### 1.4 Encode and scale
- [ ] Define `CATEGORICAL_COLS` list:
  `['utm_source', 'utm_medium', 'product_type', 'gender_segment', 'collection', 'option1_value', 'option2_value', 'acquisition_source', 'default_country', 'gender_segment_affinity', 'refund_reason', 'ticket_category', 'resolved_by', 'support_channel', 'financial_status']`
- [ ] Fill NaN in all categoricals with `'unknown'`
- [ ] `LabelEncoder` each categorical col → integer codes; save `encoder_map: dict[col → LabelEncoder]` for later use
- [ ] Define `NUMERIC_COLS` = all remaining non-target numeric columns
- [ ] `StandardScaler` on numeric cols; save scaler
- [ ] Return `feature_meta` dict: `{ 'cat_cols': [...], 'num_cols': [...], 'cat_vocab_sizes': {col: n_unique}, 'encoders': encoder_map, 'scaler': scaler }`

### 1.5 Target extraction
- [ ] Function `get_X_y(df, target_col)`:
  - Drop `target_col` from features
  - If target is categorical: label-encode it, return `task_type='classification'`, `n_classes=k`
  - If target is boolean/binary int (2 unique vals): return `task_type='binary'`
  - If target is numeric: return `task_type='regression'`
  - Drop rows where `target_col` is NaN
  - Return `X (np.ndarray)`, `y (np.ndarray)`, `task_type`, `n_classes`

---

## Phase 2 — `nn/model.py`
> Defines `PrettyFlyNet` and the training loop.

### 2.1 Dataset class
- [ ] `class PrettyFlyDataset(Dataset)`:
  - `__init__(self, X_cat, X_num, y)` — store as tensors
  - `__len__` and `__getitem__` returning `(X_cat[i], X_num[i], y[i])`
  - `X_cat`: `LongTensor` (shape: `[n, n_cat_cols]`)
  - `X_num`: `FloatTensor` (shape: `[n, n_num_cols]`)
  - `y`: `FloatTensor` for regression/binary, `LongTensor` for multi-class

### 2.2 Network definition
- [ ] `class PrettyFlyNet(nn.Module)`:
  - `__init__(self, cat_vocab_sizes, n_num_features, task_type, n_classes=1, emb_dim=8)`:
    - `nn.ModuleList` of `nn.Embedding(vocab_size + 1, emb_dim)` for each categorical col
    - `emb_output_dim = len(cat_vocab_sizes) * emb_dim`
    - `input_dim = emb_output_dim + n_num_features`
    - MLP: `BatchNorm1d(input_dim)` → `Linear(input_dim, 256)` → `ReLU` → `Dropout(0.3)` → `Linear(256, 128)` → `ReLU` → `Dropout(0.2)` → `Linear(128, 64)` → `ReLU`
    - Output head:
      - `regression`: `Linear(64, 1)`
      - `binary`: `Linear(64, 1)` + `Sigmoid`
      - `classification`: `Linear(64, n_classes)`
  - `forward(self, x_cat, x_num)`:
    - Embed each cat col, concatenate all embeddings
    - Concatenate with `x_num`
    - Pass through MLP + output head

### 2.3 Loss function selector
- [ ] `get_loss_fn(task_type)`:
  - `regression` → `nn.MSELoss()`
  - `binary` → `nn.BCELoss()`
  - `classification` → `nn.CrossEntropyLoss()`

### 2.4 Training loop
- [ ] `train_model(X, y, feature_meta, task_type, n_classes, epochs=50, batch_size=512, lr=1e-3)`:
  - 80/20 train/val split (stratify if classification/binary)
  - Split `X` into `X_cat` (int cols) and `X_num` (float cols) using `feature_meta`
  - Create `DataLoader` for train and val (shuffle train, no shuffle val)
  - Instantiate `PrettyFlyNet`, `Adam` optimiser, loss fn
  - Early stopping: patience=5 on val loss; restore best weights
  - `tqdm` progress bar per epoch showing train loss + val loss
  - Return `model`, `val_loss`, `val_metric` (AUC for binary, accuracy for classification, RMSE for regression)

---

## Phase 3 — `nn/estimator.py`
> CLI entry point. Orchestrates everything and prints results.

### 3.1 CLI argument parsing
- [ ] `argparse` with args:
  - `--target` (required): column name to predict
  - `--epochs` (default: 50)
  - `--batch_size` (default: 512)
  - `--data_dir` (default: `../data`)
  - `--list_targets` (flag): print all valid target columns and exit

### 3.2 Orchestration
- [ ] Call `data_builder.build_feature_table(data_dir)` → `df`, `feature_meta`
- [ ] Validate `--target` exists in `df.columns`; if not, print valid options and exit
- [ ] Call `data_builder.get_X_y(df, target_col)` → `X`, `y`, `task_type`, `n_classes`
- [ ] Print summary: `f"Target: {target_col} | Task: {task_type} | Rows: {len(y)} | Features: {X.shape[1]}"`
- [ ] Call `model.train_model(...)` → `model`, `val_loss`, `val_metric`

### 3.3 Feature importance (permutation)
- [ ] After training, run permutation importance:
  - For each feature column `i`: shuffle column `i` in val set, record change in val loss
  - Rank by largest loss increase = most important feature
  - Print top 10 features as a ranked list with delta-loss score
- [ ] Subtask: handle cat vs num index offset correctly when shuffling `X_cat` vs `X_num`

### 3.4 Output printing
- [ ] Print section: `=== RESULTS ===`
- [ ] Print val metric: AUC (binary), accuracy (classification), RMSE (regression)
- [ ] Print top-5 most important features with scores
- [ ] Print 5 example predictions vs actuals from the val set (random sample)
- [ ] Print 1-line business interpretation, e.g.:
  - `has_refund` → `"Top return drivers: product_type, discount_pct, collection"`
  - `satisfaction_rating` → `"Satisfaction most influenced by: resolved_by, ticket_category, total_price"`

---

## Phase 4 — Validation & Testing

- [ ] Run `python nn/estimator.py --list_targets` — should print all ~40 column names
- [ ] Run `python nn/estimator.py --target has_refund --epochs 30`
  - Expect: binary task, AUC > 0.60, top feature includes `product_type` or `discount_pct`
- [ ] Run `python nn/estimator.py --target satisfaction_rating --epochs 30`
  - Expect: regression task, RMSE < 1.5 (scale is 1–5), runs on ~1,200 non-null rows
- [ ] Run `python nn/estimator.py --target total_price --epochs 30`
  - Expect: regression task, RMSE reasonable vs mean price
- [ ] Run `python nn/estimator.py --target product_type --epochs 20`
  - Expect: multi-class task (6 classes), accuracy > 1/6 baseline (~17%)
- [ ] Run `python validate.py data/` — all 20 checks must still pass (data untouched)
- [ ] Confirm total runtime < 5 min per run on CPU

---

## Phase 5 — Polish (if time allows)

- [ ] Add `--save_model` flag to `estimator.py` → saves `model.pt` + `feature_meta.pkl`
- [ ] Add `--predict` flag: load saved model, accept a JSON row of features, return prediction
- [ ] Add a `--plot` flag: save a bar chart of feature importances as `importance_{target}.png`
- [ ] Write `README_nn.md` with 3 example CLI invocations and sample output

---

## Quick Reference: Key Column Names per File

| File | Target-worthy columns |
|------|-----------------------|
| `orders.csv` | `total_price`, `total_discounts`, `financial_status` |
| `line_items.csv` | `quantity`, `price`, `total_discount` |
| `refunds.csv` | `has_refund` (engineered), `refund_reason` |
| `support_tickets.csv` | `satisfaction_rating`, `resolved_by`, `resolution_time_minutes` |
| `google_ads_daily.csv` | `conversions`, `conversion_value_gbp` |
| `customers.csv` | `orders_count`, `total_spent` |

---

## Estimated Time Breakdown

| Phase | Est. Time |
|-------|-----------|
| Phase 0 — Setup | 10 min |
| Phase 1 — data_builder.py | 75 min |
| Phase 2 — model.py | 60 min |
| Phase 3 — estimator.py | 45 min |
| Phase 4 — Validation | 30 min |
| Phase 5 — Polish | remaining |
| **Total** | **~3h 40min** |
