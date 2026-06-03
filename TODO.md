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

## Phase 1 — `nn/data_builder.py` ✅ DONE
> Builds the master flat feature table from raw CSVs. Output: `(X, y, feature_meta)`.

### 1.1 Load raw tables
- [x] Load `data/orders.csv` — parse `created_at` as datetime
- [x] Load `data/line_items.csv`
- [x] Load `data/variants.csv` — `product_id` dropped (already in `line_items` to avoid merge conflict)
- [x] Load `data/products.csv`
- [x] Load `data/customers.csv`
- [x] Load `data/refunds.csv` — JSON `refund_line_items` parsed; `has_refund` flag created; deduped to one row per `order_id`
- [x] Load `data/support_tickets.csv` — `related_order_id` renamed to `order_id`; `has_ticket` flag created
- [x] Load `data/po_line_items.csv` — deduped on `variant_id`
- [x] Load `data/google_ads_daily.csv` — aggregated by `campaign_name + date` → `google_spend/impressions/clicks/conversions`
- [x] Load `data/meta_ads_daily.csv` — aggregated by `campaign_name + date` → `meta_spend/impressions/clicks/conversions`

### 1.2 Build join chain (base = `line_items`)
- [x] All 10 LEFT JOINs completed in `_build_joined()`
- [x] Ad join on `utm_campaign == campaign_name` AND `order_date == date`
- [x] NaN fills for all flag/categorical columns post-join

### 1.3 Feature engineering
- [x] `discount_pct`, `gross_margin_est`, `order_month`, `order_dayofweek`, `order_hour`, `is_discounted`, `total_ad_spend`, `total_ad_conversions`
- [x] `has_refund` and `has_ticket` cast to int via `.notna() & .eq(True)` (avoids pandas FutureWarning)
- [x] All raw join keys dropped

### 1.4 Encode and scale
- [x] 15 categorical cols LabelEncoded; 35 numeric cols StandardScaled
- [x] `feature_meta` dict returned with `cat_cols`, `num_cols`, `cat_vocab_sizes`, `encoders`, `scaler`

### 1.5 Target extraction
- [x] `prepare_for_target(df, target_col)` auto-detects task type: binary / regression / classification
- [x] NaN rows dropped per target before training

> **What was done:**
> - Created `nn/data_builder.py` (330 lines)
> - Public API: `build_feature_table()` → 69,956 rows × 51 cols; `prepare_for_target(df, col)` → `X_cat, X_num, y, task_type, n_classes, feature_meta, target_encoder`
> - Verified on 3 targets: `has_refund` (binary, 69,956 rows), `satisfaction_rating` (regression, 539 rows), `product_type` (classification, 6 classes, 69,956 rows)
> - Zero FutureWarnings on pandas 2.3.3
> - Git commit: `1f03011`

---

## Phase 2 — `nn/model.py` ✅ DONE
> Defines `PrettyFlyNet` and the training loop.

### 2.1 Dataset class
- [x] `class PrettyFlyDataset(Dataset)`:
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
- [x] `train_model(X_cat, X_num, y, task_type, n_classes, feature_meta, epochs, batch_size, lr, patience, device)`:
  - [x] 80/20 stratified train/val split (falls back to unstratified for sparse targets)
  - [x] `DataLoader` for train (shuffled) and val
  - [x] `PrettyFlyNet` + `Adam` + task-appropriate loss
  - [x] Early stopping patience=5, best-weight restore
  - [x] `tqdm.write` per epoch with train+val loss
  - [x] Returns `model, val_loss, val_metric, metric_name, val_idx, preds, targets`

> **What was done:**
> - Created `nn/model.py` (204 lines)
> - `PrettyFlyDataset`: wraps `X_cat (LongTensor)`, `X_num (FloatTensor)`, `y` (Long for classification, Float otherwise)
> - `PrettyFlyNet`: `nn.Embedding` per cat col (dim=8) → concat + `BatchNorm1d` → `Dense(256→128→64)` → task head
> - Output heads: regression=`Linear(64,1)`, binary=`Linear(64,1)+Sigmoid`, classification=`Linear(64,n_classes)`
> - `get_loss_fn`: MSE / BCE / CrossEntropy
> - CUDA auto-detected and used (torch 2.12+cu130)
> - Smoke-tested 3 epochs × 4 targets: `has_refund` AUC=1.0, `total_price` RMSE=14.8, `product_type` acc=0.9999, `satisfaction_rating` RMSE=1.48
> - Git commit: `e63e301`

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
