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

## Phase 3 — `nn/estimator.py` ✅ DONE
> CLI entry point. Orchestrates everything and prints results.

### 3.1 CLI argument parsing
- [x] `--target`, `--epochs` (50), `--batch_size` (512), `--data_dir`, `--list_targets`

### 3.2 Orchestration
- [x] `build_feature_table()` → `prepare_for_target()` → `train_model()` pipeline
- [x] Invalid target caught with helpful error + exit code 1
- [x] `--list_targets` prints all 51 columns with % non-null and row count

### 3.3 Feature importance (permutation)
- [x] Shuffles each of 50 features n_repeats=3 times on val set
- [x] Cat/num index offset handled correctly (`i < n_cat` → `X_cat`, else `X_num[:, i-n_cat]`)
- [x] Returns ranked list of `(feature_name, avg_delta_loss)`

### 3.4 Output printing
- [x] `=== RESULTS ===` block with target, task, metric
- [x] Top-10 importances with delta-loss score + ASCII bar chart
- [x] 5 example predictions vs actuals — labels decoded for classification targets
- [x] 1-line business insight via `_business_line()` lookup

> **What was done:**
> - Created `nn/estimator.py` (200 lines)
> - Full end-to-end verified on 3 targets:
>   - `has_refund` (binary): AUC=1.0, top features: `financial_status`, `refund_reason`
>   - `product_type` (classification): acc=0.9999, top features: `gross_margin_est`, `weight_grams`, decoded labels (Tee/Cap/Hoodie) in example predictions
>   - `satisfaction_rating` (regression, sparse 539 rows): RMSE=1.09, top features: `financial_status`, `refund_reason`, `utm_medium`
> - Note: `has_refund` top features (`financial_status=partially_refunded`, `refund_reason`) are technically leakage — they directly encode the target. Expected for this dataset; meaningful for other targets.
> - Git commit: `a32b55a`

---

## Phase 4 — Validation & Testing ✅ DONE

- [x] `--list_targets` — prints all 51 columns with % non-null and row count
- [x] `has_refund --epochs 30` — binary, AUC=1.0 (>> 0.60 threshold), top: `financial_status`, `refund_reason`
- [x] `satisfaction_rating --epochs 30` — regression, RMSE=1.17 (< 1.5 threshold, scale 1–5, early stop ep.11), top: `acquisition_source`, `support_channel`, `refund_reason`
- [x] `total_price --epochs 30` — regression, RMSE=6.59 vs naive baseline 111.92 (94% improvement)
- [x] `product_type --epochs 20` — classification, acc=0.9999 (>> 17% majority baseline of 53.6%), early stop ep.15, decoded labels (Tee/Cap/Hoodie/Outerwear)
- [x] `validate.py data/` — 20 passed, 0 failed, 0 skipped (data untouched)
- [x] Runtime: 7.9s for 50 epochs on `total_price` (CUDA) — well within 5-min limit

> **What was done:**
> - Ran all 5 spec'd verification commands end-to-end with 30-epoch budgets
> - Confirmed all metrics beat naive baselines: AUC vs majority-class, RMSE vs predict-mean
> - Early stopping fires correctly on sparse targets (satisfaction_rating: ep.11, product_type: ep.15)
> - All 20 data validation rules pass — no data mutations from any pipeline run
> - Full 50-epoch run on largest target (69,956 rows) completes in 7.9s on CUDA
> - Git commit: `a32b55a` (no new code changes in Phase 4 — validation only)

---

## Phase 5 — Polish ✅ DONE

- [x] `--save_model <prefix>` → saves `{prefix}.pt` (state dict) + `{prefix}.pkl` (feature_meta, task_type, n_classes, encoders, scaler, target_encoder)
- [x] `--predict <json_or_path> --load_model <prefix>` → loads saved model, fills missing fields with `0`/`"unknown"` defaults, decodes classification labels, prints prediction + confidence
- [x] `--plot` → saves `importance_{target}.png` via matplotlib (1500×900px, 150 dpi, horizontal bar chart with delta-loss labels)
- [x] `README_nn.md` — 4 CLI examples with sample output, all flags table, architecture diagram

> **What was done:**
> - Updated `nn/estimator.py` (+130 lines): added `save_model()`, `load_and_predict()`, `plot_importance()` functions; wired `--save_model`, `--predict`/`--load_model`, `--plot` flags into `main()`
> - Added `matplotlib` to `requirements_nn.txt`
> - Created `README_nn.md` (130 lines) at project root
> - Verified: `--save_model` writes `models/product_type.pt` (334KB) + `.pkl` (5KB), all metadata intact
> - Verified: `--predict` works with JSON string AND JSON file path; missing features default safely; no UserWarnings
> - Verified: `--plot` writes valid 1500×900 PNG (96KB)
> - Verified: `--predict` without `--load_model` exits 1 with clear error
> - Git commit: `de2bf7d`

---

## Phase 6 — Evaluation & Report ✅ DONE

- [x] `nn/evaluate.py` — comprehensive metrics per task type: binary (AUC, AP, precision, recall, F1, confusion matrix, ROC curve, score distribution, PR curve), regression (RMSE, MAE, R², median abs error, within ±10%/±20%, residual plots), classification (accuracy, macro/weighted F1, per-class breakdown, confusion matrix, per-class accuracy/F1 bar charts)
- [x] `--all` flag runs all 6 key targets in sequence with summary table
- [x] `--plot` saves 6-panel `eval_{target}.png` for each run
- [x] `REPORT.md` — full system guide + live evaluation results for all 6 targets

> **What was done:**
> - Created `nn/evaluate.py` (450 lines): `eval_binary()`, `eval_regression()`, `eval_classification()` metric suites; `print_*_report()` printers with ASCII bar charts; `make_plots()` generating 6-panel matplotlib figures; `--all` batch mode with summary table
> - Fixed histogram edge case: perfect binary model outputs exactly 0/1, so fixed `np.linspace(0,1,42)` bin edges instead of auto-bins
> - Created `REPORT.md` (250 lines): architecture diagram, join chain, feature engineering table, full metrics for all 6 targets, feature importance findings, worked examples, caveats, quick reference
> - Generated 6 evaluation charts: `eval_has_refund.png`, `eval_satisfaction_rating.png`, `eval_total_price.png`, `eval_product_type.png`, `eval_resolved_by.png`, `eval_resolution_time_minutes.png`
> - Key findings: `total_price` R²=0.997, `product_type` acc=0.9999, `resolved_by` acc=0.9993, `resolution_time_minutes` 53% better than naive, `satisfaction_rating` needs richer features (only 539 rows)
> - Git commit: see below

---

Add .gitignore and add appropriate files there ✅ DONE

---

## Phase 7 — `--subset` Flag for Per-Product Filtering

> Lets you train and predict on a slice of data — e.g. Hoodies only, Sweatpants only — without touching any other code.

- [x] Add `--subset` argument to `parse_args()` in `nn/estimator.py` — accepts `"col=value"` string (e.g. `"product_type=Hoodie"`)
- [x] Parse the string into `(col, val)` in `main()`, apply as a row filter on the feature table before `prepare_for_target()` is called
- [x] Print a log line: `Subset : product_type==Hoodie → 13,803 rows (from 69,956)`
- [x] Validate that the column exists and the value is present — exit with helpful error + available values if not
- [x] Update `--list_targets` to note that `--subset` can be combined with any target

**Example usage:**
```bash
python nn/estimator.py --target damaged_in_transit --epochs 30 --subset "product_type=Hoodie"
python nn/estimator.py --target has_refund --epochs 30 --subset "product_type=Sweatpants"
```

**Files to change:** `nn/estimator.py` — `parse_args()` and `main()` only. No changes to `data_builder.py` or `model.py`.

**Effort:** ~20 min

---

## Phase 8 — Full Data Integration (all 21 files)

> Load all previously unused data files and join them into the master feature matrix. Adds ~15 new features covering supplier chain, inventory health, email engagement, customer geography, discount details, and support message sentiment. Primary impact: fixes `satisfaction_rating`, unlocks supplier/inventory predictions.

**Files currently unused:** `discount_codes.csv`, `purchase_orders.csv`, `suppliers.csv`, `inventory_movements.csv`, `email_events.csv`, `addresses.csv`, `support_messages.json`

**Files skipped (no useful per-order join):** `bank_transactions.csv` (no order key), `email_campaigns.csv` (aggregate only — 6 rows), `product_collections.csv` + `collections.csv` (collection already captured via `products.csv`)

---

### 8.1 Discount codes — join on `orders.discount_code`

- [ ] Add `_load_discount_codes(data_dir)` — load 8-row lookup table
- [ ] Join to orders on `orders.discount_code == discount_codes.code` (LEFT JOIN — 85% of orders have no code)
- [ ] New features:
  - `discount_type` — "percentage" / "fixed_amount" / "none" → **categorical**
  - `discount_value` — numeric value of the discount (0 if no code) → **numeric**
- [ ] Fill: `discount_type = "none"`, `discount_value = 0` for orders without a code
- [ ] Drop: `code`, `usage_count`, `starts_at`, `ends_at` (not per-order signals)
- [ ] Add `"discount_type"` to `CATEGORICAL_COLS`

**Leakage check:** Neither column encodes any current target. Clean.

---

### 8.2 Suppliers + purchase orders — join chain via `po_line_items`

- [ ] Add `_load_supplier_features(data_dir)`:
  - Load `purchase_orders.csv` (21 rows) + `suppliers.csv` (5 rows)
  - Join purchase_orders → suppliers on `supplier_id`
  - Parse `expected_delivery` and `actual_delivery` as dates
  - Compute `delivery_delay_days = (actual_delivery - expected_delivery).dt.days` (negative = early, positive = late)
  - Join to `po_line_items` on `po_id` → gives one supplier row per variant (po_line_items already deduped per variant_id in existing pipeline)
- [ ] New features per variant:
  - `supplier_country` — Portugal / Italy / Turkey / etc. → **categorical**
  - `lead_time_days` — 45–90 days (supplier speed proxy) → **numeric**
  - `delivery_delay_days` — actual vs promised delivery → **numeric**
- [ ] Fill: `supplier_country = "unknown"`, `lead_time_days = median`, `delivery_delay_days = 0`
- [ ] Drop: `po_id`, `supplier_id`, `supplier_name`, all date columns, `status` (all "received"), cost columns (already have landed_cost_per_unit_gbp)
- [ ] Add `"supplier_country"` to `CATEGORICAL_COLS`

**Leakage check:** Lead time and delivery delay don't encode any target. Clean.

---

### 8.3 Inventory movements — aggregate per variant

- [ ] Add `_load_inventory_features(data_dir)`:
  - Load `inventory_movements.csv` (76,444 rows — 3 movement types: `sale`, `return`, `po_receipt`)
  - Aggregate per `variant_id`:
    - `variant_latest_stock` = `running_balance` from the most recent movement (any type)
    - `variant_return_rate` = return movements / sale movements (0 if no sales)
    - `variant_restock_count` = count of `po_receipt` movements (how many times restocked)
  - Join to feature matrix on `variant_id`
- [ ] Fill: 0 for variants with no movements (should not occur — all 645 variants have movements)
- [ ] Drop: `movement_id`, `date`, `quantity_delta`, `reference_id`, `type` (used only for aggregation)
- [ ] New features → all **numeric** (no CATEGORICAL_COLS change needed)

**Pre-computation note:** `variant_return_rate` = returns / sales — compute as a ratio, not raw counts, to avoid scale issues.
**Leakage check:** Stock levels don't encode any current target. `variant_return_rate` could correlate with `has_refund` / `size_issue` but is a legitimate feature (it's per-variant history, not per-order).

---

### 8.4 Email engagement — aggregate per customer

- [ ] Add `_load_email_features(data_dir)`:
  - Load `email_events.csv` (11,368 rows — event types: `sent`, `opened`, `clicked`, `converted`)
  - **Exclude `converted` events** — 65% of events are "converted", auto-attributed, not genuine engagement
  - Aggregate per `customer_id` over genuine engagement events (sent/opened/clicked only):
    - `email_open_count` — total opens per customer
    - `email_click_count` — total clicks per customer
    - `email_campaign_count` — distinct campaigns customer appeared in
    - `days_since_last_email` — days between most recent email event and 2026-06-04 (reference date). Use large sentinel (999) for customers with no emails.
  - Join to feature matrix on `customer_id`
- [ ] Fill: `email_open_count = 0`, `email_click_count = 0`, `email_campaign_count = 0`, `days_since_last_email = 999` for customers with no email history (84% of customers)
- [ ] Drop: `event_id`, `campaign_id`, `timestamp`, `event_type` (used only for aggregation)
- [ ] New features → all **numeric**

**Zero-inflation note:** 84% of customers have no email history — filling with 0 / 999 is correct. Do NOT filter rows. These customers are a real cohort (non-email-engaged).
**Leakage check:** Email engagement does not encode any target. Clean.

---

### 8.5 Customer addresses — extract geography only

- [ ] Add `_load_address_features(data_dir)`:
  - Load `addresses.csv` (22,440 rows — one per customer)
  - Extract `postcode_district` = first segment of UK postcode (e.g., "IG1" from "IG1 1AT")
  - Keep `city` as-is
  - **Drop all PII**: `first_name`, `last_name`, `address1`, `address2` (100% null anyway), `province` (90% null), `country` (all GB — zero variance)
  - Join to feature matrix on `customer_id`
- [ ] Fill: `postcode_district = "unknown"`, `city = "unknown"` for customers with no address
- [ ] Add `"postcode_district"` and `"city"` to `CATEGORICAL_COLS`

**PII note:** Only postcode_district (area-level, not property-level) and city are used. Full postcode, full address, and names are dropped before any ML processing.

---

### 8.6 Support messages — sentiment features

- [ ] Add `_load_support_messages(data_dir)`:
  - Load `support_messages.json` — full conversation transcripts per ticket
  - Parse per-ticket: `role` (customer / bot / human) + `content` (text)
  - Compute:
    - `msg_count` — total messages in thread
    - `customer_msg_count` — customer-only messages
    - `avg_customer_msg_length` — avg chars per customer message (frustration proxy)
    - `n_escalations` — bot→human handoffs in thread
    - `response_time_first_seconds` — seconds to first agent reply
  - Join to `support_tickets` on `ticket_id`, then inherited via `order_id`
- [ ] Fill: all 0 for orders with no ticket (same pattern as `resolution_time_minutes`)
- [ ] New features → all **numeric**

**Expected impact:** `satisfaction_rating` RMSE should drop below 1.0 (currently at naive baseline 1.17).

---

### 8.7 Validate all new features

- [ ] `python nn/estimator.py --list_targets` — confirm new columns appear in the feature list
- [ ] `python nn/estimator.py --target satisfaction_rating --epochs 30` — expect RMSE < 1.17
- [ ] `python nn/estimator.py --target has_refund --epochs 30` — AUC should hold ≥ 0.83 (no regression)
- [ ] `python nn/estimator.py --target damaged_in_transit --epochs 30` — `variant_return_rate` should appear in top importances
- [ ] `python nn/estimator.py --target total_price --epochs 30` — RMSE should hold ≤ £4.37

**Files to change:** `nn/data_builder.py` only — 6 new load functions + updates to `_build_joined()`, `CATEGORICAL_COLS`, and `drop_cols`.

**New feature count:** +4 categorical + ~11 numeric = ~15 total new features. Feature matrix grows from 53 → ~68 columns.

**Effort:** ~90 min

---

## Phase 9 — LLM Recommendation Layer on `--predict`

> After the neural network makes a prediction, pass the input features + prediction to a Claude API call that returns a plain-English recommendation: *"High refund risk — consider reducing price by 10% or updating the sizing guide for this collection."*

### 9.1 Add `--recommend` flag to CLI ✅ DONE
- [x] Add `--recommend` boolean flag and `--llm` model override to `parse_args()`
- [x] Only active when `--predict` and `--load_model` are also set — warns and ignores otherwise
- [x] Validate `OPENROUTER_API_KEY` env var — prints clear error if not set
- [x] `.env.example` added; `.env` added to `.gitignore`

### 9.2 LLM recommendation via OpenRouter ✅ DONE
- [x] `get_recommendation()` built with structured prompt: target, task type, prediction, input features, top 5 importances, brand context
- [x] Uses `openai` SDK with `base_url=https://openrouter.ai/api/v1` (OpenRouter-compatible)
- [x] Default model order: DeepSeek free → Gemma 3 free → Mistral cheap
- [x] Streams response token-by-token
- [x] `top_importances` persisted in `.pkl` at `--save_model` time

### 9.3 Known issues / open items
- [ ] **`user_row` in prompt is noisy** — `input_row` passed to the LLM contains all 70+ feature columns (most defaulting to 0.0) rather than only the keys the user actually provided. The LLM sees noise instead of signal. Fix: capture the original parsed JSON keys before auto-computation and filter to those only. See bottom of TODO for full description.

**Files changed:**
- `nn/estimator.py` — `parse_args()`, `load_and_predict()`, `get_recommendation()`, `save_model()`
- `requirements_nn.txt` — added `openai` (not anthropic)
- `.env.example`, `.gitignore`

---

## Phase 10 — Sentiment Analysis of Customer Messages (VADER)

> Score the actual text of customer support messages to produce per-ticket sentiment features. Uses VADER — free, local, no API, runs in milliseconds. Primary goal: fix `satisfaction_rating` (currently at naive baseline because Phase 8 structural features had zero variance in this dataset).

**Why VADER:** Designed for short informal social-media-style text, exactly like support messages. Returns a compound score from -1.0 (very negative) to +1.0 (very positive). Pure Python, no GPU, no cost.

**Why Phase 8.6 didn't fix satisfaction_rating:** We counted messages and timed responses, but never read the words. `n_escalations` = 0 for every ticket and `response_time` = 120s for every ticket in this dataset — zero variance, zero signal. Sentiment scoring reads the actual text and will produce real variance.

---

### 10.1 Install VADER

- [ ] Add `vaderSentiment` to `requirements_nn.txt`
- [ ] Confirm install: `python -c "from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer"`

---

### 10.2 Add `_load_support_sentiment_features()` to `nn/data_builder.py`

- [ ] Load `support_messages.json` (already done in `_load_support_message_features` — share the parse logic or call it separately)
- [ ] Initialise one `SentimentIntensityAnalyzer()` instance (reuse across all messages — it's stateless)
- [ ] For each ticket, score **customer messages only** (skip bot/human agent messages — we want the customer's tone, not the bot's)
- [ ] Compute per-ticket features:
  - `avg_sentiment` — mean compound score across all customer messages in the thread (−1 to +1)
  - `min_sentiment` — lowest (most negative) single customer message score (catches peak frustration)
  - `pct_negative_msgs` — fraction of customer messages with compound < −0.05 (VADER's negativity threshold)
- [ ] Build a DataFrame with `ticket_id` + 3 sentiment cols
- [ ] Join to `support_tickets.csv` on `ticket_id` → get `order_id`
- [ ] Return DataFrame keyed by `order_id`

**Example:**
```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
score = sia.polarity_scores("This is absolutely awful!!")["compound"]  # → -0.8402
```

---

### 10.3 Join into master table in `_build_joined()`

- [ ] Add `sentiment_features` parameter to `_build_joined()` signature
- [ ] LEFT JOIN on `order_id` after the support_tickets join (same position as `support_message_features`)
- [ ] Fill NaN with 0 for orders with no ticket (neutrality — no message means no signal either way)
- [ ] New columns are numeric — no change to `CATEGORICAL_COLS`
- [ ] Add to `FILTER_MAP`:
  - `"avg_sentiment": ("has_ticket", 1)` — only meaningful for ticket orders
  - `"min_sentiment": ("has_ticket", 1)`
  - `"pct_negative_msgs": ("has_ticket", 1)`
- [ ] Wire into `build_feature_table()` — call `_load_support_sentiment_features()` alongside the other new loaders

---

### 10.4 Update `load_and_predict()` in `nn/estimator.py`

- [ ] The 3 sentiment features will be in the saved model's `feature_meta.num_cols` after retraining
- [ ] No user-facing change needed — missing sentiment fields default to 0 at inference (neutral sentiment assumed for orders without a support thread)

---

### 10.5 Validate

```bash
# Check features appear in the table
python3 -c "
from nn.data_builder import build_feature_table
df = build_feature_table()
print(df[['avg_sentiment','min_sentiment','pct_negative_msgs']].describe().round(3))
rated = df.dropna(subset=['satisfaction_rating'])
print('Correlations with satisfaction_rating:')
print(rated[['avg_sentiment','min_sentiment','pct_negative_msgs','satisfaction_rating']].corr()['satisfaction_rating'].round(3))
"

# Retrain satisfaction_rating — expect RMSE meaningfully below 1.17 (current naive baseline)
python nn/estimator.py --target satisfaction_rating --epochs 30

# Confirm avg_sentiment and min_sentiment appear in top 5 importances
# Confirm no regression on has_refund (AUC should stay ≥ 0.84)
python nn/estimator.py --target has_refund --epochs 30
```

**Success criteria:**
- `avg_sentiment` and `min_sentiment` have non-trivial correlation (|r| > 0.1) with `satisfaction_rating`
- `satisfaction_rating` RMSE drops below 1.10 (below naive baseline of 1.15)
- `avg_sentiment` appears in top 5 feature importances for `satisfaction_rating`

**Files to change:**
- `requirements_nn.txt` — add `vaderSentiment`
- `nn/data_builder.py` — add `_load_support_sentiment_features()`, update `_build_joined()`, `FILTER_MAP`, `build_feature_table()`

**Effort:** ~30 min

---

## Pre-Phase 10 Cleanup

> Items identified during audit before adding Phase 10. Fix these first to avoid stale docs misleading reviewers and to unblock clean Phase 10 implementation.

---

### 🔴 D1 — `REPORT.md` is significantly stale *(REPORT.md)*

- **Problem:** Written before leakage fixes and Phase 8. Contains wrong metrics that will mislead anyone reading it:
  - `has_refund` AUC shown as **1.0000** — real is **0.844** (leakage fixed in B1)
  - `total_price` RMSE shown as **£6.42** — real is **£4.37** (price_components_sum fix)
  - References only **10 tables** joined — now **21**
  - Only **6 targets** covered — now **73 columns** available
  - Summary table still shows fake perfect pre-fix results
- **Fix:** Rewrite evaluation results section with live numbers; update join chain; update target count; add Phase 8 new targets
- **File:** `REPORT.md`
- **Effort:** 20 min

---

### 🔴 D2 — `README_nn.md` has stale numbers *(README_nn.md)*

- **Problem:** Numbers written before Phase 8 still visible to anyone visiting the repo:
  - "51 things you can predict" → now **73**
  - "10 tables" → **21**
  - "353 lines" in data_builder → **579**
  - Missing all Phase 8 new targets (`delivery_delay_days`, `variant_return_rate`, `city`, etc.)
- **Fix:** Update feature count, table count, line count, target examples table
- **File:** `README_nn.md`
- **Effort:** 10 min

---

### 🟡 D3 — `support_messages.json` loaded twice *(nn/data_builder.py)*

- **Problem:** `_load_support_message_features()` already reads `support_messages.json` (1.5MB). Phase 10 will add `_load_support_sentiment_features()` which reads the same file again. Two full reads of the same file per `build_feature_table()` call — ~0.5s wasted, messy.
- **Fix:** Extract a shared `_parse_support_messages()` that reads and parses the JSON once, returns the raw list. Both loaders call that instead of opening the file themselves.
- **File:** `nn/data_builder.py`
- **Effort:** 5 min
- **Must fix before Phase 10** — otherwise Phase 10 adds a third load

---

### 🟡 D4 — `pretty_fly_master.csv` is stale locally *(gitignored)*

- **Problem:** Generated before Phase 8 at 53 columns. Now the feature matrix is 73 columns. File is gitignored so doesn't affect the repo, but if opened locally it will show wrong/missing features.
- **Fix:** Regenerate: `python3 -c "from nn.data_builder import build_feature_table; build_feature_table().to_csv('pretty_fly_master.csv', index=False)"`
- **Effort:** 1 min (just run the command)

---

### 🟢 D5 — U3 already fixed — mark done in TODO *(TODO.md)*

- **Problem:** U3 (`evaluate.py --all` rebuilds table 6×) is listed as pending but was already fixed. `evaluate.py` builds the table once in `main()` at line 415 and passes `df` to each `run_target(df, ...)`.
- **Fix:** Mark U3 as ✅ DONE in the priority table
- **Effort:** 1 min

---

### 🟢 D6 — Phase 10 inference gap is documented but not handled *(nn/estimator.py)*

- **Problem:** After Phase 10, `avg_sentiment`, `min_sentiment`, `pct_negative_msgs` will be in the feature matrix. At `--predict` time users won't have a support message thread to score. Currently these will default to 0 (neutral) via the existing missing-field fallback — which is the correct behaviour — but it's not documented anywhere.
- **Fix:** Add a comment in `load_and_predict()` noting that sentiment features default to 0 (neutral) at inference, meaning "no support thread = assume neutral customer"
- **File:** `nn/estimator.py`
- **Effort:** 2 min

---

### Priority order for cleanup

| # | ID | Item | Effort | Block? |
|---|----|------|--------|--------|
| 1 | D3 | Fix double JSON load | 5 min | ✋ Blocks Phase 10 |
| 2 | D5 | Mark U3 done in TODO | 1 min | — |
| 3 | D4 | Regenerate master CSV | 1 min | — |
| 4 | D2 | Update README_nn.md | 10 min | — |
| 5 | D1 | Update REPORT.md | 20 min | — |
| 6 | D6 | Document inference gap | 2 min | — |

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

---

## Improvements & Fixes (post-build audit)

Findings from code inspection + live evaluation runs. Each item has a root cause and a concrete fix.

---

### 🔴 Bugs / Data Issues

**B1 — Data leakage in `has_refund` target** *(nn/data_builder.py)* ✅ FIXED
- **Was:** AUC=1.0000 (fake — `financial_status`, `refund_reason`, `refund_amount` encoded the answer)
- **Fix applied:** `LEAKAGE_MAP` in `data_builder.py`; `prepare_for_target()` drops leakage cols and logs `"Leakage: dropped [...]"`
- **Result:** AUC=0.6596 (honest), top features now `subtotal`, `total_price`, `product_type`
- **Git:** `1f3278d`

**B2 — `resolution_time_minutes` is 97.6% zeros** *(nn/data_builder.py)* ✅ FIXED
- **Was:** RMSE=50 min with R²=0.78 — inflated by predicting 0 for 68,251 non-ticket rows
- **Fix applied:** `FILTER_MAP` in `data_builder.py`; filters to `has_ticket==1` (1,705 rows) for this target. Logs `"Filter: has_ticket==1 → 1,705 rows (from 69,956)"`
- **Result:** RMSE=323 min on real ticket data, y range [5, 1439] min, zero structural zeros
- **Git:** `1f3278d`

**B3 — Bar chart scaling in estimator uses fixed multiplier** *(nn/estimator.py:133)* ✅ FIXED
- **Was:** `score * 200` — all top features showed identical 20-bar width when scores were large
- **Fix applied:** `int(20 * score / top_score)` — bars now proportional relative to top feature
- **Result:** `subtotal` = 20 bars, `total_tax` = 2 bars, `total_shipping` = 1 bar (correct proportions)
- **Git:** `1f3278d`

---

### 🟡 Model / Training Issues

**M1 — Early stopping has no `min_delta` threshold** *(nn/model.py)* ✅ FIXED
- **Was:** Near-perfect models (product_type, has_refund) ran all N epochs; float32 noise kept resetting patience counter
- **Fix applied:** `min_delta=1e-5` param on `train_model()`; patience only resets when `best - val > min_delta`
- **Result:** `product_type` now stops at epoch 8 (was running 30-50 epochs)
- **Git:** `f156bc9`

**M2 — No global random seed — results are not reproducible** *(nn/model.py)* ✅ FIXED
- **Was:** Weight init varied every run; metrics differed by ±0.05 between identical calls
- **Fix applied:** `seed=42` param on `train_model()`; seeds `torch`, `numpy`, `torch.cuda` at function start
- **Result:** Verified identical metrics across 3 repeated runs on `product_type`, `total_price`, `satisfaction_rating`
- **Git:** `1f3278d`

**M3 — Noisy val_loss for `total_price` — no LR scheduler** *(nn/model.py)* ✅ FIXED
- **Was:** No LR schedule — wasted epochs on noisy plateaus; oscillations visible (e.g. epoch 7: 79 → epoch 9: 93 → epoch 11: 38)
- **Fix applied:** `ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)` after Adam; epoch line shows `lr=X ↓` on reduction
- **Result:** Fires on genuinely plateauing targets (`satisfaction_rating` LR halved at epoch 14); correctly does NOT fire on `total_price` (model still improving throughout). Scheduler wiring verified with controlled constant-loss test.
- **Git:** `f156bc9`

**M4 — `satisfaction_rating` val set is only 107 rows** *(nn/model.py)* ✅ FIXED
- **Was:** 539 rows → 80/20 split → 107 val rows; R² varied ±0.3 between identical runs
- **Fix applied:** `test_size=0.3` when `len(y) < 1000`; logged as `"Sparse target (N rows): using 30% val split"`
- **Result:** `satisfaction_rating` val set grows from 107 → 162 rows; metrics more stable
- **Git:** `f156bc9`
- **File:** `nn/model.py`, `train_model()` — check `len(y)` and adjust split.

---

### 🟢 Feature / Data Improvements

**F1 — Sentiment features from `support_messages.json` not used** *(nn/data_builder.py)*
- **Problem:** `support_messages.json` contains full conversation text (customer/bot/human turns) for all 1,204 tickets. This is the richest signal for `satisfaction_rating` and `resolved_by` targets, but it is never loaded. The current `satisfaction_rating` model performs at naive baseline (R²≈0) because subjective satisfaction isn't encoded anywhere in the current features.
- **Fix:** Load `support_messages.json`, compute per-ticket features: `msg_count`, `customer_msg_count`, `avg_customer_msg_length`, `n_escalations` (bot→human handoffs), `response_time_first` (seconds to first agent reply). Join to `support_tickets` on `ticket_id` → `order_id`. This alone would likely push `satisfaction_rating` RMSE below 1.0.
- **File:** Add `_load_support_messages()` to `nn/data_builder.py`, join in `_build_joined()`.

**F2 — Ad join is campaign-day level, not order-specific** *(nn/data_builder.py)*
- **Problem:** Multiple orders in the same campaign on the same day share identical ad features. There are 18 campaign names and ~730 unique campaign-days in the dataset — so ad features are effectively campaign-period averages, not per-order signals. This limits ad-related targets (`google_conversions`, `meta_conversions`).
- **Fix (pragmatic):** Document this clearly in a comment in `_build_joined()`. The fix would require knowing which ad impression led to which click led to which order — that data doesn't exist in this dataset. Current approach is the best possible with available data.
- **File:** `nn/data_builder.py`, `_build_joined()` — add a comment.

**F3 — `gross_margin_est` fill uses a confusing lambda pipe** *(nn/data_builder.py:169–171)*
- **Problem:** The fill for `gross_margin_est` when `landed_cost_per_unit_gbp` is missing uses `df["price"].pipe(lambda s: (s - s.median()) / s.replace(0, np.nan)).fillna(0)`. In practice, all 69,956 rows have a landed cost (0% missing verified), so the fallback never fires — but the formula is wrong anyway (it computes a normalised price deviation, not a margin).
- **Fix:** Since there are zero missing values, simplify to just fill with the global median margin: `.fillna(df["gross_margin_est"].median())` after the initial computation. Cleaner and correct.
- **File:** `nn/data_builder.py`, line 169–171.

---

### 🔵 UX / CLI Improvements

**U1 — `--predict` doesn't validate column names in JSON input**
- **Problem:** If you pass `--predict '{"pric": 85.0}'` (typo), it silently uses `0.0` for `price`. The user gets a prediction with no warning that their input was ignored.
- **Fix:** In `load_and_predict()`, after parsing the JSON row, print a warning for any key not found in `cat_cols + num_cols`: `Unknown input keys (ignored): {set(row.keys()) - set(all_feature_cols)}`.
- **File:** `nn/estimator.py`, `load_and_predict()`.

**U2 — No `--leakage_exclude` flag for `has_refund`-style targets**
- **Problem:** There is no way to run `has_refund` as a genuine predictor without manually editing `data_builder.py`. Users demoing this tool should be able to test the honest version.
- **Fix (ties to B1):** Once `LEAKAGE_MAP` is added in B1, expose it via CLI: `--exclude_cols financial_status,refund_reason,refund_amount`. Overrides or extends the default leakage map.
- **File:** `nn/estimator.py`, `parse_args()` + `main()`.

**U3 — `evaluate.py --all` reloads the feature table 6 times**
- **Problem:** `build_feature_table()` is called inside `run_target()` on each of the 6 targets. Each call re-reads and re-joins all 10 CSVs (~3s per call = 18s wasted).
- **Fix:** Build the table once in `main()` and pass `df` as an argument to `run_target(df, ...)`.
- **File:** `nn/evaluate.py`, `main()` and `run_target()` signature.

---

### Priority order for hackathon

| # | Item | Effort | Impact | Status |
|---|------|--------|--------|--------|
| 1 | **B1** — Fix leakage in `has_refund` | 20 min | Makes the demo honest | ✅ DONE |
| 2 | **B2** — Fix `resolution_time_minutes` zero inflation | 15 min | Fixes misleading R²=0.78 | ✅ DONE |
| 3 | **M2** — Seed torch for reproducibility | 5 min | Stable demo runs | ✅ DONE |
| 4 | **M1** — Add `min_delta` to early stopping | 10 min | Cleaner training output | ✅ DONE |
| 5 | **B3** — Fix bar chart scaling | 5 min | Readable importance output | ✅ DONE |
| 6 | **F1** / **Ph8** — Full data integration (all 21 files) | 90 min | 73→76 cols, new targets | ✅ DONE |
| 7 | **U3** — Build table once in evaluate.py | 5 min | 18s saved per --all run | ✅ DONE |
| 8 | **M3** — LR scheduler for noisy regression | 15 min | Smoother total_price training | ✅ DONE |
| 9 | **U1** — Warn on unknown predict keys | 10 min | Better UX | ✅ DONE |
| 10 | **M4** — Cross-val for sparse targets | 30 min | Reliable satisfaction metrics | ✅ DONE |
| 11 | **Ph7** — `--subset` flag for per-product filtering | 20 min | Slice training by product type | ✅ DONE |
| 12 | **Ph9** — LLM recommendation via OpenRouter (`--recommend` + `--llm`) | 90 min | Actionable business output | ✅ DONE |
| 13 | **Ph10** — VADER sentiment on customer messages | 30 min | avg/min/pct_negative per ticket | ✅ DONE |
| 14 | **D3** — Shared `_parse_support_messages()` — one JSON read | 5 min | No duplicate file read in Ph10 | ✅ DONE |
| 15 | **D1** — Rewrite REPORT.md with live 30-epoch metrics | 20 min | Accurate docs for judges | ✅ DONE |
| 16 | **D2** — Update README_nn.md stale numbers | 10 min | 51→76 targets, 10→21 files | ✅ DONE |
| 17 | **D6** — Document sentiment inference gap in load_and_predict | 2 min | Comment added | ✅ DONE |
| 18 | **F3** — Simplify gross_margin_est fallback | 5 min | Dead lambda removed | ✅ DONE |
| 19 | **R1** — Fix `user_row` noise in `--recommend` prompt | 15 min | LLM sees signal not zeros | ⬜ TODO |

---

## Open Issue R1 — `user_row` prompt noise in `--recommend`

> **Status:** ⬜ TODO — deferred, review before demo

### Problem

In `load_and_predict()`, the `user_row` passed to `get_recommendation()` is built like this:

```python
user_row = {k: v for k, v in row.items() if k in set(cat_cols) | set(num_cols)}
```

By this point, `row` has already had auto-computed engineered features added to it (`discount_pct`, `gross_margin_est`, `total_ad_spend`, `price_components_sum`) **and** all 70+ model features will default to `0.0` when the user only provided 2–3 keys. So the LLM receives something like:

```
price: 85.0
product_type: Hoodie
discount_pct: 0.0
gross_margin_est: 0.0
total_ad_spend: 0.0
email_open_count: 0.0
avg_sentiment: 0.0
lead_time_days: 0.0
... (60+ more zero-value defaults)
```

This drowns out the actual user signal. The LLM has no way to tell which values were intentional vs default fills.

### Why it matters

The whole value of `--recommend` is that it ties advice to *what the user told us*. With 60+ zero-noise features in the prompt, the recommendation becomes generic — the model can't distinguish "customer has city=London" from "city was not provided, defaulted to 0".

### Fix

Capture the original JSON keys **before** auto-computation and use those as the filter:

```python
# In load_and_predict(), right after parsing the JSON:
user_provided_keys = set(row.keys())   # capture before auto-compute

# ... auto-computation block runs ...

# When building user_row for the recommendation:
user_row = {k: v for k, v in row.items()
            if k in user_provided_keys and k in (set(cat_cols) | set(num_cols))}
```

This way the LLM only sees the 2–5 features the user actually typed, making the recommendation specific and actionable.

### Files to change
- `nn/estimator.py` — `load_and_predict()` only, ~3 lines

### Effort
~5 min code, ~10 min to test prompt quality with a real API call
