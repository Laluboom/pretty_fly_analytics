# Pretty Fly Neural Estimator — How to Use

Every feature, flag, and function with concrete examples. Run all commands from the project root (`pretty_fly_data_pack/`).

---

## Contents

1. [Installation](#1-installation)
2. [Validate the dataset](#2-validate-the-dataset)
3. [List all valid targets](#3-list-all-valid-targets)
4. [Train a model](#4-train-a-model)
5. [Save a model to disk](#5-save-a-model-to-disk)
6. [Save a model + plot importance chart](#6-save-a-model--plot-importance-chart)
7. [Train on a data slice — `--subset`](#7-train-on-a-data-slice----subset)
8. [Predict from a saved model — `--predict`](#8-predict-from-a-saved-model----predict)
9. [Get an LLM recommendation — `--recommend`](#9-get-an-llm-recommendation----recommend)
10. [Override the LLM model — `--llm`](#10-override-the-llm-model----llm)
11. [Full evaluation suite — `evaluate.py`](#11-full-evaluation-suite----evaluatepy)
12. [Evaluate all 11 key targets at once — `--all`](#12-evaluate-all-11-key-targets-at-once----all)
13. [Custom data directory — `--data_dir`](#13-custom-data-directory----data_dir)
14. [Python API — use modules directly](#14-python-api----use-modules-directly)

---

## 1. Installation

```bash
pip install -r requirements_nn.txt
```

Installs: `torch`, `pandas`, `numpy`, `scikit-learn`, `tqdm`, `matplotlib`, `vaderSentiment`, `openai`

For `--recommend` (LLM layer), set your OpenRouter key:

```bash
cp .env.example .env
# edit .env and paste your key
export OPENROUTER_API_KEY=sk-or-...
```

---

## 2. Validate the dataset

Runs 20 reconciliation rules (row counts, financial totals, referential integrity) with 1-penny tolerance. Does not modify any data.

```bash
python validate.py data/
```

Expected output:
```
20 passed, 0 failed, 0 skipped
```

Exit code 0 on full pass, 1 if any rule fails.

---

## 3. List all valid targets

Prints all 76 columns in the feature matrix with their non-null percentage and usable row count.

```bash
python nn/estimator.py --list_targets
```

Example output (truncated):
```
Valid target columns:
  avg_sentiment                        2.4% non-null  (1,686 rows)
  city                               100.0% non-null  (69,956 rows)
  damaged_in_transit                  14.3% non-null  (10,025 rows)
  delivery_delay_days                100.0% non-null  (69,956 rows)
  has_refund                         100.0% non-null  (69,956 rows)
  resolution_time_minutes              2.4% non-null  (1,705 rows)
  satisfaction_rating                  0.8% non-null  (539 rows)
  total_price                        100.0% non-null  (69,956 rows)
  ...
```

> Sparse targets (low non-null %) train on fewer rows — the model automatically switches to a 70/30 val split when the target has fewer than 1,000 rows.

---

## 4. Train a model

Trains from scratch on a target column and prints results: metric, top-10 feature importances, 5 example predictions.

```bash
# Binary classification — predicts refund probability
python nn/estimator.py --target has_refund

# Regression — predicts order value
python nn/estimator.py --target total_price

# Multi-class classification — predicts product category
python nn/estimator.py --target product_type

# Sparse regression — only 539 rows with a support ticket
python nn/estimator.py --target satisfaction_rating

# Train with more epochs (default is 50; early stopping will fire earlier if appropriate)
python nn/estimator.py --target total_price --epochs 100

# Train with a custom batch size (default 512)
python nn/estimator.py --target satisfaction_rating --batch_size 32
```

**What it prints:**
```
==============================
  RESULTS
==============================
  Target  : has_refund
  Task    : binary
  AUC     : 0.8436

  Top 10 Feature Importances (permutation):
  Rank  Feature                             Delta Loss
  ----------------------------------------------------
  1     size_issue                           0.79418  ████████████████████
  2     damaged_in_transit                   0.27823  ███████
  3     product_type                         0.04211  █
  ...

  5 Example Predictions vs Actuals (val set sample):
  #       Predicted         Actual
  ----------------------------------
  1           0.081              0
  2           0.923              1
  ...

  Business insight: Top return drivers: size_issue, damaged_in_transit, product_type
==============================
```

---

## 5. Save a model to disk

Saves two files: `{prefix}.pt` (model weights) and `{prefix}.pkl` (feature metadata, encoders, scaler, task type, top importances). Required for `--predict` and `--recommend` later.

```bash
# Save to models/ directory
python nn/estimator.py --target has_refund --epochs 30 --save_model models/has_refund

# Save regression model
python nn/estimator.py --target total_price --epochs 50 --save_model models/total_price

# Save classification model
python nn/estimator.py --target product_type --epochs 30 --save_model models/product_type

# Save a sparse target (small dataset)
python nn/estimator.py --target satisfaction_rating --batch_size 32 --save_model models/satisfaction
```

Output:
```
Model saved: models/has_refund.pt  +  models/has_refund.pkl
```

> The `models/` directory is created automatically. It is gitignored — model files are not committed.

---

## 6. Save a model + plot importance chart

`--plot` saves `importance_{target}.png` — a horizontal bar chart of the top-15 features ranked by permutation importance. Can be combined with any training run.

```bash
python nn/estimator.py --target has_refund --epochs 30 --save_model models/has_refund --plot

python nn/estimator.py --target total_price --epochs 50 --plot

# Plot only, no save
python nn/estimator.py --target product_type --epochs 20 --plot
```

Output file: `importance_has_refund.png` (saved in current working directory)

> PNG files are gitignored.

---

## 7. Train on a data slice — `--subset`

Filters the training data to rows matching `col=value` before training. Useful for product-specific models.

```bash
# Train has_refund predictor on Hoodies only
python nn/estimator.py --target has_refund --epochs 30 --subset "product_type=Hoodie"

# Train on Sweatpants only
python nn/estimator.py --target total_price --epochs 30 --subset "product_type=Sweatpants"

# Train on a specific supplier country
python nn/estimator.py --target delivery_delay_days --epochs 30 --subset "supplier_country=Portugal"

# Train on London customers only
python nn/estimator.py --target has_refund --epochs 30 --subset "city=London"

# Combine with --save_model
python nn/estimator.py --target has_refund --epochs 30 \
  --subset "product_type=Hoodie" \
  --save_model models/has_refund_hoodies
```

Log line printed on filter:
```
Subset : product_type==Hoodie → 13,803 rows (from 69,956)
```

**If the column or value doesn't exist:**
```
Error: --subset value 'Onesie' not found in column 'product_type'.
  Available values: ['Cap', 'Hoodie', 'Outerwear', 'Sweatpants', 'Tee', 'Trainer']
```

---

## 8. Predict from a saved model — `--predict`

Loads a saved model and predicts on a single JSON row. Fields not provided default to 0 / "unknown". Engineered features (`discount_pct`, `gross_margin_est`, `total_ad_spend`, `price_components_sum`) are auto-computed from components if their inputs are present.

```bash
# Minimal input — most features default
python nn/estimator.py \
  --predict '{"price": 85.0, "product_type": "Hoodie"}' \
  --load_model models/has_refund

# Richer input — engineered features computed automatically
python nn/estimator.py \
  --predict '{"price": 129.17, "subtotal": 129.17, "total_shipping": 4.99, "total_tax": 26.0, "product_type": "Sweatpants", "collection": "Autumn 25"}' \
  --load_model models/total_price

# Classification target — output is a decoded label with confidence
python nn/estimator.py \
  --predict '{"price": 45.0, "weight_grams": 120}' \
  --load_model models/product_type

# Predict from a JSON file instead of inline string
echo '{"price": 85.0, "product_type": "Hoodie"}' > input.json
python nn/estimator.py --predict input.json --load_model models/has_refund
```

**Typo / unknown key warning:**
```bash
python nn/estimator.py \
  --predict '{"pric": 85.0, "product_type": "Hoodie"}' \
  --load_model models/has_refund
```
```
Warning: unknown input keys (ignored): ['pric']
Predicting: has_refund
Input features used: 19 categorical + 57 numeric
Prediction : 0  (probability=0.0834)
```

**Example outputs by task type:**
```
# Binary
Prediction : 1  (probability=0.8231)

# Regression
Prediction : 127.4392

# Classification
Prediction : Hoodie  (confidence=0.9871)
```

> Sentiment features (`avg_sentiment`, `min_sentiment`, `pct_negative_msgs`) default to 0 at inference — no support thread means neutral customer, which is the correct assumption for new orders.

---

## 9. Get an LLM recommendation — `--recommend`

After predicting, sends the prediction + input features + top-5 importances to an LLM via OpenRouter and streams a plain-English business recommendation. Requires `OPENROUTER_API_KEY`.

```bash
export OPENROUTER_API_KEY=sk-or-...

# Get refund risk + recommendation
python nn/estimator.py \
  --predict '{"price": 129.17, "product_type": "Hoodie", "collection": "Autumn 25"}' \
  --load_model models/has_refund \
  --recommend

# Recommend on resolution time prediction
python nn/estimator.py \
  --predict '{"ticket_category": "sizing", "support_channel": "email"}' \
  --load_model models/resolution_time \
  --recommend

# Recommend after subset-trained model
python nn/estimator.py \
  --predict '{"price": 85.0, "collection": "Core"}' \
  --load_model models/has_refund_hoodies \
  --recommend
```

Example output:
```
============================================================
  RECOMMENDATION  (deepseek/deepseek-chat-v3-0324:free)
============================================================
Risk/Outlook : High refund risk (81%) — sizing and transit damage are the primary drivers.
Key signal   : size_issue is the dominant feature; this Hoodie collection has a historically
               elevated return rate due to fit inconsistency.

Suggested actions:
1. Add a detailed fit guide to the Autumn 25 Hoodie PDP before the next campaign.
2. Review the supplier packing process — damaged_in_transit is the #2 signal.
3. Consider size-specific stock weighting: size M and L show the highest return rates.
============================================================
```

> `--recommend` is silently ignored if `--predict` and `--load_model` are not also set.

---

## 10. Override the LLM model — `--llm`

By default `--recommend` tries models in this order:
1. `deepseek/deepseek-chat-v3-0324:free` (free)
2. `google/gemma-3-27b-it:free` (free)
3. `mistralai/mistral-small-3.1-24b-instruct` (cheap)

Override with any OpenRouter model ID:

```bash
# Use Gemma 3 explicitly
python nn/estimator.py \
  --predict '{"price": 85.0, "product_type": "Tee"}' \
  --load_model models/has_refund \
  --recommend \
  --llm "google/gemma-3-27b-it:free"

# Use Mistral
python nn/estimator.py \
  --predict '{"price": 85.0}' \
  --load_model models/total_price \
  --recommend \
  --llm "mistralai/mistral-small-3.1-24b-instruct"
```

> `--llm` is silently ignored if used without `--predict` and `--load_model`.

---

## 11. Full evaluation suite — `evaluate.py`

Trains a fresh model, computes deep metric suites, and optionally saves a 6-panel chart. More thorough than `estimator.py` — intended for measuring model quality rather than day-to-day use.

### Binary targets — metrics produced:
AUC, Average Precision, Accuracy, Precision, Recall, F1, Positive Rate, Confusion Matrix, ROC curve, Score Distribution, Precision-Recall curve

### Regression targets — metrics produced:
RMSE, MAE, R², Median Abs Error, Max Error, Within ±10%, Within ±20%, Residual Mean/Std, Naive RMSE baseline, Improvement over naive, Predicted vs Actual scatter, Residual histogram

### Classification targets — metrics produced:
Accuracy, Macro F1, Weighted F1, Per-class breakdown, Confusion Matrix, Per-class accuracy/F1 bar charts, Class distribution actual vs predicted

```bash
# Single binary target
python nn/evaluate.py --target has_refund --epochs 30

# Single regression target with chart
python nn/evaluate.py --target total_price --epochs 30 --plot

# Sparse regression (auto-uses 30% val split)
python nn/evaluate.py --target satisfaction_rating --epochs 50

# Multi-class target
python nn/evaluate.py --target product_type --epochs 20

# Phase 8/10 targets
python nn/evaluate.py --target damaged_in_transit --epochs 30 --plot
python nn/evaluate.py --target delivery_delay_days --epochs 30 --plot
python nn/evaluate.py --target variant_return_rate --epochs 30
python nn/evaluate.py --target city --epochs 30
python nn/evaluate.py --target size_issue --epochs 30

# Any column in the feature matrix is valid (not just the 11 KEY_TARGETS)
python nn/evaluate.py --target gross_margin_est --epochs 30 --plot
python nn/evaluate.py --target resolved_by --epochs 30
python nn/evaluate.py --target resolution_time_minutes --epochs 30 --plot
```

`--plot` saves `eval_{target}.png` (6-panel chart) in the current directory.

---

## 12. Evaluate all 11 key targets at once — `--all`

Trains all 11 key targets sequentially using a single shared feature table build, then prints a summary table.

```bash
# Run all 11 key targets, 30 epochs each
python nn/evaluate.py --all --epochs 30

# Run all 11 key targets + save a chart for each
python nn/evaluate.py --all --epochs 30 --plot

# Quick run, fewer epochs
python nn/evaluate.py --all --epochs 10
```

End-of-run summary table:
```
==============================================================
  SUMMARY — ALL TARGETS
==============================================================
  Target                         Task             Metric          Value
  ------------------------------------------------------------
  has_refund                     binary           AUC            0.8436
  satisfaction_rating            regression       RMSE           1.2125
  total_price                    regression       RMSE           4.8600
  product_type                   classification   Accuracy       1.0000
  resolved_by                    classification   Accuracy       1.0000
  resolution_time_minutes        regression       RMSE         315.2100
  damaged_in_transit             binary           AUC            0.8290
  size_issue                     binary           AUC            0.7500
  delivery_delay_days            regression       RMSE              ...
  variant_return_rate            regression       RMSE              ...
  city                           classification   Accuracy          ...
```

The 11 key targets and their default batch sizes:

| Target | Task | Batch size |
|--------|------|-----------|
| `has_refund` | binary | 512 |
| `satisfaction_rating` | regression | 64 |
| `total_price` | regression | 512 |
| `product_type` | classification | 512 |
| `resolved_by` | classification | 512 |
| `resolution_time_minutes` | regression | 512 |
| `damaged_in_transit` | binary | 512 |
| `size_issue` | binary | 512 |
| `delivery_delay_days` | regression | 512 |
| `variant_return_rate` | regression | 512 |
| `city` | classification | 512 |

---

## 13. Custom data directory — `--data_dir`

Both `estimator.py` and `evaluate.py` default to looking for `data/` relative to the project root. Override with `--data_dir` if your data lives elsewhere.

```bash
# estimator.py
python nn/estimator.py --target has_refund --data_dir /path/to/data

# evaluate.py
python nn/evaluate.py --target total_price --data_dir /path/to/data

# Useful in containerised or CI environments
python nn/estimator.py --target product_type --data_dir $DATA_PATH --save_model $MODEL_PATH/product_type
```

---

## 14. Python API — use modules directly

All functions are importable for use in notebooks or custom scripts.

### Build the feature table

```python
from nn.data_builder import build_feature_table

df = build_feature_table()
# Returns: pd.DataFrame, 69,956 rows × 76 columns

# Custom data path
df = build_feature_table(data_dir="/path/to/data")

print(df.shape)         # (69956, 76)
print(df.columns.tolist())
print(df[["has_refund", "total_price", "product_type"]].describe())
```

### Prepare data for a specific target

```python
from nn.data_builder import build_feature_table, prepare_for_target

df = build_feature_table()

X_cat, X_num, y, task_type, n_classes, feature_meta, target_encoder = prepare_for_target(df, "has_refund")
# X_cat: np.ndarray [n, 19] — label-encoded categoricals
# X_num: np.ndarray [n, 57] — standardised numerics
# y:     np.ndarray [n]     — target values
# task_type: "binary" | "regression" | "classification"
# n_classes: int (1 for binary/regression, k for classification)
# feature_meta: dict with cat_cols, num_cols, cat_vocab_sizes, encoders, scaler
# target_encoder: LabelEncoder or None
```

### Train a model

```python
from nn.model import train_model

model, val_loss, val_metric, metric_name, val_idx, preds, targets = train_model(
    X_cat, X_num, y,
    task_type=task_type,
    n_classes=n_classes,
    feature_meta=feature_meta,
    epochs=30,
    batch_size=512,
)
# model:      PrettyFlyNet (PyTorch module, best weights restored)
# val_loss:   float — final validation loss
# val_metric: float — AUC / accuracy / RMSE
# metric_name: str
# val_idx:    np.ndarray — indices of validation rows in the original array
# preds:      np.ndarray — validation predictions
# targets:    np.ndarray — validation ground truth
```

### Compute permutation importance

```python
from nn.estimator import permutation_importance

device = next(model.parameters()).device

ranked = permutation_importance(
    model,
    X_cat[val_idx], X_num[val_idx], y[val_idx],
    task_type, feature_meta, device,
    n_repeats=3,
)
# Returns: [(feature_name, avg_delta_loss), ...] sorted descending

for feat, score in ranked[:10]:
    print(f"{feat}: {score:.5f}")
```

### Load a saved model and predict

```python
from nn.estimator import load_and_predict

load_and_predict(
    prefix="models/has_refund",
    raw_input='{"price": 85.0, "product_type": "Hoodie"}',
    recommend=False,
)

# With LLM recommendation (requires OPENROUTER_API_KEY in env)
load_and_predict(
    prefix="models/has_refund",
    raw_input='{"price": 85.0, "product_type": "Hoodie"}',
    recommend=True,
    llm_model="google/gemma-3-27b-it:free",
)

# From a JSON file
load_and_predict(
    prefix="models/has_refund",
    raw_input="input.json",
)
```

### Save a model

```python
from nn.estimator import save_model

save_model(
    model=model,
    feature_meta=feature_meta,
    task_type=task_type,
    n_classes=n_classes,
    target_col="has_refund",
    target_encoder=target_encoder,
    prefix="models/has_refund",
    top_importances=ranked,   # pass ranked list from permutation_importance()
)
```

### Plot feature importance

```python
from nn.estimator import plot_importance

plot_importance(
    ranked=ranked,
    target_col="has_refund",
    metric_name="AUC",
    val_metric=0.8436,
)
# Saves: importance_has_refund.png
```

### Use the evaluation metric suites directly

```python
from nn.evaluate import eval_binary, eval_regression, eval_classification

# Binary
metrics = eval_binary(preds, targets, threshold=0.5)
print(metrics["AUC"], metrics["F1"])

# Regression
metrics = eval_regression(preds, targets)
print(metrics["RMSE"], metrics["R²"])

# Classification (pass the target_encoder for readable class names)
metrics = eval_classification(preds, targets, target_encoder=te)
print(metrics["Accuracy"], metrics["Macro F1"])
print(metrics["Classification Report"])
```

---

## Quick reference card

```
TRAINING
  --target <col>            Column to predict (any of 76)
  --epochs N                Max epochs, default 50 (estimator) / 30 (evaluate)
  --batch_size N            Batch size, default 512
  --subset "col=val"        Filter rows before training
  --data_dir <path>         Override default data/ location

SAVING / LOADING
  --save_model <prefix>     Save .pt + .pkl to this prefix
  --load_model <prefix>     Load saved model for --predict

OUTPUT
  --list_targets            Print all 76 columns with non-null %
  --plot                    Save importance/evaluation chart PNG

PREDICTION
  --predict '{"k": v}'      JSON string or file path (requires --load_model)

LLM
  --recommend               Call OpenRouter LLM after predict (requires API key)
  --llm <model_id>          Override default model (deepseek → gemma → mistral)

EVALUATE.PY ONLY
  --all                     Run all 11 key targets in sequence
```
