# Pretty Fly — Universal Neural Estimator

A single neural network that predicts **any variable** in the Pretty Fly dataset from all other variables. Pick a target, train, and get a ranked list of what drives it — no custom model per question needed.

---

## Quick start

```bash
pip install -r requirements_nn.txt
```

Run from the project root (`pretty_fly_data_pack/`).

---

## Usage

### List all valid prediction targets

```bash
python nn/estimator.py --list_targets
```

Prints all 51 columns with their non-null percentage and row count:

```
  has_refund                           100.0% non-null  (69,956 rows)
  satisfaction_rating                    0.8% non-null  (539 rows)
  total_price                          100.0% non-null  (69,956 rows)
  product_type                         100.0% non-null  (69,956 rows)
  ...
```

---

### Example 1 — What drives returns?

```bash
python nn/estimator.py --target has_refund --epochs 30 --plot --save_model models/has_refund
```

**Sample output:**
```
============================================================
  RESULTS
============================================================
  Target  : has_refund
  Task    : binary
  val_AUC   : 1.0000

  Top 10 Feature Importances (permutation):
  Rank  Feature                             Delta Loss
  ----------------------------------------------------
  1     financial_status                       1.56130  ████████████████████
  2     refund_reason                          0.32038  ████████████████████
  3     refund_amount                          0.00001
  ...

  5 Example Predictions vs Actuals (val set sample):
  #         Predicted         Actual
  ----------------------------------
  1             0.000              0
  2             1.000              1
  ...

  Business insight: Top return drivers: financial_status, refund_reason, refund_amount
============================================================

Importance chart saved: importance_has_refund.png
Model saved: models/has_refund.pt  +  models/has_refund.pkl
```

---

### Example 2 — What predicts customer satisfaction?

```bash
python nn/estimator.py --target satisfaction_rating --epochs 30 --batch_size 64 --plot
```

**Sample output:**
```
  Target  : satisfaction_rating
  Task    : regression
  val_RMSE  : 1.1466

  Top 10 Feature Importances:
  1     acquisition_source    ...
  2     support_channel       ...
  3     refund_reason         ...

  Business insight: Satisfaction most influenced by: acquisition_source, support_channel, refund_reason
```

> Note: only 539 rows have a `satisfaction_rating` (support tickets with ratings). The model trains on this subset automatically.

---

### Example 3 — Classify product type from all other signals

```bash
python nn/estimator.py --target product_type --epochs 20 --plot --save_model models/product_type
```

**Sample output:**
```
  Target  : product_type
  Task    : classification
  val_accuracy: 0.9999

  5 Example Predictions vs Actuals:
  #         Predicted         Actual
  ----------------------------------
  1               Tee            Tee
  2         Outerwear      Outerwear
  3               Cap            Cap
```

Predictions are decoded back to readable labels (Tee / Hoodie / Cap / Trainer / Sweatpants / Outerwear).

---

### Example 4 — Save a trained model, then predict from a JSON row

**Train and save:**
```bash
python nn/estimator.py --target product_type --epochs 20 --save_model models/product_type
```

**Predict on a new row:**
```bash
python nn/estimator.py \
  --predict '{"price": 85.0, "weight_grams": 320, "option1_value": "M", "collection": "Core"}' \
  --load_model models/product_type
```

**Output:**
```
Predicting: product_type
Input features used: 14 categorical + 35 numeric
Prediction : Hoodie  (confidence=0.9231)
```

Missing fields default to `0` (numeric) or `"unknown"` (categorical) — you only need to pass the fields you know.

---

## All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--target` | required | Column name to predict |
| `--epochs` | 50 | Max training epochs (early stopping applies) |
| `--batch_size` | 512 | Training batch size |
| `--data_dir` | `../data` | Path to the data/ directory |
| `--list_targets` | — | Print all valid target columns and exit |
| `--save_model` | — | Save `{prefix}.pt` + `{prefix}.pkl` after training |
| `--load_model` | — | Path prefix of saved model (used with `--predict`) |
| `--predict` | — | JSON string or path to JSON file; requires `--load_model` |
| `--plot` | — | Save `importance_{target}.png` bar chart |

---

## Architecture

```
Input
  ├── 15 categorical columns → nn.Embedding(vocab+1, dim=8) each
  └── 35 numeric columns → StandardScaled float32

Concatenation → BatchNorm1d
  → Linear(input_dim, 256) → ReLU → Dropout(0.3)
  → Linear(256, 128)       → ReLU → Dropout(0.2)
  → Linear(128, 64)        → ReLU
  → Output head (auto-selected by target type):
      binary         → Linear(64,1) + Sigmoid   — BCELoss   → reports AUC
      regression     → Linear(64,1)              — MSELoss   → reports RMSE
      classification → Linear(64, n_classes)     — CrossEntropy → reports accuracy
```

Training uses Adam (lr=1e-3), early stopping patience=5, 80/20 val split.
Feature importance uses permutation (shuffle each feature 3× on val set, measure loss increase).

---

## File structure

```
pretty_fly_data_pack/
├── nn/
│   ├── data_builder.py   — loads + joins 10 tables → 69,956-row feature matrix
│   ├── model.py          — PrettyFlyNet, training loop, early stopping
│   └── estimator.py      — CLI entry point
├── requirements_nn.txt
└── README_nn.md          — this file
```
