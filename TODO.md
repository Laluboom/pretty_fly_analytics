# TODO — Pretty Fly Universal Neural Estimator

Ranked, best first. Reviewed 2026-09-28.

> Nothing from the 2026-09-17 review was implemented — no source file has changed
> since `b7e9099` (5 June). Items 1–3 below are re-confirmed, not re-guessed: every
> figure was recounted from the CSVs this run.
>
> The full Phase 0–10 build log is in git history (`git show b7e9099:TODO.md`) and
> summarised in `SESSION_BRIEF.md` / `REPORT.md`.

---

## 1. `[DONE 2026-09-28]` `LEAKAGE_MAP` is a hand-maintained list and it has now drifted twice

`LEAKAGE_MAP` (`nn/data_builder.py:451-462`) is the only thing standing between this
project and meaningless scores. It is updated by hand, and it has been missed twice —
both times when a *new* feature turned out to be a function of an *existing* target.

**1a — `has_ticket` leaks perfectly through the Phase 10 sentiment columns (new this review).**

`avg_sentiment`, `min_sentiment` and `pct_negative_msgs` are merged on `order_id` and
`fillna(0)` for orders with no ticket (`nn/data_builder.py:377-380`) — structurally
identical to `msg_count`, which *is* dropped. They are not in
`LEAKAGE_MAP["has_ticket"]` (`nn/data_builder.py:453-456`).

Counted from the data this run:

- 1,204 tickets, each on a distinct order → 1,705 line-item rows carry `has_ticket=1`
- **all 1,204 tickets have ≥1 customer message**, so all 1,705 positive rows get a real
  VADER score and every one of the 68,251 negatives is exactly `0.0`
- `avg_sentiment != 0` ⟹ `has_ticket == 1`, covering **100%** of the positive class

Git shows the mechanism exactly: `ec57f66` (Phase 10) added the three columns to
`_build_joined` **and** added three entries to `FILTER_MAP` — and touched `LEAKAGE_MAP`
not at all. The author remembered one map and forgot the other.

`has_ticket` is not in `KEY_TARGETS` (`nn/evaluate.py:348-362`), so no published number
is wrong — but `--target has_ticket` is a documented, supported call that will return a
near-1.0 AUC that means nothing.

**1b — `has_refund` still leaks through `size_issue` and `damaged_in_transit` (carried from 2026-09-17, untouched).**

Both are pure functions of `refund_reason` (`nn/data_builder.py:423-424`), which *is*
dropped; they are not. `refund_reason` is `"none"` on every non-refunded row, so
`size_issue == 1` implies `has_refund == 1` with certainty. Re-counted from
`data/refunds.csv`: size + damage reasons are 3,145/5,843 = **53.8%** of refunds, which
is almost exactly the reported recall of 0.5461 (`REPORT.md:170`) — and `size_issue`
(Δ=0.794) and `damaged_in_transit` (Δ=0.278) are the top two features at `REPORT.md:180`.
The model is not predicting refunds; it is reading back the column that was dropped.
`REPORT.md:461` calls this AUC "leakage-free". It is not.

**Do:** add the three sentiment columns to `LEAKAGE_MAP["has_ticket"]`; add
`size_issue` and `damaged_in_transit` to `LEAKAGE_MAP["has_refund"]`, and cross-drop
them from each other's entries (both are functions of the same column). Re-run
`python nn/evaluate.py --target has_refund --epochs 30` and write the honest AUC into
`REPORT.md` §3.1 and the summary table. An honest 0.6 beats a fake 0.84.

_~45 min, mostly the re-run. Item 4 is what stops this happening a third time._

---

## 2. [BUG] Refund labels are applied at order grain to a line-item table

The feature table's grain is the line item (69,956 rows). `_load_refunds` parses
`refund_line_items` into `refund_variant_ids` at `nn/data_builder.py:68` — the exact
column needed to label correctly — and throws it away on the very next line
(`nn/data_builder.py:69`). The merge is on `order_id` (`nn/data_builder.py:356`), so
every line item in a partially-refunded order is marked refunded.

Recounted from the CSVs this run:

- 5,843 refunds, no order refunded twice, and **every** `refund_line_items` array has
  length exactly 1
- those 5,843 orders contain **10,025** line items
- so **4,182 rows — 41.7% of the positive class — are mislabelled**

`REPORT.md:174` reports "Positive rate (actual) 14.33%" = 10,025/69,956. The true
line-item refund rate is **8.35%**. The same inflation flows into `FILTER_MAP`
(`nn/data_builder.py:466-476`): `damaged_in_transit` and `size_issue` train on
"refund rows only (10,025)" (`REPORT.md:468`), 42% of which were never refunded but
still carry the order's `refund_reason`.

**Do:** keep `refund_variant_ids` through the merge and set
`has_refund = variant_id in refund_variant_ids` in `_build_joined`, where `variant_id`
is still present (it is not dropped until `_engineer_features`, `nn/data_builder.py:427`).
Blank `refund_reason` / `refund_amount` on rows that fail that test. Re-run and correct
the positive-rate figures in `REPORT.md`.

_~45 min._

---

## 3. [BUG] Every reported metric comes from weights the run threw away

`nn/model.py:212` restores `best_weights`, but `all_preds` / `all_targets` are
reassigned at the top of every epoch's validation block (`nn/model.py:174`) and still
hold the **final** epoch's predictions. The metric computed at `nn/model.py:215-226`
therefore describes the final-epoch model, not the restored best one. With `patience=5`
that is a model five epochs past its best checkpoint.

This contaminates everything downstream: `nn/evaluate.py:378-391` feeds the same stale
`preds`/`targets` into `eval_binary` / `eval_regression` / `eval_classification`, so all
of `REPORT.md` §3 measures a discarded model — while `--save_model`
(`nn/estimator.py:527`) persists the good one. The printed number and the shipped
artefact are different models.

Same function, second issue: if `val_loss` is NaN on epoch 1, `inf - nan > min_delta` is
False, so `best_weights` stays `None` and `model.load_state_dict(None)` crashes with a
bare `AttributeError` five epochs later.

**Do:** after `model.load_state_dict(best_weights)`, run one more no-grad pass over
`val_loader` to rebuild `all_preds`/`all_targets` before computing the metric. Guard
`best_weights is None` with a clear error.

_~30 min._

---

## 4. [TEST] The one test that would have caught items 1a and 1b — and the next one automatically

There are no tests. `validate.py` is the hackathon's vendor-dataset reconciler (20 rules
over the raw CSVs), not a test of anything in `nn/`.

Don't write a test that hardcodes today's leakage list — that just restates the bug.
Write the structural one:

1. `build_feature_table()` → assert `df["has_refund"].sum() == 5843` (line-item grain,
   not 10,025) — this is item 2's regression guard
2. for **every** key in `LEAKAGE_MAP`, plus `has_ticket`, run `prepare_for_target` and
   assert that no surviving feature column `c` is a deterministic function of the
   target — e.g. fail if `df.groupby(c)[target].nunique().max() == 1` for a `c` whose
   non-modal value covers a non-trivial share of rows

Rule 2 fails today on `has_ticket` (three sentiment columns) and on `has_refund`
(`size_issue`, `damaged_in_transit`), and it will fail the next time someone adds a
Phase-11 feature derived from an existing target without updating the map. That is the
actual fix; item 1 is just paying off the current balance.

Cache `build_feature_table()` in a module-scoped fixture so the suite builds the table
once (~20 s) rather than per test.

_~45 min._

---

## 5. [QUICK WIN ~15min] `--target quantity` trains a full model, then crashes

`quantity` is constant: all 69,956 line items have `quantity == 1` (verified against
`data/line_items.csv` this run). `get_X_y` (`nn/data_builder.py:532`) tests
`set(df[target].unique()).issubset({0, 1, True, False})` — `{1}` satisfies that — so it
is classified as a **binary** task with one class. Training then runs to completion
against all-ones before `roc_auc_score` (`nn/model.py:219`) dies with a bare
`ValueError: Only one class present in y_true`.

`quantity` is not an obscure column: `_business_line` (`nn/estimator.py:122`) advertises
it as a headline target ("Units per order driven by: ..."), and it appears in
`--list_targets` at 100% non-null. It is also a dead feature for every *other* target —
a zero-variance column occupying a slot in `num_cols`.

**Do:** in `prepare_for_target` (`nn/data_builder.py:597`), after the filter/leakage
steps, raise a clear `ValueError` when `df_clean[target_col].nunique() < 2`, naming the
column and its single value. Drop the `quantity` line from `_business_line` while you
are there. Cheap, and it turns a 2-minute wasted training run into an instant message.

---

## Runners-up — verified, not in the top 5

- **`--predict` defaults missing numerics to raw `0.0`, not "no information."**
  `nn/estimator.py:270` builds the row with `row.get(col, 0.0)` and then scales it, so an
  unspecified feature lands well below the mean rather than at it. Measured at line-item
  grain: `price` → z = −1.55, `weight_grams` → −1.90, `orders_count` → −1.43,
  `total_spent` → −1.33. The documented example (`REFERENCE.md`, `HOW_TO_USE.md`)
  supplies 3 of ~56 numeric features, so ~53 of them arrive as out-of-distribution
  values. Fix: default to `scaler.mean_[i]`, which scales to exactly 0.
- **Unseen categoricals at `--predict` time fall back to index 0 silently**
  (`nn/estimator.py:261-264`) — the alphabetically first *real* class, with no warning.
  `nn/model.py:40` already allocates `vocab_size + 1` embedding rows; the reserved OOV
  slot is never used.
- **Encoders and the scaler are fit on the full table before the train/val split**
  (`nn/data_builder.py:620` → `nn/model.py:124`), so val rows inform the scaling and the
  category vocabulary. Mild next to items 1–2, but it is the same family of problem.
- **Dependencies are entirely unpinned** (`requirements_nn.txt`, 8 bare names) while
  `save_model` pickles live sklearn `LabelEncoder` / `StandardScaler` objects
  (`nn/estimator.py:189-190`). A minor sklearn bump silently invalidates every saved model.
