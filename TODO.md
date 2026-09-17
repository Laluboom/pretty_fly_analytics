# TODO — Pretty Fly Universal Neural Estimator

Ranked, best first. Reviewed 2026-09-17.

> The full Phase 0–10 build log that used to live in this file is preserved in git
> history (`git show b7e9099:TODO.md`) and summarised in `SESSION_BRIEF.md` and
> `REPORT.md`. Everything in it was marked done; this file is now forward-looking only.

---

## 1. [BUG] `has_refund` is still leaking — via the two flags derived from `refund_reason`

`LEAKAGE_MAP["has_refund"]` (`nn/data_builder.py:452`) drops `financial_status`,
`refund_reason` and `refund_amount` — but **not** `size_issue` and
`damaged_in_transit`, which `_engineer_features` builds as pure functions of
`refund_reason` (`nn/data_builder.py:423-424`). `refund_reason` is `"none"` on every
non-refunded row, so `size_issue == 1` implies `has_refund == 1` with certainty.

The evidence is already sitting in `REPORT.md`:

- `REPORT.md:180` — top two features are `size_issue` (Δ=0.794) and `damaged_in_transit` (Δ=0.278)
- `REPORT.md:170` — recall **0.5461**; size/damage reasons are **53.8%** of all refunds (3,145/5,843)
- `REPORT.md:169` — precision **0.9936**, only 7 false positives across 11,987 negatives

The model is not predicting refunds. It is reading a downstream encoding of the
column that was dropped for leakage, and it recovers almost exactly the refunds
whose reason happens to leak. `REPORT.md:461` calls this AUC "leakage-free". It is not.

**Do:** add `"size_issue"` and `"damaged_in_transit"` to `LEAKAGE_MAP["has_refund"]`;
also cross-drop them from each other's entries (both are functions of the same
column, so each leaks into the other on the `has_refund==1` subset). Re-run
`python nn/evaluate.py --target has_refund --epochs 30` and write the honest AUC
into `REPORT.md` §3.1 and the summary table. An honest 0.6 beats a fake 0.84.

_~30 min, mostly the re-run._

---

## 2. [BUG] Refund labels are applied at order grain to a line-item table

The feature table's grain is the line item (69,956 rows). `_load_refunds` parses
`refund_line_items` into `refund_variant_ids` at `nn/data_builder.py:68` — the exact
data needed to label correctly — and then throws the column away on the next line
(`nn/data_builder.py:69`). The merge is on `order_id` (`nn/data_builder.py:356`), so
every line item in a partially-refunded order is labelled as refunded.

Verified against the CSVs:

- 5,843 refunds, each covering exactly **one** line item
- those orders contain **10,025** line items
- so **4,182 rows — 41.7% of the positive class — are mislabelled**

`REPORT.md:174` reports "Positive rate (actual) 14.33%" = 10,025/69,956. The true
line-item refund rate is 8.35%. The same inflation flows into `FILTER_MAP`
(`nn/data_builder.py:466-476`): `damaged_in_transit` and `size_issue` train on
"refund rows only (10,025)" (`REPORT.md:468`), 42% of which were never refunded but
carry the order's `refund_reason`.

**Do:** keep `refund_variant_ids` through the merge and set
`has_refund = variant_id in refund_variant_ids` in `_build_joined` (where
`variant_id` is still present — it isn't dropped until `_engineer_features:426`).
Blank `refund_reason`/`refund_amount` on rows that fail that test. Re-run and
update the positive-rate figures in `REPORT.md`.

_~45 min._

---

## 3. [BUG] Every reported metric comes from weights the run threw away

`nn/model.py:212` restores `best_weights`, but `all_preds` / `all_targets`
(`nn/model.py:174`, reassigned at the top of every epoch's validation block) still
hold the **last** epoch's predictions. The metric computed at `nn/model.py:215-226`
therefore describes the final-epoch model, not the restored best one. With
`patience=5` that is a model five epochs past its best checkpoint.

This contaminates everything downstream: `evaluate.py:378-391` feeds the same stale
`preds`/`targets` into `eval_binary`/`eval_regression`/`eval_classification`, so the
whole of `REPORT.md` §3 measures a discarded model — while `--save_model` saves the
good one. The printed number and the shipped artefact are different models.

Same function, second issue: if `val_loss` is NaN on epoch 1 (`inf - nan > min_delta`
is False), `best_weights` stays `None` and `model.load_state_dict(None)` crashes with
a bare `AttributeError` five epochs later.

**Do:** after `model.load_state_dict(best_weights)`, run one more no-grad pass over
`val_loader` to rebuild `all_preds`/`all_targets` before computing the metric. Guard
`best_weights is None` with a clear error.

_~30 min._

---

## 4. [QUICK WIN ~15min] Warn when an inference input hits an unseen category

`nn/estimator.py:261-264`: at `--predict` time, any categorical value not in the
fitted `LabelEncoder` silently falls back to index `0` — the alphabetically first
*real* class. Predict for a customer in a city the training set never saw and they
get quietly treated as whoever sorts first, with no output at all to say so. The
existing unknown-*key* warning at `estimator.py:233` covers the wrong failure mode.

Note `nn/model.py:40` already allocates `vocab_size + 1` embedding rows — there is a
reserved OOV slot that nothing ever uses.

**Do:** collect the unseen `(col, value)` pairs in that loop and print one
`Warning: unseen categorical values (mapped to fallback): ...` line before the
prediction. Cheap, and it makes a genuinely silent failure visible in a demo.

---

## 5. [TEST] One test file, aimed squarely at items 1 and 2

There are no tests. `validate.py` is the hackathon's dataset reconciler, not a test
of `nn/`. The single test that would have caught both leakage bugs above:

1. `build_feature_table()` → assert `df["has_refund"].sum() == 5843` (line-item grain,
   not 10,025)
2. for each target in `LEAKAGE_MAP`, run `prepare_for_target` and assert no remaining
   feature column is a deterministic function of the target — e.g. assert no feature
   column `c` has `df.groupby(c)[target].nunique().max() == 1` on a column with a
   non-trivial positive rate

Cache `build_feature_table()` in a module-scoped fixture so the suite runs once
(~20 s) rather than per test.

_~45 min._
