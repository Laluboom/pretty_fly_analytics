# Daily brief — pretty_fly_analytics (2026-11-10)

## What I looked at

All four modules in `nn/` end to end (`data_builder` 649 lines, `model` 229, `estimator`
538, `evaluate` 450), `REPORT.md`, `TODO.md`, `README_nn.md`, `HOW_TO_USE.md`, and the raw
data — `line_items`, `orders`, `customers`, `addresses`, `variants`, `email_events`,
`inventory_movements`, `purchase_orders`, `po_line_items`, `refunds`, `support_tickets`,
`support_messages.json`. No browser check — this is a CLI/ML project, nothing to serve.
The review box still has no `pip` and no working `ensurepip`, so I recounted everything with
the stdlib `csv`/`json` modules rather than guessing.

Since the last review (2026-10-19), one of the five proposed tasks landed — `f31404b`, the
four missing `LEAKAGE_MAP` keys. **The other four were never touched** and are carried
forward.

## The finding

Three consecutive reviews have now patched `LEAKAGE_MAP` by hand. This run I went looking
for why the same class of bug keeps recurring, and the answer is one line up the stack:
**`nn/model.py:124` splits rows, but the table's grain is the line item while every one of
the 11 `KEY_TARGETS` is an attribute of a coarser entity** — an order, a customer, or a
variant. Sibling rows from the same entity land on both sides of the split carrying an
identical feature vector *and* an identical label.

I measured it rather than asserting it. I rebuilt the exact order+customer-grain feature
vector a `has_refund` run sees *after* the `LEAKAGE_MAP` drops — money columns, utm,
discount, month/dow/hour, the six customer columns, city, postcode district, email
aggregates. **49,791 of 49,793 orders have a globally unique fingerprint.** The primary key
is fully reconstructable from surviving features. Simulating the same 80/20 split: 42.7% of
val rows (5,974/13,991) have a byte-identical train sibling, and that sibling's label is
correct in 5,974 of 5,974 cases. A pure lookup — no model, no training — scores
**accuracy 0.9466, precision 1.000, recall 0.617** against a naive baseline of 0.8606. The
published AUC is 0.8436.

`variant_return_rate` is worse. It is a per-variant constant on a table with 645 variants
and 69,956 rows, so **100%** of val rows have their variant in train, and the variant is
uniquely identified by `(variant_price, weight_grams, inventory_quantity, option1, option2)`
for all 645. A per-variant lookup gives **RMSE 0.000000, R² 1.000000**.

This matters most for what `REPORT.md` is currently promising. Three sections are flagged
"pending re-run" after the leakage fixes. **Those re-runs will not produce honest numbers** —
the lookup channel is still wide open, and no column drop can close it. Only a grouped split
can.

## The second finding

`python nn/evaluate.py --all` has never completed. `delivery_delay_days` is identically 0 on
every row — all 21 purchase orders have `actual_delivery == expected_delivery`, so all 645
variants get 0. It is `KEY_TARGETS[8]`, and `get_X_y` types it **binary** (not the
"regression" `REPORT.md:517` claims) because `{0.0}.issubset({0,1,True,False})` is satisfied
by a one-element set. So `--all` trains eight full models, then `roc_auc_score` raises
`Only one class present in y_true` and the run dies before reaching `variant_return_rate` or
`city`.

The proof was already in the repo: the summary table at `REPORT.md:505` has figures for the
first eight targets and `—` for exactly rows 9, 10 and 11. The command is documented in four
places and has never run to the end. Same root cause as the known `quantity` and
`n_escalations` crashes — this is just the first instance inside `KEY_TARGETS`.

## What I'm proposing

1. **[BUG]** Switch the split to `GroupShuffleSplit` with a per-target group key. Three
   touch points: keep `order_id`/`customer_id`/`variant_id` past
   `nn/data_builder.py:426-431`, exclude them from `cat_cols`/`num_cols`, and return the
   group key from `get_X_y` so it survives the `dropna` and `FILTER_MAP` filters. ~55 min.
2. **[QUICK WIN ~15min]** Raise a clear `ValueError` on `nunique() < 2` in
   `prepare_for_target`, and have the `--all` loop skip-and-continue. Smallest change that
   produces visible progress: `--all` reaches all 11 targets for the first time.
3. **[BUG]** Rebuild `all_preds`/`all_targets` after restoring `best_weights`
   (`nn/model.py:212` vs `215-226`). Should land with #1, before any re-run. ~30 min.
4. **[TEST]** The structural suite — now with a group guard as its first and most valuable
   rule (zero train/val entity overlap per target; fails today at 42.7% and 100%). ~45 min.
5. **[BUG]** Label `has_refund` at variant grain. Carried forward, still 41.7% of the
   positive class mislabelled. ~45 min.

## Read on health

The code is well-written and the data loading is careful — that has not changed. The problem
is that the *evaluation* has a structural flaw the code quality masks, and this run found its
root rather than another symptom. Stop patching `LEAKAGE_MAP`, fix the split, add the test,
withdraw the numbers. Do not build Phase 11.
