# TODO — pretty_fly_analytics

Reviewed 2026-11-10. Ranked, best first. Previous review (2026-10-19) landed item 1 of 5
(`f31404b` — the four missing `LEAKAGE_MAP` keys); **items 2–5 were never touched** and are
carried forward below as items 2–5 again, with one new finding promoted above all of them.

No code was modified in this review. No ML deps on the review box (no `pip`, no working
`ensurepip`), so every number below was recounted from the raw CSV/JSON with the stdlib.

---

## 1. [BUG] The val split is row-level on a line-item table — so every published number includes an entity lookup that no `LEAKAGE_MAP` entry can remove

`nn/model.py:124` splits **rows**:

```python
train_idx, val_idx = train_test_split(idx, test_size=val_size, random_state=42, stratify=stratify)
```

But the table's grain is the line item (69,956 rows) while **every one of the 11
`KEY_TARGETS` is an attribute of a coarser entity** — an order, a customer or a variant.
Sibling rows from the same entity therefore land on both sides of the split carrying an
identical feature vector *and* an identical label. This is the channel that survives every
`LEAKAGE_MAP` fix, because it is not a column — it is the split.

**Measured this run (stdlib, seed 0, the same 80/20 proportion `train_model` uses):**

*Order grain — `has_refund`, `total_price`, `satisfaction_rating`, `resolved_by`, `resolution_time_minutes`:*

- 49,793 orders over 69,956 rows. Items per order: 35,060×1, 10,224×2, 3,588×3, 921×4 —
  **49.88% of rows sit in a multi-item order.**
- I rebuilt the feature vector a `has_refund` run actually sees at order+customer grain
  (the five money columns, `utm_source`/`utm_medium`/`utm_campaign`, `discount_code`,
  month/dow/hour, the six `customers.csv` columns, `city`, `postcode_district`, the four
  email aggregates) — with `financial_status`, `refund_reason`, `refund_amount`,
  `size_issue`, `damaged_in_transit` dropped per `LEAKAGE_MAP`. Result: **49,791 of 49,793
  orders have a globally unique fingerprint.** The primary key is fully reconstructable
  from surviving features.
- So **42.7% of val rows (5,974/13,991) have a byte-identical train sibling**, and that
  sibling's label matches in **5,974/5,974 = 100%** of cases.
- A pure lookup — no model, no training — scores **accuracy 0.9466, precision 1.000,
  recall 0.6171** on `has_refund`, against a naive all-zero baseline of 0.8606. 61.7% of
  val positives are directly readable from train. The published AUC is 0.8436
  (`REPORT.md:174`).

*Variant grain — `variant_return_rate`, `delivery_delay_days`:*

- 645 variants over 69,956 rows (median 85 rows/variant, min 2), so **100% of val rows
  have their variant in train**, and `(variant_price, weight_grams, inventory_quantity,
  option1_value, option2_value)` identifies all **645/645** uniquely.
- `variant_return_rate` is a per-variant constant. A per-variant train lookup gives
  **RMSE 0.000000, R² 1.000000** (naive-mean RMSE 0.042528). Any figure published for this
  `KEY_TARGET` (`nn/evaluate.py:360`) would be a memorisation score, not a prediction.

This reframes the three pending re-runs in `REPORT.md` §3.1/3.4–3.6: **re-running after the
`LEAKAGE_MAP` fix will not produce honest numbers**, because the lookup channel is still
open. Fix the split first or the re-run is wasted.

**Do:** thread a per-target group key through to the split.

1. `nn/data_builder.py:426-431` — stop dropping `order_id`, `customer_id`, `variant_id`;
   move them to a `GROUP_COLS` set.
2. `_encode_and_scale` (`:502`) — exclude `GROUP_COLS` from both `cat_cols` and `num_cols`;
   `get_X_y` (`:540`) — return the group column alongside `X_cat`/`X_num`/`y` so it is
   subset identically by the `dropna` at `:541` and the `FILTER_MAP` filter at `:633`.
3. Add `GROUP_MAP = {target: group_col}` — `order_id` for the five order-grain targets,
   `variant_id` for `product_type`/`variant_return_rate`/`delivery_delay_days`,
   `customer_id` for `city` — and swap `train_test_split` for
   `sklearn.model_selection.GroupShuffleSplit` in `train_model`, falling back to the
   current behaviour when a target has no group.

_~55 min of plumbing. If that is too much for one sitting, do the measurement half only:
add a `groups` argument to `train_model` and print the train/val entity overlap, so the
number is in front of whoever next quotes a metric._

---

## 2. [QUICK WIN ~15min] `nn/evaluate.py --all` has never completed — it dies on target 9 of 11

`delivery_delay_days` is **identically 0 on every row**. `data/purchase_orders.csv` has 21
POs and `actual_delivery == expected_delivery` on all 21 (verified: delay distribution is
`{0: 21}`, zero nulls), so all 645 `po_line_items` variants get 0, so all 69,956 rows get 0.

It is `KEY_TARGETS[8]` (`nn/evaluate.py:359`). `get_X_y` (`nn/data_builder.py:555`) types
it **binary** — `{0.0}.issubset({0, 1, True, False})` is satisfied by a one-element set —
not the "regression" `REPORT.md:517` claims. So `--all` builds the table, trains eight full
models, then `roc_auc_score` (`nn/model.py:219`) raises a bare
`ValueError: Only one class present in y_true` and the run dies before it reaches
`variant_return_rate` or `city`.

The corroboration is sitting in `REPORT.md` already: the summary table at `:505-519` has
figures for the first eight `KEY_TARGETS` and **`—` for exactly rows 9, 10 and 11**. The
command documented in four places (`REPORT.md:158`, `:495`, `README_nn.md:45`, `:127`,
`HOW_TO_USE.md:395-407`) has never run to the end.

This is the third instance of the same single-value bug, and the first one inside
`KEY_TARGETS`:

- **`quantity`** — all 69,956 line items are `1`. `_business_line`
  (`nn/estimator.py:122`) advertises it as a headline target; `--list_targets` shows it at
  100% non-null.
- **`n_escalations`** — identically 0 on all 1,204 tickets. The rule at
  `nn/data_builder.py:241-244` counts a `bot` message immediately followed by a `human`
  one, but the senders in `data/support_messages.json` are `customer` (3,108), `bot`
  (1,204) and **`agent`** (700) — no `human` sender exists (same stale name at `:231` and
  `:252`). Renaming to `agent` does not help: the transcripts have only two shapes,
  `customer|bot|customer|agent|customer` (700) and `customer|bot|customer` (504), so `bot`
  is never *adjacent* to `agent`. And redefining it as "an agent replied at all" is
  exactly `resolved_by == "human"`, i.e. a fresh perfect leak. Leave the feature dead;
  just stop it crashing.

**Do:** two edits.

1. `prepare_for_target` (`nn/data_builder.py:620-649`) — after the filter and leakage
   steps, raise a clear `ValueError` when `df_clean[target_col].nunique() < 2`, naming the
   column and its single value.
2. `nn/evaluate.py:428-430` — wrap the `--all` loop body in `try/except ValueError`,
   print `SKIPPED: <target> — <reason>`, and continue, so `--all` reaches all 11 targets
   and the summary table shows which three are unrunnable.

Also drop the `quantity` line from `_business_line` (`nn/estimator.py:122`). Verifiable
here with `python3 -m py_compile` plus the stdlib recount — no deps needed.

---

## 3. [BUG] Every reported metric comes from weights the run threw away

_Carried forward from 2026-10-19 — never touched. Unchanged in the source._

`nn/model.py:212` restores `best_weights`, but `all_preds` / `all_targets` are reassigned
at the top of every epoch's validation block (`nn/model.py:174`) and still hold the
**final** epoch's predictions. The metric computed at `nn/model.py:215-226` therefore
describes the final-epoch model, not the restored best one. With `patience=5` that is a
model five epochs past its best checkpoint.

This contaminates everything downstream: `nn/evaluate.py:376-391` feeds the same stale
`preds`/`targets` into `eval_binary` / `eval_regression` / `eval_classification`, so all of
`REPORT.md` §3 measures a discarded model — while `permutation_importance`
(`nn/estimator.py:514`) and `--save_model` (`nn/estimator.py:527`) both use the *restored*
one. The printed number, the importance ranking and the shipped artefact do not describe
the same weights.

Same function, second issue: if `val_loss` is NaN on epoch 1, `inf - nan > min_delta` is
False, so `best_weights` stays `None` and `model.load_state_dict(None)` crashes with a bare
`AttributeError` five epochs later.

**Do:** after `model.load_state_dict(best_weights)`, run one more no-grad pass over
`val_loader` to rebuild `all_preds`/`all_targets` before computing the metric. Guard
`best_weights is None` with a clear error.

_~30 min. Affects all 11 published numbers, so it should land before any re-run — alongside
item 1._

---

## 4. [TEST] The one test that would have caught items 1–3 and every past drift

_Carried forward from 2026-10-19 — never touched. There is still no `tests/` dir and no CI;
`validate.py` (1,123 lines) is the hackathon's vendor-CSV reconciler and asserts nothing
about `nn/`._

The last three bug-shaped commits (`b7e9099`, `43f7423`, `f31404b`) all fixed a
hand-maintained list by hand. A test that hardcodes today's leakage list would just restate
the bug. Write the structural one — it is target-agnostic, so it covers columns nobody has
thought about yet:

1. **Group guard** (item 1's regression test — the new one, and the most valuable):
   for every `KEY_TARGET`, compute its group key and assert the train and val index sets
   share **zero** groups. Fails today at 42.7% overlap on order-grain targets and 100% on
   variant-grain ones.
2. **Grain guard** (item 5's regression test): `build_feature_table()` → assert
   `df["has_refund"].sum() == 5850`, not 10,025.
3. **Leakage guard**: for every column in `KEY_TARGETS` *plus* every key in `LEAKAGE_MAP`,
   run `prepare_for_target` and fail if any surviving feature `c` is a near-deterministic
   function of the target — e.g. fail when `df.groupby(c)[target].nunique().max() == 1`
   and `c`'s non-modal values cover a non-trivial share of rows. Report the offending
   `(target, column)` pairs by name.
4. **Degeneracy guard**: assert every column in `num_cols` has `nunique() >= 2`. Catches
   `n_escalations` and `delivery_delay_days`, both of which currently occupy an input slot
   and a permutation-importance iteration while carrying zero information.

Cache `build_feature_table()` in a module-scoped fixture — it is a ~20 s build and every
test needs it.

_~45 min to write. Cannot be **executed** here (no pytest, no pandas); needs a run on a
machine with the deps._

---

## 5. [BUG] Refund labels are applied at order grain to a line-item table

_Carried forward from 2026-10-19 — never touched._

`_load_refunds` parses `refund_line_items` into `refund_variant_ids` at
`nn/data_builder.py:68` — the exact column needed to label correctly — and discards it on
the very next line (`:69`). The merge is on `order_id` (`:356`), so every line item in a
partially-refunded order is marked refunded. `_engineer_features` even lists
`refund_variant_ids` in `drop_cols` (`:429`), a defensive entry for a column that never
survives that far.

- 5,843 refunds, no order refunded twice, and **every** `refund_line_items` array has
  length exactly 1
- those 5,843 orders contain **10,025** line items (3,016 have 1, 1,714 have 2, 871 have 3,
  242 have 4)
- so **4,182 rows — 41.7% of the positive class — are mislabelled**

`REPORT.md:174` reports "Positive rate (actual) 14.33%" = 10,025/69,956. The true line-item
rate is **8.36%**. The inflation flows into `FILTER_MAP`: `damaged_in_transit` and
`size_issue` train on "refund rows only (10,025)", 42% of which were never refunded but
still carry the order's `refund_reason`.

The fix lands: **every** refund's variant is present in its order's line items (0 misses in
5,843). 5,836 match exactly one row and 7 match two (a repeated variant within one order),
so the correct positive count is **5,850**, not 5,843.

**Do:** keep `refund_variant_ids` through the merge and set
`has_refund = variant_id in refund_variant_ids` in `_build_joined`, where `variant_id` is
still present (it is not dropped until `:426-431`). Blank `refund_reason` / `refund_amount`
on rows that fail the test. Note in a comment that the 7 duplicate-variant orders
double-count `refund_amount`. Withdraw the positive-rate figures in `REPORT.md` §3.1.

_~45 min._

---

## Runners-up — verified, not in the top 5

- **`REPORT.md:190` still says "Interpretation: Leakage-safe"** directly under the
  `⚠️ Not leakage-free` banner added at `:164-172`, and still lists `size_issue` (Δ=0.794)
  and `damaged_in_transit` (Δ=0.278) as the top features — the two columns the banner says
  were wrongly present. A reader skimming to the interpretation gets the pre-fix story.
  One-paragraph delete.
- **`--predict` defaults missing numerics to raw `0.0`, not "no information."**
  `nn/estimator.py:270` builds the row with `row.get(col, 0.0)` and then scales it, so an
  unspecified feature lands well below the mean rather than at it. The documented example
  (`REFERENCE.md`, `HOW_TO_USE.md`) supplies 3 of ~56 numeric features. Fix: default to
  `scaler.mean_[i]`, which scales to exactly 0.
- **Unseen categoricals at `--predict` time fall back to index 0 silently**
  (`nn/estimator.py:261-264`) — the alphabetically first *real* class, with no warning.
  `nn/model.py:40` already allocates `vocab_size + 1` embedding rows; the reserved OOV slot
  is never used.
- **Encoders and the scaler are fit on the full table before the train/val split**
  (`nn/data_builder.py:516`, `:524` → `nn/model.py:124`), so val rows inform the scaling and
  the category vocabulary. Small next to item 1, but it is the same root cause — the split
  happens after everything that should respect it.
- **`make_plots`' classification branch can raise on a shape mismatch.** `n` comes from the
  classes present in `targets` (`nn/evaluate.py:78`), but `counts_pred` is
  `np.bincount(preds, minlength=n)` (`:321`), which is longer than `x` if the model ever
  predicts a class index absent from the val targets — `ax4.bar` then fails. Latent today,
  reachable on a sparse multi-class target.
- **Latent row-explosion in the join chain.** `_load_refunds` and `_load_po_line_items`
  call `drop_duplicates`, but the `support_tickets` (`:362`), `addresses` (`:351`) and
  `support_message_features` (`:371`) merges do not. Safe *today* — all five merge keys are
  unique — but one duplicate row in a refreshed CSV silently multiplies line items.
- **Dependencies are entirely unpinned** (`requirements_nn.txt`, 8 bare names; `torch` is
  CPU/CUDA-ambiguous) while `save_model` pickles live sklearn `LabelEncoder` /
  `StandardScaler` objects (`nn/estimator.py:189-190`). A minor sklearn bump silently
  invalidates every saved model. Can't be done here — pinning needs a resolver.
- **Stale comment:** `nn/data_builder.py:143` says "Multiple POs per variant — keep most
  recent." `data/po_line_items.csv` has 645 rows and 645 distinct variants — exactly one PO
  line each, so the `sort_values`/`drop_duplicates` is a no-op.
