# TODO — Pretty Fly Universal Neural Estimator

Ranked, best first. Reviewed 2026-10-19.

> **Item 1 of the 2026-09-28 list is done** (`43f7423`, merged in `#1`): the three sentiment
> columns are now in `LEAKAGE_MAP["has_ticket"]` and `size_issue` / `damaged_in_transit`
> are cross-dropped for `has_refund` (`nn/data_builder.py:451-466`). Verified by reading
> the map, not by trusting the commit message.
>
> Items 2, 3 and 5 from that list are **untouched and re-confirmed** — they are items 4, 5
> and 3 below. Every figure in this file was recounted from the CSVs this run with the
> stdlib `csv`/`json` modules.
>
> **Environment blocker, read before picking anything up.** There is no `pip` and no
> working `ensurepip` on this box, so `torch`/`pandas`/`sklearn` cannot be installed and
> **nothing here can be trained or re-measured in place**. That is exactly why `43f7423`
> landed a correct code fix with the REPORT figure left flagged rather than corrected. Every
> task below is therefore scoped so the deliverable is *code + an explicitly withdrawn
> figure*, never "write the new number." The re-runs need Visesh's machine.

---

## 1. `[DONE 2026-10-19]` Three more key targets are one-column lookups — and REPORT's stated mechanism for two of them is factually wrong

`LEAKAGE_MAP` (`nn/data_builder.py:451-466`) is keyed by target and has **6 keys**.
`KEY_TARGETS` (`nn/evaluate.py:348-362`) has **11 entries**, and the feature table offers
~76 columns as targets. So the map protects 4 of the 11 published targets; the other 7 get
no leakage handling at all. Three of them are badly affected, and in two cases `REPORT.md`
confidently explains the perfect score with the wrong mechanism.

**1a — `resolved_by` (Accuracy 1.0000, `REPORT.md:274`) is `resolution_time_minutes` read back.**

Counted from `data/support_tickets.csv` this run:

- bot-resolved tickets: `resolution_time_minutes` ∈ [5, 29], n=504, mean 16.9
- human-resolved tickets: ∈ [60, 1439], n=700, mean 756.5
- **zero overlap.** A single threshold at 29 separates bot from human with 100% accuracy.

On the unfiltered table a third class appears (`none` = no ticket → `resolution_time_minutes`
is `fillna(0.0)` at `nn/data_builder.py:366`), so one column gives a perfect **three-way**
split: `0` / `5–29` / `60+`. `msg_count` does the same job independently — the message
transcripts have only two shapes, so `msg_count` ∈ {0, 3, 5} maps 1:1 onto
{none, bot, human}. So does `customer_msg_count` ∈ {0, 2, 3}.

`REPORT.md:290` says *"`support_channel` now leads — the channel a customer contacts
through almost perfectly determines who handles them."* **This is wrong.** Cross-tabbed
this run: chat 251 human / 168 bot, email 356/259, instagram_dm 93/77. Majority-vote
accuracy of `support_channel` → `resolved_by` is **0.5814 — exactly the naive baseline**
(700/1204). Channel carries no handler signal. `support_channel` tops the permutation
ranking only because it is `"none"` on 97.56% of rows, so shuffling it destroys the
model's ability to spot the majority class. The ranking is measuring *has_ticket
detection*, and §3.5 reads it as a business insight.

`resolved_by` is also the one ticket-derived column missing from `FILTER_MAP`
(`nn/data_builder.py:469-480`), which is why it is scored against a 97.56% "none"
baseline and why "+2.4 pp" looks modest instead of alarming.

**1b — `resolution_time_minutes` (RMSE 315, "33% better than naive 473", `REPORT.md:310-312`) is the same leak in reverse.**

There is no `LEAKAGE_MAP["resolution_time_minutes"]` key, so `resolved_by` survives as a
feature. `REPORT.md:310` already shows it dominating at Δ=136,908 and `REPORT.md:328`
notes the two are "mutually predictive… each is the strongest predictor of the other" —
describing the leak as a finding. Measured this run: predicting just the **group mean of
`resolved_by`** (bot 16.9, human 756.5) gives RMSE **304.1** against a naive-mean RMSE of
475.0. The model's 315 is *worse than the one-column lookup*. The entire "33% better than
naive" is that column, and nothing else.

**1c — `product_type` (Accuracy 1.0000, `REPORT.md:245`) is `weight_grams`.**

`weight_grams` has exactly **6 distinct values across all 645 variants, each mapping to
exactly one of the 6 product types**: 120g→Cap, 280g→Tee, 520g→Sweatpants, 650g→Hoodie,
850g→Trainer, 900g→Outerwear. It is a literal lookup table. `REPORT.md:264` instead
credits `supplier_country` and `gross_margin_est` — "supplier origin and margin perfectly
separate all 6 product types." They are co-leaks, not the mechanism: every top-ranked
feature there (`supplier_country`, `gross_margin_est`, `variant_price`, `price`,
`lead_time_days`, `weight_grams`) is a product- or variant-level attribute broadcast onto
line items. The model is joining, not predicting. The business question as posed
("can we identify product category from all other signals?") is trivially yes.

**1d — `city` is `postcode_district`, and the summary row describes the wrong dataset.**

`postcode_district` is derived from the postcode at `nn/data_builder.py:204` and kept as a
feature; `city` comes from the same row of `data/addresses.csv`. All **40 districts map to
exactly one city** — majority-vote accuracy 1.0000 against a 75.0% London baseline. No
`LEAKAGE_MAP["city"]` key exists. `REPORT.md:480` publishes no number (`—`), so nothing
false is in print yet, but `city` is a documented `--all` target that will return 1.000.
That row also says *"20+ UK cities"*: there are **11** cities and only London is in the UK
(Madrid, New York, Chicago, Brooklyn, Los Angeles, Amsterdam, Dublin, Paris, Berlin, Rome).
`default_country` is a weaker partial leak — 80.6% majority-vote vs the 75.0% baseline.

**Do:** add `FILTER_MAP["resolved_by"] = ("has_ticket", 1)` and
`FILTER_MAP["customer_msg_count"] = ("has_ticket", 1)` (the latter is missing while
`msg_count` is present — same drift). Add four `LEAKAGE_MAP` keys:
`resolved_by` → `[resolution_time_minutes, msg_count, customer_msg_count, satisfaction_rating, n_escalations, response_time_first_seconds]`;
`resolution_time_minutes` → `[resolved_by, msg_count, customer_msg_count]`;
`product_type` → `[weight_grams, variant_price, price, gross_margin_est, supplier_country, lead_time_days, collection]`;
`city` → `[postcode_district]`.
Then **withdraw** the §3.4, §3.5, §3.6 figures and the three summary-table rows
(`REPORT.md:473-480`) the way §3.1 was already withdrawn — strike the number, state the
mechanism, mark pending re-run. Replace the three wrong interpretation paragraphs
(`:264`, `:290`, and the `:328` "mutually predictive" bullet) with what the data actually
says. Fix "20+ UK cities" → "11 cities, 1 UK".

_~50 min, all of it editing — no training required, which is the point._

---

## 2. [TEST] The one test that would have caught 1a–1d, `43f7423`, and the next drift automatically

There are no tests, no `tests/` directory and no CI. `validate.py` (1,123 lines) is the
hackathon's vendor-dataset reconciler — 20 rules over the raw CSVs — and asserts nothing
about `nn/`.

The most recent bug-shaped commit is `43f7423`, which fixed two `LEAKAGE_MAP` omissions by
hand. A test that hardcodes today's leakage list would just restate the bug. Write the
structural one instead — it is target-agnostic, so it covers columns nobody has thought
about yet:

1. **Grain guard** (item 4's regression test): `build_feature_table()` →
   assert `df["has_refund"].sum() == 5850`, not 10,025.
2. **Leakage guard**: for every column in `KEY_TARGETS` *plus* every key in `LEAKAGE_MAP`,
   run `prepare_for_target` and fail if any surviving feature `c` is a near-deterministic
   function of the target — e.g. fail when `df.groupby(c)[target].nunique().max() == 1`
   and `c`'s non-modal values cover a non-trivial share of rows. Report the offending
   `(target, column)` pairs by name.
3. **Degeneracy guard**: assert every column in `num_cols` has `nunique() >= 2`, so a
   zero-variance feature like `n_escalations` (item 3) is caught at build time rather than
   occupying an input slot silently.

Rule 2 fails **today** on `resolved_by`, `resolution_time_minutes`, `product_type` and
`city`, and would have failed on `has_ticket`/`has_refund` before `43f7423`. That is the
actual fix; item 1 only pays off the current balance.

Cache `build_feature_table()` in a module-scoped fixture — it is a ~20 s build and every
test needs it.

_~45 min to write. Note it cannot be **executed** here (no pytest, no pandas); it will
need a run on a machine with the deps._

---

## 3. [QUICK WIN ~15min] Constant columns train a full model, then crash on the last line

Two offerable targets are single-valued, and `get_X_y` (`nn/data_builder.py:536`) types
them as binary because `set(df[target].unique()).issubset({0, 1, True, False})` is
satisfied by a one-element set. Training runs to completion before `roc_auc_score`
(`nn/model.py:219`) dies with a bare `ValueError: Only one class present in y_true`.

- **`quantity`** — all **69,956** line items are `1` (verified against
  `data/line_items.csv`). Not obscure: `_business_line` (`nn/estimator.py:122`) advertises
  it as a headline target ("Units per order driven by: …") and `--list_targets` shows it at
  100% non-null.
- **`n_escalations`** — identically **0** on all 1,204 tickets, so `0` on all 69,956 rows.
  It is a `FILTER_MAP` target (`nn/data_builder.py:475`), so `--target n_escalations`
  filters to 1,705 rows of zeros and then crashes the same way.

`n_escalations` is dead for a reason worth knowing before anyone "fixes" it. The escalation
rule at `nn/data_builder.py:240-244` counts a `bot` message immediately followed by a
`human` one. The senders in `data/support_messages.json` are `customer` (3,108), `bot`
(1,204) and **`agent`** (700) — there is no `human` sender, so the predicate never matches
(the same stale name sits at `:231` and `:252`). Renaming it to `agent` does **not** help:
the transcripts have only two shapes — `customer|bot|customer|agent|customer` (700) and
`customer|bot|customer` (504) — so `bot` is never *adjacent* to `agent` and the count stays
0. And the signal you would get from redefining it as "an agent replied at all" is
precisely `resolved_by == "human"`, i.e. a fresh perfect leak (item 1a). Leave the feature
dead; just stop it crashing.

**Do:** in `prepare_for_target` (`nn/data_builder.py:601-630`), after the filter and
leakage steps, raise a clear `ValueError` when `df_clean[target_col].nunique() < 2`, naming
the column and its single value. Drop the `quantity` line from `_business_line`
(`nn/estimator.py:122`). Verifiable here with `python3 -m py_compile` plus the stdlib CSV
recount — no deps needed.

---

## 4. [BUG] Refund labels are applied at order grain to a line-item table

The feature table's grain is the line item. `_load_refunds` parses `refund_line_items` into
`refund_variant_ids` at `nn/data_builder.py:68` — the exact column needed to label
correctly — and discards it on the very next line (`:69`). The merge is on `order_id`
(`:356`), so every line item in a partially-refunded order is marked refunded.
`_engineer_features` even lists `refund_variant_ids` in `drop_cols` (`:429`), a defensive
entry for a column that never survives that far.

Recounted this run:

- 5,843 refunds, no order refunded twice, and **every** `refund_line_items` array has
  length exactly 1
- those 5,843 orders contain **10,025** line items (3,016 have 1, 1,714 have 2, 871 have 3,
  242 have 4)
- so **4,182 rows — 41.7% of the positive class — are mislabelled**

`REPORT.md:174` reports "Positive rate (actual) 14.33%" = 10,025/69,956. The true
line-item rate is **8.36%**. The inflation flows into `FILTER_MAP`: `damaged_in_transit`
and `size_issue` train on "refund rows only (10,025)", 42% of which were never refunded
but still carry the order's `refund_reason`.

The fix is viable — I checked the join actually lands: **every** refund's variant is
present in its order's line items (0 misses in 5,843). 5,836 match exactly one row and 7
match two (a repeated variant within one order), so the correct positive count is
**5,850**, *not* 5,843. The 2026-09-28 list proposed asserting 5,843; that would have
failed by 7.

**Do:** keep `refund_variant_ids` through the merge and set
`has_refund = variant_id in refund_variant_ids` in `_build_joined`, where `variant_id` is
still present (it is not dropped until `:426-431`). Blank `refund_reason` / `refund_amount`
on rows that fail the test. Note in a comment that the 7 duplicate-variant orders
double-count `refund_amount`. Withdraw the positive-rate figures in `REPORT.md` §3.1.

_~45 min._

---

## 5. [BUG] Every reported metric comes from weights the run threw away

`nn/model.py:212` restores `best_weights`, but `all_preds` / `all_targets` are reassigned at
the top of every epoch's validation block (`nn/model.py:174`) and still hold the **final**
epoch's predictions. The metric computed at `nn/model.py:215-226` therefore describes the
final-epoch model, not the restored best one. With `patience=5` that is a model five epochs
past its best checkpoint.

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

_~30 min. Affects all 11 published numbers, so it should land before any re-run._

---

## Runners-up — verified this run, not in the top 5

- **`--predict` defaults missing numerics to raw `0.0`, not "no information."**
  `nn/estimator.py:270` builds the row with `row.get(col, 0.0)` and then scales it, so an
  unspecified feature lands well below the mean rather than at it. The documented example
  (`REFERENCE.md`, `HOW_TO_USE.md`) supplies 3 of ~56 numeric features. Fix: default to
  `scaler.mean_[i]`, which scales to exactly 0. (z-scores quoted on 2026-09-28 could not be
  re-measured this run — no pandas — but the code path is unchanged.)
- **Unseen categoricals at `--predict` time fall back to index 0 silently**
  (`nn/estimator.py:261-264`) — the alphabetically first *real* class, with no warning.
  `nn/model.py:40` already allocates `vocab_size + 1` embedding rows; the reserved OOV slot
  is never used.
- **Encoders and the scaler are fit on the full table before the train/val split**
  (`nn/data_builder.py:504-505` → `nn/model.py:124`), so val rows inform the scaling and the
  category vocabulary.
- **`make_plots`' classification branch can raise on a shape mismatch.** `n` comes from the
  classes present in `targets` (`nn/evaluate.py:78`), but `counts_pred` is
  `np.bincount(preds, minlength=n)` (`:321`), which is longer than `x` if the model ever
  predicts a class index absent from the val targets — `ax4.bar` then fails. Latent today,
  reachable on a sparse multi-class target.
- **Latent row-explosion in the join chain.** `_load_refunds` and `_load_po_line_items`
  call `drop_duplicates`, but the `support_tickets` (`:362`), `addresses` (`:351`) and
  `support_message_features` (`:371`) merges do not. Safe *today* — I checked all five merge
  keys and every one is unique — but one duplicate row in a refreshed CSV silently
  multiplies line items.
- **Dependencies are entirely unpinned** (`requirements_nn.txt`, 8 bare names; `torch` is
  CPU/CUDA-ambiguous) while `save_model` pickles live sklearn `LabelEncoder` /
  `StandardScaler` objects (`nn/estimator.py:189-190`). A minor sklearn bump silently
  invalidates every saved model. Can't be done here — pinning needs a resolver, and there
  is no pip.
- **Stale comment:** `nn/data_builder.py:143` says "Multiple POs per variant — keep most
  recent." `data/po_line_items.csv` has 645 rows and 645 distinct variants — exactly one PO
  line each, and the `sort_values`/`drop_duplicates` is a no-op.
