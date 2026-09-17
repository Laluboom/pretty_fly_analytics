# Review brief — 2026-09-17

## What I looked at

All four files in `nn/` end to end (`data_builder.py`, `model.py`, `estimator.py`,
`evaluate.py` — 1,843 lines), plus `REPORT.md`, the old `TODO.md`, `requirements*.txt`
and `.gitignore`. Then I cross-checked the report's claims against the raw CSVs with
stdlib-only scripts (no pandas installed on this machine) — refund counts, refund
reason distribution, line-items-per-order, ticket/address/PO join cardinality.

Not a web project, so no browser check.

Worth noting up front: the last code commit is 5 June 2026, the end of the hackathon.
Three and a half months of nothing. This sits in `01_active_projects` but it is
functionally a finished demo. That's fine — just don't plan around it being in motion.

## What I found

**The headline number is wrong, and the report contains its own proof.**

`REPORT.md` leads with `has_refund` at AUC 0.8436, labelled "leakage-free (prev. fake
1.000)". The previous leakage fix dropped `financial_status`, `refund_reason` and
`refund_amount` (`nn/data_builder.py:452`). But `_engineer_features` creates
`size_issue` and `damaged_in_transit` as pure functions of `refund_reason`
(`nn/data_builder.py:423-424`), and those two were never added to `LEAKAGE_MAP`.
Since `refund_reason` is `"none"` on every non-refunded row, `size_issue == 1` implies
`has_refund == 1` with certainty.

The report's own numbers confirm it. Its top two features are `size_issue` (Δ=0.794)
and `damaged_in_transit` (Δ=0.278). Its precision is 0.9936 — seven false positives
across 11,987 negatives. And its recall is 0.5461, against the 53.8% of refunds
(3,145 of 5,843, counted directly from `data/refunds.csv`) whose reason is size- or
damage-related. The model recovers almost exactly the refunds whose reason leaks, and
essentially nothing else. There is no evidence here that it predicts refunds at all.

**Second, the labels themselves are at the wrong grain.** The feature table is one row
per line item (69,956). `_load_refunds` parses `refund_line_items` into
`refund_variant_ids` on `nn/data_builder.py:68` — exactly the data needed to label
correctly — and discards the column on the very next line, then merges on `order_id`
(`:356`). Every line item in a partially-refunded order is marked refunded. I counted
it: 5,843 refunds, each covering exactly one line item, in orders holding 10,025 line
items. 4,182 rows — 41.7% of the positive class — are false. `REPORT.md:174` reports
a 14.33% positive rate, which is 10,025/69,956; the true rate is 8.35%. The same
inflation propagates into the `damaged_in_transit` and `size_issue` training sets,
whose row count the report states as "10,025".

**Third, the metrics describe a model that was thrown away.** `nn/model.py:212`
restores the best checkpoint, but `all_preds`/`all_targets` still hold the final
epoch's predictions from `nn/model.py:174`, and the metric is computed from those at
`:215-226`. With `patience=5`, that is a model five epochs past its best.
`evaluate.py:378-391` consumes the same stale arrays, so all of `REPORT.md` §3 is
affected — while `--save_model` writes out the good weights. The printed score and the
shipped artefact are different models.

Smaller things: an unseen categorical at `--predict` time silently maps to encoder
index 0, the alphabetically first real class, with no warning at all
(`nn/estimator.py:261-264`) — and `nn/model.py:40` already reserves a `vocab_size + 1`
OOV embedding slot that nothing uses. Dependencies are entirely unpinned while saved
models pickle live sklearn encoder objects. There are no tests; `validate.py` is the
vendor's dataset reconciler, not a test of `nn/`.

## What I'm proposing

Five tasks, ranked. One and two are label-correctness fixes and should be done in that
order, since fixing the leak first gives you a clean baseline to measure the relabel
against. Three makes the reported numbers honest. The quick win is the unseen-category
warning — fifteen minutes, and it turns a completely silent failure into a visible one,
which matters if this ever gets demoed again. Five is the one test that would have
caught both one and two: assert the `has_refund` positive count is 5,843, and assert no
surviving feature is a deterministic function of the target.

The old `TODO.md` was a 44 KB phase-by-phase build log with every item already ticked —
no live tasks in it, so nothing was preserved. It's in git history at `b7e9099` and
summarised in `SESSION_BRIEF.md`.

The code itself is good: clear structure, honest comments, sensible early stopping and
scheduling, real thought given to leakage and zero-inflation. The problem is that the
leakage audit stopped one derivation short, and nothing verifies the conclusions.
