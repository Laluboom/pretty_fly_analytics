# Daily review — pretty_fly_analytics — 2026-09-28

## What I looked at

`nn/data_builder.py`, `nn/model.py`, `nn/estimator.py` and `nn/evaluate.py` in full,
`REPORT.md` for the published figures, and the raw CSVs in `data/` to check the numbers
rather than accept them. Not a web project, so no browser check. None of the project's
dependencies (pandas, torch, sklearn, vaderSentiment, matplotlib, openai) are installed
in this environment, so I could not run the pipeline — every figure below was recounted
from the raw CSVs with the stdlib, and every code claim is anchored to a line.

First thing worth saying: **nothing from the 2026-09-17 review was implemented.** No
source file has changed since `b7e9099` on 5 June. I re-derived the three carried-over
findings independently rather than trusting the previous run, and all three hold — the
refund figures (5,843 / 10,025 / 41.7%) and the 53.8% size-and-damage share came out
identical on a fresh count.

## The main finding — the leakage map has drifted a second time

`LEAKAGE_MAP` (`nn/data_builder.py:451`) is the single thing keeping this project's
scores meaningful, and it is maintained entirely by hand. It has now been missed twice,
both times because a *new* feature turned out to be a function of an *existing* target.

The new one is `has_ticket`. The three Phase 10 sentiment columns — `avg_sentiment`,
`min_sentiment`, `pct_negative_msgs` — are merged on `order_id` and `fillna(0)` for
orders with no ticket (`nn/data_builder.py:377-380`). That is structurally identical to
`msg_count`, which *is* listed in `LEAKAGE_MAP["has_ticket"]`. The sentiment columns are
not. From the data: 1,204 tickets on 1,204 distinct orders → 1,705 positive line-item
rows, and **every one of those 1,204 tickets has at least one customer message**, so
every positive gets a real VADER score while all 68,251 negatives sit at exactly `0.0`.
`avg_sentiment != 0` implies `has_ticket == 1` across **100%** of the positive class —
a strictly more complete leak than the `has_refund` one found last time, which only
covered 53.8%.

`git show ec57f66` nails the mechanism: the Phase 10 commit added the three columns to
`_build_joined` **and** added three entries to `FILTER_MAP` — and did not touch
`LEAKAGE_MAP` at all. The author remembered one map and forgot the other. Mitigating
factor: `has_ticket` is not in `KEY_TARGETS`, so no published number is wrong. But
`--target has_ticket` is a supported call that will hand back a near-1.0 AUC meaning
nothing.

That is why item 4 is a test rather than more documentation. Patching the map a third
time fixes today's balance and nothing else; a structural assertion — for every
`LEAKAGE_MAP` key, fail if any surviving feature is a deterministic function of the
target — fails on both known leaks today and catches the Phase 11 version automatically.

## What else I'm proposing

Items 2 and 3 are carried over unchanged because they are still real and still the
highest-value things after the leakage. Refund labels are applied at order grain to a
line-item table (`nn/data_builder.py:69`, `:356`) — the code parses exactly the column
it needs and discards it one line later, leaving 4,182 rows, 41.7% of positives,
mislabelled. And every metric in `REPORT.md` §3 describes the final-epoch model rather
than the restored best checkpoint (`nn/model.py:212` vs `215-226`), so the printed
number and the artefact `--save_model` writes are different models.

The quick win is new and small: `--target quantity` trains a complete model and *then*
crashes. `quantity` is constant — all 69,956 line items are `1` — so `get_X_y`
(`:532`) types it as binary-with-one-class via the `issubset({0,1,True,False})` branch,
and `roc_auc_score` (`nn/model.py:219`) dies with a bare sklearn `ValueError`. It is not
an obscure column: `_business_line` (`nn/estimator.py:122`) advertises it as a headline
target. A `nunique() < 2` guard in `prepare_for_target` turns a wasted training run into
an instant message.

Two verified runners-up I left out of the top five and recorded at the bottom of
`TODO.md`: `--predict` defaults missing numerics to raw `0.0` before scaling
(`nn/estimator.py:270`), which puts them 1.3–1.9 standard deviations from the mean
instead of at it — the documented example supplies 3 of ~56 numeric features, so ~53
arrive out of distribution; and the encoders and scaler are fit on the full table before
the train/val split.

## Verdict

The engineering is good — clean module boundaries, thoughtful `FILTER_MAP`/`LEAKAGE_MAP`
design, honest baselines in the metric suites. The problem is that the guardrail is
manual, and it has now failed twice in the same way. Fix the two leaks, fix the grain,
then write the test that makes the third drift impossible. Until then `REPORT.md` should
be read as a demo artefact, not a result.
