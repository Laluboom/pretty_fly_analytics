# Daily review — pretty_fly_analytics (2026-10-19)

## What I looked at

30 commits of history, the four `nn/` source files end to end (1,847 lines), `REPORT.md`
in full, and the previous two reviews' `TODO.md`. Then I recounted every data claim
against the CSVs directly — `line_items`, `refunds`, `support_tickets`,
`support_messages.json`, `products`, `variants`, `addresses`, `customers`, `orders`,
`po_line_items`. No browser check: this is a CLI/ML project, nothing to serve.

**I could not train anything.** `/usr/bin/python3` has no `pip` and no working
`ensurepip`, so `torch`/`pandas`/`sklearn` are unavailable. All verification below is
from reading code plus stdlib `csv`/`json` recounts. That constraint is itself a finding
— it's exactly why the last doer run (`43f7423`) landed a correct code fix but left the
`REPORT.md` figure flagged instead of corrected.

## Good news first

The 2026-09-28 top item actually got done and merged (PR #1). I verified it by reading
`LEAKAGE_MAP` rather than trusting the commit message: the three sentiment columns are in
`has_ticket`, and `size_issue`/`damaged_in_transit` are cross-dropped for `has_refund`
(`nn/data_builder.py:451-466`). Items 2, 3 and 5 from that list are untouched and I've
re-confirmed all three.

## What I found

**The hand-maintained-map problem has recurred a third time, and this time it's worse
than "a column was missed."** `LEAKAGE_MAP` is keyed by target and has 6 keys;
`KEY_TARGETS` publishes 11. So 7 published targets get no leakage handling whatsoever.
Three of them are one-column lookups:

- **`resolved_by`, Accuracy 1.0000** is `resolution_time_minutes` read back. Bot-resolved
  tickets take 5–29 minutes, human-resolved take 60–1439. Zero overlap — a single
  threshold at 29 gives 100%. `msg_count` and `customer_msg_count` are independent perfect
  separators too, since the transcripts have only two shapes.
- **`resolution_time_minutes`, RMSE 315, "33% better than naive 473"** is the same leak in
  reverse. Just predicting the group mean of `resolved_by` (bot 16.9 min, human 756.5)
  gives RMSE **304** — better than the model. The entire headline improvement is that one
  column.
- **`product_type`, Accuracy 1.0000** is `weight_grams`, which has exactly 6 distinct
  values across all 645 variants, one per product type: 120g→Cap, 280g→Tee,
  520g→Sweatpants, 650g→Hoodie, 850g→Trainer, 900g→Outerwear. A literal lookup table.
- **`city`** is `postcode_district` — all 40 districts map to exactly one of 11 cities.
  No number is published yet, but it's a documented `--all` target that will return 1.000.

What makes this more than a map-maintenance chore is that **`REPORT.md` explains two of
these perfect scores with the wrong mechanism.** §3.5 states "the channel a customer
contacts through almost perfectly determines who handles them." It doesn't:
`support_channel` → `resolved_by` is 58.14% by majority vote, which is *exactly* the naive
baseline (chat 251h/168b, email 356h/259b, instagram 93h/77b). `support_channel` tops the
permutation ranking only because it's `"none"` on 97.56% of rows — the ranking is measuring
"is this row a ticket at all," and the report reads it as business insight. §3.4 similarly
credits `supplier_country` and margin for `product_type` when the mechanism is weight.
§3.6 even *notices* that `resolved_by` and `resolution_time_minutes` are "mutually
predictive… each the strongest predictor of the other" and files it as a finding rather
than a leak.

I also re-verified the two carried bugs and found the previous review's proposed test
assertion was off. The refund grain bug is real — 4,182 rows (41.7% of positives)
mislabelled, true line-item rate 8.36% not the published 14.33%. But the correct positive
count is **5,850, not 5,843**: 7 orders contain the refunded variant twice. The
2026-09-28 TODO proposed asserting 5,843, which would have failed by 7. Good news on
feasibility though — every refund's variant does appear in its order's line items (0
misses in 5,843), so the fix joins cleanly.

Finally, a small one with a nice root cause. `n_escalations` is identically zero on all
69,956 rows: the rule looks for a `bot` message followed by a `human` one, but the senders
in the JSON are `customer`, `bot` and **`agent`** — there is no `human`. Renaming it won't
help either, because the only two conversation shapes are
`customer|bot|customer|agent|customer` and `customer|bot|customer`, so `bot` is never
adjacent to `agent`. And redefining escalation as "an agent replied at all" would give you
exactly `resolved_by == "human"` — a fresh perfect leak. Best left dead; it just shouldn't
crash.

## What I'm proposing

Ranked in `TODO.md`. The shape of it: **stop publishing the numbers, then add the test
that stops this recurring** — not build Phase 11.

1. **[BUG]** Add the four missing `LEAKAGE_MAP` keys plus `FILTER_MAP["resolved_by"]` and
   `["customer_msg_count"]`; withdraw the §3.4/§3.5/§3.6 figures the way §3.1 already was;
   rewrite the two wrong interpretation paragraphs. ~50 min, no training needed — which is
   the point, given the env.
2. **[TEST]** The structural leakage test. Not a hardcoded list of today's leaks, which
   would just restate the bug: a target-agnostic near-determinism assertion over every
   `KEY_TARGET`, plus the `has_refund.sum() == 5850` grain guard and a zero-variance guard
   on `num_cols`. It fails today on four targets, would have caught `43f7423`, and catches
   the next drift for free. This is the answer to "which single test would have helped."
3. **[QUICK WIN ~15min]** Guard `nunique() < 2` in `prepare_for_target` so `quantity` and
   `n_escalations` fail instantly rather than after a full training run.
4. **[BUG]** Label refunds at variant grain, target 5,850 positives.
5. **[BUG]** The reported metric comes from weights the run discarded (`nn/model.py:174`
   vs `:212`), so the printed score, the importance ranking and the saved artefact describe
   three different models. 30 minutes, and it affects all 11 numbers — it should land
   before any re-run.

## Verdict

Clean, genuinely well-built code whose published results are mostly artifacts. The
engineering isn't the problem; the write-up is. Two things need Visesh: a machine with the
deps so the re-runs can actually happen, and a call on whether `REPORT.md` is still a
document anyone relies on — if it is, items 1 and 5 are urgent; if it's a hackathon
souvenir, item 2 plus honest caveats is enough and this repo should drop out of the active
tier.
