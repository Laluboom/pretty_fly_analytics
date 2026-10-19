# REFERENCE — pretty_fly_analytics

## What this is

A universal tabular neural estimator over the **Pretty Fly** hackathon dataset
(Wayflyer × Fin AI, London, 3–5 June 2026). Point it at any column in a ~76-column joined
feature table and it trains a PyTorch MLP to predict that column from all the others, then
reports permutation importance and — optionally — an LLM-written business recommendation.
Last functional source commit 5 June 2026; since then only review notes and one leakage
fix (`43f7423`). A finished demo, not an evolving system.

## Layout

```
data/                 21 vendor CSVs + support_messages.json (~25 MB, committed)
validate.py           vendor reconciliation validator, 20 rules — not a test of nn/
nn/data_builder.py    21 files → one flat 69,956-row table (630 lines)
nn/model.py           PrettyFlyNet: cat embeddings + 3-layer MLP; train loop (229)
nn/estimator.py       CLI entry point; permutation importance; save/predict/LLM (538)
nn/evaluate.py        full metric suites + 5-panel charts; 11 key targets (450)
```

`data_builder` joins line_items ← orders ← variants ← products ← customers, plus suppliers,
inventory, email, addresses, discount codes, tickets, message stats and VADER sentiment.
Grain is the **line item** (49,793 orders → 69,956 rows).

## Running it

```bash
pip install -r requirements_nn.txt    # torch, pandas, numpy, sklearn, tqdm,
                                      # matplotlib, vaderSentiment, openai
python nn/estimator.py --list_targets
python nn/estimator.py --target has_refund --epochs 30 --plot
python nn/estimator.py --target total_price --save_model models/total_price
python nn/estimator.py --load_model models/total_price \
       --predict '{"subtotal": 120, "total_shipping": 4.99, "total_tax": 24}'
python nn/evaluate.py --all --epochs 20     # 11 targets + summary table
```

Flags: `--subset col=value`, `--batch_size`, `--data_dir`, `--recommend` (needs
`OPENROUTER_API_KEY`; see `.env.example`), `--llm <openrouter-model>`. Full walkthrough in
`HOW_TO_USE.md`. **The review box has no `pip` and no working `ensurepip`**, so nothing
here can be trained or re-measured there — re-runs need a machine with the deps.

## Current state — do not trust any published number

`REPORT.md` covers 11 targets. Five verified defects mean most are wrong, all optimistic.

- **Leakage on 4 of the 11 key targets.** `LEAKAGE_MAP` (`nn/data_builder.py:451`) is
  hand-maintained, keyed by target, and has 6 keys — so 7 published targets get no
  protection. `resolved_by` (acc 1.000) is `resolution_time_minutes` read back (bot 5–29
  min, human 60–1439, zero overlap). `product_type` (1.000) is `weight_grams` — 6 values,
  one per type. `city` is `postcode_district` — 40 districts, each → 1 of 11 cities.
  `resolution_time_minutes` (RMSE 315) is beaten by the `resolved_by` group mean (304).
  `REPORT.md:264` and `:290` both explain their perfect score with the wrong mechanism —
  `support_channel` → `resolved_by` is 0.5814, exactly the naive baseline.
- **Refund labels are order-grain on a line-item table** (`:69`, `:356`): 10,025 rows
  marked refunded, only 5,850 line items were — 41.7% of positives wrong.
- **Metrics come from the final epoch, not the restored best checkpoint**
  (`nn/model.py:212` vs `215-226`), so printed scores, the importance ranking and the
  `--save_model` artefact describe different weights.
- **Constant targets crash after a full training run.** `quantity` (all rows `1`) and
  `n_escalations` (identically 0 — the rule looks for a `human` sender that doesn't exist)
  are typed binary-with-one-class and `roc_auc_score` raises (`nn/model.py:219`).

Evidence and fixes in `TODO.md`. `REPORT.md`, `SESSION_BRIEF.md` and `README_nn.md` all
pre-date this review and still quote the affected figures.

No test suite, no CI. Dependencies unpinned, which matters because saved models pickle live
sklearn `LabelEncoder` / `StandardScaler` objects (`nn/estimator.py:189`). `models/`,
generated `*.png`, `.env`, `baby_guide.md` and `pretty_fly_master.csv` are gitignored.
