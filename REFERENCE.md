# REFERENCE — pretty_fly_analytics

## What this is

A universal tabular neural estimator over the **Pretty Fly** hackathon dataset
(Wayflyer × Fin AI, London, 3–5 June 2026). Point it at any column in a ~76-column joined
feature table and it trains a PyTorch MLP to predict that column from all the others, then
reports permutation importance and — optionally — an LLM-written business recommendation.
Last functional source commit 5 June 2026; since then only review notes and two leakage
fixes (`43f7423`, `f31404b`). A finished demo, not an evolving system.

## Layout

```
data/                 21 vendor CSVs + support_messages.json (~25 MB, committed)
validate.py           vendor reconciliation validator, 20 rules — not a test of nn/
nn/data_builder.py    21 files → one flat 69,956-row table (649 lines)
nn/model.py           PrettyFlyNet: cat embeddings + 3-layer MLP; train loop (229)
nn/estimator.py       CLI entry point; permutation importance; save/predict/LLM (538)
nn/evaluate.py        full metric suites + 5-panel charts; 11 key targets (450)
```

`data_builder` joins line_items ← orders ← variants ← products ← customers, plus suppliers,
inventory, email, addresses, discount codes, tickets, message stats and VADER sentiment.
Grain is the **line item** (49,793 orders → 645 variants → 69,956 rows). That grain
mismatch is the project's central defect — see below.

## Running it

```bash
pip install -r requirements_nn.txt    # torch, pandas, numpy, sklearn, tqdm,
                                      # matplotlib, vaderSentiment, openai
python nn/estimator.py --list_targets
python nn/estimator.py --target has_refund --epochs 30 --plot
python nn/estimator.py --target total_price --save_model models/total_price
python nn/estimator.py --load_model models/total_price \
       --predict '{"subtotal": 120, "total_shipping": 4.99, "total_tax": 24}'
python nn/evaluate.py --all --epochs 20     # crashes at target 9/11 — see below
```

Flags: `--subset col=value`, `--batch_size`, `--data_dir`, `--recommend` (needs
`OPENROUTER_API_KEY`; see `.env.example`), `--llm <openrouter-model>`. Full walkthrough in
`HOW_TO_USE.md`. **The review box has no `pip` and no working `ensurepip`**, so nothing here
can be trained or re-measured there — re-runs need a machine with the deps.

## Current state — do not trust any published number

`REPORT.md` covers 11 targets. Six verified defects mean most are wrong, all optimistic.

- **The val split is row-level on a line-item table** (`nn/model.py:124`), but all 11
  targets are order-, customer- or variant-level attributes. 49,791 of 49,793 orders have a
  globally unique fingerprint in the *surviving* feature columns, so 42.7% of val rows have
  a byte-identical train sibling with the identical label. A pure lookup — no model — gets
  accuracy 0.9466 / precision 1.000 on `has_refund` vs a 0.8606 baseline. `variant_return_rate`
  is worse: 100% variant overlap, lookup R² = 1.000. No `LEAKAGE_MAP` entry can close this;
  only a `GroupShuffleSplit` can. **The three pending re-runs in `REPORT.md` are therefore
  premature.**
- **`nn/evaluate.py --all` has never finished.** `delivery_delay_days` is identically 0
  (all 21 POs have `actual_delivery == expected_delivery`), gets typed binary by
  `get_X_y` (`nn/data_builder.py:555`), trains fully, then `roc_auc_score`
  (`nn/model.py:219`) raises. The summary table at `REPORT.md:505` shows `—` for exactly
  targets 9, 10 and 11. Same bug kills `quantity` (all `1`) and `n_escalations`
  (identically 0 — the rule looks for a `human` sender that does not exist).
- **Column leakage on 4 of the 11 targets** — `LEAKAGE_MAP` (`nn/data_builder.py:451`) is
  hand-maintained and keyed by target, and has drifted three times. `f31404b` added the
  four missing keys; `REPORT.md:264` and `:290` still explained two perfect scores via the
  wrong mechanism before that fix.
- **Refund labels are order-grain on a line-item table** (`:69`, `:356`): 10,025 rows
  marked refunded, only 5,850 line items were — 41.7% of positives wrong.
- **Metrics come from the final epoch, not the restored best checkpoint**
  (`nn/model.py:212` vs `215-226`), so printed scores, the importance ranking and the
  `--save_model` artefact describe different weights.

Evidence and fixes in `TODO.md`. `REPORT.md`, `SESSION_BRIEF.md` and `README_nn.md` all
pre-date this review and still quote the affected figures.

No test suite, no CI. Dependencies unpinned, which matters because saved models pickle live
sklearn `LabelEncoder` / `StandardScaler` objects (`nn/estimator.py:189`). `models/`,
generated `*.png`, `.env`, `baby_guide.md` and `pretty_fly_master.csv` are gitignored.
