# REFERENCE — pretty_fly_analytics

## What this is

A universal tabular neural estimator over the **Pretty Fly** hackathon dataset
(Wayflyer × Fin AI, London, 3–5 June 2026). Point it at any column in a 76-column
joined feature table and it trains a PyTorch MLP to predict that column from all the
others, then reports permutation importance and — optionally — an LLM-written business
recommendation.

Built during the hackathon; last code commit 5 June 2026. Nothing has been committed
since, so treat it as a finished, working demo rather than an actively evolving system.

## Layout

```
data/                 21 vendor CSVs + support_messages.json (~25 MB, committed)
validate.py           vendor reconciliation validator, 20 rules — not a test of nn/
nn/data_builder.py    21 files → one flat 69,956-row × 76-col table (626 lines)
nn/model.py           PrettyFlyNet: cat embeddings + 3-layer MLP; train loop (229 lines)
nn/estimator.py       CLI entry point; permutation importance; save/predict/LLM (538)
nn/evaluate.py        full metric suites + 5-panel charts; 11 key targets (450)
```

`data_builder` joins line_items ← orders ← variants ← products ← customers, plus
suppliers, inventory, email, addresses, discount codes, tickets, message stats and
VADER sentiment. Grain is the **line item**.

## Running it

```bash
pip install -r requirements_nn.txt          # torch, pandas, numpy, sklearn, tqdm,
                                            # matplotlib, vaderSentiment, openai
python nn/estimator.py --list_targets
python nn/estimator.py --target has_refund --epochs 30 --plot
python nn/estimator.py --target total_price --save_model models/total_price
python nn/estimator.py --load_model models/total_price \
       --predict '{"subtotal": 120, "total_shipping": 4.99, "total_tax": 24}'
python nn/evaluate.py --all --epochs 20     # 11 targets + summary table
```

Flags: `--subset col=value`, `--batch_size`, `--data_dir`, `--recommend` (needs
`OPENROUTER_API_KEY`; see `.env.example`), `--llm <openrouter-model>`.
Full flag-by-flag walkthrough in `HOW_TO_USE.md`.

## Current state — read this before trusting any number

`REPORT.md` presents metrics for 11 targets. Three verified defects mean several of
them are wrong, in the optimistic direction:

- **`has_refund` is not leakage-free.** `size_issue` and `damaged_in_transit` are
  derived from `refund_reason` (`nn/data_builder.py:423-424`) but are not dropped by
  `LEAKAGE_MAP` (`nn/data_builder.py:452`). The quoted AUC 0.8436 is inflated.
- **Refund labels are order-grain on a line-item table** (`nn/data_builder.py:69, 356`).
  10,025 rows are marked refunded; only 5,843 line items actually were.
- **Reported metrics use the final-epoch model, not the restored best checkpoint**
  (`nn/model.py:212` vs `215-226`), so printed scores don't match saved models.

Details, evidence and fixes in `TODO.md`. `REPORT.md`, `SESSION_BRIEF.md` and
`README_nn.md` all pre-date this review and still quote the affected figures.

No test suite. Dependencies are unpinned, which matters because saved models pickle
live sklearn `LabelEncoder`/`StandardScaler` objects (`nn/estimator.py:189`).

`models/`, generated `*.png`, `.env`, `baby_guide.md` and `pretty_fly_master.csv` are
gitignored and not in the repo.
