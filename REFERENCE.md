# REFERENCE — pretty_fly_analytics

## What this is

A universal tabular neural estimator over the **Pretty Fly** hackathon dataset
(Wayflyer × Fin AI, London, 3–5 June 2026). Point it at any column in a 76-column
joined feature table and it trains a PyTorch MLP to predict that column from all the
others, then reports permutation importance and — optionally — an LLM-written business
recommendation.

Built during the hackathon; last source commit 5 June 2026. Nothing has been committed
since except review notes, so treat it as a finished, working demo rather than an
actively evolving system.

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

`REPORT.md` presents metrics for 11 targets. Four verified defects mean several are
wrong, all in the optimistic direction:

- **Target leakage, twice.** `LEAKAGE_MAP` (`nn/data_builder.py:451`) is hand-maintained
  and has drifted. `has_refund` still sees `size_issue` / `damaged_in_transit`, both pure
  functions of the dropped `refund_reason` (`:423-424`) — the quoted AUC 0.8436 is
  inflated. `has_ticket` sees the three Phase 10 sentiment columns (`:377-380`), which are
  non-zero on 100% of positives and exactly 0 on every negative — a perfect leak.
- **Refund labels are order-grain on a line-item table** (`:69`, `:356`). 10,025 rows are
  marked refunded; only 5,843 line items were. 41.7% of positives are wrong.
- **Reported metrics use the final-epoch model, not the restored best checkpoint**
  (`nn/model.py:212` vs `215-226`), so printed scores describe a different model than
  `--save_model` writes to disk.
- **`--target quantity` crashes** after a full training run: the column is constant
  (all 69,956 rows are `1`), so it is typed binary-with-one-class and `roc_auc_score`
  raises (`nn/model.py:219`).

Details, evidence and fixes in `TODO.md`. `REPORT.md`, `SESSION_BRIEF.md` and
`README_nn.md` all pre-date this review and still quote the affected figures.

No test suite. Dependencies are unpinned, which matters because saved models pickle live
sklearn `LabelEncoder` / `StandardScaler` objects (`nn/estimator.py:189`).

`models/`, generated `*.png`, `.env`, `baby_guide.md` and `pretty_fly_master.csv` are
gitignored and not in the repo.
