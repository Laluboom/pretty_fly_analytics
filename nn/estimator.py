import argparse
import sys
import os
import numpy as np
import torch

# Allow running from project root or from nn/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nn.data_builder import build_feature_table, prepare_for_target
from nn.model import train_model, PrettyFlyNet, get_loss_fn, PrettyFlyDataset
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# 3.1  CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pretty Fly Universal Neural Estimator — predict any variable from all others."
    )
    parser.add_argument("--target",      type=str,   default=None,  help="Column to predict")
    parser.add_argument("--epochs",      type=int,   default=50,    help="Max training epochs (default: 50)")
    parser.add_argument("--batch_size",  type=int,   default=512,   help="Batch size (default: 512)")
    parser.add_argument("--data_dir",    type=str,   default=None,  help="Path to data/ directory")
    parser.add_argument("--list_targets",action="store_true",       help="Print all valid target columns and exit")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 3.3  Permutation importance
# ---------------------------------------------------------------------------

def permutation_importance(model, X_cat, X_num, y, task_type, feature_meta, device, n_repeats=3):
    loss_fn = get_loss_fn(task_type)
    model.eval()

    all_feature_names = feature_meta["cat_cols"] + feature_meta["num_cols"]
    n_cat = len(feature_meta["cat_cols"])
    n_num = len(feature_meta["num_cols"])

    X_cat_t = torch.tensor(X_cat, dtype=torch.long).to(device)
    X_num_t = torch.tensor(X_num, dtype=torch.float32).to(device)
    if task_type == "classification":
        y_t = torch.tensor(y, dtype=torch.long).to(device)
    else:
        y_t = torch.tensor(y, dtype=torch.float32).to(device)

    # baseline loss
    with torch.no_grad():
        out = model(X_cat_t, X_num_t)
        if task_type in ("regression", "binary"):
            baseline = loss_fn(out.squeeze(), y_t).item()
        else:
            baseline = loss_fn(out, y_t).item()

    importances = []
    rng = np.random.default_rng(42)

    for i in range(len(all_feature_names)):
        delta_sum = 0.0
        for _ in range(n_repeats):
            if i < n_cat:
                # shuffle cat column i
                col_backup = X_cat_t[:, i].clone()
                perm = rng.permutation(len(y))
                X_cat_t[:, i] = X_cat_t[perm, i]
                with torch.no_grad():
                    out = model(X_cat_t, X_num_t)
                X_cat_t[:, i] = col_backup
            else:
                # shuffle num column (i - n_cat)
                j = i - n_cat
                col_backup = X_num_t[:, j].clone()
                perm = rng.permutation(len(y))
                X_num_t[:, j] = X_num_t[perm, j]
                with torch.no_grad():
                    out = model(X_cat_t, X_num_t)
                X_num_t[:, j] = col_backup

            if task_type in ("regression", "binary"):
                shuffled_loss = loss_fn(out.squeeze(), y_t).item()
            else:
                shuffled_loss = loss_fn(out, y_t).item()

            delta_sum += shuffled_loss - baseline

        importances.append(delta_sum / n_repeats)

    ranked = sorted(zip(all_feature_names, importances), key=lambda x: x[1], reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# 3.4  Output printer
# ---------------------------------------------------------------------------

def _business_line(target_col, top_features):
    top3 = ", ".join(f[0] for f in top_features[:3])
    lines = {
        "has_refund":          f"Top return drivers: {top3}",
        "satisfaction_rating": f"Satisfaction most influenced by: {top3}",
        "total_price":         f"Order value driven by: {top3}",
        "quantity":            f"Units per order driven by: {top3}",
        "resolved_by":         f"Ticket escalation predicted by: {top3}",
        "resolution_time_minutes": f"Resolution time driven by: {top3}",
    }
    return lines.get(target_col, f"Top predictors of {target_col}: {top3}")


def print_results(target_col, task_type, metric_name, val_metric,
                  ranked_importance, preds, targets, target_encoder):
    print()
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Target  : {target_col}")
    print(f"  Task    : {task_type}")
    print(f"  {metric_name:10s}: {val_metric:.4f}")
    print()

    print("  Top 10 Feature Importances (permutation):")
    print(f"  {'Rank':<5} {'Feature':<35} {'Delta Loss':>10}")
    print("  " + "-" * 52)
    for rank, (feat, score) in enumerate(ranked_importance[:10], 1):
        bar = "█" * min(int(score * 200), 20) if score > 0 else ""
        print(f"  {rank:<5} {feat:<35} {score:>10.5f}  {bar}")

    print()
    print("  5 Example Predictions vs Actuals (val set sample):")
    print(f"  {'#':<4} {'Predicted':>14} {'Actual':>14}")
    print("  " + "-" * 34)
    rng = np.random.default_rng(7)
    sample_idx = rng.choice(len(preds), size=min(5, len(preds)), replace=False)
    for i, idx in enumerate(sample_idx, 1):
        pred_val = preds[idx]
        true_val = targets[idx]
        if task_type == "classification" and target_encoder is not None:
            pred_label = target_encoder.inverse_transform([int(pred_val)])[0]
            true_label = target_encoder.inverse_transform([int(true_val)])[0]
            print(f"  {i:<4} {str(pred_label):>14} {str(true_label):>14}")
        elif task_type == "binary":
            print(f"  {i:<4} {pred_val:>14.3f} {true_val:>14.0f}")
        else:
            print(f"  {i:<4} {pred_val:>14.2f} {true_val:>14.2f}")

    print()
    print(f"  Business insight: {_business_line(target_col, ranked_importance)}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 3.2  Main orchestration
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Build feature table
    df = build_feature_table(args.data_dir)

    # --list_targets
    if args.list_targets or args.target is None:
        print("\nValid target columns:")
        for col in sorted(df.columns):
            nulls = df[col].isna().sum()
            pct   = 100 * (1 - nulls / len(df))
            print(f"  {col:<35}  {pct:5.1f}% non-null  ({len(df) - nulls:,} rows)")
        if args.target is None:
            print("\nRe-run with --target <column_name>")
            sys.exit(0)

    # Validate target
    if args.target not in df.columns:
        print(f"\nError: '{args.target}' not found in feature table.")
        print("Run with --list_targets to see valid options.")
        sys.exit(1)

    # Prepare data
    X_cat, X_num, y, task_type, n_classes, feature_meta, target_encoder = prepare_for_target(
        df, args.target
    )

    # Train
    model, val_loss, val_metric, metric_name, val_idx, preds, targets = train_model(
        X_cat, X_num, y,
        task_type, n_classes, feature_meta,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Permutation importance on val set
    device = next(model.parameters()).device
    print("\nComputing feature importances...")
    ranked = permutation_importance(
        model,
        X_cat[val_idx], X_num[val_idx], y[val_idx],
        task_type, feature_meta, device,
    )

    # Print results
    print_results(
        args.target, task_type, metric_name, val_metric,
        ranked, preds, targets, target_encoder,
    )


if __name__ == "__main__":
    main()
