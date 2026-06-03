"""
evaluate.py — Comprehensive evaluation for any trained PrettyFlyNet target.

Usage:
    python nn/evaluate.py --target has_refund --epochs 30
    python nn/evaluate.py --target total_price --epochs 30 --plot
    python nn/evaluate.py --all --epochs 20        # run all 6 key targets
"""

import argparse
import os
import sys
import numpy as np
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nn.data_builder import build_feature_table, prepare_for_target
from nn.model import train_model
from nn.estimator import permutation_importance

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score,
)


# ---------------------------------------------------------------------------
# Metric suites
# ---------------------------------------------------------------------------

def eval_binary(preds, targets, threshold=0.5):
    pred_labels = (preds >= threshold).astype(int)
    pos_rate    = targets.mean()
    return {
        "AUC":              roc_auc_score(targets, preds),
        "Average Precision":average_precision_score(targets, preds),
        "Accuracy":         accuracy_score(targets, pred_labels),
        "Precision":        precision_score(targets, pred_labels, zero_division=0),
        "Recall":           recall_score(targets, pred_labels, zero_division=0),
        "F1":               f1_score(targets, pred_labels, zero_division=0),
        "Positive Rate":    pos_rate,
        "Pred Positive Rate": pred_labels.mean(),
        "Confusion Matrix": confusion_matrix(targets.astype(int), pred_labels),
    }


def eval_regression(preds, targets):
    mae   = mean_absolute_error(targets, preds)
    rmse  = float(np.sqrt(mean_squared_error(targets, preds)))
    r2    = r2_score(targets, preds)
    range_ = targets.max() - targets.min()
    within_10pct = np.mean(np.abs(preds - targets) <= 0.1 * np.abs(targets).clip(1)) * 100
    within_20pct = np.mean(np.abs(preds - targets) <= 0.2 * np.abs(targets).clip(1)) * 100
    residuals    = preds - targets
    return {
        "RMSE":              rmse,
        "MAE":               mae,
        "R²":                r2,
        "Median Abs Error":  float(np.median(np.abs(preds - targets))),
        "Max Error":         float(np.abs(preds - targets).max()),
        "Within ±10%":       within_10pct,
        "Within ±20%":       within_20pct,
        "Residual Mean":     float(residuals.mean()),
        "Residual Std":      float(residuals.std()),
        "Target Mean":       float(targets.mean()),
        "Target Std":        float(targets.std()),
        "Naive RMSE (mean)": float(targets.std()),
    }


def eval_classification(preds, targets, target_encoder=None):
    preds_int   = preds.astype(int)
    targets_int = targets.astype(int)
    classes     = sorted(np.unique(targets_int))
    labels      = (target_encoder.inverse_transform(classes)
                   if target_encoder else [str(c) for c in classes])
    acc          = accuracy_score(targets_int, preds_int)
    macro_f1     = f1_score(targets_int, preds_int, average="macro", zero_division=0)
    weighted_f1  = f1_score(targets_int, preds_int, average="weighted", zero_division=0)
    report       = classification_report(
        targets_int, preds_int,
        labels=classes, target_names=labels,
        zero_division=0,
    )
    cm = confusion_matrix(targets_int, preds_int, labels=classes)
    n_classes    = len(classes)
    naive_acc    = np.bincount(targets_int).max() / len(targets_int)
    return {
        "Accuracy":        acc,
        "Macro F1":        macro_f1,
        "Weighted F1":     weighted_f1,
        "N Classes":       n_classes,
        "Naive Accuracy":  naive_acc,
        "Classification Report": report,
        "Confusion Matrix": cm,
        "Class Labels":    labels,
    }


# ---------------------------------------------------------------------------
# Printers
# ---------------------------------------------------------------------------

W = 62

def _bar(val, max_val, width=28):
    if max_val == 0:
        return ""
    filled = int(width * min(val / max_val, 1.0))
    return "█" * filled + "░" * (width - filled)


def print_binary_report(target_col, metrics, ranked):
    cm = metrics.pop("Confusion Matrix")
    print(f"\n{'='*W}")
    print(f"  BINARY CLASSIFICATION REPORT  |  target: {target_col}")
    print(f"{'='*W}")
    print(f"  {'Metric':<28} {'Value':>10}")
    print(f"  {'-'*40}")
    for k, v in metrics.items():
        print(f"  {k:<28} {v:>10.4f}")
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"            Pred 0   Pred 1")
    print(f"  Actual 0  {cm[0,0]:>6}   {cm[0,1]:>6}")
    print(f"  Actual 1  {cm[1,0]:>6}   {cm[1,1]:>6}")
    _print_importance(ranked)


def print_regression_report(target_col, metrics, ranked):
    print(f"\n{'='*W}")
    print(f"  REGRESSION REPORT  |  target: {target_col}")
    print(f"{'='*W}")
    print(f"  {'Metric':<28} {'Value':>10}")
    print(f"  {'-'*40}")
    skip = {"Target Mean", "Target Std", "Naive RMSE (mean)"}
    for k, v in metrics.items():
        if k in skip:
            continue
        unit = "%" if "%" in k else ""
        print(f"  {k:<28} {v:>10.4f}{unit}")
    print(f"\n  Baseline comparison:")
    print(f"  {'Naive RMSE (predict mean)':<28} {metrics['Naive RMSE (mean)']:>10.4f}")
    print(f"  {'Model RMSE':<28} {metrics['RMSE']:>10.4f}")
    improvement = (1 - metrics['RMSE'] / metrics['Naive RMSE (mean)']) * 100
    print(f"  {'Improvement over naive':<28} {improvement:>10.1f}%")
    _print_importance(ranked)


def print_classification_report(target_col, metrics, ranked):
    report = metrics.pop("Classification Report")
    cm     = metrics.pop("Confusion Matrix")
    labels = metrics.pop("Class Labels")
    print(f"\n{'='*W}")
    print(f"  CLASSIFICATION REPORT  |  target: {target_col}")
    print(f"{'='*W}")
    print(f"  {'Metric':<28} {'Value':>10}")
    print(f"  {'-'*40}")
    skip = {"N Classes", "Naive Accuracy"}
    for k, v in metrics.items():
        if k in skip:
            continue
        print(f"  {k:<28} {v:>10.4f}")
    print(f"\n  Baseline (majority class):  {metrics['Naive Accuracy']:.4f}")
    print(f"  Improvement:                {(metrics['Accuracy'] - metrics['Naive Accuracy'])*100:+.1f}pp")
    print(f"\n  Per-class breakdown:")
    print(report)
    print(f"  Confusion Matrix:")
    header = "         " + "".join(f"{l[:8]:>10}" for l in labels)
    print(f"  {header}")
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>10}" for v in row)
        print(f"  {labels[i][:8]:<8} {row_str}")
    _print_importance(ranked)


def _print_importance(ranked, top_n=10):
    print(f"\n  Top {top_n} Feature Importances (permutation, Δ loss):")
    print(f"  {'Rank':<5} {'Feature':<32} {'Score':>8}  {'Bar'}")
    print(f"  {'-'*60}")
    max_score = max(s for _, s in ranked[:top_n]) if ranked else 1
    for rank, (feat, score) in enumerate(ranked[:top_n], 1):
        bar = _bar(score, max_score)
        print(f"  {rank:<5} {feat:<32} {score:>8.5f}  {bar}")
    print(f"{'='*W}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(target_col, task_type, preds, targets, ranked, metrics, target_encoder=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Evaluation Report — target: {target_col}", fontsize=15, fontweight="bold", y=0.98)
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1: Feature importance (always)
    ax1 = fig.add_subplot(gs[0, :2])
    top15 = ranked[:15]
    names  = [f for f, _ in reversed(top15)]
    scores = [s for _, s in reversed(top15)]
    ax1.barh(names, scores, color="#2d6a9f")
    ax1.set_xlabel("Δ Loss (permutation importance)")
    ax1.set_title("Feature Importances")
    ax1.spines[["top","right"]].set_visible(False)

    if task_type == "binary":
        # Panel 2: ROC curve
        from sklearn.metrics import roc_curve
        ax2 = fig.add_subplot(gs[0, 2])
        fpr, tpr, _ = roc_curve(targets, preds)
        ax2.plot(fpr, tpr, color="#2d6a9f", lw=2)
        ax2.plot([0,1],[0,1],"--", color="grey", lw=1)
        ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
        ax2.set_title(f"ROC Curve  (AUC={metrics['AUC']:.4f})")
        ax2.spines[["top","right"]].set_visible(False)

        # Panel 3: Score distribution — use fixed [0,1] bin edges to handle perfect models
        ax3 = fig.add_subplot(gs[1, 0])
        bin_edges = np.linspace(0.0, 1.0, 42)
        ax3.hist(preds[targets==0], bins=bin_edges, alpha=0.6, label="Actual 0", color="#e07b39")
        ax3.hist(preds[targets==1], bins=bin_edges, alpha=0.6, label="Actual 1", color="#2d6a9f")
        ax3.axvline(0.5, color="red", linestyle="--", lw=1, label="threshold")
        ax3.set_xlabel("Predicted Probability"); ax3.set_title("Score Distribution")
        ax3.legend(fontsize=8); ax3.spines[["top","right"]].set_visible(False)

        # Panel 4: Confusion matrix heatmap
        ax4 = fig.add_subplot(gs[1, 1])
        cm = metrics["Confusion Matrix"]
        im = ax4.imshow(cm, cmap="Blues")
        ax4.set_xticks([0,1]); ax4.set_yticks([0,1])
        ax4.set_xticklabels(["Pred 0","Pred 1"]); ax4.set_yticklabels(["Actual 0","Actual 1"])
        for i in range(2):
            for j in range(2):
                ax4.text(j, i, str(cm[i,j]), ha="center", va="center",
                         color="white" if cm[i,j] > cm.max()/2 else "black")
        ax4.set_title("Confusion Matrix")

        # Panel 5: Precision-Recall curve
        from sklearn.metrics import precision_recall_curve
        ax5 = fig.add_subplot(gs[1, 2])
        prec, rec, _ = precision_recall_curve(targets, preds)
        ap = metrics["Average Precision"]
        ax5.plot(rec, prec, color="#2d6a9f", lw=2)
        ax5.set_xlabel("Recall"); ax5.set_ylabel("Precision")
        ax5.set_title(f"Precision-Recall  (AP={ap:.4f})")
        ax5.spines[["top","right"]].set_visible(False)

    elif task_type == "regression":
        # Panel 2: Predicted vs Actual scatter
        ax2 = fig.add_subplot(gs[0, 2])
        lim = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
        ax2.scatter(targets, preds, alpha=0.15, s=6, color="#2d6a9f")
        ax2.plot(lim, lim, "r--", lw=1)
        ax2.set_xlabel("Actual"); ax2.set_ylabel("Predicted")
        ax2.set_title(f"Predicted vs Actual  (R²={metrics['R²']:.4f})")
        ax2.spines[["top","right"]].set_visible(False)

        # Panel 3: Residual distribution
        ax3 = fig.add_subplot(gs[1, 0])
        residuals = preds - targets
        ax3.hist(residuals, bins=60, color="#2d6a9f", edgecolor="white", linewidth=0.3)
        ax3.axvline(0, color="red", linestyle="--", lw=1)
        ax3.set_xlabel("Residual (Predicted − Actual)"); ax3.set_title("Residual Distribution")
        ax3.spines[["top","right"]].set_visible(False)

        # Panel 4: Residuals vs Actual
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.scatter(targets, residuals, alpha=0.15, s=6, color="#e07b39")
        ax4.axhline(0, color="red", linestyle="--", lw=1)
        ax4.set_xlabel("Actual"); ax4.set_ylabel("Residual")
        ax4.set_title("Residuals vs Actual")
        ax4.spines[["top","right"]].set_visible(False)

        # Panel 5: Error distribution (|residual|)
        ax5 = fig.add_subplot(gs[1, 2])
        abs_err = np.abs(residuals)
        ax5.hist(abs_err, bins=60, color="#5a9e6f", edgecolor="white", linewidth=0.3)
        ax5.axvline(np.median(abs_err), color="red", linestyle="--", lw=1, label=f"median={np.median(abs_err):.2f}")
        ax5.set_xlabel("|Residual|"); ax5.set_title("Absolute Error Distribution")
        ax5.legend(fontsize=8); ax5.spines[["top","right"]].set_visible(False)

    else:  # classification
        labels = metrics.get("Class Labels", [])
        cm     = metrics.get("Confusion Matrix")

        # Panel 2: Confusion matrix heatmap
        ax2 = fig.add_subplot(gs[0, 2])
        n = len(labels)
        im = ax2.imshow(cm, cmap="Blues")
        ax2.set_xticks(range(n)); ax2.set_yticks(range(n))
        short = [l[:6] for l in labels]
        ax2.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
        ax2.set_yticklabels(short, fontsize=7)
        for i in range(n):
            for j in range(n):
                ax2.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=7,
                         color="white" if cm[i,j] > cm.max()/2 else "black")
        ax2.set_title(f"Confusion Matrix\nacc={metrics['Accuracy']:.4f}")

        # Panel 3: Per-class accuracy
        ax3 = fig.add_subplot(gs[1, 0])
        per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(1)
        ax3.barh(labels, per_class_acc, color="#2d6a9f")
        ax3.axvline(metrics["Accuracy"], color="red", linestyle="--", lw=1, label="overall")
        ax3.set_xlabel("Accuracy"); ax3.set_title("Per-class Accuracy")
        ax3.set_xlim(0, 1); ax3.legend(fontsize=8)
        ax3.spines[["top","right"]].set_visible(False)

        # Panel 4: Class distribution
        ax4 = fig.add_subplot(gs[1, 1])
        counts_true = np.bincount(targets.astype(int), minlength=n)
        counts_pred = np.bincount(preds.astype(int),   minlength=n)
        x = np.arange(n)
        ax4.bar(x - 0.2, counts_true, 0.4, label="Actual",    color="#2d6a9f")
        ax4.bar(x + 0.2, counts_pred, 0.4, label="Predicted", color="#e07b39", alpha=0.8)
        ax4.set_xticks(x); ax4.set_xticklabels([l[:6] for l in labels], rotation=45, ha="right", fontsize=7)
        ax4.set_title("Class Distribution: Actual vs Predicted")
        ax4.legend(fontsize=8); ax4.spines[["top","right"]].set_visible(False)

        # Panel 5: Per-class F1
        ax5 = fig.add_subplot(gs[1, 2])
        f1s = f1_score(targets.astype(int), preds.astype(int), labels=list(range(n)), average=None, zero_division=0)
        ax5.barh(labels, f1s, color="#5a9e6f")
        ax5.axvline(metrics["Macro F1"], color="red", linestyle="--", lw=1, label="macro avg")
        ax5.set_xlabel("F1 Score"); ax5.set_title("Per-class F1")
        ax5.set_xlim(0, 1); ax5.legend(fontsize=8)
        ax5.spines[["top","right"]].set_visible(False)

    out = f"eval_{target_col}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Evaluation chart saved: {out}")


# ---------------------------------------------------------------------------
# Key targets
# ---------------------------------------------------------------------------

KEY_TARGETS = [
    ("has_refund",              512),
    ("satisfaction_rating",      64),
    ("total_price",             512),
    ("product_type",            512),
    ("resolved_by",             512),
    ("resolution_time_minutes", 512),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_target(df, target_col, epochs, batch_size, plot):
    print(f"\n{'#'*W}")
    print(f"  EVALUATING: {target_col}")
    print(f"{'#'*W}")

    X_cat, X_num, y, task_type, n_classes, fm, te = prepare_for_target(df, target_col)

    model, val_loss, val_metric, metric_name, val_idx, preds, targets = train_model(
        X_cat, X_num, y, task_type, n_classes, fm,
        epochs=epochs, batch_size=batch_size,
    )

    device = next(model.parameters()).device
    print("\nComputing permutation importances...")
    ranked = permutation_importance(
        model, X_cat[val_idx], X_num[val_idx], y[val_idx],
        task_type, fm, device,
    )

    if task_type == "binary":
        metrics = eval_binary(preds, targets)
        print_binary_report(target_col, dict(metrics), ranked)
        if plot:
            make_plots(target_col, task_type, preds, targets, ranked, metrics, te)

    elif task_type == "regression":
        metrics = eval_regression(preds, targets)
        print_regression_report(target_col, dict(metrics), ranked)
        if plot:
            make_plots(target_col, task_type, preds, targets, ranked, metrics, te)

    else:
        metrics = eval_classification(preds, targets, te)
        print_classification_report(target_col, dict(metrics), ranked)
        if plot:
            make_plots(target_col, task_type, preds, targets, ranked, metrics, te)

    return {
        "target": target_col, "task_type": task_type,
        "metric_name": metric_name, "val_metric": val_metric,
        "metrics": metrics, "ranked": ranked,
    }


def main():
    parser = argparse.ArgumentParser(description="Comprehensive evaluation for PrettyFlyNet")
    parser.add_argument("--target",  type=str, default=None, help="Single target column to evaluate")
    parser.add_argument("--all",     action="store_true",    help="Evaluate all 6 key targets")
    parser.add_argument("--epochs",  type=int, default=30,   help="Training epochs (default: 30)")
    parser.add_argument("--plot",    action="store_true",    help="Save evaluation plots")
    parser.add_argument("--data_dir",type=str, default=None)
    args = parser.parse_args()

    df = build_feature_table(args.data_dir)

    if args.all:
        results = []
        for target_col, batch_size in KEY_TARGETS:
            r = run_target(df, target_col, args.epochs, batch_size, args.plot)
            results.append(r)

        # Summary table
        print(f"\n\n{'='*W}")
        print("  SUMMARY — ALL TARGETS")
        print(f"{'='*W}")
        print(f"  {'Target':<30} {'Task':<16} {'Metric':<14} {'Value':>8}")
        print(f"  {'-'*60}")
        for r in results:
            print(f"  {r['target']:<30} {r['task_type']:<16} {r['metric_name']:<14} {r['val_metric']:>8.4f}")

    elif args.target:
        run_target(df, args.target, args.epochs,
                   64 if args.target == "satisfaction_rating" else 512,
                   args.plot)
    else:
        print("Provide --target <col> or --all")
        sys.exit(1)


if __name__ == "__main__":
    main()
