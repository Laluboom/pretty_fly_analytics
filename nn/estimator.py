import argparse
import sys
import os
import json
import pickle
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
    parser.add_argument("--save_model",  type=str,   default=None,  help="Save model + metadata to this path prefix (e.g. models/has_refund)")
    parser.add_argument("--predict",     type=str,   default=None,  help="JSON string or path to JSON file of feature values; requires --load_model")
    parser.add_argument("--load_model",  type=str,   default=None,  help="Path prefix of saved model to load for --predict")
    parser.add_argument("--plot",        action="store_true",        help="Save feature importance bar chart as importance_{target}.png")
    parser.add_argument("--subset",      type=str,   default=None,  help="Filter rows before training, e.g. \"product_type=Hoodie\"")
    parser.add_argument("--recommend",   action="store_true",        help="After --predict, call an LLM via OpenRouter for a plain-English business recommendation (requires OPENROUTER_API_KEY)")
    parser.add_argument("--model",       type=str,   default=None,  help=f"OpenRouter model to use with --recommend (default: {OPENROUTER_MODELS[0]})")
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
    top_score = max((s for _, s in ranked_importance[:10]), default=1) or 1
    for rank, (feat, score) in enumerate(ranked_importance[:10], 1):
        # B3: normalize bar width relative to top feature so all bars are visible
        bar = "█" * int(20 * score / top_score) if score > 0 else ""
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
# 5.1  Save model
# ---------------------------------------------------------------------------

def save_model(model, feature_meta, task_type, n_classes, target_col, target_encoder, prefix,
               top_importances=None):
    os.makedirs(os.path.dirname(prefix) if os.path.dirname(prefix) else ".", exist_ok=True)
    torch.save(model.state_dict(), f"{prefix}.pt")
    meta = {
        "feature_meta":    feature_meta,
        "task_type":       task_type,
        "n_classes":       n_classes,
        "target_col":      target_col,
        "target_encoder":  target_encoder,
        "n_num_features":  len(feature_meta["num_cols"]),
        "top_importances": top_importances or [],
    }
    with open(f"{prefix}.pkl", "wb") as f:
        pickle.dump(meta, f)
    print(f"\nModel saved: {prefix}.pt  +  {prefix}.pkl")


# ---------------------------------------------------------------------------
# 5.2  Load model + predict from JSON row
# ---------------------------------------------------------------------------

def load_and_predict(prefix, raw_input, data_dir=None, recommend=False, model=None):
    with open(f"{prefix}.pkl", "rb") as f:
        meta = pickle.load(f)

    feature_meta   = meta["feature_meta"]
    task_type      = meta["task_type"]
    n_classes      = meta["n_classes"]
    target_col     = meta["target_col"]
    target_encoder = meta["target_encoder"]
    n_num          = meta["n_num_features"]

    model = PrettyFlyNet(
        cat_vocab_sizes=feature_meta["cat_vocab_sizes"],
        n_num_features=n_num,
        task_type=task_type,
        n_classes=n_classes,
    )
    model.load_state_dict(torch.load(f"{prefix}.pt", map_location="cpu"))
    model.eval()

    # Parse input
    if os.path.isfile(raw_input):
        with open(raw_input) as f:
            row = json.load(f)
    else:
        row = json.loads(raw_input)

    # Warn on unknown keys before auto-computation adds its own keys to row
    all_feature_cols = set(feature_meta["cat_cols"]) | set(feature_meta["num_cols"])
    unknown_keys = set(row.keys()) - all_feature_cols
    if unknown_keys:
        print(f"Warning: unknown input keys (ignored): {sorted(unknown_keys)}")

    # Auto-compute engineered features from raw inputs if components are present
    def _get(k, default=0.0):
        return float(row.get(k, default))

    row["discount_pct"] = row.get("discount_pct",
        min(_get("total_discounts") / _get("subtotal"), 1.0) if _get("subtotal") else 0.0)
    row["gross_margin_est"] = row.get("gross_margin_est",
        (_get("price") - _get("landed_cost_per_unit_gbp")) / _get("price") if _get("price") else 0.0)
    row["total_ad_spend"] = row.get("total_ad_spend",
        _get("google_spend") + _get("meta_spend"))
    row["total_ad_conversions"] = row.get("total_ad_conversions",
        _get("google_conversions") + _get("meta_conversions"))
    row["price_components_sum"] = row.get("price_components_sum",
        _get("subtotal") + _get("total_shipping") + _get("total_tax") - _get("total_discounts"))

    # Build cat and num arrays, filling missing with defaults
    cat_cols = feature_meta["cat_cols"]
    num_cols = feature_meta["num_cols"]
    encoders = feature_meta["encoders"]
    scaler   = feature_meta["scaler"]

    # Encode categoricals
    x_cat = []
    for col in cat_cols:
        val = str(row.get(col, "unknown"))
        le  = encoders[col]
        if val in le.classes_:
            x_cat.append(le.transform([val])[0])
        else:
            x_cat.append(0)  # fallback to first class for unseen values

    # Sentiment features (avg_sentiment, min_sentiment, pct_negative_msgs) default to 0 here,
    # meaning "no support thread = neutral customer" — correct behaviour for inference on new orders.

    # Scale numerics — build a single-row DataFrame so sklearn doesn't warn about feature names
    import pandas as pd
    num_row = pd.DataFrame([[float(row.get(col, 0.0)) for col in num_cols]], columns=num_cols)
    num_scaled = scaler.transform(num_row)[0].astype(np.float32)

    x_cat_t = torch.tensor(np.array([x_cat]), dtype=torch.long)
    x_num_t = torch.tensor(np.array([num_scaled]), dtype=torch.float32)

    with torch.no_grad():
        out = model(x_cat_t, x_num_t)

    print(f"\nPredicting: {target_col}")
    print(f"Input features used: {len(x_cat)} categorical + {len(num_scaled)} numeric")

    prediction_str = ""
    if task_type == "binary":
        prob = out.squeeze().item()
        label = 1 if prob >= 0.5 else 0
        prediction_str = f"{label}  (probability={prob:.4f})"
        print(f"Prediction : {prediction_str}")
    elif task_type == "classification":
        probs = torch.softmax(out, dim=1).squeeze().numpy()
        pred_idx = int(probs.argmax())
        if target_encoder:
            pred_label = target_encoder.inverse_transform([pred_idx])[0]
            prediction_str = f"{pred_label}  (confidence={probs[pred_idx]:.4f})"
        else:
            pred_label = f"class {pred_idx}"
            prediction_str = f"{pred_label}  (confidence={probs[pred_idx]:.4f})"
        print(f"Prediction : {prediction_str}")
    else:
        val = out.squeeze().item()
        prediction_str = f"{val:.4f}"
        print(f"Prediction : {prediction_str}")

    if recommend:
        # Load top importances from saved metadata if available
        top_importances = meta.get("top_importances", [])
        # Filter out auto-computed keys added before the warning check
        user_row = {k: v for k, v in row.items()
                    if k in set(cat_cols) | set(num_cols)}
        get_recommendation(target_col, task_type, prediction_str, user_row, top_importances,
                           model=model)


# ---------------------------------------------------------------------------
# 5.3  Plot feature importance
# ---------------------------------------------------------------------------

def plot_importance(ranked, target_col, metric_name, val_metric):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_n   = 15
    top     = ranked[:top_n]
    names   = [f for f, _ in reversed(top)]
    scores  = [s for _, s in reversed(top)]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, scores, color="#2d6a9f", edgecolor="white")
    ax.set_xlabel("Permutation Importance (Δ Loss)", fontsize=11)
    ax.set_title(
        f"Feature Importances — target: {target_col}\n{metric_name} = {val_metric:.4f}",
        fontsize=13, fontweight="bold",
    )
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out_path = f"importance_{target_col}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nImportance chart saved: {out_path}")


# ---------------------------------------------------------------------------
# Ph9  LLM recommendation layer
# ---------------------------------------------------------------------------

# OpenRouter model preference order — first available is used
OPENROUTER_MODELS = [
    "deepseek/deepseek-chat-v3-0324:free",   # free, best quality
    "google/gemma-3-27b-it:free",             # free, solid fallback
    "mistralai/mistral-small-3.1-24b-instruct",  # cheap, last resort
]


def get_recommendation(target_col, task_type, prediction_str, input_row, top_importances,
                       model=None):
    """
    Call an LLM via OpenRouter with the model's prediction and top feature importances
    to produce a plain-English business recommendation for Pretty Fly.

    Parameters
    ----------
    target_col      : str        — e.g. "has_refund"
    task_type       : str        — "binary" | "regression" | "classification"
    prediction_str  : str        — human-readable prediction, e.g. "1 (probability=0.81)"
    input_row       : dict       — the raw JSON the user supplied
    top_importances : list       — [(feature_name, score), ...] top 5 from permutation importance
    model           : str | None — override the default model preference order
    """
    from openai import OpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("\nError: OPENROUTER_API_KEY environment variable is not set.")
        print("  export OPENROUTER_API_KEY=sk-or-...")
        return

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    chosen_model = model or OPENROUTER_MODELS[0]

    importance_lines = "\n".join(
        f"  {i+1}. {name} (importance score: {score:.4f})"
        for i, (name, score) in enumerate(top_importances[:5])
    )
    input_lines = "\n".join(f"  {k}: {v}" for k, v in input_row.items())

    prompt = f"""You are a data analyst assistant for Pretty Fly, a London-based streetwear brand.
A neural network has just made the following prediction:

Target variable : {target_col}
Task type       : {task_type}
Prediction      : {prediction_str}

Input features provided:
{input_lines}

Top 5 features driving this prediction (permutation importance):
{importance_lines}

Brand context: Pretty Fly sells hoodies, tees, caps, outerwear, sweatpants and trainers.
Their customers are primarily UK-based. Data covers 24 months of orders, returns, support
tickets, supplier chain, email campaigns, and ad spend.

Based on the prediction and the signals above, give a concise, actionable business
recommendation. Structure your response as:

Risk/Outlook : [one line summary of what the prediction means]
Key signal   : [the most important feature and what it tells us]

Suggested actions:
1. [specific action]
2. [specific action]
3. [specific action]

Keep the tone direct and practical. No generic advice — tie every action to the specific
prediction and features shown."""

    print("\n" + "=" * 60)
    print(f"  RECOMMENDATION  ({chosen_model})")
    print("=" * 60)

    stream = client.chat.completions.create(
        model=chosen_model,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# 3.2  Main orchestration
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --predict mode: skip training entirely, load saved model
    if args.predict:
        if not args.load_model:
            print("Error: --predict requires --load_model <prefix>")
            sys.exit(1)
        if args.recommend and not os.environ.get("OPENROUTER_API_KEY"):
            print("Error: --recommend requires OPENROUTER_API_KEY to be set.")
            print("  export OPENROUTER_API_KEY=sk-or-...")
            sys.exit(1)
        load_and_predict(args.load_model, args.predict, args.data_dir,
                         recommend=args.recommend, model=args.model)
        return

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

    # --subset: filter rows before training
    if args.subset:
        if "=" not in args.subset:
            print(f"Error: --subset must be in the form 'col=value', got: {args.subset!r}")
            sys.exit(1)
        sub_col, sub_val = args.subset.split("=", 1)
        if sub_col not in df.columns:
            print(f"Error: --subset column '{sub_col}' not found. Run --list_targets to see valid columns.")
            sys.exit(1)
        matches = df[sub_col].astype(str) == sub_val
        if not matches.any():
            print(f"Error: --subset value '{sub_val}' not found in column '{sub_col}'.")
            print(f"  Available values: {sorted(df[sub_col].astype(str).unique())}")
            sys.exit(1)
        before = len(df)
        df = df[matches].copy()
        print(f"Subset : {sub_col}=={sub_val} → {len(df):,} rows (from {before:,})")

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

    # --save_model
    if args.save_model:
        save_model(model, feature_meta, task_type, n_classes,
                   args.target, target_encoder, args.save_model,
                   top_importances=ranked)

    # --plot
    if args.plot:
        plot_importance(ranked, args.target, metric_name, val_metric)


if __name__ == "__main__":
    main()
