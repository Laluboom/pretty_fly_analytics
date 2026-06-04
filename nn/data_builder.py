import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _path(filename):
    return os.path.join(DATA_DIR, filename)


# ---------------------------------------------------------------------------
# 1.1  Raw table loaders
# ---------------------------------------------------------------------------

def _load_orders():
    df = pd.read_csv(_path("orders.csv"), parse_dates=["created_at"])
    return df[[
        "order_id", "customer_id", "created_at",
        "subtotal", "total_discounts", "total_shipping", "total_tax", "total_price",
        "utm_source", "utm_medium", "utm_campaign",
        "discount_code", "financial_status",
    ]]


def _load_line_items():
    return pd.read_csv(_path("line_items.csv"))[[
        "line_item_id", "order_id", "variant_id", "product_id",
        "quantity", "price", "total_discount",
    ]]


def _load_variants():
    df = pd.read_csv(_path("variants.csv"))
    return df[[
        "variant_id",
        "price", "weight_grams", "inventory_quantity",
        "option1_value", "option2_value",
    ]].rename(columns={"price": "variant_price"})


def _load_products():
    return pd.read_csv(_path("products.csv"))[[
        "product_id", "product_type", "gender_segment", "collection",
    ]]


def _load_customers():
    return pd.read_csv(_path("customers.csv"))[[
        "customer_id", "orders_count", "total_spent",
        "acquisition_source", "default_country",
        "gender_segment_affinity", "accepts_marketing",
    ]]


def _load_refunds():
    df = pd.read_csv(_path("refunds.csv"))

    def _extract_variant_ids(val):
        try:
            return json.loads(val) if isinstance(val, str) else []
        except Exception:
            return []

    df["refund_variant_ids"] = df["refund_line_items"].apply(_extract_variant_ids)
    df = df[["order_id", "amount", "reason"]].rename(
        columns={"amount": "refund_amount", "reason": "refund_reason"}
    )
    df["has_refund"] = True
    # keep first refund per order (edge case: multiple partial refunds)
    return df.drop_duplicates("order_id")


def _load_support_tickets():
    df = pd.read_csv(_path("support_tickets.csv"))
    df = df.rename(columns={
        "related_order_id": "order_id",
        "category": "ticket_category",
        "channel": "support_channel",
    })
    df["has_ticket"] = True
    return df[[
        "order_id", "has_ticket", "ticket_category",
        "resolved_by", "resolution_time_minutes",
        "satisfaction_rating", "support_channel",
    ]].dropna(subset=["order_id"])


def _load_po_line_items():
    return pd.read_csv(_path("po_line_items.csv"))[
        ["variant_id", "landed_cost_per_unit_gbp"]
    ].drop_duplicates("variant_id")


def _load_google_ads():
    df = pd.read_csv(_path("google_ads_daily.csv"), parse_dates=["date"])
    return df.groupby(["campaign_name", "date"], as_index=False).agg(
        google_spend=("spend_gbp", "sum"),
        google_impressions=("impressions", "sum"),
        google_clicks=("clicks", "sum"),
        google_conversions=("conversions", "sum"),
    )


def _load_meta_ads():
    df = pd.read_csv(_path("meta_ads_daily.csv"), parse_dates=["date"])
    return df.groupby(["campaign_name", "date"], as_index=False).agg(
        meta_spend=("spend_gbp", "sum"),
        meta_impressions=("impressions", "sum"),
        meta_clicks=("clicks", "sum"),
        meta_conversions=("conversions", "sum"),
    )


# ---------------------------------------------------------------------------
# 1.2  Join chain
# ---------------------------------------------------------------------------

def _build_joined(
    line_items, orders, variants, products,
    customers, po_line_items, refunds, support_tickets,
    google_ads, meta_ads,
):
    df = line_items.merge(orders, on="order_id", how="left")
    df = df.merge(variants, on="variant_id", how="left")
    df = df.merge(products, on="product_id", how="left")
    df = df.merge(customers, on="customer_id", how="left")
    df = df.merge(po_line_items, on="variant_id", how="left")
    df = df.merge(refunds, on="order_id", how="left")
    df["has_refund"] = df["has_refund"].notna() & df["has_refund"].eq(True)
    df["refund_reason"] = df["refund_reason"].fillna("none")
    df["refund_amount"] = df["refund_amount"].fillna(0.0)
    df = df.merge(support_tickets, on="order_id", how="left")
    df["has_ticket"] = df["has_ticket"].notna() & df["has_ticket"].eq(True)
    df["ticket_category"] = df["ticket_category"].fillna("none")
    df["resolved_by"] = df["resolved_by"].fillna("none")
    df["resolution_time_minutes"] = df["resolution_time_minutes"].fillna(0.0)
    df["satisfaction_rating"] = df["satisfaction_rating"]  # keep NaN — sparse target
    df["support_channel"] = df["support_channel"].fillna("none")

    # ad join key: utm_campaign + order date
    df["order_date"] = df["created_at"].dt.normalize()
    df = df.merge(
        google_ads, left_on=["utm_campaign", "order_date"],
        right_on=["campaign_name", "date"], how="left",
    ).drop(columns=["campaign_name", "date"], errors="ignore")
    df = df.merge(
        meta_ads, left_on=["utm_campaign", "order_date"],
        right_on=["campaign_name", "date"], how="left",
    ).drop(columns=["campaign_name", "date"], errors="ignore")

    for col in ["google_spend", "google_impressions", "google_clicks", "google_conversions",
                "meta_spend", "meta_impressions", "meta_clicks", "meta_conversions"]:
        df[col] = df[col].fillna(0.0)

    return df


# ---------------------------------------------------------------------------
# 1.3  Feature engineering
# ---------------------------------------------------------------------------

def _engineer_features(df):
    df = df.copy()
    df["discount_pct"] = (df["total_discounts"] / df["subtotal"].replace(0, np.nan)).clip(0, 1).fillna(0)
    df["gross_margin_est"] = (
        (df["price"] - df["landed_cost_per_unit_gbp"]) / df["price"].replace(0, np.nan)
    ).fillna(df["price"].pipe(lambda s: (s - s.median()) / s.replace(0, np.nan)).fillna(0))
    df["order_month"] = df["created_at"].dt.month
    df["order_dayofweek"] = df["created_at"].dt.dayofweek
    df["order_hour"] = df["created_at"].dt.hour
    df["is_discounted"] = df["discount_code"].notna().astype(int)
    df["total_ad_spend"] = df["google_spend"] + df["meta_spend"]
    df["total_ad_conversions"] = df["google_conversions"] + df["meta_conversions"]
    # Pre-computed sum so total_price model sees the exact algebraic identity
    df["price_components_sum"] = (
        df["subtotal"] + df["total_shipping"] + df["total_tax"] - df["total_discounts"]
    )
    df["accepts_marketing"] = df["accepts_marketing"].astype(int)
    df["has_refund"] = df["has_refund"].astype(int)
    df["has_ticket"] = df["has_ticket"].astype(int)
    df["damaged_in_transit"] = (df["refund_reason"] == "damaged_in_transit").astype(int)
    df["size_issue"] = df["refund_reason"].isin(["size_too_small", "size_too_large"]).astype(int)

    drop_cols = [
        "order_id", "line_item_id", "variant_id", "product_id", "customer_id",
        "created_at", "order_date", "discount_code", "utm_campaign",
        "refund_variant_ids",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df


# ---------------------------------------------------------------------------
# 1.4  Encode and scale
# ---------------------------------------------------------------------------

CATEGORICAL_COLS = [
    "utm_source", "utm_medium",
    "product_type", "gender_segment", "collection",
    "option1_value", "option2_value",
    "acquisition_source", "default_country", "gender_segment_affinity",
    "refund_reason", "ticket_category", "resolved_by", "support_channel",
    "financial_status",
]

# B1: columns that directly encode the target and must be excluded as features.
# Without this, models reach AUC=1.0 by reading the answer rather than predicting it.
LEAKAGE_MAP = {
    "has_refund":               ["financial_status", "refund_reason", "refund_amount"],
    "has_ticket":               ["ticket_category", "resolved_by", "resolution_time_minutes",
                                 "satisfaction_rating", "support_channel"],
    "damaged_in_transit":       ["refund_reason", "refund_amount", "has_refund", "financial_status"],
    "size_issue":               ["refund_reason", "refund_amount", "has_refund", "financial_status"],
    "landed_cost_per_unit_gbp": ["gross_margin_est"],
    "variant_price":            ["price", "gross_margin_est"],
}

# B2: targets where the feature table contains rows that are structurally 0
# (not a real measurement) — filter to meaningful rows only before training.
# resolution_time_minutes is 0.0 for 97.6% of rows (orders with no ticket).
FILTER_MAP = {
    "resolution_time_minutes": ("has_ticket", 1),
    "damaged_in_transit":      ("has_refund", 1),
    "size_issue":              ("has_refund", 1),
}


def _encode_and_scale(df, target_col):
    df = df.copy()

    feature_cols = [c for c in df.columns if c != target_col]
    cat_cols = [c for c in CATEGORICAL_COLS if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    for col in cat_cols:
        df[col] = df[col].fillna("unknown").astype(str)

    encoder_map = {}
    cat_vocab_sizes = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoder_map[col] = le
        cat_vocab_sizes[col] = len(le.classes_)

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    feature_meta = {
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "cat_vocab_sizes": cat_vocab_sizes,
        "encoders": encoder_map,
        "scaler": scaler,
    }
    return df, feature_meta


# ---------------------------------------------------------------------------
# 1.5  Target extraction
# ---------------------------------------------------------------------------

def get_X_y(df, feature_meta, target_col):
    df = df.dropna(subset=[target_col]).copy()

    cat_cols = feature_meta["cat_cols"]
    num_cols = feature_meta["num_cols"]

    unique_vals = df[target_col].nunique()
    dtype = df[target_col].dtype

    if dtype == object or (dtype.kind in ("i", "u") and unique_vals <= 20 and unique_vals > 2):
        le = LabelEncoder()
        y = le.fit_transform(df[target_col].astype(str)).astype(np.int64)
        task_type = "classification"
        n_classes = len(le.classes_)
        target_encoder = le
    elif unique_vals == 2 or set(df[target_col].unique()).issubset({0, 1, True, False}):
        y = df[target_col].astype(float).values
        task_type = "binary"
        n_classes = 1
        target_encoder = None
    else:
        y = df[target_col].astype(float).values
        task_type = "regression"
        n_classes = 1
        target_encoder = None

    X_cat = df[cat_cols].values.astype(np.int64)
    X_num = df[num_cols].values.astype(np.float32)

    return X_cat, X_num, y, task_type, n_classes, target_encoder


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_feature_table(data_dir=None):
    global DATA_DIR
    if data_dir:
        DATA_DIR = data_dir

    print("Loading tables...")
    orders = _load_orders()
    line_items = _load_line_items()
    variants = _load_variants()
    products = _load_products()
    customers = _load_customers()
    po_line_items = _load_po_line_items()
    refunds = _load_refunds()
    support_tickets = _load_support_tickets()
    google_ads = _load_google_ads()
    meta_ads = _load_meta_ads()

    print("Joining tables...")
    df = _build_joined(
        line_items, orders, variants, products,
        customers, po_line_items, refunds, support_tickets,
        google_ads, meta_ads,
    )

    print("Engineering features...")
    df = _engineer_features(df)

    print(f"Feature table ready: {len(df):,} rows x {df.shape[1]} columns")
    return df


def prepare_for_target(df, target_col):
    if target_col not in df.columns:
        available = sorted(df.columns.tolist())
        raise ValueError(f"'{target_col}' not found. Available columns:\n{available}")

    df_clean = df.copy()

    # B2 FIRST: filter rows — must run before leakage drop because the filter column
    # may itself be a leakage column (e.g. has_refund for damaged_in_transit).
    if target_col in FILTER_MAP:
        filter_col, filter_val = FILTER_MAP[target_col]
        if filter_col in df_clean.columns:
            before = len(df_clean)
            df_clean = df_clean[df_clean[filter_col] == filter_val].copy()
            print(f"Filter : {filter_col}=={filter_val} → {len(df_clean):,} rows (from {before:,})")

    # B1 SECOND: drop leakage columns that directly encode this target
    leakage_cols = LEAKAGE_MAP.get(target_col, [])
    if leakage_cols:
        drop = [c for c in leakage_cols if c in df_clean.columns]
        df_clean = df_clean.drop(columns=drop)
        print(f"Leakage: dropped {drop}")

    df_encoded, feature_meta = _encode_and_scale(df_clean, target_col)
    X_cat, X_num, y, task_type, n_classes, target_encoder = get_X_y(df_encoded, feature_meta, target_col)

    print(f"Target : {target_col}")
    print(f"Task   : {task_type}  |  Classes: {n_classes}  |  Rows: {len(y):,}  |  Features: {X_cat.shape[1] + X_num.shape[1]}")

    return X_cat, X_num, y, task_type, n_classes, feature_meta, target_encoder
