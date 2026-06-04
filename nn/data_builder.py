import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _path(filename):
    return os.path.join(DATA_DIR, filename)


# ---------------------------------------------------------------------------
# 1.1  Raw table loaders — original 10 tables
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
# 1.1b  New table loaders — Phase 8 data integration
# ---------------------------------------------------------------------------

def _load_discount_codes():
    """8-row lookup: discount code → type + value. Join on orders.discount_code."""
    df = pd.read_csv(_path("discount_codes.csv"))[["code", "type", "value"]]
    return df.rename(columns={"type": "discount_type", "value": "discount_value"})


def _load_supplier_features():
    """Chain: po_line_items → purchase_orders → suppliers → per-variant supplier features."""
    po_li = pd.read_csv(_path("po_line_items.csv"))[["po_id", "variant_id"]]
    pos = pd.read_csv(_path("purchase_orders.csv"),
                      parse_dates=["expected_delivery", "actual_delivery"])
    pos["delivery_delay_days"] = (
        pos["actual_delivery"] - pos["expected_delivery"]
    ).dt.days.fillna(0).astype(int)
    pos = pos[["po_id", "supplier_id", "delivery_delay_days"]]

    suppliers = pd.read_csv(_path("suppliers.csv"))[
        ["supplier_id", "country", "lead_time_days"]
    ].rename(columns={"country": "supplier_country"})

    merged = po_li.merge(pos, on="po_id", how="left")
    merged = merged.merge(suppliers, on="supplier_id", how="left")
    # Multiple POs per variant — keep most recent (po_ids are sequential)
    merged = merged.sort_values("po_id").drop_duplicates("variant_id", keep="last")
    return merged[["variant_id", "supplier_country", "lead_time_days", "delivery_delay_days"]]


def _load_inventory_features():
    """Aggregate 76k inventory movements to per-variant stock features."""
    df = pd.read_csv(_path("inventory_movements.csv"))

    # Latest stock balance per variant
    latest = (
        df.sort_values("date")
        .groupby("variant_id", as_index=False)
        .last()[["variant_id", "running_balance"]]
        .rename(columns={"running_balance": "variant_latest_stock"})
    )

    # Return rate: returns / sales per variant
    sales   = df[df["type"] == "sale"].groupby("variant_id").size().rename("_sales")
    returns = df[df["type"] == "return"].groupby("variant_id").size().rename("_returns")
    restocks = df[df["type"] == "po_receipt"].groupby("variant_id").size().rename("variant_restock_count")

    rates = pd.concat([sales, returns, restocks], axis=1).fillna(0).reset_index()
    rates["variant_return_rate"] = (
        rates["_returns"] / rates["_sales"].replace(0, np.nan)
    ).fillna(0).round(4)
    rates = rates[["variant_id", "variant_return_rate", "variant_restock_count"]]

    return latest.merge(rates, on="variant_id", how="left").fillna(0)


def _load_email_features():
    """Aggregate per-customer email engagement (excludes auto-attributed 'converted' events)."""
    df = pd.read_csv(_path("email_events.csv"), parse_dates=["timestamp"])
    # 'converted' events are auto-attributed post-purchase, not genuine engagement
    df = df[df["event_type"] != "converted"].copy()

    opens    = df[df["event_type"] == "opened"].groupby("customer_id").size().rename("email_open_count")
    clicks   = df[df["event_type"] == "clicked"].groupby("customer_id").size().rename("email_click_count")
    campaigns = df.groupby("customer_id")["campaign_id"].nunique().rename("email_campaign_count")

    REFERENCE_DATE = pd.Timestamp("2026-06-04")
    days_since = (
        (REFERENCE_DATE - df.groupby("customer_id")["timestamp"].max())
        .dt.days
        .rename("days_since_last_email")
    )

    result = pd.concat([opens, clicks, campaigns, days_since], axis=1).fillna(0).reset_index()
    result = result.astype({
        "email_open_count": int,
        "email_click_count": int,
        "email_campaign_count": int,
        "days_since_last_email": int,
    })
    return result


def _load_address_features():
    """Extract non-PII geography: city + postcode district (e.g. 'IG1' from 'IG1 1AT')."""
    df = pd.read_csv(_path("addresses.csv"))[["customer_id", "city", "postcode"]]
    df["postcode_district"] = df["postcode"].str.split(" ").str[0].fillna("unknown")
    df["city"] = df["city"].fillna("unknown")
    # Drop province (90% null), country (all GB), full postcode, PII already excluded by column selection
    return df[["customer_id", "city", "postcode_district"]]


def _parse_support_messages():
    """Read support_messages.json once; returns the raw list. Call this once and pass to loaders."""
    with open(_path("support_messages.json")) as f:
        return json.load(f)


def _load_support_message_features(raw):
    """Per-ticket conversation features from pre-parsed message list → join via order_id."""
    # Build ticket_id → order_id map from support_tickets
    tickets = pd.read_csv(_path("support_tickets.csv"))[
        ["ticket_id", "related_order_id"]
    ].rename(columns={"related_order_id": "order_id"})

    rows = []
    for ticket in raw:
        tid = ticket["ticket_id"]
        msgs = ticket.get("messages", [])
        if not msgs:
            continue

        customer_msgs = [m for m in msgs if m["sender"] == "customer"]
        agent_msgs    = [m for m in msgs if m["sender"] in ("bot", "human")]

        msg_count            = len(msgs)
        customer_msg_count   = len(customer_msgs)
        avg_customer_msg_len = (
            np.mean([len(m["body"]) for m in customer_msgs]) if customer_msgs else 0
        )

        # Escalation = bot message immediately followed by a human message
        senders = [m["sender"] for m in msgs]
        n_escalations = sum(
            1 for i in range(len(senders) - 1)
            if senders[i] == "bot" and senders[i + 1] == "human"
        )

        # Time from first customer message to first agent reply (seconds)
        try:
            first_customer_ts = pd.Timestamp(next(
                m["timestamp"] for m in msgs if m["sender"] == "customer"
            ))
            first_agent_ts = pd.Timestamp(next(
                m["timestamp"] for m in msgs if m["sender"] in ("bot", "human")
            ))
            response_time_first_seconds = max(
                0, (first_agent_ts - first_customer_ts).total_seconds()
            )
        except StopIteration:
            response_time_first_seconds = 0

        rows.append({
            "ticket_id": tid,
            "msg_count": msg_count,
            "customer_msg_count": customer_msg_count,
            "avg_customer_msg_length": round(avg_customer_msg_len, 1),
            "n_escalations": n_escalations,
            "response_time_first_seconds": response_time_first_seconds,
        })

    msg_features = pd.DataFrame(rows)
    # Join ticket_id → order_id
    result = msg_features.merge(tickets, on="ticket_id", how="left").dropna(subset=["order_id"])
    return result.drop(columns=["ticket_id"])


def _load_support_sentiment_features(raw):
    """VADER sentiment scored on customer messages only → per-order features via ticket_id."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    tickets = pd.read_csv(_path("support_tickets.csv"))[
        ["ticket_id", "related_order_id"]
    ].rename(columns={"related_order_id": "order_id"})

    rows = []
    for ticket in raw:
        tid = ticket["ticket_id"]
        customer_msgs = [
            m for m in ticket.get("messages", []) if m["sender"] == "customer"
        ]
        if not customer_msgs:
            continue

        scores = [sia.polarity_scores(m["body"])["compound"] for m in customer_msgs]
        rows.append({
            "ticket_id":          tid,
            "avg_sentiment":      round(float(np.mean(scores)), 4),
            "min_sentiment":      round(float(np.min(scores)), 4),
            "pct_negative_msgs":  round(sum(s < -0.05 for s in scores) / len(scores), 4),
        })

    sentiment = pd.DataFrame(rows)
    result = sentiment.merge(tickets, on="ticket_id", how="left").dropna(subset=["order_id"])
    return result.drop(columns=["ticket_id"])


# ---------------------------------------------------------------------------
# 1.2  Join chain
# ---------------------------------------------------------------------------

def _build_joined(
    line_items, orders, variants, products,
    customers, po_line_items, refunds, support_tickets,
    google_ads, meta_ads,
    discount_codes, supplier_features, inventory_features,
    email_features, address_features, support_message_features,
    sentiment_features,
):
    df = line_items.merge(orders, on="order_id", how="left")
    df = df.merge(variants, on="variant_id", how="left")
    df = df.merge(products, on="product_id", how="left")
    df = df.merge(customers, on="customer_id", how="left")
    df = df.merge(po_line_items, on="variant_id", how="left")

    # --- New: supplier chain (joins via variant_id, same as po_line_items) ---
    df = df.merge(supplier_features, on="variant_id", how="left")
    df["supplier_country"] = df["supplier_country"].fillna("unknown")
    df["lead_time_days"] = df["lead_time_days"].fillna(df["lead_time_days"].median())
    df["delivery_delay_days"] = df["delivery_delay_days"].fillna(0)

    # --- New: inventory features (per variant) ---
    df = df.merge(inventory_features, on="variant_id", how="left")
    for col in ["variant_latest_stock", "variant_return_rate", "variant_restock_count"]:
        df[col] = df[col].fillna(0)

    # --- New: discount code details (join on discount_code from orders) ---
    df = df.merge(
        discount_codes, left_on="discount_code", right_on="code", how="left"
    ).drop(columns=["code"], errors="ignore")
    df["discount_type"] = df["discount_type"].fillna("none")
    df["discount_value"] = df["discount_value"].fillna(0)

    # --- New: email engagement (per customer) ---
    df = df.merge(email_features, on="customer_id", how="left")
    df["email_open_count"]       = df["email_open_count"].fillna(0).astype(int)
    df["email_click_count"]      = df["email_click_count"].fillna(0).astype(int)
    df["email_campaign_count"]   = df["email_campaign_count"].fillna(0).astype(int)
    # Customers with no email history get sentinel 999 (not 0 — 0 would mean "opened 0 times today")
    df["days_since_last_email"]  = df["days_since_last_email"].fillna(999).astype(int)

    # --- New: address geography (per customer) ---
    df = df.merge(address_features, on="customer_id", how="left")
    df["city"]               = df["city"].fillna("unknown")
    df["postcode_district"]  = df["postcode_district"].fillna("unknown")

    # --- Original refunds ---
    df = df.merge(refunds, on="order_id", how="left")
    df["has_refund"]    = df["has_refund"].notna() & df["has_refund"].eq(True)
    df["refund_reason"] = df["refund_reason"].fillna("none")
    df["refund_amount"] = df["refund_amount"].fillna(0.0)

    # --- Original support tickets ---
    df = df.merge(support_tickets, on="order_id", how="left")
    df["has_ticket"]               = df["has_ticket"].notna() & df["has_ticket"].eq(True)
    df["ticket_category"]          = df["ticket_category"].fillna("none")
    df["resolved_by"]              = df["resolved_by"].fillna("none")
    df["resolution_time_minutes"]  = df["resolution_time_minutes"].fillna(0.0)
    df["satisfaction_rating"]      = df["satisfaction_rating"]  # keep NaN — sparse target
    df["support_channel"]          = df["support_channel"].fillna("none")

    # --- New: support message features (per order via ticket) ---
    df = df.merge(support_message_features, on="order_id", how="left")
    for col in ["msg_count", "customer_msg_count", "avg_customer_msg_length",
                "n_escalations", "response_time_first_seconds"]:
        df[col] = df[col].fillna(0)

    # --- New: VADER sentiment features (customer message tone per ticket) ---
    df = df.merge(sentiment_features, on="order_id", how="left")
    # Orders with no ticket → neutral sentiment (0 = no signal either way)
    for col in ["avg_sentiment", "min_sentiment", "pct_negative_msgs"]:
        df[col] = df[col].fillna(0)

    # --- Original ads ---
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
    ).fillna(0)
    df["order_month"]      = df["created_at"].dt.month
    df["order_dayofweek"]  = df["created_at"].dt.dayofweek
    df["order_hour"]       = df["created_at"].dt.hour
    df["is_discounted"]    = df["discount_code"].notna().astype(int)
    df["total_ad_spend"]   = df["google_spend"] + df["meta_spend"]
    df["total_ad_conversions"] = df["google_conversions"] + df["meta_conversions"]
    # Pre-computed sum: gives total_price model the algebraic identity directly
    df["price_components_sum"] = (
        df["subtotal"] + df["total_shipping"] + df["total_tax"] - df["total_discounts"]
    )
    df["accepts_marketing"]     = df["accepts_marketing"].astype(int)
    df["has_refund"]            = df["has_refund"].astype(int)
    df["has_ticket"]            = df["has_ticket"].astype(int)
    df["damaged_in_transit"]    = (df["refund_reason"] == "damaged_in_transit").astype(int)
    df["size_issue"]            = df["refund_reason"].isin(["size_too_small", "size_too_large"]).astype(int)

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
    # Phase 8 additions
    "discount_type", "supplier_country", "city", "postcode_district",
]

# B1: columns that directly encode the target — drop before training to prevent leakage.
LEAKAGE_MAP = {
    "has_refund":               ["financial_status", "refund_reason", "refund_amount"],
    "has_ticket":               ["ticket_category", "resolved_by", "resolution_time_minutes",
                                 "satisfaction_rating", "support_channel",
                                 "msg_count", "customer_msg_count", "avg_customer_msg_length",
                                 "n_escalations", "response_time_first_seconds"],
    # satisfaction_rating: message features are the SIGNAL we want — do NOT drop them
    "damaged_in_transit":       ["refund_reason", "refund_amount", "has_refund", "financial_status"],
    "size_issue":               ["refund_reason", "refund_amount", "has_refund", "financial_status"],
    "landed_cost_per_unit_gbp": ["gross_margin_est"],
    "variant_price":            ["price", "gross_margin_est"],
}

# B2: targets where most rows are structural zeros — filter to meaningful rows before training.
FILTER_MAP = {
    "resolution_time_minutes":      ("has_ticket", 1),
    "damaged_in_transit":           ("has_refund", 1),
    "size_issue":                   ("has_refund", 1),
    "msg_count":                    ("has_ticket", 1),
    "avg_customer_msg_length":      ("has_ticket", 1),
    "n_escalations":                ("has_ticket", 1),
    "response_time_first_seconds":  ("has_ticket", 1),
    "avg_sentiment":                ("has_ticket", 1),
    "min_sentiment":                ("has_ticket", 1),
    "pct_negative_msgs":            ("has_ticket", 1),
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
    orders          = _load_orders()
    line_items      = _load_line_items()
    variants        = _load_variants()
    products        = _load_products()
    customers       = _load_customers()
    po_line_items   = _load_po_line_items()
    refunds         = _load_refunds()
    support_tickets = _load_support_tickets()
    google_ads      = _load_google_ads()
    meta_ads        = _load_meta_ads()

    print("Loading new data sources...")
    discount_codes          = _load_discount_codes()
    supplier_features       = _load_supplier_features()
    inventory_features      = _load_inventory_features()
    email_features          = _load_email_features()
    address_features        = _load_address_features()
    raw_messages             = _parse_support_messages()
    support_message_features = _load_support_message_features(raw_messages)
    sentiment_features       = _load_support_sentiment_features(raw_messages)

    print("Joining tables...")
    df = _build_joined(
        line_items, orders, variants, products,
        customers, po_line_items, refunds, support_tickets,
        google_ads, meta_ads,
        discount_codes, supplier_features, inventory_features,
        email_features, address_features, support_message_features,
        sentiment_features,
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
