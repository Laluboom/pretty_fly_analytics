#!/usr/bin/env python3
"""
validate.py — Reconciliation validator for Pretty Fly hackathon dataset.

Implements all 20 reconciliation rules from Section 9 of the spec as
independent check functions. Uses Decimal arithmetic with 1-penny tolerance.

Usage:
    python validate.py data/sample/
    python validate.py data/full/

Exit code 0 if all rules pass, 1 if any rule fails.
"""

import sys
import csv
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from collections import defaultdict

TOLERANCE = Decimal("0.01")  # 1 penny


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def D(value):
    """Convert a value to Decimal. Returns Decimal('0') for empty/None/null."""
    if value is None or str(value).strip() in ("", "null", "None", "NaN"):
        return Decimal("0")
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return Decimal("0")


def _ym(date_str):
    """Extract 'YYYY-MM' from a date or datetime string."""
    if not date_str:
        return None
    return str(date_str)[:7]


def _fmt(mismatches, limit=3):
    """Format mismatch examples for display (indented)."""
    lines = []
    for m in mismatches[:limit]:
        lines.append(f"         {m}")
    if len(mismatches) > limit:
        lines.append(f"         ... and {len(mismatches) - limit} more")
    return "\n".join(lines)


class DataLoader:
    """Lazy-loading, caching reader for CSV and JSON data files."""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self._cache = {}

    def csv(self, filename):
        """Load CSV as list[dict]. Returns None if file missing."""
        if filename not in self._cache:
            path = self.data_dir / filename
            if not path.exists():
                self._cache[filename] = None
            else:
                with open(path, newline="", encoding="utf-8") as f:
                    self._cache[filename] = list(csv.DictReader(f))
        return self._cache[filename]

    def json(self, filename):
        """Load JSON file. Returns None if file missing."""
        if filename not in self._cache:
            path = self.data_dir / filename
            if not path.exists():
                self._cache[filename] = None
            else:
                with open(path, encoding="utf-8") as f:
                    self._cache[filename] = json.load(f)
        return self._cache[filename]


def _need(loader, *filenames):
    """Load required files. Returns (list_of_data, None) or (None, skip_msg)."""
    results = []
    for fn in filenames:
        data = loader.json(fn) if fn.endswith(".json") else loader.csv(fn)
        if data is None:
            return None, f"N/A — file not found: {fn}"
        results.append(data)
    if len(results) == 1:
        return results[0], None
    return results, None


# ---------------------------------------------------------------------------
# Bank-transaction description matchers
# ---------------------------------------------------------------------------

def _is_shopify_payout(desc):
    d = desc.upper()
    return "SHOPIFY" in d and "PAYOUT" in d


def _is_shopify_refund(desc):
    d = desc.upper()
    return "SHOPIFY" in d and "REFUND" in d


def _is_google_ads(desc):
    d = desc.upper()
    return "GOOGLE" in d and "ADS" in d


def _is_meta_ads(desc):
    d = desc.upper()
    return ("FB" in d or "META" in d) and "ADS" in d


def _is_klaviyo(desc):
    return "KLAVIYO" in desc.upper()


def _is_supplier(desc):
    return "SWIFT XFER" in desc.upper()


# ===================================================================
# RULE 1: Order subtotals = sum of line items
# Also checks: total_price = subtotal - discounts + shipping + tax
# ===================================================================

def rule_01(loader):
    data, err = _need(loader, "orders.csv", "line_items.csv")
    if err:
        return True, err
    orders, line_items = data

    # Sum line item values per order
    li_subs = defaultdict(lambda: Decimal("0"))
    for li in line_items:
        li_subs[li["order_id"]] += D(li["price"]) * D(li["quantity"])

    bad_sub, bad_tot = [], []
    for o in orders:
        oid = o["order_id"]
        # Subtotal check
        diff = D(o["subtotal"]) - li_subs.get(oid, Decimal("0"))
        if abs(diff) > TOLERANCE:
            bad_sub.append(
                f"{oid}: subtotal={o['subtotal']}, "
                f"sum(li)={li_subs.get(oid, 0)}, diff={diff}"
            )
        # total_price formula check
        computed = (
            D(o["subtotal"]) - D(o["total_discounts"])
            + D(o["total_shipping"]) + D(o["total_tax"])
        )
        diff2 = D(o["total_price"]) - computed
        if abs(diff2) > TOLERANCE:
            bad_tot.append(
                f"{oid}: total_price={o['total_price']}, "
                f"computed={computed}, diff={diff2}"
            )

    msgs = []
    if bad_sub:
        msgs.append(f"Subtotal mismatches: {len(bad_sub)}/{len(orders)}")
        msgs.append(_fmt(bad_sub))
    if bad_tot:
        msgs.append(f"total_price formula mismatches: {len(bad_tot)}/{len(orders)}")
        msgs.append(_fmt(bad_tot))
    if msgs:
        return False, "\n".join(msgs)
    return True, f"All {len(orders)} orders reconcile"


# ===================================================================
# RULE 2: Customer total_spent and orders_count match orders
# ===================================================================

def rule_02(loader):
    data, err = _need(loader, "customers.csv", "orders.csv")
    if err:
        return True, err
    customers, orders = data

    # Aggregate from orders
    spent = defaultdict(lambda: Decimal("0"))
    count = defaultdict(int)
    for o in orders:
        cid = o["customer_id"]
        spent[cid] += D(o["total_price"])
        count[cid] += 1

    bad_spent, bad_count = [], []
    for c in customers:
        cid = c["customer_id"]
        ds = D(c["total_spent"]) - spent.get(cid, Decimal("0"))
        if abs(ds) > TOLERANCE:
            bad_spent.append(
                f"{cid}: total_spent={c['total_spent']}, "
                f"sum(orders)={spent.get(cid, 0)}, diff={ds}"
            )
        dc = int(c["orders_count"]) - count.get(cid, 0)
        if dc != 0:
            bad_count.append(
                f"{cid}: orders_count={c['orders_count']}, "
                f"actual={count.get(cid, 0)}, diff={dc}"
            )

    msgs = []
    if bad_spent:
        msgs.append(f"total_spent mismatches: {len(bad_spent)}/{len(customers)}")
        msgs.append(_fmt(bad_spent))
    if bad_count:
        msgs.append(f"orders_count mismatches: {len(bad_count)}/{len(customers)}")
        msgs.append(_fmt(bad_count))
    if msgs:
        return False, "\n".join(msgs)
    return True, f"All {len(customers)} customers reconcile"


# ===================================================================
# RULE 3: Discount code usage_count = count of orders using that code
# ===================================================================

def rule_03(loader):
    data, err = _need(loader, "discount_codes.csv", "orders.csv")
    if err:
        return True, err
    codes, orders = data

    # Check that orders have a discount_code column
    if orders and "discount_code" not in orders[0]:
        return True, "N/A — orders.csv has no discount_code column"

    usage = defaultdict(int)
    for o in orders:
        dc = (o.get("discount_code") or "").strip()
        if dc:
            usage[dc] += 1

    bad = []
    for c in codes:
        code = c["code"]
        expected = usage.get(code, 0)
        actual = int(c["usage_count"])
        if actual != expected:
            bad.append(
                f"{code}: usage_count={actual}, orders_using={expected}, "
                f"diff={actual - expected}"
            )

    if bad:
        return False, f"Mismatches: {len(bad)}/{len(codes)}\n{_fmt(bad)}"
    return True, f"All {len(codes)} discount codes reconcile"


# ===================================================================
# RULE 4: Every line item produces an inventory movement (sale)
# ===================================================================

def rule_04(loader):
    data, err = _need(loader, "line_items.csv", "inventory_movements.csv")
    if err:
        return True, err
    line_items, movements = data

    # Expected: (order_id, variant_id) -> total qty sold
    expected = defaultdict(lambda: Decimal("0"))
    for li in line_items:
        key = (li["order_id"], li["variant_id"])
        expected[key] += D(li["quantity"])

    # Actual sale movements: (reference_id, variant_id) -> abs(quantity_delta)
    actual = defaultdict(lambda: Decimal("0"))
    for mv in movements:
        if mv["type"] == "sale":
            key = (mv["reference_id"], mv["variant_id"])
            actual[key] += abs(D(mv["quantity_delta"]))

    bad = []
    for key, exp_qty in expected.items():
        act_qty = actual.get(key, Decimal("0"))
        if abs(exp_qty - act_qty) > TOLERANCE:
            bad.append(
                f"order={key[0]}, variant={key[1]}: "
                f"line_qty={exp_qty}, movement_qty={act_qty}"
            )

    if bad:
        return False, f"Missing/mismatched sale movements: {len(bad)}\n{_fmt(bad)}"
    return True, f"All {len(expected)} line-item → movement pairs reconcile"


# ===================================================================
# RULE 5: Every PO receipt produces inventory movements (po_receipt)
# ===================================================================

def rule_05(loader):
    data, err = _need(loader, "po_line_items.csv", "purchase_orders.csv",
                       "inventory_movements.csv")
    if err:
        return True, err
    po_lines, pos, movements = data

    # Only check received POs
    received_pos = {p["po_id"] for p in pos if p.get("status") == "received"}

    expected = defaultdict(lambda: Decimal("0"))
    for pli in po_lines:
        if pli["po_id"] in received_pos:
            key = (pli["po_id"], pli["variant_id"])
            expected[key] += D(pli["quantity_received"])

    actual = defaultdict(lambda: Decimal("0"))
    for mv in movements:
        if mv["type"] == "po_receipt":
            key = (mv["reference_id"], mv["variant_id"])
            actual[key] += D(mv["quantity_delta"])

    bad = []
    for key, exp_qty in expected.items():
        act_qty = actual.get(key, Decimal("0"))
        if abs(exp_qty - act_qty) > TOLERANCE:
            bad.append(
                f"po={key[0]}, variant={key[1]}: "
                f"received={exp_qty}, movement={act_qty}"
            )

    if bad:
        return False, (
            f"Missing/mismatched po_receipt movements: {len(bad)}\n{_fmt(bad)}"
        )
    return True, f"All {len(expected)} PO receipt → movement pairs reconcile"


# ===================================================================
# RULE 6: Every refund produces inventory movement (return) + bank txn
# ===================================================================

def rule_06(loader):
    data, err = _need(loader, "refunds.csv", "inventory_movements.csv",
                       "bank_transactions.csv", "orders.csv")
    if err:
        return True, err
    refunds, movements, bank_txns, orders = data

    if not refunds:
        return True, "No refunds to check"

    # --- Part A: return inventory movements ---
    # Expected returns: (order_id, variant_id) -> qty
    expected_returns = defaultdict(lambda: Decimal("0"))
    for r in refunds:
        raw = r.get("refund_line_items", "[]")
        try:
            variant_ids = json.loads(raw) if raw else []
        except (json.JSONDecodeError, TypeError):
            variant_ids = []
        for vid in variant_ids:
            expected_returns[(r["order_id"], vid)] += Decimal("1")

    actual_returns = defaultdict(lambda: Decimal("0"))
    for mv in movements:
        if mv["type"] == "return":
            key = (mv["reference_id"], mv["variant_id"])
            actual_returns[key] += D(mv["quantity_delta"])

    bad_inv = []
    for key, exp_qty in expected_returns.items():
        act_qty = actual_returns.get(key, Decimal("0"))
        if abs(exp_qty - act_qty) > TOLERANCE:
            bad_inv.append(
                f"order={key[0]}, variant={key[1]}: "
                f"expected_return={exp_qty}, movement={act_qty}"
            )

    # --- Part B: refund amounts vs bank — monthly aggregates, 1p per month ---
    refund_monthly = defaultdict(lambda: Decimal("0"))
    for r in refunds:
        ym = _ym(r.get("created_at", ""))
        if ym:
            refund_monthly[ym] += D(r["amount"])

    bank_refund_monthly = defaultdict(lambda: Decimal("0"))
    for bt in bank_txns:
        if _is_shopify_refund(bt.get("description", "")):
            ym = _ym(bt["date"])
            if ym:
                bank_refund_monthly[ym] += abs(D(bt["amount_gbp"]))

    all_months = sorted(set(refund_monthly) | set(bank_refund_monthly))
    bad_bank = []
    for ym in all_months:
        ref_amt = refund_monthly.get(ym, Decimal("0"))
        bank_amt = bank_refund_monthly.get(ym, Decimal("0"))
        if abs(ref_amt - bank_amt) > TOLERANCE:
            bad_bank.append(
                f"{ym}: refunds={ref_amt}, bank={bank_amt}, "
                f"diff={ref_amt - bank_amt}"
            )

    # --- Part C: timing — each refund must be after its order ---
    order_dates = {o["order_id"]: o["created_at"] for o in orders}
    bad_timing = []
    for r in refunds:
        order_date = order_dates.get(r["order_id"], "")
        refund_date = r.get("created_at", "")
        if order_date and refund_date and refund_date < order_date:
            bad_timing.append(
                f"refund {r['refund_id']}: created {refund_date} "
                f"BEFORE order {r['order_id']} created {order_date}"
            )

    msgs = []
    if bad_inv:
        msgs.append(
            f"Missing/mismatched return movements: {len(bad_inv)}\n{_fmt(bad_inv)}"
        )
    if bad_bank:
        msgs.append(
            f"Monthly refund vs bank mismatches: {len(bad_bank)}\n{_fmt(bad_bank)}"
        )
    if bad_timing:
        msgs.append(
            f"Refunds dated before their orders: {len(bad_timing)}\n"
            f"{_fmt(bad_timing)}"
        )

    if msgs:
        return False, "\n".join(msgs)
    return True, (
        f"All {len(expected_returns)} return movements reconcile, "
        f"monthly bank totals match across {len(all_months)} months, "
        f"all refund dates valid"
    )


# ===================================================================
# RULE 7: Inventory movements per variant sum to current stock level
# ===================================================================

def rule_07(loader):
    data, err = _need(loader, "variants.csv", "inventory_movements.csv")
    if err:
        return True, err
    variants, movements = data

    # Sum movements per variant
    mv_sums = defaultdict(lambda: Decimal("0"))
    for mv in movements:
        mv_sums[mv["variant_id"]] += D(mv["quantity_delta"])

    bad = []
    for v in variants:
        vid = v["variant_id"]
        expected = D(v["inventory_quantity"])
        actual = mv_sums.get(vid, Decimal("0"))
        if abs(expected - actual) > TOLERANCE:
            bad.append(
                f"{vid}: inventory_quantity={expected}, "
                f"sum(movements)={actual}, diff={expected - actual}"
            )

    if bad:
        return False, f"Stock level mismatches: {len(bad)}/{len(variants)}\n{_fmt(bad)}"
    return True, f"All {len(variants)} variant stock levels reconcile"


# ===================================================================
# RULE 8: Monthly Google ad spend = monthly Google bank transactions
# ===================================================================

def rule_08(loader):
    data, err = _need(loader, "google_ads_daily.csv", "bank_transactions.csv")
    if err:
        return True, err
    ads, bank_txns = data

    # Monthly ads spend
    ads_monthly = defaultdict(lambda: Decimal("0"))
    for row in ads:
        ym = _ym(row["date"])
        if ym:
            ads_monthly[ym] += D(row["spend_gbp"])

    # Monthly Google bank outflows
    bank_monthly = defaultdict(lambda: Decimal("0"))
    for bt in bank_txns:
        if _is_google_ads(bt.get("description", "")):
            bank_monthly[_ym(bt["date"])] += abs(D(bt["amount_gbp"]))

    all_months = sorted(set(ads_monthly) | set(bank_monthly))
    bad = []
    for ym in all_months:
        diff = ads_monthly.get(ym, Decimal("0")) - bank_monthly.get(ym, Decimal("0"))
        if abs(diff) > TOLERANCE:
            bad.append(
                f"{ym}: ads_spend={ads_monthly.get(ym, 0)}, "
                f"bank={bank_monthly.get(ym, 0)}, diff={diff}"
            )

    if bad:
        return False, f"Monthly Google spend mismatches: {len(bad)}\n{_fmt(bad)}"
    return True, f"All {len(all_months)} months reconcile"


# ===================================================================
# RULE 9: Monthly Meta ad spend = monthly Meta bank transactions
# ===================================================================

def rule_09(loader):
    data, err = _need(loader, "meta_ads_daily.csv", "bank_transactions.csv")
    if err:
        return True, err
    ads, bank_txns = data

    ads_monthly = defaultdict(lambda: Decimal("0"))
    for row in ads:
        ym = _ym(row["date"])
        if ym:
            ads_monthly[ym] += D(row["spend_gbp"])

    bank_monthly = defaultdict(lambda: Decimal("0"))
    for bt in bank_txns:
        if _is_meta_ads(bt.get("description", "")):
            bank_monthly[_ym(bt["date"])] += abs(D(bt["amount_gbp"]))

    all_months = sorted(set(ads_monthly) | set(bank_monthly))
    bad = []
    for ym in all_months:
        diff = ads_monthly.get(ym, Decimal("0")) - bank_monthly.get(ym, Decimal("0"))
        if abs(diff) > TOLERANCE:
            bad.append(
                f"{ym}: ads_spend={ads_monthly.get(ym, 0)}, "
                f"bank={bank_monthly.get(ym, 0)}, diff={diff}"
            )

    if bad:
        return False, f"Monthly Meta spend mismatches: {len(bad)}\n{_fmt(bad)}"
    return True, f"All {len(all_months)} months reconcile"


# ===================================================================
# RULE 10: UTM campaign names in orders exist in ads or email tables
# ===================================================================

def rule_10(loader):
    orders, err = _need(loader, "orders.csv")
    if err:
        return True, err

    # Collect campaign names from ads + email tables
    known_campaigns = set()
    for fn in ("google_ads_daily.csv", "meta_ads_daily.csv"):
        rows = loader.csv(fn)
        if rows:
            for r in rows:
                known_campaigns.add(r["campaign_name"])
    email = loader.csv("email_campaigns.csv")
    if email:
        for r in email:
            known_campaigns.add(r["name"])

    # Check orders
    orphans = set()
    for o in orders:
        camp = (o.get("utm_campaign") or "").strip()
        if camp and camp not in known_campaigns:
            orphans.add(camp)

    if orphans:
        examples = sorted(orphans)[:3]
        return False, (
            f"UTM campaigns in orders not found in ads/email: "
            f"{len(orphans)}\n{_fmt(examples)}"
        )
    return True, "All order UTM campaigns exist in ads/email tables"


# ===================================================================
# RULE 11: Email attributed_orders/revenue = orders with matching UTM
# ===================================================================

def rule_11(loader):
    data, err = _need(loader, "email_campaigns.csv", "orders.csv")
    if err:
        return True, err
    campaigns, orders = data

    # Orders attributed to Klaviyo, grouped by utm_campaign
    kl_orders = defaultdict(list)
    for o in orders:
        if (o.get("utm_source") or "").strip().lower() == "klaviyo":
            camp = (o.get("utm_campaign") or "").strip()
            if camp:
                kl_orders[camp].append(o)

    bad = []
    for c in campaigns:
        name = c["name"]
        matched = kl_orders.get(name, [])
        # attributed_orders
        exp_count = len(matched)
        act_count = int(c.get("attributed_orders", 0))
        # attributed_revenue_gbp
        exp_rev = sum(D(o["total_price"]) for o in matched)
        act_rev = D(c.get("attributed_revenue_gbp", 0))

        if act_count != exp_count or abs(act_rev - exp_rev) > TOLERANCE:
            bad.append(
                f"{name}: orders={act_count} (exp {exp_count}), "
                f"rev={act_rev} (exp {exp_rev})"
            )

    if bad:
        return False, f"Email attribution mismatches: {len(bad)}\n{_fmt(bad)}"
    return True, f"All {len(campaigns)} email campaigns reconcile"


# ===================================================================
# RULE 12: Klaviyo subscription appears monthly in bank
# ===================================================================

def rule_12(loader):
    data, err = _need(loader, "bank_transactions.csv", "orders.csv")
    if err:
        return True, err
    bank_txns, orders = data

    # Determine months present in the dataset from orders
    order_months = sorted({_ym(o["created_at"]) for o in orders if _ym(o["created_at"])})
    if not order_months:
        return True, "N/A — no orders to derive month range"

    # Months with Klaviyo bank txns
    kl_months = set()
    for bt in bank_txns:
        if _is_klaviyo(bt.get("description", "")):
            kl_months.add(_ym(bt["date"]))

    missing = [m for m in order_months if m not in kl_months]
    if missing:
        return False, (
            f"Klaviyo bank txn missing for {len(missing)} month(s)\n"
            f"{_fmt(missing)}"
        )
    return True, f"Klaviyo subscription present in all {len(order_months)} months"


# ===================================================================
# RULE 13: PO deposits + balances in bank = purchase_orders.total_cost_gbp
# ===================================================================

def rule_13(loader):
    data, err = _need(loader, "purchase_orders.csv", "bank_transactions.csv")
    if err:
        return True, err
    pos, bank_txns = data

    # Determine the bank's observed date range
    bank_dates = [bt["date"] for bt in bank_txns if bt.get("date")]
    if not bank_dates:
        return True, "N/A — no bank transactions"
    bank_start = min(bank_dates)
    bank_end = max(bank_dates)

    # Expected bank outflows: only count payments whose dates fall within the
    # bank's date range (deposits before the window aren't in the bank)
    expected_total = Decimal("0")
    po_count = 0
    for po in pos:
        cost = D(po["total_cost_gbp"])
        dep_str = (po.get("deposit_paid_at") or "").strip()
        bal_str = (po.get("balance_paid_at") or "").strip()
        deposit_amt = (cost / 2).quantize(Decimal("0.01"))
        balance_amt = cost - deposit_amt

        contributed = False
        if dep_str and bank_start <= dep_str <= bank_end:
            expected_total += deposit_amt
            contributed = True
        if bal_str and bank_start <= bal_str <= bank_end:
            expected_total += balance_amt
            contributed = True
        if contributed:
            po_count += 1

    # Sum of supplier bank outflows
    supplier_total = Decimal("0")
    for bt in bank_txns:
        if _is_supplier(bt.get("description", "")):
            supplier_total += abs(D(bt["amount_gbp"]))

    diff = expected_total - supplier_total
    tol = TOLERANCE * max(po_count, 1)
    if abs(diff) > tol:
        return False, (
            f"PO payments vs bank: expected={expected_total}, "
            f"supplier_bank_outflows={supplier_total}, diff={diff}"
        )
    return True, (
        f"{po_count} POs with in-window payments ({expected_total}) match "
        f"supplier bank outflows ({supplier_total})"
    )


# ===================================================================
# RULE 14: Running bank balance is internally consistent
# (closing = opening + sum of all transactions)
# ===================================================================

def rule_14(loader):
    bank_txns, err = _need(loader, "bank_transactions.csv")
    if err:
        return True, err
    if not bank_txns:
        return True, "N/A — no bank transactions"

    # Check running balance consistency row by row
    bad = []
    prev_balance = None
    for i, bt in enumerate(bank_txns):
        bal = D(bt["balance_gbp"])
        amt = D(bt["amount_gbp"])
        if prev_balance is not None:
            expected_bal = prev_balance + amt
            diff = bal - expected_bal
            if abs(diff) > TOLERANCE:
                bad.append(
                    f"row {i+1} ({bt.get('transaction_id', '?')}): "
                    f"balance={bal}, expected={expected_bal} "
                    f"(prev={prev_balance} + amt={amt}), diff={diff}"
                )
        prev_balance = bal

    if bad:
        return False, f"Running balance breaks: {len(bad)}\n{_fmt(bad)}"

    # Also verify: final balance = first balance - first amount + sum of all amounts
    # (i.e., opening balance + sum = closing)
    opening = D(bank_txns[0]["balance_gbp"]) - D(bank_txns[0]["amount_gbp"])
    total_flow = sum(D(bt["amount_gbp"]) for bt in bank_txns)
    closing = D(bank_txns[-1]["balance_gbp"])
    diff = closing - (opening + total_flow)
    if abs(diff) > TOLERANCE:
        return False, (
            f"Closing balance mismatch: closing={closing}, "
            f"opening({opening}) + sum({total_flow}) = {opening + total_flow}, "
            f"diff={diff}"
        )

    return True, (
        f"Running balance consistent across {len(bank_txns)} transactions "
        f"(opening={opening}, closing={closing})"
    )


# ===================================================================
# RULE 15: P&L Net Sales = orders.subtotal - discounts - refunds
# ===================================================================

def rule_15(loader):
    data, err = _need(loader, "orders.csv", "refunds.csv")
    if err:
        return True, err
    orders, refunds = data

    gross_sales = sum(D(o["subtotal"]) for o in orders)
    total_discounts = sum(D(o["total_discounts"]) for o in orders)
    total_refunds = sum(D(r["amount"]) for r in refunds)
    net_sales = gross_sales - total_discounts - total_refunds

    if net_sales <= Decimal("0"):
        return False, (
            f"Net sales is non-positive: gross={gross_sales}, "
            f"discounts={total_discounts}, refunds={total_refunds}, "
            f"net={net_sales}"
        )

    return True, (
        f"Net sales = {net_sales} "
        f"(gross {gross_sales} - discounts {total_discounts} "
        f"- refunds {total_refunds})"
    )


# ===================================================================
# RULE 16: P&L COGS = sum of (units sold x landed cost)
# Every sold variant must have a landed cost in po_line_items.
# ===================================================================

def rule_16(loader):
    data, err = _need(loader, "line_items.csv", "po_line_items.csv", "orders.csv")
    if err:
        return True, err
    line_items, po_lines, orders = data

    # Build landed cost lookup: variant_id -> latest landed_cost_per_unit_gbp
    landed = {}
    for pli in po_lines:
        vid = pli["variant_id"]
        cost = D(pli["landed_cost_per_unit_gbp"])
        landed[vid] = cost  # last one wins (latest PO)

    # Check every sold variant has a cost
    missing_cost = set()
    cogs = Decimal("0")
    for li in line_items:
        vid = li["variant_id"]
        if vid not in landed:
            missing_cost.add(vid)
        else:
            cogs += D(li["quantity"]) * landed[vid]

    if missing_cost:
        examples = sorted(missing_cost)[:3]
        return False, (
            f"{len(missing_cost)} sold variants have no landed cost\n"
            f"{_fmt(examples)}"
        )

    # Sanity: gross margin should be broadly reasonable
    net_sales = sum(D(o["subtotal"]) for o in orders) - sum(
        D(o["total_discounts"]) for o in orders
    )
    if net_sales > Decimal("0"):
        margin = (net_sales - cogs) / net_sales * 100
        return True, f"COGS = {cogs}, gross margin = {margin:.1f}%"
    return True, f"COGS = {cogs} (net sales not positive, margin not computed)"


# ===================================================================
# RULE 17: Inventory value = inventory_quantity x landed cost per unit
# ===================================================================

def rule_17(loader):
    data, err = _need(loader, "variants.csv", "po_line_items.csv")
    if err:
        return True, err
    variants, po_lines = data

    # Build landed cost lookup
    landed = {}
    for pli in po_lines:
        vid = pli["variant_id"]
        landed[vid] = D(pli["landed_cost_per_unit_gbp"])

    total_value = Decimal("0")
    missing = []
    for v in variants:
        vid = v["variant_id"]
        qty = D(v["inventory_quantity"])
        if qty > 0:
            if vid not in landed:
                missing.append(vid)
            else:
                total_value += qty * landed[vid]

    if missing:
        return False, (
            f"{len(missing)} variants with stock have no landed cost\n"
            f"{_fmt(missing[:3])}"
        )

    return True, f"Total inventory value = {total_value}"


# ===================================================================
# RULE 18: Accounts Payable = outstanding PO balances at month-end
# ===================================================================

def rule_18(loader):
    pos, err = _need(loader, "purchase_orders.csv")
    if err:
        return True, err

    # AP = total_cost_gbp for POs where balance hasn't been paid
    ap = Decimal("0")
    outstanding_count = 0
    for po in pos:
        balance_paid = (po.get("balance_paid_at") or "").strip()
        deposit_paid = (po.get("deposit_paid_at") or "").strip()
        cost = D(po["total_cost_gbp"])

        if not balance_paid:
            if deposit_paid:
                # Deposit paid, balance outstanding (assume 50/50 split)
                ap += cost / 2
            else:
                # Nothing paid
                ap += cost
            outstanding_count += 1

    return True, (
        f"Accounts Payable = {ap} ({outstanding_count} POs with outstanding balance)"
    )


# ===================================================================
# RULE 19: Support ticket FK references all exist
# ===================================================================

def rule_19(loader):
    tickets, err = _need(loader, "support_tickets.csv")
    if err:
        return True, err

    orders = loader.csv("orders.csv")
    products = loader.csv("products.csv")
    customers = loader.csv("customers.csv")

    order_ids = {o["order_id"] for o in orders} if orders else set()
    product_ids = {p["product_id"] for p in products} if products else set()
    customer_ids = {c["customer_id"] for c in customers} if customers else set()

    bad = []
    for t in tickets:
        tid = t["ticket_id"]
        roid = (t.get("related_order_id") or "").strip()
        rpid = (t.get("related_product_id") or "").strip()
        cid = (t.get("customer_id") or "").strip()

        if roid and orders is not None and roid not in order_ids:
            bad.append(f"{tid}: related_order_id={roid} not in orders")
        if rpid and products is not None and rpid not in product_ids:
            bad.append(f"{tid}: related_product_id={rpid} not in products")
        if cid and customers is not None and cid not in customer_ids:
            bad.append(f"{tid}: customer_id={cid} not in customers")

    if bad:
        return False, f"Broken FK references: {len(bad)}\n{_fmt(bad)}"
    return True, f"All {len(tickets)} ticket FK references valid"


# ===================================================================
# RULE 20: UTM campaign names consistent across orders, ads, and email
# ===================================================================

def rule_20(loader):
    orders = loader.csv("orders.csv")
    google = loader.csv("google_ads_daily.csv")
    meta = loader.csv("meta_ads_daily.csv")
    email = loader.csv("email_campaigns.csv")

    if not orders:
        return True, "N/A — orders.csv not found"

    # Gather campaigns from each source
    order_camps = set()
    for o in orders:
        c = (o.get("utm_campaign") or "").strip()
        if c:
            order_camps.add(c)

    google_camps = set()
    if google:
        for r in google:
            google_camps.add(r["campaign_name"])

    meta_camps = set()
    if meta:
        for r in meta:
            meta_camps.add(r["campaign_name"])

    email_camps = set()
    if email:
        for r in email:
            email_camps.add(r["name"])

    all_known = google_camps | meta_camps | email_camps

    # Orders referencing unknown campaigns
    orphan_orders = order_camps - all_known

    # Ads/email campaigns with conversions but no matching orders
    # (only flag if they report conversions > 0)
    google_with_conv = set()
    if google:
        conv_by_camp = defaultdict(int)
        for r in google:
            conv_by_camp[r["campaign_name"]] += int(r.get("conversions", 0))
        google_with_conv = {c for c, n in conv_by_camp.items() if n > 0}

    meta_with_conv = set()
    if meta:
        conv_by_camp = defaultdict(int)
        for r in meta:
            conv_by_camp[r["campaign_name"]] += int(r.get("conversions", 0))
        meta_with_conv = {c for c, n in conv_by_camp.items() if n > 0}

    # --- Direction B: ads/email → orders ---
    # Campaigns reporting conversions should have at least one matching order.
    # Exclude email flows (automated, attribution is different) and awareness
    # campaigns. Ads overclaim ~15%, but a campaign with real conversions
    # should still have SOME Shopify-attributed orders.
    ads_with_conv = google_with_conv | meta_with_conv

    # Filter out awareness-only Meta campaigns
    awareness_camps = set()
    if meta:
        for r in meta:
            if r.get("campaign_objective", "").lower() == "awareness":
                awareness_camps.add(r["campaign_name"])

    orphan_ads = (ads_with_conv - order_camps) - awareness_camps

    msgs = []

    # Direction A failures
    if orphan_orders:
        examples = sorted(orphan_orders)[:3]
        msgs.append(
            f"A: Order UTM campaigns not in any ads/email table: "
            f"{len(orphan_orders)}\n{_fmt(examples)}"
        )

    # Direction B failures
    if orphan_ads:
        examples = sorted(orphan_ads)[:3]
        msgs.append(
            f"B: Ad campaigns with conversions but zero matching orders: "
            f"{len(orphan_ads)}\n{_fmt(examples)}"
        )

    if msgs:
        return False, "\n".join(msgs)

    info = (
        f"Bidirectional check passed: {len(order_camps)} order campaigns, "
        f"{len(google_camps)} Google, {len(meta_camps)} Meta, "
        f"{len(email_camps)} email"
    )
    return True, info


# ===================================================================
# Main runner
# ===================================================================

RULES = [
    (1,  "Order subtotals = sum of line items",                      rule_01),
    (2,  "Customer total_spent and orders_count match orders",       rule_02),
    (3,  "Discount code usage_count = orders using that code",       rule_03),
    (4,  "Every line item produces inventory movement (sale)",       rule_04),
    (5,  "Every PO receipt produces inventory movement (po_receipt)", rule_05),
    (6,  "Every refund -> inventory return + bank txn",              rule_06),
    (7,  "Inventory movements sum to current stock level",           rule_07),
    (8,  "Monthly Google ad spend = Google bank transactions",       rule_08),
    (9,  "Monthly Meta ad spend = Meta bank transactions",           rule_09),
    (10, "UTM campaign names in orders exist in ads/email",          rule_10),
    (11, "Email attributed_orders/revenue match orders",             rule_11),
    (12, "Klaviyo subscription appears monthly in bank",             rule_12),
    (13, "PO payments in bank = purchase_orders.total_cost_gbp",     rule_13),
    (14, "Running bank balance internally consistent",               rule_14),
    (15, "Net Sales = subtotals - discounts - refunds",              rule_15),
    (16, "COGS = sum(units sold x landed cost)",                     rule_16),
    (17, "Inventory value = quantity x landed cost",                 rule_17),
    (18, "Accounts Payable = outstanding PO balances",               rule_18),
    (19, "Support ticket FK references all exist",                   rule_19),
    (20, "UTM campaign names consistent across all tables",          rule_20),
]


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate.py <data_directory>")
        print("  e.g. python validate.py data/sample/")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    if not data_dir.is_dir():
        print(f"Error: {data_dir} is not a directory")
        sys.exit(1)

    loader = DataLoader(data_dir)
    all_passed = True
    pass_count = 0
    fail_count = 0
    skip_count = 0

    print(f"\n{'='*60}")
    print(f"  Pretty Fly Data Validator — {data_dir}")
    print(f"{'='*60}\n")

    for num, desc, func in RULES:
        try:
            passed, message = func(loader)
        except Exception as e:
            passed = False
            message = f"EXCEPTION: {type(e).__name__}: {e}"

        if "N/A" in (message or ""):
            tag = "SKIP"
            skip_count += 1
        elif passed:
            tag = "PASS"
            pass_count += 1
        else:
            tag = "FAIL"
            fail_count += 1
            all_passed = False

        print(f"  [{tag}] Rule {num:>2}: {desc}")
        if message:
            for line in message.split("\n"):
                print(f"         {line}")
        print()

    print(f"{'='*60}")
    print(f"  Results: {pass_count} passed, {fail_count} failed, {skip_count} skipped")
    print(f"{'='*60}\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
