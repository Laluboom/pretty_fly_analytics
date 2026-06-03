# Pretty Fly - Hackathon Data Pack

**Wayflyer × Fin AI - Build the Future of E-commerce**
*Wednesday 3 - Friday 5 June 2026, London*

---

## Welcome

You're holding 24 months of data for a fictional London streetwear brand called **Pretty Fly**. They've been selling premium tees, hoodies, sweatpants, caps and trainers since June 2024 - menswear at first, with a small womenswear line that launched in December 2025.

Your job over the next three days is to build them an AI-powered tool that helps them run their business better.

## Pick your path

There's no required direction. Three rough framings - pick one, blend them, or invent your own.

**Customer-facing.** Tools Pretty Fly might put in front of their customers - a smarter storefront, support agents, sizing assistants, post-purchase concierges, returns prediction, personalised journeys, loyalty experiences.

**Operator-focused.** Tools Pretty Fly's team uses internally - inventory and cash forecasting, ad efficiency triage, conversational P&L reporting, margin diagnostics, support automation, anomaly detection.

**Surprise us.** The dataset is rich. If you see a direction that doesn't fit either lane, take it.

The best builds often live at the seam between lanes. But knowing which direction you're pointing keeps the work focused.

## Judging Criteria

- **Execution** - does it work?
- **Business value** - would Pretty Fly pay for this?
- **Demo quality** - is your pitch compelling?

## What's in the box

```
pretty_fly_data_pack/
├── README.md                   ← this file
├── data/
│   ├── products.csv
│   ├── variants.csv
│   ├── collections.csv
│   ├── product_collections.csv
│   ├── customers.csv
│   ├── addresses.csv
│   ├── orders.csv
│   ├── line_items.csv
│   ├── discount_codes.csv
│   ├── refunds.csv
│   ├── suppliers.csv
│   ├── purchase_orders.csv
│   ├── po_line_items.csv
│   ├── inventory_movements.csv
│   ├── google_ads_daily.csv
│   ├── meta_ads_daily.csv
│   ├── email_campaigns.csv
│   ├── email_events.csv
│   ├── support_tickets.csv
│   ├── support_messages.json
│   └── bank_transactions.csv
├── validate.py                 ← reconciliation checks (more on this below)
└── requirements.txt
```

| File | Rows |
|------|------|
| products.csv | 62 |
| variants.csv | 645 |
| collections.csv | 9 |
| product_collections.csv | 62 |
| customers.csv | 22,440 |
| addresses.csv | 22,440 |
| orders.csv | 49,793 |
| line_items.csv | 69,956 |
| discount_codes.csv | 8 |
| refunds.csv | 5,843 |
| suppliers.csv | 5 |
| purchase_orders.csv | 21 |
| po_line_items.csv | 645 |
| inventory_movements.csv | 76,444 |
| google_ads_daily.csv | 5,110 |
| meta_ads_daily.csv | 3,102 |
| email_campaigns.csv | 6 |
| email_events.csv | 11,368 |
| support_tickets.csv | 1,204 |
| support_messages.json | 1,204 conversations |
| bank_transactions.csv | 560 |

Everything reconciles. Bank transactions tie to orders. Ad spend ties to bank. PO payments tie to supplier deliveries. Inventory movements tie to sales. If you compute something and the numbers don't match, the bug is almost certainly yours - but we've shipped a validator so you can prove it (see "Verifying your numbers" below).

## Quick start

```python
import pandas as pd

orders     = pd.read_csv("data/orders.csv", parse_dates=["created_at"])
line_items = pd.read_csv("data/line_items.csv")
products   = pd.read_csv("data/products.csv")
variants   = pd.read_csv("data/variants.csv")

# A typical join: orders → line items → variants → products
sold = (line_items
    .merge(orders[["order_id", "created_at", "customer_id"]], on="order_id")
    .merge(variants[["variant_id", "product_id", "sku"]], on="variant_id")
    .merge(products[["product_id", "title", "product_type", "gender_segment"]], on="product_id"))

# Monthly revenue
sold["month"] = sold["created_at"].dt.to_period("M")
sold.groupby("month").apply(lambda x: (x["price"] * x["quantity"]).sum())
```

## The dataset at a glance

A short tour of each table and how they connect.

**Catalogue: `products`, `variants`, `collections`, `product_collections`**
Products are the abstract item ("Heritage Hoodie"). Variants are the actual sellable units - a hoodie in Washed Black, size M, has its own SKU and inventory level. Products belong to one or more collections (Core, SS25 Drop, Womens, etc.). Each variant has a price and a current inventory quantity.

**Customers: `customers`, `addresses`**
Around 22,400 unique customers. Each row has an acquisition source (the UTM of their first order), a default country, and a derived `gender_segment_affinity` based on their purchase history. Useful when reasoning about cohorts.

**Orders: `orders`, `line_items`, `discount_codes`, `refunds`**
Around 50,000 orders across the 24 months. Each order has subtotals, discounts, shipping, VAT (UK orders include VAT; export orders don't), and UTM attribution. Line items break each order down by variant. Refunds are linked back to orders and have a reason. Discount codes are real Pretty Fly codes that appear on orders.

**Inventory and supply: `suppliers`, `purchase_orders`, `po_line_items`, `inventory_movements`**
Pretty Fly buys from suppliers in Portugal, Italy, and Turkey. Each purchase order has a deposit payment, a balance payment, and a delivery date. Every receipt, sale, and return creates an `inventory_movement` - sum them per variant and you'll get the current stock level. The gap between paying suppliers and earning back revenue is the brand's cash conversion cycle.

**Advertising: `google_ads_daily`, `meta_ads_daily`**
Daily spend, clicks, impressions, conversions and conversion value across both platforms over 24 months. Campaign names match the `utm_campaign` field on orders, so you can tie marketing spend directly to attributed revenue. (Standard caveat: ad platforms overclaim conversions vs. what Shopify actually attributes.)

**Email: `email_campaigns`, `email_events`**
Klaviyo-style. Campaigns are one-offs (a drop announcement); flows are automated (welcome series, abandoned cart). Events show every send, open, click, and conversion at the customer level. Email-attributed revenue is fully reconcilable.

**Support: `support_tickets`, `support_messages`**
Around 1,200 tickets across the 24 months. Channels are email, live chat, and Instagram DM. Messages are realistic conversations between customers, Pretty Fly's human agents, and their existing bot triage. Around 40% of tickets are bot-resolved today; the rest go to humans. Each ticket can link to a customer, an order, or a product.

**Banking: `bank_transactions`**
Every penny in and out of the business over 24 months. Counterparties have realistic-but-messy descriptions (`SHOPIFY* 8829 PAYOUT REF#82910`, `FB*ADS9921XK MENLO PARK`). The `counterparty` field is fully populated with the clean entity name behind each transaction. There's also a `raw_category` field with a coarse classification (`SUPPLIER`, `PAYROLL`, `SAAS`, `RENT`, `MARKETING`, `SHIPPING`, `FULFILMENT`, `PAYOUT`, `REFUND`, `OTHER`) - useful for a fast first-pass P&L. The `category` field is **deliberately blank** - categorising these more finely is a build direction in itself.

## Build inspiration

We're not going to tell you what to build. But here are some questions Pretty Fly's team probably can't answer cleanly today, and that the data should let you tackle:

- Where is cash actually getting tied up? What would it cost to fix?
- Which marketing pounds are working hardest, and which should they pull tomorrow?
- Are there customer segments quietly outperforming - or quietly underperforming - that they should treat differently?
- Which support tickets could a bot resolve end-to-end if it had access to order and product data?
- Which products would they reorder right now if they could see clearly? Which would they walk away from?
- What does their P&L actually look like this month, and how would they know without an accountant?
- If you were Pretty Fly's CFO, what would worry you in this data? If you were their head of growth, what would excite you?
- What does their storefront experience leave on the table that an AI agent could pick up?

**A note before you dive in.** Pretty Fly is a believable, moderately healthy business - they're not in crisis, but they have the same kinds of inefficiencies any real DTC brand carries. Some of these are visible in the data. The most interesting builds will surface insights an operator wouldn't have spotted on their own - and turn those insights into something usable.

**If you can name a specific SKU, a specific date range, a specific customer cohort, or a specific £-figure in your pitch, you're winning.** Specificity is the strongest signal that you've done the work.

## A note on building P&Ls from this data

P&L tools are one of the most natural build paths in this dataset, so a few pointers if that's your direction.

A few P&L lines need more than just bank transactions:

- **COGS is recognised when units are sold, not when suppliers are paid.** To compute it cleanly, multiply units sold (from `line_items`) by their landed cost per unit (from `po_line_items`).
- **Shopify payment processing fees are embedded inside payout amounts** - payouts equal gross orders minus fees and refunds for that batch. Derive the fee by comparing payout amounts to the orders included.
- **Returns affect both the `refunds` table and `bank_transactions`** (via SHOPIFY REFUND BATCH lines). The refunds table is more granular.

The simplest first-pass P&L groups bank transactions by `raw_category` and adjusts for the items above. The more accurate version joins through `orders`, `line_items`, `po_line_items`, and `refunds`. Both are reasonable scopes - pick what fits your time.

## Verifying your numbers

We've shipped a validation script with 20 reconciliation rules. If you're ever unsure whether a number you've computed is right - or whether a discrepancy is your code or the data - run it:

```bash
pip install -r requirements.txt
python validate.py data/
```

It checks order subtotals, customer aggregates, inventory balances, ad spend vs bank, refund timing, UTM consistency across tables, and a dozen other things. All rules should pass on the dataset you've been given. If any fail, tell us - but they shouldn't.

The same script is a useful reference for the relationships between tables. Read it if you're stuck on how something links to something else.

## What's not in here

To save you searching for it:

- **No real customer data.** Every name, email, and address is synthetic. Emails are `firstname.lastname@example-fake.com` style.
- **No live API.** This is a snapshot, not a connection. If you want to simulate one, build it.
- **No payment processor data** beyond what Shopify exposes - no Stripe-level fraud signals, no card-level info.
- **No web analytics.** No GA, no session data, no page views. If your build needs traffic data, you'll need to assume or generate it.
- **No images.** We've described products but not pictured them. Generate or stub as needed.

## Tips for a strong build

A few things we've seen separate the strongest hackathon builds from the rest.

**Spend the first few hours exploring the data.** The strongest builds rest on a specific finding, not a generic idea. Coming back on Day 2 with "we noticed that…" beats coming back with "we built a tool that…" Day 1 is for investigation; Day 2 is for building on what you found.

**Don't try to use every table.** A focused build using three tables well beats a broad build using twelve tables superficially. The dataset is rich precisely so you can pick your slice - not so you have to cover all of it.

**Demos beat slides.** A working interface speaks louder than mockups every time. Spend the time getting the demo working live for Friday rather than polishing static screens.

**AI is a tool, not the product.** Judges aren't impressed by "we used the latest model." They're impressed by what you built *with* it - the prompt design, the grounding, the architecture choices, the user experience.

**Have fun.** This is a fictional brand with real-shaped data. There's no wrong answer. Ship something you'd be proud to demo on Friday.

## Practical tips

- **Timestamps are UTC.** Convert to Europe/London if you care about local-time patterns (Black Friday spikes, drop launches, etc.).
- **Currency is GBP.** Some supplier invoices are in EUR or USD with conversion noted; everything reports in GBP.
- **VAT is inclusive in UK retail prices.** A £50 tee on the site is £50 to a UK customer (with £8.33 VAT inside); a US customer pays £41.67 net.
- **The dataset window is 1 June 2024 → 31 May 2026.** "Today" in this world is the morning of 1 June 2026.
- **Womenswear launched in December 2025**, six months before the dataset ends. Pre-launch, the brand was menswear-only.
- **Discount codes are case-sensitive** in the data as they would be on Shopify.
- **`utm_campaign` is the master key for marketing attribution** - it threads through orders, ads, and email.

## Getting help during the hackathon

Wayflyer folks are on-site all three days and available online to answer questions in Discord: https://discord.gg/gt3MUwRZ7. Find us if you're stuck on the data, want to talk through a build idea, or need a quick reality check on whether something's worth pursuing.

Tools down Friday at 7pm. Pitches start immediately after.

Build something wonderful.

---

*Pretty Fly, Wayflyer, and Fin AI are unrelated entities and the data in this pack is entirely fictional. Any resemblance to real brands, real campaigns, or real customer behaviour is coincidental and probably means we've done our jobs well.*