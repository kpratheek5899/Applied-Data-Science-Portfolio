# Nova Retail Economy Specification

## Project Overview

Nova Retail is a fictional omnichannel retailer designed to simulate a modern digital-first retail business.

The objective of this project is to build a decision science platform capable of:

* Simulating realistic retail demand
* Estimating demand elasticity
* Recovering true economic parameters
* Optimizing product prices
* Generating business recommendations under multiple objectives

The project serves as the foundation for:

* Econometric Modeling
* Bayesian Inference
* Pricing Optimization
* Scenario Simulation
* Decision Intelligence

---

# Business Objective

The platform recommends product prices based on selected business goals.

## Decision Variable

Only one variable is optimized:

```text
Price
```

All other business variables remain fixed during optimization.

---

# Optimization Goals

## Goal 1: Maximize Profit

Objective:

```text
Profit = Revenue - Product Cost
```

Recommendation seeks the price producing the highest expected profit.

---

## Goal 2: Maximize Revenue

Objective:

```text
Revenue = Price × Units
```

Recommendation seeks the price producing the highest expected revenue.

---

## Goal 3: Protect Inventory

Objective:

Maintain healthy inventory levels while minimizing stockout risk.

Recommendation seeks the price that preserves inventory while maintaining reasonable financial performance.

---

# Retail Structure

## Regions

* West
* Central
* East

## Stores

20 stores distributed across regions.

## Sales Channels

* Web
* Mobile App
* Buy Online Pickup In Store (BOPIS)
* Physical Store

## Product Categories

* Electronics
* Fitness
* Home
* Outdoor

## Total Products

50 SKUs

---

# Product Types

Each SKU belongs to one of the following product types.

| Product Type    | Price Elasticity | Promo Sensitivity |
| --------------- | ---------------- | ----------------- |
| Premium         | -0.8             | Low               |
| Commodity       | -2.0             | Medium            |
| Seasonal        | -1.2             | High              |
| Promo Sensitive | -1.8             | Very High         |

## Interpretation

Example:

Elasticity = -1.5

```text
1% increase in price
→ 1.5% decrease in units
```

---

# Marketing Channels

The business invests in four marketing channels.

| Channel     | Elasticity |
| ----------- | ---------- |
| Paid Search | 0.20       |
| Paid Social | 0.08       |
| Display     | 0.05       |
| Email       | 0.15       |

## Interpretation

Example:

Search Elasticity = 0.20

```text
10% increase in search spend
→ 2% increase in units
```

---

# Retail Calendar

The economy contains major retail events.

## Memorial Day

Window:

```text
7 days before
Holiday
7 days after
```

Demand Multiplier:

```text
1.25x
```

Marketing Multiplier:

```text
1.50x
```

---

## Independence Day

Demand Multiplier:

```text
1.20x
```

Marketing Multiplier:

```text
1.40x
```

---

## Labor Day

Demand Multiplier:

```text
1.30x
```

Marketing Multiplier:

```text
1.60x
```

---

## Black Friday

Demand Multiplier:

```text
2.50x
```

Marketing Multiplier:

```text
3.00x
```

Inventory Stress Multiplier:

```text
2.00x
```

---

## Cyber Monday

Demand Multiplier:

```text
2.00x
```

Marketing Multiplier:

```text
2.50x
```

---

## Holiday Season

December 1 – December 24

Demand Multiplier:

```text
1.75x
```

Marketing Multiplier:

```text
2.00x
```

---

# Demand Pull-Forward Logic

Major events create temporary demand distortions.

## Pre-Event Window

14 days before event.

Demand Multiplier:

```text
0.90x
```

Consumers delay purchases while waiting for promotions.

---

## Event Window

Demand spikes.

Multiplier depends on event.

---

## Post-Event Hangover

14 days after event.

Demand Multiplier:

```text
0.85x
```

Demand has already been consumed.

---

# Promotion Rules

Promotion depth may take values:

```text
0%
5%
10%
15%
20%
```

Promotion lift is applied on top of elasticity effects.

Example:

```text
10% Promotion

→ +18% Demand Lift
```

---

# Inventory System

Inventory behaves dynamically.

## Weekly Replenishment

Stores receive inventory every 7 days.

Target inventory:

```text
80% of maximum capacity
```

---

# Stock Status Rules

| Inventory Remaining | Status               |
| ------------------- | -------------------- |
| >40%                | In Stock             |
| 15% - 40%           | Low Stock            |
| 1% - 15%            | Limited Availability |
| 0%                  | Out Of Stock         |

---

# Digital Experience Effects

Stock visibility impacts conversion.

## Customer Messages

* Available Today
* In Stock
* Only A Few Left
* Limited Availability
* Out Of Stock

Inventory messaging influences conversion rate.

---

# Demand Generation Equation

Conceptual demand model:

Demand =

Base Demand

× Price Effect

× Promotion Effect

× Marketing Effect

× Holiday Effect

× Inventory Visibility Effect

× Random Noise

---

# Data Schema

## Time

* date
* year
* month
* week
* day_of_week

---

## Business Structure

* region
* store_id
* channel
* category
* sku
* product_type

---

## Commercial Variables

* price
* cost
* promotion_depth
* promotion_flag

---

## Marketing Variables

* search_spend
* social_spend
* display_spend
* email_flag

---

## Inventory Variables

* inventory_on_hand
* inventory_capacity
* inventory_pct_remaining
* stock_status

---

## Digital Variables

* web_sessions
* app_sessions
* page_views
* conversion_rate

---

## Event Variables

* event_name
* event_flag
* pre_event_flag
* post_event_flag

---

## Sales Variables

* units
* revenue
* gross_profit

---

## Operational Variables

* store_capacity
* fulfillment_capacity
* labor_hours

---

# Expected Outputs

For every recommendation scenario:

* Recommended Price
* Price Change %
* Expected Units Change %
* Expected Revenue Change %
* Expected Profit Change %
* Ending Inventory
* Stock Status
* Stockout Risk
* Business Explanation

---

# Future Enhancements

* Marketing Budget Optimization
* Inventory Transfer Optimization
* Multi-SKU Optimization
* GenAI Decision Copilot
* Reinforcement Learning Pricing Engine
