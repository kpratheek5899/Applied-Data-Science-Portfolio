# Retail Pricing & Capacity Optimization Engine

## Project Overview

Nova Retail is a fictional modern omnichannel retailer with a strong digital footprint across e-commerce, mobile app, BOPIS, and physical store sales.

The company wants to understand how pricing, promotions, marketing, inventory visibility, stock status messaging, and operational capacity influence demand and profitability.

This project builds an end-to-end decision science system that simulates realistic retail data, estimates demand elasticities, and recommends pricing decisions under business constraints.

## Business Problem

Retailers often make pricing and promotional decisions without fully accounting for:

- Price elasticity
- Digital traffic
- Marketing pressure
- Online inventory visibility
- Stock status messaging
- Fulfillment capacity
- Store labor constraints
- Product margin differences

The goal of this project is to estimate demand response and optimize pricing decisions while respecting operational constraints.

## Objective

Build a portfolio-grade decision science engine that can:

1. Simulate modern omnichannel retail data
2. Estimate price elasticity using statistical and Bayesian methods
3. Compare OLS and Bayesian elasticity estimates
4. Recover known parameters from simulated data
5. Optimize pricing decisions to maximize profit
6. Account for inventory, fulfillment capacity, and operational constraints
7. Present recommendations through a Streamlit dashboard

## Channels

Nova Retail sells through:

- Website
- Mobile App
- Buy Online Pickup In Store
- Store Walk-in

## Key Business Variables

### Commercial Variables

- Price
- Discount depth
- Promotion flag
- Product cost
- Margin
- Units sold
- Revenue
- Profit

### Digital Variables

- Website sessions
- Mobile app sessions
- Product page views
- Add-to-cart rate
- Conversion rate
- Online stock visibility
- Stock status message

### Inventory Variables

- Inventory on hand
- Days of supply
- Stock status
- Stockout flag
- Low stock flag

### Marketing Variables

- Paid search spend
- Paid social spend
- Display spend
- Email campaign flag
- Organic traffic

### Operational Variables

- Store capacity
- Fulfillment capacity
- Labor hours
- BOPIS capacity
- Capacity utilization

## Methods

This project will use:

- Python
- Data simulation
- Regression modeling
- Bayesian hierarchical modeling
- Applied econometrics
- Parameter recovery
- Constrained optimization
- Streamlit dashboarding

## Planned Architecture

```text
Synthetic Retail Data
        ↓
Exploratory Analysis
        ↓
OLS Elasticity Baseline
        ↓
Bayesian Elasticity Model
        ↓
Parameter Recovery
        ↓
Pricing Optimization
        ↓
Scenario Simulator
        ↓
Streamlit Dashboard