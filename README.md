# German Electricity Market: Explaining and Forecasting Day-Ahead Power Prices

Analysis of German day-ahead electricity prices using electricity demand, wind generation, solar generation, and historical price information.

---

## Research Question

> How are electricity demand and renewable generation associated with German day-ahead electricity prices, and can they help predict price?

### Subquestions

- When are electricity prices highest and lowest?
- How do demand, wind, and solar generation vary over time?
- Is electricity demand associated with price?
- Is renewable generation associated with price?
- Under what demand and renewable-generation conditions do price extremes occur?
- Can historical price information and energy variables improve short-term price prediction?

---

## Dataset

Data was collected from the **Energy-Charts API** for Germany.

| Attribute | Details |
|---|---|
| Country | Germany |
| Period | 2026-01-01 to 2026-09-13 |
| Resolution | 15 minutes |
| Observations | 24,572 |
| Time zone | Europe/Berlin |
| Price | Day-ahead electricity price |
| Demand | Electricity load |
| Renewable generation | Onshore wind and solar generation |

### Data Quality

The dataset contains:

- No missing values
- No duplicate timestamps

During the daylight-saving transition, four local 15-minute timestamps do not occur in the `Europe/Berlin` local-time sequence. This produces one longer local-time gap.

This was treated as a **calendar/time-zone effect rather than ordinary missing data**, so the observations were not imputed or removed.

---

## Analysis Workflow

The analysis followed this workflow:

1. Collected electricity price and power-generation data from the Energy-Charts API.
2. Converted timestamps to `Europe/Berlin`.
3. Merged price and power data using timestamps.
4. Performed data-quality checks.
5. Conducted exploratory data analysis.
6. Examined correlations between price, demand, and renewable generation.
7. Engineered renewable-generation, time, and calendar features.
8. Tested historical price lags at:
   - 15 minutes
   - 24 hours
   - 7 days
9. Compared Linear Regression with Gradient Boosting.
10. Tuned Gradient Boosting using `TimeSeriesSplit` to preserve chronological order.
11. Evaluated residuals and feature importance.
12. Compared model errors across different price regimes.
13. Evaluated the final models against a persistence baseline.

---

## Exploratory Data Analysis

### Correlation Findings

Renewable generation showed a strong negative linear association with electricity price:

**Pearson correlation: `r = -0.72`**

Overall load-price correlation was weakly positive:

**Pearson correlation: `r = 0.11`**

However, the load-price relationship varied substantially by hour of day.

This demonstrates why simple overall correlations can hide important time-dependent patterns.

---

## Feature Engineering

The analysis introduced several derived features.

### Renewable Generation

```text
renewable_generation = wind_onshore + solar
