# German Electricity Market: Explaining and Forecasting Day-Ahead Power Prices

I built this project to understand how electricity demand and renewable generation relate to electricity prices in Germany, and to see how well those variables can help predict short-term prices.

The main focus was not just getting a good model score. I also wanted to understand which features actually helped, how much recent price history matters, and where the model still struggles.

---

## Research Question

**How are electricity demand and renewable generation associated with German day-ahead electricity prices, and can they help predict price?**

I looked at this through a few smaller questions:

- How do electricity prices behave over time?
- How are demand, wind, and solar generation related to price?
- Does the relationship between demand and price change by hour?
- Does historical price information improve prediction?
- How well can a model predict very low and very high prices?

---

## Dataset

The data was collected from the **Energy-Charts API** for Germany.

| | |
|---|---|
| Country | Germany |
| Period | 2026-01-01 to 2026-09-13 |
| Resolution | 15 minutes |
| Observations | 24,572 |
| Time zone | Europe/Berlin |

### Variables

- `load` — electricity demand
- `wind_onshore` — onshore wind generation
- `solar` — solar generation
- `price` — day-ahead electricity price

The timestamps from the price and power data were converted to `Europe/Berlin` and matched before merging the datasets.

### Data quality

- 0 missing values
- 0 duplicate timestamps

There are four fewer local 15-minute timestamps than a simple calendar calculation would suggest because of the daylight-saving transition. I treated this as a time-zone/calendar effect rather than ordinary missing data.

---

## What I Did

The analysis followed a fairly straightforward process:

1. Collected price and power data from the API.
2. Cleaned and checked the timestamps.
3. Merged the datasets.
4. Explored the distributions and relationships between variables.
5. Looked at correlations between price, load, and renewable generation.
6. Created time and renewable-generation features.
7. Tested different historical price lags.
8. Built a Linear Regression model.
9. Built a Gradient Boosting model.
10. Tuned Gradient Boosting using time-series cross-validation.
11. Checked residuals and feature importance.
12. Compared errors for very low and very high prices.
13. Compared everything against a simple persistence baseline.

---

## Exploratory Analysis

One of the first things I looked at was the relationship between price and renewable generation.

The correlation between total renewable generation and price was:

**r = -0.72**

Load and price had a much weaker overall correlation:

**r = 0.11**

However, the load-price relationship changed considerably depending on the hour of the day. That was a useful reminder that looking at one overall correlation can hide time-dependent patterns.

---

## Feature Engineering

I created a combined renewable-generation variable:

```python
renewable_generation = wind_onshore + solar
```

I also extracted time information from the timestamp:

- `hour`
- `day_of_week`
- `month`

For the hour of day, I used cyclical encoding:

- `hour_sin`
- `hour_cos`

This represents the 24-hour cycle without treating 23:00 and 00:00 as far apart.

### Historical price features

I tested several lagged price variables:

| Feature | Meaning |
|---|---|
| `price_lag_1` | previous 15 minutes |
| `price_lag_2` | previous 30 minutes |
| `price_lag_4` | previous 1 hour |
| `price_lag_96` | previous 24 hours |
| `price_lag_672` | previous 7 days |

The 30-minute and 1-hour lags did not add enough useful information to keep them in the final feature set.

The 24-hour and 7-day lags did improve the results, so they were retained.

Weekday features were also retained after testing.

I did not use month-of-year in the final model because the dataset only covers January through September, so I did not have a full annual cycle.

---

## Models

I compared four approaches:

- Persistence baseline
- Linear Regression
- Gradient Boosting
- Tuned Gradient Boosting

### Persistence baseline

The persistence model simply uses the previous observed price as the prediction.

```python
baseline_pred = merged_df["price"].shift(1).loc[y_test.index]
```

This is a useful baseline for electricity prices because recent prices are highly persistent.

### Gradient Boosting tuning

For the Gradient Boosting model, I used `TimeSeriesSplit` instead of randomly mixing observations between training and validation.

Best parameters:

- `learning_rate = 0.05`
- `max_depth = 4`
- `n_estimators = 200`

The test set was kept separate from the hyperparameter tuning process.

---

## Model Performance

All final models were evaluated on the same chronological test period.

| Model | MAE (€/MWh) | RMSE (€/MWh) | R² |
|---|---|---|---|
| Persistence Baseline | 9.011 | 14.059 | 0.9581 |
| Linear Regression | 8.383 | 12.955 | 0.9644 |
| Gradient Boosting (baseline) | 8.333 | 12.797 | 0.9653 |
| Gradient Boosting (tuned) | 8.250 | 12.646 | 0.9661 |

The tuned Gradient Boosting model reduced:

- MAE by ~8.45%
- RMSE by ~10.05%

compared with the persistence baseline. The improvement over Linear Regression was real, but fairly small.

---

## Feature Importance

The tuned Gradient Boosting model relied heavily on recent price history.

| Feature | Importance |
|---|---|
| `price_lag_1` | ~0.970 |
| `price_lag_96` | ~0.009 |
| `wind_onshore` | ~0.005 |
| `solar` | ~0.005 |
| `hour_sin` | ~0.005 |
| `price_lag_672` | ~0.004 |

The previous 15-minute price was by far the most important feature. So although demand and renewable generation were part of the analysis, recent price history ended up carrying most of the predictive signal for this model.

These feature-importance values describe how the model used the variables. They do not establish causality.

---

## Residual Analysis

I also checked whether the Linear Regression residuals still had temporal structure.

| Lag | Time interval | Residual autocorrelation |
|---|---|---|
| 1 | 15 minutes | 0.060 |
| 2 | 30 minutes | 0.082 |
| 4 | 1 hour | 0.440 |
| 96 | 24 hours | 0.515 |
| 672 | 7 days | 0.444 |

The residuals were only weakly autocorrelated at 15 and 30 minutes, but much more strongly correlated at 1 hour, 24 hours, and 7 days. This suggested that the linear model was capturing a lot of the short-term behaviour, but still leaving some temporal structure unexplained.

---

## Error Analysis

Overall model performance does not tell the whole story, especially for electricity prices. I compared the models for the lowest and highest 5% of actual prices.

| Price region | Linear MAE (€/MWh) | Tuned Gradient Boosting MAE (€/MWh) |
|---|---|---|
| Lowest 5% | 2.155 | 1.565 |
| Highest 5% | 18.820 | 20.778 |

Gradient Boosting performed better for the lowest-price observations, including negative-price periods. For the highest-price observations, Linear Regression actually performed better. So the nonlinear model did not solve the problem of extreme upward price spikes.

---

## Key Findings

- Renewable generation had a strong negative linear association with electricity price (r = -0.72).
- Overall load-price correlation was weakly positive (r = 0.11).
- The relationship between load and price changed significantly depending on the hour of the day.
- The previous 15-minute price was the strongest predictive feature.
- The 24-hour price lag improved the model.
- The 7-day price lag provided another small improvement.
- The 30-minute and 1-hour lags did not provide enough additional value to keep.
- Weekday features improved the Linear Regression model.
- Recent price history explained a large part of short-term price behaviour.
- Gradient Boosting improved overall performance, but the improvement was modest.
- Extreme upward price spikes were still difficult to predict.

---

## Final Conclusion

The main thing I found from this project is that electricity price history is extremely important for short-term prediction.

Renewable generation had a strong negative association with price, and load showed a more complicated relationship that changed throughout the day. However, when it came to prediction, the previous 15-minute price dominated the model.

The final tuned Gradient Boosting model achieved:

- MAE = 8.250 €/MWh
- RMSE = 12.646 €/MWh
- R² = 0.9661

It performed better than both the persistence baseline and the Linear Regression model on the overall test period, but the improvement was not dramatic.

The model also behaved differently across the price distribution. It handled very low prices better than Linear Regression, but it was worse at the highest-price observations.

So the result is not "renewables and demand can accurately predict electricity prices on their own." The stronger conclusion is that recent price behaviour contains most of the short-term predictive signal, while the other variables add smaller amounts of information.

---

## Limitations

- The dataset covers less than one full year.
- Annual seasonality could not be properly evaluated.
- Final model evaluation used one chronological train/test split.
- Time-series cross-validation was used for Gradient Boosting hyperparameter tuning.
- Feature importance does not imply causation.
- Extreme price spikes remain difficult to predict.
- The model uses only a selected set of observable variables and does not include every factor affecting electricity prices.

---

## Tools

- Python
- pandas
- NumPy
- Matplotlib
- scikit-learn
- requests
- Jupyter Notebook

---

## Project Structure

```
.
├── german_electricity_analysis.ipynb
├── README.md
└── requirements.txt
```

---

## Reproducibility

Clone the repository:

```bash
git clone https://github.com/rinshina/german-electricity-price-analysis
cd german-electricity-price-analysis
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook
```

Then open `german_electricity_analysis.ipynb` and run the notebook from top to bottom.

The notebook retrieves the data directly from the Energy-Charts API, so there is no separate dataset file required.

---

## Data Source

[Energy-Charts API — Fraunhofer ISE](https://www.energy-charts.info/)

The project uses the API to retrieve German electricity price and power data:

- Day-ahead electricity price
- Electricity load
- Onshore wind generation
- Solar generation

---

## Author

**Rinshina Febin K**
[GitHub](https://github.com/rinshina) · [LinkedIn](https://linkedin.com/in/febinkp/)
