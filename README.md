# Premier League Forwards: 2026-27 Forecast

A forecasting model and interactive dashboard for Premier League forwards. It projects each forward's goals, assists and minutes for 2026-27 from five seasons of past data, and reports the uncertainty around every number.

![Top 3 projected scorers](assets/top3_forecast.png)

## Highlights

- **Forecasts with ranges, not single numbers.** Every projection comes with a 10th to 90th percentile band and the probability of reaching 10, 15 or 20 goals.
- **Shrinkage-based rate model.** Goals, assists and penalty attempts per 90 are pulled toward the league average in proportion to how little a player has played, so small samples do not produce extreme forecasts.
- **Honest backtest.** Trained on earlier seasons only and tested on 2024-25 and 2025-26, against a "repeat last season" baseline.
- **Interactive Streamlit dashboard** with leaderboard, player profiles, comparisons, over/under performers and a Golden Boot race.

## Backtest results

Tested on 163 player-seasons (100 regulars with 900+ minutes for the per-90 comparison). Lower error is better.

| Metric | Model | Repeat last season | League average |
|---|---|---|---|
| Goals per 90 error (regulars) | **0.116** | 0.138 | 0.134 |
| Assists per 90 error (regulars) | **0.078** | 0.097 | 0.086 |
| Season goals error | **3.60** | 3.77 | n/a |
| Season assists error | **1.64** | 1.76 | n/a |
| Season G+A error | **4.71** | 5.10 | n/a |
| Minutes error | 683 | 694 | n/a |
| Scorer ranking (Spearman) | 0.50 | 0.53 | n/a |

- The rate model clearly beats the baselines.
- The minutes model only marginally beats repeating last season, and the model does not beat the baseline at ranking scorers.
- The 10th to 90th percentile range for goals contains the actual total 82% of the time (ideal: 80%). For goals plus assists it is 77%.

## Dashboard

| Tab | What it shows |
|---|---|
| Leaderboard | Rank forwards by goals, assists or G+A with filters for club, age, minutes and sample size |
| Player profile | Season history plus the 2026-27 forecast with ranges, minutes and per-90 trends |
| Compare | Radar comparison of up to four players, as percentiles among regulars |
| Over / under performers | Last season's raw rate against the forecast rate, to spot regression risks and bounce-back candidates |
| Golden Boot race | Approximate probability of finishing top scorer or top three |
| Model report | Backtest results, calibration, method and limitations |

## Project structure

```
.
├── app.py                          # Streamlit dashboard
├── forward_forecast_pipeline.py    # Cleaning, models, backtest, forecast
├── requirements.txt
├── assets/                         # Images used in this README
└── data/
    ├── panel_clean.csv             # One row per player-season (702 rows, 375 players)
    ├── forecast_2026_27.csv        # Forecasts for 100 forwards
    └── backtest_metrics.csv        # Backtest results by fold and pooled
```

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Rebuild the forecast from raw data

The raw workbook is not included in this repository. It is an Excel file with one sheet per season (2021-22 to 2025-26) and these columns:

`Player, Nation, Position, Squad, Age, Born, Matches Played, Starts, Minutes Played, 90s, Goals Scored, Assists, Goals + Assists, Non-Penalty Goals, Penalty Goals, Penalty Attempts, Yellow Card, Red Cards, Gls/90, Ast/90, G+A /90, G-PK /90, G+A-PK /90`

```bash
python forward_forecast_pipeline.py --input PL_Forwards_5Years.xlsx --outdir data
```

This regenerates the three CSVs in `data/`, and the dashboard picks them up on the next run. The pipeline needs `pandas`, `numpy` and `openpyxl`.

## Method

1. **Cleaning.** The season sheets are stacked and mid-season transfers are merged into one row per player-season (15 such rows in this data).
2. **Rate model.** For non-penalty goals/90, assists/90 and penalty attempts/90, each player's rate is a Poisson-Gamma posterior: the league average acts as a prior worth `k` virtual 90-minute games, and past seasons are added with exponentially decaying weights. The decay and `k` are tuned on training seasons only.
3. **Minutes model.** Ridge regression on last season's minutes, a recency-weighted minutes average, starts share, minutes per appearance, age, age squared, number of past seasons and whether the player played last season.
4. **Simulation.** 4,000 simulated seasons per player combine the minutes uncertainty (bootstrapped from training residuals) with Gamma-uncertain rates and Poisson scoring. Penalty goals are modelled as penalty attempts times the league conversion rate. The simulations give expected values, percentile ranges and threshold probabilities.
5. **Backtest.** Rolling origin: for each test season the model is tuned and fitted only on earlier seasons.

## Limitations

- **Minutes are the weakest link.** Injuries and transfers are not visible in this data, and the minutes model barely beats repeating last season.
- **Survivor bias.** Players who left the league or lost the forward label have no next-season row, so the model cannot learn from them. Forecasts assume each player stays in the league as a forward.
- **Small sample.** About 160 backtest cases, so small differences between models are not conclusive.
- **Conservative at the top.** Shrinkage pulls consistent elite scorers toward the average, which is why the forecast for the highest scorers sits well below their last season.
- **No shot data.** There is no xG, shots or xA, so there is no threat model and no separation of finishing skill from luck.
- **Golden Boot tab is approximate.** It is rebuilt from each player's forecast mean and range, treating players as independent, not from the original simulations.

## Roadmap

- Add shots, xG and xA to build a threat model and a finishing-versus-luck split.
- Save the simulated seasons so the Golden Boot race is exact.
- Add a fixture-adjusted view and an FPL expected-points layer.
- Add forecast-versus-actual tracking as the 2026-27 season progresses.

## Data note

Forecasts are statistical estimates, not predictions of certainty. Check the terms of your data source before redistributing the raw statistics or files derived from them.
