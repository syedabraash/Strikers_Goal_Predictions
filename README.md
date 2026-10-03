# Premier League forwards: 2026-27 forecast dashboard

## Run
```
pip install -r requirements.txt
streamlit run app.py
```

## Refresh the data
```
python forward_forecast_pipeline.py --input PL_Forwards_5Years.xlsx --outdir data
```
The dashboard reads `data/forecast_2026_27.csv`, `data/panel_clean.csv` and `data/backtest_metrics.csv`.

## Tabs
Leaderboard, Player profile, Compare, Over / under performers, Golden Boot race (approximate), Model report.
