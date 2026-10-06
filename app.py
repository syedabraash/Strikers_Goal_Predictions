"""
Premier League forwards: 2026-27 forecast dashboard.

Reads the CSVs produced by forward_forecast_pipeline.py (stored in ./data):
  forecast_2026_27.csv, panel_clean.csv, backtest_metrics.csv

Run:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).parent / "data"
FORECAST_SEASON = "2026-27"
C_GOALS, C_ASSISTS, C_MIN = "#1f77b4", "#ff7f0e", "#2ca02c"

st.set_page_config(page_title="PL Forwards Forecast", page_icon="⚽", layout="wide")


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    fc = pd.read_csv(DATA / "forecast_2026_27.csv")
    panel = pd.read_csv(DATA / "panel_clean.csv")
    bt = pd.read_csv(DATA / "backtest_metrics.csv", index_col=0)

    last = panel[panel["yr"] == panel["yr"].max()].set_index("Player")
    fc["last_NPG90"] = fc["Player"].map(last["NPG90"])
    fc["last_A90"] = fc["Player"].map(last["A90"])
    fc["last_G"] = fc["Player"].map(last["G"])
    fc["last_A"] = fc["Player"].map(last["A"])
    return fc, panel, bt


@st.cache_data
def simulate_golden_boot(means, sds, n_sims=20000, seed=7):
    """Approximate title race from each player's forecast mean and p10-p90 spread.

    Season goals are drawn from a negative binomial matched to those two numbers,
    independently across players. Ties are broken at random.
    """
    rng = np.random.default_rng(seed)
    m, s = np.array(means), np.array(sds)
    var = np.maximum(s ** 2, m * 1.05)  # NB needs variance > mean
    r = m ** 2 / (var - m)
    p = r / (r + m)
    draws = rng.negative_binomial(r[None, :], p[None, :], size=(n_sims, len(m)))
    order = np.argsort(-(draws + rng.random(draws.shape) * 0.5), axis=1)
    win = np.bincount(order[:, 0], minlength=len(m)) / n_sims
    top3 = sum(np.bincount(order[:, k], minlength=len(m)) for k in range(3)) / n_sims
    return win, top3


fc, panel, bt = load_data()
METRICS = {
    "Goals": ("exp_goals", "goals_p10", "goals_p90", C_GOALS),
    "Assists": ("exp_ast", "ast_p10", "ast_p90", C_ASSISTS),
    "Goals + assists": ("exp_ga", "ga_p10", "ga_p90", "#9467bd"),
}

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("Premier League forwards: 2026-27 forecast")
st.caption(
    f"{len(fc)} forwards who played in the Premier League in 2025-26. "
    "Forecasts assume each player stays in the league as a forward. "
    "Shaded ranges and error bars show the 10th to 90th percentile."
)

tab_lb, tab_pl, tab_cmp, tab_gap, tab_gb, tab_model = st.tabs(
    ["Leaderboard", "Player profile", "Compare", "Over / under performers",
     "Golden Boot race", "Model report"]
)

# ----------------------------------------------------------------------------
# 1. Leaderboard
# ----------------------------------------------------------------------------
with tab_lb:
    c1, c2, c3, c4 = st.columns([2, 2, 2, 1.4])
    squads = c1.multiselect("Club", sorted(fc["Squad"].unique()))
    age_lo, age_hi = c2.slider("Age in 2026-27", int(fc["Age_next"].min()),
                               int(fc["Age_next"].max()),
                               (int(fc["Age_next"].min()), int(fc["Age_next"].max())))
    min_proj = c3.slider("Min. projected minutes", 0, 3000, 900, step=100)
    hide_low = c4.checkbox("Hide low-sample", value=True,
                           help="Players with one short season of history.")
    metric = st.radio("Rank by", list(METRICS), horizontal=True)

    d = fc[(fc["Age_next"].between(age_lo, age_hi)) & (fc["proj_min"] >= min_proj)]
    if squads:
        d = d[d["Squad"].isin(squads)]
    if hide_low:
        d = d[~d["low_sample"]]
    col, lo, hi, colour = METRICS[metric]
    d = d.sort_values(col, ascending=False)

    if d.empty:
        st.info("No players match these filters.")
    else:
        top_n = st.slider("Players in chart", 5, min(40, max(5, len(d))),
                          min(15, max(5, len(d))))
        top = d.head(top_n)
        fig = go.Figure(go.Bar(
            x=top[col], y=top["Player"], orientation="h", marker_color=colour,
            error_x=dict(type="data", symmetric=False,
                         array=top[hi] - top[col], arrayminus=top[col] - top[lo]),
            customdata=np.stack([top["Squad"], top[lo], top[hi]], axis=1),
            hovertemplate="<b>%{y}</b> (%{customdata[0]})<br>Expected: %{x:.1f}"
                          "<br>10-90% range: %{customdata[1]:.0f} to %{customdata[2]:.0f}"
                          "<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=max(320, 26 * top_n), margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title=f"Expected {metric.lower()} in {FORECAST_SEASON}")
        st.plotly_chart(fig, width="stretch")

        table = pd.DataFrame({
            "Player": d["Player"], "Club": d["Squad"], "Age": d["Age_next"],
            "Proj. minutes": d["proj_min"].round(0).astype(int),
            "Exp. goals": d["exp_goals"].round(1),
            "Goals range": d["goals_p10"].astype(int).astype(str) + " to " + d["goals_p90"].astype(int).astype(str),
            "Exp. assists": d["exp_ast"].round(1),
            "Exp. G+A": d["exp_ga"].round(1),
            "P(10+ goals)": d["P_goals_ge_10"], "P(15+ goals)": d["P_goals_ge_15"],
            "Low sample": d["low_sample"],
        })
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "P(10+ goals)": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "P(15+ goals)": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        })

# ----------------------------------------------------------------------------
# 2. Player profile
# ----------------------------------------------------------------------------
with tab_pl:
    ranked = fc.sort_values("exp_goals", ascending=False)
    name = st.selectbox("Player", ranked["Player"].tolist(),
                        format_func=lambda n: f"{n} ({ranked.loc[ranked['Player'] == n, 'Squad'].iloc[0]})")
    r = fc[fc["Player"] == name].iloc[0]
    h = panel[panel["Player"] == name].sort_values("yr")

    st.subheader(f"{name}  |  {r['Squad']}  |  age {int(r['Age_next'])} in {FORECAST_SEASON}")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Projected minutes", f"{r['proj_min']:,.0f}")
    k2.metric("Expected goals", f"{r['exp_goals']:.1f}")
    k2.caption(f"range {r['goals_p10']:.0f} to {r['goals_p90']:.0f}")
    k3.metric("Expected assists", f"{r['exp_ast']:.1f}")
    k3.caption(f"range {r['ast_p10']:.0f} to {r['ast_p90']:.0f}")
    k4.metric("Expected G+A", f"{r['exp_ga']:.1f}")
    k4.caption(f"range {r['ga_p10']:.0f} to {r['ga_p90']:.0f}")
    k5.metric("P(10+ goals)", f"{r['P_goals_ge_10']:.0%}")
    k5.caption(f"15+: {r['P_goals_ge_15']:.0%}  |  20+: {r['P_goals_ge_20']:.0%}")
    if r["low_sample"]:
        st.warning("Low sample: only one short season of history, so this forecast leans on the league average.")

    seasons = h["Season"].tolist()
    cats = seasons + [f"{FORECAST_SEASON} (forecast)"]
    n = len(seasons)
    pad = [None]

    def err(lo_col, hi_col, mid):
        return dict(type="data", symmetric=False,
                    array=[0] * n + [r[hi_col] - r[mid]], arrayminus=[0] * n + [r[mid] - r[lo_col]])

    left, right = st.columns([3, 2])
    fig = go.Figure()
    fig.add_bar(x=cats, y=h["G"].tolist() + pad, name="Goals", marker_color=C_GOALS, offsetgroup="g")
    fig.add_bar(x=cats, y=[None] * n + [r["exp_goals"]], name="Goals (forecast)", marker_color=C_GOALS,
                marker_pattern_shape="/", offsetgroup="g", error_y=err("goals_p10", "goals_p90", "exp_goals"))
    fig.add_bar(x=cats, y=h["A"].tolist() + pad, name="Assists", marker_color=C_ASSISTS, offsetgroup="a")
    fig.add_bar(x=cats, y=[None] * n + [r["exp_ast"]], name="Assists (forecast)", marker_color=C_ASSISTS,
                marker_pattern_shape="/", offsetgroup="a", error_y=err("ast_p10", "ast_p90", "exp_ast"))
    fig.update_layout(barmode="group", title="Goals and assists per season", height=380,
                      margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", y=-0.2))
    left.plotly_chart(fig, width="stretch")

    fig = go.Figure()
    fig.add_scatter(x=seasons, y=h["Min"], mode="lines+markers", name="Actual", line_color=C_MIN)
    fig.add_scatter(x=[seasons[-1], cats[-1]], y=[h["Min"].iloc[-1], r["proj_min"]], mode="lines+markers",
                    name="Forecast", line=dict(color=C_MIN, dash="dash"))
    fig.update_layout(title="Minutes played", height=380, margin=dict(l=0, r=0, t=40, b=0),
                      yaxis_range=[0, 3500], legend=dict(orientation="h", y=-0.2))
    right.plotly_chart(fig, width="stretch")

    fig = go.Figure()
    for lab, col, fcol, colr in [("Goals/90 (non-penalty)", "NPG90", "rate_NPG", C_GOALS),
                                 ("Assists/90", "A90", "rate_A", C_ASSISTS)]:
        fig.add_scatter(x=seasons, y=h[col], mode="lines+markers", name=lab, line_color=colr)
        fig.add_scatter(x=[seasons[-1], cats[-1]], y=[h[col].iloc[-1], r[fcol]], mode="lines+markers",
                        name=f"{lab} forecast", line=dict(color=colr, dash="dash"), showlegend=False)
    fig.update_layout(title="Per-90 rates (forecast rates are shrunk toward the league average)",
                      height=340, margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

    hist = h[["Season", "Squad", "Age", "Matches", "Starts", "Min", "G", "A", "NPG", "PG", "PA", "NPG90", "A90"]].copy()
    hist.columns = ["Season", "Club", "Age", "Apps", "Starts", "Minutes", "Goals", "Assists",
                    "Non-pen goals", "Pen goals", "Pen attempts", "NPG/90", "Ast/90"]
    st.dataframe(hist.round(2), hide_index=True, width="stretch")

# ----------------------------------------------------------------------------
# 3. Compare
# ----------------------------------------------------------------------------
with tab_cmp:
    st.caption("Percentiles are relative to forwards projected for 900+ minutes.")
    default = fc.sort_values("exp_goals", ascending=False)["Player"].head(3).tolist()
    picks = st.multiselect("Players (up to 4)", sorted(fc["Player"]), default=default, max_selections=4)
    axes = [("Exp. goals", "exp_goals"), ("Exp. assists", "exp_ast"), ("Goal rate /90", "rate_NPG"),
            ("Assist rate /90", "rate_A"), ("Proj. minutes", "proj_min"),
            ("P(10+ goals)", "P_goals_ge_10"), ("Penalty duty", "rate_PA")]
    pool = fc[fc["proj_min"] >= 900]

    if not picks:
        st.info("Pick at least one player.")
    else:
        fig = go.Figure()
        for p in picks:
            row = fc[fc["Player"] == p].iloc[0]
            vals = [(pool[c] <= row[c]).mean() * 100 for _, c in axes]
            labels = [a for a, _ in axes]
            fig.add_scatterpolar(r=vals + vals[:1], theta=labels + labels[:1], fill="toself",
                                 name=f"{p} ({row['Squad']})", opacity=0.55)
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], ticksuffix="")), height=520,
                          margin=dict(l=40, r=40, t=20, b=20), legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, width="stretch")

        cmp = fc[fc["Player"].isin(picks)].set_index("Player").loc[picks]
        show = pd.DataFrame({
            "Club": cmp["Squad"], "Age": cmp["Age_next"], "Proj. minutes": cmp["proj_min"].round(0),
            "Exp. goals": cmp["exp_goals"].round(1), "Exp. assists": cmp["exp_ast"].round(1),
            "Goals/90": cmp["rate_NPG"].round(2), "Assists/90": cmp["rate_A"].round(2),
            "P(10+ goals)": (cmp["P_goals_ge_10"] * 100).round(0).astype(int).astype(str) + "%",
            "Pen attempts /90": cmp["rate_PA"].round(2)})
        st.dataframe(show, width="stretch")

# ----------------------------------------------------------------------------
# 4. Over / under performers
# ----------------------------------------------------------------------------
with tab_gap:
    st.caption("Compares last season's raw rate with the forecast rate. The forecast blends recent seasons with the "
               "league average, weighted by how much each player has played, so a big gap means last season looks "
               "hard to repeat (or was held back).")
    g1, g2 = st.columns([2, 2])
    kind = g1.radio("Rate", ["Goals/90 (non-penalty)", "Assists/90"], horizontal=True)
    min_last = g2.slider("Min. minutes in 2025-26", 300, 2500, 900, step=100)
    last_col, fc_col = ("last_NPG90", "rate_NPG") if kind.startswith("Goals") else ("last_A90", "rate_A")

    g = fc[fc["Min_last_season"] >= min_last].copy()
    g["gap"] = g[last_col] - g[fc_col]
    if g.empty:
        st.info("No players match this minutes filter.")
    else:
        extremes = pd.concat([g.nlargest(5, "gap"), g.nsmallest(5, "gap")])["Player"]
        g["label"] = np.where(g["Player"].isin(extremes), g["Player"].str.split().str[-1], "")
        fig = px.scatter(g, x=last_col, y=fc_col, size="Min_last_season", color="gap", text="label",
                         hover_name="Player", hover_data={"Squad": True, last_col: ":.2f", fc_col: ":.2f",
                                                          "Min_last_season": True, "gap": ":.2f", "label": False},
                         color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                         labels={last_col: "2025-26 raw rate", fc_col: "2026-27 forecast rate", "gap": "Gap"})
        top = float(max(g[last_col].max(), g[fc_col].max())) * 1.05
        fig.add_shape(type="line", x0=0, y0=0, x1=top, y1=top, line=dict(dash="dot", color="grey"))
        fig.update_traces(textposition="top center")
        fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")

        def view(df):
            return pd.DataFrame({"Player": df["Player"], "Club": df["Squad"],
                                 "2025-26 rate": df[last_col].round(2), "Forecast rate": df[fc_col].round(2),
                                 "Gap": df["gap"].round(2), "Minutes": df["Min_last_season"]})
        a, b = st.columns(2)
        a.markdown("**Regression risk** (last season well above forecast)")
        a.dataframe(view(g.nlargest(8, "gap")), hide_index=True, width="stretch")
        b.markdown("**Bounce-back candidates** (last season well below forecast)")
        b.dataframe(view(g.nsmallest(8, "gap")), hide_index=True, width="stretch")

# ----------------------------------------------------------------------------
# 5. Golden Boot race
# ----------------------------------------------------------------------------
with tab_gb:
    st.caption("Approximation: each player's season goals are drawn from a distribution matched to the forecast mean "
               "and 10-90% range, independently across players. Transfers and injuries beyond what the minutes model "
               "captures are not modelled.")
    size = st.slider("Players in the race", 10, 40, 25)
    pool = fc.sort_values("exp_goals", ascending=False).head(size).reset_index(drop=True)
    sds = ((pool["goals_p90"] - pool["goals_p10"]) / 2.563).clip(lower=1.0)
    win, top3 = simulate_golden_boot(tuple(pool["exp_goals"]), tuple(sds))
    res = pool.assign(win=win, top3=top3).sort_values("win", ascending=False)

    fig = go.Figure(go.Bar(x=res["win"].head(15), y=res["Player"].head(15), orientation="h",
                           marker_color=C_GOALS, text=[f"{v:.0%}" for v in res["win"].head(15)],
                           textposition="outside"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=480, margin=dict(l=0, r=40, t=10, b=0), xaxis_tickformat=".0%",
                      xaxis_title="Probability of finishing top scorer (ties split)")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame({"Player": res["Player"], "Club": res["Squad"],
                               "Exp. goals": res["exp_goals"].round(1),
                               "Win": res["win"], "Top 3": res["top3"]}),
                 hide_index=True, width="stretch", column_config={
        "Win": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        "Top 3": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})

# ----------------------------------------------------------------------------
# 6. Model report
# ----------------------------------------------------------------------------
with tab_model:
    num = lambda k, col="pooled": float(pd.to_numeric(bt.loc[k, col], errors="coerce"))
    st.markdown("Backtest: the model is trained on earlier seasons only and tested on 2024-25 and 2025-26 "
                f"({int(num('n_players'))} player-seasons pooled). Naive = each player's most recent season repeated.")

    pairs = [("Goals", "goals_MAE"), ("Assists", "assists_MAE"), ("Goals + assists", "GA_MAE"),
             ("Minutes (/100)", "minutes_MAE")]
    rows = []
    for lab, k in pairs:
        scale = 100 if "Minutes" in lab else 1
        rows.append((lab, num(f"{k}_model") / scale, num(f"{k}_naive_last") / scale))
    cmp_df = pd.DataFrame(rows, columns=["Target", "Model", "Naive (last season)"])
    fig = go.Figure()
    fig.add_bar(x=cmp_df["Target"], y=cmp_df["Model"], name="Model", marker_color=C_GOALS)
    fig.add_bar(x=cmp_df["Target"], y=cmp_df["Naive (last season)"], name="Naive (last season)", marker_color="#9aa0a6")
    fig.update_layout(barmode="group", title="Mean absolute error (lower is better)", height=380,
                      margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Goals/90 error", f"{num('NPG90_MAE_EB_model'):.3f}",
              f"{num('NPG90_MAE_EB_model') - num('NPG90_MAE_last_season_raw'):+.3f} vs last season", delta_color="inverse")
    m2.metric("Assists/90 error", f"{num('A90_MAE_EB_model'):.3f}",
              f"{num('A90_MAE_EB_model') - num('A90_MAE_last_season_raw'):+.3f} vs last season", delta_color="inverse")
    m3.metric("Goals 10-90% range hit rate", f"{num('goals_p10_p90_coverage'):.0%}", help="Ideal: 80%")
    m4.metric("Scorer ranking (Spearman)", f"{num('goals_spearman_model'):.2f}",
              f"{num('goals_spearman_model') - num('goals_spearman_naive_last'):+.2f} vs last season")

    with st.expander("How the forecast works"):
        st.markdown(
            "1. **Clean:** five seasons stacked, mid-season transfers merged into one row per player-season.\n"
            "2. **Rates:** non-penalty goals/90, assists/90 and penalty attempts/90 are shrunk toward the league average "
            "(Poisson-Gamma model), with recent seasons weighted more.\n"
            "3. **Minutes:** ridge regression on last season's minutes, starts share, age and history.\n"
            "4. **Simulation:** 4,000 simulated seasons per player give the expected values and 10-90% ranges.\n")
    with st.expander("Limitations"):
        st.markdown(
            "- Minutes are the weakest link: the minutes model barely beats repeating last season, and injuries and "
            "transfers are not visible in this data.\n"
            "- Players who left the league or lost the forward label have no next-season row, which biases training.\n"
            "- Only about 160 backtest cases, so small differences between models are not conclusive.\n"
            "- No shots, xG or xA yet. Adding them would allow a real threat model and a finishing-vs-luck split.\n"
            "- Forecast rates are conservative for consistent elite scorers because of shrinkage.\n")

st.divider()
st.caption("Source data: Premier League forwards, 2021-22 to 2025-26, as supplied. Forecasts are statistical estimates, not predictions of certainty.")
