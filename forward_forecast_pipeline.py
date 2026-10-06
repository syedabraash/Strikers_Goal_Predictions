#!/usr/bin/env python3
"""
Premier League forwards: cleaning, baselines and next-season forecasts.

Steps
  1. Load the 5 season sheets, stack them, merge split-season (transfer) rows
     into one row per player-season.
  2. Baseline rate model: empirical-Bayes (Poisson-Gamma) shrinkage of
     non-penalty goals/90, assists/90 and penalty attempts/90, with exponential
     recency decay over past seasons.
  3. Minutes model: ridge regression on prior minutes, starts share, age.
  4. Monte-Carlo simulation -> p10 / p50 / p90 bands for goals, assists, G+A.
  5. Rolling-origin backtest (train on earlier seasons, test on the next one)
     and a 2026-27 forecast.

Usage
  python forward_forecast_pipeline.py --input PL_Forwards_5Years.xlsx --outdir out

Requires: pandas, numpy, openpyxl
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MAX_MIN = 38 * 90
N_SIMS = 4000
SEED = 42
MIN_FEATURES = ["last_min", "prior_min_w", "starts_share", "min_per_match",
                "age_c", "age_c2", "n_prior", "played_last"]


# ----------------------------------------------------------------------------
# 1. Loading and cleaning
# ----------------------------------------------------------------------------
def load_raw(path):
    sheets = pd.read_excel(path, sheet_name=None)
    frames = []
    for name, df in sheets.items():
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        df["yr"] = int(name[:4])
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    return raw.rename(columns={
        "Matches Played": "Matches", "Minutes Played": "Min", "Goals Scored": "G",
        "Assists": "A", "Penalty Goals": "PG", "Penalty Attempts": "PA",
        "Non-Penalty Goals": "NPG", "Yellow Card": "YC", "Red Cards": "RC"})


def build_panel(raw):
    """One row per player-season. Transfers inside a season are summed."""
    born_conflict = raw.groupby("Player")["Born"].nunique()
    if (born_conflict > 1).any():
        print("WARNING: players with conflicting birth years (possible name clashes):",
              list(born_conflict[born_conflict > 1].index))
    main_squad = (raw.sort_values("Min").groupby(["Player", "yr"]).tail(1)
                  [["Player", "yr", "Squad"]])
    sums = raw.groupby(["Player", "yr"]).agg(
        Nation=("Nation", "first"), Born=("Born", "first"), Age=("Age", "max"),
        Matches=("Matches", "sum"), Starts=("Starts", "sum"), Min=("Min", "sum"),
        G=("G", "sum"), A=("A", "sum"), NPG=("NPG", "sum"), PG=("PG", "sum"),
        PA=("PA", "sum"), YC=("YC", "sum"), RC=("RC", "sum"),
        n_stints=("Squad", "nunique")).reset_index()
    panel = sums.merge(main_squad, on=["Player", "yr"])
    panel["Season"] = panel["yr"].astype(str) + "-" + (panel["yr"] + 1).astype(str).str[2:]
    panel["n90"] = panel["Min"] / 90
    for c, out in [("G", "G90"), ("A", "A90"), ("NPG", "NPG90"), ("PA", "PA90")]:
        panel[out] = panel[c] / panel["n90"]
    panel["GA"] = panel["G"] + panel["A"]
    panel["starts_share"] = panel["Starts"] / panel["Matches"].clip(lower=1)
    panel["pen_taker"] = (panel["PA"] >= 2).astype(int)
    return panel.sort_values(["yr", "Player"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. Rate model: Poisson-Gamma shrinkage with recency decay
# ----------------------------------------------------------------------------
def posterior(panel, t, stat, decay, k):
    """Gamma posterior for a per-90 rate using seasons < t.

    prior: Gamma(mu*k, k)  (mean = league rate, worth k 'virtual' 90s)
    data : decayed sums of events and 90s
    """
    hist = panel[panel["yr"] < t]
    mu = hist[stat].sum() / hist["n90"].sum()
    w = decay ** (t - 1 - hist["yr"])
    tmp = pd.DataFrame({"Player": hist["Player"], "ws": w * hist[stat],
                        "wn": w * hist["n90"]}).groupby("Player").sum()
    alpha = mu * k + tmp["ws"]
    beta = k + tmp["wn"]
    return pd.DataFrame({"alpha": alpha, "beta": beta, "rate": alpha / beta}), mu


def tune_rate(panel, train_years, stat,
              decays=(0.2, 0.4, 0.6, 0.8, 1.0), ks=(2, 4, 6, 10, 15, 25, 40, 60, 100)):
    best = (None, np.inf)
    for d in decays:
        for k in ks:
            se = wsum = 0.0
            for t in train_years:
                post, _ = posterior(panel, t, stat, d, k)
                tgt = panel[panel["yr"] == t].set_index("Player")
                j = tgt.join(post["rate"].rename("pred"), how="inner")
                se += (j["n90"] * (j[stat] / j["n90"] - j["pred"]) ** 2).sum()
                wsum += j["n90"].sum()
            score = se / wsum
            if score < best[1]:
                best = ((d, k), score)
    return best[0]


# ----------------------------------------------------------------------------
# 3. Minutes model: ridge regression
# ----------------------------------------------------------------------------
def minutes_features(panel, t, decay=0.6):
    hist = panel[panel["yr"] < t].sort_values("yr").copy()
    hist["w"] = decay ** (t - 1 - hist["yr"])
    hist["wmin"] = hist["w"] * hist["Min"]
    g = hist.groupby("Player")
    last = g.tail(1).set_index("Player")
    f = pd.DataFrame(index=last.index)
    f["last_min"] = last["Min"]
    f["prior_min_w"] = g["wmin"].sum() / g["w"].sum()
    f["starts_share"] = last["Starts"] / last["Matches"].clip(lower=1)
    f["min_per_match"] = last["Min"] / last["Matches"].clip(lower=1)
    f["age"] = last["Age"] + (t - last["yr"])
    f["age_c"] = f["age"] - 26
    f["age_c2"] = f["age_c"] ** 2
    f["n_prior"] = g.size()
    f["played_last"] = (last["yr"] == t - 1).astype(int)
    f["last_yr"] = last["yr"]
    f["Squad_last"] = last["Squad"]
    f["Min_last_season"] = last["Min"]
    return f


def fit_ridge(X, y, alpha):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    ym = y.mean()
    beta = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - ym))
    return {"mu": mu, "sd": sd, "beta": beta, "ym": ym}


def predict_ridge(m, X):
    return np.clip(m["ym"] + ((X - m["mu"]) / m["sd"]) @ m["beta"], 0, MAX_MIN)


def minutes_training_set(panel, years):
    rows = []
    for t in years:
        f = minutes_features(panel, t)
        y = panel[panel["yr"] == t].set_index("Player")["Min"].rename("y")
        j = f.join(y, how="inner")
        j["t"] = t
        rows.append(j)
    return pd.concat(rows)


def tune_ridge_alpha(panel, train_years, alphas=(0.1, 1, 10, 30, 100)):
    if len(train_years) < 2:
        return 10.0
    data = minutes_training_set(panel, train_years)
    best = (None, np.inf)
    for a in alphas:
        errs = []
        for t in train_years:
            tr, te = data[data["t"] != t], data[data["t"] == t]
            m = fit_ridge(tr[MIN_FEATURES].values.astype(float), tr["y"].values, a)
            p = predict_ridge(m, te[MIN_FEATURES].values.astype(float))
            errs.append(np.abs(p - te["y"].values).mean())
        if np.mean(errs) < best[1]:
            best = (a, np.mean(errs))
    return best[0]


# ----------------------------------------------------------------------------
# 4. Fit / predict / simulate
# ----------------------------------------------------------------------------
def fit_models(panel, train_years):
    models = {"rate": {}}
    for stat in ["NPG", "A", "PA"]:
        models["rate"][stat] = tune_rate(panel, train_years, stat)
    alpha = tune_ridge_alpha(panel, train_years)
    data = minutes_training_set(panel, train_years)
    X, y = data[MIN_FEATURES].values.astype(float), data["y"].values
    ridge = fit_ridge(X, y, alpha)
    resid = y - predict_ridge(ridge, X)
    pred = predict_ridge(ridge, X)
    bucket = np.digitize(pred, [900, 1900])
    pools = {}
    for b in range(3):
        r = resid[bucket == b]
        pools[b] = r if len(r) >= 15 else resid
    models.update(ridge=ridge, ridge_alpha=alpha, resid_pools=pools)
    return models


def predict_year(panel, t, models, n_sims=N_SIMS, seed=SEED):
    f = minutes_features(panel, t)
    out = f[["last_yr", "Squad_last", "Min_last_season", "n_prior", "age"]].copy()
    out["proj_min"] = predict_ridge(models["ridge"], f[MIN_FEATURES].values.astype(float))

    post = {}
    for stat, (d, k) in models["rate"].items():
        p, mu = posterior(panel, t, stat, d, k)
        post[stat] = p.reindex(out.index)
        out["rate_" + stat] = post[stat]["rate"]
        out["lg_" + stat] = mu

    hist = panel[panel["yr"] < t]
    conv = hist["PG"].sum() / max(hist["PA"].sum(), 1)

    out["exp_npg"] = out["rate_NPG"] * out["proj_min"] / 90
    out["exp_pg"] = out["rate_PA"] * out["proj_min"] / 90 * conv
    out["exp_goals"] = out["exp_npg"] + out["exp_pg"]
    out["exp_ast"] = out["rate_A"] * out["proj_min"] / 90
    out["exp_ga"] = out["exp_goals"] + out["exp_ast"]

    # --- Monte Carlo ---
    rng = np.random.default_rng(seed)
    n = len(out)
    pm = out["proj_min"].values
    bucket = np.digitize(pm, [900, 1900])
    mins = np.empty((n, n_sims))
    for b in range(3):
        idx = np.where(bucket == b)[0]
        if len(idx):
            mins[idx] = pm[idx, None] + rng.choice(models["resid_pools"][b], size=(len(idx), n_sims))
    mins = np.clip(mins, 0, MAX_MIN)

    def draw_rate(stat):
        a, b = post[stat]["alpha"].values, post[stat]["beta"].values
        return rng.gamma(a[:, None], 1.0 / b[:, None], size=(n, n_sims))

    npg = rng.poisson(draw_rate("NPG") * mins / 90)
    ast = rng.poisson(draw_rate("A") * mins / 90)
    pen_att = rng.poisson(draw_rate("PA") * mins / 90)
    pg = rng.binomial(pen_att, conv)
    goals, ga = npg + pg, npg + pg + ast

    for name, arr in [("goals", goals), ("ast", ast), ("ga", ga)]:
        q = np.percentile(arr, [10, 50, 90], axis=1)
        out[f"{name}_p10"], out[f"{name}_p50"], out[f"{name}_p90"] = q
    for th in (5, 10, 15, 20):
        out[f"P_goals_ge_{th}"] = (goals >= th).mean(axis=1)
    out["conv"] = conv
    return out


# ----------------------------------------------------------------------------
# 5. Backtest
# ----------------------------------------------------------------------------
def fold_metrics(j):
    """j: predictions joined with actual target-season stats."""
    m = {}
    m["n_players"] = len(j)
    m["minutes_MAE_model"] = (j["proj_min"] - j["Min"]).abs().mean()
    m["minutes_MAE_naive_last"] = (j["Min_last_season"] - j["Min"]).abs().mean()
    m["goals_MAE_model"] = (j["exp_goals"] - j["G"]).abs().mean()
    m["goals_MAE_naive_last"] = (j["G_last"] - j["G"]).abs().mean()
    m["assists_MAE_model"] = (j["exp_ast"] - j["A"]).abs().mean()
    m["assists_MAE_naive_last"] = (j["A_last"] - j["A"]).abs().mean()
    m["GA_MAE_model"] = (j["exp_ga"] - j["GA"]).abs().mean()
    m["GA_MAE_naive_last"] = ((j["G_last"] + j["A_last"]) - j["GA"]).abs().mean()
    m["goals_spearman_model"] = j["exp_goals"].corr(j["G"], method="spearman")
    m["goals_spearman_naive_last"] = j["G_last"].corr(j["G"], method="spearman")
    m["goals_p10_p90_coverage"] = ((j["G"] >= j["goals_p10"]) & (j["G"] <= j["goals_p90"])).mean()
    m["ga_p10_p90_coverage"] = ((j["GA"] >= j["ga_p10"]) & (j["GA"] <= j["ga_p90"])).mean()

    r = j[j["n90"] >= 9]  # regulars: rate error is meaningful
    m["n_regulars_900min"] = len(r)
    m["NPG90_MAE_EB_model"] = (r["rate_NPG"] - r["NPG90"]).abs().mean()
    m["NPG90_MAE_last_season_raw"] = (r["last_NPG90"] - r["NPG90"]).abs().mean()
    m["NPG90_MAE_league_mean"] = (r["lg_NPG"] - r["NPG90"]).abs().mean()
    m["A90_MAE_EB_model"] = (r["rate_A"] - r["A90"]).abs().mean()
    m["A90_MAE_last_season_raw"] = (r["last_A90"] - r["A90"]).abs().mean()
    m["A90_MAE_league_mean"] = (r["lg_A"] - r["A90"]).abs().mean()
    return m


def backtest(panel, test_years):
    pooled, rows = [], []
    for T in test_years:
        train_years = list(range(panel["yr"].min() + 1, T))
        models = fit_models(panel, train_years)
        pred = predict_year(panel, T, models)
        # naive baselines use each player's most recent season before T
        recent = (panel[panel["yr"] < T].sort_values("yr").groupby("Player").tail(1)
                  .set_index("Player")[["G", "A", "NPG90", "A90"]]
                  .rename(columns={"G": "G_last", "A": "A_last",
                                   "NPG90": "last_NPG90", "A90": "last_A90"}))
        tgt = panel[panel["yr"] == T].set_index("Player")[["Min", "n90", "G", "A", "GA", "NPG90", "A90"]]
        j = pred.join(recent).join(tgt, how="inner")
        pooled.append(j)
        met = fold_metrics(j)
        met.update(fold=f"{T}-{str(T + 1)[2:]}",
                   tuned_decay_k_NPG=str(models["rate"]["NPG"]),
                   tuned_decay_k_A=str(models["rate"]["A"]),
                   ridge_alpha=models["ridge_alpha"])
        rows.append(met)
    pooled_met = fold_metrics(pd.concat(pooled))
    pooled_met["fold"] = "pooled"
    rows.append(pooled_met)
    return pd.DataFrame(rows).set_index("fold").T


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="PL_Forwards_5Years.xlsx")
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    raw = load_raw(args.input)
    panel = build_panel(raw)
    last_yr = int(panel["yr"].max())
    print(f"{len(raw)} raw rows -> {len(panel)} player-seasons, {panel['Player'].nunique()} players")
    print(f"merged split-season rows: {len(raw) - len(panel)}")
    panel.to_csv(out / "panel_clean.csv", index=False)

    # Backtest: predict the last two seasons from earlier data only
    test_years = [last_yr - 1, last_yr]
    bt = backtest(panel, test_years)
    bt.to_csv(out / "backtest_metrics.csv")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print("\n=== Backtest (rolling origin) ===")
        print(bt)

    # Final fit and next-season forecast
    next_yr = last_yr + 1
    models = fit_models(panel, list(range(panel["yr"].min() + 1, next_yr)))
    fc = predict_year(panel, next_yr, models)
    fc = fc[(fc["last_yr"] == last_yr) & (fc["Min_last_season"] >= 180)].copy()
    cur = panel[panel["yr"] == last_yr].set_index("Player")
    fc["Nation"] = cur["Nation"]
    fc["low_sample"] = (fc["n_prior"] == 1) & (fc["Min_last_season"] < 900)
    fc = fc.reset_index().rename(columns={"Squad_last": "Squad", "age": "Age_next"})
    fc["Age_next"] = fc["Age_next"].astype(int)
    cols = ["Player", "Nation", "Squad", "Age_next", "n_prior", "Min_last_season", "proj_min",
            "rate_NPG", "rate_A", "rate_PA", "exp_goals", "goals_p10", "goals_p50", "goals_p90",
            "exp_ast", "ast_p10", "ast_p50", "ast_p90", "exp_ga", "ga_p10", "ga_p50", "ga_p90",
            "P_goals_ge_5", "P_goals_ge_10", "P_goals_ge_15", "P_goals_ge_20", "low_sample"]
    fc = fc[cols].sort_values("exp_goals", ascending=False)
    fc.to_csv(out / f"forecast_{next_yr}_{str(next_yr + 1)[2:]}.csv", index=False)
    print(f"\n=== Top 15 projected scorers {next_yr}-{str(next_yr + 1)[2:]} "
          "(assumes the player stays in the PL as a forward) ===")
    show = ["Player", "Squad", "Age_next", "proj_min", "exp_goals", "goals_p10", "goals_p90",
            "exp_ast", "P_goals_ge_15"]
    with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 200):
        print(fc[show].head(15).to_string(index=False))
    print(f"\nTuned params {next_yr}: {models['rate']}, ridge alpha {models['ridge_alpha']}")
    print(f"Files written to {out.resolve()}")


if __name__ == "__main__":
    main()
