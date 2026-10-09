"""Sensitivity grids for two assumed scenarios.

S1: an assumed annual growth penalty applied to counties flagged by a rule
(2010 goods-producing share at or above the regional median).
S2: a reduction in distance between corridor counties, which changes market
potential; the GDP level gain is expm1(elasticity * change in log market
potential) over a range of elasticities. Inputs are 2019 GDP and distances.
"""
import numpy as np
import pandas as pd

ELASTICITIES = (0.0014, 0.01, 0.03, 0.05)
DELTAS = (0.005, 0.01, 0.02)
YEARS_FWD = 16


def s1_grid(log_gdp_2019, gdp_2019, flagged):
    flagged = np.asarray(flagged, float)
    rows = []
    for dlt in DELTAS:
        pen = -dlt * YEARS_FWD * flagged
        lvl = np.expm1(pen)
        rows.append(dict(
            delta_pp_per_year=dlt * 100,
            flagged_median_pct=float(100 * np.median(lvl[flagged > 0])),
            flagged_p10_pct=float(100 * np.percentile(lvl[flagged > 0], 10)),
            flagged_p90_pct=float(100 * np.percentile(lvl[flagged > 0], 90)),
            region_gdp_change_pct=float(100 * (np.sum(gdp_2019 * np.exp(pen))
                                               / np.sum(gdp_2019) - 1)),
            n_flagged=int(flagged.sum())))
    return pd.DataFrame(rows)


def s1_grid2d(gdp_2019, flagged, deltas=DELTAS, years_grid=(4, 8, 12, 16)):
    """S1 heatmap tables: regional GDP change and flagged-county median change (%)."""
    flagged = np.asarray(flagged, float)
    reg, med = [], []
    for dlt in deltas:
        rrow, mrow = [], []
        for yrs in years_grid:
            pen = -dlt * yrs * flagged
            rrow.append(100 * (np.sum(gdp_2019 * np.exp(pen)) / np.sum(gdp_2019) - 1))
            mrow.append(100 * np.median(np.expm1(pen[flagged > 0])))
        reg.append(rrow)
        med.append(mrow)
    return (pd.DataFrame(reg, index=[f"{d*100:.1f} pp/yr" for d in deltas],
                         columns=[f"{y} yr" for y in years_grid]),
            pd.DataFrame(med, index=[f"{d*100:.1f} pp/yr" for d in deltas],
                         columns=[f"{y} yr" for y in years_grid]))


def s2_grid2d(dist, gdp_levels, corridor_mask, mults=(0.9, 0.8, 0.7, 0.6),
              elasticities=ELASTICITIES):
    """S2 heatmap table: median corridor GDP gain (%) by elasticity and distance multiplier."""
    cm = np.asarray(corridor_mask) > 0
    w0 = 1.0 / np.maximum(dist, 10.0)
    np.fill_diagonal(w0, 0.0)
    mp0 = w0 @ gdp_levels
    rows = []
    for e in elasticities:
        row = []
        for mult in mults:
            d1 = np.where(np.outer(cm, cm) & (dist > 0), dist * mult, dist)
            w1 = 1.0 / np.maximum(d1, 10.0)
            np.fill_diagonal(w1, 0.0)
            dln = np.log(w1 @ gdp_levels) - np.log(mp0)
            row.append(100 * np.median(np.expm1(e * dln[cm])))
        rows.append(row)
    return pd.DataFrame(rows, index=[f"eps = {e:g}" for e in elasticities],
                        columns=[f"x {m:.1f}" for m in mults])


def s2_band(dist, gdp_levels, corridor_mask, mult=0.7):
    w0 = 1.0 / np.maximum(dist, 10.0)
    np.fill_diagonal(w0, 0.0)
    cm = np.asarray(corridor_mask) > 0
    d1 = np.where(np.outer(cm, cm) & (dist > 0), dist * mult, dist)
    w1 = 1.0 / np.maximum(d1, 10.0)
    np.fill_diagonal(w1, 0.0)
    dln = np.log(w1 @ gdp_levels) - np.log(w0 @ gdp_levels)
    rows = []
    for e in ELASTICITIES:
        gain = np.expm1(e * dln)
        rows.append(dict(
            elasticity=e,
            median_dlnMP_corridor=float(np.median(dln[cm])),
            median_corridor_gain_pct=float(100 * np.median(gain[cm])),
            max_corridor_gain_pct=float(100 * gain[cm].max()),
            region_gain_pct=float(100 * np.sum(gdp_levels * gain) / gdp_levels.sum()),
            n_corridor=int(cm.sum())))
    return pd.DataFrame(rows), dln
