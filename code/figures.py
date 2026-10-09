"""Figures 1 to 4: error maps, predicted vs actual, rolling skill, scenario grids.

Inputs: county geometries (Albers Equal Area, EPSG:5070), hindcast county
results, rolling metrics, scenario tables. Outputs: PDF and 300-dpi PNG files.
"""
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "figure.dpi": 300,
})
SOURCE_MAP = "Source: BEA Regional Accounts; author's calculations. Boundaries: Census Bureau (2020), BEA geography."
SOURCE_CHART = "Source: author's calculations (replication package, results/rolling_metrics.csv)."


def _save(fig, outdir, stem):
    outdir = Path(outdir)
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [f"{stem}.pdf", f"{stem}.png"]


def _map_decor(ax, fig):
    """Add a 200 km scale bar and a north arrow (projected metre coordinates)."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xlim(x0 - 0.01 * (x1 - x0), x1 + 0.01 * (x1 - x0))
    ax.set_ylim(y0 - 0.01 * (y1 - y0), y1 + 0.035 * (y1 - y0))
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bx, by = x0 + 0.03 * (x1 - x0), y0 + 0.03 * (y1 - y0)
    L = 200_000
    ax.plot([bx, bx + L], [by, by], color="black", lw=3.0, solid_capstyle="butt",
            clip_on=False, zorder=10)
    ax.text(bx + L / 2, by + 0.016 * (y1 - y0), "200 km", ha="center",
            va="bottom", fontsize=8.5)
    nx = x1 - 0.035 * (x1 - x0)
    ax.annotate("", xy=(nx, y1 - 0.012 * (y1 - y0)),
                xytext=(nx, y1 - 0.075 * (y1 - y0)),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3),
                clip_on=False)
    ax.text(nx, y1 - 0.095 * (y1 - y0), "N", ha="center", va="top",
            fontsize=11, fontweight="bold")
    ax.set_axis_off()


def fig1_choropleths(geoms, county_df, outdir):
    """Maps of hindcast error (sim - actual), 2010 origin, state x year FE, conditional policy."""
    g = geoms.copy()
    g["state2"] = g.GeoFIPS.str[:2]
    states = g.dissolve(by="state2")
    df = county_df.copy()
    df["err_gdp"] = df.growth_gdp_sim - df.growth_gdp_actual
    df["err_pop"] = df.growth_pop_sim - df.growth_pop_actual
    g = g.merge(df[["GeoFIPS", "err_gdp", "err_pop"]], on="GeoFIPS", how="left")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, col, ttl in zip(axes, ("err_gdp", "err_pop"),
                            ("(a) Real GDP growth error", "(b) Population growth error")):
        lim = float(np.nanpercentile(np.abs(g[col]), 98))
        g.plot(column=col, ax=ax, cmap="RdBu_r", vmin=-lim, vmax=lim,
               edgecolor="0.75", linewidth=0.08, legend=True,
               legend_kwds=dict(label="Error (log points, sim − actual)",
                                shrink=0.85))
        states.boundary.plot(ax=ax, color="0.4", linewidth=0.55)
        ax.set_title(ttl)
        _map_decor(ax, fig)
    fig.suptitle("Figure 1. Frozen-hindcast errors, 2010 origin, state×year FE, conditional policy",
                 fontsize=11)
    fig.text(0.5, -0.015, SOURCE_MAP, ha="center", fontsize=8)
    return _save(fig, outdir, "fig1_error_choropleths")


def fig2_scatter(county_df, metrics_row_gdp, metrics_row_pop, outdir):
    """Scatter of simulated against actual growth for GDP and population."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3))
    for ax, (a, s, ttl, mr) in zip(axes, [
            ("growth_gdp_actual", "growth_gdp_sim", "(a) Real GDP growth", metrics_row_gdp),
            ("growth_pop_actual", "growth_pop_sim", "(b) Population growth", metrics_row_pop)]):
        ax.scatter(county_df[a], county_df[s], s=7, alpha=0.5, color="#2c5f8a", edgecolors="none")
        lim = [min(county_df[a].min(), county_df[s].min()),
               max(county_df[a].max(), county_df[s].max())]
        ax.plot(lim, lim, "k--", lw=0.9)
        ax.set_xlabel("Actual log growth 2010–2019")
        ax.set_ylabel("Simulated log growth (conditional)")
        ax.set_title(f"{ttl}\ncorr = {mr['pearson']:.2f}, demeaned RMSE = {mr['demeaned_rmse']:.3f}")
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Figure 2. Predicted against actual growth, 2010 origin, state×year FE",
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.text(0.5, 0.005, SOURCE_CHART, ha="center", fontsize=8)
    return _save(fig, outdir, "fig2_pred_vs_actual")


def fig3_rolling_skill(metrics_df, outdir):
    """Demeaned RMSE by forecast origin for the simulation, GNN and baselines."""
    sim = metrics_df[(metrics_df.model == "simulation") &
                     (metrics_df.spec == "state_year_fe") &
                     (metrics_df.policy == "conditional")]
    gnn = metrics_df[(metrics_df.spec == "neural_gnn") &
                     (metrics_df.policy == "conditional")]
    base = metrics_df[metrics_df.spec == "baseline"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for ax, out_name, ttl in zip(axes, ("gdp", "pop"), ("(a) Real GDP", "(b) Population")):
        s = sim[sim.outcome == out_name].sort_values("origin")
        ax.plot(s.origin, s.demeaned_rmse, marker="o", lw=1.6, color="#2c5f8a",
                label="Simulation (state×year FE, conditional)")
        if len(gnn):
            gn = gnn[gnn.outcome == out_name].sort_values("origin")
            ax.plot(gn.origin, gn.demeaned_rmse, marker="^", lw=1.5,
                    color="#009E73", label="Graph neural network (conditional)")
        b = base[base.outcome == out_name]
        shr = (b[b.model.str.startswith("shrunk")].groupby("origin").demeaned_rmse.min())
        ax.plot(shr.index, shr.values, marker="D", lw=1.4, color="#b35806",
                label="Best honest shrunk trend")
        per = b[b.model == "persistence"].set_index("origin")
        ax.plot(per.index, per.demeaned_rmse, marker="s", lw=1.2, color="0.45",
                label="Persistence (= SD of actual growth)")
        ax.set_xlabel("Forecast origin (hindcast to 2019)")
        ax.set_ylabel("Demeaned RMSE (log points)")
        ax.set_title(ttl)
        ax.set_xticks([2006, 2008, 2010, 2012])
        ax.legend(loc="upper right" if out_name == "pop" else "lower left",
                  frameon=False)
        ax.grid(axis="y", color="0.88", lw=0.7)
    fig.suptitle("Figure 3. Rolling-origin allocation skill against honest baselines", fontsize=11)
    fig.text(0.5, -0.015, SOURCE_CHART, ha="center", fontsize=8)
    return _save(fig, outdir, "fig3_rolling_skill")


def _heatmap(ax, mat, title):
    vals = mat.to_numpy(float)
    im = ax.imshow(vals, cmap="cividis", aspect="auto")
    ax.set_xticks(range(len(mat.columns)), mat.columns)
    ax.set_yticks(range(len(mat.index)), mat.index)
    mid = 0.5 * (vals.min() + vals.max())
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if v < mid else "black")
    ax.set_title(title)
    return im


def fig4_scenario_grids(s1_reg, s1_med, s2_grid, outdir):
    """Heatmaps of the S1 and S2 scenario grids."""
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4))
    im = _heatmap(axes[0, 0], s1_reg, "(a) S1 drought grid: regional GDP change (%)")
    axes[0, 0].set_xlabel("Horizon"); axes[0, 0].set_ylabel("Penalty on flagged counties")
    fig.colorbar(im, ax=axes[0, 0], shrink=0.9, label="%")
    im = _heatmap(axes[0, 1], s1_med, "(b) S1: flagged-county median GDP change (%)")
    axes[0, 1].set_xlabel("Horizon"); axes[0, 1].set_ylabel("Penalty on flagged counties")
    fig.colorbar(im, ax=axes[0, 1], shrink=0.9, label="%")
    im = _heatmap(axes[1, 0], s2_grid, "(c) S2 corridor: median corridor GDP gain (%)")
    axes[1, 0].set_xlabel("Corridor distance multiplier"); axes[1, 0].set_ylabel("Elasticity ε")
    fig.colorbar(im, ax=axes[1, 0], shrink=0.9, label="%")
    axes[1, 1].axis("off")
    axes[1, 1].text(0.02, 0.95,
                    "Reading the grids\n\n"
                    "S1: the penalty δ is an assumption, not an\n"
                    "estimated drought elasticity. Outcomes scale\n"
                    "almost linearly in δ — the slope is the content.\n\n"
                    "S2: gains are expm1(ε·Δln MP) at a median\n"
                    "corridor Δln MP of 0.107 (×0.7 distances).\n"
                    "Even the largest cell of the grid (ε = 0.05,\n"
                    "distances ×0.6) leaves the median corridor\n"
                    "county under +1%.",
                    va="top", ha="left", fontsize=9)
    fig.suptitle("Figure 4. Scenario sensitivity grids — SIMULATION OUTPUT, not forecasts",
                 fontsize=11)
    fig.text(0.5, -0.005, "Source: author's calculations; assumptions as stated in Section 5.",
             ha="center", fontsize=8)
    fig.tight_layout()
    return _save(fig, outdir, "fig4_scenario_grids")


def build_all(geoms_path, county_csv, metrics_csv, s1_reg, s1_med, s2_grid, outdir):
    geoms = gpd.read_file(geoms_path)
    geoms["GeoFIPS"] = geoms.GeoFIPS.astype(str).str.zfill(5)
    county = pd.read_csv(county_csv, dtype={"GeoFIPS": str})
    county["GeoFIPS"] = county.GeoFIPS.str.zfill(5)
    metrics = pd.read_csv(metrics_csv)
    sel = metrics[(metrics.origin == 2010) & (metrics.spec == "state_year_fe") &
                  (metrics.policy == "conditional") & (metrics.model == "simulation")]
    row_g = sel[sel.outcome == "gdp"].iloc[0]
    row_p = sel[sel.outcome == "pop"].iloc[0]
    made = []
    made += fig1_choropleths(geoms, county, outdir)
    made += fig2_scatter(county, row_g, row_p, outdir)
    made += fig3_rolling_skill(metrics, outdir)
    made += fig4_scenario_grids(s1_reg, s1_med, s2_grid, outdir)
    return made
