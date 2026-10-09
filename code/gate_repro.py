"""Baseline comparison at the 2010 origin for the archived v1 ensemble hindcast.

Scores the ensemble median (data/v1_hindcast_county.csv) against persistence,
trend and shrunk-trend baselines, with 2,000 state-clustered bootstrap
replicates. Inputs: v1 hindcast, county panel and archived metrics in data/.
Outputs: results/gate2010_*.csv and results/gate2010_summary.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

LAMBDAS = np.round(np.arange(0.0, 1.01, 0.1), 2)
BOOT_REPS = 2000
BOOT_SEED = 20261006


def _metrics(pred, act):
    pred, act = np.asarray(pred, float), np.asarray(act, float)
    err = pred - act
    bias = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    err_sd = float(np.sqrt(max(np.mean(err ** 2) - bias ** 2, 0.0)))
    if np.std(pred) > 0:
        pear = float(np.corrcoef(pred, act)[0, 1])
        spear = float(pd.Series(pred).corr(pd.Series(act), method="spearman"))
    else:
        pear, spear = np.nan, np.nan
    return dict(rmse=rmse, mae=float(np.mean(np.abs(err))), bias=bias,
                err_sd=err_sd, bias_sq_share_of_mse=float(bias ** 2 / rmse ** 2),
                pearson=pear, spearman=spear, n=int(len(err)))


def run(data_dir, out_dir):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    hc = pd.read_csv(data_dir / "v1_hindcast_county.csv", dtype={"GeoFIPS": str})
    hc["GeoFIPS"] = hc["GeoFIPS"].str.zfill(5)
    panel = pd.read_csv(data_dir / "county_panel.csv", dtype={"GeoFIPS": str})
    arch = pd.read_csv(data_dir / "v1_hindcast_metrics.csv")
    states_sorted = sorted(hc["state"].unique())
    assert len(hc) == 841 and len(states_sorted) == 9

    # check the trend column against log(2010) - log(2001) from the panel
    piv = {}
    for col, name in [("pop", "pop"), ("rgdp_k2017", "gdp")]:
        pp = panel.pivot(index="GeoFIPS", columns="year", values=col)
        piv[name] = np.log(pp[2010]) - np.log(pp[2001])
    order = hc["GeoFIPS"].values
    trend_check = {}
    for name, col in [("pop", "growth_pop_trend_baseline"),
                      ("gdp", "growth_gdp_trend_baseline")]:
        pt = piv[name].reindex(order).to_numpy()
        trend_check[name] = float(np.max(np.abs(pt - hc[col].to_numpy())))
        assert trend_check[name] < 1e-9

    OUTCOMES = {
        "pop": dict(act="growth_pop_actual_10_19", sim="growth_pop_sim_median",
                    trend="growth_pop_trend_baseline", panel_col="pop"),
        "gdp": dict(act="growth_gdp_actual_10_19", sim="growth_gdp_sim_median",
                    trend="growth_gdp_trend_baseline", panel_col="rgdp_k2017"),
    }
    verify = {}
    for out, cfg in OUTCOMES.items():
        act = hc[cfg["act"]].to_numpy()
        for key, pred in [("simulation_ensemble_median_expected_hazard", hc[cfg["sim"]].to_numpy()),
                          ("persistence_zero_growth", np.zeros(len(hc))),
                          ("county_trend_extrapolation_2002_2010_mean_x9", hc[cfg["trend"]].to_numpy())]:
            var = "POP growth 2010-19 (log)" if out == "pop" else "GDP growth 2010-19 (log)"
            row = arch[(arch.variable == var) & (arch.model == key)].iloc[0]
            m = _metrics(pred, act)
            verify[f"{out}:{key}"] = dict(rmse_absdiff=float(abs(m["rmse"] - row.rmse)),
                                          bias_absdiff=float(abs(m["bias"] - row.bias)))
            assert abs(m["rmse"] - row.rmse) < 1e-6 and abs(m["bias"] - row.bias) < 1e-6

    # lambda validation inside the calibration window
    panel = panel.sort_values(["GeoFIPS", "year"])
    panel["state"] = panel["GeoFIPS"].map(dict(zip(hc.GeoFIPS, hc.state)))
    val_rows, lam_val = [], {}
    SPLITS = [("A_freeze2005", list(range(2002, 2006)), 2005, 5),
              ("B_freeze2006", list(range(2002, 2007)), 2006, 4),
              ("C_freeze2007", list(range(2002, 2008)), 2007, 3)]
    for out, cfg in OUTCOMES.items():
        col = cfg["panel_col"]
        panel["_l"] = np.log(panel[col])
        panel["_g"] = panel.groupby("GeoFIPS")["_l"].diff()
        lam_val[out] = {}
        for sname, train_years, freeze, horizon in SPLITS:
            tr = panel[panel.year.isin(train_years)].groupby("GeoFIPS")["_g"].mean()
            lvl_f = panel[panel.year == freeze].set_index("GeoFIPS")["_l"]
            lvl_10 = panel[panel.year == 2010].set_index("GeoFIPS")["_l"]
            vact = lvl_10 - lvl_f
            idx = tr.index.intersection(vact.index)
            tr, vact = tr.loc[idx], vact.loc[idx]
            st = panel[panel.year == 2010].set_index("GeoFIPS")["state"].loc[idx]
            anchors = {"regional": pd.Series(tr.mean(), index=idx),
                       "state": tr.groupby(st).transform("mean")}
            for aname, anch in anchors.items():
                curve = []
                for lam in LAMBDAS:
                    pred = horizon * (lam * tr + (1 - lam) * anch)
                    curve.append(float(np.sqrt(np.mean((pred - vact) ** 2))))
                best = int(np.argmin(curve))
                val_rows.append(dict(outcome=out, split=sname, anchor=aname,
                                     best_lambda=float(LAMBDAS[best]),
                                     best_rmse=curve[best],
                                     rmse_at_lambda0=curve[0], rmse_at_lambda1=curve[-1],
                                     curve=";".join(f"{x:.4f}" for x in curve)))
                if sname == "A_freeze2005":
                    lam_val[out][aname] = float(LAMBDAS[best])
        g = panel[panel.year.between(2002, 2010)]
        grp = g.groupby("GeoFIPS")["_g"]
        means, within_var = grp.mean(), grp.var(ddof=1)
        sig_w2 = float(within_var.mean())
        sig_b2 = max(float(means.var(ddof=1)) - sig_w2 / 9.0, 0.0)
        lam_eb = sig_b2 / (sig_b2 + sig_w2 / 9.0) if (sig_b2 + sig_w2) > 0 else 0.0
        lam_val[out]["EB"] = float(lam_eb)
        lam_val[out]["EB_var_of_means"] = float(means.var(ddof=1))
        lam_val[out]["EB_noise"] = float(sig_w2 / 9.0)
        val_rows.append(dict(outcome=out, split="EB_variance_ratio", anchor="n/a",
                             best_lambda=float(lam_eb), best_rmse=np.nan,
                             rmse_at_lambda0=np.nan, rmse_at_lambda1=np.nan, curve=""))
        panel = panel.drop(columns=["_l", "_g"])
    val_df = pd.DataFrame(val_rows)
    consensus = {}
    for out in OUTCOMES:
        consensus[out] = {}
        for aname in ["regional", "state"]:
            sel = val_df[(val_df.outcome == out) & (val_df.anchor == aname) &
                         (val_df.split != "EB_variance_ratio")]
            consensus[out][aname] = float(np.median(sel.best_lambda))

    # baseline predictions
    preds, meta = {}, {}
    for out, cfg in OUTCOMES.items():
        act = hc[cfg["act"]].to_numpy()
        trend = hc[cfg["trend"]].to_numpy()
        reg_ex = np.full(len(hc), trend.mean())
        state_ex = hc.groupby("state")[cfg["trend"]].transform("mean").to_numpy()
        reg_or = np.full(len(hc), act.mean())
        state_or = hc.groupby("state")[cfg["act"]].transform("mean").to_numpy()

        def add(name, pred, **kw):
            preds[(out, name)] = np.asarray(pred, float)
            meta[name] = kw

        add("simulator_ensemble_median", hc[cfg["sim"]].to_numpy(), kind="simulator", oracle=False)
        add("persistence_zero_growth", np.zeros(len(hc)), kind="reference", oracle=False)
        add("raw_county_trend_lambda1", trend, kind="reference", oracle=False)
        add("regional_mean_EXANTE_calibration", reg_ex, kind="mean", oracle=False)
        add("state_mean_EXANTE_calibration", state_ex, kind="mean", oracle=False)
        add("regional_mean_ORACLE_realized", reg_or, kind="mean", oracle=True)
        add("state_mean_ORACLE_realized", state_or, kind="mean", oracle=True)
        for aname, anch_ex in [("regional", reg_ex), ("state", state_ex)]:
            lamA = lam_val[out][aname]
            add(f"shrunk_{aname}_EXANTE_lambdaA{lamA:.1f}", lamA * trend + (1 - lamA) * anch_ex,
                kind="shrunk_nonoracle_primary", oracle=False, lam=lamA, anchor=aname)
            lamC = consensus[out][aname]
            add(f"shrunk_{aname}_EXANTE_lambdaConsensus{lamC:.1f}",
                lamC * trend + (1 - lamC) * anch_ex,
                kind="shrunk_nonoracle_consensus", oracle=False, lam=lamC, anchor=aname)
    curve_rows, oracle_best = [], {}
    for out, cfg in OUTCOMES.items():
        act = hc[cfg["act"]].to_numpy()
        trend = hc[cfg["trend"]].to_numpy()
        variants = {
            "regional_EXANTE": np.full(len(hc), trend.mean()),
            "state_EXANTE": hc.groupby("state")[cfg["trend"]].transform("mean").to_numpy(),
            "regional_ORACLE": np.full(len(hc), act.mean()),
            "state_ORACLE": hc.groupby("state")[cfg["act"]].transform("mean").to_numpy(),
        }
        for vname, anch in variants.items():
            rms = []
            for lam in LAMBDAS:
                m = _metrics(lam * trend + (1 - lam) * anch, act)
                rms.append(m["rmse"])
                curve_rows.append(dict(outcome=out, variant=vname, lam=float(lam), **m,
                                       oracle_anchor=("ORACLE" in vname)))
            best_i = int(np.argmin(rms))
            oracle_best[(out, vname)] = (float(LAMBDAS[best_i]), float(rms[best_i]))
            if "ORACLE" in vname:
                nm = f"shrunk_{vname}_lambdaOracleBest{LAMBDAS[best_i]:.1f}"
                preds[(out, nm)] = LAMBDAS[best_i] * trend + (1 - LAMBDAS[best_i]) * anch
                meta[nm] = dict(kind="shrunk_oracle_best", oracle=True,
                                lam=float(LAMBDAS[best_i]), anchor=vname)
    curve_df = pd.DataFrame(curve_rows)
    best_nonoracle, best_honest = {}, {}
    for out in OUTCOMES:
        cands = {nm: _metrics(p, hc[OUTCOMES[out]["act"]].to_numpy())["rmse"]
                 for (o, nm), p in preds.items()
                 if o == out and meta[nm].get("kind") == "shrunk_nonoracle_primary"}
        best_nonoracle[out] = min(cands, key=cands.get)
        honest = {nm: _metrics(p, hc[OUTCOMES[out]["act"]].to_numpy())["rmse"]
                  for (o, nm), p in preds.items()
                  if o == out and not meta[nm].get("oracle") and nm != "simulator_ensemble_median"}
        best_honest[out] = min(honest, key=honest.get)
    rows = []
    for (out, name), pred in sorted(preds.items()):
        m = _metrics(pred, hc[OUTCOMES[out]["act"]].to_numpy())
        rows.append(dict(outcome=out, model=name, kind=meta[name].get("kind"),
                         oracle=meta[name].get("oracle"), lam=meta[name].get("lam"),
                         anchor=meta[name].get("anchor"), **m))
    metrics_df = pd.DataFrame(rows).sort_values(["outcome", "rmse"])

    # state-clustered bootstrap of RMSE differences
    rng = np.random.default_rng(BOOT_SEED)
    state_idx = {s: np.where(hc["state"].to_numpy() == s)[0] for s in states_sorted}
    boot_models = {}
    for out in OUTCOMES:
        names = ["persistence_zero_growth", "raw_county_trend_lambda1",
                 "regional_mean_EXANTE_calibration", "state_mean_EXANTE_calibration",
                 "regional_mean_ORACLE_realized", "state_mean_ORACLE_realized",
                 best_nonoracle[out]]
        names += [nm for (o, nm) in preds if o == out and
                  meta[nm].get("kind") in ("shrunk_nonoracle_consensus", "shrunk_oracle_best")]
        boot_models[out] = sorted(set(names))
    boot_rows, per_state_rows = [], []
    for out, cfg in OUTCOMES.items():
        act = hc[cfg["act"]].to_numpy()
        sim_pred = preds[(out, "simulator_ensemble_median")]

        def sse_by_state(pred):
            err2 = (pred - act) ** 2
            return np.array([err2[state_idx[s]].sum() for s in states_sorted]), \
                np.array([len(state_idx[s]) for s in states_sorted])

        sim_sse, counts = sse_by_state(sim_pred)
        sim_rmse_full = float(np.sqrt(sim_sse.sum() / counts.sum()))
        for name in boot_models[out]:
            base_sse, _ = sse_by_state(preds[(out, name)])
            base_rmse_full = float(np.sqrt(base_sse.sum() / counts.sum()))
            draws = rng.integers(0, len(states_sorted), size=(BOOT_REPS, len(states_sorted)))
            S = len(states_sorted)
            w = np.bincount((np.arange(BOOT_REPS)[:, None] * S + draws).ravel(),
                            minlength=BOOT_REPS * S).reshape(BOOT_REPS, S).astype(float)
            n_star = w @ counts
            diffs = np.sqrt((w @ sim_sse) / n_star) - np.sqrt((w @ base_sse) / n_star)
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            boot_rows.append(dict(
                outcome=out, comparator=name,
                comparator_oracle=bool(meta[name].get("oracle")),
                rmse_sim=sim_rmse_full, rmse_comparator=base_rmse_full,
                rmse_diff_sim_minus_comparator=sim_rmse_full - base_rmse_full,
                boot_ci_lo=float(lo), boot_ci_hi=float(hi),
                boot_share_diff_lt0=float(np.mean(diffs < 0)),
                ci_excludes_zero=bool(lo > 0 or hi < 0), reps=BOOT_REPS,
                clusters=len(states_sorted), seed=BOOT_SEED))
            if name == best_nonoracle[out]:
                for s in states_sorted:
                    idx = state_idx[s]
                    per_state_rows.append(dict(
                        outcome=out, state=s, n=int(len(idx)),
                        rmse_sim=float(np.sqrt(np.mean((sim_pred[idx] - act[idx]) ** 2))),
                        rmse_best_nonoracle_shrunk=float(np.sqrt(np.mean(
                            (preds[(out, name)][idx] - act[idx]) ** 2)))))
    boot_df = pd.DataFrame(boot_rows)

    metrics_df.to_csv(out_dir / "gate2010_baseline_metrics.csv", index=False)
    curve_df.to_csv(out_dir / "gate2010_lambda_curves.csv", index=False)
    val_df.to_csv(out_dir / "gate2010_lambda_validation.csv", index=False)
    boot_df.to_csv(out_dir / "gate2010_bootstrap_ci.csv", index=False)
    pd.DataFrame(per_state_rows).to_csv(out_dir / "gate2010_per_state_diffs.csv", index=False)
    summary = dict(
        trend_column_vs_panel_max_abs_diff=trend_check,
        metric_convention_verification=verify,
        lambda_validation_primary_splitA={k: {kk: vv for kk, vv in v.items()}
                                          for k, v in lam_val.items()},
        lambda_consensus=consensus,
        best_nonoracle_shrunk=best_nonoracle,
        best_honest_baseline_overall=best_honest)
    (out_dir / "gate2010_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
