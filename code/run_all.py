#!/usr/bin/env python3
"""Runs the full analysis: python run_all.py

Reads the files in data/ and writes results/: 2010 baseline comparison for the
v1 hindcast, rolling-origin hindcasts and inference for four specifications,
the neural benchmark, the NRI sensitivity, the spatial-weight diagnostic,
scenario grids, figures, and run_manifest.json.
"""
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluation as ev
import gate_repro
import model as M
import neural as NN
import scenarios as SC

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RES = ROOT / "results"
FIG = RES / "figures"
ORIGINS = (2006, 2008, 2010, 2012)
SPECS = ("pooled", "state_fe", "state_year_fe", "county_fe")
POLICIES = ("mean", "conditional")
EST_START = 2003


def sha256(p):
    h = hashlib.sha256()
    h.update(Path(p).read_bytes())
    return h.hexdigest()


def main():
    t_start = time.time()
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    panel = pd.read_csv(DATA / "county_panel.csv", dtype={"GeoFIPS": str})
    static = pd.read_csv(DATA / "county_static.csv", dtype={"GeoFIPS": str})
    dist = np.load(DATA / "dist_km.npy")
    order = static.sort_values("GeoFIPS").index.to_numpy()
    if not np.array_equal(order, np.arange(len(static))):
        static = static.iloc[order].reset_index(drop=True)
        dist = dist[np.ix_(order, order)]
    d = M.Data(panel, static, dist)
    g, p = d.growth()
    state = d.state

    # 1. 2010 baseline comparison for the archived v1 hindcast
    gate_summary = gate_repro.run(DATA, RES)
    print("[1/6] gate reproduction done")

    # 2. rolling-origin evaluation
    metric_rows, infer_rows, lam_rows, coef_rows = [], [], [], []
    cv_rows, nn_info = [], dict(selected_wd={}, train_seconds_gnn=0.0,
                                train_seconds_mlp=0.0, per_origin_seconds={})
    headline_county = None
    A_graph = NN.graph(d)
    Z_head, _ = d.zmat(M.HEADLINE_STATICS)
    for origin in ORIGINS:
        H = 2019 - origin
        levels = {"gdp": d.lg, "pop": d.lp, "goods_share": d.sg, "services_share": d.ss}
        lam_primary, lam_eb = {}, {}
        for out_name, lvl in levels.items():
            rows_v, prim, eb = ev.select_lambda(lvl, d.yidx, origin, d.state_codes)
            lam_primary[out_name] = prim
            lam_eb[out_name] = eb
            for r in rows_v:
                lam_rows.append(dict(origin=origin, outcome=out_name, **r))
            lam_rows.append(dict(origin=origin, outcome=out_name, freeze=-1,
                                 anchor="EB", primary_split=False,
                                 best_lambda=eb["lambda_eb"], best_rmse=np.nan,
                                 rmse_lambda0=eb["var_of_county_means"],
                                 rmse_lambda1=eb["sampling_noise"], curve="",
                                 var_of_county_means=eb["var_of_county_means"],
                                 sampling_noise=eb["sampling_noise"]))
        base_preds, base_trend = {}, {}
        actuals = {}
        for out_name, lvl in levels.items():
            bp, tr = ev.build_baselines(lvl, d.yidx, origin, state, d.state_codes,
                                        lam_primary[out_name])
            base_preds[out_name] = bp
            base_trend[out_name] = tr
            actuals[out_name] = lvl[:, d.yidx[2019]] - lvl[:, d.yidx[origin]]
        # share outcomes scored only where both endpoint shares are observed
        share_mask = {}
        for out_name, obs in (("goods_share", d.sg_obs), ("services_share", d.sg_obs)):
            share_mask[out_name] = obs[:, d.yidx[origin]] & obs[:, d.yidx[2019]]
        # baseline rows (policy independent)
        for out_name in levels:
            act = actuals[out_name]
            mask = share_mask.get(out_name)
            for bname, bp in base_preds[out_name].items():
                a_, p_ = (act[mask], bp[mask]) if mask is not None else (act, bp)
                m = ev.metrics(p_, a_)
                metric_rows.append(dict(origin=origin, horizon=H, spec="baseline",
                                        policy="n/a", outcome=out_name, model=bname,
                                        n=int(len(a_)), **{k: round(v, 5) for k, v in m.items()},
                                        state_demeaned_rmse=round(
                                            ev.state_demeaned_rmse(p_, a_, state[mask] if mask is not None else state), 5)))
        # fit and hindcast; clustered standard errors only at origin 2010,
        # where the coefficients are saved
        frame = M.build_frame(d, EST_START, origin, M.HEADLINE_STATICS)
        fits = {spec: M.fit(d, EST_START, origin, spec, frame=frame,
                            cluster_se=(origin == 2010)) for spec in SPECS}
        if origin == 2010:
            for spec in SPECS:
                for kind in ("gdp", "pop"):
                    rec = fits[spec][kind]
                    for term, val in rec["coef"].items():
                        coef_rows.append(dict(spec=spec, equation=kind, term=term,
                                              coef=val, se_cluster_county=rec["se"].get(term, np.nan),
                                              r2=rec["r2"], n=rec["n"]))
        sim_preds = {}
        for spec in SPECS:
            for pol in POLICIES:
                sgdp, spop, sgf, ssf = M.hindcast(d, fits[spec], origin, 2019, pol)
                preds = {"gdp": sgdp, "pop": spop,
                         "goods_share": sgf - d.sg[:, d.yidx[origin]],
                         "services_share": ssf - d.ss[:, d.yidx[origin]]}
                sim_preds[(spec, pol)] = preds
                for out_name, pr in preds.items():
                    act = actuals[out_name]
                    mask = share_mask.get(out_name)
                    a_, p_ = (act[mask], pr[mask]) if mask is not None else (act, pr)
                    m = ev.metrics(p_, a_)
                    metric_rows.append(dict(origin=origin, horizon=H, spec=spec,
                                            policy=pol, outcome=out_name, model="simulation",
                                            n=int(len(a_)), **{k: round(v, 5) for k, v in m.items()},
                                            state_demeaned_rmse=round(
                                                ev.state_demeaned_rmse(p_, a_, state[mask] if mask is not None else state), 5)))
                if origin == 2010 and spec == "state_year_fe":
                    headline_county = pd.DataFrame(dict(
                        GeoFIPS=d.fips, state=state, policy=pol,
                        growth_gdp_actual=actuals["gdp"], growth_gdp_sim=sgdp,
                        growth_pop_actual=actuals["pop"], growth_pop_sim=spop))
                    headline_county.to_csv(RES / f"hindcast_county_2010_headline_{pol}.csv",
                                           index=False)
        # inference: simulation vs best shrunk trend and vs persistence
        for spec in SPECS:
            for pol in POLICIES:
                for out_name in ("gdp", "pop", "goods_share", "services_share"):
                    act = actuals[out_name]
                    sim = sim_preds[(spec, pol)][out_name]
                    mask = share_mask.get(out_name)
                    if mask is not None:
                        act_b = {k: v[mask] for k, v in base_preds[out_name].items()}
                        act, sim = act[mask], sim[mask]
                        state_i = state[mask]
                    else:
                        act_b = base_preds[out_name]
                        state_i = state
                    shrunk_names = [k for k in act_b if k.startswith("shrunk_")]
                    key = "demeaned_rmse" if pol == "conditional" else "rmse"
                    def score(bp):
                        mm = ev.metrics(bp, act)
                        return mm["demeaned_rmse"] if key == "demeaned_rmse" else mm["rmse"]
                    best_shrunk = min(shrunk_names, key=lambda k: score(act_b[k]))
                    for comp_name, comp in ((best_shrunk, act_b[best_shrunk]),
                                            ("persistence", act_b["persistence"])):
                        if pol == "conditional":
                            diff, lo, hi = ev.clustered_bootstrap_demeaned_diff(sim, comp, act, state_i)
                        else:
                            diff, lo, hi = ev.clustered_bootstrap_rmse_diff(sim, comp, act, state_i)
                        dm, dm_p, dm_c, dm_pc = ev.diebold_mariano(
                            sim, comp, act, state_i, demeaned=(pol == "conditional"))
                        infer_rows.append(dict(origin=origin, spec=spec, policy=pol,
                                               outcome=out_name, comparator=comp_name,
                                               loss=key, diff_sim_minus_comp=round(diff, 5),
                                               boot_ci_lo=round(lo, 5), boot_ci_hi=round(hi, 5),
                                               dm=round(dm, 3), dm_p=dm_p,
                                               dm_clustered=round(dm_c, 3), dm_p_clustered=dm_pc))
        # neural benchmark: GNN, and aspatial MLP for GDP and population
        t_nn = time.time()
        levels_nn = {"gdp": d.lg, "pop": d.lp, "goods_share": d.sg,
                     "services_share": d.ss}
        gnn_preds, mlp_preds = {}, {}
        for out_name, lvl in levels_nn.items():
            obs_nn = d.sg_obs if out_name in ("goods_share", "services_share") else None
            net, stats_nn, wd_sel, cv_r, sec = NN.train_gnn(
                d, A_graph, g, p, Z_head, origin, lvl, obs=obs_nn)
            nn_info["selected_wd"][f"{origin}:{out_name}"] = wd_sel
            nn_info["train_seconds_gnn"] += sec
            for r in cv_r:
                cv_rows.append(dict(origin=origin, outcome=out_name, **r))
            raw = NN.forecast_gnn(net, stats_nn, d, g, p, Z_head, origin, H)
            gnn_preds[out_name] = {
                "mean": raw,
                "conditional": NN.conditional_shift(raw, actuals[out_name], d.state_codes)}
            if out_name in ("gdp", "pop"):
                X0 = NN.snapshot_features(d, origin, g, p, Z_head)
                mlp_fn, sec_m = NN.train_mlp_ablation(d, g, p, Z_head, origin, lvl)
                nn_info["train_seconds_mlp"] += sec_m
                raw_m = mlp_fn(X0) * H
                mlp_preds[out_name] = {
                    "mean": raw_m,
                    "conditional": NN.conditional_shift(raw_m, actuals[out_name], d.state_codes)}
        nn_info["per_origin_seconds"][str(origin)] = round(time.time() - t_nn, 1)
        for spec_nn, preds_nn in (("neural_gnn", gnn_preds),
                                  ("neural_mlp_aspatial", mlp_preds)):
            for pol in POLICIES:
                for out_name, by_pol in preds_nn.items():
                    pr = by_pol[pol]
                    act = actuals[out_name]
                    mask = share_mask.get(out_name)
                    a_, p_ = (act[mask], pr[mask]) if mask is not None else (act, pr)
                    st_ = state[mask] if mask is not None else state
                    mm = ev.metrics(p_, a_)
                    metric_rows.append(dict(origin=origin, horizon=H, spec=spec_nn,
                                            policy=pol, outcome=out_name,
                                            model=spec_nn, n=int(len(a_)),
                                            **{k: round(v, 5) for k, v in mm.items()},
                                            state_demeaned_rmse=round(
                                                ev.state_demeaned_rmse(p_, a_, st_), 5)))
        # neural inference: GNN vs best shrunk trend and persistence,
        # state_year_fe simulation vs GNN, GNN vs MLP
        def _infer(spec_row, a_pred, comp_name, b_pred, out_name, pol, act_full, mask):
            if mask is not None:
                a_i, b_i, act_i, st_i = a_pred[mask], b_pred[mask], act_full[mask], state[mask]
            else:
                a_i, b_i, act_i, st_i = a_pred, b_pred, act_full, state
            if pol == "conditional":
                diff, lo, hi = ev.clustered_bootstrap_demeaned_diff(a_i, b_i, act_i, st_i)
            else:
                diff, lo, hi = ev.clustered_bootstrap_rmse_diff(a_i, b_i, act_i, st_i)
            dm, dm_p, dm_c, dm_pc = ev.diebold_mariano(
                a_i, b_i, act_i, st_i, demeaned=(pol == "conditional"))
            infer_rows.append(dict(origin=origin, spec=spec_row, policy=pol,
                                   outcome=out_name, comparator=comp_name,
                                   loss="demeaned_rmse" if pol == "conditional" else "rmse",
                                   diff_sim_minus_comp=round(diff, 5),
                                   boot_ci_lo=round(lo, 5), boot_ci_hi=round(hi, 5),
                                   dm=round(dm, 3), dm_p=dm_p,
                                   dm_clustered=round(dm_c, 3), dm_p_clustered=dm_pc))
        for pol in POLICIES:
            key = "demeaned_rmse" if pol == "conditional" else "rmse"
            for out_name in levels_nn:
                act = actuals[out_name]
                mask = share_mask.get(out_name)
                act_b = ({k: v[mask] for k, v in base_preds[out_name].items()}
                         if mask is not None else base_preds[out_name])
                def score_nn(bp, _act=act, _mask=mask):
                    aa = _act[_mask] if _mask is not None else _act
                    mm = ev.metrics(bp, aa)
                    return mm["demeaned_rmse"] if key == "demeaned_rmse" else mm["rmse"]
                shrunk_names = [k for k in act_b if k.startswith("shrunk_")]
                best_shrunk = min(shrunk_names, key=lambda k: score_nn(act_b[k]))
                for comp_name in (best_shrunk, "persistence"):
                    _infer("neural_gnn", gnn_preds[out_name][pol], comp_name,
                           base_preds[out_name][comp_name], out_name, pol, act, mask)
                _infer("state_year_fe", sim_preds[("state_year_fe", pol)][out_name],
                       "neural_gnn", gnn_preds[out_name][pol], out_name, pol, act, mask)
                if out_name in mlp_preds:
                    _infer("neural_gnn", gnn_preds[out_name][pol],
                           "neural_mlp_aspatial", mlp_preds[out_name][pol],
                           out_name, pol, act, mask)
        if origin == 2010:
            pd.DataFrame(dict(
                GeoFIPS=d.fips, state=state,
                growth_gdp_actual=actuals["gdp"], growth_pop_actual=actuals["pop"],
                gnn_gdp_mean=gnn_preds["gdp"]["mean"],
                gnn_gdp_conditional=gnn_preds["gdp"]["conditional"],
                gnn_pop_mean=gnn_preds["pop"]["mean"],
                gnn_pop_conditional=gnn_preds["pop"]["conditional"],
                mlp_pop_mean=mlp_preds["pop"]["mean"],
                mlp_gdp_mean=mlp_preds["gdp"]["mean"])
            ).to_csv(RES / "hindcast_county_2010_neural.csv", index=False)
        print(f"      neural benchmark done ({time.time() - t_nn:.0f}s)")
        print(f"[2/6] origin {origin} done ({time.time() - t_start:.0f}s)")

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(RES / "rolling_metrics.csv", index=False)
    pd.DataFrame(infer_rows).to_csv(RES / "rolling_inference.csv", index=False)
    pd.DataFrame(lam_rows).to_csv(RES / "lambda_validation_all_origins.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(RES / "coefficients_2010.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(RES / "neural_cv.csv", index=False)

    # 3. NRI sensitivity at origin 2010
    nri_rows = []
    for label, cols in (("nri_free_headline", M.HEADLINE_STATICS),
                        ("with_nri_v120", M.HEADLINE_STATICS + M.NRI_STATICS)):
        frame_n = M.build_frame(d, EST_START, 2010, cols)
        for spec in ("pooled", "state_year_fe"):
            est = M.fit(d, EST_START, 2010, spec, static_cols=cols,
                        frame=frame_n, cluster_se=False)
            for pol in POLICIES:
                sgdp, spop, _, _ = M.hindcast(d, est, 2010, 2019, pol, static_cols=cols)
                for out_name, pr, lvl in (("gdp", sgdp, d.lg), ("pop", spop, d.lp)):
                    act = lvl[:, d.yidx[2019]] - lvl[:, d.yidx[2010]]
                    m = ev.metrics(pr, act)
                    nri_rows.append(dict(variant=label, spec=spec, policy=pol,
                                         outcome=out_name,
                                         **{k: round(v, 5) for k, v in m.items()}))
    pd.DataFrame(nri_rows).to_csv(RES / "nri_sensitivity.csv", index=False)
    print("[3/6] NRI sensitivity done")

    # 4. spatial-weight diagnostic
    diag_rows = []
    wl_backup = d.wl.copy()
    for wlabel, W in (("local_100km", wl_backup),
                      ("all_region", (M.mp_weights(dist) / M.mp_weights(dist).sum(1, keepdims=True)))):
        d.wl = W
        frame_w = M.build_frame(d, EST_START, 2010, M.HEADLINE_STATICS)
        est = M.fit(d, EST_START, 2010, "pooled", frame=frame_w, cluster_se=False)
        for kind, own, spat in (("gdp", "g_lag", "sg_lag"), ("pop", "p_lag", "sp_lag")):
            diag_rows.append(dict(weights=wlabel, equation=kind,
                                  own_lag=est[kind]["coef"][own],
                                  spatial_lag=est[kind]["coef"][spat],
                                  feedback_sum=est[kind]["coef"][own] + est[kind]["coef"][spat],
                                  r2=est[kind]["r2"]))
    d.wl = wl_backup
    pd.DataFrame(diag_rows).to_csv(RES / "spatial_diagnostic.csv", index=False)
    print("[4/6] spatial diagnostic done")

    # 5. scenario grids
    panel19 = panel[panel.year == 2019].set_index("GeoFIPS").reindex(d.fips)
    gdp19 = panel19.rgdp_k2017.to_numpy(float) * 1000.0  # thousands of dollars to dollars
    share10 = d.sg[:, d.yidx[2010]]
    flagged = (share10 >= np.median(share10)).astype(float)
    s1 = SC.s1_grid(np.log(gdp19), gdp19, flagged)
    s1.to_csv(RES / "S1_grid.csv", index=False)
    corridor_mask = static.set_index("GeoFIPS").reindex(d.fips).corridor.to_numpy(float)
    s2, dln = SC.s2_band(dist, gdp19, corridor_mask)
    s2.to_csv(RES / "S2_band.csv", index=False)
    s1_reg, s1_med = SC.s1_grid2d(gdp19, flagged)
    s1_reg.to_csv(RES / "S1_grid2d_regional.csv")
    s1_med.to_csv(RES / "S1_grid2d_flagged.csv")
    s2_grid = SC.s2_grid2d(dist, gdp19, corridor_mask)
    s2_grid.to_csv(RES / "S2_grid2d.csv")
    print("[5/6] scenarios done")

    # 6. figures
    import figures as FIGS
    figure_files = FIGS.build_all(
        DATA / "county_geometries.geojson",
        RES / "hindcast_county_2010_headline_conditional.csv",
        RES / "rolling_metrics.csv", s1_reg, s1_med, s2_grid, FIG)
    print("[6/6] figures done:", figure_files)

    # run manifest
    import geopandas
    import matplotlib as mpl
    import scipy
    import shapely
    import sklearn
    import statsmodels
    manifest = dict(
        run_date="2026-10-06", runtime_seconds=round(time.time() - t_start, 1),
        python=platform.python_version(),
        expected_outputs=dict(
            figures=[f"results/figures/{f}" for f in figure_files],
            tables=["gate2010_baseline_metrics.csv", "gate2010_bootstrap_ci.csv",
                    "gate2010_lambda_curves.csv", "gate2010_lambda_validation.csv",
                    "gate2010_per_state_diffs.csv", "rolling_metrics.csv",
                    "rolling_inference.csv", "lambda_validation_all_origins.csv",
                    "coefficients_2010.csv", "nri_sensitivity.csv",
                    "spatial_diagnostic.csv", "S1_grid.csv", "S2_band.csv",
                    "S1_grid2d_regional.csv", "S1_grid2d_flagged.csv",
                    "S2_grid2d.csv",
                    "hindcast_county_2010_headline_conditional.csv",
                    "hindcast_county_2010_headline_mean.csv",
                    "neural_cv.csv", "hindcast_county_2010_neural.csv"]),
        packages=dict(numpy=np.__version__, pandas=pd.__version__,
                      scipy=scipy.__version__, statsmodels=statsmodels.__version__,
                      matplotlib=mpl.__version__, geopandas=geopandas.__version__,
                      shapely=shapely.__version__, sklearn=sklearn.__version__),
        neural=dict(seed=NN.NN_SEED, hidden=NN.HIDDEN, wd_grid=list(NN.WD_GRID),
                    epochs_final=NN.EPOCHS_FINAL, epochs_cv=NN.EPOCHS_CV,
                    lr=NN.LR, min_window=NN.MIN_WINDOW,
                    state_folds=[sorted(f) for f in NN.STATE_FOLDS],
                    mlp_alpha=NN.MLP_ALPHA,
                    graph_edges=int(A_graph.nnz),
                    selected_wd=nn_info["selected_wd"],
                    train_seconds_gnn=round(nn_info["train_seconds_gnn"], 1),
                    train_seconds_mlp=round(nn_info["train_seconds_mlp"], 1),
                    per_origin_seconds=nn_info["per_origin_seconds"]),
        seed_bootstrap=ev.BOOT_SEED, bootstrap_reps=ev.BOOT_REPS,
        origins=list(ORIGINS), specs=list(SPECS), policies=list(POLICIES),
        estimation_start=EST_START, n_counties=int(len(d.fips)),
        data_sha256={f.name: sha256(f) for f in sorted(DATA.glob("*")) if f.is_file()},
        gate_reproduction=gate_summary)
    (RES / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", RES / "run_manifest.json", f"({manifest['runtime_seconds']}s)")


if __name__ == "__main__":
    main()
