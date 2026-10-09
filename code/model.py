"""Recursive spatial panel model of county growth.

Fits GDP and population growth equations and goods/services share drift
equations on a calibration window, then iterates them forward from a forecast
origin. Inputs: county panel, static county attributes, distance matrix.
Outputs: cumulative growth and final sector shares for each county.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

FLOOR = 10.0
YEARS = list(range(2001, 2020))

HEADLINE_STATICS = ["topo_code", "jan_temp_f", "water_share", "amenity_scale",
                    "log_density_2001"]
NRI_STATICS = ["hazard_composite", "risk_drought", "coastal"]


def local_weights(dist, radius=100.0, min_nb=8):
    w = np.where((dist > 0) & (dist <= radius), 1.0 / np.maximum(dist, FLOOR), 0.0)
    for i in range(len(w)):
        if (w[i] > 0).sum() < min_nb:
            nn = np.argsort(dist[i])[1:min_nb + 1]
            w[i, nn] = 1.0 / np.maximum(dist[i, nn], FLOOR)
    return w / w.sum(1, keepdims=True)


def mp_weights(dist):
    w = 1.0 / np.maximum(dist, FLOOR)
    np.fill_diagonal(w, 0.0)
    return w


class Data:
    """Panel arrays aligned to the static file's GeoFIPS order."""

    def __init__(self, panel, static, dist):
        self.static = static.sort_values("GeoFIPS").reset_index(drop=True)
        self.fips = self.static.GeoFIPS.tolist()
        n = len(self.fips)
        idx = {f: i for i, f in enumerate(self.fips)}
        self.yidx = {y: j for j, y in enumerate(YEARS)}
        self.lg = np.full((n, len(YEARS)), np.nan)
        self.lp = self.lg.copy()
        sg = np.full((n, len(YEARS)), np.nan)
        ss = np.full((n, len(YEARS)), np.nan)
        for r in panel.itertuples():
            i, k = idx[r.GeoFIPS], self.yidx[r.year]
            self.lg[i, k] = np.log(r.rgdp_k2017)
            self.lp[i, k] = np.log(r.pop)
            sg[i, k] = r.share_goods
            ss[i, k] = r.share_services
        assert np.isfinite(self.lg).all() and np.isfinite(self.lp).all()
        self.sg_obs = ~np.isnan(sg)
        # fill missing shares (within-county ffill/bfill, then regional median)
        # for starting values; the drift equations use observed cells only
        med_g, med_s = np.nanmedian(sg), np.nanmedian(ss)
        self.sg = pd.DataFrame(sg).ffill(axis=1).bfill(axis=1).fillna(med_g).to_numpy(float)
        self.ss = pd.DataFrame(ss).ffill(axis=1).bfill(axis=1).fillna(med_s).to_numpy(float)
        self.dist = dist
        self.wl = local_weights(dist)
        self.wm = mp_weights(dist)
        self.state = self.static.state_abbr.to_numpy()
        self.state_codes = pd.Categorical(self.state).codes
        self.states = sorted(np.unique(self.state))
        # log population density in 2001
        pop01 = np.exp(self.lp[:, self.yidx[2001]])
        dens = np.log(pop01 / self.static.land_km2.to_numpy(float))
        self.static = self.static.assign(log_density_2001=dens)

    def growth(self):
        g = np.full_like(self.lg, np.nan)
        g[:, 1:] = np.diff(self.lg, axis=1)
        p = np.full_like(self.lp, np.nan)
        p[:, 1:] = np.diff(self.lp, axis=1)
        return g, p

    def zmat(self, cols):
        Z = self.static[cols].to_numpy(float)
        mu, sd = np.nanmean(Z, 0), np.nanstd(Z, 0)
        Z = (Z - mu) / sd
        return np.nan_to_num(Z), list(cols)


def _covariates(d, lg_lag, lp_lag, g_lag, p_lag, sg_lag, ss_lag, kind):
    mp = np.log(d.wm @ np.exp(lg_lag))
    if kind == "gdp":
        return {"lnGDP": lg_lag, "lnMP": mp, "g_lag": g_lag,
                "sg_lag": d.wl @ np.nan_to_num(g_lag),
                "share_goods": sg_lag, "share_serv": ss_lag}
    if kind == "pop":
        return {"lnPOP": lp_lag, "lnMP": mp, "lnGDPpc": lg_lag - lp_lag,
                "g_lag": g_lag, "p_lag": p_lag,
                "sp_lag": d.wl @ np.nan_to_num(p_lag)}
    # drift equations (dependent variable is the change in the share)
    return {"share_goods": sg_lag, "share_serv": ss_lag,
            "lnGDPpc": lg_lag - lp_lag}


DRIFT_STATICS = ["topo_code", "water_share"]


def build_frame(d, y0, y1, static_cols):
    """Long estimation frame for years y0..y1."""
    g, p = d.growth()
    Z, zcols = d.zmat(static_cols)
    Zd, zdcols = d.zmat(DRIFT_STATICS)
    rows = []
    for yr in range(y0, y1 + 1):
        k = d.yidx[yr]
        base = dict(i=np.arange(len(d.fips)), year=yr, state=d.state_codes)
        for kind, y in (("gdp", g), ("pop", p)):
            X = _covariates(d, d.lg[:, k - 1], d.lp[:, k - 1], g[:, k - 1],
                            p[:, k - 1], d.sg[:, k - 1], d.ss[:, k - 1], kind)
            df = pd.DataFrame(X)
            df["y"] = y[:, k]
            for j, c in enumerate(zcols):
                df[c] = Z[:, j]
            rows.append(df.assign(**base, kind=kind))
        for kind, lvl in (("dgoods", d.sg), ("dserv", d.ss)):
            X = _covariates(d, d.lg[:, k - 1], d.lp[:, k - 1], g[:, k - 1],
                            p[:, k - 1], d.sg[:, k - 1], d.ss[:, k - 1], "drift")
            df = pd.DataFrame(X)
            df["y"] = lvl[:, k] - lvl[:, k - 1]
            df["obs"] = d.sg_obs[:, k] & d.sg_obs[:, k - 1]
            for j, c in enumerate(zdcols):
                df[c] = Zd[:, j]
            rows.append(df.assign(**base, kind=kind))
    return pd.concat(rows, ignore_index=True), zcols, zdcols


def _ols(y, X, groups, cluster_se=True):
    mod = sm.OLS(np.asarray(y, float), np.asarray(X, float))
    if cluster_se:
        return mod.fit(cov_type="cluster", cov_kwds={"groups": np.asarray(groups)})
    return mod.fit()


def fit(d, y0, y1, spec="pooled", static_cols=HEADLINE_STATICS, frame=None,
        cluster_se=True):
    """Estimate the four equations on years y0..y1; returns a dict by equation."""
    if frame is None:
        frame = build_frame(d, y0, y1, static_cols)
    fr, zcols, zdcols = frame
    out = {}
    for kind in ("gdp", "pop", "dgoods", "dserv"):
        f = fr[fr.kind == kind].copy()
        if kind in ("dgoods", "dserv"):
            f = f[f.obs].drop(columns=["obs"])
            eff_spec = "pooled"
            xs_static = zdcols
        else:
            eff_spec = spec
            xs_static = zcols
        dyn = [c for c in f.columns if c not in
               ("y", "year", "i", "state", "kind", "obs") + tuple(zcols) + tuple(zdcols)
               and f[c].notna().any()]
        xs = dyn + [c for c in xs_static if c in f.columns]
        f = f.dropna(subset=["y"] + xs)
        rec = dict(spec=eff_spec, xs=xs, kind=kind)
        if eff_spec == "county_fe":
            xs_use = dyn  # static covariates drop out under the within transformation
            yv = (f.y - f.groupby("i").y.transform("mean")).to_numpy(float)
            Xd = f[xs_use] - f.groupby("i")[xs_use].transform("mean")
            yrs = sorted(f.year.unique())
            yd = pd.get_dummies(f.year, prefix="yr", drop_first=True, dtype=float)
            yd = yd - yd.mean(axis=0)  # county-demeaned year dummies
            X = pd.concat([Xd.reset_index(drop=True), yd.reset_index(drop=True)], axis=1)
            res = _ols(yv, X.to_numpy(float), f.i.to_numpy(), cluster_se)
            names = list(X.columns)
            coef = dict(zip(names, res.params))
            se = dict(zip(names, res.bse))
            beta = np.array([coef[c] for c in xs_use])
            # county intercepts: mean residual after covariates and year effects
            year_eff = {yrs[0]: 0.0}
            year_eff.update({y: coef[f"yr_{y}"] for y in yrs[1:]})
            lin = f[xs_use].to_numpy(float) @ beta + np.array(
                [year_eff[y] for y in f.year])
            alpha = pd.Series(f.y.to_numpy(float) - lin).groupby(
                f.i.to_numpy()).mean()
            rec.update(coef={c: coef[c] for c in xs_use}, se={c: se[c] for c in xs_use},
                       xs=xs_use, county_fe=alpha.reindex(range(len(d.fips))).fillna(
                           float(alpha.mean())).to_numpy(float),
                       year_eff=year_eff,
                       year_eff_mean=float(np.mean(list(year_eff.values()))))
        elif eff_spec == "state_year_fe":
            yrs = sorted(f.year.unique())
            cell_id = (f.state.astype(np.int64) * 10000 + f.year.astype(np.int64)).astype(int)
            cd = pd.get_dummies(cell_id, prefix="cell", dtype=float)
            X = pd.concat([f[xs].reset_index(drop=True), cd.reset_index(drop=True)], axis=1)
            res = _ols(f.y.to_numpy(float), X.to_numpy(float), f.i.to_numpy(), cluster_se)
            names = list(X.columns)
            coef = dict(zip(names, res.params))
            se = dict(zip(names, res.bse))
            cell_eff = {}
            for cname in cd.columns:
                cid = int(cname.split("_")[1])
                cell_eff[(cid // 10000, cid % 10000)] = float(coef[cname])
            means = {}
            for s_code in np.unique(d.state_codes):
                vals = [cell_eff[(int(s_code), y)] for y in yrs
                        if (int(s_code), y) in cell_eff]
                means[int(s_code)] = float(np.mean(vals))
            rec.update(coef={c: coef[c] for c in xs}, se={c: se[c] for c in xs},
                       cell_eff=cell_eff, cell_mean_by_state=means)
        else:
            yrs = sorted(f.year.unique())
            parts = [f[xs].reset_index(drop=True)]
            if eff_spec == "state_fe":
                sd = pd.get_dummies(f.state, prefix="st", drop_first=True, dtype=float)
                parts.append(sd.reset_index(drop=True))
            yd = pd.get_dummies(f.year, prefix="yr", drop_first=True, dtype=float)
            parts.append(yd.reset_index(drop=True))
            X = pd.concat(parts, axis=1)
            X.insert(0, "const", 1.0)
            res = _ols(f.y.to_numpy(float), X.to_numpy(float), f.i.to_numpy(), cluster_se)
            names = list(X.columns)
            coef = dict(zip(names, res.params))
            se = dict(zip(names, res.bse))
            year_eff = {yrs[0]: 0.0}
            year_eff.update({y: coef[f"yr_{y}"] for y in yrs[1:]})
            rec.update(coef=coef, se=se, const=float(coef["const"]),
                       year_eff=year_eff,
                       year_eff_mean=float(np.mean(list(year_eff.values()))))
            if eff_spec == "state_fe":
                rec["state_eff"] = {int(c.split("_")[1]): float(coef[c])
                                    for c in names if c.startswith("st_")}
        rec.update(r2=float(res.rsquared), n=int(res.nobs),
                   resid_sd=float(np.std(res.resid, ddof=1)))
        out[kind] = rec
    return out


def _predict_base(est_kind, d, X, Zfull, zcols_of_kind):
    """Prediction excluding any year/cell effect."""
    m = est_kind
    p = np.zeros(len(d.fips))
    if m["spec"] == "county_fe":
        p = p + m["county_fe"]
    elif m["spec"] != "state_year_fe":
        p = p + m.get("const", 0.0)
    for name in m["xs"]:
        if name in X:
            p = p + m["coef"][name] * X[name]
        else:  # static covariate, taken from Zfull
            j = zcols_of_kind.index(name)
            p = p + m["coef"][name] * Zfull[:, j]
    if m["spec"] == "state_fe":
        se = m.get("state_eff", {})
        p = p + np.array([se.get(int(s), 0.0) for s in d.state_codes], float)
    return p


def hindcast(d, est, t0, t1, year_policy="mean", static_cols=HEADLINE_STATICS):
    """Recursive hindcast from t0 to t1 (cumulative growth and final shares)."""
    g, p = d.growth()
    Z, zcols = d.zmat(static_cols)
    Zd, zdcols = d.zmat(DRIFT_STATICS)
    # year_policy "mean": calibration mean year effect; "conditional": predicted
    # mean (state mean under state_year_fe) is shifted to the realised mean
    k0 = d.yidx[t0]
    lg, lp = d.lg[:, k0].copy(), d.lp[:, k0].copy()
    gl, pl = g[:, k0].copy(), p[:, k0].copy()
    sg, ss = d.sg[:, k0].copy(), d.ss[:, k0].copy()
    for yr in range(t0 + 1, t1 + 1):
        k = d.yidx[yr]
        preds = {}
        for kind in ("gdp", "pop"):
            X = _covariates(d, lg, lp, gl, pl, sg, ss, kind)
            base = _predict_base(est[kind], d, X, Z, zcols)
            m = est[kind]
            if m["spec"] == "state_year_fe":
                eff = np.array([m["cell_mean_by_state"][int(s)]
                                for s in d.state_codes], float)
                pred = base + eff
                if year_policy == "conditional":
                    act = (g if kind == "gdp" else p)[:, k]
                    shift = pd.Series(act).groupby(d.state_codes).transform("mean")
                    own = pd.Series(pred).groupby(d.state_codes).transform("mean")
                    pred = pred + (shift - own).to_numpy(float)
            else:
                pred = base + m["year_eff_mean"]
                if year_policy == "conditional":
                    act = (g if kind == "gdp" else p)[:, k]
                    pred = pred + (np.nanmean(act) - pred.mean())
            preds[kind] = pred
        # share drift, always with the mean year effect
        Xd = _covariates(d, lg, lp, gl, pl, sg, ss, "drift")
        dg = _predict_base(est["dgoods"], d, Xd, Zd, zdcols) + est["dgoods"]["year_eff_mean"]
        ds = _predict_base(est["dserv"], d, Xd, Zd, zdcols) + est["dserv"]["year_eff_mean"]
        sg = np.clip(sg + dg, 0.01, 0.95)
        ss = np.clip(ss + ds, 0.01, 0.95)
        tot = sg + ss
        over = tot > 0.99
        if over.any():
            sg[over] *= 0.99 / tot[over]
            ss[over] *= 0.99 / tot[over]
        lg, lp = lg + preds["gdp"], lp + preds["pop"]
        gl, pl = preds["gdp"], preds["pop"]
    return lg - d.lg[:, k0], lp - d.lp[:, k0], sg, ss
