"""Baselines, shrinkage selection and inference for the hindcast evaluation.

Error is pred - actual. Demeaned RMSE removes the mean of prediction and actual.
Shrinkage lambda is chosen inside the calibration window (trend fitted to a
freeze year, scored on growth from the freeze year to the origin) and by an
empirical-Bayes variance ratio.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

LAMBDAS = np.round(np.arange(0.0, 1.01, 0.1), 2)
BOOT_REPS = 2000
BOOT_SEED = 20261006


def metrics(pred, act):
    pred, act = np.asarray(pred, float), np.asarray(act, float)
    err = pred - act
    bias = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    err_sd = float(np.sqrt(max(np.mean(err ** 2) - bias ** 2, 0.0)))
    dm = (pred - pred.mean()) - (act - act.mean())
    return dict(rmse=rmse, mae=float(np.mean(np.abs(err))), bias=bias,
                err_sd=err_sd, demeaned_rmse=float(np.sqrt(np.mean(dm ** 2))),
                pearson=float(np.corrcoef(pred, act)[0, 1]) if pred.std() > 0 else np.nan,
                spearman=float(spearmanr(pred, act).statistic) if pred.std() > 0 else np.nan)


def state_demeaned_rmse(pred, act, state):
    df = pd.DataFrame({"p": pred, "a": act, "s": state})
    dm = (df.p - df.groupby("s").p.transform("mean")) - \
         (df.a - df.groupby("s").a.transform("mean"))
    return float(np.sqrt(np.mean(dm ** 2)))


def select_lambda(levels, yidx, origin, state_codes):
    """Choose shrinkage lambda for one outcome; levels is an [n, T] array."""
    annual = np.full_like(levels, np.nan)
    annual[:, 1:] = np.diff(levels, axis=1)
    rows, primary = [], {}
    splits = [F for F in (origin - 5, origin - 4, origin - 3) if F >= 2003]
    for si, F in enumerate(splits):
        tr_years = [y for y in range(2002, F + 1)]
        tr_mean = np.nanmean(annual[:, [yidx[y] for y in tr_years]], axis=1)
        horizon = origin - F
        val_act = levels[:, yidx[origin]] - levels[:, yidx[F]]
        anchors = {"regional": np.full(len(levels), tr_mean.mean()),
                   "state": pd.Series(tr_mean).groupby(state_codes).transform("mean").to_numpy()}
        for aname, anch in anchors.items():
            curve = [float(np.sqrt(np.mean(
                (horizon * (lam * tr_mean + (1 - lam) * anch) - val_act) ** 2)))
                for lam in LAMBDAS]
            best = int(np.argmin(curve))
            rows.append(dict(freeze=F, anchor=aname, primary_split=(si == 0),
                             best_lambda=float(LAMBDAS[best]), best_rmse=curve[best],
                             rmse_lambda0=curve[0], rmse_lambda1=curve[-1],
                             curve=";".join(f"{x:.4f}" for x in curve)))
            if si == 0:
                primary[aname] = float(LAMBDAS[best])
    # empirical-Bayes variance ratio on annual increments, 2002 to origin
    yrs = [yidx[y] for y in range(2002, origin + 1)]
    A = annual[:, yrs]
    m = A.shape[1]
    means = np.nanmean(A, axis=1)
    within = np.nanmean((A - means[:, None]) ** 2 * m / (m - 1), axis=1)
    sig_w2 = float(np.mean(within))
    var_means = float(np.var(means, ddof=1))
    noise = sig_w2 / m
    sig_b2 = max(var_means - noise, 0.0)
    lam_eb = sig_b2 / (sig_b2 + noise) if (sig_b2 + noise) > 0 else 0.0
    eb = dict(var_of_county_means=var_means, sampling_noise=noise,
              sigma_b2=sig_b2, sigma_w2=sig_w2, lambda_eb=float(lam_eb), m_years=int(m))
    return rows, primary, eb


def build_baselines(levels, yidx, origin, state, state_codes, lam_primary):
    """Baseline predictions of cumulative change from origin to 2019, and the trend."""
    H = 2019 - origin
    span = origin - 2001
    total_cal = levels[:, yidx[origin]] - levels[:, yidx[2001]]
    trend = total_cal * H / span
    act = levels[:, yidx[2019]] - levels[:, yidx[origin]]
    reg_ex = np.full(len(levels), trend.mean())
    st_ex = pd.Series(trend).groupby(state_codes).transform("mean").to_numpy()
    reg_or = np.full(len(levels), act.mean())
    st_or = pd.Series(act).groupby(state_codes).transform("mean").to_numpy()
    preds = {
        "persistence": np.zeros(len(levels)),
        "regional_mean_exante": reg_ex,
        "state_mean_exante": st_ex,
        "regional_mean_oracle": reg_or,
        "state_mean_oracle": st_or,
        "county_trend_raw": trend,
    }
    for aname, anch in (("regional", reg_ex), ("state", st_ex)):
        lam = lam_primary[aname]
        preds[f"shrunk_{aname}_primary_lambda{lam:.1f}"] = lam * trend + (1 - lam) * anch
    return preds, trend


def clustered_bootstrap_rmse_diff(pred_a, pred_b, act, state, reps=BOOT_REPS,
                                  seed=BOOT_SEED):
    """State-clustered bootstrap interval for RMSE(a) - RMSE(b)."""
    states = np.unique(state)
    idx = {s: np.where(state == s)[0] for s in states}
    def sse(pred):
        e2 = (pred - act) ** 2
        return np.array([e2[idx[s]].sum() for s in states]), \
            np.array([len(idx[s]) for s in states])
    sa, counts = sse(pred_a)
    sb, _ = sse(pred_b)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(states), size=(reps, len(states)))
    S = len(states)
    w = np.bincount((np.arange(reps)[:, None] * S + draws).ravel(),
                    minlength=reps * S).reshape(reps, S).astype(float)
    n_star = w @ counts
    diffs = np.sqrt((w @ sa) / n_star) - np.sqrt((w @ sb) / n_star)
    point = float(np.sqrt(sa.sum() / counts.sum()) - np.sqrt(sb.sum() / counts.sum()))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, float(lo), float(hi)


def clustered_bootstrap_demeaned_diff(pred_a, pred_b, act, state, reps=BOOT_REPS,
                                      seed=BOOT_SEED):
    """State-clustered bootstrap interval for the demeaned RMSE difference.

    Demeaning uses the full-sample means in every resample."""
    def dmse(pred):
        dm = (pred - pred.mean()) - (act - act.mean())
        return dm ** 2
    ea, eb = dmse(pred_a), dmse(pred_b)
    states = np.unique(state)
    idx = {s: np.where(state == s)[0] for s in states}
    sa = np.array([ea[idx[s]].sum() for s in states])
    sb = np.array([eb[idx[s]].sum() for s in states])
    counts = np.array([len(idx[s]) for s in states])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(states), size=(reps, len(states)))
    S = len(states)
    w = np.bincount((np.arange(reps)[:, None] * S + draws).ravel(),
                    minlength=reps * S).reshape(reps, S).astype(float)
    n_star = w @ counts
    diffs = np.sqrt((w @ sa) / n_star) - np.sqrt((w @ sb) / n_star)
    point = float(np.sqrt(sa.sum() / counts.sum()) - np.sqrt(sb.sum() / counts.sum()))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, float(lo), float(hi)


def diebold_mariano(pred_a, pred_b, act, state, demeaned=False):
    """Cross-sectional Diebold-Mariano statistic on squared error differences."""
    if demeaned:
        ea = (pred_a - pred_a.mean()) - (act - act.mean())
        eb = (pred_b - pred_b.mean()) - (act - act.mean())
    else:
        ea, eb = pred_a - act, pred_b - act
    d = ea ** 2 - eb ** 2
    n = len(d)
    from scipy.stats import norm
    dm = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    p = 2 * (1 - norm.cdf(abs(dm)))
    # state-clustered standard error of the mean of d
    states = np.unique(state)
    G = len(states)
    ss = np.array([d[state == s].sum() - (state == s).sum() * d.mean() for s in states])
    var = (G / (G - 1.0)) * np.sum(ss ** 2) / n ** 2
    se_c = np.sqrt(var) if var > 0 else np.nan
    dm_c = d.mean() / se_c if se_c and np.isfinite(se_c) else np.nan
    p_c = 2 * (1 - norm.cdf(abs(dm_c))) if np.isfinite(dm_c) else np.nan
    return float(dm), float(p), float(dm_c), float(p_c)
