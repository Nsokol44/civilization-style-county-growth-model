"""Graph neural network benchmark over the county graph, plus an aspatial MLP.

A two-layer GraphSAGE-style network (NumPy, full-batch Adam) is trained on
calibration sub-windows to predict annualised change; the forecast multiplies
the prediction at the origin by the horizon. The MLP uses the same features
and windows without the graph. Inputs: Data object, growth arrays. Outputs:
cumulative predictions per county.
"""
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neural_network import MLPRegressor

import model as M

NN_SEED = 20261006
WD_GRID = (0.03, 0.1, 0.3)
HIDDEN = 16
LR = 0.01
EPOCHS_FINAL = 300
EPOCHS_CV = 200
MIN_WINDOW = 3
STATE_FOLDS = ({"AL", "KY", "SC"}, {"FL", "MS", "TN"}, {"GA", "NC", "VA"})
MLP_ALPHA = 3.0

FEATURE_NAMES = ["lnGDP", "lnPOP", "lnGDPpc", "lnMP", "g_t", "p_t",
                 "share_goods", "share_services", "trend_g", "trend_p"]


def graph(d):
    """Local spatial weights as a sparse adjacency matrix."""
    return sparse.csr_matrix(d.wl)


def snapshot_features(d, t, g, p, Z):
    """Node features at year t from data dated t or earlier."""
    k = d.yidx[t]
    lg, lp = d.lg[:, k], d.lp[:, k]
    mp = np.log(d.wm @ np.exp(lg))
    yrs = [d.yidx[y] for y in range(max(2002, t - 4), t + 1)]
    tg = np.nanmean(g[:, yrs], axis=1)
    tp = np.nanmean(p[:, yrs], axis=1)
    state_oh = np.eye(len(d.states))[d.state_codes]
    return np.column_stack([lg, lp, lg - lp, mp, g[:, k], p[:, k],
                            d.sg[:, k], d.ss[:, k], tg, tp, Z, state_oh])


def conditional_shift(pred, act, state_codes):
    """Shift predictions so each state mean equals the realised state mean."""
    shift = pd.Series(act).groupby(state_codes).transform("mean").to_numpy()
    own = pd.Series(pred).groupby(state_codes).transform("mean").to_numpy()
    return pred + (shift - own)


class SageNet:
    """Two GraphSAGE-style layers and a linear head, trained with full-batch Adam."""

    def __init__(self, n_in, hidden=HIDDEN, seed=NN_SEED, wd=0.1):
        rng = np.random.default_rng(seed)

        def init(fi, fo):
            return rng.normal(0.0, np.sqrt(2.0 / fi), (fi, fo))

        self.Ws1, self.Wn1, self.b1 = init(n_in, hidden), init(n_in, hidden), np.zeros(hidden)
        self.Ws2, self.Wn2, self.b2 = init(hidden, hidden), init(hidden, hidden), np.zeros(hidden)
        self.Wo = rng.normal(0.0, np.sqrt(1.0 / hidden), (hidden, 1))
        self.bo = np.zeros(1)
        self.wd = float(wd)

    def _params(self):
        return [self.Ws1, self.Wn1, self.b1, self.Ws2, self.Wn2,
                self.b2, self.Wo, self.bo]

    def _set(self, ps):
        (self.Ws1, self.Wn1, self.b1, self.Ws2, self.Wn2,
         self.b2, self.Wo, self.bo) = ps

    def forward(self, A, X):
        N1 = A @ X
        Z1 = X @ self.Ws1 + N1 @ self.Wn1 + self.b1
        H1 = np.maximum(Z1, 0)
        N2 = A @ H1
        Z2 = H1 @ self.Ws2 + N2 @ self.Wn2 + self.b2
        H2 = np.maximum(Z2, 0)
        out = (H2 @ self.Wo + self.bo).ravel()
        return out, (X, N1, Z1, H1, N2, Z2, H2)

    def predict(self, A, X):
        return self.forward(A, X)[0]

    def train(self, A, snaps, epochs, node_mask=None):
        """Fit on snaps, a list of (X, Y, target_mask); node_mask limits the loss to a subset of counties."""
        params = self._params()
        ms = [np.zeros_like(x) for x in params]
        vs = [np.zeros_like(x) for x in params]
        wd = self.wd
        nm = np.ones(len(snaps[0][0]), bool) if node_mask is None else node_mask
        for ep in range(1, epochs + 1):
            grads = [np.zeros_like(x) for x in params]
            for X, Y, Tm in snaps:
                out, (X_, N1, Z1, H1, N2, Z2, H2) = self.forward(A, X)
                Msk = Tm & nm[:, None]
                err = np.where(Msk, out[:, None] - Y, 0.0)
                n = float(Msk.sum())
                if n == 0:
                    continue
                dout = (2.0 * err.sum(1) / n)[:, None]
                gWo = H2.T @ dout + wd * self.Wo
                gbo = dout.sum(0)
                dH2 = dout @ self.Wo.T
                dZ2 = dH2 * (Z2 > 0)
                gWs2 = H1.T @ dZ2 + wd * self.Ws2
                gWn2 = N2.T @ dZ2 + wd * self.Wn2
                gb2 = dZ2.sum(0)
                dH1 = dZ2 @ self.Ws2.T + A.T @ (dZ2 @ self.Wn2.T)
                dZ1 = dH1 * (Z1 > 0)
                gWs1 = X_.T @ dZ1 + wd * self.Ws1
                gWn1 = N1.T @ dZ1 + wd * self.Wn1
                gb1 = dZ1.sum(0)
                for i, gg in enumerate((gWs1, gWn1, gb1, gWs2, gWn2,
                                        gb2, gWo, gbo)):
                    grads[i] += gg / len(snaps)
            for i in range(len(params)):
                ms[i] = 0.9 * ms[i] + 0.1 * grads[i]
                vs[i] = 0.999 * vs[i] + 0.001 * grads[i] ** 2
                mh = ms[i] / (1 - 0.9 ** ep)
                vh = vs[i] / (1 - 0.999 ** ep)
                params[i] -= LR * mh / (np.sqrt(vh) + 1e-8)
            self._set(params)
        return self


def _snap_data(d, g, p, Z, origin, lvl, obs):
    """Training data per start year: features, annualised changes to each end year, and mask."""
    out = []
    for s in range(2003, origin - MIN_WINDOW + 1):
        X = snapshot_features(d, s, g, p, Z)
        Ys, Ms = [], []
        for e in range(s + MIN_WINDOW, origin + 1):
            ann = (lvl[:, d.yidx[e]] - lvl[:, d.yidx[s]]) / (e - s)
            if obs is not None:
                m = obs[:, d.yidx[e]] & obs[:, d.yidx[s]]
            else:
                m = np.ones(len(d.fips), bool)
            Ys.append(ann)
            Ms.append(m)
        out.append((X, np.column_stack(Ys), np.column_stack(Ms)))
    return out


def _standardise(snap_raw):
    Xall = np.vstack([s[0] for s in snap_raw])
    xm, xsd = Xall.mean(0), Xall.std(0)
    xsd[xsd == 0] = 1.0
    yall = np.concatenate([s[1][s[2]] for s in snap_raw])
    ym, ysd = float(yall.mean()), float(yall.std())
    snaps = [((X - xm) / xsd, (Y - ym) / ysd, Mk) for X, Y, Mk in snap_raw]
    return snaps, (xm, xsd, ym, ysd)


def train_gnn(d, A, g, p, Z, origin, lvl, obs=None):
    """Select weight decay by state-grouped validation, then fit the final GNN."""
    t0 = time.time()
    snap_raw = _snap_data(d, g, p, Z, origin, lvl, obs)
    snaps, stats = _standardise(snap_raw)
    xm, xsd, ym, ysd = stats
    state = d.state
    cv_rows = []
    best_wd, best_mse = None, np.inf
    for wd in WD_GRID:
        fold_mses = []
        for fold_states in STATE_FOLDS:
            held = np.array([s in fold_states for s in state])
            net = SageNet(snaps[0][0].shape[1], wd=wd)
            net.train(A, snaps, EPOCHS_CV, node_mask=~held)
            se, cnt = 0.0, 0
            for X, Y, Tm in snaps:
                pr = net.predict(A, X)
                Msk = Tm & held[:, None]
                se += float(np.sum(np.where(Msk, pr[:, None] - Y, 0.0) ** 2))
                cnt += int(Msk.sum())
            fold_mses.append(se / cnt)
            cv_rows.append(dict(wd=wd, fold="+".join(sorted(fold_states)),
                                val_mse=se / cnt))
        pooled = float(np.mean(fold_mses))
        for r in cv_rows:
            if r["wd"] == wd:
                r.setdefault("pooled_val_mse", pooled)
        if pooled < best_mse:
            best_mse, best_wd = pooled, wd
    net = SageNet(snaps[0][0].shape[1], wd=best_wd)
    net.train(A, snaps, EPOCHS_FINAL)
    return net, stats, best_wd, cv_rows, time.time() - t0


def forecast_gnn(net, stats, d, g, p, Z, origin, H):
    """Cumulative prediction from the origin snapshot."""
    xm, xsd, ym, ysd = stats
    A = graph(d)
    X0 = snapshot_features(d, origin, g, p, Z)
    ann = net.predict(A, (X0 - xm) / xsd) * ysd + ym
    return ann * H


def train_mlp_ablation(d, g, p, Z, origin, lvl):
    """Fit the MLP without the graph; returns a prediction function and the fit time."""
    t0 = time.time()
    snap_raw = _snap_data(d, g, p, Z, origin, lvl, None)
    Xs, ys = [], []
    for X, Y, Mk in snap_raw:
        for w in range(Y.shape[1]):
            Xs.append(X[Mk[:, w]])
            ys.append(Y[Mk[:, w], w])
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    xm, xsd = X.mean(0), X.std(0)
    xsd[xsd == 0] = 1.0
    ym, ysd = float(y.mean()), float(y.std())
    mlp = MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu",
                       solver="adam", alpha=MLP_ALPHA, max_iter=300,
                       early_stopping=False, random_state=NN_SEED)
    mlp.fit((X - xm) / xsd, (y - ym) / ysd)

    def predict_fn(X0):
        return mlp.predict((X0 - xm) / xsd) * ysd + ym

    return predict_fn, time.time() - t0
