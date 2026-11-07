"""Finite-group mapping to E[D_3] and the resulting m_E, n*, and R_text.

Target t(x) = E[D_3] over 200 three-vote draws from all votes of a fitting item. Predictor features:
probabilities sorted in descending order, 1 - max p, and entropy. Map g = ridge on standardized degree-2
features (alpha by 5-fold item CV) or isotonic on 1 - max p, whichever has the smaller CV MSE.
(i) the model output is mapped, human groups are raw; (ii) human k-groups are mapped with g_k as well.
Fitting items: dev (BRIGHTER, ChaosNLI) or the out-of-fold items of the other folds (DICES-990).

  python s4_map.py --votes DIR --split DIR --res3 DIR --res4 DIR --out FILE.json
"""

import argparse
import glob
import json
import pickle

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from s3_nstar import N_BOOT, N_DRAW, corr_rows, nstar

N_FIT_DRAW_T, N_FIT_DRAW_K = 200, 20
ALPHAS = (0.01, 0.1, 1, 10, 100)


def feats(P):
    s = -np.sort(-P, -1)
    ent = -(np.clip(P, 1e-12, 1) * np.log(np.clip(P, 1e-12, 1))).sum(-1)
    return np.concatenate([s, (1 - s[..., :1]), ent[..., None]], -1)


def D(P):
    return 1 - P.max(-1)


def target(Vs, rng, n=3):
    return np.array([np.mean([D(V[rng.permutation(len(V))[:n]].mean(0)) for _ in range(N_FIT_DRAW_T)])
                     for V in Vs])


def fit_map(X, y, groups):
    """Return (predict_fn, info). X: rows x features; groups: item index per row."""
    gkf = GroupKFold(5)
    best = None
    for a in ALPHAS:
        err = []
        for tr, te in gkf.split(X, y, groups):
            m = make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(alpha=a)).fit(X[tr], y[tr])
            err.append(np.mean((m.predict(X[te]) - y[te]) ** 2))
        if best is None or np.mean(err) < best[1]:
            best = (a, float(np.mean(err)))
    iso_err = []
    for tr, te in gkf.split(X, y, groups):
        m = IsotonicRegression(out_of_bounds="clip").fit(X[tr, -2], y[tr])
        iso_err.append(np.mean((m.predict(X[te, -2]) - y[te]) ** 2))
    iso_err = float(np.mean(iso_err))
    ridge = make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(alpha=best[0])).fit(X, y)
    iso = IsotonicRegression(out_of_bounds="clip").fit(X[:, -2], y)
    fns = {"ridge": lambda Z: ridge.predict(Z), "iso": lambda Z: iso.predict(Z[:, -2])}
    main = "ridge" if best[1] <= iso_err else "iso"
    return fns, {"main": main, "ridge_alpha": best[0], "cv_mse_ridge": best[1], "cv_mse_iso": iso_err}


def draws_vec(Vs, n, kmax, rng):
    """Same random stream as s3_nstar.draws; also keeps the B-group mean vectors."""
    K = Vs[0].shape[1]
    A = np.zeros((N_DRAW, len(Vs))); B = np.zeros((kmax, N_DRAW, len(Vs))); BV = np.zeros((kmax, N_DRAW, len(Vs), K))
    for j, V in enumerate(Vs):
        for d in range(N_DRAW):
            p = rng.permutation(len(V))
            A[d, j] = 1 - V[p[:n]].mean(0).max()
            cum = np.cumsum(V[p[n:n + kmax]], 0) / np.arange(1, kmax + 1)[:, None]
            BV[:, d, j] = cum
            B[:, d, j] = 1 - cum.max(1)
    return A, B, BV


def ens(runs, ids, split):
    Ps = []
    for f in runs:
        r = pickle.load(open(f, "rb"))
        pos = {i: j for j, i in enumerate(r[split]["ids"])}
        Ps.append(r[split]["P"][[pos[i] for i in ids]])
    return np.mean(Ps, 0)


def nstar_block(m_vec, A, Bmat, n_items):
    """m_vec: model predictor per item; Bmat: (kmax, draws, items) human predictors."""
    def st(idx):
        m = corr_rows(m_vec[idx][None], A[:, idx]).mean()
        c = np.array([corr_rows(A[:, idx], Bmat[k][:, idx]).mean() for k in range(len(Bmat))])
        return m, c
    m, c = st(np.arange(n_items))
    brng = np.random.default_rng(1)
    boots = np.array([nstar(*st(brng.integers(0, n_items, n_items))) for _ in range(N_BOOT)])
    lo, hi = (float(np.quantile(np.where(np.isfinite(boots), boots, 1e9), p)) for p in (0.025, 0.975))
    return {"m": float(m), "c": c.tolist(), "nstar": nstar(m, c), "ci95": [lo, hi if hi < 1e9 else float("inf")],
            "boot_share_above_kmax": float(np.mean(~np.isfinite(boots)))}


def analyse(name, Vfit, Pfit, Vtest, Ptest, n, kmax, fit_groups=None, keep=False):
    """fit_groups: None (one fit set) or list of (fit_idx, test_idx) for cross-fitting on the test pool.
    keep: also return the arrays (mapped model prediction, A, B) for later bootstraps (not saved to JSON)."""
    rng = np.random.default_rng(0)
    A, B, BV = draws_vec(Vtest, n, kmax, rng)
    out = {"items": len(Vtest), "raw": nstar_block(D(Ptest), A, B, len(Vtest))}
    splits = fit_groups or [(None, np.arange(len(Vtest)))]
    Em = {"ridge": np.zeros(len(Vtest)), "iso": np.zeros(len(Vtest))}
    Eh = {"ridge": np.zeros_like(B), "iso": np.zeros_like(B)}
    infos = []
    for fi, ti in splits:
        Vf = Vfit if fi is None else [Vtest[i] for i in fi]
        Pf = Pfit if fi is None else Ptest[fi]
        t = target(Vf, np.random.default_rng(2), n)
        fns, info = fit_map(feats(Pf), t, np.arange(len(Vf)))
        info_k = []
        for s in Em:
            Em[s][ti] = fns[s](feats(Ptest[ti]))
        krng = np.random.default_rng(1)
        for k in range(1, kmax + 1):
            X, y, g = [], [], []
            for j, V in enumerate(Vf):
                for _ in range(N_FIT_DRAW_K):
                    X.append(feats(V[krng.permutation(len(V))[:k]].mean(0))); y.append(t[j]); g.append(j)
            fk, ik = fit_map(np.array(X), np.array(y), np.array(g))
            info_k.append(ik)
            Z = feats(BV[k - 1][:, ti].reshape(-1, BV.shape[-1]))
            for s in Eh:
                Eh[s][k - 1][:, ti] = fk[s](Z).reshape(N_DRAW, len(ti))
        infos.append({"model_map": info, "human_maps": info_k})
    main_m = infos[0]["model_map"]["main"]
    main_h = [ik["main"] for ik in infos[0]["human_maps"]]
    out["map_info"] = infos
    for s in ("ridge", "iso"):
        out[f"E_i_{s}"] = nstar_block(Em[s], A, B, len(Vtest))
        out[f"E_ii_{s}"] = nstar_block(Em[s], A, Eh[s], len(Vtest))
    # main = the map with the smaller CV MSE on the (first) fitting set
    out["E_i_main"] = dict(out[f"E_i_{main_m}"], map=main_m)
    Ehm = np.stack([Eh[main_h[k]][k] for k in range(kmax)])
    out["E_ii_main"] = dict(nstar_block(Em[main_m], A, Ehm, len(Vtest)), map_model=main_m, map_human=main_h)
    print(name, "raw n*", out["raw"]["nstar"], "E(i)", out["E_i_main"]["nstar"], out["E_i_main"]["ci95"],
          "E(ii)", out["E_ii_main"]["nstar"], out["E_ii_main"]["ci95"], "m_E", round(out["E_i_main"]["m"], 3))
    if keep:
        return out, {"Em": Em[main_m], "A": A, "B": B}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--votes", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--res3", required=True)
    ap.add_argument("--res4", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = lambda d: {x["id"]: x["V"] for x in pickle.load(open(f"{a.votes}/{d}.pkl", "rb"))["items"]}  # noqa: E731
    out = {}

    # BRIGHTER English (models of s3_judge and thinned-target models)
    V = L("brighter_eng"); sp = json.load(open(f"{a.split}/splits.json"))["brighter_eng"]
    dev, test = sp["dev"], [i for i in sp["test"] if len(V[i]) >= 8]
    for tag, rd in (("main", f"{a.res3}/brighter_eng/main"), ("thin1", f"{a.res4}/brighter_eng/thin1"),
                    ("thin3", f"{a.res4}/brighter_eng/thin3")):
        runs = sorted(glob.glob(f"{rd}/seed*/probs.pkl"))
        if runs:
            out[f"brighter_eng/{tag}"] = analyse(f"brighter_eng/{tag}", [V[i] for i in dev], ens(runs, dev, "dev"),
                                                 [V[i] for i in test], ens(runs, test, "test"), 3, 5)

    # ChaosNLI (dev fit) and DICES-990 (cross-fitting on out-of-fold predictions)
    sp2 = json.load(open(f"{a.split}/splits_e2.json"))
    V = L("chaosnli"); runs = sorted(glob.glob(f"{a.res4}/chaosnli/main/seed*/probs.pkl"))
    if runs:
        c = sp2["chaosnli"]
        out["chaosnli/main"] = analyse("chaosnli/main", [V[i] for i in c["dev"]], ens(runs, c["dev"], "dev"),
                                       [V[i] for i in c["test"]], ens(runs, c["test"], "test"), 3, 20)
    V = L("dices_990"); folds = sp2["dices_990"]["folds"]
    if all(glob.glob(f"{a.res4}/dices_990/fold{f}/seed*/probs.pkl") for f in range(5)):
        ids = [i for fl in folds for i in fl]
        P = np.concatenate([ens(sorted(glob.glob(f"{a.res4}/dices_990/fold{f}/seed*/probs.pkl")), fl, "test")
                            for f, fl in enumerate(folds)])
        pos = np.cumsum([0] + [len(fl) for fl in folds])
        cf = [(np.setdiff1d(np.arange(len(ids)), np.arange(pos[f], pos[f + 1])), np.arange(pos[f], pos[f + 1]))
              for f in range(5)]
        out["dices_990/main"] = analyse("dices_990/main", None, None, [V[i] for i in ids], P, 3, 20, cf)

    b = out.get("brighter_eng/main", {})
    out["H7"] = {"nstar_E_i": b.get("E_i_main", {}).get("nstar"), "nstar_E_ii": b.get("E_ii_main", {}).get("nstar")}
    out["H7"]["supported"] = all(v is not None and v < 2 for v in out["H7"].values())
    h8 = {d: {"raw": out[d]["raw"]["nstar"], "E_i": out[d]["E_i_main"]["nstar"]}
          for d in ("chaosnli/main", "dices_990/main") if d in out}
    out["H8"] = {"values": h8, "supported": len(h8) == 2 and all(v < 3 for x in h8.values() for v in x.values())}
    out["E3"] = {t: {"m_raw": out[f"brighter_eng/{t}"]["raw"]["m"], "m_E": out[f"brighter_eng/{t}"]["E_i_main"]["m"]}
                 for t in ("thin1", "thin3", "main") if f"brighter_eng/{t}" in out}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("H7", "H8", "E3")}, indent=1))


if __name__ == "__main__":
    main()
