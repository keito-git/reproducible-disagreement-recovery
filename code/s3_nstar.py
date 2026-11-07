"""Human curve c(n, k), model agreement m(n), and the annotator-equivalent count n*.

c(n, k) = corr(D_n^A, D_k^B) for disjoint human groups and m(n) = corr(D_model, D_n^A), on a fixed
item set and the same draws (200). n* interpolates c(n, .) at m(n); 500 item bootstraps give a 95% interval.

  python s3_nstar.py --votes DIR --res DIR --out FILE.json
"""

import argparse
import glob
import json
import pickle

import numpy as np

N_DRAW, N_BOOT = 200, 500
SETTINGS = {"brighter_eng": [(3, 5)], "goemotions": [(2, 3), (3, 2)]}


def corr_rows(x, Y):
    # Pearson correlation of vector x with every row of Y (draws x items) / or row-wise pairs.
    x = x - x.mean(-1, keepdims=True); Y = Y - Y.mean(-1, keepdims=True)
    return (x * Y).sum(-1) / np.sqrt((x ** 2).sum(-1) * (Y ** 2).sum(-1))


def draws(Vs, n, kmax, rng):
    """D_n^A (draws, items) and D_k^B for k = 1..kmax (kmax, draws, items)."""
    A = np.zeros((N_DRAW, len(Vs))); B = np.zeros((kmax, N_DRAW, len(Vs)))
    for j, V in enumerate(Vs):
        for d in range(N_DRAW):
            p = rng.permutation(len(V))
            A[d, j] = 1 - V[p[:n]].mean(0).max()
            cum = np.cumsum(V[p[n:n + kmax]], 0)
            B[:, d, j] = 1 - (cum / np.arange(1, kmax + 1)[:, None]).max(1)
    return A, B


def nstar(m, c):
    k = np.arange(1, len(c) + 1)
    if m >= c[-1]:
        return float("inf")
    if m < c[0]:
        return 0.0
    i = np.searchsorted(c, m) if np.all(np.diff(c) > 0) else np.argmax(c >= m)
    i = max(i, 1)
    return float(k[i - 1] + (m - c[i - 1]) / (c[i] - c[i - 1]))


def stats(Dm, A, B, idx):
    m = corr_rows(Dm[idx][None], A[:, idx]).mean()
    c = np.array([corr_rows(A[:, idx], B[k][:, idx]).mean() for k in range(len(B))])
    return m, c


def model_D(res_dir, ids):
    runs = sorted(glob.glob(f"{res_dir}/seed*/probs.pkl"))
    Ps = []
    for f in runs:
        r = pickle.load(open(f, "rb"))
        pos = {i: j for j, i in enumerate(r["dev"]["ids"] + r["test"]["ids"])}
        P = np.concatenate([r["dev"]["P"], r["test"]["P"]])
        Ps.append(P[[pos[i] for i in ids]])
    ens = np.mean(Ps, 0)
    return 1 - ens.max(1), [1 - P.max(1) for P in Ps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--votes", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sp = json.load(open(f"{a.split}/splits.json"))
    out = {}
    for data, settings in SETTINGS.items():
        V = {x["id"]: x["V"] for x in pickle.load(open(f"{a.votes}/{data}.pkl", "rb"))["items"]}
        pool = sp[data]["test"] if data == "brighter_eng" else sp[data]["dev"] + sp[data]["test"]
        for n, kmax in settings:
            ids = [i for i in pool if len(V[i]) >= n + kmax]
            rng = np.random.default_rng(0)
            A, B = draws([V[i] for i in ids], n, kmax, rng)
            for tag in ["main"] + (["frac0.25", "frac0.5"] if data == "brighter_eng" else []):
                rd = f"{a.res}/{data}/{tag}"
                if not glob.glob(f"{rd}/seed*/probs.pkl"):
                    continue
                Dm, singles = model_D(rd, ids)
                allidx = np.arange(len(ids))
                m, c = stats(Dm, A, B, allidx)
                brng = np.random.default_rng(1)
                boots = []
                for _ in range(N_BOOT):
                    idx = brng.integers(0, len(ids), len(ids))
                    mb, cb = stats(Dm, A, B, idx)
                    boots.append(nstar(mb, cb))
                boots = np.array(boots)
                fin = boots[np.isfinite(boots)]
                out[f"{data}/{tag}/n{n}"] = {
                    "items": len(ids), "m": float(m), "c": c.tolist(), "nstar": nstar(m, c),
                    "nstar_ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
                    "boot_share_above_kmax": float(np.mean(~np.isfinite(boots))),
                    "boot_median_finite": float(np.median(fin)) if len(fin) else None,
                    "m_single_seeds": [float(corr_rows(d[None], A).mean()) for d in singles]}
    H5 = {"brighter_eng_n3": out.get("brighter_eng/main/n3", {}).get("nstar", None),
          "goemotions_n2": out.get("goemotions/main/n2", {}).get("nstar", None)}
    out["H5"] = {"values": H5, "brighter_eng_ge3": H5["brighter_eng_n3"] is not None and H5["brighter_eng_n3"] >= 3,
                 "goemotions_ge2": H5["goemotions_n2"] is not None and H5["goemotions_n2"] >= 2}
    fr = [out.get(f"brighter_eng/{t}/n3", {}).get("nstar") for t in ("frac0.25", "frac0.5", "main")]
    out["H6"] = {"nstar_by_frac": fr, "increasing": all(x is not None for x in fr) and fr[0] < fr[1] < fr[2]}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k in ("H5", "H6")}, indent=1))
    for k, v in out.items():
        if "/" in k:
            print(k, "items", v["items"], "m", round(v["m"], 3), "c", [round(x, 3) for x in v["c"]],
                  "n*", v["nstar"], "CI", v["nstar_ci95"])


if __name__ == "__main__":
    main()
