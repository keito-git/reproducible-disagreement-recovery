"""BRIGHTER German: r33 and raw model correlation on items with >= 6, 7, 8 annotators.

Compares r33 and the raw model correlation on test items with >=6, >=7, >=8 annotators (same draw scheme as s4_map).
"""
import glob, json, pickle
import numpy as np

V = {x["id"]: x["V"] for x in pickle.load(open("data/votes/brighter_deu.pkl", "rb"))["items"]}
te = json.load(open("data/split/splits_j.json"))["brighter_deu"]["test"]
runs = sorted(glob.glob("results_raw/s13/brighter_deu/main/seed*/probs.pkl"))
Ps = []
for f in runs:
    r = pickle.load(open(f, "rb"))["test"]; pos = {i: j for j, i in enumerate(r["ids"])}
    Ps.append(r["P"][[pos[i] for i in te]])
P = np.mean(Ps, 0); dm = 1 - P.max(1)
out = {}
for mn in (6, 7, 8):
    idx = [j for j, i in enumerate(te) if len(V[i]) >= mn]
    rng = np.random.default_rng(0); cs, ms = [], []
    for _ in range(200):
        A, B = [], []
        for j in idx:
            v = V[te[j]]; p = rng.permutation(len(v))
            A.append(1 - v[p[:3]].mean(0).max()); B.append(1 - v[p[3:6]].mean(0).max())
        A, B = np.array(A), np.array(B)
        cs.append(np.corrcoef(A, B)[0, 1]); ms.append(np.corrcoef(dm[idx], A)[0, 1])
    r33, m = float(np.mean(cs)), float(np.mean(ms))
    out[f">={mn}"] = {"items": len(idx), "r33": r33, "m_raw": m, "R_raw": m * m / r33}
print(json.dumps(out, indent=1))
