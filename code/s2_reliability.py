"""Split-half reproducibility r_nn of the per-item disagreement D_n and the Spearman-Brown check.

r_nn: items with >= 2n votes; votes split at random into two disjoint groups of n; Pearson (and
Spearman) correlation of D_n between the groups; 200 splits. Spearman-Brown predictions from r_33
(or r_22 for GoEmotions) are compared with measured values on the same item set.

  python s2_reliability.py --votes DIR --out FILE.json
"""

import argparse
import json
import pickle

import numpy as np
from scipy.stats import ks_2samp, pearsonr, spearmanr

N_SPLIT = 200


def D(V):
    return 1.0 - V.mean(0).max()


def split_rel(items, n, rng):
    its = [x["V"] for x in items if len(x["V"]) >= 2 * n]
    rp, rs = [], []
    for _ in range(N_SPLIT):
        a, b = [], []
        for V in its:
            p = rng.permutation(len(V))
            a.append(D(V[p[:n]])); b.append(D(V[p[n:2 * n]]))
        rp.append(pearsonr(a, b)[0]); rs.append(spearmanr(a, b)[0])
    return {"n": n, "items": len(its), "pearson": float(np.mean(rp)), "pearson_sd": float(np.std(rp)),
            "spearman": float(np.mean(rs))}


def sb(r, k):
    return k * r / (1 + (k - 1) * r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--votes", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    L = lambda name: pickle.load(open(f"{a.votes}/{name}.pkl", "rb"))["items"]
    out = {}

    # H1: Spearman-Brown on the many-vote datasets, predicted and measured on one item set
    for name, min_votes in (("chaosnli", 40), ("dices_350", 40), ("dices_990", 40), ("brighter_eng", 20)):
        items = [x for x in L(name) if len(x["V"]) >= min_votes]
        ns = [3, 5, 10] + ([20] if min_votes >= 40 else [])
        rel = {n: split_rel(items, n, rng) for n in ns}
        r3 = rel[3]["pearson"]
        out[name] = {"item_set_min_votes": min_votes, "rel": rel,
                     "sb_from_r33": {n: float(sb(r3, n / 3)) for n in ns},
                     "abs_err": {n: float(abs(sb(r3, n / 3) - rel[n]["pearson"])) for n in ns if n > 3}}

    # H2: emotion data at n = 3
    be = L("brighter_eng")
    out["H2_brighter_eng"] = split_rel(be, 3, rng)
    ge = L("goemotions")
    r22 = split_rel(ge, 2, rng)
    out["H2_goemotions"] = {"r22": r22, "r33_sb": float(sb(r22["pearson"], 1.5))}

    # Is the 5-vote subset of GoEmotions a selected subset?
    d5 = [D(x["V"][rng.permutation(len(x["V"]))[:3]]) for x in ge if len(x["V"]) == 5]
    d3 = [D(x["V"]) for x in ge if len(x["V"]) == 3]
    ks = ks_2samp(d5, d3)
    out["goemotions_5vote_selection"] = {"n5": len(d5), "n3": len(d3), "mean_D3_of_5vote": float(np.mean(d5)),
                                         "mean_D3_of_3vote": float(np.mean(d3)), "ks_stat": float(ks.statistic),
                                         "ks_p": float(ks.pvalue)}
    out["H1"] = {k: v["abs_err"] for k, v in out.items() if "abs_err" in v}
    out["H2"] = {"brighter_eng_r33": out["H2_brighter_eng"]["pearson"],
                 "goemotions_r33_sb": out["H2_goemotions"]["r33_sb"]}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("H1", "H2", "goemotions_5vote_selection")}, indent=1))


if __name__ == "__main__":
    main()
