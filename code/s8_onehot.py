"""BRIGHTER English with one-hot votes (one class drawn per annotator, seed 400).

Recomputes the human curves, the finite-n map, n* and the three-way split with the existing ensembles.

  python s8_onehot.py --votes DIR --split DIR --res3 DIR --res6 DIR --s5 FILE.json --out FILE.json
"""

import argparse
import glob
import json
import pickle

import numpy as np

from s4_map import analyse, ens


def onehot(Vs, rng):
    out = []
    for V in Vs:
        cls = [rng.choice(V.shape[1], p=row / row.sum()) for row in V]
        out.append(np.eye(V.shape[1])[cls])
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("votes", "split", "res3", "res6", "s5", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    items = pickle.load(open(f"{a.votes}/brighter_eng.pkl", "rb"))["items"]
    ids = [x["id"] for x in items]
    O = dict(zip(ids, onehot([x["V"] for x in items], np.random.default_rng(400))))
    sp = json.load(open(f"{a.split}/splits.json"))["brighter_eng"]
    dev, test = sp["dev"], [i for i in sp["test"] if len(O[i]) >= 8]
    out = {}
    for tag, rd in (("main", f"{a.res3}/brighter_eng/main"), ("n1868_thin7", f"{a.res6}/brighter_eng/n1868_thin7")):
        runs = sorted(glob.glob(f"{rd}/seed*/probs.pkl"))
        r = analyse(f"onehot/{tag}", [O[i] for i in dev], ens(runs, dev, "dev"),
                    [O[i] for i in test], ens(runs, test, "test"), 3, 5)
        r33, m = r["raw"]["c"][2], r["E_i_main"]["m"]
        r["decomp"] = {"r33": r33, "m_E": m, "noise": 1 - r33, "explained": m ** 2, "unexplained": r33 - m ** 2,
                       "explained_share_of_stable": m ** 2 / r33}
        out[f"brighter_eng_onehot/{tag}"] = r
    ref = json.load(open(a.s5))["chaosnli/rl_thin7"]["E_i_main"]["nstar"]
    v = out["brighter_eng_onehot/n1868_thin7"]["E_i_main"]["nstar"]
    out["H11"] = {"brighter_onehot_n1868_thin7": v, "chaosnli_rl_thin7": ref, "supported": v < ref}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out["H11"], indent=1))
    for k in out:
        if "decomp" in out.get(k, {}):
            print(k, {x: round(y, 3) for x, y in out[k]["decomp"].items()})


if __name__ == "__main__":
    main()
