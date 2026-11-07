"""n* and R_text = m^2 / r33 for the direct regression models.

  python s9_map.py --votes DIR --split DIR --res9 DIR --snap DIR --out FILE.json
"""

import argparse
import glob
import json
import pickle

import numpy as np

from s4_map import draws_vec, nstar_block


def ens_pred(runs, ids, split):
    Ps = []
    for f in runs:
        r = pickle.load(open(f, "rb"))
        pos = {i: j for j, i in enumerate(r[split]["ids"])}
        Ps.append(r[split]["pred"][[pos[i] for i in ids]])
    return np.mean(Ps, 0)


def block(Vtest, pred, kmax):
    A, B, _ = draws_vec(Vtest, 3, kmax, np.random.default_rng(0))
    r = nstar_block(pred, A, B, len(Vtest))
    r33 = r["c"][2]
    r["R_text"] = r["m"] ** 2 / r33
    r["r33"] = r33
    return r


def main():
    ap = argparse.ArgumentParser()
    for k in ("votes", "split", "res9", "snap", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    L = lambda d: {x["id"]: x["V"] for x in pickle.load(open(f"{a.votes}/{d}.pkl", "rb"))["items"]}  # noqa: E731
    out = {}
    V = L("brighter_eng"); sp = json.load(open(f"{a.split}/splits.json"))["brighter_eng"]
    test = [i for i in sp["test"] if len(V[i]) >= 8]
    for tag in ("full", "n1868_thin7"):
        runs = sorted(glob.glob(f"{a.res9}/brighter_eng/{tag}/seed*/pred.pkl"))
        out[f"brighter_eng/{tag}"] = block([V[i] for i in test], ens_pred(runs, test, "test"), 5)
    sp2 = json.load(open(f"{a.split}/splits_e2.json"))
    V = L("chaosnli"); test = sp2["chaosnli"]["test"]
    runs = sorted(glob.glob(f"{a.res9}/chaosnli/rl_thin7/seed*/pred.pkl"))
    out["chaosnli/rl_thin7"] = block([V[i] for i in test], ens_pred(runs, test, "test"), 20)
    V = L("dices_990"); folds = sp2["dices_990"]["folds"]
    ids = [i for fl in folds for i in fl]
    P = np.concatenate([ens_pred(sorted(glob.glob(f"{a.res9}/dices_990/thin7/fold{f}/seed*/pred.pkl")), fl, "test")
                        for f, fl in enumerate(folds)])
    out["dices_990/thin7"] = block([V[i] for i in ids], P, 20)
    # distribution-model counterparts (E(i)) from the earlier snapshots
    S = {f: json.load(open(f"{a.snap}/{f}")) for f in ("s4_map.json", "s5_map.json", "s6_map.json")}
    ref = {"brighter_eng/full": S["s4_map.json"]["brighter_eng/main"],
           "brighter_eng/n1868_thin7": S["s6_map.json"]["brighter_eng/n1868_thin7"],
           "chaosnli/rl_thin7": S["s5_map.json"]["chaosnli/rl_thin7"],
           "dices_990/thin7": S["s5_map.json"]["dices_990/thin7"]}
    for k, d in ref.items():
        r33 = d["raw"]["c"][2]
        out[k]["dist_E_i"] = {"m": d["E_i_main"]["m"], "nstar": d["E_i_main"]["nstar"],
                              "R_text": d["E_i_main"]["m"] ** 2 / r33}
        out[k]["gain_R_text"] = out[k]["R_text"] - out[k]["dist_E_i"]["R_text"]
    out["H12"] = {k: out[k]["gain_R_text"] for k in ("brighter_eng/full", "brighter_eng/n1868_thin7")}
    out["H12"]["supported"] = all(v < 0.10 for v in list(out["H12"].values()))
    json.dump(out, open(a.out, "w"), indent=1)
    for k in ref:
        print(k, "direct m", round(out[k]["m"], 3), "n*", out[k]["nstar"], "R", round(out[k]["R_text"], 3),
              "| dist R", round(out[k]["dist_E_i"]["R_text"], 3), "gain", round(out[k]["gain_R_text"], 3))
    print(json.dumps(out["H12"], indent=1))


if __name__ == "__main__":
    main()
