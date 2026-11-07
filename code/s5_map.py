"""Matched condition: seven training votes and no MNLI checkpoint.

  python s5_map.py --votes DIR --split DIR --res5 DIR --out FILE.json
"""

import argparse
import glob
import json
import pickle

import numpy as np

from s4_map import analyse, ens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--votes", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--res5", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = lambda d: {x["id"]: x["V"] for x in pickle.load(open(f"{a.votes}/{d}.pkl", "rb"))["items"]}  # noqa: E731
    sp2 = json.load(open(f"{a.split}/splits_e2.json"))
    out = {}
    V = L("chaosnli"); c = sp2["chaosnli"]
    for tag in ("mnli_thin7", "mnli_thin3", "rl_full", "rl_thin7"):
        runs = sorted(glob.glob(f"{a.res5}/chaosnli/{tag}/seed*/probs.pkl"))
        out[f"chaosnli/{tag}"] = analyse(f"chaosnli/{tag}", [V[i] for i in c["dev"]], ens(runs, c["dev"], "dev"),
                                         [V[i] for i in c["test"]], ens(runs, c["test"], "test"), 3, 20)
    V = L("dices_990"); folds = sp2["dices_990"]["folds"]
    ids = [i for fl in folds for i in fl]
    pos = np.cumsum([0] + [len(fl) for fl in folds])
    cf = [(np.setdiff1d(np.arange(len(ids)), np.arange(pos[f], pos[f + 1])), np.arange(pos[f], pos[f + 1]))
          for f in range(5)]
    for tag in ("thin7", "thin3"):
        P = np.concatenate([ens(sorted(glob.glob(f"{a.res5}/dices_990/{tag}/fold{f}/seed*/probs.pkl")), fl, "test")
                            for f, fl in enumerate(folds)])
        out[f"dices_990/{tag}"] = analyse(f"dices_990/{tag}", None, None, [V[i] for i in ids], P, 3, 20, cf)
    h9 = {k: out[k]["E_i_main"]["nstar"] for k in ("chaosnli/mnli_thin7", "chaosnli/rl_thin7", "dices_990/thin7")}
    out["H9"] = {"values": h9, "supported": all(v >= 3 for v in h9.values())}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out["H9"], indent=1))


if __name__ == "__main__":
    main()
