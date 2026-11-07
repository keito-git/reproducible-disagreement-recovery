"""Matched condition: BRIGHTER English with 1,868 training items (the ChaosNLI training size).

  python s6_map.py --votes DIR --split DIR --res6 DIR --s5 FILE.json --out FILE.json
"""

import argparse
import glob
import json
import pickle

from s4_map import analyse, ens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--votes", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--res6", required=True)
    ap.add_argument("--s5", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    V = {x["id"]: x["V"] for x in pickle.load(open(f"{a.votes}/brighter_eng.pkl", "rb"))["items"]}
    sp = json.load(open(f"{a.split}/splits.json"))["brighter_eng"]
    dev, test = sp["dev"], [i for i in sp["test"] if len(V[i]) >= 8]
    out = {}
    for tag in ("n1868_thin7", "n1868_full"):
        runs = sorted(glob.glob(f"{a.res6}/brighter_eng/{tag}/seed*/probs.pkl"))
        out[f"brighter_eng/{tag}"] = analyse(f"brighter_eng/{tag}", [V[i] for i in dev], ens(runs, dev, "dev"),
                                             [V[i] for i in test], ens(runs, test, "test"), 3, 5)
    ref = json.load(open(a.s5))["chaosnli/rl_thin7"]["E_i_main"]["nstar"]
    v = out["brighter_eng/n1868_thin7"]["E_i_main"]["nstar"]
    out["H10"] = {"brighter_n1868_thin7": v, "chaosnli_rl_thin7": ref, "supported": v < ref}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out["H10"], indent=1))


if __name__ == "__main__":
    main()
