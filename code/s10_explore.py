"""One-hot redraws, item-bootstrap intervals of R_text and of dataset differences, and R_text(n) for n = 2, 3, 4.

(a) one-hot votes redrawn 50 times (seeds 1000..1049): n*_E(i) and R_text for BRIGHTER n1868_thin7;
(b) item-bootstrap 95% CIs of R_text = m_E^2 / r33 per condition, and of domain differences
    (independent bootstraps per dataset, 500 each, seed 7);
(c) R_text(n) = m_E(n)^2 / r_nn for n = 2, 3, 4 (distribution models, E(i) map fitted to E[D_n]).

  python s10_explore.py --votes DIR --split DIR --res3 DIR --res4 DIR --res5 DIR --res6 DIR --res9 DIR --out FILE.json
"""

import argparse
import glob
import json
import pickle

import numpy as np

from s3_nstar import corr_rows
from s4_map import analyse, draws_vec, ens
from s8_onehot import onehot
from s9_map import ens_pred

N_BOOT = 500


def boot_R(Em, A, B, n, rng):
    N = A.shape[1]
    out = []
    for _ in range(N_BOOT):
        i = rng.integers(0, N, N)
        m = corr_rows(Em[i][None], A[:, i]).mean()
        r = corr_rows(A[:, i], B[n - 1][:, i]).mean()
        out.append(m ** 2 / r)
    return np.array(out)


def summ(x):
    return {"mean": float(np.mean(x)), "ci95": [float(np.quantile(x, .025)), float(np.quantile(x, .975))]}


def main():
    ap = argparse.ArgumentParser()
    for k in ("votes", "split", "res3", "res4", "res5", "res6", "res9", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    L = lambda d: {x["id"]: x["V"] for x in pickle.load(open(f"{a.votes}/{d}.pkl", "rb"))["items"]}  # noqa: E731
    out = {"a_onehot_redraws": [], "b_boot": {}, "b_diff": {}, "c_by_n": {}}
    sp = json.load(open(f"{a.split}/splits.json"))["brighter_eng"]
    sp2 = json.load(open(f"{a.split}/splits_e2.json"))
    Vb = L("brighter_eng"); dev = sp["dev"]; test_b = [i for i in sp["test"] if len(Vb[i]) >= 8]

    # conditions: name -> (Vfit, Pfit, Vtest, Ptest, kmax, fit_groups)
    def bri(rd):
        runs = sorted(glob.glob(f"{rd}/seed*/probs.pkl"))
        return [Vb[i] for i in dev], ens(runs, dev, "dev"), [Vb[i] for i in test_b], ens(runs, test_b, "test"), 5, None
    Vc = L("chaosnli"); c = sp2["chaosnli"]

    def cha(rd):
        runs = sorted(glob.glob(f"{rd}/seed*/probs.pkl"))
        return ([Vc[i] for i in c["dev"]], ens(runs, c["dev"], "dev"), [Vc[i] for i in c["test"]],
                ens(runs, c["test"], "test"), 20, None)
    Vd = L("dices_990"); folds = sp2["dices_990"]["folds"]; ids = [i for fl in folds for i in fl]
    pos = np.cumsum([0] + [len(fl) for fl in folds])
    cf = [(np.setdiff1d(np.arange(len(ids)), np.arange(pos[f], pos[f + 1])), np.arange(pos[f], pos[f + 1]))
          for f in range(5)]

    def dic(rd):
        P = np.concatenate([ens(sorted(glob.glob(f"{rd}/fold{f}/seed*/probs.pkl")), fl, "test")
                            for f, fl in enumerate(folds)])
        return None, None, [Vd[i] for i in ids], P, 20, cf
    conds = {"emotion_full": bri(f"{a.res3}/brighter_eng/main"),
             "emotion_matched": bri(f"{a.res6}/brighter_eng/n1868_thin7"),
             "nli_mnli_full": cha(f"{a.res4}/chaosnli/main"),
             "nli_matched": cha(f"{a.res5}/chaosnli/rl_thin7"),
             "safety_full": dic(f"{a.res4}/dices_990"),
             "safety_7votes": dic(f"{a.res5}/dices_990/thin7")}

    # (b) and (c)
    boots = {}
    for name, (Vf, Pf, Vt, Pt, km, fg) in conds.items():
        for n in (2, 3, 4):
            if 2 * n > min(len(v) for v in Vt):
                continue
            r, arr = analyse(f"{name}/n{n}", Vf, Pf, Vt, Pt, n, max(n, 3) if n != 3 else km, fg, keep=True)
            rnn = r["raw"]["c"][n - 1]
            R = r["E_i_main"]["m"] ** 2 / rnn
            out["c_by_n"].setdefault(name, {})[f"n{n}"] = {"r_nn": rnn, "m_E": r["E_i_main"]["m"], "R_text": R}
            if n == 3:
                bs = boot_R(arr["Em"], arr["A"], arr["B"], 3, np.random.default_rng(7))
                boots[name] = bs
                out["b_boot"][name] = dict(summ(bs), point=R)
    # direct regressors, n = 3
    for name, key, Vt, idl, km in (("emotion_matched_direct", "brighter_eng/n1868_thin7", [Vb[i] for i in test_b],
                                    test_b, 5),
                                   ("nli_matched_direct", "chaosnli/rl_thin7", [Vc[i] for i in c["test"]],
                                    c["test"], 20)):
        runs = sorted(glob.glob(f"{a.res9}/{key}/seed*/pred.pkl"))
        if not runs:
            continue
        A, B, _ = draws_vec(Vt, 3, km, np.random.default_rng(0))
        bs = boot_R(ens_pred(runs, idl, "test"), A, B, 3, np.random.default_rng(7))
        boots[name] = bs
        out["b_boot"][name] = summ(bs)
    runs = [sorted(glob.glob(f"{a.res9}/dices_990/thin7/fold{f}/seed*/pred.pkl")) for f in range(5)]
    if all(runs):
        P = np.concatenate([ens_pred(r, fl, "test") for r, fl in zip(runs, folds)])
        A, B, _ = draws_vec([Vd[i] for i in ids], 3, 20, np.random.default_rng(0))
        boots["safety_7votes_direct"] = boot_R(P, A, B, 3, np.random.default_rng(7))
        out["b_boot"]["safety_7votes_direct"] = summ(boots["safety_7votes_direct"])
    for x, y in (("emotion_matched", "nli_matched"), ("safety_7votes", "emotion_matched"),
                 ("safety_7votes", "nli_matched"), ("nli_mnli_full", "nli_matched"),
                 ("emotion_matched_direct", "nli_matched_direct")):
        if x in boots and y in boots:
            out["b_diff"][f"{x} - {y}"] = summ(boots[x] - boots[y])

    # (a) one-hot redraws
    items = pickle.load(open(f"{a.votes}/brighter_eng.pkl", "rb"))["items"]
    raw = {x["id"]: x["V"] for x in items}
    runs = sorted(glob.glob(f"{a.res6}/brighter_eng/n1868_thin7/seed*/probs.pkl"))
    Pd, Pt = ens(runs, dev, "dev"), ens(runs, test_b, "test")
    for s in range(1000, 1050):
        O = dict(zip(raw, onehot(list(raw.values()), np.random.default_rng(s))))
        r = analyse(f"onehot_seed{s}", [O[i] for i in dev], Pd, [O[i] for i in test_b], Pt, 3, 5)
        out["a_onehot_redraws"].append({"seed": s, "nstar_E_i": r["E_i_main"]["nstar"],
                                        "R_text": r["E_i_main"]["m"] ** 2 / r["raw"]["c"][2]})
        json.dump(out, open(a.out, "w"), indent=1)
    v = np.array([x["nstar_E_i"] for x in out["a_onehot_redraws"]])
    out["a_summary"] = {"nstar_E_i": summ(v), "share_below_2.73": float(np.mean(v < 2.73))}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("b_boot", "b_diff", "c_by_n", "a_summary")}, indent=1))


if __name__ == "__main__":
    main()
