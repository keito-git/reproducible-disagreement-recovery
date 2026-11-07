"""Comparison predictors, frozen LLM, second datasets, and robustness checks.

Result keys: J1 majority-label classifiers (E(i) map) -> R_text; J2 surface-feature ridge -> R_text;
J3 frozen-LLM embedding ridge -> R_text; J4 second datasets (DICES-350, BRIGHTER German) -> R_text;
J5 checks: one-hot emotion CI and difference to NLI, ridge-alpha range sensitivity, per-seed R_text.
All R_text use n = 3; bootstraps as in s10_explore (500, seed 7).

  python s13_map.py --votes DIR --raw DIR --split DIR --res DIR --out FILE.json
"""

import argparse
import glob
import json
import pickle
import string

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import s4_map
from s4_judge import load_texts
from s4_map import analyse, draws_vec, ens, nstar_block
from s8_onehot import onehot
from s9_direct import d3_target
from s10_explore import boot_R, summ


def train_targets(Vd, tr, thin=7):
    rt = np.random.default_rng(200)   # same thinning stream as s4_judge / s9_direct
    Vtr = [Vd[i][rt.permutation(len(Vd[i]))[:thin]] for i in tr]
    ry = np.random.default_rng(500)
    return np.array([d3_target(v, ry) for v in Vtr])


def surface(t):
    segs = t if isinstance(t, tuple) else (t,)
    f = []
    for s in segs:
        w = s.split()
        f += [len(s), len(w), s.count("!"), s.count("?"), sum(c in string.punctuation for c in s),
              sum(c.isupper() for c in s) / max(len(s), 1)]
    if len(segs) == 2:
        a, b = (set(x.lower().split()) for x in segs)
        f.append(len(a & b) / max(len(a | b), 1))
    return f


def reg_R(Vt, pred, kmax, extra=None):
    A, B, _ = draws_vec(Vt, 3, kmax, np.random.default_rng(0))
    r = nstar_block(pred, A, B, len(Vt))
    r["r33"] = r["c"][2]
    r["R_text"] = r["m"] ** 2 / r["r33"]
    bs = boot_R(pred, A, B, 3, np.random.default_rng(7))
    r["R_boot"] = summ(bs)
    return r, bs


def dist_R(name, Vf, Pf, Vt, Pt, kmax, fg=None):
    r, arr = analyse(name, Vf, Pf, Vt, Pt, 3, kmax, fg, keep=True)
    R = r["E_i_main"]["m"] ** 2 / r["raw"]["c"][2]
    bs = boot_R(arr["Em"], arr["A"], arr["B"], 3, np.random.default_rng(7))
    return {"R_text": R, "R_boot": summ(bs), "m_E": r["E_i_main"]["m"], "r33": r["raw"]["c"][2],
            "nstar_E_i": r["E_i_main"]["nstar"]}, bs


def main():
    ap = argparse.ArgumentParser()
    for k in ("votes", "raw", "split", "res", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    R = a.res
    L = lambda d: {x["id"]: x for x in pickle.load(open(f"{a.votes}/{d}.pkl", "rb"))["items"]}  # noqa: E731
    out, boots = {}, {}
    dump = lambda: json.dump(out, open(a.out, "w"), indent=1)  # noqa: E731

    # ---- the three matched conditions: test sets, train sets, texts
    Xb = L("brighter_eng"); Vb = {i: x["V"] for i, x in Xb.items()}
    sp = json.load(open(f"{a.split}/splits.json"))["brighter_eng"]
    dev_b, test_b = sp["dev"], [i for i in sp["test"] if len(Vb[i]) >= 8]
    tr_b = [sp["train"][i] for i in np.sort(np.random.default_rng(300).permutation(len(sp["train"]))[:1868])]
    sp2 = json.load(open(f"{a.split}/splits_e2.json"))
    Vc = {i: x["V"] for i, x in L("chaosnli").items()}; c = sp2["chaosnli"]
    Vd = {i: x["V"] for i, x in L("dices_990").items()}; folds = sp2["dices_990"]["folds"]
    ids_d = [i for fl in folds for i in fl]
    pos = np.cumsum([0] + [len(fl) for fl in folds])
    cf = [(np.setdiff1d(np.arange(len(ids_d)), np.arange(pos[f], pos[f + 1])), np.arange(pos[f], pos[f + 1]))
          for f in range(5)]
    Tc, Td = load_texts(a.raw, "chaosnli"), load_texts(a.raw, "dices_990")
    txt = {"brighter_eng": lambda i: Xb[i]["text"], "chaosnli": lambda i: Tc[i], "dices_990": lambda i: Td[i]}

    def fit_pred(data, feat, model_fn):
        """Ridge regressors on matched training sets (DICES: per fold) -> test predictions in evaluation order."""
        if data == "brighter_eng":
            m = model_fn().fit(feat(tr_b), train_targets(Vb, tr_b)); return m.predict(feat(test_b))
        if data == "chaosnli":
            m = model_fn().fit(feat(c["train"]), train_targets(Vc, c["train"])); return m.predict(feat(c["test"]))
        P = []
        for f, fl in enumerate(folds):
            tr = [i for g, gl in enumerate(folds) if g != f for i in gl]
            P.append(model_fn().fit(feat(tr), train_targets(Vd, tr)).predict(feat(fl)))
        return np.concatenate(P)
    evalset = {"brighter_eng": ([Vb[i] for i in test_b], 5), "chaosnli": ([Vc[i] for i in c["test"]], 20),
               "dices_990": ([Vd[i] for i in ids_d], 20)}

    # ---- surface features
    out["J2_surface"] = {}
    for data in evalset:
        sf = lambda ids, d=data: np.array([surface(txt[d](i)) for i in ids], float)  # noqa: E731
        mk = lambda: make_pipeline(StandardScaler(), PolynomialFeatures(2), RidgeCV(np.logspace(-2, 4, 13)))  # noqa
        r, _ = reg_R(*[evalset[data][0]], fit_pred(data, sf, mk), evalset[data][1])
        out["J2_surface"][data] = {k: r[k] for k in ("m", "R_text", "R_boot", "nstar")}
        print("J2", data, round(r["R_text"], 3), flush=True)
    dump()

    # ---- frozen LLM embeddings
    out["J3_llm"] = {}
    for data in evalset:
        f = f"{R}/s13/llm/{data}.pkl"
        if not glob.glob(f):
            continue
        E = pickle.load(open(f, "rb")); row = {i: j for j, i in enumerate(E["ids"])}
        ef = lambda ids, E=E, row=row: E["X"][[row[i] for i in ids]]  # noqa: E731
        mk = lambda: make_pipeline(StandardScaler(), RidgeCV(np.logspace(0, 6, 13)))  # noqa: E731
        r, bs = reg_R(evalset[data][0], fit_pred(data, ef, mk), evalset[data][1])
        boots[f"llm/{data}"] = bs
        out["J3_llm"][data] = {k: r[k] for k in ("m", "R_text", "R_boot", "nstar")}
        print("J3", data, round(r["R_text"], 3), flush=True)
    dump()

    # ---- majority-label classifiers vs soft-label models (matched conditions)
    out["J1_hard"] = {}
    conds = {"brighter_eng": ("brighter_eng/n1868_thin7", f"{R}/s6/brighter_eng/n1868_thin7"),
             "chaosnli": ("chaosnli/rl_thin7", f"{R}/s5/chaosnli/rl_thin7"),
             "dices_990": ("dices_990/thin7", f"{R}/s5/dices_990/thin7")}
    for data, (key, softdir) in conds.items():
        res = {}
        for tag, rd in (("hard", f"{R}/s13/hard/{key}"), ("soft", softdir)):
            if data == "dices_990":
                P = np.concatenate([ens(sorted(glob.glob(f"{rd}/fold{f}/seed*/probs.pkl")), fl, "test")
                                    for f, fl in enumerate(folds)])
                res[tag], boots[f"{tag}/{data}"] = dist_R(f"{tag}/{data}", None, None, evalset[data][0], P, 3, cf)
            else:
                runs = sorted(glob.glob(f"{rd}/seed*/probs.pkl"))
                dv, te = (dev_b, test_b) if data == "brighter_eng" else (c["dev"], c["test"])
                Vx = Vb if data == "brighter_eng" else Vc
                res[tag], boots[f"{tag}/{data}"] = dist_R(f"{tag}/{data}", [Vx[i] for i in dv], ens(runs, dv, "dev"),
                                                          [Vx[i] for i in te], ens(runs, te, "test"), 3)
        res["ratio_hard_to_soft"] = res["hard"]["R_text"] / res["soft"]["R_text"]
        out["J1_hard"][data] = res
        print("J1", data, round(res["hard"]["R_text"], 3), round(res["soft"]["R_text"], 3), flush=True)
    out["J1_hard"]["all_ratio_ge_0.9"] = all(out["J1_hard"][d]["ratio_hard_to_soft"] >= 0.9 for d in conds)
    dump()

    # ---- second datasets
    out["J4"] = {}
    spj = json.load(open(f"{a.split}/splits_j.json"))
    V3 = {i: x["V"] for i, x in L("dices_350").items()}; f3 = spj["dices_350"]["folds"]
    ids3 = [i for fl in f3 for i in fl]; p3 = np.cumsum([0] + [len(fl) for fl in f3])
    cf3 = [(np.setdiff1d(np.arange(len(ids3)), np.arange(p3[f], p3[f + 1])), np.arange(p3[f], p3[f + 1]))
           for f in range(5)]
    for tag in ("full", "thin7"):
        P = np.concatenate([ens(sorted(glob.glob(f"{R}/s13/dices_350/{tag}/fold{f}/seed*/probs.pkl")), fl, "test")
                            for f, fl in enumerate(f3)])
        out["J4"][f"dices_350/{tag}"], boots[f"dices_350/{tag}"] = dist_R(f"dices_350/{tag}", None, None,
                                                                          [V3[i] for i in ids3], P, 20, cf3)
    Vg = {i: x["V"] for i, x in L("brighter_deu").items()}; sg = spj["brighter_deu"]
    test_g = [i for i in sg["test"] if len(Vg[i]) >= 6]
    runs = sorted(glob.glob(f"{R}/s13/brighter_deu/main/seed*/probs.pkl"))
    out["J4"]["brighter_deu/main"], boots["brighter_deu/main"] = dist_R(
        "brighter_deu/main", [Vg[i] for i in sg["dev"]], ens(runs, sg["dev"], "dev"),
        [Vg[i] for i in test_g], ens(runs, test_g, "test"), 3)
    out["J4"]["brighter_deu/main"]["test_items"] = len(test_g)
    soft_b = out["J1_hard"]["brighter_eng"]["soft"]["R_text"]
    r350, rdeu = out["J4"]["dices_350/thin7"]["R_text"], out["J4"]["brighter_deu/main"]["R_text"]
    out["H13"] = {"dices_350_thin7": r350, "brighter_deu": rdeu, "brighter_eng_matched": soft_b,
                  "supported": r350 > rdeu and r350 > soft_b}
    for x, y in (("dices_350/thin7", "brighter_deu/main"), ("dices_350/thin7", "soft/brighter_eng"),
                 ("brighter_deu/main", "soft/chaosnli"), ("soft/brighter_eng", "soft/chaosnli")):
        out.setdefault("diff", {})[f"{x} - {y}"] = summ(boots[x] - boots[y])
    print("H13", out["H13"], flush=True)
    dump()

    # ---- robustness checks
    out["J5"] = {}
    runs6 = sorted(glob.glob(f"{R}/s6/brighter_eng/n1868_thin7/seed*/probs.pkl"))
    O = {i: v for i, v in zip(Vb, onehot(list(Vb.values()), np.random.default_rng(400)))}
    out["J5"]["onehot_emotion"], boots["onehot"] = dist_R("onehot", [O[i] for i in dev_b], ens(runs6, dev_b, "dev"),
                                                          [O[i] for i in test_b], ens(runs6, test_b, "test"), 5)
    out["diff"]["onehot_emotion - soft/chaosnli"] = summ(boots["onehot"] - boots["soft/chaosnli"])
    old = s4_map.ALPHAS
    s4_map.ALPHAS = old + (1000, 10000)
    out["J5"]["alpha_wide"] = {}
    for tag, rd in (("brighter_eng/main", f"{R}/s3/brighter_eng/main"), ("brighter_eng/n1868_thin7", f"{R}/s6/brighter_eng/n1868_thin7")):
        runs = sorted(glob.glob(f"{rd}/seed*/probs.pkl"))
        r = analyse(f"alpha/{tag}", [Vb[i] for i in dev_b], ens(runs, dev_b, "dev"), [Vb[i] for i in test_b],
                    ens(runs, test_b, "test"), 3, 3)
        out["J5"]["alpha_wide"][tag] = {"m_E_ridge": r["E_i_ridge"]["m"], "m_E_iso": r["E_i_iso"]["m"],
                                        "main": r["map_info"][0]["model_map"]}
    s4_map.ALPHAS = old
    dump()
    out["J5"]["per_seed"] = {}
    single = {"emotion_full": ("b", f"{R}/s3/brighter_eng/main"), "emotion_matched": ("b", f"{R}/s6/brighter_eng/n1868_thin7"),
              "nli_mnli_full": ("c", f"{R}/s4/chaosnli/main"), "nli_matched": ("c", f"{R}/s5/chaosnli/rl_thin7"),
              "safety_full": ("d", f"{R}/s4/dices_990"), "safety_7votes": ("d", f"{R}/s5/dices_990/thin7")}
    for name, (kind, rd) in single.items():
        vals = []
        for s in range(3):
            if kind == "d":
                P = np.concatenate([ens([f"{rd}/fold{f}/seed{s}/probs.pkl"], fl, "test") for f, fl in enumerate(folds)])
                r = analyse(f"{name}/s{s}", None, None, evalset["dices_990"][0], P, 3, 3, cf)
            else:
                Vx, dv, te = (Vb, dev_b, test_b) if kind == "b" else (Vc, c["dev"], c["test"])
                run = [f"{rd}/seed{s}/probs.pkl"]
                r = analyse(f"{name}/s{s}", [Vx[i] for i in dv], ens(run, dv, "dev"), [Vx[i] for i in te],
                            ens(run, te, "test"), 3, 3)
            vals.append(r["E_i_main"]["m"] ** 2 / r["raw"]["c"][2])
        out["J5"]["per_seed"][name] = {"R_text": vals, "sd": float(np.std(vals, ddof=1))}
        print("seed", name, np.round(vals, 3), flush=True)
    dump()
    print("J_DONE", flush=True)


if __name__ == "__main__":
    main()
