"""Judgment models for all datasets (distribution, vote thinning, train-size matching, majority-label targets).

Same recipe as s3_judge.train, with sentence-pair inputs. ChaosNLI uses SNLI + MNLI-m items split
60/10/30 (seed 0); DICES-990 uses 5-fold cross-fitting (seed 0); BRIGHTER English thinning keeps the
train items of s3_judge and builds each target from 1 or 3 of its votes (seed 200).

  python s4_judge.py split --votes DIR --raw DIR --out DIR
  python s4_judge.py train --votes DIR --raw DIR --split DIR --data {chaosnli,dices_990,brighter_eng,dices_350,brighter_deu} \
      --model NAME --lr LR --seed S [--fold F] [--thin K] [--ntrain N] [--hard] --out DIR
"""

import argparse
import json
import math
import os
import pickle

import numpy as np

from s3_judge import js

MAXLEN = {"chaosnli": 128, "dices_990": 256, "brighter_eng": 128, "dices_350": 256, "brighter_deu": 128}
CHAOS_FILES = ("chaosNLI_snli", "chaosNLI_mnli_m")


def load_texts(raw, data):
    """id -> text or (first, second) pair."""
    if data == "chaosnli":
        out = {}
        for f in CHAOS_FILES:
            for line in open(f"{raw}/chaosnli/raw/{f}.jsonl", encoding="utf-8"):
                d = json.loads(line)
                out[d["uid"]] = (d["example"]["premise"], d["example"]["hypothesis"])
        return out
    if data in ("dices_990", "dices_350"):
        import pandas as pd
        g = pd.read_csv(f"{raw}/{data}.csv", low_memory=False).groupby("item_id").first()
        return {int(k): (str(r["context"]), str(r["response"])) for k, r in g.iterrows()}
    return None


def cmd_split(a):
    os.makedirs(a.out, exist_ok=True)
    ch = pickle.load(open(f"{a.votes}/chaosnli.pkl", "rb"))["items"]
    ids = np.array([x["id"] for x in ch if x["subset"] in CHAOS_FILES])
    p = np.random.default_rng(0).permutation(len(ids))
    nt, nd = int(round(0.6 * len(ids))), int(round(0.1 * len(ids)))
    sp = {"chaosnli": {"train": ids[p[:nt]].tolist(), "dev": ids[p[nt:nt + nd]].tolist(),
                       "test": ids[p[nt + nd:]].tolist()}}
    di = np.array([x["id"] for x in pickle.load(open(f"{a.votes}/dices_990.pkl", "rb"))["items"]])
    fold = np.random.default_rng(0).permutation(len(di)) % 5
    sp["dices_990"] = {"folds": [di[fold == f].tolist() for f in range(5)]}
    json.dump(sp, open(f"{a.out}/splits_e2.json", "w"))
    print({"chaosnli": {k: len(v) for k, v in sp["chaosnli"].items()},
           "dices_990": [len(f) for f in sp["dices_990"]["folds"]]})


def encode(tok, texts, maxlen):
    if isinstance(texts[0], tuple):
        return tok(list(texts), truncation="only_first", max_length=maxlen, padding="max_length",
                   return_tensors="pt")
    return tok(list(texts), truncation=True, max_length=maxlen, padding="max_length", return_tensors="pt")


def train(model_name, nc, texts, targets, seed, lr, maxlen, epochs=3, min_steps=300, bs=32):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    if isinstance(texts[0], tuple):
        tok.truncation_side = "left"     # pairs: drop the start of the first segment (context / premise)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=nc, attn_implementation="eager").cuda()
    if "mnli" in model_name:
        # checkpoint head order is (contradiction, neutral, entailment); ChaosNLI counts are (e, n, c)
        with torch.no_grad():
            for prm in (model.classifier.out_proj.weight, model.classifier.out_proj.bias):
                prm.copy_(prm[[2, 1, 0]].clone())
    enc = encode(tok, texts, maxlen)
    T = torch.as_tensor(targets, dtype=torch.float32)
    n = len(texts)
    total = max(epochs * math.ceil(n / bs), min_steps)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    g = torch.Generator().manual_seed(seed)
    step = 0
    model.train()
    while step < total:
        for idx in torch.randperm(n, generator=g).split(bs):
            if step >= total:
                break
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=enc["input_ids"][idx].cuda(),
                               attention_mask=enc["attention_mask"][idx].cuda()).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), T[idx].cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
            step += 1
    model.eval()
    return model, tok


def predict(model, tok, texts, maxlen, bs=64):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            e = encode(tok, texts[i:i + bs], maxlen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(input_ids=e["input_ids"].cuda(), attention_mask=e["attention_mask"].cuda()).logits
            out.append(torch.softmax(lg.float(), -1).cpu().numpy())
    return np.concatenate(out)


def cmd_train(a):
    os.makedirs(a.out, exist_ok=True)
    V = {x["id"]: x for x in pickle.load(open(f"{a.votes}/{a.data}.pkl", "rb"))["items"]}
    T = load_texts(a.raw, a.data)
    text = (lambda i: T[i]) if T is not None else (lambda i: V[i]["text"])
    if a.data == "brighter_eng":
        sp = json.load(open(f"{a.split}/splits.json"))["brighter_eng"]
        evals = {"dev": sp["dev"], "test": sp["test"]}
        tr = list(sp["train"])
        if a.ntrain > 0:  # matched condition: subsample train items to match ChaosNLI
            tr = [tr[i] for i in np.sort(np.random.default_rng(300).permutation(len(tr))[:a.ntrain])]
    elif a.data in ("chaosnli", "brighter_deu"):
        f = "splits_e2.json" if a.data == "chaosnli" else "splits_j.json"
        sp = json.load(open(f"{a.split}/{f}"))[a.data]
        evals = {"dev": sp["dev"], "test": sp["test"]}
        tr = list(sp["train"])
    else:
        f = "splits_e2.json" if a.data == "dices_990" else "splits_j.json"
        folds = json.load(open(f"{a.split}/{f}"))[a.data]["folds"]
        evals = {"dev": [], "test": folds[a.fold]}
        tr = [i for f, fl in enumerate(folds) if f != a.fold for i in fl]
    if a.thin > 0:
        rng = np.random.default_rng(200)
        tgt = np.stack([V[i]["V"][rng.permutation(len(V[i]["V"]))[:a.thin]].mean(0) for i in tr])
    else:
        tgt = np.stack([V[i]["V"].mean(0) for i in tr])
    if a.hard:  # majority-label targets (ties -> lowest class index)
        tgt = np.eye(tgt.shape[1])[tgt.argmax(1)]
    nc = tgt.shape[1]
    ml = MAXLEN[a.data]
    model, tok = train(a.model, nc, [text(i) for i in tr], tgt, a.seed, a.lr, ml)
    res = {}
    for s, ids in evals.items():
        P = predict(model, tok, [text(i) for i in ids], ml) if ids else np.zeros((0, nc))
        res[s] = {"ids": ids, "P": P}
    pickle.dump(res, open(f"{a.out}/probs.pkl", "wb"))
    ev = "dev" if evals["dev"] else "test"
    pd_ = np.stack([V[i]["V"].mean(0) for i in evals[ev]])
    info = {"model": a.model, "lr": a.lr, "seed": a.seed, "fold": a.fold, "thin": a.thin, "hard": a.hard,
            "n_train": len(tr),
            f"{ev}_jsd": float(js(pd_, res[ev]["P"]).mean())}
    json.dump(info, open(f"{a.out}/info.json", "w"), indent=1)
    print(json.dumps(info))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["split", "train"])
    ap.add_argument("--votes", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--split")
    ap.add_argument("--data")
    ap.add_argument("--model")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--thin", type=int, default=0)
    ap.add_argument("--ntrain", type=int, default=0)
    ap.add_argument("--hard", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"split": cmd_split, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
