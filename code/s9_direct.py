"""Direct regression model of the finite-group disagreement E[D_3 | x].

Same encoder, splits, train items, vote thinning and step budget as s4_judge.train; the head has one output
and the loss is MSE to t(x) = E[D_3] over 200 three-vote draws of the item's (thinned) training votes (seed 500).

  python s9_direct.py --votes DIR --raw DIR --split DIR --data {chaosnli,dices_990,brighter_eng} \
      --seed S [--fold F] [--thin K] [--ntrain N] --out DIR
"""

import argparse
import json
import math
import os
import pickle

import numpy as np

from s4_judge import MAXLEN, encode, load_texts

N_T = 200


def d3_target(V, rng, n=3):
    return float(np.mean([1 - V[rng.permutation(len(V))[:n]].mean(0).max() for _ in range(N_T)]))


def train(model_name, texts, y, seed, lr, maxlen, epochs=3, min_steps=300, bs=32):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    if isinstance(texts[0], tuple):
        tok.truncation_side = "left"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, attn_implementation="eager").cuda()
    enc = encode(tok, texts, maxlen)
    T = torch.as_tensor(y, dtype=torch.float32)
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
                out = model(input_ids=enc["input_ids"][idx].cuda(),
                            attention_mask=enc["attention_mask"][idx].cuda()).logits
            loss = torch.nn.functional.mse_loss(out.float().squeeze(-1), T[idx].cuda())
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
            out.append(lg.float().squeeze(-1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    for k in ("votes", "raw", "split", "data", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--model", default="roberta-large")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--thin", type=int, default=0)
    ap.add_argument("--ntrain", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    V = {x["id"]: x for x in pickle.load(open(f"{a.votes}/{a.data}.pkl", "rb"))["items"]}
    T = load_texts(a.raw, a.data)
    text = (lambda i: T[i]) if T is not None else (lambda i: V[i]["text"])
    if a.data == "brighter_eng":
        sp = json.load(open(f"{a.split}/splits.json"))["brighter_eng"]
        evals, tr = {"dev": sp["dev"], "test": sp["test"]}, list(sp["train"])
        if a.ntrain > 0:  # same subsample as s4_judge --ntrain
            tr = [tr[i] for i in np.sort(np.random.default_rng(300).permutation(len(tr))[:a.ntrain])]
    elif a.data == "chaosnli":
        sp = json.load(open(f"{a.split}/splits_e2.json"))["chaosnli"]
        evals, tr = {"dev": sp["dev"], "test": sp["test"]}, list(sp["train"])
    else:
        folds = json.load(open(f"{a.split}/splits_e2.json"))["dices_990"]["folds"]
        evals = {"dev": [], "test": folds[a.fold]}
        tr = [i for f, fl in enumerate(folds) if f != a.fold for i in fl]
    rt = np.random.default_rng(200)  # same thinning stream as s4_judge --thin
    Vtr = [V[i]["V"][rt.permutation(len(V[i]["V"]))[:a.thin]] if a.thin > 0 else V[i]["V"] for i in tr]
    ry = np.random.default_rng(500)
    y = np.array([d3_target(v, ry) for v in Vtr])
    ml = MAXLEN[a.data]
    model, tok = train(a.model, [text(i) for i in tr], y, a.seed, a.lr, ml)
    res = {s: {"ids": ids, "pred": predict(model, tok, [text(i) for i in ids], ml) if ids else np.zeros(0)}
           for s, ids in evals.items()}
    pickle.dump(res, open(f"{a.out}/pred.pkl", "wb"))
    info = {"data": a.data, "seed": a.seed, "fold": a.fold, "thin": a.thin, "ntrain": len(tr),
            "train_target_mean": float(y.mean()), "train_target_sd": float(y.std())}
    json.dump(info, open(f"{a.out}/info.json", "w"), indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
