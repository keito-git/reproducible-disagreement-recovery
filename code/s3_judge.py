"""Distribution models for BRIGHTER English: model and learning-rate selection on dev JSD, then three seeds.

Training recipe (eager attention, bf16 autocast, AdamW,
weight decay 0.01, batch 32, 3 epochs and at least 300 steps, 6% linear warmup), with the
model and learning rate as arguments and max length 128.

  python s3_judge.py split --votes DIR --out DIR
  python s3_judge.py train --votes DIR --split DIR --data {brighter_eng,goemotions} \
      --model NAME --lr LR --seed S [--frac F] --out DIR
"""

import argparse
import json
import math
import os
import pickle

import numpy as np

MAXLEN = 128


def cmd_split(a):
    os.makedirs(a.out, exist_ok=True)
    # BRIGHTER English: items with >= 8 votes -> test 40%, dev 10%, rest to train; < 8 votes -> train.
    it = pickle.load(open(f"{a.votes}/brighter_eng.pkl", "rb"))["items"]
    ids = np.array([x["id"] for x in it]); nv = np.array([len(x["V"]) for x in it])
    rng = np.random.default_rng(0)
    big = rng.permutation(np.flatnonzero(nv >= 8))
    nt, nd = int(round(0.4 * len(big))), int(round(0.1 * len(big)))
    sp = {"test": ids[big[:nt]].tolist(), "dev": ids[big[nt:nt + nd]].tolist()}
    sp["train"] = ids[np.setdiff1d(np.arange(len(ids)), big[:nt + nd])].tolist()
    # GoEmotions: the official simplified split ids.
    from datasets import load_dataset
    simp = load_dataset("google-research-datasets/go_emotions", "simplified")
    have = {x["id"] for x in pickle.load(open(f"{a.votes}/goemotions.pkl", "rb"))["items"]}
    sg = {s: [i for i in simp[s]["id"] if i in have] for s in ("train", "validation", "test")}
    json.dump({"brighter_eng": sp, "goemotions": {"train": sg["train"], "dev": sg["validation"], "test": sg["test"]}},
              open(f"{a.out}/splits.json", "w"))
    print({k: len(v) for k, v in sp.items()}, {k: len(v) for k, v in sg.items()})


def train(model_name, nc, texts, targets, seed, lr, epochs=3, min_steps=300, bs=32):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=nc, attn_implementation="eager").cuda()
    enc = tok(list(texts), truncation=True, max_length=MAXLEN, padding="max_length", return_tensors="pt")
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


def predict(model, tok, texts, bs=128):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            e = tok(list(texts[i:i + bs]), truncation=True, max_length=MAXLEN, padding="max_length",
                    return_tensors="pt")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(input_ids=e["input_ids"].cuda(), attention_mask=e["attention_mask"].cuda()).logits
            out.append(torch.softmax(lg.float(), -1).cpu().numpy())
    return np.concatenate(out)


def js(p, q):
    p, q = np.clip(p, 1e-12, 1), np.clip(q, 1e-12, 1)
    m = (p + q) / 2
    kl = lambda a, b: (a * np.log2(a / b)).sum(1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def cmd_train(a):
    os.makedirs(a.out, exist_ok=True)
    V = {x["id"]: x for x in pickle.load(open(f"{a.votes}/{a.data}.pkl", "rb"))["items"]}
    sp = json.load(open(f"{a.split}/splits.json"))[a.data]
    tr = list(sp["train"])
    if a.frac < 1:
        tr = [tr[i] for i in np.sort(np.random.default_rng(100).permutation(len(tr))[:int(a.frac * len(tr))])]
    nc = next(iter(V.values()))["V"].shape[1]
    model, tok = train(a.model, nc, [V[i]["text"] for i in tr], np.stack([V[i]["V"].mean(0) for i in tr]),
                       a.seed, a.lr)
    res = {}
    for s in ("dev", "test"):
        P = predict(model, tok, [V[i]["text"] for i in sp[s]])
        res[s] = {"ids": sp[s], "P": P}
    pickle.dump(res, open(f"{a.out}/probs.pkl", "wb"))
    pd = np.stack([V[i]["V"].mean(0) for i in sp["dev"]])
    info = {"model": a.model, "lr": a.lr, "seed": a.seed, "frac": a.frac, "n_train": len(tr),
            "dev_jsd": float(js(pd, res["dev"]["P"]).mean())}
    json.dump(info, open(f"{a.out}/info.json", "w"), indent=1)
    print(json.dumps(info))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["split", "train"])
    ap.add_argument("--votes", required=True)
    ap.add_argument("--split")
    ap.add_argument("--data")
    ap.add_argument("--model")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"split": cmd_split, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
