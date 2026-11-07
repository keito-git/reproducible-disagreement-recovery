"""Per-item vote matrices (BRIGHTER English, ChaosNLI, DICES; GoEmotions is built but not used in the paper).

Each dataset becomes a list of (n_votes, K) arrays, one per item; a row is one annotator's
probability vector. Writes <out>/<name>.pkl with {"classes", "items": [{"id", "text", "V"}]}.

  python s1_votes.py --raw DIR --hf DIR --out DIR
"""

import argparse
import csv
import json
import os
import pickle
from collections import defaultdict

import numpy as np


def save(out, name, classes, items, info):
    pickle.dump({"classes": classes, "items": items}, open(f"{out}/{name}.pkl", "wb"))
    n = np.array([len(it["V"]) for it in items])
    info[name] = {"items": len(items), "K": len(classes),
                  "votes_per_item": {int(k): int(v) for k, v in zip(*np.unique(n, return_counts=True))}}


def goemotions(info, out):
    from datasets import load_dataset
    raw = load_dataset("google-research-datasets/go_emotions", "raw")["train"]
    skip = ("text", "id", "author", "subreddit", "link_id", "parent_id", "created_utc", "rater_id",
            "example_very_unclear")
    cols = [c for c in raw.features.keys() if c not in skip]
    acc, text = defaultdict(list), {}
    for row in raw:
        if row["example_very_unclear"]:
            continue
        chosen = [i for i, c in enumerate(cols) if row[c] == 1]
        if not chosen:
            continue
        v = np.zeros(len(cols)); v[chosen] = 1.0 / len(chosen)
        acc[row["id"]].append(v); text[row["id"]] = row["text"]
    save(out, "goemotions", cols, [{"id": k, "text": text[k], "V": np.array(v)} for k, v in acc.items()], info)


def brighter_eng(raw, info, out):
    emos = ["anger", "sadness", "fearanx", "happiness", "surprise", "social_warmth"]
    it, text = defaultdict(dict), {}
    rows = list(csv.reader(open(f"{raw}/eng.csv", encoding="utf-8")))
    for r in rows[1:]:
        it[r[0]][r[2]] = np.array([int(x) for x in r[3:]]); text[r[0]] = r[1]
    items = []
    for k, d in it.items():
        A = np.stack([d[e] for e in emos], 1) >= 1          # (raters, 6)
        V = np.zeros((len(A), 7))
        for j, a in enumerate(A):
            if a.any():
                V[j, :6] = a / a.sum()
            else:
                V[j, 6] = 1.0
        items.append({"id": k, "text": text[k], "V": V})
    save(out, "brighter_eng", emos + ["none"], items, info)


def chaosnli(raw, info, out):
    items = []
    for f in ("chaosNLI_snli", "chaosNLI_mnli_m", "chaosNLI_alphanli"):
        for line in open(f"{raw}/chaosnli/raw/{f}.jsonl", encoding="utf-8"):
            d = json.loads(line)
            c = np.array(d["label_count"], int)
            V = np.repeat(np.eye(len(c)), c, axis=0)       # exchangeable votes from the counts
            items.append({"id": d["uid"], "text": None, "V": V, "subset": f})
    # alphaNLI has 2 classes; pad to 3 so that one array shape serves all items
    for x in items:
        if x["V"].shape[1] == 2:
            x["V"] = np.c_[x["V"], np.zeros(len(x["V"]))]
    save(out, "chaosnli", ["c0", "c1", "c2"], items, info)


def dices(raw, info, out):
    import pandas as pd
    cls = ["Yes", "No", "Unsure"]
    for f in ("350", "990"):
        d = pd.read_csv(f"{raw}/dices_{f}.csv", low_memory=False)
        items = []
        for k, g in d.groupby("item_id"):
            y = g["Q_overall"].map({c: i for i, c in enumerate(cls)}).dropna().astype(int).values
            items.append({"id": int(k), "text": None, "V": np.eye(3)[y]})
        save(out, f"dices_{f}", cls, items, info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    info = {}
    goemotions(info, a.out)
    brighter_eng(a.raw, info, a.out)
    chaosnli(a.raw, info, a.out)
    dices(a.raw, info, a.out)
    json.dump(info, open(f"{a.out}/votes_info.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
