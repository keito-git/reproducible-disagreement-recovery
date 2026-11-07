"""Votes for BRIGHTER German and splits for the second datasets.

BRIGHTER German: one row per (text, emotion) with one column per annotator (1 = marked, blank = not marked).
The label set includes 'neutral', so a rater who annotated the text marked at least one class; raters with no mark
on any row did not annotate the text. Each rater's vote is the mark vector divided by its sum (as for English).
Splits (seed 0): DICES-350 in 5 folds; BRIGHTER German 60/10/30.

  python s11_prep.py --raw DIR --votes DIR --split DIR
"""

import argparse
import json
import pickle

import numpy as np
import pandas as pd

CLS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


def main():
    ap = argparse.ArgumentParser()
    for k in ("raw", "votes", "split"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    d = pd.read_csv(f"{a.raw}/deu_individuals.csv")
    ann = [c for c in d.columns if "Annotator" in c]
    items = []
    for tid, g in d.groupby("text_id"):
        g = g.set_index("emotion").reindex(CLS)
        M = g[ann].notna().values.T.astype(float)          # (raters, classes)
        M = M[M.sum(1) > 0]
        items.append({"id": str(tid), "text": str(g["text"].dropna().iloc[0]), "V": M / M.sum(1, keepdims=True)})
    pickle.dump({"classes": CLS, "items": items}, open(f"{a.votes}/brighter_deu.pkl", "wb"))
    n = np.array([len(x["V"]) for x in items])
    print("brighter_deu items", len(items), dict(zip(*[v.tolist() for v in np.unique(n, return_counts=True)])))
    ids = np.array([x["id"] for x in items])
    p = np.random.default_rng(0).permutation(len(ids))
    nt, nd = int(round(0.6 * len(ids))), int(round(0.1 * len(ids)))
    sp = {"brighter_deu": {"train": ids[p[:nt]].tolist(), "dev": ids[p[nt:nt + nd]].tolist(),
                           "test": ids[p[nt + nd:]].tolist()}}
    di = np.array([x["id"] for x in pickle.load(open(f"{a.votes}/dices_350.pkl", "rb"))["items"]])
    fold = np.random.default_rng(0).permutation(len(di)) % 5
    sp["dices_350"] = {"folds": [di[fold == f].tolist() for f in range(5)]}
    json.dump(sp, open(f"{a.split}/splits_j.json", "w"))
    print({k: ({s: len(v) for s, v in x.items()} if "train" in x else [len(f) for f in x["folds"]]) for k, x in sp.items()})


if __name__ == "__main__":
    main()
