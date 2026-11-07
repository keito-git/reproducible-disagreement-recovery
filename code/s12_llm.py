"""Frozen LLM sentence embeddings (mean of last hidden states, eager attention, bf16).

Pairs are joined as 'Premise: ... / Hypothesis: ...' (ChaosNLI) or 'Context: ... / Response: ...' (DICES);
the start of the text is truncated when it exceeds 512 tokens. Saves {ids, X} per dataset.

  python s12_llm.py --votes DIR --raw DIR --split DIR --model PATH_OR_NAME --out DIR
"""

import argparse
import json
import os
import pickle

import numpy as np

from s4_judge import load_texts

FMT = {"chaosnli": "Premise: {}\nHypothesis: {}", "dices_990": "Context: {}\nResponse: {}"}


def main():
    ap = argparse.ArgumentParser()
    for k in ("votes", "raw", "split", "model", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    import torch
    from transformers import AutoModel, AutoTokenizer
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.truncation_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    sp = json.load(open(f"{a.split}/splits.json"))["brighter_eng"]
    sp2 = json.load(open(f"{a.split}/splits_e2.json"))
    sets = {"brighter_eng": sp["train"] + sp["dev"] + sp["test"],
            "chaosnli": sp2["chaosnli"]["train"] + sp2["chaosnli"]["dev"] + sp2["chaosnli"]["test"],
            "dices_990": [i for fl in sp2["dices_990"]["folds"] for i in fl]}
    for data, ids in sets.items():
        V = {x["id"]: x for x in pickle.load(open(f"{a.votes}/{data}.pkl", "rb"))["items"]}
        T = load_texts(a.raw, data)
        texts = [FMT[data].format(*T[i]) if T is not None else V[i]["text"] for i in ids]
        X = []
        with torch.no_grad():
            for b in range(0, len(texts), 16):
                e = tok(texts[b:b + 16], truncation=True, max_length=512, padding=True, return_tensors="pt").to("cuda")
                h = model(**e).last_hidden_state.float()
                m = e["attention_mask"].unsqueeze(-1).float()
                X.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
        X = np.concatenate(X)
        pickle.dump({"ids": ids, "X": X.astype(np.float32)}, open(f"{a.out}/{data}.pkl", "wb"))
        print(data, X.shape, flush=True)


if __name__ == "__main__":
    main()
