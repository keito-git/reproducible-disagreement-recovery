"""Three-way partition of Var(D_3) per dataset (values plotted in Figure 2).

noise = 1 - r33, text-explained = m_E^2, stable-unexplained = r33 - m_E^2, where r33 = c(3,3)
(disjoint 3-person groups) and m_E = corr(mapped model prediction, D_3). Reads the s4/s5/s6 JSON outputs.

  python s7_decomp.py --snap DIR --out DIR
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROWS = [  # (label, json file, key)
    ("Emotion: BRIGHTER-eng (full)", "s4_map.json", "brighter_eng/main"),
    ("Emotion: BRIGHTER-eng (matched)", "s6_map.json", "brighter_eng/n1868_thin7"),
    ("NLI: ChaosNLI (MNLI init, full)", "s4_map.json", "chaosnli/main"),
    ("NLI: ChaosNLI (matched)", "s5_map.json", "chaosnli/rl_thin7"),
    ("Safety: DICES-990 (full)", "s4_map.json", "dices_990/main"),
    ("Safety: DICES-990 (7 votes)", "s5_map.json", "dices_990/thin7"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    J = {f: json.load(open(f"{a.snap}/{f}")) for f in {r[1] for r in ROWS}}
    res = []
    for lab, f, k in ROWS:
        d = J[f][k]
        r33 = d["raw"]["c"][2]
        m = d["E_i_main"]["m"]
        res.append({"label": lab, "r33": r33, "m_E": m, "noise": 1 - r33, "explained": m ** 2,
                    "unexplained": r33 - m ** 2, "explained_share_of_stable": m ** 2 / r33,
                    "nstar_E_i": d["E_i_main"]["nstar"]})
    json.dump(res, open(f"{a.out}/s7_decomp.json", "w"), indent=1)
    for r in res:
        print(f"{r['label']:34s} r33={r['r33']:.3f} m_E={r['m_E']:.3f} noise={r['noise']:.3f} "
              f"expl={r['explained']:.3f} unexpl={r['unexplained']:.3f} expl/stable={r['explained_share_of_stable']:.2f}")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    y = range(len(res))[::-1]
    parts = [("explained", "Text-predictable by current model", "#3b6ea8"), ("unexplained", "Repeatable but not captured", "#a9c4e2"),
             ("noise", "Non-repeatable at 3 raters", "#d9d9d9")]
    left = [0.0] * len(res)
    for key, name, col in parts:
        v = [r[key] for r in res]
        ax.barh(list(y), v, left=left, color=col, label=name, edgecolor="white", linewidth=0.5)
        left = [l + x for l, x in zip(left, v)]
    ax.set_yticks(list(y)); ax.set_yticklabels([r["label"] for r in res])
    ax.set_xlim(0, 1); ax.set_xlabel("Share of variance of per-item disagreement $D_3$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.25), ncol=3, frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{a.out}/fig_decomp.pdf"); fig.savefig(f"{a.out}/fig_decomp.png", dpi=200)


if __name__ == "__main__":
    main()
