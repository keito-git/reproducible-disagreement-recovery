"""P4 manuscript Figure 2 (v2): three-way partition of Var(D_3) on the main datasets only.

Panel (b) of s14_fig.py is dropped because Tables 3 and 4 already report R_text with intervals.
Reads only saved analysis outputs; computes nothing new.

  python s18_fig_partition_v2.py --decomp FILE.json --out DIR
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "font.size": 9,
})

DECOMP_NAMES = ["Emotion, full", "Emotion, matched", "NLI, MNLI checkpoint", "NLI, matched",
                "Safety, full", "Safety, matched votes"]
COLORS = {"explained": "#3b6ea8", "unexplained": "#e3a64a", "noise": "#d9d9d9"}
LABELS = {"explained": "Text-predictable by the model", "unexplained": "Repeatable but not captured",
          "noise": "Non-repeatable at 3 annotators"}


def main(a):
    dec = json.load(open(a.decomp))
    assert len(dec) == len(DECOMP_NAMES)
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    y = list(range(len(dec)))[::-1]
    for yi, row in zip(y, dec):
        left = 0.0
        for key in ("explained", "unexplained", "noise"):
            ax.barh(yi, row[key], left=left, color=COLORS[key], edgecolor="black", linewidth=0.4, height=0.68)
            left += row[key]
    ax.set_yticks(y)
    ax.set_yticklabels(DECOMP_NAMES)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"Share of Var($D_3$)")
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[k], edgecolor="black", linewidth=0.4) for k in COLORS]
    # legend below the axis so that it cannot overlap any bar
    ax.legend(handles, [LABELS[k] for k in COLORS], loc="upper center", bbox_to_anchor=(0.42, -0.24),
              ncol=3, frameon=False, fontsize=8, handlelength=1.2, columnspacing=1.2)
    fig.tight_layout()
    os.makedirs(a.out, exist_ok=True)
    fig.savefig(f"{a.out}/fig_partition_v2.pdf", bbox_inches="tight")
    fig.savefig(f"{a.out}/fig_partition_v2.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp"); ap.add_argument("--out")
    main(ap.parse_args())
