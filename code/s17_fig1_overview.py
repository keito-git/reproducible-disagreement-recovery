"""Figure 1 of the article: overview of the five steps in the order of Sections 4.1-4.5.

Boxes, arrows, and small schematic glyphs only. Glyph curves are qualitative shapes with no
numbers on them; symbols follow Sections 3 and 4 of the article.

  python s17_fig1_overview.py --out DIR
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "font.size": 8,
})

GREY = ("#f2f2f2", "#555555")
HUMAN = ("#e8f0f8", "#3b6ea8")
MODEL = ("#fbf0e2", "#c07a2c")
CORE = ("#e9f3e6", "#3d7a33")
CLASS_COLORS = ["#3b6ea8", "#e3a64a", "#8fbf7f"]
INK = "#333333"


def module(ax, x, y, w, h, title, style, lw=0.9):
    fc, ec = style
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=8, fontweight="bold",
            color=INK, linespacing=1.15)


def arrow(ax, pts, label=None, lpos=None, color=INK):
    """Orthogonal polyline arrow through pts; label is placed at lpos."""
    xs, ys = zip(*pts)
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=0.9)
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9, shrinkA=0, shrinkB=0))
    if label:
        ax.text(*lpos, label, ha="center", va="bottom", fontsize=7.5, color=color)


def mini_axes(ax, x0, y0, w, h, xlabel, ylabel):
    """L-shaped axes; the y label sits above the y axis and the x label below the right end."""
    ax.plot([x0, x0 + w], [y0, y0], color=INK, lw=0.7)
    ax.plot([x0, x0], [y0, y0 + h], color=INK, lw=0.7)
    ax.text(x0 + w, y0 - 0.1, xlabel, ha="right", va="top", fontsize=7.5)
    ax.text(x0, y0 + h + 0.08, ylabel, ha="left", va="bottom", fontsize=7.5)


def sat(u):
    return 0.85 * (1 - np.exp(-2.6 * u))


def main(a):
    fig, ax = plt.subplots(figsize=(7.2, 3.96))
    ax.set_xlim(0, 20); ax.set_ylim(0, 11); ax.set_aspect("equal"); ax.axis("off")
    rng = np.random.default_rng(0)
    kk = np.linspace(0, 1, 60)

    # input: items x annotators vote matrix
    module(ax, 0.2, 2.6, 2.6, 5.8, "Annotated\nitems", GREY)
    gx, gy, s = 0.8, 4.0, 0.3
    for r in range(6):
        for c in range(6):
            ax.add_patch(Rectangle((gx + c * s, gy + r * s), s * 0.88, s * 0.88,
                                   fc=CLASS_COLORS[rng.integers(3)], ec="none"))
    ax.text(gx + 3 * s, gy + 6 * s + 0.1, "annotators", ha="center", va="bottom", fontsize=7.5)
    ax.text(0.52, gy + 3 * s, "items", ha="center", va="center", fontsize=7.5, rotation=90)
    ax.text(1.5, 3.3, r"votes $\mathbf{v}_{ij}$", ha="center", va="center", fontsize=8)

    # step 1: human reproducibility curve
    module(ax, 3.6, 6.6, 6.2, 4.2, "Step 1: Reproducibility curve", HUMAN)
    rx, ry, rs = 3.95, 8.6, 0.36
    for c in range(8):
        col = HUMAN[1] if c < 3 else (MODEL[1] if c < 6 else "#bbbbbb")
        ax.add_patch(Rectangle((rx + c * rs, ry), rs * 0.85, rs * 0.85, fc=col, ec="none"))
    ax.text(rx + 4 * rs, ry + rs + 0.1, "disjoint groups", ha="center", va="bottom", fontsize=7.5)
    ax.text(rx + 1.45 * rs, ry - 0.1, r"$A_t$: $n$", ha="center", va="top", fontsize=7.5, color=HUMAN[1])
    ax.text(rx + 4.45 * rs, ry - 0.1, r"$B_{k,t}$: $k$", ha="center", va="top", fontsize=7.5, color=MODEL[1])
    ax.text(rx + 4 * rs, 7.2, r"$D_{A_t}(x)$, $D_{B_{k,t}}(x)$", ha="center", va="center", fontsize=8)
    arrow(ax, [(6.95, 8.75), (7.45, 8.75)])
    cx, cy, cw, ch = 7.9, 7.2, 1.6, 1.9
    mini_axes(ax, cx, cy, cw, ch, r"$k$", r"$c(n,k)$")
    ax.plot(cx + kk * cw, cy + ch * sat(kk), color=HUMAN[1], lw=1.1)
    kn = 0.4
    ax.plot(cx + kn * cw, cy + ch * sat(kn), "o", ms=3, color=HUMAN[1])
    ax.text(cx + kn * cw + 0.1, cy + ch * sat(kn) - 0.1, r"$r_{nn}$", ha="left", va="top", fontsize=7.5,
            color=HUMAN[1])

    # step 2: judgment model
    module(ax, 3.6, 0.3, 2.9, 5.2, "Step 2:\nJudgment\nmodel", MODEL)
    ax.add_patch(Rectangle((3.95, 2.35), 0.8, 1.0, fc="white", ec=INK, lw=0.6))
    for i in range(3):
        ax.plot([4.07, 4.63 - 0.15 * (i == 2)], [3.12 - 0.27 * i] * 2, color="#999999", lw=0.6)
    ax.text(4.35, 2.2, r"text $x$", ha="center", va="top", fontsize=7.5)
    arrow(ax, [(4.85, 2.85), (5.2, 2.85)])
    for i, (hgt, col) in enumerate(zip((0.9, 0.55, 0.22), CLASS_COLORS)):
        ax.add_patch(Rectangle((5.35 + i * 0.3, 2.35), 0.23, hgt, fc=col, ec="none"))
    ax.plot([5.28, 6.25], [2.35, 2.35], color=INK, lw=0.6)
    ax.text(5.77, 2.2, r"$\hat{\mathbf{p}}(x)$", ha="center", va="top", fontsize=7.5)
    ax.text(5.05, 1.05, "trained on\nvote distributions", ha="center", va="center", fontsize=7.5)

    # step 3: finite-group mapping
    module(ax, 6.9, 0.3, 2.9, 5.2, "Step 3:\nFinite-group\nmapping", MODEL)
    mx, my, mw, mh = 7.45, 2.0, 1.95, 1.3
    mini_axes(ax, mx, my, mw, mh, r"$1-\mathrm{max}_c\,\hat{p}_c$", r"$t_n(x)$")
    ax.plot(mx + kk * mw, my + mh * 0.9 * kk ** 1.6, color=MODEL[1], lw=1.1)
    ax.text(8.35, 1.0, r"$E(x)=g(\hat{\mathbf{p}}(x))$", ha="center", va="center", fontsize=8)

    # step 4: reliability normalization (core of the framework)
    module(ax, 11.4, 2.6, 4.35, 5.8, "Step 4: Reliability\nnormalization", CORE, lw=2.0)
    ax.text(13.575, 6.55, r"$m_E(n)=\mathrm{corr}(E, D_{A_t})$", ha="center", va="center", fontsize=8)
    ax.text(13.575, 5.8, r"$R_{\mathrm{text}}(n)=m_E(n)^2/r_{nn}$", ha="center", va="center", fontsize=8.5)
    bx, by, bw, bh = 11.85, 4.75, 3.45, 0.4
    parts = [(0.12, "#3b6ea8"), (0.28, "#e3a64a"), (0.60, "#d9d9d9")]
    left = bx
    for frac, col in parts:
        ax.add_patch(Rectangle((left, by), bw * frac, bh, fc=col, ec="black", lw=0.4))
        left += bw * frac
    labels = (r"$m_E^2$: text-predictable", r"$r_{nn}-m_E^2$: not captured", r"$1-r_{nn}$: non-repeatable")
    for i, (lab, (_, col)) in enumerate(zip(labels, parts)):
        yy = 4.15 - 0.45 * i
        ax.add_patch(Rectangle((11.75, yy - 0.13), 0.26, 0.26, fc=col, ec="black", lw=0.4))
        ax.text(12.12, yy, lab, ha="left", va="center", fontsize=7.2)

    # step 5: annotator-equivalent count
    module(ax, 16.45, 2.6, 3.4, 5.8, "Step 5: Annotator-\nequivalent count", HUMAN)
    nx, ny, nw, nh = 17.0, 3.55, 2.45, 2.55
    mini_axes(ax, nx, ny, nw, nh, r"$k$", r"$c(n,k)$")
    ax.plot(nx + kk * nw, ny + nh * sat(kk), color=HUMAN[1], lw=1.1)
    km = 0.55
    ym = ny + nh * sat(km)
    ax.plot([nx, nx + km * nw], [ym, ym], ls="--", color=CORE[1], lw=0.9)
    ax.plot([nx + km * nw] * 2, [ny, ym], ls="--", color=CORE[1], lw=0.9)
    ax.text(nx + 0.08, ym + 0.08, r"$m_E(n)$", ha="left", va="bottom", fontsize=7.5, color=CORE[1])
    ax.text(nx + km * nw, ny - 0.1, r"$n^*(n)$", ha="center", va="top", fontsize=7.5, color=CORE[1])

    # flows between modules
    arrow(ax, [(2.8, 7.6), (3.6, 7.6)])
    arrow(ax, [(2.8, 3.4), (3.6, 3.4)])
    arrow(ax, [(6.5, 2.85), (6.9, 2.85)])
    arrow(ax, [(9.8, 7.6), (11.4, 7.6)], r"$r_{nn}$, $D_{A_t}$", (10.6, 7.68))
    arrow(ax, [(9.8, 3.4), (11.4, 3.4)], r"$E(x)$", (10.6, 3.48))
    arrow(ax, [(15.75, 5.5), (16.45, 5.5)])
    arrow(ax, [(9.8, 10.0), (18.15, 10.0), (18.15, 8.4)], r"$c(n,k)$", (14.0, 10.08), color=HUMAN[1])

    os.makedirs(a.out, exist_ok=True)
    fig.savefig(f"{a.out}/fig_overview.pdf", bbox_inches="tight")
    fig.savefig(f"{a.out}/fig_overview.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out")
    main(ap.parse_args())
