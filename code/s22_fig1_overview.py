"""Figure 1 of the article: overview of the five steps in the order of Sections 4.1-4.5.

Each schematic glyph carries a short gray note directly below it that states what the glyph shows.
Every symbol, formula, and term in the figure is written exactly as in Sections 3 and 4.

Panels tile one rectangle at the printed text width. Square thin frames, black headings,
and two semantic colors (human side blue, model side orange) plus grays. Glyph curves are
qualitative shapes with no numbers on them; symbols follow Sections 3 and 4 of the article.

  python s22_fig1_overview.py --out DIR
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "font.size": 7.5,
})

W, H = 6.48, 3.30           # inches; W is the text width of the article
TOP = 3.12                  # top edge of all panels; the strip above carries the c(n,k) link
INK = "#222222"
NOTE = "#555555"
FRAME = "#8a8a8a"
HUMAN = "#3b6ea8"
HUMAN_LIGHT = "#a9c4e2"
MODEL = "#c07a2c"
MODEL_LIGHT = "#e3a64a"
GRAYS = ["#3c3c3c", "#8f8f8f", "#d4d4d4"]
FS, FS_HEAD, FS_NOTE = 7.5, 8.0, 7.0
PAD_X = 0.10


def panel(ax, x, y, w, h, number, title, lines=1, emph=False):
    """Square frame, bold step number, regular title, thin rule; returns the y of the rule."""
    ax.add_patch(Rectangle((x, y), w, h, fc="#f4f4f4" if emph else "none",
                           ec="#2a2a2a" if emph else FRAME, lw=1.3 if emph else 0.6))
    bx, by = x + PAD_X, y + h - 0.055
    if number:
        ax.text(bx, by, number, ha="left", va="top", fontsize=FS_HEAD, color=INK, fontweight="bold")
        bx += 0.155
    ax.text(bx, by, title, ha="left", va="top", fontsize=FS_HEAD, color=INK, linespacing=1.1)
    ry = y + h - 0.07 - 0.135 * lines - 0.03
    ax.plot([x + 0.06, x + w - 0.06], [ry, ry], color="#bbbbbb", lw=0.5)
    return ry


def note(ax, x, y, text):
    """Gray explanatory note, centered at x, top at y."""
    ax.text(x, y, text, ha="center", va="top", fontsize=FS_NOTE, color=NOTE, linespacing=1.15)


def arrow(ax, p0, p1, color=INK):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=color,
                                lw=0.7, shrinkA=0, shrinkB=0))


def axes_glyph(ax, x0, y0, w, h, xlabel, ylabel):
    """L-shaped axes; y label above the y axis, x label below the right end."""
    ax.plot([x0, x0 + w], [y0, y0], color=INK, lw=0.6)
    ax.plot([x0, x0], [y0, y0 + h], color=INK, lw=0.6)
    ax.text(x0 + w, y0 - 0.03, xlabel, ha="right", va="top", fontsize=FS)
    ax.text(x0 - 0.02, y0 + h + 0.02, ylabel, ha="left", va="bottom", fontsize=FS)


def sat(u):
    """Qualitative saturating shape of the curve c(n,k)."""
    return 0.88 * (1 - np.exp(-2.6 * u))


def main(a):
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
    rng = np.random.default_rng(3)
    kk = np.linspace(0, 1, 80)

    y0 = 0.02
    ph = TOP - y0
    xi, wi = 0.02, 0.80            # input
    xm, wm = 1.02, 2.30            # step 1 above, steps 2 and 3 below
    x4, w4 = 3.74, 1.42            # step 4
    x5, w5 = 5.36, 1.10            # step 5
    hb = 1.45                      # height of the lower middle row
    y1 = y0 + hb + 0.18            # bottom of step 1
    w23 = (wm - 0.18) / 2
    x3 = xm + w23 + 0.18
    y_low = y0 + 0.77              # flow level of the lower row

    # input: items x annotators vote matrix
    ry = panel(ax, xi, y0, wi, ph, "", "Annotated\nitems", lines=2)
    s, nr, nc = 0.10, 12, 6
    gx = xi + 0.17
    gy = ry - 0.20 - nr * s
    probs = rng.dirichlet([1.2, 1.2, 1.2], size=nr)
    for r in range(nr):
        for c in range(nc):
            ax.add_patch(Rectangle((gx + c * s, gy + r * s), s * 0.86, s * 0.86,
                                   fc=GRAYS[rng.choice(3, p=probs[r])], ec="none"))
    ax.text(gx - 0.08, gy + nr * s / 2, r"items $x_i$", ha="center", va="center", fontsize=FS, rotation=90)
    ax.text(gx + nc * s / 2, gy - 0.05, "annotator\n" r"votes $\mathbf{v}_{ij}$", ha="center", va="top", fontsize=FS)
    note(ax, xi + wi / 2, gy - 0.36, "one row per item,\none cell per vote")
    for i, lab in enumerate(("class 1", "class 2", "class 3")):
        yy = y0 + 0.62 - 0.15 * i
        ax.add_patch(Rectangle((gx, yy - 0.04), 0.08, 0.08, fc=GRAYS[i], ec="none"))
        ax.text(gx + 0.12, yy, lab, ha="left", va="center", fontsize=FS)

    # step 1: human reproducibility curve
    ry = panel(ax, xm, y1, wm, TOP - y1, "(1)", "Reproducibility curve")
    rs, rx = 0.13, xm + 0.14
    rowy = ry - 0.36
    ax.text(rx + 4 * rs, rowy + rs + 0.03, "disjoint annotator groups", ha="center", va="bottom",
            fontsize=FS)
    for c in range(8):
        fc = HUMAN if c < 3 else (HUMAN_LIGHT if c < 6 else "#d4d4d4")
        ax.add_patch(Rectangle((rx + c * rs, rowy), rs * 0.86, rs * 0.86, fc=fc, ec="none"))
    for c0, lab in ((0, r"$A_t$: $n$"), (3, r"$B_{k,t}$: $k$")):
        xa, xb = rx + c0 * rs, rx + (c0 + 3) * rs - 0.02
        ax.plot([xa, xa, xb, xb], [rowy - 0.03, rowy - 0.06, rowy - 0.06, rowy - 0.03], color=HUMAN, lw=0.5)
        ax.text((xa + xb) / 2, rowy - 0.08, lab, ha="center", va="top", fontsize=FS, color=HUMAN)
    ax.text(rx + 4 * rs, rowy - 0.40, r"$D_{A_t}(x_i)$ vs. $D_{B_{k,t}}(x_i)$", ha="center", va="center",
            fontsize=FS)
    note(ax, rx + 4 * rs, rowy - 0.52, "disagreement of each group,\ncorrelated over items")
    arrow(ax, (rx + 8 * rs + 0.04, rowy + 0.05), (rx + 8 * rs + 0.24, rowy + 0.05))
    cx, cw = xm + 1.62, 0.56
    cy = y1 + 0.42
    ch = ry - 0.17 - cy
    axes_glyph(ax, cx, cy, cw, ch, r"$k$", r"$c(n,k)$")
    ax.plot(cx + kk * cw, cy + ch * sat(kk), color=HUMAN, lw=1.0)
    kn = 0.42
    ax.plot(cx + kn * cw, cy + ch * sat(kn), "o", ms=2.6, color=HUMAN)
    ax.plot([cx + kn * cw] * 2, [cy, cy + ch * sat(kn)], ls=":", color=HUMAN, lw=0.6)
    ax.text(cx + kn * cw + 0.05, cy + ch * sat(kn) - 0.05, r"$r_{nn}$", ha="left", va="top",
            fontsize=FS, color=HUMAN)
    ax.text(cx + kn * cw, cy - 0.03, r"$n$", ha="center", va="top", fontsize=FS)
    note(ax, cx + cw / 2 - 0.05, cy - 0.15, "agreement between\ngroups of $n$ and $k$")

    # step 2: judgment model
    ry = panel(ax, xm, y0, w23, hb, "(2)", "Judgment\nmodel", lines=2)
    tx, ty = xm + PAD_X, y_low - 0.15
    ax.add_patch(Rectangle((tx, ty), 0.30, 0.30, fc="white", ec=INK, lw=0.5))
    ax.text(tx + 0.15, ty + 0.15, r"$x$", ha="center", va="center", fontsize=FS + 0.5)
    ax.text(tx + 0.15, ty - 0.03, "text", ha="center", va="top", fontsize=FS)
    arrow(ax, (tx + 0.34, ty + 0.15), (tx + 0.50, ty + 0.15))
    bx = tx + 0.56
    for i, (hgt, col) in enumerate(zip((0.30, 0.18, 0.07), (MODEL, MODEL_LIGHT, "#f0d3a6"))):
        ax.add_patch(Rectangle((bx + i * 0.11, ty), 0.085, hgt, fc=col, ec="none"))
    ax.plot([bx - 0.03, bx + 0.33], [ty, ty], color=INK, lw=0.5)
    ax.text(bx + 0.15, ty - 0.03, r"$\hat{\mathbf{p}}(x)$", ha="center", va="top", fontsize=FS)
    note(ax, xm + w23 / 2, ty - 0.15, "predicted vote\ndistribution of $x$")
    ax.text(xm + w23 / 2, y0 + 0.11, r"target: mean vote $\bar{\mathbf{v}}(x_i)$", ha="center",
            va="center", fontsize=FS)

    # step 3: finite-group mapping
    ry = panel(ax, x3, y0, w23, hb, "(3)", "Finite-group\nmapping", lines=2)
    mx, mw = x3 + 0.17, w23 - 0.32
    my = y0 + 0.62
    mh = ry - 0.17 - my
    axes_glyph(ax, mx, my, mw, mh, r"$1-\mathrm{max}_c\,\hat{p}_c$", r"$t_n(x)$")
    u = rng.uniform(0.05, 0.95, 18)
    ax.scatter(mx + u * mw, my + mh * np.clip(0.85 * u ** 1.4 + rng.normal(0, 0.07, u.size), 0.02, 0.98),
               s=2.5, color=MODEL_LIGHT, lw=0)
    ax.plot(mx + kk * mw, my + mh * 0.85 * kk ** 1.4, color=MODEL, lw=1.0)
    ax.text(mx + mw + 0.01, my + mh * 0.85 + 0.01, r"$g$", ha="left", va="bottom", fontsize=FS,
            color=MODEL)
    note(ax, x3 + w23 / 2, my - 0.15, "curve: $g$ fitted on\ndevelopment items")
    ax.text(x3 + w23 / 2, y0 + 0.11, r"$E(x)=g(\hat{\mathbf{p}}(x))$", ha="center", va="center",
            fontsize=FS)

    # step 4: reliability normalization (core of the framework)
    ry = panel(ax, x4, y0, w4, ph, "(4)", "Reliability\nnormalization", lines=2, emph=True)
    sx, sw, sh = x4 + 0.22, 0.55, 0.55
    sy = ry - 0.20 - sh
    axes_glyph(ax, sx, sy, sw, sh, r"$E(x_i)$", r"$D_{A_t}(x_i)$")
    e = rng.uniform(0.05, 0.95, 26)
    ax.scatter(sx + e * sw, sy + sh * np.clip(0.15 + 0.6 * e + rng.normal(0, 0.15, e.size), 0.03, 0.97),
               s=2.5, color=INK, lw=0)
    ax.text(sx + sw + 0.07, sy + sh * 0.55, r"$\rightarrow\, m_E(n)$", ha="left", va="center", fontsize=FS)
    note(ax, x4 + w4 / 2, sy - 0.15, "one dot per item")
    # Eq. (7), broken after the sum so that it fits the panel
    ax.text(x4 + w4 / 2, sy - 0.33, r"$m_E(n)=\frac{1}{T}\Sigma_{t=1}^{T}$", ha="center", va="center", fontsize=FS)
    ax.text(x4 + w4 / 2, sy - 0.47, r"$\mathrm{corr}_i(E(x_i),D_{A_t}(x_i))$", ha="center", va="center",
            fontsize=FS)
    ax.text(x4 + w4 / 2, sy - 0.64, r"$R_{\mathrm{text}}(n)=\frac{m_E(n)^2}{r_{nn}}$", ha="center",
            va="center", fontsize=FS + 0.5)
    bx, bw, bh = x4 + 0.12, w4 - 0.24, 0.13
    by = sy - 1.02
    parts = [(0.14, HUMAN), (0.30, MODEL_LIGHT), (0.56, "#d4d4d4")]
    left = bx
    for frac, col in parts:
        ax.add_patch(Rectangle((left, by), bw * frac, bh, fc=col, ec=INK, lw=0.4))
        left += bw * frac
    xr = bx + bw * 0.44
    ax.plot([bx, bx, xr, xr], [by + bh + 0.03, by + bh + 0.06, by + bh + 0.06, by + bh + 0.03],
            color=INK, lw=0.5)
    ax.text((bx + xr) / 2, by + bh + 0.07, r"$r_{nn}$", ha="center", va="bottom", fontsize=FS)
    note(ax, x4 + w4 / 2, by - 0.09, r"shares of Var$(D_n)$")
    # legend: the three terms and labels of Eq. (8)
    labels = ((r"$m_E(n)^2$", "text-predictable by the model"),
              (r"$r_{nn}-m_E(n)^2$", "repeatable but not captured"),
              (r"$1-r_{nn}$", "non-repeatable at $n$ annotators"))
    for i, ((term, lab), (_, col)) in enumerate(zip(labels, parts)):
        yy = by - 0.28 - 0.235 * i
        ax.add_patch(Rectangle((bx, yy - 0.04), 0.08, 0.08, fc=col, ec=INK, lw=0.4))
        ax.text(bx + 0.12, yy, term, ha="left", va="center", fontsize=FS)
        ax.text(bx, yy - 0.12, lab, ha="left", va="center", fontsize=FS_NOTE)

    # step 5: annotator-equivalent count
    ry = panel(ax, x5, y0, w5, ph, "(5)", "Annotator-\nequivalent count", lines=2)
    nx, nw = x5 + 0.16, w5 - 0.30
    ny = y0 + 0.95
    nh = ry - 0.20 - ny
    axes_glyph(ax, nx, ny, nw, nh, r"$k$", r"$c(n,k)$")
    ax.plot(nx + kk * nw, ny + nh * sat(kk), color=HUMAN, lw=1.0)
    km = 0.55
    ym = ny + nh * sat(km)
    ax.plot([nx, nx + km * nw], [ym, ym], ls="--", color=INK, lw=0.6)
    ax.plot([nx + km * nw] * 2, [ny, ym], ls="--", color=INK, lw=0.6)
    ax.text(nx + 0.04, ym + 0.03, r"$m_E(n)$", ha="left", va="bottom", fontsize=FS)
    ax.text(nx + km * nw, ny - 0.04, r"$n^*(n)$", ha="center", va="top", fontsize=FS)
    ax.text(x5 + w5 / 2, ny - 0.30, r"$n^{*}(n)=k$ such that", ha="center", va="center", fontsize=FS)
    ax.text(x5 + w5 / 2, ny - 0.45, r"$c(n,k)=m_E(n)$", ha="center", va="center", fontsize=FS)
    note(ax, x5 + w5 / 2, ny - 0.60, "number of annotators\nthat agree equally well\nwith group $A_t$")

    # flows between panels
    arrow(ax, (xi + wi, y1 + 0.6), (xm, y1 + 0.6))
    arrow(ax, (xi + wi, y_low), (xm, y_low))
    arrow(ax, (xm + w23, y_low), (x3, y_low))
    ya = y1 + 0.60
    arrow(ax, (xm + wm, ya), (x4, ya))
    ax.text((xm + wm + x4) / 2, ya + 0.035, r"$r_{nn}$, $D_{A_t}$", ha="center", va="bottom", fontsize=FS)
    arrow(ax, (xm + wm, y_low), (x4, y_low))
    ax.text((xm + wm + x4) / 2, y_low + 0.03, r"$E(x)$", ha="center", va="bottom", fontsize=FS)
    arrow(ax, (x4 + w4, y0 + ph / 2), (x5, y0 + ph / 2))

    # the curve of step 1 is also read in step 5
    yl = TOP + 0.09
    xs, xe = xm + wm - 0.30, x5 + w5 / 2
    lx, gap = (x4 + w4 / 2 + xe) / 2, 0.22
    ax.plot([xs, xs, lx - gap / 2], [TOP, yl, yl], color=HUMAN, lw=0.7)
    ax.plot([lx + gap / 2, xe], [yl, yl], color=HUMAN, lw=0.7)
    arrow(ax, (xe, yl), (xe, TOP), color=HUMAN)
    ax.text(lx, yl, r"$c(n,k)$", ha="center", va="center", fontsize=FS, color=HUMAN,
            bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"))

    os.makedirs(a.out, exist_ok=True)
    fig.savefig(f"{a.out}/fig_overview_v7.pdf")
    fig.savefig(f"{a.out}/fig_overview_v7.png", dpi=300)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    main(ap.parse_args())
