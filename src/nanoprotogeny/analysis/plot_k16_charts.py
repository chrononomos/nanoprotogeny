# Copyright 2026 Santos C. Borom
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
########################################################################
r"""
plot_k16_charts.py
==================
Export publication-quality PNG and PDF of the nitrogenase_closed_loop
k=16 tower dataset analysis (n_orbs=76, full CAS target).

Three panels:
  1. Step energy profile — k=16 vs base (k=2)
  2. Janus crossing gap δ_CI per step — all below 1.6 mHa QPE threshold
  3. Kummer convergence from k=16 toward QPE threshold at k=21

Usage
-----
    python plot_k16_charts.py                    # saves both PNG and PDF
    python plot_k16_charts.py --format png
    python plot_k16_charts.py --format pdf
    python plot_k16_charts.py --dpi 300
    python plot_k16_charts.py --out /some/dir
"""

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── constants ─────────────────────────────────────────────────────────────────

E_INF       = -57.35773765        # Kummer fixed point (Ha)
E_JANUS_K16 = -57.3242            # E_Janus at k=16
E_SEED      = 491.864             # non-Janus E_seed reference (full 76-orb system)

BASE_SEED = [
    120.539, 120.774, 120.995, 121.201, 121.393, 121.571,
    121.735, 121.886, 121.886, 121.735, 121.571, 121.393,
    121.201, 120.995, 120.774, 120.539,
]
BASE_JANUS_STEPS = {3: 2.1519, 11: 1.7801}   # step → δ_CI mHa (QPE-resolvable)

JANUS_STEPS_K16 = {
    1: 0.2014, 2: 0.1433, 3: 0.0241, 4: 0.3463, 5: 0.3496,
    9: 0.0097, 10: 0.3496, 11: 0.3463, 12: 0.0241, 13: 0.1433,
}
NON_JANUS_E = {
    0: 490.0314, 6: 501.6844, 7: 502.9428,
    8: 502.9428, 14: 491.8641, 15: 490.0314,
}
STEPS = list(range(16))
QPE_THRESHOLD = 1.6   # mHa

# ── style ─────────────────────────────────────────────────────────────────────

LABEL_FS = 9
TICK_FS  = 8
TITLE_FS = 10
ANNOT_FS = 7.5

def apply_spine_style(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#cccccc")
    ax.tick_params(axis="both", which="both", length=3, width=0.5,
                   labelsize=TICK_FS, colors="#444444")
    ax.set_facecolor("white")
    ax.grid(True, linewidth=0.35, color="#e8e8e8", linestyle="-")
    ax.set_axisbelow(True)


# ── figure ────────────────────────────────────────────────────────────────────

def build_figure():
    fig = plt.figure(figsize=(11, 11), facecolor="white")
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.48,
                            top=0.95, bottom=0.07, left=0.09, right=0.97)

    # ── panel 1: step energy profile ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    apply_spine_style(ax1)

    # base k=2
    ax1.plot(STEPS, BASE_SEED,
             color="#bbbbbb", linewidth=1.0, linestyle="--",
             label="base k=2  CAS(4,4)  (~121 Ha)", zorder=1)
    for js in BASE_JANUS_STEPS:
        ax1.plot(js, BASE_SEED[js], marker="D", markersize=5,
                 color="#888880", markeredgecolor="white", markeredgewidth=0.8, zorder=3)

    # k=16
    k16_y = [E_JANUS_K16 if i in JANUS_STEPS_K16 else NON_JANUS_E[i] for i in STEPS]
    ax1.plot(STEPS, k16_y,
             color="#378ADD", linewidth=2.0, linestyle="-",
             label="k=16  76 orbs  (10 Janus steps)", zorder=2)
    for js in JANUS_STEPS_K16:
        ax1.plot(js, E_JANUS_K16, marker="o", markersize=5,
                 color="#D85A30", markeredgecolor="white", markeredgewidth=0.8, zorder=4)

    for js in JANUS_STEPS_K16:
        ax1.axvline(js, color="#dddddd", linewidth=0.5, linestyle=":", zorder=0)

    ax1.set_xlabel("Step  n", fontsize=LABEL_FS, color="#444")
    ax1.set_ylabel("E_ref  (Ha)", fontsize=LABEL_FS, color="#444")
    ax1.set_title(
        "QPE target energy per step — k=16 (76 orbs) vs base k=2 (CAS 4,4)\n"
        "Orange dots = k=16 Janus steps (10)  |  diamond = base Janus steps (2)",
        fontsize=TITLE_FS, fontweight="normal", color="#222", pad=8)
    ax1.set_xticks(STEPS)
    ax1.set_xticklabels([str(n) for n in STEPS], fontsize=TICK_FS)
    legend1 = ax1.legend(fontsize=ANNOT_FS, framealpha=0.9, edgecolor="#dddddd",
                         loc="upper right", handlelength=1.8, labelspacing=0.4)
    legend1.get_frame().set_linewidth(0.5)

    # ── panel 2: δ_CI bar chart ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    apply_spine_style(ax2)

    k16_steps = sorted(JANUS_STEPS_K16.keys())
    k16_dci   = [JANUS_STEPS_K16[s] for s in k16_steps]
    x_base    = [len(k16_steps) + 0, len(k16_steps) + 1]
    base_dci  = [BASE_JANUS_STEPS[3], BASE_JANUS_STEPS[11]]

    x_k16 = np.arange(len(k16_steps))
    bars_k16 = ax2.bar(x_k16, k16_dci, color="#E24B4A", width=0.7,
                       label="k=16 crossings (unresolvable)")
    bars_base = ax2.bar([x_k16[-1]+2, x_k16[-1]+3], base_dci, color="#1D9E75",
                        width=0.7, label="base k=2 crossings (resolvable)")

    ax2.axhline(QPE_THRESHOLD, color="#BA7517", linewidth=1.2, linestyle="--",
                alpha=0.8, label=f"QPE threshold {QPE_THRESHOLD} mHa", zorder=0)

    tick_positions = list(x_k16) + [x_k16[-1]+2, x_k16[-1]+3]
    tick_labels    = [f"n={s}" for s in k16_steps] + ["n=3\n(base)", "n=11\n(base)"]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=TICK_FS)
    ax2.set_ylabel("δ_CI  (mHa)", fontsize=LABEL_FS, color="#444")
    ax2.set_title(
        "Janus crossing gap δ_CI — k=16 all unresolvable vs base k=2 both resolvable",
        fontsize=TITLE_FS, fontweight="normal", color="#222", pad=8)
    legend2 = ax2.legend(fontsize=ANNOT_FS, framealpha=0.9, edgecolor="#dddddd",
                         loc="upper right", handlelength=1.4, labelspacing=0.4)
    legend2.get_frame().set_linewidth(0.5)

    # annotate
    for bar, val in zip(bars_k16, k16_dci):
        if val > 0.05:
            ax2.text(bar.get_x()+bar.get_width()/2, val+0.02,
                     f"{val:.3f}", ha="center", va="bottom",
                     fontsize=ANNOT_FS-0.5, color="#C94040")
    for bar, val in zip(bars_base, base_dci):
        ax2.text(bar.get_x()+bar.get_width()/2, val+0.02,
                 f"{val:.3f}", ha="center", va="bottom",
                 fontsize=ANNOT_FS-0.5, color="#1D6B4E")

    # ── panel 3: Kummer convergence from k=16 ────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    apply_spine_style(ax3)

    conv_ks = list(range(16, 25))
    conv_e  = [E_INF + (E_SEED - E_INF) * 0.5**(k-2) for k in conv_ks]

    ax3.axhline(E_INF, color="#A32D2D", linewidth=0.9, linestyle="--",
                alpha=0.7, label=f"E_∞ = {E_INF:.3f} Ha", zorder=1)

    ax3.plot(conv_ks, conv_e, color="#7F77DD", linewidth=2.0, zorder=2,
             label="E_Janus(k) continuing from k=16")
    for k, e in zip(conv_ks, conv_e):
        delta_mha = (e - E_INF)*1000
        color = "#BA7517" if k == 21 else "#7F77DD"
        ms = 8 if k == 21 else 5
        ax3.scatter([k], [e], color=color, s=ms**2, zorder=3,
                    edgecolors="white", linewidths=1.0)
        dy = 0.006 if k < 20 else -0.004
        ax3.annotate(f"Δ={delta_mha:.2f}\nmHa",
                     xy=(k, e), xytext=(0, 14 if dy > 0 else -22),
                     textcoords="offset points",
                     ha="center", fontsize=ANNOT_FS-0.5, color="#666",
                     arrowprops=dict(arrowstyle="-", color="#dddddd", lw=0.4))

    # QPE threshold annotation at k=21
    e21 = conv_e[conv_ks.index(21)]
    ax3.annotate("QPE threshold\ncrossed here\n(1.048 mHa)",
                 xy=(21, e21), xytext=(21.5, e21 + 0.012),
                 fontsize=ANNOT_FS, color="#BA7517",
                 arrowprops=dict(arrowstyle="->", color="#BA7517", lw=0.9))

    ax3.set_xlabel("Tower level  k", fontsize=LABEL_FS, color="#444")
    ax3.set_ylabel("E_Janus(k)  (Ha)", fontsize=LABEL_FS, color="#444")
    ax3.set_title(
        "E_Janus(k) convergence toward E_∞  [from k=16 E_seed ≈ 491.9 Ha]",
        fontsize=TITLE_FS, fontweight="normal", color="#222", pad=8)
    ax3.set_xticks(conv_ks)
    ax3.set_xticklabels([f"k={k}" for k in conv_ks], fontsize=TICK_FS)
    legend3 = ax3.legend(fontsize=ANNOT_FS, framealpha=0.9, edgecolor="#dddddd",
                         loc="lower right", handlelength=1.8, labelspacing=0.4)
    legend3.get_frame().set_linewidth(0.5)

    # overall caption
    fig.text(
        0.5, 0.01,
        "nitrogenase_closed_loop · k=16 · n_orbs=76 · Group B (m=4) · "
        "E_∞=−57.358 Ha · 10 Janus steps, all δ_CI < 1.6 mHa · QPE threshold at k=21 · "
        "k_MQE=6.212×10¹² s⁻¹",
        ha="center", fontsize=6.5, color="#888888",
    )

    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--format", choices=["png", "pdf", "both"], default="both")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--out", default=None,
                        help="Output directory (default: same directory as this script)")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out) if args.out else pathlib.Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fig   = build_figure()
    saved = []

    if args.format in ("png", "both"):
        p = out_dir / "nitrogenase_closed_loop_k16_charts.png"
        fig.savefig(p, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        saved.append(str(p))

    if args.format in ("pdf", "both"):
        p = out_dir / "nitrogenase_closed_loop_k16_charts.pdf"
        fig.savefig(p, bbox_inches="tight", facecolor="white")
        saved.append(str(p))

    plt.close(fig)
    for s in saved:
        print(f"Saved → {s}")


if __name__ == "__main__":
    main()
