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
"""
plot_mqe_rates.py — MQE Rate Visualisation

Reads a JSON file produced by compute_mqe_rates.py (--output-json) and
generates a multi-panel figure for each mechanism found.

Usage
-----
    python plot_mqe_rates.py rates_310K.json
    python plot_mqe_rates.py rates_310K.json --format pdf --dpi 200
    python plot_mqe_rates.py rates_310K.json --out figs/

Output files
------------
    <mech>_mqe_rates.png  (and/or .pdf)

For a single mechanism: a 4-panel dashboard.
For multiple mechanisms: an additional summary comparison figure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np

# ── Reference enzyme rates (s⁻¹) for context ─────────────────────────────────
REFERENCE_RATES = [
    ("kBT/h (310 K)",          6.46e12, "#aaaaaa", "--"),
    ("Carbonic anhydrase",      1.0e6,   "#4db6ac", "-"),
    ("Nitrogenase (measured)",  8.0e0,   "#ef5350", "-"),
    ("ATP synthase",            1.0e2,   "#7986cb", "-"),
    ("Rubredoxin e⁻ transfer",  5.0e8,   "#66bb6a", "-"),
]

# ── Enzyme rate regime bands ──────────────────────────────────────────────────
REGIMES = {
    "barrierless":       ("#e8f5e9", "barrierless\n(≤ 0.5 mHa)"),
    "fast_enzymatic":    ("#e3f2fd", "fast enzymatic\n(0.5–10 mHa)"),
    "industrial_catalyst": ("#fff3e0", "industrial\n(10–60 mHa)"),
    "high_barrier":      ("#fce4ec", "high barrier\n(60–1000 mHa)"),
}


# ─── helpers ─────────────────────────────────────────────────────────────────

def _fmt_sci(v: float, sig: int = 3) -> str:
    if v == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(v))))
    mant = v / 10**exp
    return f"{mant:.{sig-1}f} × 10$^{{{exp}}}$"


def _barrier_str(b: dict) -> str:
    mHa = b.get("dE_barrier_mHa")
    if mHa is None:
        return "N/A"
    if abs(mHa) < 1e-3:
        return "≈ 0 mHa"
    return f"{mHa:.4f} mHa"


# ─── single-mechanism dashboard ───────────────────────────────────────────────

def plot_mechanism_dashboard(mech_data: dict, kBT_h: float, T_K: float, out_dir: str, fmt: list[str], dpi: int) -> None:
    name     = mech_data["mechanism"]
    rate     = mech_data.get("rate", {})
    barrier  = mech_data.get("barrier", {})
    lz       = mech_data.get("lz_diagnostics", {})
    topo     = mech_data.get("topology", {})
    lindblad = mech_data.get("lindblad", {})
    cas      = mech_data.get("cas", {})

    k_mqe    = rate.get("k_MQE_per_s")
    regime   = barrier.get("regime", "?")
    log10k   = rate.get("log10_k_MQE")
    half_life = rate.get("half_life_s")

    fig = plt.figure(figsize=(14, 10), facecolor="white")
    fig.suptitle(
        f"MQE Rate Dashboard — {name.replace('_', ' ')}   (T = {T_K:.1f} K)",
        fontsize=14, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35,
                           left=0.08, right=0.96, top=0.92, bottom=0.08)

    ax_rate  = fig.add_subplot(gs[0, :])   # top: rate on log scale
    ax_lz    = fig.add_subplot(gs[1, 0])   # bottom-left: LZ diagnostics
    ax_topo  = fig.add_subplot(gs[1, 1])   # bottom-right: topology / meta

    # ── Panel 1: Rate constant with reference lines ───────────────────────────
    ax = ax_rate
    ax.set_xlim(-1, 14)
    ax.set_xlabel("log₁₀(k / s⁻¹)", fontsize=10)
    ax.set_title("Rate Constant vs Enzymatic References", fontsize=11, pad=6)

    # Regime shading
    regime_spans = {
        "barrierless":           (12.0, 14.0),
        "fast_enzymatic":        (10.0, 12.0),
        "industrial_catalyst":   (7.0, 10.0),
        "high_barrier":          (3.0, 7.0),
    }
    colors_span = {
        "barrierless":           "#e8f5e9",
        "fast_enzymatic":        "#e3f2fd",
        "industrial_catalyst":   "#fff3e0",
        "high_barrier":          "#fce4ec",
    }
    for r, (lo, hi) in regime_spans.items():
        ax.axvspan(lo, hi, color=colors_span[r], alpha=0.5, zorder=0)
        ax.text((lo + hi) / 2, 0.92, r.replace("_", "\n"),
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7, color="#555555")

    # Reference lines
    for label, val, color, ls in REFERENCE_RATES:
        lv = math.log10(val)
        ax.axvline(lv, color=color, linestyle=ls, linewidth=1.2, alpha=0.8)
        ax.text(lv + 0.07, 0.6, label, transform=ax.get_xaxis_transform(),
                fontsize=7.5, color=color, rotation=90, va="center")

    # k_MQE marker
    if k_mqe and k_mqe > 0:
        ax.axvline(log10k, color="#d32f2f", linewidth=2.5, zorder=5)
        ax.text(log10k + 0.1, 0.18,
                f"k_MQE = {_fmt_sci(k_mqe)}\n(log₁₀ = {log10k:.2f})",
                transform=ax.get_xaxis_transform(),
                fontsize=9, color="#d32f2f", fontweight="bold")

    ax.set_yticks([])
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlim(-1, 14)

    # ── Panel 2: LZ diagnostics table ────────────────────────────────────────
    ax = ax_lz
    ax.axis("off")
    ax.set_title("Landau–Zener Diagnostics", fontsize=11, pad=4)

    H_AB   = lz.get("H_AB_Ha")
    dCI    = lz.get("delta_CI_Ha")
    Gamma  = lz.get("Gamma_classical")
    w_cl   = lz.get("w_LZ_classical")
    mu_amu = lz.get("reduced_mass_amu")
    v_nuc  = lz.get("v_nuc_bohr_per_au")

    rows = [
        ("H_AB", f"{H_AB:.3e} Ha" if H_AB is not None else "N/A"),
        ("H₁ₑ", f"{lz.get('H12_1e_Ha', 0):.3e} Ha"),
        ("H₂ₑ", f"{lz.get('H12_2e_Ha', 0):.3e} Ha"),
        ("δ_CI", f"{dCI*1000:.4f} mHa" if dCI else "N/A"),
        ("Γ_cl", f"{Gamma:.3f}" if Gamma else "N/A"),
        ("w_LZ_cl", f"{w_cl:.4f}" if w_cl else "N/A"),
        ("μ_red", f"{mu_amu:.2f} amu" if mu_amu else "N/A"),
        ("v_nuc", f"{v_nuc:.3e} bohr/au" if v_nuc else "N/A"),
    ]
    for i, (lbl, val) in enumerate(rows):
        y = 0.95 - i * 0.11
        ax.text(0.02, y, lbl, transform=ax.transAxes,
                fontsize=9, fontweight="bold", color="#333333")
        ax.text(0.42, y, val, transform=ax.transAxes,
                fontsize=9, color="#111111")
    ax.text(0.02, 0.95 - len(rows) * 0.11,
            "Note: For Case III (4|m), Berry\nphase guarantees w_LZ = 1\nindependently of Γ_cl.",
            transform=ax.transAxes, fontsize=7.5, color="#666666", style="italic")

    # ── Panel 3: Topology + barrier + meta ───────────────────────────────────
    ax = ax_topo
    ax.axis("off")
    ax.set_title("Topology & Rate Summary", fontsize=11, pad=4)

    half_str = (f"{half_life:.3e} s" if half_life else "N/A")
    rows2 = [
        ("Mechanism",     name.replace("_", " ")),
        ("Tower level",   mech_data.get("tower_level", "?")),
        ("CAS",           f"({cas.get('nelec_active','?')},{cas.get('ncas','?')})"),
        ("m modulus",     str(mech_data.get("m_modulus", "?"))),
        ("Winding",       topo.get("winding_condition", "?")),
        ("Case",          topo.get("case", "?")),
        ("w_LZ",          f"{topo.get('w_LZ', '?')} ({topo.get('w_LZ_source','').split('(')[0].strip()})"),
        ("n*",            str(lindblad.get("n_star", "?"))),
        ("p(k*)",         str(lindblad.get("p_k_star", "?"))),
        ("ΔE‡",           _barrier_str(barrier)),
        ("Regime",        regime),
        ("k_MQE",         _fmt_sci(k_mqe) + " s⁻¹" if k_mqe else "N/A"),
        ("Half-life",     half_str),
        ("Spectral class", mech_data.get("spectral_class", "?")),
    ]
    for i, (lbl, val) in enumerate(rows2):
        y = 0.97 - i * 0.067
        ax.text(0.01, y, lbl + ":", transform=ax.transAxes,
                fontsize=8.5, fontweight="bold", color="#333333")
        ax.text(0.40, y, val, transform=ax.transAxes,
                fontsize=8.5, color="#111111")

    # Colour-coded regime badge
    badge_color = {"barrierless": "#388e3c", "fast_enzymatic": "#1565c0",
                   "industrial_catalyst": "#e65100", "high_barrier": "#b71c1c",
                   "pathological_scaffold": "#4a148c"}.get(regime, "#666666")
    rect = mpatches.FancyBboxPatch(
        (0.01, -0.02), 0.98, 0.06,
        boxstyle="round,pad=0.01",
        transform=ax.transAxes, clip_on=False,
        facecolor=badge_color, alpha=0.12, edgecolor=badge_color,
    )
    ax.add_patch(rect)
    ax.text(0.50, 0.01, f"Regime: {regime}", transform=ax.transAxes,
            ha="center", fontsize=9, fontweight="bold", color=badge_color)

    for ext in fmt:
        path = os.path.join(out_dir, f"{name}_mqe_rates.{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ─── multi-mechanism comparison ───────────────────────────────────────────────

def plot_comparison(mechs: list[dict], kBT_h: float, T_K: float, out_dir: str, fmt: list[str], dpi: int) -> None:
    if len(mechs) < 2:
        return  # nothing to compare

    ok = [m for m in mechs if m.get("ok") and m["rate"].get("k_MQE_per_s")]
    if not ok:
        return

    names     = [m["mechanism"].replace("_", "\n") for m in ok]
    log10ks   = [m["rate"]["log10_k_MQE"] for m in ok]
    barriers  = [m["barrier"].get("dE_barrier_mHa") or 0.0 for m in ok]
    regimes   = [m["barrier"].get("regime", "?") for m in ok]

    bar_colors = [
        {"barrierless": "#43a047", "fast_enzymatic": "#1e88e5",
         "industrial_catalyst": "#fb8c00", "high_barrier": "#e53935",
         "pathological_scaffold": "#8e24aa"}.get(r, "#78909c")
        for r in regimes
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
    fig.suptitle(f"MQE Rate Comparison — {T_K:.1f} K", fontsize=13, fontweight="bold")

    # Left: log10(k_MQE)
    ax = axes[0]
    x  = np.arange(len(ok))
    bars = ax.bar(x, log10ks, color=bar_colors, edgecolor="white", linewidth=0.8)
    ax.axhline(math.log10(kBT_h), color="#aaa", linestyle="--", linewidth=1, label="kBT/h")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("log₁₀(k_MQE / s⁻¹)", fontsize=10)
    ax.set_title("Rate Constants", fontsize=11)
    ax.legend(fontsize=8)
    for bar, lk in zip(bars, log10ks):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{lk:.1f}", ha="center", va="bottom", fontsize=8)

    # Right: ΔE‡
    ax = axes[1]
    bars2 = ax.bar(x, barriers, color=bar_colors, edgecolor="white", linewidth=0.8)
    ax.axhline(0.5, color="#aaa", linestyle="--", linewidth=1, label="QPE threshold (0.5 mHa)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("ΔE‡ / mHa", fontsize=10)
    ax.set_title("Energy Barrier", fontsize=11)
    ax.legend(fontsize=8)
    for bar, dE in zip(bars2, barriers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{dE:.3g}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    for ext in fmt:
        path = os.path.join(out_dir, f"mqe_rates_comparison.{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise MQE rate JSON from compute_mqe_rates.py")
    parser.add_argument("json_file", help="Path to rates JSON (--output-json from compute_mqe_rates.py)")
    parser.add_argument("--format", choices=["png", "pdf", "both"], default="png")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory (default: same as json_file)")
    args = parser.parse_args()

    with open(args.json_file) as f:
        data = json.load(f)

    T_K    = data.get("temperature_K", 298.15)
    kBT_h  = data.get("kBT_h_per_s", 6.25e12)
    mechs  = data.get("mechanisms", [])

    if not mechs:
        sys.exit("No mechanisms in JSON.")

    fmt      = ["png", "pdf"] if args.format == "both" else [args.format]
    out_dir  = args.out or os.path.dirname(os.path.abspath(args.json_file))
    os.makedirs(out_dir, exist_ok=True)

    for mech_data in mechs:
        if not mech_data.get("ok"):
            print(f"  Skipping {mech_data['mechanism']} (failed)")
            continue
        print(f"Plotting: {mech_data['mechanism']}")
        plot_mechanism_dashboard(mech_data, kBT_h, T_K, out_dir, fmt, args.dpi)

    plot_comparison(mechs, kBT_h, T_K, out_dir, fmt, args.dpi)
    print("Done.")


if __name__ == "__main__":
    main()
