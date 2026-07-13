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
plot_tower_charts.py
====================
Generic publication-quality Kummer tower chart for any MQE hybrid or
tower-scaffold dataset.  Reads manifest.json and (optionally) step files
— no hardcoded mechanism data.

Two panels:

  Panel 1 — Step energy profile: computed step energies as background,
             with the Janus step(s) showing E_Janus(k) at selected
             Kummer levels k (algebraically derived from δ₀·p^{-(k-2)}).

  Panel 2 — E_Janus(k) convergence toward E_∞ (Kummer fixed point)
             from k=2 to k_conv + 3.

Supported manifest formats
--------------------------
  * hybrid_protocol  (datasets/hybrids/hybridtower/)
  * seed_protocol    (datasets/stoichiometry-zetazeros/iwasawatower/)

Usage
-----
    python plot_tower_charts.py <dataset_dir>
    python plot_tower_charts.py <dataset_dir> --format png --dpi 300
    python plot_tower_charts.py <dataset_dir> --out figs/
    python plot_tower_charts.py <dataset_dir> --k-levels 5

<dataset_dir> must contain manifest.json.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── style ─────────────────────────────────────────────────────────────────────

LABEL_FS = 9
TICK_FS  = 8
TITLE_FS = 10
ANNOT_FS = 7.5

# Colours for successive k levels (base → convergence)
_LEVEL_COLORS = [
    "#aaaaaa",  # k=2  (base / dashed)
    "#378ADD",
    "#1D9E75",
    "#BA7517",
    "#7F77DD",
    "#D85A30",  # convergence level
]

_JANUS_MKW = dict(marker="o", markersize=5, markeredgewidth=0.8,
                  markeredgecolor="white", zorder=4)


def _spine_style(ax: plt.Axes) -> None:
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color("#cccccc")
    ax.tick_params(axis="both", which="both", length=3, width=0.5,
                   labelsize=TICK_FS, colors="#444444")
    ax.set_facecolor("white")
    ax.grid(True, linewidth=0.35, color="#e8e8e8", linestyle="-")
    ax.set_axisbelow(True)


# ── manifest loading ───────────────────────────────────────────────────────────

def _load_manifest(dataset_dir: pathlib.Path) -> dict:
    p = dataset_dir / "manifest.json"
    if not p.exists():
        sys.exit(f"[plot] manifest.json not found in {dataset_dir}")
    with open(p) as fh:
        return json.load(fh)


def _algebraic(manifest: dict) -> dict:
    """Return step0_algebraic from whichever protocol key is present."""
    for key in ("hybrid_protocol", "seed_protocol"):
        proto = manifest.get(key) or {}
        alg   = proto.get("step0_algebraic") or {}
        if alg:
            return alg
    return {}


def _step1(manifest: dict) -> dict:
    """Return step1 seed block (handles both protocol key variants)."""
    for key in ("hybrid_protocol", "seed_protocol"):
        proto = manifest.get(key) or {}
        for s1key in ("step1_janus_pyscf", "step1_seed"):
            s1 = proto.get(s1key) or {}
            if s1:
                return s1
    return {}


def _resolve_step_file(
    dataset_dir: pathlib.Path,
    mechanism: str,
    step_n: int,
) -> Optional[pathlib.Path]:
    """Search for step_{n:02d}.json in dataset_dir, then sibling hybrid dir."""
    fname = f"step_{step_n:02d}.json"
    p = dataset_dir / fname
    if p.exists():
        return p
    sibling = dataset_dir.parent.parent.parent / mechanism / fname
    if sibling.exists():
        return sibling
    return None


# ── data extraction ────────────────────────────────────────────────────────────

def _choose_k_levels(k_conv: int, n_levels: int) -> list[int]:
    """Pick n_levels evenly-spaced Kummer levels from 2 to k_conv."""
    if k_conv <= 2:
        return [2]
    if k_conv - 2 < n_levels - 1:
        return list(range(2, k_conv + 1))
    ks = sorted({
        round(2 + i * (k_conv - 2) / (n_levels - 1))
        for i in range(n_levels)
    })
    return ks


def extract(manifest: dict, dataset_dir: pathlib.Path) -> dict:
    """
    Pull all chart data from the manifest and (optionally) step files.

    Returns a plain dict consumed by build_figure().
    """
    alg   = _algebraic(manifest)
    s1    = _step1(manifest)
    mech  = manifest["mechanism"]

    E_inf   = float(alg["E_inf_Ha"])
    E_init  = float(s1.get("E_seed_Ha", 0.0))
    delta0  = E_init - E_inf           # Kummer initial residual

    p_base  = int(manifest.get("tower_p", 2))
    k_conv  = int(manifest.get("tower_level_k", 18))
    M       = int(manifest.get("M_steps", 8))
    jsteps  = [int(j) for j in (manifest.get("janus_steps") or [])]
    n_orbs  = int(manifest.get("n_orbs_base", 4))
    sc      = manifest.get("scaffold_class") or manifest.get("spectral_class") or "?"
    m_mod   = int(manifest.get("m_modulus", 4))

    # Per-step energies: use fci_energies_Ha as the background series.
    # This keeps the Janus step at E_init (= e_janus(k=2)) by construction,
    # while non-Janus steps carry the algebraically computed Weyl PES values.
    fci_e = [float(v) for v in (manifest.get("fci_energies_Ha") or [])]
    if not fci_e:
        fci_e = [0.0] * M
    while len(fci_e) < M:
        fci_e.append(fci_e[-1] if fci_e else 0.0)
    fci_e = fci_e[:M]

    # δ_CI at each Janus step (from step files, best-effort)
    janus_dci: dict[int, float] = {}
    for js in jsteps:
        sp = _resolve_step_file(dataset_dir, mech, js)
        if sp is not None:
            with open(sp) as fh:
                sd = json.load(fh)
            dci = (sd.get("mqe_step") or {}).get("delta_CI_Ha")
            if dci is not None:
                janus_dci[js] = float(dci) * 1000.0   # Ha → mHa

    def e_janus(k: int) -> float:
        return E_inf + delta0 * (p_base ** -(k - 2))

    return {
        "mechanism":    mech,
        "E_inf":        E_inf,
        "E_init":       E_init,
        "delta0":       delta0,
        "p_base":       p_base,
        "k_conv":       k_conv,
        "M_steps":      M,
        "janus_steps":  jsteps,
        "janus_dci":    janus_dci,
        "fci_e":        fci_e,
        "n_orbs_base":  n_orbs,
        "spectral_class": sc,
        "m_modulus":    m_mod,
        "e_janus":      e_janus,
    }


# ── figure ────────────────────────────────────────────────────────────────────

def build_figure(td: dict, n_k_levels: int = 5) -> plt.Figure:
    mech    = td["mechanism"]
    E_inf   = td["E_inf"]
    delta0  = td["delta0"]
    p_base  = td["p_base"]
    k_conv  = td["k_conv"]
    M       = td["M_steps"]
    jsteps  = td["janus_steps"]
    jdci    = td["janus_dci"]
    fci_e   = td["fci_e"]
    n_orbs  = td["n_orbs_base"]
    sc      = td["spectral_class"]
    m_mod   = td["m_modulus"]
    ej      = td["e_janus"]

    steps  = list(range(M))
    k_show = _choose_k_levels(k_conv, n_k_levels)

    fig = plt.figure(figsize=(10, 8), facecolor="white")
    gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45,
                            top=0.93, bottom=0.08, left=0.09, right=0.97)

    # ── Panel 1: step energy profile ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    _spine_style(ax1)

    for idx, k in enumerate(k_show):
        color = _LEVEL_COLORS[min(idx, len(_LEVEL_COLORS) - 1)]
        ls    = "--" if k == 2 else "-"
        lw    = 1.0 if k == 2 else 1.6
        ek    = ej(k)

        # Background steps stay at fci_e; Janus steps dip to E_Janus(k)
        series = [ek if s in jsteps else fci_e[s] for s in steps]

        # n_orbs estimate per level (n_orbs_base × (k-1), conventional)
        n_est  = n_orbs * (k - 1) if k > 2 else n_orbs
        label  = f"k={k}  ({n_est} orbs)" + ("  [base]" if k == 2 else "")
        ax1.plot(steps, series, color=color, linewidth=lw, linestyle=ls,
                 label=label, zorder=2)

        for js in jsteps:
            ax1.plot(js, ek, color=color, **_JANUS_MKW)

    # Janus vertical guides + annotations
    for js in jsteps:
        ax1.axvline(js, color="#cccccc", linewidth=0.6, linestyle=":", zorder=0)
        dci_str = f"\nδ_CI={jdci[js]:.3f} mHa" if js in jdci else ""
        ax1.text(js, 0.02, f"n={js}\nJanus{dci_str}",
                 transform=ax1.get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=ANNOT_FS, color="#999999")

    ax1.set_xlabel("Step  n", fontsize=LABEL_FS, color="#444")
    ax1.set_ylabel("E_ref  (Ha)", fontsize=LABEL_FS, color="#444")
    ax1.set_title(
        f"QPE target energy per step — {mech.replace('_', ' ')}  "
        f"Kummer tower  k = {k_show[0]}–{k_show[-1]}",
        fontsize=TITLE_FS, fontweight="normal", color="#222", pad=8)
    ax1.set_xticks(steps)
    ax1.set_xticklabels([str(s) for s in steps], fontsize=TICK_FS)
    leg1 = ax1.legend(fontsize=ANNOT_FS, framealpha=0.9, edgecolor="#dddddd",
                      loc="best", handlelength=1.8, handletextpad=0.5,
                      labelspacing=0.35)
    leg1.get_frame().set_linewidth(0.5)

    # ── Panel 2: E_Janus(k) convergence ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    _spine_style(ax2)

    # Show k=2 through k_conv+3 to reveal where the curve is heading
    k_all  = list(range(2, k_conv + 4))
    ej_all = [ej(k) for k in k_all]

    ax2.axhline(E_inf, color="#A32D2D", linewidth=0.9, linestyle="--",
                alpha=0.7, label=f"E_∞ = {E_inf:.5f} Ha", zorder=1)

    ax2.plot(k_all, ej_all, color="#D85A30", linewidth=2.0, zorder=2,
             label="E_Janus(k)")
    ax2.scatter(k_all, ej_all, color="#D85A30", s=40, zorder=3,
                edgecolors="white", linewidths=1.2)

    # Annotate convergence point
    ej_conv     = ej(k_conv)
    delta_mHa   = abs(ej_conv - E_inf) * 1000.0
    ax2.annotate(
        f"k={k_conv}\nΔ={delta_mHa:.3f} mHa",
        xy=(k_conv, ej_conv),
        xytext=(10, 10), textcoords="offset points",
        fontsize=ANNOT_FS, color="#D85A30", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#D85A30", lw=0.8),
    )

    # ÷p annotations between the first few k pairs
    for i in range(1, min(5, len(k_all))):
        k0, e0 = k_all[i - 1], ej_all[i - 1]
        k1, e1 = k_all[i],     ej_all[i]
        span   = abs(e0 - e1)
        ax2.text((k0 + k1) / 2, (e0 + e1) / 2 + span * 0.15,
                 f"÷{p_base}", ha="center", va="bottom",
                 fontsize=ANNOT_FS - 0.5, color="#D85A30", alpha=0.7)

    ax2.set_xlabel("Tower level  k", fontsize=LABEL_FS, color="#444")
    ax2.set_ylabel("E_Janus(k)  (Ha)", fontsize=LABEL_FS, color="#444")
    ax2.set_title(
        f"E_Janus(k) convergence toward E_∞  "
        f"[Kummer ratio = 1/{p_base} exact]",
        fontsize=TITLE_FS, fontweight="normal", color="#222", pad=8)
    ax2.set_xticks(k_all)
    ax2.set_xticklabels(
        [f"k={k}" for k in k_all],
        fontsize=TICK_FS - 1,
        rotation=45 if len(k_all) > 10 else 0,
    )
    leg2 = ax2.legend(fontsize=ANNOT_FS, framealpha=0.9, edgecolor="#dddddd",
                      loc="upper right", handlelength=1.8, handletextpad=0.5)
    leg2.get_frame().set_linewidth(0.5)

    # Footer
    dci_str = (
        ", ".join(f"n={js}: {jdci[js]:.3f} mHa" for js in sorted(jdci))
        if jdci else "δ_CI unavailable"
    )
    fig.text(
        0.5, 0.01,
        f"{mech} · {sc} (m={m_mod}) · E_∞ = {E_inf:.5f} Ha · "
        f"Janus δ_CI: {dci_str} · k_conv = {k_conv}",
        ha="center", fontsize=6.5, color="#888888",
    )

    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset_dir",
        help="Directory containing manifest.json (hybrid or tower-scaffold)",
    )
    parser.add_argument(
        "--format", choices=["png", "pdf", "both"], default="both",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--out", default=None,
        help="Output directory (default: same as dataset_dir)",
    )
    parser.add_argument(
        "--k-levels", type=int, default=5,
        dest="k_levels",
        help="Number of Kummer levels to show in panel 1 (default: 5)",
    )
    args = parser.parse_args()

    ddir     = pathlib.Path(args.dataset_dir).resolve()
    manifest = _load_manifest(ddir)
    td       = extract(manifest, ddir)
    fig      = build_figure(td, n_k_levels=args.k_levels)

    mech    = td["mechanism"]
    out_dir = pathlib.Path(args.out).resolve() if args.out else ddir
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    if args.format in ("png", "both"):
        p = out_dir / f"{mech}_tower_charts.png"
        fig.savefig(p, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        saved.append(str(p))
    if args.format in ("pdf", "both"):
        p = out_dir / f"{mech}_tower_charts.pdf"
        fig.savefig(p, bbox_inches="tight", facecolor="white")
        saved.append(str(p))

    plt.close(fig)
    for s in saved:
        print(f"Saved → {s}")


if __name__ == "__main__":
    main()
