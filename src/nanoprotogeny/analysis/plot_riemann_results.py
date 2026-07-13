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
plot_riemann_results.py
=======================
Export publication-quality PNG and PDF from any ``mqe run --riemann`` JSON
output file.  Works for any mechanism in the standard mqe_riemann_validation
format — nitrogenase, PSII, hydrogenase, or any future catalog entry.

Three panels per mechanism:
  1. Spectral identification residuals — δ at each Janus crossing vs QPE threshold
  2. Riemann zero fingerprint — E_Janus(γ_k) for 20 zeros, matched zero highlighted
  3. Non-Janus step profile — E_ref variation along the catalytic path

Usage
-----
    python plot_riemann_results.py results.json
    python plot_riemann_results.py results.json --format png --dpi 300
    python plot_riemann_results.py results.json --mechanism psii
    python plot_riemann_results.py results.json --out /some/dir
    python plot_riemann_results.py results.json --n-zeros 30
"""

import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Riemann zeta non-trivial zeros (first 30, imaginary parts) ───────────────
# Standard mathematical constants — OEIS A002410 / LMFDB.
RIEMANN_ZEROS = [
    14.134725141734693, 21.022039638771555, 25.010857580145688,
    30.424876125859513, 32.935061587739190, 37.586178158825671,
    40.918719012147495, 43.327073280914999, 48.005150881167160,
    49.773832477672301, 52.970321477714461, 56.446247697063246,
    59.347044002602353, 60.831778524609995, 65.112544048081652,
    67.079810529494174, 69.546401711173957, 72.067157674481895,
    75.704690699083933, 77.144840068874805, 79.337375020249367,
    82.910380854086030, 84.735492981329459, 87.425274613125229,
    88.809111208594021, 92.491899270660498, 94.651344040519886,
    95.870634228245332, 98.831194218193692, 101.317851006956279,
]

QPE_THRESHOLD_MHA = 1.6   # mHa — spectral identification pass criterion

# ── style ─────────────────────────────────────────────────────────────────────

LABEL_FS = 9
TICK_FS  = 8
TITLE_FS = 10
ANNOT_FS = 7.5

C_PASS   = "#1D9E75"
C_FAIL   = "#E24B4A"
C_PURPLE = "#7F77DD"
C_BLUE   = "#378ADD"
C_ORANGE = "#D85A30"
C_AMBER  = "#BA7517"
C_GREY   = "#888780"


def apply_spine_style(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#cccccc")
    ax.tick_params(axis="both", which="both", length=3, width=0.5,
                   labelsize=TICK_FS, colors="#444444")
    ax.set_facecolor("white")
    ax.grid(True, linewidth=0.35, color="#e8e8e8", linestyle="-")
    ax.set_axisbelow(True)


# ── data helpers ──────────────────────────────────────────────────────────────

def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def list_mechanisms(data: dict) -> list[str]:
    return list(data.get("mqe_riemann_validation", {}).keys())


def get_mechanism(data: dict, name: str | None = None) -> tuple[str, dict]:
    mechs = data.get("mqe_riemann_validation", {})
    if not mechs:
        raise ValueError("No 'mqe_riemann_validation' key in JSON.")
    if name is None:
        name = next(iter(mechs))
    if name not in mechs:
        raise KeyError(f"{name!r} not found. Available: {list(mechs)}")
    return name, mechs[name]


def classify_steps(qpe_results: dict) -> tuple[list[dict], list[dict]]:
    """Split qpe_results into Janus (riemann_exact) and tower_direct entries."""
    janus, tower = [], []
    for k, v in qpe_results.items():
        entry = {"step": int(k), **v}
        if v.get("method") == "riemann_exact":
            janus.append(entry)
        else:
            tower.append(entry)
    janus.sort(key=lambda x: x["step"])
    tower.sort(key=lambda x: x["step"])
    return janus, tower


def infer_nstar_dt(janus_entry: dict) -> float:
    """Back-compute n*·Δt from a Janus result: E_riemann = −s·γ / (n*·Δt)."""
    s   = janus_entry["s_value"]
    g   = janus_entry["gamma_k"]
    e_r = janus_entry["E_riemann"]
    return -s * g / e_r


def ejanus_spectrum(s: float, nstar_dt: float, n: int = 20) -> np.ndarray:
    return np.array([-s * g / nstar_dt for g in RIEMANN_ZEROS[:n]])


# ── figure ────────────────────────────────────────────────────────────────────

def build_figure(
    mech_name: str,
    mech:      dict,
    n_zeros:   int = 20,
) -> plt.Figure:

    qpe    = mech.get("qpe_results", {})
    step_e = mech.get("step_reference_energies", [])
    M      = mech.get("M_steps", len(step_e))
    m      = mech.get("m", "?")
    N      = mech.get("N_orbitals", "?")
    grp    = mech.get("scaffold_class", "")
    s_val  = mech.get("s_value", 0.0)
    ph_ok  = mech.get("phase_closure_ok", False)
    ch_ok  = mech.get("chemical_accuracy_ok", False)
    sec    = mech.get("elapsed_s", 0.0)

    janus_steps, tower_steps = classify_steps(qpe)
    janus_ids = {j["step"] for j in janus_steps}

    # Spectral reference: first Janus step with a full set of fields
    ref_janus = next(
        (j for j in janus_steps if j.get("E_riemann") is not None), None
    )
    nstar_dt    = infer_nstar_dt(ref_janus) if ref_janus else None
    e_riem_ref  = ref_janus["E_riemann"]   if ref_janus else None
    e_tower_ref = ref_janus["E_tower"]     if ref_janus else None
    gamma_match = ref_janus.get("gamma_k") if ref_janus else None
    zero_idx    = (ref_janus.get("zero_index", 1) - 1) if ref_janus else None  # 0-based

    spectrum = ejanus_spectrum(s_val, nstar_dt, n_zeros) if nstar_dt else np.array([])

    # Non-Janus step energies (raw → offset from first non-Janus step in mHa)
    nj_pairs = [
        (n, step_e[n])
        for n in range(M)
        if n not in janus_ids and n < len(step_e) and step_e[n] is not None
    ]
    if nj_pairs:
        base_e     = nj_pairs[0][1]
        nj_x       = [p[0] for p in nj_pairs]
        nj_offset  = [(p[1] - base_e) * 1000.0 for p in nj_pairs]
    else:
        nj_x, nj_offset = [], []

    # Janus residuals
    janus_res = [(j["step"], j["residual_mHa"])
                 for j in janus_steps if j.get("residual_mHa") is not None]

    # ── layout ────────────────────────────────────────────────────────────────
    n_panels = (
        (1 if janus_res else 0)
        + (1 if len(spectrum) > 0 else 0)
        + (1 if nj_x else 0)
    )
    if n_panels == 0:
        n_panels = 1

    fig = plt.figure(
        figsize=(10, max(4, 3.8 * n_panels)),
        facecolor="white",
    )
    gs = gridspec.GridSpec(
        n_panels, 1, figure=fig, hspace=0.52,
        top=0.93, bottom=0.07, left=0.10, right=0.96,
    )

    status_str = "PASS ✓" if ch_ok else "FAIL ✗"
    status_col = C_PASS if ch_ok else C_FAIL
    fig.suptitle(
        f"PATH-R Riemann scaffold — {mech_name}   [{status_str}]   "
        f"N={N}   {grp}   m={m}",
        fontsize=11, fontweight="normal", color=status_col, y=0.98,
    )

    panel = 0

    # ── panel 1: Janus residuals ──────────────────────────────────────────────
    if janus_res:
        ax = fig.add_subplot(gs[panel]); panel += 1
        apply_spine_style(ax)

        xs  = [r[0] for r in janus_res]
        res = [r[1] for r in janus_res]
        bar_cols = [C_PASS if r <= QPE_THRESHOLD_MHA else C_FAIL for r in res]

        ax.bar(xs, res, color=bar_cols, width=min(0.6, 0.9 / max(len(xs), 1)),
               zorder=2, edgecolor="white", linewidth=0.5)
        ax.axhline(QPE_THRESHOLD_MHA, color=C_AMBER, linewidth=1.2,
                   linestyle="--", alpha=0.85, label=f"QPE threshold  {QPE_THRESHOLD_MHA} mHa")

        for x_i, r in zip(xs, res):
            ax.text(x_i, r + QPE_THRESHOLD_MHA * 0.04,
                    f"{r:.4f}", ha="center", va="bottom",
                    fontsize=ANNOT_FS, color="#444")

        title_extra = ""
        if gamma_match is not None and zero_idx is not None:
            title_extra = f"   [matched zero k={zero_idx+1},  γ_k = {gamma_match:.6f}]"
        ax.set_title(
            "Spectral identification residuals |E_tower − E_Janus(γ_k)| at Janus steps"
            + title_extra,
            fontsize=TITLE_FS, fontweight="normal", color="#222", pad=8,
        )
        ax.set_ylabel("residual  (mHa)", fontsize=LABEL_FS, color="#444")
        ax.set_xlabel("step  n", fontsize=LABEL_FS, color="#444")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"n={x}" for x in xs], fontsize=TICK_FS)
        y_max = max(max(res), QPE_THRESHOLD_MHA) * 1.45
        ax.set_ylim(0, y_max)
        leg = ax.legend(fontsize=ANNOT_FS, framealpha=0.9, edgecolor="#dddddd",
                        handlelength=1.4, loc="upper right")
        leg.get_frame().set_linewidth(0.5)

        # Horizontal band for QPE window
        ax.axhspan(0, QPE_THRESHOLD_MHA, color=C_AMBER, alpha=0.04, zorder=0)

    # ── panel 2: Riemann zero fingerprint ─────────────────────────────────────
    if len(spectrum) > 0:
        ax = fig.add_subplot(gs[panel]); panel += 1
        apply_spine_style(ax)

        zk    = list(range(1, n_zeros + 1))
        b_col = [C_PASS if i == zero_idx else C_PURPLE for i in range(n_zeros)]
        ax.bar(zk, spectrum, color=b_col, width=0.7, zorder=2,
               edgecolor="white", linewidth=0.4)

        if e_tower_ref is not None:
            ax.axhline(e_tower_ref, color=C_BLUE, linewidth=1.2, linestyle="-.",
                       alpha=0.75, label=f"E_tower = {e_tower_ref:.4f} Ha")
        if e_riem_ref is not None:
            ax.axhline(e_riem_ref, color=C_GREY, linewidth=0.8, linestyle=":",
                       alpha=0.8, label=f"E_∞ = {e_riem_ref:.4f} Ha")

        matched_lbl = ""
        if zero_idx is not None:
            matched_lbl = f"  [k={zero_idx+1} matched, green]"
        ax.set_title(
            f"Riemann zero fingerprint — {n_zeros} zeros in eigenphase window" + matched_lbl,
            fontsize=TITLE_FS, fontweight="normal", color="#222", pad=8,
        )
        ax.set_xlabel("zero index  k", fontsize=LABEL_FS, color="#444")
        ax.set_ylabel("E_Janus(γ_k)  (Ha)", fontsize=LABEL_FS, color="#444")
        ax.set_xticks(zk)
        ax.set_xticklabels([str(k) for k in zk], fontsize=TICK_FS)
        leg = ax.legend(fontsize=ANNOT_FS, framealpha=0.9, edgecolor="#dddddd",
                        handlelength=1.4, loc="upper right")
        leg.get_frame().set_linewidth(0.5)

    # ── panel 3: non-Janus step profile ───────────────────────────────────────
    if nj_x:
        ax = fig.add_subplot(gs[panel]); panel += 1
        apply_spine_style(ax)

        min_off = min(nj_offset)
        pt_cols = [C_ORANGE if o <= min_off + 0.2 else C_BLUE for o in nj_offset]

        ax.plot(nj_x, nj_offset, color=C_BLUE, linewidth=1.8,
                marker="o", markersize=4,
                markeredgecolor="white", markeredgewidth=0.8, zorder=2,
                label="non-Janus E_ref")

        # Colour the minimum region
        for x_i, y_i, c in zip(nj_x, nj_offset, pt_cols):
            ax.plot(x_i, y_i, "o", markersize=4,
                    color=c, markeredgecolor="white", markeredgewidth=0.8, zorder=3)

        ax.axhline(0, color="#cccccc", linewidth=0.6, linestyle=":", zorder=0)

        # Annotate Janus step positions
        y_lo = min(nj_offset) - abs(min(nj_offset)) * 0.12
        for js in sorted(janus_ids):
            ax.axvline(js, color="#eeeeee", linewidth=0.6, linestyle=":", zorder=0)
            ax.text(js, y_lo, f"n={js}\nJanus",
                    ha="center", va="top",
                    fontsize=ANNOT_FS - 0.5, color="#bbbbbb")

        ax.set_title(
            "Non-Janus step reference energies along catalytic path  "
            "[offset from n=0 baseline, mHa]",
            fontsize=TITLE_FS, fontweight="normal", color="#222", pad=8,
        )
        ax.set_xlabel("step  n", fontsize=LABEL_FS, color="#444")
        ax.set_ylabel("ΔE from step 0  (mHa)", fontsize=LABEL_FS, color="#444")
        ax.set_xticks(list(range(M)))
        ax.set_xticklabels([str(n) for n in range(M)], fontsize=TICK_FS)
        leg = ax.legend(fontsize=ANNOT_FS, framealpha=0.9, edgecolor="#dddddd",
                        handlelength=1.4, loc="upper right")
        leg.get_frame().set_linewidth(0.5)

    # footer
    fig.text(
        0.5, 0.01,
        f"{mech_name} · PATH-R Part B · N={N} orbs · "
        f"ℤ_{m} phase closure {'✓' if ph_ok else '✗'} · "
        f"QPE threshold {QPE_THRESHOLD_MHA} mHa · elapsed {sec:.2f}s",
        ha="center", fontsize=6.5, color="#888888",
    )

    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("json", help="Path to mqe run --riemann JSON output")
    parser.add_argument(
        "--mechanism", "-m", default=None,
        help="Mechanism name to plot (default: all mechanisms in file)",
    )
    parser.add_argument(
        "--format", choices=["png", "pdf", "both"], default="both",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--n-zeros", type=int, default=20,
        help="Riemann zeros to display in fingerprint panel (max 30, default 20)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory (default: same directory as the JSON file)",
    )
    args = parser.parse_args()

    n_zeros = min(args.n_zeros, len(RIEMANN_ZEROS))
    data    = load_results(args.json)
    mechs   = data.get("mqe_riemann_validation", {})

    if not mechs:
        print("ERROR: no mqe_riemann_validation found in JSON.", file=sys.stderr)
        sys.exit(1)

    to_plot = [args.mechanism] if args.mechanism else list(mechs.keys())
    out_dir = (
        pathlib.Path(args.out) if args.out
        else pathlib.Path(args.json).parent
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in to_plot:
        if name not in mechs:
            print(f"WARNING: {name!r} not in file, skipping.", file=sys.stderr)
            continue

        fig  = build_figure(name, mechs[name], n_zeros=n_zeros)
        stem = f"{name}_riemann_validation"

        if args.format in ("png", "both"):
            p = out_dir / f"{stem}.png"
            fig.savefig(p, dpi=args.dpi, bbox_inches="tight", facecolor="white")
            print(f"Saved → {p}")

        if args.format in ("pdf", "both"):
            p = out_dir / f"{stem}.pdf"
            fig.savefig(p, bbox_inches="tight", facecolor="white")
            print(f"Saved → {p}")

        plt.close(fig)


if __name__ == "__main__":
    main()
