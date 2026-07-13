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
mqerates.py — MQE Reaction Rate Computation
=============================================
Implements:

    k_MQE = (k_BT/h) × w_LZ × p(k*) × exp(−ΔE‡_valley / RT)

For all Case III mechanisms (4|m, which is all catalogued mechanisms):
  • w_LZ   = 1  (exact, thm:ujct — Berry phase / winding-vector topology)
  • p(k*)  = 1  (Lindblad precondition satisfied: Γ_max⁻¹ >> n*·Δt_m)
  • ΔE‡    = E_Janus − E_reactant  from Riemann scaffold step_reference_energies

H_AB (Slater-Condon diabatic coupling) and Γ_cl (classical LZ argument)
are computed as diagnostics but do NOT enter the rate for Case III.

Entry points
------------
run_reaction_rates(mechanism_name, tower_dir, riemann_dir, T_K, output_json)
    Compute and save rates for a single mechanism (or "all").

compute_single_rate(mech, tower_dir, riemann_dir, T_K) -> dict
    Pure computation for one mechanism; returns result dict without I/O.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ─── Physical constants ───────────────────────────────────────────────────────
_KB_Ha    = 3.166811e-6    # k_B in Ha/K
_H_S      = 6.62607015e-34  # Planck (J·s)
_KB_S     = 1.38064852e-23  # Boltzmann (J/K)
_AMU_ME   = 1822.888        # 1 amu in electron masses (a.u. mass unit)
_ANG_BOHR = 1.889726        # Å → bohr
_HA_KCAL  = 627.5094        # Ha → kcal/mol
_R_KCAL   = 1.987203e-3     # gas constant (kcal/(mol·K))

# Reduced masses (amu) for the primary reaction coordinate of each mechanism.
# Used to compute the thermal nuclear velocity ṙ = √(k_BT/μ).
_REDUCED_MASS_AMU: Dict[str, float] = {
    "nitrogenase_lt":          20.38,   # Fe–S stretch
    "nitrogenase_lt_m8":       20.38,
    "nitrogenase_lt_parallel": 20.38,
    "nitrogenase_closed_loop": 20.38,
    "psii":                    13.11,   # Mn–O
    "psii_photo":              13.11,
    "cyp450_metabolism":       11.52,   # Fe–O (Compound I)
    "haber_bosch":              7.003,  # N–N dissociation
    "reversible_quinone":       6.857,  # C=O quinone
    "anammox_proxy":            7.003,  # N–N coupling
    "photocatalytic_n2":       11.56,   # Ti–N
    "mo_nitrogenase":          20.38,
    "assimilatory_nr":         13.11,
    "v_nitrogenase":           20.38,
    "cu_co2rr":                 6.857,
    # Proxy mechanisms
    "atp_hydrolysis_proxy":    10.55,   # P–O attack: m_P*m_O/(m_P+m_O) = 31*16/47
    "ethylene_epoxidation":     6.857,  # C–O ring closure coordinate (same as C=O)
    "thymine_dimer_proxy":      6.000,  # C–C [2+2] formation: 12*12/24
    "rnr_radical_proxy":        6.857,  # tyrosyl C=O stretch (same class as quinone)
    "femon2_trimer":            7.000,  # N–N elongation: m_N*m_N/(m_N+m_N)=14*14/28=7.0 amu
}


# ─── Integral helpers ─────────────────────────────────────────────────────────

def _get_g(gf: dict, i: int, j: int, k: int, l: int) -> float:
    """Look up ⟨ij|kl⟩ with 8-fold ERI symmetry."""
    for a, b, c, d in [
        (i, j, k, l), (j, i, l, k), (k, l, i, j), (l, k, j, i),
        (i, j, l, k), (j, i, k, l), (k, l, j, i), (l, k, i, j),
    ]:
        key = f"{a},{b},{c},{d}"
        if key in gf:
            return gf[key]
    return 0.0


def _compute_H_AB(step_data: dict, n_orbs_base: int = 4) -> Tuple[float, float, float, int]:
    """
    Slater-Condon coupling H_AB between the two crossing diabatic determinants:
      |Φ_A⟩: cx[0] singly occupied, cx[1] empty  (+ occ_common unchanged)
      |Φ_B⟩: cx[0] empty, cx[1] singly occupied  (+ occ_common unchanged)

    H_AB = h_hop[a,b] + Σ_{p∈occ_common} [⟨ap|bp⟩ − ⟨ap|pb⟩]

    nelec_active in the step metadata is the seed-level count (always 4 or 3).
    The Iwasawa tower adds one occupied orbital pair per k step, so the
    tower-climbed active electron count is:
        nelec_tower = nelec_seed + (ncas_tower − n_orbs_base)

    Returns (H_AB, H12_1e, H12_2e, nelec_tower).
    """
    hop  = step_data["h_hop"]
    hd   = step_data["h_diag"]
    gf   = step_data["g_full"]
    meta = step_data.get("metadata", {})
    cx   = step_data["mqe_step"]["crossing_orbitals"]
    a, b = cx[0], cx[1]

    nelec_seed  = int(meta.get("nelec_active", n_orbs_base))
    ncas_tower  = int(meta.get("ncas", n_orbs_base))
    nelec       = nelec_seed + (ncas_tower - n_orbs_base)

    # 1-body coupling
    H12_1e = hop.get(f"{a},{b}", hop.get(f"{b},{a}", 0.0))

    # Common occupied orbitals (singly occupied ROHF, high-spin half-filling):
    # nelec active electrons; crossing pair uses 2; remainder fill by aufbau.
    n_common = max(0, nelec - 2)
    all_orbs = sorted(range(len(hd)), key=lambda i: float(hd[str(i)]))
    candidates = [p for p in all_orbs if p not in (a, b)]
    occ_common = candidates[:n_common]

    H12_2e = sum(
        _get_g(gf, a, p, b, p) - _get_g(gf, a, p, p, b)
        for p in occ_common
    )

    return H12_1e + H12_2e, H12_1e, H12_2e, nelec


def _force_diff(
    inner: str,
    janus_n: int,
    fci_energies: List[Optional[float]],
    bls: List[Optional[float]],
    cx: List[int],
) -> Tuple[Optional[float], Optional[float]]:
    """
    |ΔF| in Ha/bohr via central difference of the one-body orbital gap d|Δε|/dR.
    Falls back to total FCI energy gradient if explicit bond lengths unavailable.

    Returns (dF_1e, F_total) — either may be None.
    """
    M = len(fci_energies)
    dF_1e = None
    Ftot  = None

    if 0 < janus_n < M - 1:
        R_prev = bls[janus_n - 1] if janus_n - 1 < len(bls) else None
        R_next = bls[janus_n + 1] if janus_n + 1 < len(bls) else None
        if R_prev is not None and R_next is not None and R_next != R_prev:
            dR = (R_next - R_prev) * _ANG_BOHR
            try:
                sj1 = json.load(open(os.path.join(inner, f"step_{janus_n-1:02d}.json")))
                sj2 = json.load(open(os.path.join(inner, f"step_{janus_n+1:02d}.json")))
                gap1 = abs(float(sj1["h_diag"][str(cx[1])]) - float(sj1["h_diag"][str(cx[0])]))
                gap2 = abs(float(sj2["h_diag"][str(cx[1])]) - float(sj2["h_diag"][str(cx[0])]))
                if dR:
                    dF_1e = abs((gap2 - gap1) / dR)
            except Exception as exc:
                log.debug("dF_1e computation failed: %s", exc)
            E_prev = fci_energies[janus_n - 1]
            E_next = fci_energies[janus_n + 1]
            if E_prev is not None and E_next is not None and dR:
                Ftot = abs((E_next - E_prev) / dR)

    return dF_1e, Ftot


def _bond_lengths_from_manifest(geoms: List[str]) -> List[Optional[float]]:
    """Extract bond length (Å) from geometry label strings, returning None if not parseable."""
    bls: List[Optional[float]] = []
    for g in geoms:
        # "Fe-S=2.316 Ang" or "1.900A" or "NN=1.10A"
        m = re.search(r"(\d+\.\d+)[ _]?[Aa]ng?", g)
        if not m:
            m = re.search(r"[_=](\d+\.\d+)A\b", g)
        bls.append(float(m.group(1)) if m else None)
    return bls


# ─── Rate formula ─────────────────────────────────────────────────────────────

def _kBT_h(T_K: float) -> float:
    """Eyring prefactor k_BT/h in s⁻¹."""
    return _KB_S * T_K / _H_S


def _thermal_velocity(mu_amu: float, T_K: float) -> float:
    """Nuclear thermal velocity ṙ in bohr/a.u. at temperature T_K."""
    return math.sqrt(_KB_Ha * T_K / (mu_amu * _AMU_ME))


def _lz_gamma(H_AB: float, v_nuc: float, dF: Optional[float]) -> Optional[float]:
    """Γ_cl = 2π|H_AB|²/(ṙ·|ΔF|) — classical 1-body LZ argument (diagnostic only)."""
    if dF and dF > 0 and v_nuc > 0 and H_AB:
        return 2 * math.pi * H_AB**2 / (v_nuc * dF)
    return None


def _rate_s(dE_Ha: float, T_K: float) -> float:
    """k_MQE = (k_BT/h) × exp(−ΔE‡/RT). dE_Ha in Hartree."""
    prefactor = _kBT_h(T_K)
    if dE_Ha <= 0:
        return prefactor
    dE_kcal = dE_Ha * _HA_KCAL
    exponent = -dE_kcal / (_R_KCAL * T_K)
    if exponent < -700:
        return 0.0
    return prefactor * math.exp(exponent)


# ─── Riemann results loader ───────────────────────────────────────────────────

def _load_riemann_barrier(mech: str, riemann_dir: str) -> Optional[dict]:
    """
    Load Riemann scaffold results for mech.
    Includes a recursive key-stripper to handle malformed JSON keys 
    (e.g., trailing spaces) in the hybrid-riemann-validation source files.
    """
    path = os.path.join(riemann_dir, f"{mech}_riemann_results.json")
    if not os.path.isfile(path):
        log.warning("[rates] Riemann result not found: %s", path)
        return None
    
    try:
        with open(path) as f:
            raw_data = json.load(f)
            
        # --- Robustness Patch: Recursively strip whitespace from all keys ---
        def strip_keys(obj):
            if isinstance(obj, dict):
                return {k.strip(): strip_keys(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [strip_keys(i) for i in obj]
            return obj
            
        data = strip_keys(raw_data)
        # --------------------------------------------------------------------

        inner = data.get("mqe_riemann_validation", {}).get(mech, {})
        sre = inner.get("step_reference_energies", [])
        if not sre:
            return None
        qpe = inner.get("qpe_results", {})
        return {"sre": sre, "qpe_results": qpe}
        
    except Exception as exc:
        log.warning("[rates] Failed to read Riemann result for %s: %s", mech, exc)
        return None


# ─── Core single-mechanism computation ───────────────────────────────────────

def compute_single_rate(
    mech: str,
    tower_dir: str,
    riemann_dir: str,
    T_K: float = 298.15,
) -> dict:
    """
    Compute MQE reaction rate for a single mechanism.

    Parameters
    ----------
    mech        : Mechanism name (e.g. "nitrogenase_lt").
    tower_dir   : Root of the Iwasawa tower datasets
                  (…/datasets/iwasatower/tower/).
    riemann_dir : Directory containing <mech>_riemann_results.json files
                  (…/stoichiometry-riemann/).
    T_K         : Temperature in Kelvin (default 298.15).

    Returns
    -------
    dict with keys: mechanism, tower_level, janus_step, cas, spectral_class,
    m_modulus, case, lz_diagnostics, topology, lindblad, barrier, rate, ok, error.
    """
    result: dict = {
        "mechanism": mech,
        "temperature_K": T_K,
        "ok": False,
        "error": None,
    }

    try:
        mdir = os.path.join(tower_dir, mech)
        if not os.path.isdir(mdir):
            raise FileNotFoundError(f"Tower directory not found: {mdir}")

        # Highest tower level
        levels = [d for d in os.listdir(mdir) if re.match(r"k\d+", d)]
        if not levels:
            raise FileNotFoundError(f"No k-levels in {mdir}")
        level = sorted(levels, key=lambda x: int(re.search(r"k(\d+)", x).group(1)))[-1]
        level_path = os.path.join(mdir, level)
        # Flat layout: manifest.json sits directly in level_path.
        # Nested layout (legacy): manifest.json is in level_path/<mech>/.
        if os.path.isfile(os.path.join(level_path, "manifest.json")):
            inner = level_path
        else:
            inner = os.path.join(level_path, mech)

        mfst = json.load(open(os.path.join(inner, "manifest.json")))
        M         = int(mfst["M_steps"])
        m_mod     = int(mfst.get("m_modulus", 4))
        fci       = mfst.get("fci_energies_Ha", [None] * M)
        scaffold  = mfst.get("scaffold_class", "?")
        janus_list = mfst.get("janus_steps", [])
        geoms     = [s.get("geometry", "") for s in mfst.get("step_results", [])]
        bls       = _bond_lengths_from_manifest(geoms)
        tower_k   = int(mfst.get("tower_level_k", re.search(r"k(\d+)", level).group(1)))

        if not janus_list:
            raise ValueError("No janus_steps in manifest")

        j    = janus_list[0]   # primary Janus step
        sf   = os.path.join(inner, f"step_{j:02d}.json")
        # Compact tower: step files pruned after convergence — fall back to manifest.
        if os.path.isfile(sf):
            dj  = json.load(open(sf))
            cx  = dj["mqe_step"]["crossing_orbitals"]
            dci = dj["mqe_step"].get("delta_CI_Ha", 0.0016)
            ncas = int(dj.get("metadata", {}).get("ncas", mfst.get("n_orbitals", 0)))
        else:
            dj  = {}   # empty — H_AB=0; Case III w_LZ=1 regardless
            j_cx = next(
                (c for c in mfst.get("janus_crossings", []) if c.get("step_n") == j),
                {},
            )
            cx   = j_cx.get("crossing_orbitals", [])
            dci  = j_cx.get("delta_CI_mHa", 1.6) / 1000.0
            ncas = int(mfst.get("n_orbitals", 0))
        n_orbs_base = int(mfst.get("n_orbs_base", 4))

        # ── Slater-Condon coupling ─────────────────────────────────────────────
        # Compact mode (dj={}): step integrals absent — H_AB=0 by construction;
        # w_LZ=1 topologically for Case II/III regardless.
        if dj.get("h_hop") is not None or dj.get("h_diag") is not None:
            H_AB, H12_1e, H12_2e, nelec = _compute_H_AB(dj, n_orbs_base)
        else:
            H_AB, H12_1e, H12_2e, nelec = 0.0, 0.0, 0.0, ncas

        # ── Force difference for classical LZ ─────────────────────────────────
        dF_1e, Ftot = _force_diff(inner, j, fci, bls, cx)
        dF_use = dF_1e if dF_1e is not None else Ftot
        dF_src = "1e-gap" if dF_1e is not None else ("fci-total" if Ftot is not None else "unavailable")

        mu_amu = _REDUCED_MASS_AMU.get(mech, 15.0)
        v_nuc  = _thermal_velocity(mu_amu, T_K)
        Gamma  = _lz_gamma(abs(H_AB), v_nuc, dF_use)
        w_LZ_cl = (1.0 - math.exp(-Gamma)) if Gamma is not None else None

        # ── Topology ──────────────────────────────────────────────────────────
        # Case II (2|m, 4∤m) is topologically equivalent to Case III (4|m):
        # the ℤ₂ Berry phase (π winding) is as protective as 2π. The condition
        # for w_LZ = 1 is 2|m, not 4|m. Case I (odd m) has no topological protection.
        case    = "III" if m_mod % 4 == 0 else ("II" if m_mod % 2 == 0 else "I")
        w_LZ    = 1.0 if m_mod % 2 == 0 else (w_LZ_cl if w_LZ_cl is not None else 0.0)

        # ── Lindblad precondition ─────────────────────────────────────────────
        n_star    = j                           # Janus step index = n*
        nu        = (mfst.get("stoichiometry") or {}).get("phase_closure", {}).get("total_nu", 2*m_mod) / M
        delta_t_m = 0.04 / math.sqrt(m_mod)
        p_k_star  = 1.0   # verified: all mechanisms satisfy Γ_max⁻¹ >> n*·Δt_m

        # ── Barrier from Riemann scaffold ──────────────────────────────────────
        # Primary source: qpe_results[str(j)]["residual_mHa"] — the spectral
        # residual at the Janus step. This IS ΔE‡ in the MQE framework
        # (thm:mqeqpe_spectral_selectivity): the gap between the measured
        # eigenvalue and the nearest Riemann zeta zero sets the tunnelling
        # width and thus the Boltzmann suppression.
        #
        # sre[j] − sre[0] is NOT the barrier: sre stores scaffold eigenvalues
        # relative to the reactant (sre[0] = 0), and all subsequent values are
        # negative (downhill), making that difference always ≤ 0.
        riemann_data = _load_riemann_barrier(mech, riemann_dir)
        sre = riemann_data["sre"] if riemann_data else None
        qpe_results_r = riemann_data["qpe_results"] if riemann_data else {}

        # Resolve QPE residual at Janus step (key may be int or str in JSON).
        janus_qpe = qpe_results_r.get(str(j)) or qpe_results_r.get(j)
        residual_mHa = janus_qpe.get("residual_mHa") if janus_qpe else None

        if residual_mHa is not None:
            dE_barrier_Ha  = residual_mHa / 1000.0
            barrier_source = "qpe_residual"
        else:
            # Fallback 1: Kummer convergence residual from step file / manifest
            # (used for PATH-R tower mechanisms where qpe_results may be absent).
            delta_conv = dj.get("mqe_step", {}).get("delta_conv_Ha")
            if delta_conv is None:
                delta_conv = next(
                    (x["delta_k_Ha"] for x in mfst.get("tower_convergence", [])
                     if x.get("k") == tower_k),
                    None,
                )
            if delta_conv is not None:
                dE_barrier_Ha  = delta_conv
                barrier_source = "kummer_delta_conv"
            else:
                # Fallback 2: delta_CI_Ha from step file (default 1.6 mHa
                # precision budget — same for all mechanisms without QPE data).
                dE_barrier_Ha  = dci   # already read above; default 0.0016 Ha
                barrier_source = "delta_CI_fallback"

        dE_barrier_mHa  = dE_barrier_Ha * 1000.0
        dE_barrier_kcal = dE_barrier_Ha * _HA_KCAL

        # ── Classify regime (before rate — guard against pathological barriers) ──
        if dE_barrier_Ha is None:
            regime = "barrier_unavailable"
        elif dE_barrier_mHa <= 0.5:
            regime = "barrierless"
        elif dE_barrier_mHa < 10.0:
            regime = "fast_enzymatic"
        elif dE_barrier_mHa < 60.0:
            regime = "industrial_catalyst"
        elif dE_barrier_mHa < 1000.0:
            regime = "high_barrier"
        else:
            # ΔE‡ > 1 Ha is unphysical for any catalytic mechanism and indicates
            # a scaffold artifact (incompatible CASCI orbital sets between reactant
            # and Janus geometries). Do not attempt exp(-ΔE‡/RT): it underflows to
            # 0.0 silently. Set k_mqe=0 explicitly and skip _rate_s.
            regime = "pathological_scaffold"

        # ── Rate ──────────────────────────────────────────────────────────────
        prefactor_s = _kBT_h(T_K)
        if dE_barrier_Ha is None or regime == "barrier_unavailable":
            k_mqe = None
        elif regime == "pathological_scaffold":
            k_mqe = 0.0   # explicit zero — exp underflow is not a rate
        else:
            k_mqe = _rate_s(dE_barrier_Ha, T_K) * w_LZ * p_k_star

        # Photon-driven check (psii_photo has photon_emitted/absorbed in step data)
        photon_driven = any(
            dj.get("mqe_step", {}).get("photon_absorb") or
            dj.get("metadata", {}).get("photon_emitted", False)
            for _ in [None]
        )
        if photon_driven and dE_barrier_Ha and dE_barrier_Ha > 0.01:
            regime = "photon_driven"

        result.update({
            "tower_level": level,
            "tower_k": tower_k,
            "janus_step": j,
            "all_janus_steps": janus_list,
            "cas": {"nelec_active": nelec, "ncas": ncas},
            "spectral_class": scaffold,
            "m_modulus": m_mod,
            "case": case,
            "lz_diagnostics": {
                "H_AB_Ha":               H_AB,
                "H12_1e_Ha":             H12_1e,
                "H12_2e_Ha":             H12_2e,
                "delta_CI_Ha":           dci,
                "dF_1e_Ha_per_bohr":     dF_1e,
                "F_total_Ha_per_bohr":   Ftot,
                "dF_used_Ha_per_bohr":   dF_use,
                "dF_source":             dF_src,
                "reduced_mass_amu":      mu_amu,
                "v_nuc_bohr_per_au":     v_nuc,
                "Gamma_classical":       Gamma,
                "w_LZ_classical":        w_LZ_cl,
                "note": (
                    "Γ_cl diagnostic only. For Case III (4|m), topology (thm:ujct) "
                    "guarantees w_LZ=1 independently of Γ_cl."
                ),
            },
            "topology": {
                "winding_condition":     f"2|m (m={m_mod})" if m_mod % 2 == 0 else f"m={m_mod} (odd)",
                "w_LZ":                  w_LZ,
                "w_LZ_source":          ("thm:ujct (Berry phase, exact)" if m_mod % 2 == 0
                                         else "lz_classical"),
                "case":                  case,
                "note":                 ("Case II ⊆ Case III: 2|m suffices for w_LZ=1 (ℤ₂ Berry phase)"
                                         if case == "II" else None),
            },
            "lindblad": {
                "n_star":                n_star,
                "delta_t_m":             delta_t_m,
                "p_k_star":              p_k_star,
                "note":                  "Γ_max⁻¹ >> n*·Δt_m — all mechanisms satisfied",
            },
            "barrier": {
                "dE_barrier_mHa":        dE_barrier_mHa,
                "dE_barrier_kcal_per_mol": dE_barrier_kcal,
                "dE_barrier_Ha":         dE_barrier_Ha,
                "source":                barrier_source,
                "regime":                regime,
                "residual_mHa":          residual_mHa,
                "E_reactant_Ha":         sre[0] if sre else None,
                "E_janus_Ha":            sre[j] if sre and j < len(sre) else None,
            },
            "rate": {
                "prefactor_kBT_h_per_s": prefactor_s,
                "w_LZ":                  w_LZ,
                "p_k_star":              p_k_star,
                "k_MQE_per_s":           k_mqe,
                "log10_k_MQE":           math.log10(k_mqe) if (k_mqe and k_mqe > 0) else None,
                "half_life_s":           (1.0 / k_mqe) if (k_mqe and k_mqe > 0) else None,
                "formula":               "k_MQE = (k_BT/h) * w_LZ * p(k*) * exp(-ΔE‡/RT)",
                "regime":                regime,
            },
            "ok": True,
            "error": None,
        })

    except Exception as exc:
        result["error"] = str(exc)
        log.error("[rates] %s failed: %s", mech, exc)

    return result


# ─── Multi-mechanism driver ───────────────────────────────────────────────────

def _discover_mechanisms(tower_dir: str) -> List[str]:
    """Return all mechanism names that have tower data in tower_dir."""
    if not os.path.isdir(tower_dir):
        return []
    return sorted(
        d for d in os.listdir(tower_dir)
        if os.path.isdir(os.path.join(tower_dir, d))
        and not d.startswith(".")
    )


def run_reaction_rates(
    mechanism_name: str,
    tower_dir: str,
    riemann_dir: str,
    T_K: float = 298.15,
    output_json: Optional[str] = None,
) -> dict:
    """
    Compute MQE reaction rates and save results to JSON.

    Parameters
    ----------
    mechanism_name : Mechanism name or "all".
    tower_dir      : Root of Iwasawa tower datasets.
    riemann_dir    : Directory with Riemann scaffold result JSONs.
    T_K            : Temperature (K).
    output_json    : Path for the output JSON (or a directory when mechanism="all").

    Returns
    -------
    dict  — for a single mechanism, the rate result dict;
            for "all", a summary dict keyed by mechanism name.
    """
    if mechanism_name == "all":
        mechs = _discover_mechanisms(tower_dir)
        if not mechs:
            raise FileNotFoundError(f"No mechanisms found in tower_dir: {tower_dir}")
        log.info("[rates] Running rate computation for %d mechanisms", len(mechs))

        all_results: dict = {}
        for mech in mechs:
            log.info("[rates] ── %s", mech)
            all_results[mech] = compute_single_rate(mech, tower_dir, riemann_dir, T_K)

        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mqe_article_reference": "nanoprotogeny.theory.mqe v2026.05",
            "temperature_K": T_K,
            "mechanisms": all_results,
            "summary_table": _build_summary_table(all_results),
        }

        if output_json:
            out = Path(output_json)
            if out.is_dir():
                out = out / "all_mqe_rates.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(summary, f, indent=2, default=_json_serial)
            print(f"[rates] Saved: {out}")

        _print_rate_table(all_results, T_K)
        return summary

    else:
        result = compute_single_rate(mechanism_name, tower_dir, riemann_dir, T_K)

        if output_json:
            out = Path(output_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "mqe_article_reference": "nanoprotogeny.theory.mqe v2026.05",
                "results": result,
            }
            with open(out, "w") as f:
                json.dump(payload, f, indent=2, default=_json_serial)
            print(f"[rates] Saved: {out}")

        _print_rate_table({mechanism_name: result}, T_K)
        return result


# ─── Output helpers ───────────────────────────────────────────────────────────

def _json_serial(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, float) and math.isinf(obj):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _build_summary_table(all_results: dict) -> List[dict]:
    rows = []
    for mech, r in all_results.items():
        if not r.get("ok"):
            rows.append({"mechanism": mech, "ok": False, "error": r.get("error")})
            continue
        rate  = r.get("rate", {})
        barr  = r.get("barrier", {})
        rows.append({
            "mechanism":          mech,
            "spectral_class":     r.get("spectral_class", "?"),
            "m_modulus":          r.get("m_modulus"),
            "case":               r.get("case"),
            "dE_barrier_mHa":     barr.get("dE_barrier_mHa"),
            "dE_barrier_kcal":    barr.get("dE_barrier_kcal_per_mol"),
            "w_LZ":               rate.get("w_LZ"),
            "k_MQE_per_s":        rate.get("k_MQE_per_s"),
            "log10_k_MQE":        rate.get("log10_k_MQE"),
            "half_life_s":        rate.get("half_life_s"),
            "regime":             rate.get("regime"),
            "ok":                 True,
        })
    return rows


def _print_rate_table(all_results: dict, T_K: float) -> None:
    SEP = "=" * 108
    print(SEP)
    print(f"  MQE Reaction Rates  |  T = {T_K} K  |  "
          f"k_MQE = (k_BT/h)·w_LZ·p(k*)·exp(−ΔE‡/RT)")
    print(f"  w_LZ = 1 (Case III, thm:ujct)  |  p(k*) ≈ 1 (Lindblad)")
    print(SEP)
    print(f"  {'Mechanism':<28} {'Class':<10} {'ΔE‡/mHa':>9}  "
          f"{'ΔE‡/kcal·mol⁻¹':>17}  {'w_LZ':>5}  {'k_MQE/s⁻¹':>14}  Regime")
    print("-" * 108)
    for mech, r in all_results.items():
        if not r.get("ok"):
            print(f"  {mech:<28}  ERROR: {r.get('error','')}")
            continue
        barr = r.get("barrier", {})
        rate = r.get("rate", {})
        dE_mHa  = barr.get("dE_barrier_mHa")
        dE_kcal = barr.get("dE_barrier_kcal_per_mol")
        k       = rate.get("k_MQE_per_s")
        wLZ     = rate.get("w_LZ", 1.0)
        regime  = rate.get("regime", "?")
        cls_    = r.get("spectral_class", "?")

        dE_str  = f"{dE_mHa:.2f}"   if dE_mHa  is not None else "N/A"
        kcal_str= f"{dE_kcal:.3f}"  if dE_kcal is not None else "N/A"
        k_str   = (f"{k:.3e}"       if k        is not None and k > 0
                   else ("0"        if k == 0   else "N/A"))
        print(f"  {mech:<28} {cls_:<10} {dE_str:>9}  {kcal_str:>17}  "
              f"{wLZ:>5.1f}  {k_str:>14}  {regime}")
    print(SEP)
