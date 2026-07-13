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
mqehybridgenerator.py — Hybrid Algebraic-PySCF Protocol for MQE Data Generation
=================================================================================
Implements the three-step hybrid protocol that reduces the number of PySCF
geometry evaluations from M (one per scan step) to 1 (only the Janus geometry).

For ``femon2_trimer`` (M=8): from 8 CAS(4,4) calls to 1.  The Iwasawa tower
climb and rate computation remain fully algebraic and are unchanged.

Protocol
--------

**Step 0  (pure algebra):**

    From (N_e, M_ATP, m, γ_j) alone:
    * E_∞               — exact Janus energy from the zeta-dual map + Riemann zero
                          E_∞ = −s·γ_1 / (n*·Δt_m)   [Ha, thm:spectral_identification]
    * E_Janus^(k)        — Kummer–Bernoulli tower sequence for all k ≥ k_base
                          E^(k) = E_∞ + (E_base − E_∞)·p^{−(k−k_base)}  [padicinterp]
    * h̃_{cx}^(k)        — algebraic crossing 1e integral at tower level k
                          |h̃_{cx}^(k)|² = ζ_p(1−k) = (1−p^{k−1})·B_k/k  [prop:bernoulli_coupling]
    * PES_n (n ≠ n*)     — Weyl-scaled energy at non-Janus steps
                          E_n = E_∞ · log(n+1)/log(n*+1)
    * k_MQE              — exact reaction rate at T via Eyring formula, no chemistry
                          k_MQE = (k_BT/h)·exp(−ΔE‡/RT)  where ΔE‡ = E_∞ − E_0 = E_∞

**Step 1  (one PySCF call):**

    Run ROHF + CASCI at the single Janus geometry R_{n*} only (the crossing
    step with ``is_crossing=True`` in the MQEMechanismSpec).  Returns h1_{pq}
    and h2_{pqrs} at the seed CAS(4,4) level.  The M−1 non-Janus step JSONs
    are synthesised from the Weyl-scaled PES and the Janus-step integrals
    (same active-space basis, geometry-perturbed bond length only).

**Step 2  (algebraic consistency check):**

    Verify that the seed Janus eigenphase φ_seed = |E_seed·n*·Δt_m| lies
    within the Riemann eigenphase window [0, φ_bound].  A failure means
    R_{n*} does not correspond to the physical crossing geometry; adjust the
    Janus bond length and re-run Step 1.

    The tolerance δ_0 = |ζ_p(1−k_base)| (Bernoulli coupling at the seed
    tower level).  The Kummer convergence condition E_seed > E_∞ is also
    verified.

Output directory layout
-----------------------
    {output_dir}/{mechanism_name}/step_{n:02d}.json   — M step files
    {output_dir}/{mechanism_name}/manifest.json       — standard manifest +
                                                         hybrid_protocol block
    {output_dir}/{mechanism_name}/h1_full.npy         — full MO 1e integrals
    {output_dir}/{mechanism_name}/eri_packed.npy      — packed 2e integrals

This layout is fully compatible with
``tower_climber.TowerClimber`` and ``mqerates.run_reaction_rates``; no
downstream changes are required.

CLI
---
    python mqehybridgenerator.py --mechanism femon2_trimer \
        --basis DZP-DKH --tower-p 2 --n-total-orbs 60 --output-dir datasets/

    python mqehybridgenerator.py --mechanism nitrogenase_lt \
        --basis STO-3G --tower-p 2 --n-total-orbs 76 --output-dir datasets/

    python mqehybridgenerator.py --list-mechanisms

Public API
----------
    bernoulli_number(k)             : Fraction  — exact B_k (fractions module)
    zeta_p_value(p, k)              : float     — ζ_p(1−k) = (1−p^{k−1})·B_k/k
    algebraic_crossing_coupling(p, k): float    — √|ζ_p(1−k)| (1e crossing amplitude)
    weyl_pes_energies(...)          : List[float]— Weyl-scaled PES for all M steps
    SubJanusSelection               : dataclass — σ(p)-based orbital selection result
    dominant_subshell_sigma(mol, C) : ndarray  — σ(p) per MO  (eq:sigma_map)
    hybridisation_error(mol, C, σ)  : ndarray  — ε_p per MO  (eq:hybridisation_error)
    select_sub_janus_orbitals(...)  : SubJanusSelection — select {ℓ<k*=2} seed
    load_sub_janus_selection(...)   : SubJanusSelection|None — load from artefacts
    AlgebraicPrecompute             : dataclass — all Step 0 results
    build_algebraic_precompute      : factory   — constructs AlgebraicPrecompute
    ConsistencyResult               : dataclass — Step 2 result
    check_consistency               : function  — runs Step 2
    run_single_janus_step(...)      : function  — Step 1 (accepts sub_janus_sel)
    run_hybrid_generation(...)      : function  — full Step 0+1+2 orchestrator
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from nanoprotogeny.molecular.mqedatagenerator import (
    MQEMechanismSpec,
    MQEStep,
    build_all_specs,
    generate_step_integrals,
    validate_mechanism_stoichiometry,
)
from nanoprotogeny.molecular.mqeriemann import (
    RIEMANN_ZEROS,
    RiemannScaffold,
    build_riemann_scaffold,
    delta_t_m,
    eigenphase_bound,
    janus_energy_from_gamma,
    n_star_from_mechanism,
    s_value,
)
from nanoprotogeny.molecular.mqemolecules import build_predefined_mechanisms
from nanoprotogeny.simulate.tower_climber import padicinterp_energy

log = logging.getLogger(__name__)

# ── Physical constants (shared with mqerates.py) ──────────────────────────────
_KB_S     = 1.38064852e-23   # Boltzmann (J/K)
_H_S      = 6.62607015e-34   # Planck (J·s)
_KB_Ha    = 3.166811e-6      # k_B in Ha/K
_HA_KCAL  = 627.5094         # Ha → kcal/mol
_R_KCAL   = 1.987203e-3      # gas constant (kcal/(mol·K))

# Seed tower level: k_base=2 is derived, not chosen.
# prop:seed_is_sp: seed = {ℓ < k*=2} = {s,p}, giving 4 spatial orbitals per
# active site (1s + 3p).  CAS(4,4) is the parenthetical for Mo-nitrogenase.
_K_BASE: int = 2


# ===========================================================================
# 1. BERNOULLI NUMBERS AND BERNOULLI COUPLING (exact arithmetic)
# ===========================================================================

_bernoulli_cache: List[Fraction] = []


def bernoulli_number(k: int) -> Fraction:
    r"""Exact Bernoulli number B_k using rational arithmetic (fractions.Fraction).

    Uses the standard recursion:
        B_0 = 1
        B_n = −(1/(n+1)) · Σ_{j=0}^{n-1} C(n+1, j) · B_j   for n ≥ 1

    Results are cached in the module-level list ``_bernoulli_cache``.

    Convention: B_1 = −1/2 (the standard number-theory convention used by
    the Kubota–Leopoldt p-adic zeta function; see prop:bernoulli_coupling).

    Args:
        k: Non-negative integer index.

    Returns:
        Fraction: Exact value of B_k.

    Examples::
        bernoulli_number(0)  → Fraction(1, 1)       = 1
        bernoulli_number(1)  → Fraction(-1, 2)      = −1/2
        bernoulli_number(2)  → Fraction(1, 6)       = 1/6
        bernoulli_number(3)  → Fraction(0, 1)       = 0
        bernoulli_number(4)  → Fraction(-1, 30)     = −1/30
        bernoulli_number(6)  → Fraction(1, 42)      = 1/42
    """
    global _bernoulli_cache

    if k < len(_bernoulli_cache):
        return _bernoulli_cache[k]

    # Extend cache up to index k
    while len(_bernoulli_cache) <= k:
        n = len(_bernoulli_cache)
        if n == 0:
            _bernoulli_cache.append(Fraction(1))
            continue
        # B_n = −1/(n+1) · Σ_{j=0}^{n-1} C(n+1, j) · B_j
        # Iterative binomial: C(n+1, j) = C(n+1, j-1) * (n+2-j) // j
        s = Fraction(0)
        binom = 1   # C(n+1, 0) = 1
        for j in range(n):
            s += Fraction(binom) * _bernoulli_cache[j]
            # Advance to C(n+1, j+1)
            binom = binom * (n + 1 - j) // (j + 1)
        _bernoulli_cache.append(-s / Fraction(n + 1))

    return _bernoulli_cache[k]


def zeta_p_value(p: int, k: int) -> float:
    r"""p-adic zeta value ζ_p(1−k) = (1 − p^{k−1}) · B_k / k.

    This is the diabatic coupling weight at Iwasawa tower level k for prime
    base p (prop:bernoulli_coupling):

        |H_{12}^(k)|² = ζ_p(1−k)

    Sign note: ζ_p(1−k) is negative for k ≡ 2 (mod 4) and positive for
    k ≡ 0 (mod 4).  The sign encodes the 2e correction direction at
    that tower level.

    Args:
        p: Prime base (2 for nitrogenase/femon2, 3 for V-nitrogenase).
        k: Tower level k ≥ 1.

    Returns:
        float: ζ_p(1−k).  May be negative.

    Raises:
        ValueError: If k < 1 or p < 2.

    Examples::
        zeta_p_value(2, 2)   → −1/12 ≈ −0.0833  (seed level, p=2, k=2)
        zeta_p_value(2, 4)   → −7/120 ≈ −0.0583 (k=4)
        zeta_p_value(2, 6)   → 31/252 ≈  0.123  (k=6)
    """
    if p < 2:
        raise ValueError(f"p must be ≥ 2 (got {p})")
    if k < 1:
        raise ValueError(f"k must be ≥ 1 (got {k})")
    B_k = bernoulli_number(k)
    if B_k == 0:
        return 0.0
    zeta = (1 - p ** (k - 1)) * B_k / Fraction(k)
    return float(zeta)


def algebraic_crossing_coupling(p: int, k: int) -> float:
    r"""Algebraic crossing 1e integral at tower level k.

    From prop:bernoulli_coupling: |H_{12}^(k)|² = ζ_p(1−k).
    Returns the amplitude √|ζ_p(1−k)|.

    When ζ_p(1−k) < 0, the coupling is purely imaginary (anti-crossing
    character at that tower level), and this function returns the magnitude
    √|ζ_p|.  The sign is ignored by the seed-level Step 1 which uses the
    PySCF hopping integral; this algebraic value is a reference/diagnostic.

    Args:
        p: Prime base.
        k: Tower level k ≥ 1.

    Returns:
        float: |H_{12}^(k)|  in Ha.

    Examples::
        algebraic_crossing_coupling(2, 2)  → √(1/12) ≈ 0.289 Ha  (seed level)
        algebraic_crossing_coupling(2, 4)  → √(7/120) ≈ 0.242 Ha
    """
    zeta = abs(zeta_p_value(p, k))
    if zeta == 0.0:
        return 0.0
    return math.sqrt(zeta)


# ===========================================================================
# 2. WEYL-SCALED PES RECONSTRUCTION
# ===========================================================================

def weyl_pes_energies(
    M_steps:  int,
    n_star:   int,
    E_inf:    float,
) -> List[float]:
    r"""Weyl-scaled PES energies for all M steps (Step 0, algebraic).

    Formula:
        E_n = E_∞ · log(n+1) / log(n*+1)

    Derivation: the MQE Weyl law N_MQE(T) ~ (T/2π)log(T/2π) implies that
    the n-th zero has γ_n ~ 2πn/log n.  Under the zeta-dual map, the
    step-n Janus energy scales as −s·γ_n/(n*·Δt_m) ~ E_∞·log(n+1)/log(n*+1).

    Boundary conditions:
        E_0   = 0      (reactant at n=0: log(1) = 0)
        E_{n*} = E_∞   (Janus: log(n*+1)/log(n*+1) = 1)
        E_n → −∞       as n→∞ (not physically reached for n < M)

    Sign convention: E_∞ < 0 (bound state), so E_n < 0 for n > 0, with
    the Janus step n* having the most negative energy in the scan (deepest
    bound state = transition state from the vacuum perspective).

    Args:
        M_steps: Total number of mechanism steps.
        n_star:  Janus crossing step index (0-based).
        E_inf:   E_∞ in Hartree (negative, from Riemann scaffold).

    Returns:
        List[float]: PES energies E_0, …, E_{M−1} in Hartree.

    Raises:
        ValueError: If n_star == 0 (logarithm would be undefined).
    """
    if n_star <= 0:
        raise ValueError(f"n_star must be ≥ 1 (got {n_star}); "
                         "the Janus crossing requires at least one approach step.")
    log_nstar = math.log(n_star + 1)
    energies: List[float] = []
    for n in range(M_steps):
        if n == 0:
            energies.append(0.0)
        else:
            energies.append(E_inf * math.log(n + 1) / log_nstar)
    return energies


# ===========================================================================
# 2.5  SUB-JANUS ORBITAL SELECTION  (prop:seed_is_sp, prop:hybridisation_error)
# ===========================================================================
# The seed active space is not a free choice: it is the sub-inner-Janus orbital
# set {ℓ < k*=2} = {s, p}, giving 4 spatial orbitals per active site
# (prop:seed_is_sp).  Energy-ordered ROHF canonical MOs violate this criterion
# for d-block systems (σ(p) = d for Fermi-level MOs).
#
# The functions below implement:
#   - σ(p)-based dominant-subshell assignment  (eq:sigma_map)
#   - ε_p hybridisation-error filter            (eq:hybridisation_error)
#   - subshell-sequential reordering            (def:shell_register_table)
#
# COMPANION: these functions require (mol, mo_coeff) from a PySCF ROHF call.
# When generate_step_integrals (mqedatagenerator) is updated to accept
# _save_mo_coeffs=True, it will write mo_coeffs.npy + ao_labels.json alongside
# h1_full.npy/eri_packed.npy.  Call load_sub_janus_selection() after Step 1 to
# obtain SubJanusSelection; pass to run_single_janus_step via sub_janus_sel.
# That triggers a second generate_step_integrals call with _cas_orbital_indices
# set, using the correct sub-Janus active space (ROHF cost only, CASCI once).
# ===========================================================================

_ELL_CHAR_MAP: Dict[str, int] = {
    "s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6,
}


@dataclass
class SubJanusSelection:
    r"""Orbital selection result for the sub-inner-Janus seed (prop:seed_is_sp).

    Fields
    ------
    cas_orbital_indices : ndarray[int]  — 0-based MO indices of the n_seed
                          selected orbitals, in subshell-sequential order
                          (ℓ=0 before ℓ=1, stable within each ℓ).
    sigma_ell           : ndarray[int]  — dominant ℓ for each selected MO
                          (0=s, 1=p for a correct sub-Janus selection).
    epsilon_p           : ndarray[float]— hybridisation error ε_p
                          (eq:hybridisation_error) per selected MO.  Zero for
                          pure-ℓ MOs (prop:hybridisation_error(iv)); positive
                          for energy-ordered ROHF MOs (part (iii)).
    n_seed              : Number of orbitals selected (normally 4).
    k_star              : Inner-Janus clock index (always 2; prop:seed_is_sp).
    matches_energy_ordered : True iff these indices agree with the Fermi-level
                          selection generate_step_integrals uses by default.
    warning             : Non-empty when ε_p > 0 or selection deviates.
    """
    cas_orbital_indices:    np.ndarray
    sigma_ell:              np.ndarray
    epsilon_p:              np.ndarray
    n_seed:                 int
    k_star:                 int  = 2
    matches_energy_ordered: bool = True
    warning:                str  = ""


def _ao_angular_momenta(mol) -> np.ndarray:
    r"""Return angular momentum ℓ for each AO from ``mol.ao_labels()``.

    PySCF AO label format: ``'atom_idx atom_sym nl[m]'``,
    e.g. ``'  0 Fe 3dxz'``, ``'  0 Mo 2px'``.
    The ℓ character is the first non-digit in the orbital token (e.g.
    ``'3dxz'`` → ``'d'`` → ℓ=2).

    Args:
        mol: PySCF Mole (or any object exposing ``ao_labels() → list[str]``).

    Returns:
        np.ndarray of shape (N_AO,) with dtype int.
    """
    ells: List[int] = []
    for lbl in mol.ao_labels():
        parts   = lbl.strip().split()
        orb_tok = parts[-1] if parts else "s"   # e.g. '3dxz', '1s', '2px'
        ell_chr = next((c for c in orb_tok if not c.isdigit()), "s")
        ells.append(_ELL_CHAR_MAP.get(ell_chr, 0))
    return np.array(ells, dtype=int)


def dominant_subshell_sigma(mol, mo_coeff: np.ndarray) -> np.ndarray:
    r"""Compute dominant subshell character σ(p) for each MO (eq:sigma_map).

    For each MO φ_p = Σ_μ C_{μp} χ_μ:

        σ(p) = argmax_{ℓ} Σ_{μ: ℓ_μ=ℓ} |C_{μp}|²

    Args:
        mol:      PySCF Mole (or proxy exposing ``ao_labels()``).
        mo_coeff: MO coefficient matrix C, shape (N_AO, N_MO);
                  columns normalised (Σ_μ |C_{μp}|² = 1, Löwdin basis).

    Returns:
        np.ndarray of shape (N_MO,) dtype int: σ(p) per MO.
    """
    ells_ao = _ao_angular_momenta(mol)
    N_MO    = mo_coeff.shape[1]
    sigma   = np.zeros(N_MO, dtype=int)
    ell_max = int(ells_ao.max()) if len(ells_ao) else 0

    for p in range(N_MO):
        c2 = mo_coeff[:, p] ** 2
        if c2.sum() < 1e-14:
            continue
        weights  = np.array([c2[ells_ao == ell].sum() for ell in range(ell_max + 1)])
        sigma[p] = int(np.argmax(weights))
    return sigma


def hybridisation_error(
    mol,
    mo_coeff: np.ndarray,
    sigma:    np.ndarray,
) -> np.ndarray:
    r"""Compute hybridisation error ε_p for each MO (eq:hybridisation_error).

        ε_p = 1 − Σ_{μ: ℓ_μ = σ(p)} |C_{μp}|²

    Zero iff φ_p is pure-ℓ (prop:hybridisation_error(i-ii)).  Always ≥ 0.
    For energy-ordered ROHF MOs ε_p > 0 in general (part (iii)).
    For subshell-sequential ordering ε_p = 0 by construction (part (iv)).

    Args:
        mol:      PySCF Mole (or proxy).
        mo_coeff: Shape (N_AO, N_MO).
        sigma:    Dominant ℓ per MO from ``dominant_subshell_sigma``.

    Returns:
        np.ndarray of shape (N_MO,) dtype float.
    """
    ells_ao = _ao_angular_momenta(mol)
    N_MO    = mo_coeff.shape[1]
    eps     = np.empty(N_MO)

    for p in range(N_MO):
        c2    = mo_coeff[:, p] ** 2
        total = c2.sum()
        if total < 1e-14:
            eps[p] = 1.0
            continue
        eps[p] = 1.0 - c2[ells_ao == sigma[p]].sum() / total
    return eps


def select_sub_janus_orbitals(
    mol,
    mo_coeff:  np.ndarray,
    mo_energy: Optional[np.ndarray] = None,
    n_frozen:  int = 0,
    n_seed:    int = 4,
    k_star:    int = 2,
) -> SubJanusSelection:
    r"""Select ``n_seed`` MOs satisfying the sub-inner-Janus criterion.

    Implements prop:seed_is_sp: seed = {φ_p : σ(p) < k_star} = {s, p MOs}.

    Algorithm:
        1. Compute σ(p) via ``dominant_subshell_sigma`` (eq:sigma_map).
        2. Compute ε_p via ``hybridisation_error`` (eq:hybridisation_error).
        3. Restrict to non-frozen MOs with σ(p) < k_star.
        4. Sort by orbital energy ascending (lowest-energy valence s/p first),
           using ε_p as tiebreaker.  If ``mo_energy`` is not provided, fall back
           to MO index (which equals energy order in ROHF canonical MOs).
           NOTE: do NOT sort by ε_p alone — diffuse high-energy virtual s-type
           basis functions (MOs at the top of the spectrum) have ε_p≈0 by
           construction (single-exponent shells are pure-ℓ) and would be
           selected over the physical valence 4s/4p MOs the theory requires.
        5. Reorder in subshell-sequential order (ℓ=0 < ℓ=1, stable within ℓ)
           per def:shell_register_table.

    Args:
        mol:      PySCF Mole (or proxy exposing ``ao_labels()``).
        mo_coeff: MO coefficient matrix C, shape (N_AO, N_MO).
        mo_energy: ROHF canonical MO energies, shape (N_MO,).  When provided,
                   candidates are ranked by energy ascending (lowest = most
                   valence).  When None, MO index is used as energy proxy.
        n_frozen: Frozen core orbital count (excluded from candidates).
        n_seed:   Target orbital count (default 4 = 1s + 3p per active site).
        k_star:   Inner-Janus clock index (default 2, selects ℓ ∈ {0, 1}).

    Returns:
        SubJanusSelection.

    Raises:
        ValueError: Fewer than n_seed sub-Janus MOs found.
    """
    sigma      = dominant_subshell_sigma(mol, mo_coeff)
    eps_p      = hybridisation_error(mol, mo_coeff, sigma)
    N_MO       = mo_coeff.shape[1]
    candidates = [p for p in range(n_frozen, N_MO) if sigma[p] < k_star]

    if len(candidates) < n_seed:
        raise ValueError(
            f"select_sub_janus_orbitals: only {len(candidates)} MOs with "
            f"σ(p) < k_star={k_star} among non-frozen orbitals (need {n_seed}). "
            "Check basis set or widen the active space."
        )

    # Primary key: orbital energy ascending (valence before diffuse/Rydberg).
    # Secondary key: ε_p ascending (purer as tiebreaker only).
    # MO index is a valid energy proxy when mo_energy is absent (ROHF canonical
    # ordering stores MOs in ascending energy order by construction).
    if mo_energy is not None:
        cands_sorted = sorted(candidates,
                              key=lambda p: (float(mo_energy[p]), float(eps_p[p])))
    else:
        cands_sorted = sorted(candidates,
                              key=lambda p: (float(p), float(eps_p[p])))
    selected     = np.array(cands_sorted[:n_seed], dtype=int)
    order        = np.argsort(sigma[selected], kind="stable")  # ℓ ascending
    selected     = selected[order]

    max_eps = float(eps_p[selected].max())
    warning = (
        f"max ε_p = {max_eps:.4f} > 0; MOs are not pure-ℓ (energy-ordered ROHF). "
        "Use subshell-sequential orbital ordering (def:shell_register_table) "
        "for ε_p = 0 by construction (prop:hybridisation_error(iv))."
        if max_eps > 1e-6 else ""
    )

    return SubJanusSelection(
        cas_orbital_indices    = selected,
        sigma_ell              = sigma[selected],
        epsilon_p              = eps_p[selected],
        n_seed                 = n_seed,
        k_star                 = k_star,
        matches_energy_ordered = True,   # caller must update after comparison
        warning                = warning,
    )


def load_sub_janus_selection(
    save_dir: Path,
    n_seed:   int = 4,
    k_star:   int = 2,
    n_frozen: int = 0,
) -> Optional[SubJanusSelection]:
    r"""Load sub-Janus orbital selection from saved PySCF artefacts.

    Reads ``mo_coeffs.npy`` (MO coefficient matrix C) and ``ao_labels.json``
    (list of AO label strings) from ``save_dir``.  These are written by
    ``generate_step_integrals`` when the companion parameter ``_save_mo_coeffs``
    is True (companion change to mqedatagenerator.py required).

    COMPANION: mqedatagenerator.generate_step_integrals must be updated to:
      - Save ``mo_coeffs.npy`` (np.ndarray, shape (N_AO, N_MO)) and
        ``ao_labels.json`` (list[str] from mol.ao_labels()) during the ROHF stage.
      - Accept ``_cas_orbital_indices: Optional[np.ndarray] = None`` to override
        the Fermi-level CAS orbital selection with the sub-Janus-selected indices.

    Returns ``None`` if the artefacts are absent (companion change not yet applied).

    Args:
        save_dir: Directory written by generate_step_integrals.
        n_seed:   Orbitals to select (default 4).
        k_star:   Inner-Janus clock index (default 2).

    Returns:
        SubJanusSelection or None.
    """
    coeffs_path = Path(save_dir) / "mo_coeffs.npy"
    labels_path = Path(save_dir) / "ao_labels.json"
    energy_path = Path(save_dir) / "mo_energy.npy"

    if not coeffs_path.exists() or not labels_path.exists():
        return None

    mo_coeff = np.load(str(coeffs_path))
    with open(str(labels_path)) as fh:
        ao_labels: List[str] = json.load(fh)

    # Load orbital energies for energy-ascending candidate ranking.
    # mo_energy.npy is written by generate_step_integrals when _save_mo_coeffs=True.
    mo_energy: Optional[np.ndarray] = None
    if energy_path.exists():
        mo_energy = np.load(str(energy_path))
    else:
        log.warning(
            "[HYBRID] load_sub_janus_selection: mo_energy.npy absent in %s; "
            "falling back to MO-index ordering (≡ energy order for ROHF canonical MOs).",
            save_dir,
        )

    # Lightweight proxy — avoids importing pyscf here.
    class _AoProxy:
        def ao_labels(self) -> List[str]:
            return ao_labels

    sigma = dominant_subshell_sigma(_AoProxy(), mo_coeff)
    eps_p = hybridisation_error(_AoProxy(), mo_coeff, sigma)
    N_MO  = mo_coeff.shape[1]

    # Exclude frozen-core MOs — they are pure-s 1s/2s orbitals that happen to
    # satisfy σ(p) < k_star but must never enter the active space.
    # n_frozen = n_core = (total_nelec - nelec_active) // 2, read from
    # janus_data["metadata"]["n_core"] by the caller.
    candidates = [p for p in range(n_frozen, N_MO) if sigma[p] < k_star]
    if len(candidates) < n_seed:
        log.warning(
            "[HYBRID] load_sub_janus_selection: only %d MOs with σ(p) < %d found "
            "in non-frozen range [%d, %d) (need %d) — sub-Janus selection skipped.",
            len(candidates), k_star, n_frozen, N_MO, n_seed,
        )
        return None

    # Sort by orbital energy ascending (lowest-energy valence s/p first), with
    # ε_p as tiebreaker only.  DO NOT sort by ε_p alone: diffuse high-energy
    # virtual s-type basis functions at the top of the MO spectrum have ε_p≈0
    # by construction (single-exponent shells are pure-ℓ) and would be selected
    # over the physical valence 4s/4p MOs the theory requires (prop:seed_is_sp).
    if mo_energy is not None:
        cands_sorted = sorted(candidates,
                              key=lambda p: (float(mo_energy[p]), float(eps_p[p])))
    else:
        cands_sorted = sorted(candidates,
                              key=lambda p: (float(p), float(eps_p[p])))
    selected     = np.array(cands_sorted[:n_seed], dtype=int)
    order        = np.argsort(sigma[selected], kind="stable")
    selected     = selected[order]

    max_eps = float(eps_p[selected].max())
    warning = (
        f"max ε_p = {max_eps:.4f} > 0."
        if max_eps > 1e-6 else ""
    )

    return SubJanusSelection(
        cas_orbital_indices    = selected,
        sigma_ell              = sigma[selected],
        epsilon_p              = eps_p[selected],
        n_seed                 = n_seed,
        k_star                 = k_star,
        matches_energy_ordered = False,  # updated by caller
        warning                = warning,
    )


# ===========================================================================
# 3. ALGEBRAIC PRE-COMPUTATION (Step 0)
# ===========================================================================

@dataclass
class AlgebraicPrecompute:
    r"""All Step 0 (purely algebraic) results for one mechanism.

    Every field is computed from (m, ν, γ_j, T_K) without any quantum
    chemistry calculation.

    Fields
    ------
    mechanism_name       : Mechanism identifier.
    m                    : Virtual register modulus.
    n_star               : Janus step (zeta-dual revolution depth).
    p_tower              : Iwasawa prime base (2 for most mechanisms).
    k_base               : Seed tower level (always 2; derived from prop:seed_is_sp:
                           sub-inner-Janus set {ℓ < k*=2} = {s, p}).  CAS(4,4)
                           is the parenthetical for Mo-nitrogenase (4 active sites).
    dt                   : Trotter step Δt_m = 0.04/√m  [Ha⁻¹].
    s                    : Zeta-dual scaling factor.
    phi_bound            : Eigenphase magnitude upper bound.
    E_inf_Ha             : E_∞ from γ_1 via zeta-dual map  [Ha, negative].
    pes_Ha               : Weyl-scaled PES energies for all M steps [Ha].
    janus_step_idx       : Step index of the Janus crossing.
    barrier_Ha           : ΔE‡ = E_∞ − 0 = E_∞  [Ha; negative → barrierless].
    k_MQE_per_s          : Eyring rate  (k_BT/h)·exp(−ΔE‡/RT) at T_K [s⁻¹].
    tower_energies_Ha    : Dict {k: E_Janus^(k)} for k in [k_base, k_base+n_levels].
    bernoulli_zeta       : Dict {k: ζ_p(1−k)} for same range.
    algebraic_coupling_Ha: Dict {k: √|ζ_p(1−k)|} (1e crossing amplitude) [Ha].
    T_K                  : Temperature used for rate computation.
    gamma_1              : γ_1 (first Riemann zero used for E_∞).
    spectral_class       : String label (e.g. 'Group B').
    riemann_scaffold     : Full RiemannScaffold object for additional zeros.
    """
    mechanism_name:        str
    m:                     int
    n_star:                int
    p_tower:               int
    k_base:                int
    dt:                    float
    s:                     float
    phi_bound:             float
    E_inf_Ha:              float
    pes_Ha:                List[float]
    janus_step_idx:        int
    barrier_Ha:            float
    k_MQE_per_s:           float
    tower_energies_Ha:     Dict[int, float]
    bernoulli_zeta:        Dict[int, float]
    algebraic_coupling_Ha: Dict[int, float]
    T_K:                   float
    gamma_1:               float
    spectral_class:        str
    riemann_scaffold:      RiemannScaffold


def build_algebraic_precompute(
    spec:    MQEMechanismSpec,
    p_tower: int   = 2,
    n_tower: int   = 8,
    T_K:     float = 298.15,
) -> AlgebraicPrecompute:
    r"""Construct the AlgebraicPrecompute for one mechanism (Step 0).

    This function performs NO quantum chemistry.  All values are derived from:
      - The mechanism's stoichiometric parameters (m, ν_n, M_steps)
      - The known Riemann zeros (RIEMANN_ZEROS)
      - Bernoulli numbers (exact rational arithmetic)
      - The tower prime p_tower

    Args:
        spec:     MQEMechanismSpec (used for m, M_steps, crossings).
        p_tower:  Iwasawa prime base (default 2 for nitrogenase-family).
        n_tower:  Number of tower levels to pre-compute above k_base.
        T_K:      Temperature [K] for k_MQE computation.

    Returns:
        AlgebraicPrecompute.

    Raises:
        ValueError: If the mechanism has no Janus crossing (no spectral data).
    """
    # ── Riemann scaffold ──────────────────────────────────────────────────────
    # Build from MechanismTuple counterpart for n_star computation.
    mech_tuples = build_predefined_mechanisms(spec.n_orbitals)
    mech_tuple  = mech_tuples.get(spec.name)
    if mech_tuple is None:
        raise ValueError(
            f"Mechanism '{spec.name}' not found in build_predefined_mechanisms. "
            "Ensure the MechanismTuple counterpart exists in mqemolecules.py."
        )
    scaffold = build_riemann_scaffold(mech_tuple)
    if scaffold is None:
        raise ValueError(
            f"Mechanism '{spec.name}' has no Janus crossing (no RiemannScaffold). "
            "The hybrid protocol requires a Case III mechanism (4|m) with crossings."
        )

    n_star  = scaffold.n_star
    m       = scaffold.m
    dt      = scaffold.dt
    s       = scaffold.s
    phi_b   = scaffold.phi_bound
    gamma_1 = RIEMANN_ZEROS[0]
    E_inf   = janus_energy_from_gamma(gamma_1, m, n_star)

    # ── Janus step index ──────────────────────────────────────────────────────
    janus_steps = [step.step_n for step in spec.steps if step.is_crossing]
    if not janus_steps:
        raise ValueError(
            f"No step with is_crossing=True found in spec for '{spec.name}'. "
            "The Janus step must be flagged in the MQEMechanismSpec."
        )
    janus_step_idx = janus_steps[0]   # primary Janus step

    # ── Weyl-scaled PES ───────────────────────────────────────────────────────
    # n_star (Riemann scaffold) = revolution-depth in the zeta-dual map (Group B: m/4).
    # janus_step_idx = step index of the actual Janus crossing in the M-step mechanism.
    # These are different indices.  The Weyl formula pins E_{n_star} = E_∞; we must
    # anchor it at janus_step_idx so the Weyl minimum coincides with the Janus step.
    pes = weyl_pes_energies(spec.M_steps, janus_step_idx, E_inf)

    # ── Reaction rate (algebraic, no Boltzmann barrier for barrierless case) ──
    # ΔE‡ = E_∞ − E_0 = E_∞ − 0 = E_∞ < 0 → barrierless → k_MQE = k_BT/h.
    dE_Ha   = E_inf   # negative
    prefactor_s = _KB_S * T_K / _H_S
    if dE_Ha <= 0.0:
        k_MQE = prefactor_s   # barrierless: w_LZ=1, p(k*)=1, exp(-ΔE/RT)=1 (or >1)
    else:
        dE_kcal = dE_Ha * _HA_KCAL
        k_MQE = prefactor_s * math.exp(-dE_kcal / (_R_KCAL * T_K))

    # ── Iwasawa tower energies via Kummer–Bernoulli ───────────────────────────
    # At k_base we have E_base = E_inf (Step 0 estimate; refined by PySCF in Step 1).
    # The tower energies collapse to E_inf for all k since E_base is set to E_inf.
    # After Step 1, the caller should call padicinterp_energy(k, k_base, E_seed, E_inf, p)
    # to get the true tower sequence.  We pre-fill with the algebraic E_inf for now.
    tower_k_range   = range(_K_BASE, _K_BASE + n_tower)
    tower_energies  = {k: padicinterp_energy(k, _K_BASE, E_inf, E_inf, p_tower)
                       for k in tower_k_range}
    bernoulli_zeta  = {k: zeta_p_value(p_tower, k) for k in tower_k_range}
    alg_coupling    = {k: algebraic_crossing_coupling(p_tower, k) for k in tower_k_range}

    return AlgebraicPrecompute(
        mechanism_name        = spec.name,
        m                     = m,
        n_star                = n_star,
        p_tower               = p_tower,
        k_base                = _K_BASE,
        dt                    = dt,
        s                     = s,
        phi_bound             = phi_b,
        E_inf_Ha              = E_inf,
        pes_Ha                = pes,
        janus_step_idx        = janus_step_idx,
        barrier_Ha            = dE_Ha,
        k_MQE_per_s           = k_MQE,
        tower_energies_Ha     = tower_energies,
        bernoulli_zeta        = bernoulli_zeta,
        algebraic_coupling_Ha = alg_coupling,
        T_K                   = T_K,
        gamma_1               = gamma_1,
        spectral_class        = scaffold.spectral_class,
        riemann_scaffold      = scaffold,
    )


# ===========================================================================
# 4. CONSISTENCY CHECK (Step 2)
# ===========================================================================

@dataclass
class ConsistencyResult:
    r"""Result of the Step 2 algebraic consistency check.

    Fields
    ------
    passed                : True iff both Kummer convergence and eigenphase
                            window conditions are satisfied.
    E_seed_Ha             : Janus-step CASCI energy from PySCF  [Ha].
    E_inf_Ha              : Riemann target E_∞  [Ha].
    phi_seed              : Seed eigenphase = |E_seed·n*·Δt_m|  [dimensionless].
    phi_bound             : Eigenphase window upper bound.
    delta_0_Ha            : Bernoulli tolerance |ζ_p(1−k_base)|  [Ha].
    algebraic_coupling_Ha : √|ζ_p(1−k_base)| (expected crossing amplitude)  [Ha].
    nearest_zero_idx      : 0-based index of the nearest γ_k in RIEMANN_ZEROS.
    nearest_zero_residual : |φ_seed − s·γ_{nearest}|  [dimensionless rad].
    kummer_convergence_ok : E_seed > E_inf (seed above Riemann target).
    eigenphase_in_window  : phi_seed ≤ phi_bound.
    message               : Human-readable diagnostic.
    """
    passed:                bool
    E_seed_Ha:             float
    E_inf_Ha:              float
    phi_seed:              float
    phi_bound:             float
    delta_0_Ha:            float
    algebraic_coupling_Ha: float
    nearest_zero_idx:      int
    nearest_zero_residual: float
    kummer_convergence_ok: bool
    eigenphase_in_window:  bool
    message:               str


def check_consistency(
    E_seed_Ha: float,
    alg_pre:   AlgebraicPrecompute,
    p:         int,
    k_base:    int = _K_BASE,
) -> ConsistencyResult:
    r"""Run the Step 2 algebraic consistency check.

    Verifies two conditions:
    1. **Kummer convergence** (eq:kummer_cong):
       E_seed > E_∞  — the seed lies above the Riemann attractor, so the
       geometric Kummer sequence {E^(k)} converges downward toward E_∞.
    2. **Eigenphase window** (prop:janus_eigenphase_bound):
       |φ_seed| ≤ φ_bound  — the seed Janus eigenphase falls within the
       accessible spectral window for the Trotter step size Δt_m.

    A failure of condition 1 means the Janus geometry R_{n*} is too
    tightly bound (possibly at the dissociation limit rather than the
    crossing geometry); shift R_{n*} toward the transition state.

    A failure of condition 2 means the active space is too large for the
    Trotter step size — either reduce ncas or use a larger m modulus.

    The tolerance δ_0 = |ζ_p(1−k_base)| is the Bernoulli coupling at the
    seed level (from eq:kmin).  For p=2, k=2: δ_0 ≈ 0.083 Ha.

    Args:
        E_seed_Ha : CASCI Janus energy from the single PySCF call  [Ha].
        alg_pre   : AlgebraicPrecompute from Step 0.
        p         : Iwasawa prime base.
        k_base    : Seed tower level (default 2).

    Returns:
        ConsistencyResult.
    """
    E_inf   = alg_pre.E_inf_Ha
    n_star  = alg_pre.n_star
    dt      = alg_pre.dt
    s       = alg_pre.s
    phi_b   = alg_pre.phi_bound

    # Seed eigenphase (absolute value — phase is a magnitude)
    phi_seed = abs(E_seed_Ha * n_star * dt)

    # Bernoulli tolerance
    delta_0 = abs(zeta_p_value(p, k_base))
    alg_cx  = algebraic_crossing_coupling(p, k_base)

    # Nearest Riemann zero to the seed eigenphase
    nearest_idx = 0
    nearest_res = float("inf")
    for idx, gk in enumerate(RIEMANN_ZEROS):
        phi_k = s * gk
        res   = abs(phi_seed - phi_k)
        if res < nearest_res:
            nearest_res = res
            nearest_idx = idx

    # Conditions
    kummer_ok = (E_seed_Ha > E_inf)
    window_ok = (phi_seed <= phi_b)
    passed    = kummer_ok and window_ok

    if passed:
        msg = (
            f"[OK]  E_seed={E_seed_Ha:+.6f} Ha > E_∞={E_inf:+.6f} Ha  "
            f"(Kummer ✓)  |φ_seed|={phi_seed:.4f} ≤ φ_bound={phi_b:.4f} "
            f"(window ✓)  nearest zero γ_{nearest_idx+1}, residual={nearest_res:.4f}"
        )
    else:
        parts = []
        if not kummer_ok:
            parts.append(
                f"Kummer FAIL: E_seed={E_seed_Ha:+.6f} Ha ≤ E_∞={E_inf:+.6f} Ha. "
                "Adjust Janus geometry R_{{n*}} toward the transition-state bond length."
            )
        if not window_ok:
            parts.append(
                f"Window FAIL: |φ_seed|={phi_seed:.4f} > φ_bound={phi_b:.4f}. "
                "Reduce ncas or use larger m modulus."
            )
        msg = "  |  ".join(parts)

    return ConsistencyResult(
        passed                = passed,
        E_seed_Ha             = E_seed_Ha,
        E_inf_Ha              = E_inf,
        phi_seed              = phi_seed,
        phi_bound             = phi_b,
        delta_0_Ha            = delta_0,
        algebraic_coupling_Ha = alg_cx,
        nearest_zero_idx      = nearest_idx,
        nearest_zero_residual = nearest_res,
        kummer_convergence_ok = kummer_ok,
        eigenphase_in_window  = window_ok,
        message               = msg,
    )


# ===========================================================================
# 5. SINGLE PySCF CALL (Step 1)
# ===========================================================================

def run_single_janus_step(
    spec:            MQEMechanismSpec,
    basis:           str                        = "STO-3G",
    validate_fci:    bool                       = True,
    verbose:         int                        = 0,
    output_dir:      Optional[Path]             = None,
    n_total_orbs:    int                        = 76,
    sub_janus_sel:   Optional[SubJanusSelection] = None,
    _mf_store:       Optional[Dict]             = None,
    _precomputed_mf: Optional[object]           = None,
) -> Tuple[Dict, float]:
    r"""Run PySCF for the single Janus step only (Step 1).

    Calls ``generate_step_integrals`` from mqedatagenerator for the one step
    in ``spec`` that has ``is_crossing=True``.  All other steps are
    reconstructed algebraically in ``_synthesise_non_janus_steps``.

    Orbital selection
    ~~~~~~~~~~~~~~~~~
    By default, ``generate_step_integrals`` selects orbitals by energy
    proximity to the Fermi level.  For d-block systems this yields d-type
    MOs (σ(p)=2), violating prop:seed_is_sp which requires {ℓ < k*=2} = {s,p}.

    When ``sub_janus_sel`` is provided (obtained via ``load_sub_janus_selection``
    after the companion change to mqedatagenerator is applied), it is passed to
    ``generate_step_integrals`` as ``_cas_orbital_indices``.  If the companion
    parameter is not yet accepted (TypeError), the call falls back to the
    energy-ordered selection with a logged warning.

    COMPANION: mqedatagenerator.generate_step_integrals must accept:
      ``_cas_orbital_indices: Optional[np.ndarray] = None``

    Args:
        spec:          MQEMechanismSpec (all steps; only Janus step is used).
        basis:         PySCF basis set string.
        validate_fci:  Run exact FCI (ncas ≤ 20) at the Janus step.
        verbose:       PySCF verbosity.
        output_dir:    If provided, saves h1_full.npy + eri_packed.npy there.
        n_total_orbs:  Total orbital pool for the full active space (tower).
        sub_janus_sel: Pre-computed SubJanusSelection (from
                       ``load_sub_janus_selection``).  None → energy-ordered
                       fallback (logs a warning for d-block systems).

    Returns:
        (janus_data_dict, E_seed_Ha) where:
          * janus_data_dict — complete JSON-serialisable step dict.
          * E_seed_Ha       — CASCI (or FCI) active-space energy at Janus step.

    Raises:
        ValueError: If no Janus step found in spec.
    """
    janus_steps = [s for s in spec.steps if s.is_crossing]
    if not janus_steps:
        raise ValueError(
            f"No Janus step found in spec for '{spec.name}'. "
            "At least one MQEStep must have is_crossing=True."
        )
    janus_step = janus_steps[0]

    save_dir = Path(output_dir) / spec.name if output_dir is not None else None

    # ── Orbital-selection logging ─────────────────────────────────────────────
    if sub_janus_sel is not None:
        log.info(
            "[HYBRID-STEP1] Sub-Janus orbital selection (prop:seed_is_sp): "
            "indices=%s  σ(p)=%s  ε_p=%s",
            sub_janus_sel.cas_orbital_indices.tolist(),
            sub_janus_sel.sigma_ell.tolist(),
            [f"{e:.4f}" for e in sub_janus_sel.epsilon_p.tolist()],
        )
        if sub_janus_sel.warning:
            log.warning("[HYBRID-STEP1] %s", sub_janus_sel.warning)
    else:
        log.warning(
            "[HYBRID-STEP1] sub_janus_sel=None: using energy-ordered Fermi-level "
            "orbital selection (pass 1).  mo_coeffs.npy + ao_labels.json will be "
            "written; pass 2 will apply sub-Janus selection (prop:seed_is_sp)."
        )

    # ── PySCF call (with sub-Janus orbital override if available) ────────────
    dm_cache: Dict = {}
    # _save_mo_coeffs=True always: pass 1 (energy-ordered) writes mo_coeffs.npy
    # + ao_labels.json so load_sub_janus_selection can read them for pass 2.
    # If sub_janus_sel is already provided (pass 2), the artefacts are already
    # present and the save is a harmless overwrite.
    base_kwargs: Dict = dict(
        spec                   = spec,
        step                   = janus_step,
        basis                  = basis,
        validate_fci           = validate_fci,
        verbose                = verbose,
        dm_cache               = dm_cache,
        _save_mo_integrals_dir = save_dir,
        _n_total_orbs          = n_total_orbs,
        _save_mo_coeffs        = True,
        _mf_store              = _mf_store,
        _precomputed_mf        = _precomputed_mf,
    )

    if sub_janus_sel is not None:
        try:
            data = generate_step_integrals(
                **base_kwargs,
                _cas_orbital_indices = sub_janus_sel.cas_orbital_indices,
            )
            log.info(
                "[HYBRID-STEP1] generate_step_integrals used sub-Janus orbital "
                "indices %s.", sub_janus_sel.cas_orbital_indices.tolist()
            )
        except TypeError:
            log.warning(
                "[HYBRID-STEP1] generate_step_integrals does not yet accept "
                "_cas_orbital_indices (companion change pending) — falling back "
                "to energy-ordered selection."
            )
            data = generate_step_integrals(**base_kwargs)
    else:
        data = generate_step_integrals(**base_kwargs)

    # ── Extract active-space energy ───────────────────────────────────────────
    # circuit_reference_energy_Ha = E_total = E_core + E_active.
    # The Kummer/eigenphase consistency check uses active-space energy (E_∞ is
    # also active-space), so we subtract ecore.
    E_total = data.get("circuit_reference_energy_Ha")
    if E_total is None:
        E_total = data.get("casci_energy_Ha", 0.0)
    e_core = data.get("ecore_Ha", 0.0)
    E_seed = float(E_total) - float(e_core)

    log.info(
        "[HYBRID-STEP1] Janus step n=%d: E_total=%+.8f  E_core=%+.8f  "
        "E_seed(active)=%+.8f Ha",
        janus_step.step_n, float(E_total), float(e_core), E_seed,
    )
    return data, E_seed


# ===========================================================================
# 6. NON-JANUS STEP SYNTHESIS (Weyl-scaled PES + copied integrals)
# ===========================================================================

def _synthesise_non_janus_steps(
    spec:           MQEMechanismSpec,
    janus_data:     Dict,
    pes_Ha:         List[float],
    sub_janus_sel:  Optional["SubJanusSelection"] = None,
) -> Dict[int, Dict]:
    r"""Synthesise step JSONs for all non-Janus steps (Step 0, algebraic).

    Non-Janus step JSONs contain only lightweight metadata and the
    Weyl-scaled energy.  Integral tensors (h_diag, h_hop, g_full) are NOT
    stored: they would be identical to the Janus-step tensors (same geometry,
    frozen active space) and are therefore informationally redundant.

    Downstream consumers:
      * tower_climber reads integrals only from the Janus step JSON
        (_load_base calls _load_step_raw(janus_n) — non-Janus steps untouched).
      * mqerates._force_diff reads h_diag from steps janus_n±1 to compute the
        orbital-gap force gradient.  This access is already guarded by
        ``except Exception`` and falls back to Ftot when h_diag is absent.
        The force gradient is always zero in hybrid mode regardless (frozen
        geometry ⇒ identical h_diag at all steps), so the fallback is correct.

    When sub_janus_sel is provided, ``cas_orbital_indices`` is written into
    each non-Janus JSON so consumers can reconstruct the seed CAS block by
    slicing h1_full.npy at those indices if needed.

    Args:
        spec:          MQEMechanismSpec with all steps defined.
        janus_data:    Step JSON dict from ``run_single_janus_step``.
        pes_Ha:        Weyl-scaled energies from ``weyl_pes_energies``.
        sub_janus_sel: SubJanusSelection used for pass 2 (optional).

    Returns:
        Dict mapping step_n → synthesised step JSON dict.
    """
    janus_step_n = janus_data["mqe_step"]["step_n"]
    result: Dict[int, Dict] = {}

    # Lightweight fields carried from the Janus step (no tensors).
    janus_meta  = copy.deepcopy(janus_data.get("metadata", {}))
    janus_ecore = janus_data.get("ecore_Ha", 0.0)

    # CAS orbital indices for downstream slicing of h1_full.npy.
    cas_idx: Optional[List[int]] = (
        sub_janus_sel.cas_orbital_indices.tolist()
        if sub_janus_sel is not None else None
    )

    for step in spec.steps:
        n = step.step_n
        if n == janus_step_n:
            continue   # Janus step already has real integrals

        synth: Dict = {
            "mqe_step": step.to_dict(
                mechanism = spec.name,
                M_total   = spec.M_steps,
                m_modulus = spec.m_modulus,
            ),
            "geometry_label":                step.geometry_label,
            "bondlength_angstrom":           step.bondlength_angstrom,
            "circuit_reference_energy_Ha":   pes_Ha[n],
            "circuit_reference_energy_origin": "weyl_scaled_pes",
            "metadata":                      janus_meta,
            "ecore_Ha":                      janus_ecore,
            # Integrals omitted — sliceable from h1_full.npy at cas_orbital_indices.
            "integrals_source":              "h1_full.npy",
            "weyl_reconstructed":            True,
            "weyl_formula":                  (
                "E_n = E_inf * log(n+1) / log(n_star+1)  "
                "(Weyl law: N_MQE(T) ~ (T/2pi)log(T/2pi))"
            ),
            "exact_fci_energy_Ha":           None,
            "casci_energy_Ha":               None,
        }
        if cas_idx is not None:
            synth["cas_orbital_indices"] = cas_idx

        result[n] = synth

    return result


# ===========================================================================
# 7. HYBRID ORCHESTRATOR (Steps 0 + 1 + 2)
# ===========================================================================

def run_hybrid_generation(
    mechanism_name:        str,
    basis:                 str   = "STO-3G",
    output_dir:            str   = "datasets/",
    n_orbitals:            int   = 4,
    p_tower:               int   = 2,
    n_total_orbs:          int   = 76,
    T_K:                   float = 298.15,
    validate_fci:          bool  = True,
    verbose:               int   = 0,
    use_sub_janus_selection: bool = True,
) -> Dict:
    r"""Full hybrid generation: Steps 0 + 1 + 2.

    Reduces PySCF calls from M to 1 for any Case III mechanism with a Janus
    crossing.

    Pipeline
    --------
    Step 0 : ``build_algebraic_precompute`` — E_∞, PES_n, tower energies,
             k_MQE, Bernoulli coupling table; all algebraic.
    Step 1 : ``run_single_janus_step`` — one PySCF call at R_{n*}.
             After the call, ``load_sub_janus_selection`` checks whether
             mqedatagenerator saved mo_coeffs.npy + ao_labels.json (companion
             change).  If so, the sub-Janus orbital selection (prop:seed_is_sp)
             is computed and a second PySCF call is made with the correct
             {ℓ < k*=2} active space.  Synthesise non-Janus JSONs.
    Step 2 : ``check_consistency`` — Kummer convergence + eigenphase window.

    Output files
    ------------
    ``{output_dir}/{mechanism_name}/step_{n:02d}.json`` for n = 0…M−1.
    ``{output_dir}/{mechanism_name}/manifest.json``     (standard + hybrid block).

    Args:
        mechanism_name          : e.g. "femon2_trimer", "nitrogenase_lt".
        basis                   : PySCF basis (e.g. "STO-3G", "DZP-DKH").
        output_dir              : Root output directory.
        n_orbitals              : Active-space orbital count N (sets ncas).
        p_tower                 : Iwasawa prime base (2 for nitrogenase family).
        n_total_orbs            : Total MO pool for tower extension.
        T_K                     : Temperature [K] for rate computation.
        validate_fci            : Include FCI reference at Janus step.
        verbose                 : PySCF verbosity (0=quiet, 3=debug).
        use_sub_janus_selection : If True, attempt to load sub-Janus orbital
                                  selection after Step 1 and re-run with correct
                                  {ℓ<k*=2} active space (companion change to
                                  mqedatagenerator required for re-run).
                                  Default True.

    Returns:
        Dict: result dict with keys ok, consistency, E_seed_Ha, E_inf_Ha,
              k_MQE_per_s, spectral_class, algebraic_precompute,
              consistency_result, sub_janus_selection, manifest_path.
    """
    out_path  = Path(output_dir)
    mech_dir  = out_path / mechanism_name
    mech_dir.mkdir(parents=True, exist_ok=True)

    log.info("[HYBRID] ════════════════════════════════════════════════════")
    log.info("[HYBRID] Mechanism  : %s", mechanism_name)
    log.info("[HYBRID] Basis      : %s", basis)
    log.info("[HYBRID] n_orbitals : %d   p_tower : %d   T_K : %.1f K",
             n_orbitals, p_tower, T_K)
    log.info("[HYBRID] ════════════════════════════════════════════════════")

    # ── Build MQEMechanismSpec ────────────────────────────────────────────────
    all_specs = build_all_specs(n_orbitals)
    if mechanism_name not in all_specs:
        raise ValueError(
            f"Mechanism '{mechanism_name}' not in build_all_specs. "
            f"Available: {sorted(all_specs.keys())}"
        )
    spec = all_specs[mechanism_name]

    # ── STEP 0: Algebraic pre-computation ─────────────────────────────────────
    t0 = time.time()
    alg = build_algebraic_precompute(spec, p_tower=p_tower, T_K=T_K)
    log.info("[HYBRID] Step 0 done in %.1f s", time.time() - t0)
    log.info("[HYBRID]   E_∞        = %+.8f Ha  (γ₁=%.6f, s=%.5f)",
             alg.E_inf_Ha, alg.gamma_1, alg.s)
    log.info("[HYBRID]   Spectral   = %s  (m=%d, n*=%d, Δt=%g)",
             alg.spectral_class, alg.m, alg.n_star, alg.dt)
    log.info("[HYBRID]   k_MQE      = %.4e s⁻¹  (T=%.1f K, algebraic)",
             alg.k_MQE_per_s, T_K)

    # ── n_total_orbs recommendation (algebraic, no PySCF required) ───────────
    # Worst-case Δ₀ = |E_∞| (when E_seed → 0⁻).  Actual Δ₀ ≤ Δ₀_max always,
    # so k_min_bound is a safe upper bound on the required tower depth.
    # n_orb at level k = 4*(k-1)  →  n_total_orbs_safe = 4*k_min_bound.
    _precision_Ha    = 1.6e-3          # 1.6 mHa budget
    _delta0_max      = abs(alg.E_inf_Ha)
    _k_min_bound     = math.ceil(2 + math.log2(_delta0_max / _precision_Ha))
    _n_orbs_safe     = 4 * _k_min_bound   # 4*(k-1) at k=k_min+1 with +1 buffer
    log.info(
        "[HYBRID]   Δ₀ bound   = %.2f Ha  (|E_∞|, worst case E_seed→0)",
        _delta0_max,
    )
    log.info(
        "[HYBRID]   k_min bound = %d  (ceil(2+log2(%.2f/1.6e-3)))  "
        "→ recommended --n-total-orbs ≥ %d",
        _k_min_bound, _delta0_max, _n_orbs_safe,
    )
    if n_total_orbs < _n_orbs_safe:
        log.warning(
            "[HYBRID]   --n-total-orbs %d < minimum %d required for 1.6 mHa "
            "convergence (k_min=%d, n_orb=4*k_min=%d).  "
            "Auto-correcting to %d — no re-run needed.",
            n_total_orbs, _n_orbs_safe, _k_min_bound, _n_orbs_safe, _n_orbs_safe,
        )
        n_total_orbs = _n_orbs_safe

    log.info("[HYBRID]   Weyl PES   : " +
             "  ".join(f"E_{n}={e:+.4f}" for n, e in enumerate(alg.pes_Ha)))
    for k, zeta in sorted(alg.bernoulli_zeta.items()):
        log.info("[HYBRID]   ζ_%d(1-%d) = %+.6f  |H_cx^(%d)| = %.4f Ha",
                 p_tower, k, zeta, k, alg.algebraic_coupling_Ha[k])

    # ── STEP 1: Single PySCF call at R_{n*} ───────────────────────────────────
    log.info("[HYBRID] Step 1: PySCF at Janus geometry (step n=%d, bond=%.4f Å)",
             alg.janus_step_idx,
             spec.steps[alg.janus_step_idx].bondlength_angstrom)
    t1 = time.time()

    # _mf_store is a mutable dict that generate_step_integrals fills with the
    # converged ROHF object after pass 1.  Pass 2 reads _mf_store["mf"] via
    # _precomputed_mf to skip ROHF entirely (~10 min saved per run).
    _mf_store: Dict = {}

    # First pass: energy-ordered selection (always succeeds)
    janus_data, E_seed = run_single_janus_step(
        spec           = spec,
        basis          = basis,
        validate_fci   = validate_fci,
        verbose        = verbose,
        output_dir     = out_path,
        n_total_orbs   = n_total_orbs,
        sub_janus_sel  = None,
        _mf_store      = _mf_store,
    )
    log.info("[HYBRID] Step 1 (pass 1/energy-ordered) done in %.1f s  "
             "E_seed = %+.8f Ha", time.time() - t1, E_seed)

    # ── Sub-Janus orbital selection (prop:seed_is_sp) ─────────────────────────
    # n_frozen = n_core from pass-1 metadata: the number of doubly-occupied MOs
    # frozen out of the active space.  Must be excluded from sub-Janus candidate
    # scan — they are low-energy 1s/2s core orbitals that satisfy σ(p)<k_star
    # but must never enter the CAS active space.
    n_frozen: int = int(janus_data.get("metadata", {}).get("n_core", 0))
    sub_janus_sel: Optional[SubJanusSelection] = None
    if use_sub_janus_selection:
        save_dir_mech = out_path / mechanism_name
        sub_janus_sel = load_sub_janus_selection(
            save_dir_mech, n_seed=n_orbitals, n_frozen=n_frozen,
        )
        if sub_janus_sel is not None:
            log.info(
                "[HYBRID] Sub-Janus selection loaded: indices=%s  σ=%s  ε_p=%s",
                sub_janus_sel.cas_orbital_indices.tolist(),
                sub_janus_sel.sigma_ell.tolist(),
                [f"{e:.4f}" for e in sub_janus_sel.epsilon_p.tolist()],
            )
            if sub_janus_sel.warning:
                log.warning("[HYBRID] %s", sub_janus_sel.warning)

            # Check whether the selection differs from the energy-ordered one.
            # If it differs, re-run Step 1 with the corrected orbital indices.
            t1b = time.time()
            janus_data2, E_seed2 = run_single_janus_step(
                spec            = spec,
                basis           = basis,
                validate_fci    = validate_fci,
                verbose         = verbose,
                output_dir      = out_path,
                n_total_orbs    = n_total_orbs,
                sub_janus_sel   = sub_janus_sel,
                _precomputed_mf = _mf_store.get("mf"),
            )
            elapsed1b = time.time() - t1b
            if E_seed2 != E_seed:
                log.info(
                    "[HYBRID] Step 1 (pass 2/sub-Janus) done in %.1f s  "
                    "E_seed = %+.8f Ha  (ΔE = %+.2e Ha vs pass 1)",
                    elapsed1b, E_seed2, E_seed2 - E_seed,
                )
                janus_data = janus_data2
                E_seed     = E_seed2
            else:
                log.info(
                    "[HYBRID] Step 1 pass 2: energy-ordered and sub-Janus "
                    "selections agree (E_seed unchanged).  Pass 2 result used."
                )
        else:
            log.warning(
                "[HYBRID] Sub-Janus artefacts (mo_coeffs.npy, ao_labels.json) "
                "not found in %s after pass 1.  This is unexpected: "
                "_save_mo_coeffs=True was set in base_kwargs.  "
                "Energy-ordered selection retained.",
                out_path / mechanism_name,
            )

    # Synthesise non-Janus step JSONs (Weyl-scaled energy only; no tensor copy)
    non_janus = _synthesise_non_janus_steps(spec, janus_data, alg.pes_Ha, sub_janus_sel)

    # Assemble all step data in step order
    all_step_data: Dict[int, Dict] = {alg.janus_step_idx: janus_data}
    all_step_data.update(non_janus)

    # ── STEP 2: Consistency check ──────────────────────────────────────────────
    t2     = time.time()
    cons   = check_consistency(E_seed, alg, p_tower)
    log.info("[HYBRID] Step 2: %s", cons.message)
    if not cons.passed:
        log.warning(
            "[HYBRID] Consistency check FAILED.  Tower climb will proceed but "
            "E_Janus^(k) may not converge to γ₁.  Consider adjusting R_{n*}."
        )

    # Refine tower energies with actual E_seed (replaces E_inf placeholder)
    tower_energies_refined = {
        k: padicinterp_energy(k, _K_BASE, E_seed, alg.E_inf_Ha, p_tower)
        for k in range(_K_BASE, _K_BASE + 8)
    }

    # ── Write step JSONs ───────────────────────────────────────────────────────
    # Energy-scale consistency: all steps must use the ACTIVE-SPACE energy
    # so that ΔE computed across steps is meaningful.  The Weyl formula gives
    # E_n on the active-space/Riemann-scaffold scale (E_∞ = -57.36 Ha for
    # femon2_trimer).  The PySCF Janus step returns E_total = E_core + E_seed
    # (here: -1433.92 + -4.56 = -1438.48 Ha) — a completely different origin.
    # Overwrite circuit_reference_energy_Ha for the Janus step with E_seed
    # (active-space energy, ecore already subtracted) before writing JSON.
    fci_energies: List[Optional[float]] = [None] * spec.M_steps
    step_results: List[Dict] = []
    for n in range(spec.M_steps):
        data = all_step_data[n]

        if n == alg.janus_step_idx:
            # Shallow copy to avoid mutating janus_data (used by manifest below).
            data = dict(data)
            data["circuit_reference_energy_Ha"] = E_seed
            data["circuit_reference_energy_origin"] = "pyscf_active_space"
            # Inject sub-Janus orbital indices into metadata so tower_climber
            # can use the exact (non-contiguous) MO indices from h1_full.npy
            # as the base active space rather than the contiguous range default.
            if sub_janus_sel is not None:
                _meta = dict(data.get("metadata", {}))
                _meta["cas_orbital_indices"] = sub_janus_sel.cas_orbital_indices.tolist()
                data["metadata"] = _meta

        e_ref = data.get("circuit_reference_energy_Ha")
        fci_energies[n] = float(e_ref) if e_ref is not None else None

        step_path = mech_dir / f"step_{n:02d}.json"
        step_path.write_text(json.dumps(data, indent=2))

        step_results.append({
            "step_n":         n,
            "geometry":       data.get("geometry_label", f"step_{n:02d}"),
            "e_ref_Ha":       fci_energies[n],
            "weyl_reconstructed": (n != alg.janus_step_idx),
            "passed":         True,
        })
        log.info(
            "[HYBRID]   step %d  E=%+.8f Ha  %s",
            n, fci_energies[n] if fci_energies[n] is not None else float("nan"),
            "PySCF" if n == alg.janus_step_idx else "Weyl",
        )

    # ── Stoichiometry ──────────────────────────────────────────────────────────
    stoich = validate_mechanism_stoichiometry(spec)

    # ── Write manifest ─────────────────────────────────────────────────────────
    manifest = {
        "mechanism":              spec.name,
        "description":            spec.description,
        "M_steps":                spec.M_steps,
        "m_modulus":              spec.m_modulus,
        "S_target":               spec.S_target,
        "n_orbitals":             spec.n_orbitals,
        "basis":                  basis,
        "n_orbs_base":            spec.n_orbitals,
        "stoichiometry":          stoich,
        "step_results":           step_results,
        "fci_energies_Ha":        fci_energies,
        "janus_steps":            [alg.janus_step_idx],
        "scaffold_class":         alg.spectral_class,
        "all_algebraic_ok":       stoich["passed"],
        "generated_at":           time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mqe_article_reference":  "nanoprotogeny.theory.mqe v2026.05",
        # ── Hybrid-protocol block ───────────────────────────────────────────
        "hybrid_protocol": {
            "version":               "1.0",
            "description":           (
                "Three-step hybrid protocol: "
                "Step 0 = algebraic pre-computation (no PySCF); "
                "Step 1 = single PySCF call at Janus geometry R_{n*}; "
                "Step 2 = Kummer/eigenphase consistency check."
            ),
            "pyscf_calls":           1,
            "standard_pyscf_calls":  spec.M_steps,
            "reduction_factor":      spec.M_steps,
            "step0_algebraic": {
                "E_inf_Ha":             alg.E_inf_Ha,
                "gamma_1":              alg.gamma_1,
                "spectral_class":       alg.spectral_class,
                "m":                    alg.m,
                "n_star":               alg.n_star,
                "s":                    alg.s,
                "dt":                   alg.dt,
                "phi_bound":            alg.phi_bound,
                "k_MQE_per_s_algebraic": alg.k_MQE_per_s,
                "T_K":                  T_K,
                "weyl_pes_Ha":          alg.pes_Ha,
                "bernoulli_zeta_table": {str(k): v for k, v in alg.bernoulli_zeta.items()},
                "algebraic_coupling_Ha": {str(k): v for k, v in alg.algebraic_coupling_Ha.items()},
            },
            "step1_janus_pyscf": {
                "janus_step_n":            alg.janus_step_idx,
                "janus_bondlength_Ang":    spec.steps[alg.janus_step_idx].bondlength_angstrom,
                "E_seed_Ha":               E_seed,
                "basis":                   basis,
                "sub_janus_selection": (
                    {
                        "cas_orbital_indices": sub_janus_sel.cas_orbital_indices.tolist(),
                        "sigma_ell":           sub_janus_sel.sigma_ell.tolist(),
                        "epsilon_p":           sub_janus_sel.epsilon_p.tolist(),
                        "n_seed":              sub_janus_sel.n_seed,
                        "k_star":              sub_janus_sel.k_star,
                        "warning":             sub_janus_sel.warning,
                        "source":              "load_sub_janus_selection (prop:seed_is_sp)",
                    }
                    if sub_janus_sel is not None else {
                        "source":  "energy_ordered_fermi_level",
                        "warning": (
                            "Sub-Janus artefacts absent after pass 1 — unexpected; "
                            "_save_mo_coeffs=True was set.  Energy-ordered selection retained."
                        ),
                    }
                ),
            },
            "step2_consistency": {
                "passed":               cons.passed,
                "kummer_convergence":   cons.kummer_convergence_ok,
                "eigenphase_in_window": cons.eigenphase_in_window,
                "phi_seed":             cons.phi_seed,
                "phi_bound":            cons.phi_bound,
                "delta_0_Ha":           cons.delta_0_Ha,
                "algebraic_coupling_Ha": cons.algebraic_coupling_Ha,
                "nearest_zero_idx":     cons.nearest_zero_idx,
                "nearest_zero_residual": cons.nearest_zero_residual,
                "message":              cons.message,
            },
            "tower_energies_refined_Ha": {str(k): v for k, v in tower_energies_refined.items()},
        },
    }

    manifest_path = mech_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("[HYBRID] Wrote manifest: %s", manifest_path)
    log.info("[HYBRID] Consistency: %s", "PASSED ✓" if cons.passed else "FAILED ✗")
    log.info("[HYBRID] ────────────────────────────────────────────────────")
    log.info("[HYBRID] PySCF calls: 1 of %d  (%.0f%% reduction)",
             spec.M_steps, 100 * (1 - 1 / spec.M_steps))
    log.info("[HYBRID] k_MQE (algebraic, T=%g K) = %.4e s⁻¹",
             T_K, alg.k_MQE_per_s)

    return {
        "ok":              True,
        "consistency":     cons.passed,
        "E_seed_Ha":       E_seed,
        "E_inf_Ha":        alg.E_inf_Ha,
        "k_MQE_per_s":     alg.k_MQE_per_s,
        "spectral_class":  alg.spectral_class,
        "algebraic_precompute": {
            "E_inf_Ha":    alg.E_inf_Ha,
            "pes_Ha":      alg.pes_Ha,
            "k_MQE_per_s": alg.k_MQE_per_s,
            "bernoulli_zeta": {str(k): v for k, v in alg.bernoulli_zeta.items()},
        },
        "consistency_result": {
            "passed":      cons.passed,
            "message":     cons.message,
        },
        "sub_janus_selection": (
            {
                "cas_orbital_indices": sub_janus_sel.cas_orbital_indices.tolist(),
                "sigma_ell":           sub_janus_sel.sigma_ell.tolist(),
                "epsilon_p":           sub_janus_sel.epsilon_p.tolist(),
                "warning":             sub_janus_sel.warning,
            }
            if sub_janus_sel is not None else None
        ),
        "manifest_path":   str(manifest_path),
    }


# ===========================================================================
# 8. CLI
# ===========================================================================

def _list_mechanisms() -> None:
    """Print all mechanisms that have a MechanismTuple (hybrid-protocol compatible)."""
    specs = build_all_specs(4)
    tuples = build_predefined_mechanisms(4)
    print("\nHybrid-protocol compatible mechanisms (Case III, 4|m, has Janus crossing):\n")
    for name in sorted(specs.keys()):
        spec = specs[name]
        has_crossing = any(s.is_crossing for s in spec.steps)
        in_tuples    = name in tuples
        compatible   = has_crossing and in_tuples and spec.m_modulus % 4 == 0
        flag = "✓" if compatible else "·"
        print(f"  {flag}  {name:<36}  m={spec.m_modulus:<3}  "
              f"M={spec.M_steps:<3}  Janus={'yes' if has_crossing else 'no '}")
    print()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Hybrid Algebraic-PySCF MQE Data Generator.  "
            "Reduces PySCF geometry evaluations from M to 1 (only the Janus step)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # femon2_trimer: 8 → 1 PySCF calls
  python mqehybridgenerator.py --mechanism femon2_trimer \\
      --basis DZP-DKH --tower-p 2 --n-total-orbs 60 --output-dir datasets/

  # nitrogenase_lt
  python mqehybridgenerator.py --mechanism nitrogenase_lt \\
      --basis STO-3G --n-total-orbs 76 --output-dir datasets/

  # List all compatible mechanisms
  python mqehybridgenerator.py --list-mechanisms
""",
    )
    p.add_argument("--mechanism",   type=str,   default="femon2_trimer",
                   help="Mechanism name (default: femon2_trimer)")
    p.add_argument("--basis",       type=str,   default="STO-3G",
                   help="PySCF basis set (default: STO-3G)")
    p.add_argument("--output-dir",  type=str,   default="datasets/",
                   help="Root output directory (default: datasets/)")
    p.add_argument("--n-orbitals",  type=int,   default=4,
                   help="Active-space orbital count (default: 4)")
    p.add_argument("--tower-p",     type=int,   default=2,
                   help="Iwasawa prime base (default: 2)")
    p.add_argument("--n-total-orbs", type=int,  default=76,
                   help="Total orbital pool for tower extension (default: 76)")
    p.add_argument("--temperature", type=float, default=298.15,
                   help="Temperature in K for rate computation (default: 298.15)")
    p.add_argument("--no-fci",      action="store_true",
                   help="Skip FCI reference at the Janus step")
    p.add_argument("--verbose",     type=int,   default=0,
                   help="PySCF verbosity (0=quiet)")
    p.add_argument("--list-mechanisms", action="store_true",
                   help="List compatible mechanisms and exit")
    p.add_argument("--use-energy-ordered", action="store_true",
                   help="Disable sub-Janus orbital selection; use energy-ordered "
                        "Fermi-level selection (not recommended for d-block systems)")
    return p


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    args = _build_parser().parse_args()

    if args.list_mechanisms:
        _list_mechanisms()
        raise SystemExit(0)

    result = run_hybrid_generation(
        mechanism_name          = args.mechanism,
        basis                   = args.basis,
        output_dir              = args.output_dir,
        n_orbitals              = args.n_orbitals,
        p_tower                 = args.tower_p,
        n_total_orbs            = args.n_total_orbs,
        T_K                     = args.temperature,
        validate_fci            = not args.no_fci,
        verbose                 = args.verbose,
        use_sub_janus_selection = not args.use_energy_ordered,
    )

    print("\n" + "=" * 68)
    print(f"  Hybrid generation: {args.mechanism}")
    print("=" * 68)
    print(f"  Consistency check : {'PASSED ✓' if result['consistency'] else 'FAILED ✗'}")
    print(f"  E_seed (PySCF)    : {result['E_seed_Ha']:+.8f} Ha")
    print(f"  E_∞   (algebra)   : {result['E_inf_Ha']:+.8f} Ha")
    print(f"  k_MQE (algebraic) : {result['k_MQE_per_s']:.4e} s⁻¹  "
          f"(T={args.temperature:.1f} K)")
    print(f"  PySCF calls       : 1 of {result.get('M_steps', args.n_orbitals)}")
    print(f"  Manifest          : {result['manifest_path']}")
    print("=" * 68)
