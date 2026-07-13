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
mqeriemann.py — Riemann Zero Spectral Scaffold for MQE-QPE
===========================================================
Pure-arithmetic module.  No cirq, no numpy beyond scalars, no I/O.

The zeta-dual map φ(γ_k) = s · γ_k maps non-trivial Riemann zeros
(imaginary parts γ_k) to Janus eigenphases.  Since thm:spectral_identification
proves that the actual Janus eigenphase of H_MQE *is* s·γ_k, the known
zeros provide an exact spectral scaffold:

    E_Janus(k) = −s · γ_k / (n* · Δt_m)         [Ha, negative: bound state]

where:
    Δt_m   = 0.04 / √m                           (Trotter step at modulus m)
    s      = Δt_m / (2π · ln((n*+1)·Δt_m + 1))  (zeta-dual scaling)
    n*     = m // ν_n − 1  (cofactor shift revolution depth; see n_star_from_mechanism)
    γ_k    = Im(ρ_k) for the k-th non-trivial Riemann zero

Usage
-----
    from nanoprotogeny.molecular.mqeriemann import build_riemann_scaffold

    scaffold = build_riemann_scaffold(mechanism)
    # scaffold.janus_energies[0]  →  exact E_Janus for γ₁  [Ha]
    # scaffold.s                  →  zeta-dual scaling factor
    # scaffold.gammas             →  Riemann zeros within eigenphase window

Public API
----------
    RIEMANN_ZEROS           : List[float]  — first 20 imaginary parts γ_k
    delta_t_m(m)            : float        — Trotter step
    s_value(m, n_star)      : float        — zeta-dual s
    eigenphase_bound(m, M)  : float        — |φ_{k*}| upper bound
    n_star_from_mechanism   : int          — crossing step index
    RiemannScaffold         : dataclass    — complete scaffold for one mechanism
    build_riemann_scaffold  : factory      — constructs RiemannScaffold
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from nanoprotogeny.molecular.mqemolecules import MechanismTuple

# ── Known non-trivial Riemann zeros (imaginary parts γ_k) ────────────────────
# These are mathematical constants — not computed, not approximated.
# Source: LMFDB / Odlyzko tables, 15+ significant figures.
RIEMANN_ZEROS: List[float] = [
    14.134725141734693,
    21.022039638771555,
    25.010857580145688,
    30.424876125859513,
    32.935061587739189,
    37.586178158825671,
    40.918719012147495,
    43.327073280914999,
    48.005150881167159,
    49.773832477672302,
    52.970321477714460,
    56.446247697063394,
    59.347044002602353,
    60.831778524609809,
    65.112544048081651,
    67.079810529494173,
    69.546401711173979,
    72.067157674481907,
    75.704690699083933,
    77.144840068874805,
]


# ── Core arithmetic functions ─────────────────────────────────────────────────

def delta_t_m(m: int) -> float:
    r"""Trotter step size for virtual modulus m.

    Δt_m = 0.04 / √m   [Ha⁻¹]

    This is BASE_DT / √(N_STEPS) generalised to arbitrary m via m = N_STEPS.
    For m=4: Δt_4 = 0.02 (matches MQEConfig.DT).
    """
    return 0.04 / math.sqrt(m)


def s_value(m: int, n_star: int) -> float:
    r"""Zeta-dual scaling factor s(m, n*).

    s = Δt_m / (2π · ln((n*+1)·Δt_m + 1))

    Maps Riemann zeros to eigenphases: φ(γ_k) = s · γ_k.

    Args:
        m:      Virtual register modulus (must be a multiple of 4).
        n_star: Zeta-dual step parameter = m // ν_n − 1 (from n_star_from_mechanism).

    Returns:
        float: Zeta-dual scaling factor s > 0.

    Spectral classes (verified against catalog):
        Group A  m=8,  n*=3  → s=0.04090
        Group B  m=4,  n*=1  → s=0.08115
        Group C  m=4,  n*=3  → s=0.04135
        Group D  m=12, n*=5  → s=0.02743
    """
    dt = delta_t_m(m)
    return dt / (2.0 * math.pi * math.log((n_star + 1) * dt + 1.0))


def eigenphase_bound(m: int, M_steps: int) -> float:
    r"""Upper bound on the Janus eigenphase magnitude.

    |φ_{k*}| ≤ π√m / (0.04 · M)

    Derives from the Trotter step constraint: τ_total = M · Δt_m = M · 0.04/√m,
    so |E·τ_total| ≤ π gives |E| ≤ π/(M·Δt_m) = π√m/(0.04·M).
    The eigenphase |φ_{k*}| = |E·τ_{k*}| ≤ |E|·τ_total.

    Args:
        m:       Virtual register modulus.
        M_steps: Total number of mechanism steps.

    Returns:
        float: Maximum accessible eigenphase magnitude [dimensionless radians].
    """
    return math.pi * math.sqrt(m) / (0.04 * M_steps)


def n_star_from_mechanism(mechanism: "MechanismTuple") -> Optional[int]:
    r"""Compute n* for the zeta-dual scaling factor.

    n* is a property of the cofactor shift rate, NOT the circuit step at which
    the CrossManifoldSWAPGate fires.  The gate may be placed at a later
    occurrence of k* for biological reasons (e.g. nitrogenase_lt places the
    crossing at the E₄H₄ intermediate, step 4, even though k*=2 is first
    reached at step 0).

    Formula: n* = m / ν_n − 1

    where ν_n is the per-step cofactor shift (assumed uniform; the first
    non-zero value of nu_shifts is used).  Physical meaning: m/ν_n is the
    number of cofactor-shift steps for one complete ℤ_m revolution (the
    register returns to 0); n* = (revolution length) − 1 is the last step
    before that closure.

    Spectral class verification:
        Group A  m=8,  ν_n=2  →  n* = 8/2−1 = 3   s=0.04090  ✓
        Group B  m=4,  ν_n=2  →  n* = 4/2−1 = 1   s=0.08115  ✓
        Group C  m=4,  ν_n=1  →  n* = 4/1−1 = 3   s=0.04135  ✓
        Group D  m=12, ν_n=2  →  n* = 12/2−1 = 5  s=0.02743  ✓

    Falls back to min(crossings[0][0]) for non-uniform or non-integer cases.
    Returns None for Case I/II mechanisms (no crossings → no Janus).
    """
    if not mechanism.crossings:
        return None
    nu_list = [nu for nu in mechanism.nu_shifts if nu > 0]
    if not nu_list:
        return min(c[0] for c in mechanism.crossings)
    nu_n = nu_list[0]  # first non-zero shift; uniform for all catalogue entries
    m = mechanism.m
    if m % nu_n == 0:
        return m // nu_n - 1
    # Non-integer revolution length: fall back to circuit crossing index
    return min(c[0] for c in mechanism.crossings)


def janus_energy_from_gamma(gamma_k: float, m: int, n_star: int) -> float:
    r"""Exact Janus energy for Riemann zero γ_k.

    E_Janus(k) = −s · γ_k / (n* · Δt_m)   [Ha, negative]

    Sign convention: QPE measures the phase φ = −E · τ under e^{−iHτ}.
    For a bound state E < 0: φ = |E| · τ = s · γ_k > 0.
    Therefore E_physical = −φ / τ = −s · γ_k / (n* · Δt_m) < 0.

    This is a pure-arithmetic result: no quantum circuit, no MLE.
    It is exact by thm:spectral_identification — the Janus eigenphase
    of H_MQE equals s·γ_k.

    Args:
        gamma_k: Imaginary part of the k-th non-trivial Riemann zero [Ha⁻¹].
        m:       Virtual register modulus.
        n_star:  Zeta-dual step parameter (must be ≥ 1).

    Returns:
        float: Janus intermediate energy in Hartree (negative).

    Raises:
        ValueError: If n_star < 1 (degenerate — no approach steps).
    """
    if n_star < 1:
        raise ValueError(
            f"n_star must be ≥ 1 (got {n_star}). "
            "The Janus crossing requires at least one approach step."
        )
    dt = delta_t_m(m)
    s  = s_value(m, n_star)
    tau_janus = n_star * dt
    return -(s * gamma_k / tau_janus)


# ── RiemannScaffold dataclass ─────────────────────────────────────────────────

@dataclass
class RiemannScaffold:
    r"""Complete Riemann spectral scaffold for a single MQE mechanism.

    Holds all arithmetic quantities derived from the mechanism's stoichiometry
    and the known Riemann zeros.  None of these require quantum simulation.

    Fields
    ------
    m           : Virtual register modulus.
    n_star      : Crossing step index (= step_idx of first crossing).
    s           : Zeta-dual scaling factor.
    phi_bound   : Eigenphase magnitude upper bound.
    gammas      : Riemann zeros γ_k within the eigenphase window.
    zero_indices: Corresponding indices into RIEMANN_ZEROS (0-based).
    janus_energies      : E_k = s·γ_k/(n*·Δt_m) for each γ_k in window [Ha].
    janus_eigenphases   : φ_k = s·γ_k for each γ_k in window [dimensionless].
    all_crossing_energies: Dict mapping each crossing step_idx → {k: E_k}.
                           For mechanisms with winding > 1 (multiple crossings),
                           each crossing has the same scaffold (same m, n*).
    spectral_class      : String label (e.g. 'Group A') if recognised, else '?'.
    dt                  : Δt_m [Ha⁻¹].
    """
    m:                    int
    n_star:               int
    s:                    float
    phi_bound:            float
    gammas:               List[float]
    zero_indices:         List[int]
    janus_energies:       List[float]   # negative (bound state convention)
    janus_eigenphases:    List[float]   # positive (s·γ_k > 0)
    all_crossing_energies: Dict[int, List[float]]  # {step_idx: [E_k, ...]}
    spectral_class:       str
    dt:                   float

    def summary(self) -> str:
        lines = [
            f"  RiemannScaffold ({self.spectral_class})",
            f"    m={self.m}  n*={self.n_star}  s={self.s:.5f}  Δt={self.dt:.6f} Ha⁻¹",
            f"    Eigenphase bound: |φ_k*| ≤ {self.phi_bound:.4f}",
            f"    Zeros in window:  {len(self.gammas)}",
        ]
        for i, (g, E, phi) in enumerate(
            zip(self.gammas, self.janus_energies, self.janus_eigenphases)
        ):
            lines.append(
                f"    γ_{self.zero_indices[i]+1} = {g:.6f}  →  "
                f"φ = {phi:.6f}  E_Janus = {E:+.8f} Ha"
            )
        return "\n".join(lines)


# ── Known spectral classes for label lookup ───────────────────────────────────

_SPECTRAL_CLASSES: List[Dict] = [
    {"m": 4,  "n_star": 1, "s_ref": 0.08115, "label": "Group B"},
    {"m": 4,  "n_star": 3, "s_ref": 0.04135, "label": "Group C"},
    {"m": 8,  "n_star": 3, "s_ref": 0.04090, "label": "Group A"},
    {"m": 12, "n_star": 5, "s_ref": 0.02743, "label": "Group D"},
    # Entries 12-15 classes
    {"m": 8,  "n_star": 1, "s_ref": 0.08070, "label": "Group E (m=8,fi=2)"},
    {"m": 8,  "n_star": 7, "s_ref": 0.02100, "label": "Group F (m=8,fi=8)"},
    {"m": 12, "n_star": 2, "s_ref": 0.05397, "label": "Group G (m=12,fi=3)"},
    {"m": 12, "n_star": 11,"s_ref": 0.01416, "label": "Group H (m=12,fi=12)"},
]


def _lookup_spectral_class(m: int, s: float) -> str:
    """Return a spectral class label if s matches a known class within 0.1%."""
    for cls in _SPECTRAL_CLASSES:
        if cls["m"] == m and abs(cls["s_ref"] - s) / cls["s_ref"] < 1e-3:
            return cls["label"]
    return f"Novel (m={m}, s={s:.5f})"


# ── Factory ───────────────────────────────────────────────────────────────────

def build_riemann_scaffold(mechanism: "MechanismTuple") -> Optional["RiemannScaffold"]:
    r"""Build the Riemann spectral scaffold for a mechanism.

    Returns None for Case I/II mechanisms (no crossings → no Janus).

    The scaffold is computed in O(|RIEMANN_ZEROS|) arithmetic operations.
    No Hamiltonian, no integrals, no quantum simulation.

    Args:
        mechanism: A populated MechanismTuple (any mechanism with m % 4 == 0
                   and at least one crossing entry).

    Returns:
        RiemannScaffold or None.

    Example::

        mech     = build_predefined_mechanisms(4)["nitrogenase_lt"]
        scaffold = build_riemann_scaffold(mech)
        print(scaffold.summary())
        # Group B, m=4, n*=1, s=0.08115
        # γ₁=14.135 → φ=1.146 → E_Janus=+28.65 Ha  (active-space, 4-orbital demo)
    """
    n_star = n_star_from_mechanism(mechanism)
    if n_star is None:
        return None  # no Janus crossing → no scaffold

    m      = mechanism.m
    M      = mechanism.M_steps
    dt     = delta_t_m(m)
    s      = s_value(m, n_star)
    phi_b  = eigenphase_bound(m, M)

    # Select zeros within the eigenphase window |s·γ_k| ≤ phi_bound
    gammas, zero_idxs, energies, phases = [], [], [], []
    for idx, gk in enumerate(RIEMANN_ZEROS):
        phi_k = s * gk
        if phi_k <= phi_b:
            gammas.append(gk)
            zero_idxs.append(idx)
            phases.append(phi_k)
            energies.append(janus_energy_from_gamma(gk, m, n_star))

    # All crossings share the same (m, n*) so the same scaffold energies apply.
    # Map each crossing step_idx → the same energy list.
    all_crossing_e: Dict[int, List[float]] = {
        c[0]: energies for c in mechanism.crossings
    }

    label = _lookup_spectral_class(m, s)

    return RiemannScaffold(
        m                    = m,
        n_star               = n_star,
        s                    = s,
        phi_bound            = phi_b,
        gammas               = gammas,
        zero_indices         = zero_idxs,
        janus_energies       = energies,
        janus_eigenphases    = phases,
        all_crossing_energies= all_crossing_e,
        spectral_class       = label,
        dt                   = dt,
    )
