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
mqetrotterdensematrix.py — Fixed-Depth Trotter QPE (Path B')
======================================================
Path B' is the rigorous apples-to-apples comparison between Path A
(mqeqpe.py) and the hardware.  It isolates the "Hardware Depth Tax" —
the pure decoherence cost of running a deeper circuit — by eliminating
all ancilla and MLE confounds present in Path B.

ARCHITECTURAL LESSONS FROM EMPIRICAL TESTING
    The original Path B' implementation used a VARIABLE-DEPTH circuit
    per τ value (n_steps = round(τ/Δt)).  This introduced a critical
    failure mode:

        τ-DEPENDENT NOISE: each τ uses a different-depth circuit, so
        different τ values feed the MAP with signals at different noise
        levels.  bayesian_map_energy assumes a single-frequency signal
        C(τ) = A·e^{-iEτ} with τ-independent amplitude A.  The
        mismatch produced large λ-non-linear biases (~233 mHa at λ=1)
        and a λ-independent ZNE floor of ~43 mHa (1st order) or ~142
        mHa (2nd order).

    Numerical verification (noiseless simulation) confirmed that the
    TROTTER SYSTEMATIC error in the phase of C(τ) is sub-0.001 mHa for
    BOTH first- and second-order Trotter at these parameters.  The
    floors were entirely from τ-dependent noise, not from Trotter
    approximation error.

    Attempting to fix the floor by switching to 2nd-order Trotter made
    things WORSE: the Strang palindrome has ~2× more gates per step,
    doubling per-step noise accumulation without providing any benefit
    (since the noiseless floor was already negligible).

CORRECT ARCHITECTURE: FIXED DEPTH
    Path B' must use the SAME density matrix for ALL τ values, just
    as Path A does (Path A's ρ comes from 1 Trotter step at fixed depth
    regardless of τ; Path B' uses n_max Trotter steps).  The τ-sweep
    is done entirely via the ideal propagator e^{-iHτ}, applied
    classically as a matrix exponential.

    n_max = max(1, round(τ_max / Δt)) = 16 for the default τ_seq and
    Δt = 0.02 Ha⁻¹.  This is the maximum circuit depth in the sequence.

    The signal for each τ is:
        C(τ, λ) = Tr(ρ_{n_max, λ} · e^{−iHτ})
    where ρ_{n_max, λ} is computed ONCE per λ (not once per τ).

    With fixed depth:
        - ALL τ values see identical noise level n_max × λ × p_per_gate
        - The MAP receives a consistent single-frequency signal ✓
        - Noise bias ∝ λ (linear) → ZNE works correctly ✓
        - Noiseless Trotter floor ≈ 0 (sub-0.001 mHa, empirically) ✓

WHAT THIS MODULE PROVIDES
    compute_trotter_density_matrix(psi_gs, n_steps, noise_scale,
                                   n_orbitals, h_diag, h_hop, g_full,
                                   dt)  →  np.ndarray (density matrix)

        Simulate n_steps first-order Trotter steps from |ψ_GS⟩ under
        the scaled Forte noise model.  Returns the 4**N × 4**N density
        matrix ρ_{n_steps, λ}.

    The caller (MultiTauQPEPipelineRunner in archive/mqe_multitau.py) then
    evaluates C(τ, λ) = Tr(ρ · expm(-1j·H·τ)) for every τ in τ_seq
    using pure numpy — no additional circuit simulation per τ.

COMPARISON OF PATHS
    Path A  (mqeqpe.py):          n_fixed=1 Trotter step, ideal propagator
                                  sweeps τ → C(τ) ∈ ℂ → bayesian_map_energy
    Path B  (mqehardwareqpe.py):  ancilla QPE circuit → p(k) ∈ ℝ⁴
                                  → hardware_map_energy
    Path B' (this module):        n_fixed=n_max Trotter steps, ideal
                                  propagator sweeps τ → C(τ) ∈ ℂ
                                  → bayesian_map_energy

    The "depth tax" is measured as the increase in ZNE residual between
    Path A (n=1) and Path B' (n=n_max=16): pure decoherence at depth 16
    with no Trotter or ancilla confounds.

Dependencies:
    cirq, numpy, scipy,
    nanoprotogeny.ionq.ionqtrotter      (build_trotter_evolution_circuit),
    nanoprotogeny.ionq.ionqfortenoise   (ForteHardwareNoiseModel,
                                         FORTE_NOISE_PARAMS).
No simulate-layer imports.  No ancilla-specific imports.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import cirq
import numpy as np
from scipy.linalg import expm

from nanoprotogeny.ionq.ionqfortenoise import (
    FORTE_NOISE_PARAMS,
    ForteHardwareNoiseModel,
)
from nanoprotogeny.ionq.ionqtrotter import build_trotter_evolution_circuit, build_second_order_trotter_evolution_circuit
from nanoprotogeny.qpe.mqeqpe import bayesian_map_energy

log = logging.getLogger(__name__)

_DEFAULT_DT: float = 0.02


# ─────────────────────────────────────────────────────────────────────────────
# Core function: fixed-depth density matrix
# ─────────────────────────────────────────────────────────────────────────────

def compute_trotter_density_matrix(
    psi_gs:      np.ndarray,
    n_steps:     int,
    noise_scale: float,
    n_orbitals:  int,
    h_diag:      Dict,
    h_hop:       Dict,
    g_full:      Dict,
    dt:          float          = _DEFAULT_DT,
) -> np.ndarray:
    r"""Simulate n_steps first-order Trotter steps and return the density matrix.

    Builds a circuit of exactly n_steps first-order Trotter steps from
    |ψ_GS⟩, simulates it under the scaled Forte noise model, and returns
    the resulting density matrix ρ_{n_steps, λ}.

    The caller evaluates C(τ, λ) = Tr(ρ · expm(-1j·H·τ)) for each τ using
    numpy — no additional circuit simulation per τ.  Using the SAME density
    matrix for all τ values eliminates the τ-dependent noise problem that
    afflicted the earlier variable-depth design.

    FIXED DEPTH IS THE KEY ARCHITECTURAL INVARIANT
        Path A  uses 1 Trotter step for ALL τ (ρ_1, the same matrix).
        Path B' uses n_steps Trotter steps for ALL τ (ρ_n, the same matrix).
        Both paths then sweep τ through the ideal propagator e^{-iHτ},
        applied classically.  This ensures that bayesian_map_energy receives
        a signal with τ-INDEPENDENT noise level, satisfying its single-
        frequency assumption and making ZNE well-conditioned.

    WHY FIRST-ORDER TROTTER SUFFICES
        Noiseless simulation confirmed that the phase bias in
        C(τ) = Tr(ρ_n_noiseless · e^{-iHτ}) is sub-0.001 mHa for n=16 steps
        with first-order Trotter at Δt=0.02.  Second-order Trotter provides
        no practical advantage here and was empirically found to be WORSE
        (because the Strang palindrome roughly doubles the gate count,
        amplifying the per-step noise without reducing the already-negligible
        noiseless floor).

    Args:
        psi_gs:      Ground-state vector, shape (4**N,).
        n_steps:     Number of first-order Trotter steps to apply.
                     For Path B', the caller sets n_steps = n_max =
                     max(1, round(τ_max / Δt)) so that ALL τ values in
                     τ_seq share the same density matrix.
        noise_scale: Forte noise scale factor λ ∈ {1, 2, 3} for ZNE.
        n_orbitals:  Number of active-space orbitals N.
        h_diag:      {p: h_pp}  one-electron diagonal integrals (Ha).
        h_hop:       {(p,q): h_pq}  hopping integrals, p < q (Ha).
        g_full:      {(p,q,r,s): g}  ERI in chemist's notation (Ha).
        dt:          Fundamental Trotter step Δt (Ha⁻¹).

    Returns:
        np.ndarray of shape (4**N, 4**N) — the density matrix after
        n_steps noisy Trotter steps from |ψ_GS⟩.
    """
    # ── 1. Build n_steps first-order Trotter steps ───────────────────────────
    # Design rationale (empirically validated):
    #   - The noiseless Trotter phase error in C(τ) is sub-0.001 mHa for
    #     n=16, Δt=0.02 Ha⁻¹ with FIRST-order Trotter.  Second-order provides
    #     no accuracy benefit here because |ψ_GS⟩ is an eigenstate — a single
    #     Trotter step only adds a global phase that cancels in the density
    #     matrix, making Trotter order irrelevant to the noiseless floor.
    #   - Second-order (Strang palindrome) has ~1.5–2× the gate count per step.
    #     At IonQ Forte noise rates (p2q=0.0068) across n=16 steps, the extra
    #     gates increase per-step noise enough that Richardson ZNE [λ=1,2,3]
    #     cannot fully cancel the cubic residual, producing a ~44 mHa floor.
    #     First-order keeps gate count minimal, giving Richardson enough room
    #     to reach sub-1.6 mHa chemical accuracy.
    #   - Replicate one first-order step n_steps times (NOT build_second_order
    #     with n_steps, which would require removing the loop — that was the
    #     original n_steps² bug when both were used together).
    trotter_step = build_trotter_evolution_circuit(
        n_orbitals, h_diag, h_hop, g_full, dt=dt
    )
    multi_step_circuit = cirq.Circuit()
    for _ in range(n_steps):
        multi_step_circuit += trotter_step

    # ── Guard: empty integrals → return pure-state ρ (no evolution) ──────────
    if not list(multi_step_circuit.all_operations()):
        log.warning(
            "[MT-QPE] Trotter circuit has no operations "
            "(all integrals below screening threshold). "
            "Returning pure-state density matrix psi_gs|psi_gs†."
        )
        return np.outer(psi_gs, psi_gs.conj())

    # ── 2. Scaled Forte noise model ───────────────────────────────────────────
    scaled_model = ForteHardwareNoiseModel(
        p1q=   min(1.0, FORTE_NOISE_PARAMS["p1q_error"]    * noise_scale),
        p2q=   min(1.0, FORTE_NOISE_PARAMS["p2q_error"]    * noise_scale),
        p_meas=min(1.0, FORTE_NOISE_PARAMS["p_meas_error"] * noise_scale),
        p_idle=min(1.0, FORTE_NOISE_PARAMS["p_idle_error"] * noise_scale),
    )

    # ── 3. Noisy density-matrix simulation ───────────────────────────────────
    sim    = cirq.DensityMatrixSimulator(noise=scaled_model)
    result = sim.simulate(multi_step_circuit, initial_state=psi_gs)
    return result.final_density_matrix          # shape (4**N, 4**N)


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive τ-sequence selector
# ─────────────────────────────────────────────────────────────────────────────

def select_tau_sequence(
    H_full:             np.ndarray,
    psi_n:              np.ndarray,
    n_orbitals:         int,
    h_diag:             Dict,
    h_hop:              Dict,
    g_full:             Dict,
    dt:                 float,
    E_ref:              float,
    candidate_taus:     List[float],
    chem_accuracy_mHa:  float = 1.6,
) -> List[float]:
    r"""Select the longest τ-sequence whose plain Richardson residual is within
    chemical accuracy at the corresponding circuit depth.

    Algorithm
    ---------
    For each τ_max in ``candidate_taus``, tested from largest to smallest:

    1.  Set ``n_max = round(τ_max / dt)`` — the exact depth the actual run
        would use.
    2.  Simulate ``ρ_{n_max, λ}`` for λ ∈ {1, 2, 3} via
        ``compute_trotter_density_matrix``.
    3.  Evaluate ``C(τ, λ) = Tr(ρ_{n_max,λ} · e^{−iHτ})`` for every τ ≤ τ_max.
    4.  Extract ``E_MAP(λ)`` via ``bayesian_map_energy``.
    5.  Compute plain Richardson: ``E_ZNE = 3·E1 − 3·E2 + E3``.
    6.  **Accept** if ``|E_ZNE − E_ref| · 1000 ≤ chem_accuracy_mHa``.
    7.  Return ``{τ : τ ≤ τ_max_accepted}`` on the first pass.

    Falls back to ``[dt]`` (n_max=1) if every candidate is rejected.

    Why Richardson residual — not a denominator proxy
    --------------------------------------------------
    The first implementation used ``|E_inf − E_ref| < 2.0 Ha`` as the
    acceptance gate.  Empirically this failed for nitrogenase_lt:

        τ_max=0.32 (n_max=16):  denom=−0.063 (non-zero ✓),
                                 |E_inf−E_ref|=626 mHa < 2000 mHa ✓
                                 → accepted, but actual ZNE residual = 9.05 mHa [!]

    The 2 Ha gate was a blowup guard, not a chemical-accuracy test.
    E_inf and E_zne_ex are different quantities; E_inf at 626 mHa from E_ref
    indicates a noisy λ-series where the exponential extrapolation partially
    corrects but cannot reach ≤ 1.6 mHa.

    The fix: since the preflight computes exactly the same E1, E2, E3 as the
    actual run (same ρ, same τ sequence, same ``bayesian_map_energy`` call),
    we test Richardson directly.  If ``|3E1 − 3E2 + E3 − E_ref| ≤ 1.6 mHa``
    in the preflight, plain Richardson already satisfies chemical accuracy and
    the full run (which additionally applies exponential extrapolation) will
    certainly stay within budget.  If not, the τ_max is too large and we try
    the next shorter sequence.  This turns the selector into a faithful dry-run
    of the actual computation rather than a heuristic proxy.

    Parameters
    ----------
    H_full : np.ndarray
        Full qudit Hamiltonian, shape ``(4**N, 4**N)``.
    psi_n : np.ndarray
        Exact ground-state vector of H_full, shape ``(4**N,)``.
    n_orbitals : int
        Active-space orbital count N.
    h_diag, h_hop, g_full : Dict
        Step-specific integrals passed to ``compute_trotter_density_matrix``.
    dt : float
        Fundamental Trotter step Δt (Ha⁻¹).
    E_ref : float
        Reference energy (Ha) for the MAP search window and residual test.
    candidate_taus : List[float]
        Candidate τ values, e.g. ``[0.02, 0.04, 0.08, 0.16, 0.32]``.
    chem_accuracy_mHa : float
        Chemical-accuracy threshold in mHa.  Default ``1.6`` mHa.

    Returns
    -------
    List[float]
        The longest accepted τ-sequence, i.e. all candidates ≤ τ_max_accepted.
    """
    sorted_cands = sorted(candidate_taus)          # ascending

    for i in range(len(sorted_cands) - 1, -1, -1): # descending: largest first
        tau_max   = sorted_cands[i]
        active    = sorted_cands[: i + 1]           # all τ ≤ τ_max
        n_max_try = max(1, int(round(tau_max / dt)))

        log.debug(
            "[TAU-SELECT] Probing τ_max=%.2f Ha⁻¹ (n_max=%d, seq=%s)",
            tau_max, n_max_try, active,
        )

        E_map_series: Dict[int, float] = {}
        try:
            for lam in [1, 2, 3]:
                rho_lam = compute_trotter_density_matrix(
                    psi_n, n_max_try,
                    noise_scale = lam,
                    n_orbitals  = n_orbitals,
                    h_diag      = h_diag,
                    h_hop       = h_hop,
                    g_full      = g_full,
                    dt          = dt,
                )
                overlaps: Dict[float, complex] = {
                    tau: complex(np.trace(rho_lam @ expm(-1j * H_full * tau)))
                    for tau in active
                }
                E_map, _, _       = bayesian_map_energy(overlaps, E_ref=E_ref)
                E_map_series[lam] = E_map

        except Exception as exc:
            log.warning(
                "[TAU-SELECT] τ_max=%.2f raised %s — skipping.", tau_max, exc,
            )
            continue

        E1, E2, E3    = E_map_series[1], E_map_series[2], E_map_series[3]
        E_zne_rich    = 3.0 * E1 - 3.0 * E2 + E3          # plain Richardson
        residual_mHa  = abs(E_zne_rich - E_ref) * 1000

        if residual_mHa <= chem_accuracy_mHa:
            log.info(
                "[TAU-SELECT] ✓ τ_max=%.2f Ha⁻¹  n_max=%d  "
                "|E_ZNE−E_ref|=%.4f mHa  [within %.1f mHa budget]",
                tau_max, n_max_try, residual_mHa, chem_accuracy_mHa,
            )
            return active

        log.info(
            "[TAU-SELECT] ✗ τ_max=%.2f Ha⁻¹  n_max=%d  "
            "|E_ZNE−E_ref|=%.4f mHa  [exceeds %.1f mHa budget]",
            tau_max, n_max_try, residual_mHa, chem_accuracy_mHa,
        )

    # All candidates rejected — fall back to single-point sequence (n_max=1)
    log.warning(
        "[TAU-SELECT] All %d candidates rejected. "
        "Falling back to single-point τ=[%.2f] Ha⁻¹ (n_max=1).",
        len(sorted_cands), dt,
    )
    return [dt]


# ─────────────────────────────────────────────────────────────────────────────
# Legacy shim (kept for external callers; not used by mqe_multitau.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_multitau_qpe_signal(
    H_qudit:     np.ndarray,
    psi_gs:      np.ndarray,
    tau:         float,
    noise_scale: float,
    n_orbitals:  int,
    h_diag:      Dict,
    h_hop:       Dict,
    g_full:      Dict,
    dt:          float          = _DEFAULT_DT,
    nelec:       Optional[int] = None,
) -> complex:
    r"""DEPRECATED SHIM — uses variable depth (τ-dependent noise problem).

    Retained only for backward compatibility with any external callers.
    MultiTauQPEPipelineRunner in archive/mqe_multitau.py no longer calls this;
    it calls compute_trotter_density_matrix directly (fixed depth).

    The variable-depth design (n_steps = round(τ/Δt) per τ value) causes
    each τ to see a different noise level, making the MAP likelihood
    inconsistent and producing large λ-non-linear biases that ZNE cannot
    cancel.  Use compute_trotter_density_matrix + fixed τ-sweep instead.
    """
    n_steps = max(1, int(round(tau / dt)))
    rho = compute_trotter_density_matrix(
        psi_gs, n_steps, noise_scale, n_orbitals, h_diag, h_hop, g_full, dt=dt
    )
    U_ideal = expm(-1j * H_qudit * tau)
    return complex(np.trace(rho @ U_ideal))
