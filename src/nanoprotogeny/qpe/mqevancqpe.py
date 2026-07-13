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
mqevancqpe.py — MQE-QPE Signal and MLE Energy Extraction
=============================================================
Native virtual-ancilla (D-state) QPE for the MQE dual-manifold architecture.
All superseded external-ancilla paths (--hw, --anc) have been archived;
see src/nanoprotogeny/archive/.

Public API (default MQE-QPE path only):

    compute_virtual_ancilla_qpe_probs(rho_sys, H_full, tau, n_max, ...)
        Ancilla probability vector p(k|τ) with D-state η_V damping.

    select_tau_sequence_virtual_ancilla(H_full, psi_n, ..., candidate_taus)
        Adaptive preflight: returns the longest τ-sequence within
        chemical accuracy under the virtual-ancilla signal chain.

    hardware_map_energy(ancilla_probs, E_ref, ..., eta_v)
        MLE energy extraction.  eta_v < 1 activates the η_V-corrected
        model; eta_v = 1.0 gives the ideal-ancilla model.

    _project_rho_to_sector(rho, sector_indices)
        Sector projection helper: removes noise-induced leakage from ρ.

    _count_screened_ctrl_gates(n_orbitals, h_diag, h_hop, g_full, angle_scale)
        Counts screened PowerControlledGate ops for η_V computation.

    VIRTUAL_ANCILLA_D_STATE_NOISE
        Dict with p_idle_virtual for ¹⁷¹Yb⁺ D-state manifold.

Dependencies:
    numpy, scipy,
    nanoprotogeny.ionq.ionqfortenoise  (FORTE_NOISE_PARAMS),
    nanoprotogeny.qpe.mqetrotterdensematrix   (compute_trotter_density_matrix).
No simulate-layer imports.
"""

from __future__ import annotations

import logging
import numpy as np
from scipy.linalg import expm
from typing import Dict, Optional, Tuple, List

from nanoprotogeny.ionq.ionqfortenoise import (
    FORTE_NOISE_PARAMS,
)
# Peer qpe-layer import — no circular dep (mqemultitauqpe does not import mqehardwareqpe)
from nanoprotogeny.qpe.mqetrotterdensematrix import compute_trotter_density_matrix

log = logging.getLogger(__name__)

# Integer labels for the ancilla measurement outcomes.
_K_VALS = np.arange(4)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SECTOR PROJECTION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _project_rho_to_sector(
    rho:            np.ndarray,
    sector_indices: "Optional[np.ndarray]",
) -> np.ndarray:
    r"""Project ρ onto the particle-number sector and renormalise.

    After noisy Trotter evolution, depolarising channels scatter amplitude
    from the physically relevant electron-number sector into other sectors
    (including the vacuum sector at E=0).  For mechanisms with positive
    active-space energies, the vacuum sector is energetically far from the
    target sector, and this leakage produces a large uniform background in
    p(k|τ) that MLE cannot distinguish from the true signal.

    This function extracts the sub-block of ρ corresponding to the correct
    sector, normalises it, and re-embeds it in the full Hilbert space.  The
    result contains only the population and coherences within the physically
    correct electron-number sector.

    Equivalence to post-selection
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Projecting ρ → P·ρ·P / Tr(P·ρ·P) is equivalent to measuring the total
    electron number and post-selecting on the correct value.  Particle-number
    conservation is enforced by stoichiometric invariants; any leakage is
    purely a noise artefact and is legitimately removed here.

    Args:
        rho:            Density matrix, shape (4**N, 4**N).
        sector_indices: Row/column indices of the nelec_active sector in the
                        full 4**N basis.  If None or empty, ρ is returned
                        unchanged (backward-compatible for mechanisms where
                        sector projection is not available).

    Returns:
        Projected and normalised density matrix, same shape as ρ.
    """
    if sector_indices is None or len(sector_indices) == 0:
        return rho

    # Extract the sector sub-block.
    rho_sub = rho[np.ix_(sector_indices, sector_indices)]
    trace   = np.real(np.trace(rho_sub))

    if trace < 1e-12:
        log.warning(
            "[RHO-PROJECT] Sector projection: trace of sub-block = %.2e < 1e-12. "
            "Noise has completely evacuated the target sector. "
            "Returning unprojected ρ as fallback.", trace
        )
        return rho

    # Re-embed normalised sub-block into the full Hilbert space.
    dim          = rho.shape[0]
    rho_proj     = np.zeros((dim, dim), dtype=complex)
    idx          = np.ix_(sector_indices, sector_indices)
    rho_proj[idx] = rho_sub / trace
    return rho_proj


# ─────────────────────────────────────────────────────────────────────────────
# 2.  VIRTUAL-ANCILLA QPE (native dual-manifold method)
#
# Uses the D-state (²D₃/₂, τ~52ms) virtual register of an existing ion as the
# QPE ancilla instead of an external NomosIonQid.  No additional ion needed.
#
# Simulation is kept in the 256-dim system space using the block decomposition:
#
#   ρ_joint = Σ_{k,k'} |k⟩⟨k'|_V ⊗ ρ_{kk'}^L
#
# Diagonal blocks ρ_{kk}^L: computed from compute_trotter_density_matrix
# (one 256-dim noisy Trotter sim per λ, same as Part 1).
#
# Off-diagonal blocks: provide the quantum coherences C_{kk'} = Tr[ρ_{kk'}^L].
# Under D-state idle noise (p_idle = 5e-5), each gate in which ancilla sector k
# is idle (not the control target) contributes a damping factor (1-p_idle) to
# C_{k≠k'}.  The total damping is:
#
#   η_V = (1 - p_idle_virtual)^n_ctrl_gates
#
# where n_ctrl_gates is the number of screened PowerControlledGates × n_max.
# For nitrogenase at n_max=4: η_V ≈ 0.9976 — nearly perfect coherence.
#
# p(k|τ) is then:
#   p(k|τ) = (1/16) Σ_{m,m'} exp(iπk(m'−m)/2) · η_V^|m−m'| · C_{mm'}
#
# ─────────────────────────────────────────────────────────────────────────────

# D-state noise parameters for ¹⁷¹Yb⁺ virtual manifold (²D₃/₂, τ~52ms).
# p_idle_virtual: idle decoherence per gate slot on the virtual ancilla.
# Gate error rate on the virtual register reuses p2q from the logical register
# (conservative — actual D-state gate fidelity may be higher).
VIRTUAL_ANCILLA_D_STATE_NOISE: Dict[str, float] = {
    "p_idle_virtual": FORTE_NOISE_PARAMS["p_idle_error"],   # 5e-5 per gate slot
}


def _count_screened_ctrl_gates(
    n_orbitals:   int,
    h_diag:       Dict,
    h_hop:        Dict,
    g_full:       Dict,
    angle_scale:  float,
    screen_thresh: float = 1e-10,
) -> int:
    """Count PowerControlledGate operations that survive angle-scale screening.

    Mirrors the screening logic of _one_ctrl_trotter_step in ionqqputrotter.py.
    Used to compute n_ctrl_gates = count × n_max for the η_V damping formula.
    """
    count = 0
    for p in range(n_orbitals):
        if p in h_diag and abs(h_diag[p] * angle_scale) > screen_thresh:
            count += 1                                    # ZClock
    for p in range(n_orbitals):
        for q in range(p + 1, n_orbitals):
            if abs(h_hop.get((p, q), 0.0) * angle_scale) > screen_thresh:
                count += 2                               # URShift pair
    for p in range(n_orbitals):
        for q in range(p + 1, n_orbitals):
            if abs(g_full.get((p, p, q, q), 0.0) * angle_scale) > screen_thresh:
                count += 1                               # Coulomb
    for p in range(n_orbitals):
        for q in range(p + 1, n_orbitals):
            if abs(g_full.get((p, q, q, p), 0.0) * angle_scale) > screen_thresh:
                count += 1                               # Exchange
    for i in range(n_orbitals):
        for j in range(n_orbitals):
            for k in range(n_orbitals):
                for l in range(n_orbitals):
                    if len({i, j, k, l}) < 4:
                        continue
                    if abs(g_full.get((i, j, k, l), 0.0) * angle_scale) > screen_thresh:
                        count += 1                       # Scattering
    return count


def compute_virtual_ancilla_qpe_probs(
    rho_sys:         np.ndarray,
    H_full:          np.ndarray,
    tau:             float,
    n_max:           int,
    n_orbitals:      int,
    h_diag:          Dict,
    h_hop:           Dict,
    g_full:          Dict,
    dt:              float = 0.02,
    p_idle_virtual:  float = VIRTUAL_ANCILLA_D_STATE_NOISE["p_idle_virtual"],
) -> np.ndarray:
    r"""Compute QPE ancilla probabilities using the virtual D-state register.

    The virtual D-state ancilla (²D₃/₂, τ~52ms) has an idle decoherence rate
    of p_idle = 5×10⁻⁵ per gate slot, giving a coherence damping factor of:

        η_V = (1 - p_idle)^n_ctrl_gates ≈ 0.9976  (for n_max=4, 12 gates/step)

    While physically real, this 0.24% damping is NOT applied as a pre-processing
    correction to p(k) in this function.  The reason: hardware_map_energy uses
    the ideal QPE model p_model(k|τ,E) for MLE.  Applying η_V to the observed
    p(k) before MLE creates a model mismatch — the MLE fits ideal probabilities
    to slightly-damped data — which at short τ (weak signal) causes the energy
    estimate to shift by O(100 mHa), far exceeding the 0.24% physical correction.

    The correct treatment of η_V requires modifying the MLE model inside
    hardware_map_energy to use p_model_corr(k|τ,E,η_V), which is left for
    future work.  At current circuit depths (n_max ≤ 16), η_V > 0.992 and the
    correction is smaller than all other error sources; it can be safely absorbed
    into the ZNE residual.

    Accordingly, this function is an exact alias for compute_hardware_qpe_probs.
    The "virtual ancilla" designation is ARCHITECTURAL: it specifies which
    hardware register (VirtualQudit, D-state manifold) serves as the QPE clock,
    not a numerical modification to the signal.  The simulation correctly models
    system decoherence via compute_trotter_density_matrix; ancilla decoherence
    at η_V > 0.992 is negligible.

    Args:
        rho_sys:        System density matrix (256×256), from
                        compute_trotter_density_matrix.
        H_full:         Full Hamiltonian matrix (256×256).
        tau:            QPE evolution time (Ha⁻¹).
        n_max:          Fixed Trotter circuit depth (retained for API symmetry
                        with select_tau_sequence_virtual_ancilla; not used in
                        signal computation at current η_V regime).
        n_orbitals, h_diag, h_hop, g_full, dt, p_idle_virtual:
                        Retained for future η_V-corrected MLE.  Not used now.

    Returns:
        np.ndarray of shape (4,) — real ancilla probabilities p(k), k∈{0,1,2,3},
        with η_V^|m−m'| damping on the off-diagonal coherences C_{mm'}.
        Caller MUST pass the same eta_v to hardware_map_energy so the MLE model
        matches the data-generating model.
    """
    # ── Virtual ancilla coherence damping ─────────────────────────────────────
    angle_scale    = tau / n_max if n_max > 0 else dt
    gates_per_step = _count_screened_ctrl_gates(
        n_orbitals, h_diag, h_hop, g_full, angle_scale
    )
    n_ctrl_gates = gates_per_step * n_max
    eta_V        = (1.0 - p_idle_virtual) ** n_ctrl_gates

    # ── Coherence matrix with D-state damping ─────────────────────────────────
    U_tau = expm(-1j * H_full * tau)
    U = [
        np.eye(H_full.shape[0], dtype=complex),
        U_tau,
        U_tau @ U_tau,
        U_tau @ U_tau @ U_tau,
    ]
    # C[m, m'] = Tr[U^m ρ_sys (U†)^{m'}] · η_V^|m−m'|
    A = np.array([
        [np.trace(U[m] @ rho_sys @ U[mp].conj().T) * (eta_V ** abs(m - mp))
         for mp in range(4)]
        for m in range(4)
    ], dtype=complex)

    # ── p(k|τ) from the damped coherence matrix ───────────────────────────────
    k_vals = np.arange(4)
    phase  = np.exp(
        1j * np.pi / 2.0
        * k_vals[:, None, None]
        * (k_vals[None, None, :] - k_vals[None, :, None])
    )                                           # shape (4, 4, 4): phase[k, m, m']
    p = np.real(np.einsum("kmn,mn->k", phase, A)) / 16.0

    p = np.clip(p, 0.0, None)
    norm = p.sum()
    return p / norm if norm > 1e-12 else np.full(4, 0.25)


def select_tau_sequence_virtual_ancilla(
    H_full:            np.ndarray,
    psi_n:             np.ndarray,
    n_orbitals:        int,
    h_diag:            Dict,
    h_hop:             Dict,
    g_full:            Dict,
    dt:                float,
    E_ref:             float,
    candidate_taus:    "List[float]",
    chem_accuracy_mHa: float = 1.6,
    sector_indices:    Optional[np.ndarray] = None,
) -> Optional[List[float]]:
    r"""Adaptive τ-sequence selector for the virtual-ancilla QPE path.

    Probes candidate τ-sequences in descending order (longest first), using:
        compute_trotter_density_matrix  →  ρ_sys   (system noise, 256-dim)
        _project_rho_to_sector          →  ρ_sys   (sector projection if needed)
        compute_virtual_ancilla_qpe_probs →  p(k)  (D-state damping included)
        hardware_map_energy              →  E_MAP   (η_V-corrected MLE)

    Returns the longest τ-sequence whose best-of-two ZNE residual is within
    chem_accuracy_mHa, or None if every candidate fails (noise-floor).

    Sector projection is retained: the system ρ_sys can still have sector
    leakage from Trotter noise.  The projection gating (E_0_n > 0) is the
    responsibility of the caller — sector_indices is passed through.
    """
    sorted_cands = sorted(candidate_taus)

    for i in range(len(sorted_cands) - 1, -1, -1):
        tau_max   = sorted_cands[i]
        active    = sorted_cands[: i + 1]
        n_max_try = max(1, int(round(tau_max / dt)))

        # Compute η_V for this candidate so the MLE model matches p_obs.
        # angle_scale is evaluated at tau_max/n_max_try (smallest angle per step).
        angle_scale_try   = tau_max / n_max_try
        gates_per_step_try = _count_screened_ctrl_gates(
            n_orbitals, h_diag, h_hop, g_full, angle_scale_try
        )
        eta_v_try = VIRTUAL_ANCILLA_D_STATE_NOISE["p_idle_virtual"]
        eta_v_try = (1.0 - eta_v_try) ** (gates_per_step_try * n_max_try)

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
                rho_lam = _project_rho_to_sector(rho_lam, sector_indices)
                ancilla_probs = {
                    tau: compute_virtual_ancilla_qpe_probs(
                        rho_lam, H_full, tau, n_max_try,
                        n_orbitals, h_diag, h_hop, g_full, dt,
                    )
                    for tau in active
                }
                # Pass the same η_V to the MLE model so it matches p_obs.
                E_map, _, _       = hardware_map_energy(
                    ancilla_probs, E_ref=E_ref, eta_v=eta_v_try
                )
                E_map_series[lam] = E_map

        except Exception as exc:
            log.warning(
                "[VANC-TAU-SELECT] τ_max=%.2f raised %s — skipping.", tau_max, exc,
            )
            continue

        E1, E2, E3 = E_map_series[1], E_map_series[2], E_map_series[3]
        E_zne_rich = 3.0 * E1 - 3.0 * E2 + E3
        denom = E3 - 2.0 * E2 + E1
        if abs(denom) > 1e-12:
            E_inf    = (E1 * E3 - E2 ** 2) / denom
            denom_e  = E2 - E_inf
            E_zne_ex = (
                E_inf + (E1 - E_inf) ** 2 / denom_e
                if abs(denom_e) > 1e-12 else E_zne_rich
            )
        else:
            E_zne_ex = E_zne_rich

        residual_mHa = min(
            abs(E_zne_rich - E_ref),
            abs(E_zne_ex   - E_ref),
        ) * 1000.0

        if residual_mHa <= chem_accuracy_mHa:
            log.info(
                "[VANC-TAU-SELECT] ✓ τ_max=%.2f Ha⁻¹  n_max=%d  "
                "|E_ZNE−E_ref|=%.4f mHa  [within %.1f mHa budget]",
                tau_max, n_max_try, residual_mHa, chem_accuracy_mHa,
            )
            return active

        log.info(
            "[VANC-TAU-SELECT] ✗ τ_max=%.2f Ha⁻¹  n_max=%d  "
            "|E_ZNE−E_ref|=%.4f mHa  [exceeds %.1f mHa budget]",
            tau_max, n_max_try, residual_mHa, chem_accuracy_mHa,
        )

    log.warning(
        "[VANC-TAU-SELECT] All %d candidates rejected for E_ref=%.6f Ha. "
        "Virtual-ancilla QPE cannot reach %.1f mHa under current noise. "
        "Returning None — caller will record noise-floor failure.",
        len(sorted_cands), E_ref, chem_accuracy_mHa,
    )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MLE ENERGY EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def hardware_map_energy(
    ancilla_probs: Dict[float, np.ndarray],  # {tau: p(k) array shape (4,)}
    E_ref:         Optional[float] = None,
    window_width:  float           = 4.0,
    n_coarse:      int             = 500,
    n_fine:        int             = 2000,
    eta_v:         float           = 1.0,
) -> Tuple[float, np.ndarray, np.ndarray]:
    r"""Maximum-likelihood energy estimate from hardware QPE histogram data.

    Accepts the same call signature and returns the same (E_MAP, E_grid,
    L_grid) triple as bayesian_map_energy so it can replace it transparently
    at the call site in MQEPipelineRunner.run().

    QPE histogram model (ideal ancilla, eta_v = 1.0):
        p(k | τ, E) = (1/16)|Σ_{m=0}^3 e^{im(Eτ + πk/2)}|²
                    = (1/16)[4 + 6cos(φ) + 4cos(2φ) + 2cos(3φ)]   φ = Eτ + πk/2

    MQE-QPE corrected model (virtual D-state ancilla, eta_v < 1):
        p_corr(k | τ, E, η_V) = (1/16)[4 + 6η_V cos(φ)
                                         + 4η_V² cos(2φ)
                                         + 2η_V³ cos(3φ)]
    where η_V = (1 − p_idle)^n_ctrl_gates damps the off-diagonal coherences
    C_{mm'} by η_V^|m−m'|.  At η_V = 1 the two models are identical.

    The corrected model is used when eta_v < 1.0 (MQE-QPE / virtual-ancilla
    path).  It MUST match the η_V used in compute_virtual_ancilla_qpe_probs
    so that the MLE data-generating model equals the MLE fitting model.

    Log-likelihood summed over all τ and k values:
        L(E) = Σ_τ Σ_{k=0}^3 p_obs(k, τ) · log(p_model(k | τ, E, η_V) + ε)

    Args:
        ancilla_probs: {tau: p_obs} where p_obs is a (4,) real array.
        E_ref:         Reference energy (Ha).  When provided, the search
                       window is centred on E_ref.
        window_width:  Total window width (Ha).  Default 4.0 Ha (± 2.0 Ha).
        n_coarse, n_fine: Grid sizes for the two-pass search.
        eta_v:         Virtual-ancilla D-state coherence damping factor
                       η_V = (1 − p_idle)^n_ctrl_gates.  Default 1.0 (ideal
                       ancilla / Part 1 behaviour).  Pass the same η_V used
                       in compute_virtual_ancilla_qpe_probs.

    Returns:
        (E_MAP, E_fine_grid, L_fine)
    """
    eps        = 1e-12
    half_width = window_width / 2.0
    _use_corr  = (eta_v < 1.0 - 1e-9)          # use corrected model for MQE-QPE

    # ── Adaptive window ───────────────────────────────────────────────────────
    if E_ref is not None:
        E_min, E_max = E_ref - half_width, E_ref + half_width
    else:
        # Fallback: use the τ with highest information (largest τ) to get a
        # rough estimate from the modal bin.
        tau_max  = max(ancilla_probs.keys())
        p_max    = ancilla_probs[tau_max]
        k_modal  = int(np.argmax(p_max))
        # At the true energy, k* ≈ 2E0*tau/π, so E0 ≈ π*k*/2/tau.
        E_inst   = (np.pi * k_modal / 2.0) / tau_max
        E_min, E_max = E_inst - half_width, E_inst + half_width

    # ── Helper: vectorised p_model over an energy grid ───────────────────────
    def _log_likelihood(E_grid: np.ndarray) -> np.ndarray:
        """Return L(E) for each point in E_grid."""
        L = np.zeros(len(E_grid))
        m = _K_VALS                              # [0, 1, 2, 3]  shape (4,)
        for tau, p_obs in ancilla_probs.items():
            for k in range(4):
                # Phase argument φ = Eτ + πk/2  (shape: n_grid,)
                phi = E_grid * tau + np.pi * k / 2.0
                if _use_corr:
                    # MQE-QPE corrected model: off-diagonal C_{mm'} damped by η_V^|m−m'|
                    # p_corr = (1/16)[4 + 6η_V cos(φ) + 4η_V² cos(2φ) + 2η_V³ cos(3φ)]
                    eta2 = eta_v * eta_v
                    eta3 = eta2  * eta_v
                    p_model = (4.0
                               + 6.0 * eta_v  * np.cos(phi)
                               + 4.0 * eta2   * np.cos(2.0 * phi)
                               + 2.0 * eta3   * np.cos(3.0 * phi)) / 16.0
                    p_model = np.clip(p_model, 0.0, None)
                else:
                    # Ideal ancilla model: p = |D|²  D = (Σ_m exp(imφ)) / 4
                    phases  = np.exp(1j * np.outer(m, phi))   # (4, n_grid)
                    D       = phases.sum(axis=0) / 4.0        # (n_grid,)
                    p_model = np.abs(D) ** 2                   # (n_grid,)
                L += p_obs[k] * np.log(p_model + eps)
        return L

    # ── Coarse pass ───────────────────────────────────────────────────────────
    E_coarse    = np.linspace(E_min, E_max, n_coarse)
    L_coarse    = _log_likelihood(E_coarse)
    peak_coarse = E_coarse[np.argmax(L_coarse)]

    # Boundary safety: expand window if peak lands on an edge.
    if peak_coarse <= E_min + 1e-12 or peak_coarse >= E_max - 1e-12:
        E_min, E_max    = E_min - half_width, E_max + half_width
        E_coarse        = np.linspace(E_min, E_max, n_coarse)
        L_coarse        = _log_likelihood(E_coarse)
        peak_coarse     = E_coarse[np.argmax(L_coarse)]

    # ── Fine pass ─────────────────────────────────────────────────────────────
    fine_delta = (E_max - E_min) / n_coarse * 4   # ±4 coarse steps
    E_fine     = np.linspace(
        max(E_min, peak_coarse - fine_delta),
        min(E_max, peak_coarse + fine_delta),
        n_fine,
    )
    L_fine = _log_likelihood(E_fine)
    E_MAP  = float(E_fine[np.argmax(L_fine)])
    return E_MAP, E_fine, L_fine
