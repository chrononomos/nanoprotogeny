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
mqeqpe.py — Quantum Phase Estimation Signal and Bayesian MAP Energy Extraction
===============================================================================
The two-function QPE energy extraction pipeline for the MQE framework.

compute_qpe_signal(H_qudit, psi_gs, tau, noise_scale, ...)
    Computes the noisy QPE overlap C(τ,λ) = Tr(ρ_λ(Δt) · e^{−iHτ}).
    Runs exactly ONE Trotter step under the scaled Forte noise model to
    isolate the noise-induced phase bias δφ(λ) without multi-step
    decoherence.  The ideal propagator U_ideal(τ) = e^{−iHτ} sweeps
    phase across all τ values without additional noise accumulation.

bayesian_map_energy(overlaps, E_ref, window_width, n_coarse, n_fine)
    Pure-numpy Bayesian MAP energy estimate from a dict of QPE overlaps
    {τ: C(τ,λ)}.  Maximises the log-likelihood
        L(E) = Σ_τ [Re(C) cos(Eτ) − Im(C) sin(Eτ)]
    via a two-pass coarse+fine grid search with adaptive window expansion.

Together these implement the QPE measurement + Richardson ZNE workflow:
    overlaps_λ = {τ: compute_qpe_signal(..., noise_scale=λ)}
    E_MAP_λ    = bayesian_map_energy(overlaps_λ, ...)[0]
    E_ZNE      = 3·E_MAP(1) − 3·E_MAP(2) + E_MAP(3)   [Richardson]

Dependencies: cirq, numpy, scipy,
              nanoprotogeny.ionq.ionqtrotter      (build_trotter_evolution_circuit),
              nanoprotogeny.ionq.ionqfortenoise (ForteHardwareNoiseModel,
                                                    FORTE_NOISE_PARAMS),
              nanoprotogeny.molecular.mqehamiltonian (_project_hamiltonian_to_sector).
No simulate-layer imports.
"""

from __future__ import annotations

import numpy as np
import cirq
from scipy.linalg import expm
from typing import Dict, Optional, Tuple

from nanoprotogeny.ionq.ionqtrotter import build_trotter_evolution_circuit
from nanoprotogeny.ionq.ionqfortenoise import (
    ForteHardwareNoiseModel,
    FORTE_NOISE_PARAMS,
)
from nanoprotogeny.molecular.mqehamiltonian import _project_hamiltonian_to_sector

# Default Trotter step size: BASE_DT / sqrt(N_STEPS) = 0.04 / sqrt(4) = 0.02 Ha⁻¹
# Mirrors the module-level DT constant in mqe.py.  Redefine here to avoid
# importing from simulate/ and to keep this module self-contained.
_DEFAULT_DT: float = 0.02

def compute_qpe_signal(
    H_qudit:     np.ndarray,
    psi_gs:      np.ndarray,
    tau:         float,
    noise_scale: float,
    n_orbitals:  int,
    h_diag:      Dict,
    h_hop:       Dict,
    g_full:      Dict,
    dt: float = _DEFAULT_DT,  # Step size for the Trotter kernel
    nelec: Optional[int] = None,
) -> complex:
    r"""Compute noisy QPE overlap C(τ,λ) = Tr(ρ_λ(Δt) · e^{−iHτ}).
    
    ARCHITECTURAL DESIGN:
    Simulates exactly ONE Trotter step (duration Δt) to isolate the 
    noise-induced phase bias δφ(λ) without accumulating multi-step 
    decoherence that breaks single-frequency QPE resolution.
    Phase accumulation across τ is handled entirely by U_ideal(τ).

    MATHEMATICAL CONSEQUENCE (Single-Step Design):
    Because |ψ_GS⟩ is an exact eigenstate of H, a single Trotter step leaves it nearly invariant:
        U_Trot(Δt) |ψ_GS⟩ ≈ e^{-iφ(Δt)} |ψ_GS⟩  ⟹  ρ_sim ≈ |ψ_GS⟩⟨ψ_GS|
    The density matrix remains pure and stationary. The overlap becomes:
        C(τ, λ) = Tr(ρ_sim · e^{-iHτ}) ≈ ⟨ψ_GS| e^{-iHτ} |ψ_GS⟩ = e^{-iE_0τ}
    The phase accumulates linearly with τ purely from the ideal propagator U_ideal(τ). 
    Noise only introduces a small, τ-independent phase bias δφ(λ):
        C(τ, λ) ≈ A(λ) e^{-i(E_0 + δφ(λ))τ}
    This satisfies the core assumptions of:
        1. Bayesian MAP Estimation: The likelihood L(E) peaks sharply at a single frequency E_0 + δφ(λ).
        2. Richardson ZNE: δφ(λ) scales polynomially with λ (c_1λ + c_2λ²), enabling exact cancellation.

    WARNING (Why Multi-Step Simulation is Avoided):
    By appending n_steps = τ/Δt copies of the circuit, the simulation would accumulate noise and 
    Trotter error proportional to τ. At τ = 0.32 (16 steps), the state decoheres significantly:
        ρ_λ(τ) = (1-ε)|ψ_GS⟩⟨ψ_GS| + ε ρ_excited + coherences
    The overlap becomes a multi-frequency sum:
        C(τ, λ) ≈ (1-ε)e^{-iE_0τ} + Σ_{n>0} c_n e^{-iE_nτ}
    The phase of a sum of oscillating exponentials is non-linear in τ, breaking QPE phase extraction
    and invalidating polynomial Richardson cancellation.
    """

    # Project Hamiltonian to electron sector if nelec specified
    if nelec is not None:
        H_qudit = _project_hamiltonian_to_sector(H_qudit, n_orbitals, nelec)

    # 1. Build exactly ONE Trotter step (fixed evolution time Δt)
    trotter_circuit = build_trotter_evolution_circuit(
        n_orbitals, h_diag, h_hop, g_full, dt=dt
    )

    # Guard: empty integrals produce a circuit with no qudits, which makes
    # psi_gs incompatible with the simulator's qudit shape.  Return the
    # ideal noiseless overlap directly — e^{-iE_0*tau} — as a placeholder.
    if not list(trotter_circuit.all_operations()):
        import logging as _log
        _log.getLogger(__name__).warning(
            "[QPE] Trotter circuit has no operations (integrals empty or all below "
            "screening threshold). Returning ideal overlap e^{-iE_0*tau} as placeholder."
        )
        U_ideal_tau = expm(-1j * H_qudit * tau)
        return complex(np.trace(np.outer(psi_gs, psi_gs.conj()) @ U_ideal_tau))

    # 2. Scaled Forte noise model
    scaled_model = ForteHardwareNoiseModel(
        p1q=   min(1.0, FORTE_NOISE_PARAMS["p1q_error"]    * noise_scale),
        p2q=   min(1.0, FORTE_NOISE_PARAMS["p2q_error"]    * noise_scale),
        p_meas=min(1.0, FORTE_NOISE_PARAMS["p_meas_error"] * noise_scale),
        p_idle=min(1.0, FORTE_NOISE_PARAMS["p_idle_error"] * noise_scale),
    )

    # 3. Noisy simulation (1 step only)
    sim = cirq.DensityMatrixSimulator(noise=scaled_model)
    result = sim.simulate(trotter_circuit, initial_state=psi_gs)
    rho_lambda = result.final_density_matrix
    
    # 4. Ideal propagator sweeps evolution time τ for phase resolution
    U_ideal_tau = expm(-1j * H_qudit * tau)
    
    # 5. QPE phase overlap
    return complex(np.trace(rho_lambda @ U_ideal_tau))


def bayesian_map_energy(
    overlaps:     Dict[float, complex],
    E_ref:        Optional[float] = None,   # Physical reference (e.g., exact diag)
    window_width: float           = 4.0,    # Total width (+/- 2.0 Ha around ref)
    n_coarse:     int             = 500,
    n_fine:       int             = 2000,
) -> Tuple[float, np.ndarray, np.ndarray]:
    r"""Bayesian MAP energy estimate with adaptive search window.
    
    Dynamic window: E in [E_ref - half_width, E_ref + half_width].
    If E_ref is None, falls back to instantaneous phase estimate from max(tau).
    
    Args:
        overlaps: {tau: C_complex} from compute_qpe_signal.
        E_ref: Reference energy (Ha). Typically E_0 from exact diagonalization.
        window_width: Total search window width in Ha. Default 4.0 Ha (+/- 2.0 Ha).
        n_coarse, n_fine: Grid sizes for coarse and fine passes.
        
    Returns:
        (E_MAP, E_fine_grid, L_fine)
    """
    # ── Adaptive window determination ─────────────────────────────────────
    half_width = window_width / 2.0
    if E_ref is not None:
        E_min, E_max = E_ref - half_width, E_ref + half_width
    else:
        # Fallback: longest tau gives maximal phase accumulation
        tau_max = max(overlaps.keys())
        C_max = overlaps[tau_max]
        E_inst = -np.angle(C_max) / tau_max
        E_min, E_max = E_inst - half_width, E_inst + half_width

    # ── Coarse pass ──────────────────────────────────────────────────────
    E_coarse = np.linspace(E_min, E_max, n_coarse)
    L_coarse = np.zeros(n_coarse)
    for tau, C in overlaps.items():
        L_coarse += np.real(C) * np.cos(E_coarse * tau) \
                  - np.imag(C) * np.sin(E_coarse * tau)
                  
    peak_coarse = E_coarse[np.argmax(L_coarse)]
    
    # Boundary safety: expand if peak hits edge
    if peak_coarse <= E_min + 1e-12 or peak_coarse >= E_max - 1e-12:
        E_min, E_max = E_min - half_width, E_max + half_width
        E_coarse = np.linspace(E_min, E_max, n_coarse)
        L_coarse = np.zeros(n_coarse)
        for tau, C in overlaps.items():
            L_coarse += np.real(C) * np.cos(E_coarse * tau) \
                      - np.imag(C) * np.sin(E_coarse * tau)
        peak_coarse = E_coarse[np.argmax(L_coarse)]

    # ── Fine pass ────────────────────────────────────────────────────────
    fine_delta = (E_max - E_min) / n_coarse * 4  # +/- 4 coarse steps
    E_fine = np.linspace(
        max(E_min, peak_coarse - fine_delta),
        min(E_max, peak_coarse + fine_delta),
        n_fine,
    )
    L_fine = np.zeros(n_fine)
    for tau, C in overlaps.items():
        L_fine += np.real(C) * np.cos(E_fine * tau) \
                - np.imag(C) * np.sin(E_fine * tau)

    E_MAP = float(E_fine[np.argmax(L_fine)])
    return E_MAP, E_fine, L_fine


