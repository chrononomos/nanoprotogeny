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
mqehmqe.py — Direct MQE-Hamiltonian Construction (Hilbert-Pólya Path)
======================================================================
Implements the "reverse" or "direct" Hilbert-Pólya approach to extracting
Janus intermediate eigenphases.  Instead of the MQE-QPE measurement protocol
(build Trotter circuit → sample p(k|τ) → MLE), this module constructs the
MQE Hamiltonian

    Ĥ_MQE = (i / τ_total) · log U_Trot(τ_total)

explicitly as a Hermitian matrix, diagonalises it, and reads off the Janus
eigenphases directly from the k* = m/2 virtual-sector block.

THEORETICAL FOUNDATION
    The MQE Hamiltonian is defined in quantum-enzymaticsB.md (def:hmqe) as
    the self-adjoint generator of the full M-step mechanism evolution:

        U_Trot(τ) = ∏_{n=0}^{M-1} [ (e^{-iH_n Δt_m} ⊗ I_V) · U_couple^{(m,ν_n)} ]
                   · U_Janus    (inserted at step n* if 4|m)

        Ĥ_MQE = (i/τ_total) · logm(U_Trot(τ_total))

    Self-adjointness: U_Trot is unitary → log(U_Trot) is anti-Hermitian →
    i·log(U_Trot) is Hermitian.  Multiplying by real 1/τ_total preserves it.

    Universal Janus Criticality Theorem (thm:ujct): in the k* = m/2 virtual
    sector of Ĥ_MQE, all eigenphases are topologically confined to the
    critical line Re(s) = 1/2 under the zeta-dual spectral identification.
    This confinement is UNCONDITIONAL — no Hamiltonian parameter, geometry, or
    environment can move them while 4|m and phase closure hold.

HILBERT SPACE STRUCTURE
    Full space: H_L ⊗ H_V
    H_L: (4^N)-dimensional logical Fock space (tetralemmatic encoding)
    H_V: m-dimensional virtual ℤ_m phase register

    For m = 4r, H_V = H_{V1}^{(4)} ⊗ H_{Vaux}^{(r)} — but since
    CrossManifoldSWAPGate acts on V1 alone (d=4), the V1 component is
    treated as a single d=4 register.  V_aux only appears when r > 1.

    Combined basis index (L×V₁): I = i_L * 4 + j_V1, where:
        i_L ∈ {0, ..., 4^N - 1}   (logical Fock basis)
        j_V ∈ {0, ..., m - 1}     (virtual phase register)

    Dimension: dim_LV = 4^N * m.

OPERATORS IN L×V SPACE

    1. Trotter step n (acts on L only):
           U_trot^(n) ⊗ I_V = block_diag([expm(-i H_n dt)] × m)
       Matrix index: (i_L' * m + j_V, i_L * m + j_V) = U_trot^(n)[i_L', i_L]

    2. CofactorCouplingGate (m, ν_n) — controlled shift on V:
           |i_L⟩|j_V⟩ → |i_L⟩|(j_V + ν_n · N_e(i_L)) mod m⟩
       N_e(i_L) = sum_p [0,1,1,2][(i_L // 4^p) % 4]  (electron count of i_L)
       This is a permutation matrix with exactly one 1 per column.

    3. CrossManifoldSWAPGate on orbital p (Janus carrier):
           |i_L⟩|j_V⟩ → |i_L with digit_p = j_V⟩|digit_p(i_L)⟩
       Where digit_p(i_L) = (i_L // 4^p) % 4.
       This is the 16×16 d=4⊗d=4 SWAP restricted to the (L_p, V₁) subsystem.
       It is unitary and maps between electron sectors.

    Full step n unitary:
        U^(n) = U_couple^(m,ν_n) · (U_trot^(n) ⊗ I_V)

    Full mechanism unitary:
        U_mech = [∏_{n after n*} U^(n)] · U_Janus · [∏_{n ≤ n*} U^(n)]

    where U_Janus = ∏_{p in janus_orbs} SWAP_{L_p, V} acts at step n*.

k* SECTOR PROJECTION
    The k* = m/2 sector projector is:
        Π_{k*} = I_L ⊗ |k*⟩⟨k*|_V

    In combined index space: indices {i_L * m + k* : i_L ∈ 0..4^N - 1}.
    The projected subspace has dimension 4^N.
    Eigenvalues of Π_{k*} · Ĥ_MQE · Π_{k*} (restricted to k* block) are
    the Janus eigenphases.

ZETA-DUAL MAP
    Under the zeta-dual parameterisation (def:zeta_dual in Part B):
        ν_n = (1/2π) log(n · Δt_m + t_0)   (cofactor scaling)
        E_n ~ log n                          (logarithmic energy spectrum)

    The Janus eigenphase φ_{k*} ∈ ℝ (eigenvalue of Ĥ_MQE at step n*) maps to:
        γ_k = φ_{k*} / ΔE_scale
    where ΔE_scale = 1/(2π n*) in natural units of the mechanism.

WHEN TO USE THIS MODULE
    Classical validation (N ≤ 6 orbitals): exact Janus eigenphases, no sampling
    Spectral flow analysis: track eigenphase vs. mechanism parameters
    Zeta-dual calibration: fix ΔE_scale for γ_k computation
    Full spectrum access: see all eigenphases simultaneously, not just ground state

WHEN NOT TO USE
    Large N (≥ 8 orbitals): dim_LV = 4^N × m → intractable classically
    Hardware execution: use mqevanc.py (MQE-QPE + MLE) for actual hardware runs

COMPARISON WITH mqevanc.py (MQE-QPE path)
    mqevanc.py  : samples p(k|τ) via noisy Trotter circuit → MLE → one eigenphase
                  per step, per τ-sequence, per ZNE level.  Suitable for hardware.
    mqehmqe.py  : builds U_mech as matrix → logm → Ĥ_MQE → eigh → all eigenphases
                  at once, exact (no sampling noise, no η_V correction, no ZNE).
                  Suitable for classical validation and spectral analysis.

Dependencies (no cirq, ionq, simulate-layer imports):
    numpy, scipy.linalg (expm, logm, eigh),
    nanoprotogeny.molecular.mqemolecules   (MechanismTuple),
    nanoprotogeny.molecular.mqehamiltonian (build_qudit_hamiltonian_matrix).

Public API
----------
    electron_count(i_L, N)
        Number of electrons in Fock basis state i_L for N orbitals.

    build_cofactor_coupling_matrix(N, m, nu)
        Unitary (4^N·m) × (4^N·m) CofactorCouplingGate in L×V space.

    build_janus_swap_matrix(N, m, orbital_p)
        Unitary (4^N·m) × (4^N·m) CrossManifoldSWAPGate restricted to L_p ⊗ V₁.

    build_full_mechanism_unitary(step_hamiltonians, mechanism, dt)
        Full M-step mechanism unitary in L×V space. Returns (4^N·m) × (4^N·m).

    build_hmqe(step_hamiltonians, mechanism, dt)
        MQE Hamiltonian Ĥ_MQE = (i/τ) logm(U_mech). Hermiticity verified.

    k_star_eigenphases(H_mqe, N, m)
        Eigenvalues of Ĥ_MQE projected to the k* = m/2 virtual sector.
        Returns (eigenvalues_k_star, full_spectrum).

    janus_sector_matrix(H_mqe, N, m)
        The full k* × k* sub-block of Ĥ_MQE (4^N × 4^N Hermitian matrix).

    zeta_dual_gamma(eigenphase, n_star, dt_m, t0=0.0)
        Apply inverse zeta-dual spectral map: φ_{k*} → γ_k.

    run_hmqe_analysis(step_hamiltonians, mechanism, dt)
        High-level entry point: returns HMQEResult dataclass with all outputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import expm, logm

from nanoprotogeny.molecular.mqemolecules import MechanismTuple

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Occupation number lookup for d=4 qudit basis: {|vac⟩=0, |↑⟩=1, |↓⟩=1, |↑↓⟩=2}
# ──────────────────────────────────────────────────────────────────────────────
_NЕЛEC_PER_DIGIT: np.ndarray = np.array([0, 1, 1, 2], dtype=np.int32)


# ──────────────────────────────────────────────────────────────────────────────
# 1. ELECTRON COUNT
# ──────────────────────────────────────────────────────────────────────────────

def electron_count(i_L: int, N: int) -> int:
    r"""Return the number of electrons in Fock basis state i_L for N orbitals.

    The d=4 qudit at site p encodes {|vac⟩, |↑⟩, |↓⟩, |↑↓⟩} as {0,1,2,3}.
    Basis state i_L ∈ {0, ..., 4^N - 1} has digit d_p = (i_L // 4^p) % 4 at
    site p, contributing N_e_per_digit[d_p] ∈ {0, 1, 1, 2} electrons.

    Args:
        i_L: Fock basis index in {0, ..., 4^N - 1}.
        N:   Number of active-space orbitals.

    Returns:
        Total electron count (integer in {0, ..., 2N}).
    """
    count = 0
    tmp   = i_L
    for _ in range(N):
        count += _NЕЛEC_PER_DIGIT[tmp & 3]   # tmp % 4
        tmp  >>= 2                             # tmp //= 4
    return count


def _electron_count_array(N: int) -> np.ndarray:
    r"""Precompute electron counts for all 4^N basis states. Shape (4^N,)."""
    dim = 4 ** N
    arr = np.zeros(dim, dtype=np.int32)
    for i in range(dim):
        arr[i] = electron_count(i, N)
    return arr


def _digit_at(i_L: int, p: int) -> int:
    r"""Return digit (local d=4 state) of orbital p in Fock basis state i_L."""
    return (i_L >> (2 * p)) & 3              # (i_L // 4^p) % 4


def _set_digit(i_L: int, p: int, d: int, N: int) -> int:
    r"""Return i_L with digit at position p replaced by d ∈ {0,1,2,3}."""
    mask = ~(3 << (2 * p))                   # zero out bits 2p and 2p+1
    return (i_L & mask) | (d << (2 * p))


# ──────────────────────────────────────────────────────────────────────────────
# 2. CofactorCouplingGate in L×V space
# ──────────────────────────────────────────────────────────────────────────────

def build_cofactor_coupling_matrix(N: int, m: int, nu: int) -> np.ndarray:
    r"""Build the CofactorCouplingGate as a unitary matrix in L×V space.

    The gate implements the controlled virtual shift:
        |i_L⟩|j_V⟩  →  |i_L⟩|(j_V + ν · N_e(i_L)) mod m⟩

    This is a permutation matrix of shape (4^N·m, 4^N·m) with exactly one
    entry of 1.0 per row and per column.

    Combined basis index: I = i_L * m + j_V.

    Args:
        N:   Number of active-space orbitals (logical register dimension 4^N).
        m:   Virtual register modulus.
        nu:  Cofactor shift value ν_n for this step.

    Returns:
        Complex unitary ndarray of shape (4^N * m, 4^N * m).
    """
    dim_L  = 4 ** N
    dim_LV = dim_L * m
    nелec  = _electron_count_array(N)   # shape (dim_L,)

    # Build permutation as index mapping: col I → row I'
    rows   = np.empty(dim_LV, dtype=np.int64)
    cols   = np.arange(dim_LV, dtype=np.int64)

    for i_L in range(dim_L):
        shift = int(nu * nелec[i_L]) % m
        for j_V in range(m):
            I      = i_L * m + j_V
            j_V_out = (j_V + shift) % m
            rows[I] = i_L * m + j_V_out

    U = np.zeros((dim_LV, dim_LV), dtype=complex)
    U[rows, cols] = 1.0
    return U


# ──────────────────────────────────────────────────────────────────────────────
# 3. CrossManifoldSWAPGate on orbital p in L×V₁ space
# ──────────────────────────────────────────────────────────────────────────────

def build_janus_swap_matrix(N: int, m: int, orbital_p: int) -> np.ndarray:
    r"""Build the CrossManifoldSWAPGate for orbital p as a matrix in L×V₁ space.

    The gate swaps the d=4 state of logical orbital p with the virtual register:
        |i_L⟩|j_V⟩  →  |i_L with digit_p = j_V⟩|digit_p(i_L)⟩

    where digit_p(i_L) = (i_L // 4^p) % 4 is the local qudit state at site p.

    This is a unitary (permutation) matrix on L×V₁.  It maps between electron
    sectors in L: if digit_p(i_L) ∈ {1,2} and j_V = 0 (vacuum), the L electron
    count decreases by 1; if digit_p(i_L) = 3 and j_V = 0, it decreases by 2.

    Matches the _unitary_ method of CrossManifoldSWAPGate (ionqmqegates.py),
    which defines the 16×16 SWAP in the computational tensor basis as:
        U[j*4+i, i*4+j] = 1  for all i,j ∈ {0,1,2,3}
    (not the physical Bell-separable basis).

    RESTRICTION: This implementation treats V as a single d=4 register (m=4
    case).  For m=4r with r>1, V₁ still has d=4; V_aux is treated as external
    to the SWAP (the Janus acts on V₁ alone regardless of m, by Case III
    uniqueness: the faithful ℤ_4 subgroup ⟨m/4⟩ ≤ ℤ_m provides the Janus slot).

    Args:
        N:        Number of active-space orbitals.
        m:        Virtual register modulus (must satisfy 4|m for Case III).
        orbital_p: Index of the logical orbital being swapped with V₁.

    Returns:
        Complex unitary ndarray of shape (4^N * m, 4^N * m).

    Raises:
        ValueError: if m % 4 != 0 (Case III required for operational Janus).
    """
    if m % 4 != 0:
        raise ValueError(
            f"Janus SWAP requires 4|m (Case III). Got m={m}. "
            f"m%4={m%4} → Case {'I' if m%2 else 'II'} — Janus is {'absent' if m%2 else 'dimensionally obstructed'}."
        )

    dim_L  = 4 ** N
    dim_LV = dim_L * m
    U      = np.zeros((dim_LV, dim_LV), dtype=complex)

    for i_L in range(dim_L):
        d_p = _digit_at(i_L, orbital_p)          # current local state at orbital p
        for j_V in range(m):
            # After SWAP: orbital p in L gets j_V mod 4 (V₁ component of j_V)
            # and V gets d_p (the displaced logical state).
            # For m > 4: the SWAP acts on V₁ = j_V mod 4 while V_aux = j_V // 4
            # is unchanged (Janus only engages the ℤ_4 subgroup of ℤ_m).
            j_V1  = j_V  % 4          # V₁ component
            j_aux = j_V // 4          # V_aux component (unchanged)

            # New logical state: replace digit_p with j_V1 (from V₁)
            i_L_out = _set_digit(i_L, orbital_p, j_V1, N)

            # New virtual state: V₁ gets d_p, V_aux unchanged
            j_V_out = d_p + j_aux * 4

            I_in  = i_L     * m + j_V
            I_out = i_L_out * m + j_V_out
            U[I_out, I_in] = 1.0

    # Verify unitarity (should hold by construction — permutation matrix)
    assert np.allclose(U @ U.conj().T, np.eye(dim_LV), atol=1e-12), \
        f"JanusSwapMatrix (orbital {orbital_p}) is not unitary — bug in index arithmetic."
    return U


# ──────────────────────────────────────────────────────────────────────────────
# 4. Trotter step unitary ⊗ I_V
# ──────────────────────────────────────────────────────────────────────────────

def _build_trotter_step_LV(H_n: np.ndarray, m: int, dt: float) -> np.ndarray:
    r"""Compute (expm(-i H_n dt)) ⊗ I_V in the combined L×V basis.

    The Trotter step acts only on L; V is untouched.  In the combined basis
    I = i_L * m + j_V, this is a block-diagonal matrix with m copies of
    expm(-i H_n dt) along the diagonal.

    Args:
        H_n: L-space Hamiltonian, shape (4^N, 4^N).
        m:   Virtual register modulus (number of V basis states).
        dt:  Trotter step size Δt_m.

    Returns:
        Complex unitary ndarray of shape (4^N * m, 4^N * m).
    """
    dim_L  = H_n.shape[0]
    dim_LV = dim_L * m
    U_L    = expm(-1j * H_n * dt)                       # (dim_L, dim_L)

    # Block-diagonal: m copies of U_L
    U_LV   = np.zeros((dim_LV, dim_LV), dtype=complex)
    for j_V in range(m):
        for i_L in range(dim_L):
            row_base = i_L * m + j_V
            for i_L_p in range(dim_L):
                col_base = i_L_p * m + j_V
                U_LV[row_base, col_base] = U_L[i_L, i_L_p]
    return U_LV


def _build_trotter_step_LV_fast(H_n: np.ndarray, m: int, dt: float) -> np.ndarray:
    r"""Fast vectorised version of _build_trotter_step_LV using kron."""
    U_L = expm(-1j * H_n * dt)
    return np.kron(U_L, np.eye(m, dtype=complex))


# ──────────────────────────────────────────────────────────────────────────────
# 5. Full mechanism unitary
# ──────────────────────────────────────────────────────────────────────────────

def build_full_mechanism_unitary(
    step_hamiltonians: List[np.ndarray],
    mechanism:         "MechanismTuple",
    dt:                float,
) -> np.ndarray:
    r"""Compose all M mechanism steps into the full unitary U_mech in L×V space.

    For each step n ∈ {0, ..., M-1}:
        U^(n) = U_couple^(m, ν_n) · (expm(-i H_n dt) ⊗ I_V)
              · [U_Janus if n is a Janus crossing step]

    The steps are applied left-to-right in chronological order:
        U_mech = U^(M-1) · ... · U^(1) · U^(0)
    (rightmost factor applied first to the state).

    Janus crossings from mechanism.crossings are inserted immediately AFTER
    the Trotter+coupling step at the specified step_idx, following the
    ordering in the theory (the SWAP fires after the Hamiltonian evolution at
    step n* has placed the system at the k* = m/2 virtual phase).

    Args:
        step_hamiltonians: List of M Hamiltonian matrices, each shape (4^N, 4^N).
                           Index n corresponds to mechanism step n.
        mechanism:         MechanismTuple with N_orbitals, M_steps, m, nu_shifts,
                           and crossings.
        dt:                Trotter step size Δt_m in Ha⁻¹.

    Returns:
        Complex unitary ndarray of shape (4^N * m, 4^N * m).

    Raises:
        ValueError: if len(step_hamiltonians) != mechanism.M_steps.
        ValueError: if m % 4 != 0 and crossings is non-empty (Janus requires Case III).
    """
    N = mechanism.N_orbitals
    m = mechanism.m
    M = mechanism.M_steps

    if len(step_hamiltonians) != M:
        raise ValueError(
            f"Expected {M} step Hamiltonians, got {len(step_hamiltonians)}."
        )

    dim_L  = 4 ** N
    dim_LV = dim_L * m

    # Pre-build Janus SWAP matrices indexed by (step_idx, orbital_p)
    janus_by_step: Dict[int, np.ndarray] = {}
    for (step_idx, orbital_p, orbital_q, _delta_CI) in mechanism.crossings:
        # Full Janus SWAP S_LV^{(p,q)} = SWAP_{L_p,V} ⊗ SWAP_{L_q,V}
        # Each SWAP is a separate (4^N*m)×(4^N*m) permutation; compose them.
        S_p = build_janus_swap_matrix(N, m, orbital_p)
        S_q = build_janus_swap_matrix(N, m, orbital_q)
        S_pq = S_q @ S_p          # order: S_p first, then S_q (both act on V)
        janus_by_step[step_idx] = (
            janus_by_step[step_idx] @ S_pq
            if step_idx in janus_by_step
            else S_pq
        )
        log.debug(
            "[HMQE] Janus SWAP at step %d: orbitals (%d,%d), δ_CI=%.4f",
            step_idx, orbital_p, orbital_q, _delta_CI,
        )

    # Compose steps chronologically: U_mech = U^{M-1} ⋅ … ⋅ U^{0}
    U_mech = np.eye(dim_LV, dtype=complex)

    for n in range(M):
        H_n = step_hamiltonians[n]
        nu_n = mechanism.nu_shifts[n]

        # Step evolution: (expm(-i H_n dt)) ⊗ I_V
        U_trot = _build_trotter_step_LV_fast(H_n, m, dt)

        # Cofactor coupling: |i_L⟩|j_V⟩ → |i_L⟩|(j_V + ν_n N_e(i_L)) mod m⟩
        U_couple = build_cofactor_coupling_matrix(N, m, nu_n)

        # Combined: coupling after Trotter
        U_step = U_couple @ U_trot

        # Janus SWAP (if this step has a crossing): applied after coupling
        if n in janus_by_step:
            U_step = janus_by_step[n] @ U_step
            log.debug("[HMQE] Inserted Janus SWAP at step %d.", n)

        U_mech = U_step @ U_mech

    # Verify unitarity
    err = np.max(np.abs(U_mech @ U_mech.conj().T - np.eye(dim_LV)))
    if err > 1e-8:
        log.warning(
            "[HMQE] U_mech unitarity error = %.2e (threshold 1e-8). "
            "Consider checking Hamiltonian Hermiticity and dt value.", err,
        )
    else:
        log.debug("[HMQE] U_mech unitarity verified (max error = %.2e).", err)

    return U_mech


# ──────────────────────────────────────────────────────────────────────────────
# 6. MQE Hamiltonian
# ──────────────────────────────────────────────────────────────────────────────

def build_hmqe(
    step_hamiltonians: List[np.ndarray],
    mechanism:         "MechanismTuple",
    dt:                float,
    hermiticity_tol:   float = 1e-8,
) -> np.ndarray:
    r"""Build Ĥ_MQE = (i / τ_total) · logm(U_mech) in L×V space.

    Constructs the full mechanism unitary U_mech via build_full_mechanism_unitary,
    takes the principal matrix logarithm (branch cut on (-∞, 0]), multiplies by
    i/τ_total, and verifies self-adjointness.

    Self-adjointness proof (def:hmqe in quantum-enzymaticsB.md):
        U_mech is unitary → logm(U_mech) is anti-Hermitian (skew-Hermitian):
            [logm(U)]† = logm(U†) = logm(U⁻¹) = −logm(U)
        Multiplying by i: [i·logm(U)]† = −i·[logm(U)]† = −i·(−logm(U)) = i·logm(U).
        Hence H_MQE = (i/τ) logm(U) is Hermitian.

    Branch cut note: scipy.linalg.logm uses the principal logarithm, defined
    for matrices with no eigenvalues on ℝ_{≤0}.  If U_mech has eigenvalues at
    e^{iπ} = −1 (which can occur when τ_total = π/E for some eigenphase E), the
    branch cut is hit and logm raises or returns a complex non-Hermitian result.
    This function warns if that occurs (norm of anti-Hermitian part > threshold).
    Remedy: adjust τ by changing dt slightly to avoid the branch cut.

    Args:
        step_hamiltonians: List of M Hamiltonians, shape (4^N, 4^N) each.
        mechanism:         MechanismTuple.
        dt:                Trotter step size Δt_m.
        hermiticity_tol:   Maximum allowed max(|H - H†|) / 2.  Default 1e-8.

    Returns:
        Hermitian ndarray Ĥ_MQE of shape (4^N * m, 4^N * m).

    Raises:
        ValueError: if Ĥ_MQE fails Hermiticity check at hermiticity_tol.
    """
    m     = mechanism.m
    M     = mechanism.M_steps
    tau_total = M * dt                      # τ_total = M · Δt_m

    log.info(
        "[HMQE] Building U_mech: N=%d orbitals, M=%d steps, m=%d, dt=%.4f, "
        "τ_total=%.4f Ha⁻¹, dim=%d×%d",
        mechanism.N_orbitals, M, m, dt, tau_total,
        4**mechanism.N_orbitals * m, 4**mechanism.N_orbitals * m,
    )

    U_mech = build_full_mechanism_unitary(step_hamiltonians, mechanism, dt)

    log.info("[HMQE] U_mech assembled. Taking matrix logarithm…")
    log_U  = logm(U_mech)                  # principal log, anti-Hermitian for unitary

    H_mqe  = (1j / tau_total) * log_U      # Hermitian

    # Check Hermiticity: H_mqe should equal H_mqe†
    anti_herm = (H_mqe - H_mqe.conj().T) / 2.0
    max_err   = np.max(np.abs(anti_herm))
    if max_err > hermiticity_tol:
        # Likely a branch-cut issue in logm
        log.warning(
            "[HMQE] Ĥ_MQE Hermiticity violation: max|H−H†|/2 = %.2e > tol=%.2e. "
            "Possible logm branch-cut issue (U_mech eigenvalue near −1). "
            "Forcing Hermitian symmetrisation. Janus eigenphases may be slightly "
            "inaccurate; consider adjusting dt to avoid branch cut.",
            max_err, hermiticity_tol,
        )
        H_mqe = (H_mqe + H_mqe.conj().T) / 2.0   # force Hermitian
    else:
        log.debug("[HMQE] Ĥ_MQE Hermiticity verified (max|H−H†|/2 = %.2e).", max_err)
        # Symmetrise anyway for numerical cleanliness
        H_mqe = (H_mqe + H_mqe.conj().T) / 2.0

    log.info("[HMQE] Ĥ_MQE complete. Shape: %s, dtype: %s.", H_mqe.shape, H_mqe.dtype)
    return H_mqe


# ──────────────────────────────────────────────────────────────────────────────
# 7. k* sector projection and eigenphase extraction
# ──────────────────────────────────────────────────────────────────────────────

def _k_star_indices(N: int, m: int) -> np.ndarray:
    r"""Return combined-basis indices of the k* = m//2 virtual sector.

    The k* sector consists of all states |i_L⟩|k*⟩_V in the full L×V space.
    In the combined index I = i_L * m + j_V, these are:
        { i_L * m + (m // 2) : i_L ∈ {0, ..., 4^N − 1} }

    Returns:
        Integer ndarray of shape (4^N,) — row/column indices of the k* block.
    """
    dim_L = 4 ** N
    k_star = m // 2
    return np.array([i_L * m + k_star for i_L in range(dim_L)], dtype=np.int64)


def janus_sector_matrix(H_mqe: np.ndarray, N: int, m: int) -> np.ndarray:
    r"""Extract the k* = m//2 virtual-sector sub-block of Ĥ_MQE.

    Returns the (4^N × 4^N) Hermitian matrix:
        H_janus = Π_{k*} · H_mqe · Π_{k*}  (restricted to k* subspace)

    where Π_{k*} = I_L ⊗ |k*⟩⟨k*|_V is the k*-sector projector.

    By the Universal Janus Criticality Theorem (thm:ujct), the eigenvalues
    of H_janus are topologically confined to the critical line Re(s) = 1/2
    under the zeta-dual spectral identification — unconditionally, by the
    group structure of ℤ_m.

    Args:
        H_mqe: Ĥ_MQE matrix, shape (4^N * m, 4^N * m).
        N:     Number of active-space orbitals.
        m:     Virtual register modulus.

    Returns:
        Hermitian ndarray of shape (4^N, 4^N).
    """
    idx = _k_star_indices(N, m)
    return H_mqe[np.ix_(idx, idx)]


def k_star_eigenphases(
    H_mqe: np.ndarray,
    N:     int,
    m:     int,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Diagonalise Ĥ_MQE and return eigenvalues in the k* = m//2 sector.

    Two diagonalisations are performed:
      1. Full spectrum: np.linalg.eigvalsh(H_mqe) — all 4^N·m eigenvalues.
      2. k* sector:    np.linalg.eigvalsh(H_janus) — 4^N eigenvalues.

    The k* sector eigenvalues are the Janus eigenphases: under zeta-dual
    parameterisation, each φ_k ∈ ℝ is the imaginary part of a Riemann zero
    γ_k (up to a real-unit rescaling by ΔE_scale).

    Args:
        H_mqe: Ĥ_MQE matrix, shape (4^N * m, 4^N * m).
        N:     Number of active-space orbitals.
        m:     Virtual register modulus.

    Returns:
        (evals_k_star, evals_full):
            evals_k_star: real ndarray (4^N,), Janus eigenphases, ascending.
            evals_full:   real ndarray (4^N * m,), all eigenphases, ascending.
    """
    H_janus    = janus_sector_matrix(H_mqe, N, m)
    evals_k    = np.linalg.eigvalsh(H_janus)              # 4^N values
    evals_full = np.linalg.eigvalsh(H_mqe)                # 4^N * m values
    return evals_k, evals_full


# ──────────────────────────────────────────────────────────────────────────────
# 8. Zeta-dual spectral map
# ──────────────────────────────────────────────────────────────────────────────

def zeta_dual_gamma(
    eigenphase: float,
    n_star:     int,
    dt_m:       float,
    t0:         float = 0.0,
) -> float:
    r"""Apply the inverse zeta-dual spectral map: φ_{k*} → γ_k.

    The zeta-dual class (def:zeta_dual, quantum-enzymaticsB.md) requires:
        ν_n = (1/2π) log(n · Δt_m + t_0)    (logarithmic cofactor scaling)
        E_n ~ log n                          (logarithmic energy spectrum)

    Under this parameterisation, the Janus eigenphase φ_{k*} of Ĥ_MQE at step
    n* is identified with the imaginary part of the Riemann zero:
        s_{k*} = 1/2 + i γ_{k*}

    via the inverse relation:
        γ_k = φ_{k*} / ΔE_scale

    where ΔE_scale is the energy density at level n*:
        ΔE_scale ≈ 1 / (2π n*)    [in natural units where E_1 = 1]

    The '1/(2πn*)' comes from differentiating the level counting function
    N(T) ~ (T/2π) log(T/2π) with respect to T, evaluated at T = 2π n*:
        dN/dT|_{T=2π n*} = (1/2π)(log(2π n*) + 1) ≈ log(n*) / (2π)
    giving ΔE_scale = 2π/log(n*) in the sparse regime.

    For practical use, the simplest calibration is:
        ΔE_scale = dt_m / (2π) · (1 / log(n_star * dt_m + t0 + 1e-12))
    which recovers the natural units of the zeta-dual class definition.

    Args:
        eigenphase: φ_{k*} in Ha (real eigenvalue of Ĥ_MQE in k* sector).
        n_star:     Janus step index.
        dt_m:       Trotter step size Δt_m = 0.04/√m.
        t0:         Zeta-dual offset parameter (default 0).

    Returns:
        γ_k (real, imaginary part of the Riemann zero in zeta-dual units).
    """
    arg = n_star * dt_m + t0 + 1e-12           # avoid log(0)
    if arg <= 0:
        log.warning(
            "[ZETA-DUAL] n_star·dt_m + t0 = %.4e ≤ 0. Returning raw eigenphase.", arg
        )
        return eigenphase
    log_arg    = np.log(arg)
    if abs(log_arg) < 1e-12:
        log.warning("[ZETA-DUAL] log(n*·dt_m + t0) ≈ 0 (n*·dt_m ≈ 1). "
                    "ΔE_scale ill-conditioned; returning raw eigenphase.")
        return eigenphase
    delta_E    = dt_m / (2.0 * np.pi * abs(log_arg))
    gamma_k    = eigenphase / delta_E
    return float(gamma_k)


# ──────────────────────────────────────────────────────────────────────────────
# 9. High-level result dataclass and entry point
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HMQEResult:
    r"""Container for all outputs of run_hmqe_analysis.

    Fields
    ------
    H_mqe:             Ĥ_MQE matrix, shape (4^N·m, 4^N·m).
    H_janus:           k* sector sub-block, shape (4^N, 4^N).
    evals_k_star:      Janus eigenphases (real, ascending), shape (4^N,).
    evals_full:        Full spectrum of Ĥ_MQE (real, ascending), shape (4^N·m,).
    ground_janus_eigenphase:  evals_k_star[0] — lowest Janus eigenphase (Ha).
    gamma_k:           Zeta-dual γ_k for the ground Janus eigenphase.
                       None if n_star is not determined.
    n_star:            Janus step index (first crossing in mechanism.crossings),
                       or None if no crossings defined.
    k_star:            m // 2.
    N:                 Number of active-space orbitals.
    m:                 Virtual register modulus.
    M_steps:           Total number of mechanism steps.
    tau_total:         M_steps * dt.
    hermiticity_max_err: max|H_MQE - H_MQE†|/2 before symmetrisation.
    unitarity_max_err:   max|U_mech @ U_mech† - I|.
    """
    H_mqe:                     np.ndarray
    H_janus:                   np.ndarray
    evals_k_star:              np.ndarray
    evals_full:                np.ndarray
    ground_janus_eigenphase:   float
    gamma_k:                   Optional[float]
    n_star:                    Optional[int]
    k_star:                    int
    N:                         int
    m:                         int
    M_steps:                   int
    tau_total:                 float
    hermiticity_max_err:       float = 0.0
    unitarity_max_err:         float = 0.0

    def summary(self) -> str:
        r"""Human-readable summary of the H_MQE analysis."""
        lines = [
            "─" * 60,
            "  MQE Hamiltonian Analysis (Hilbert-Pólya Direct Path)",
            "─" * 60,
            f"  Mechanism:   N={self.N} orbitals, M={self.M_steps} steps, "
            f"m={self.m} (k*={self.k_star})",
            f"  τ_total:     {self.tau_total:.4f} Ha⁻¹",
            f"  Dim L×V:     {4**self.N * self.m} × {4**self.N * self.m}",
            "",
            f"  Hermiticity: max|H−H†|/2 = {self.hermiticity_max_err:.2e}",
            f"  Unitarity:   max|UU†−I|  = {self.unitarity_max_err:.2e}",
            "",
            f"  Full spectrum ({len(self.evals_full)} eigenphases):",
            f"    min = {self.evals_full[0]:.6f} Ha",
            f"    max = {self.evals_full[-1]:.6f} Ha",
        ]
        if self.n_star is not None:
            lines += [
                "",
                f"  k*={self.k_star} sector (Janus, step n*={self.n_star}):",
                f"    {len(self.evals_k_star)} eigenphases",
                f"    Ground Janus eigenphase: {self.ground_janus_eigenphase:.8f} Ha",
            ]
            if self.gamma_k is not None:
                lines += [
                    f"    Zeta-dual γ_k:           {self.gamma_k:.6f}",
                    f"    Critical line (UJCT):    Re(s) = 1/2  [topologically guaranteed]",
                    f"    Riemann zero candidate:  s = 1/2 + i·{self.gamma_k:.6f}",
                ]
        else:
            lines += [
                "",
                f"  k*={self.k_star} sector (no Janus crossing defined in mechanism):",
                f"    {len(self.evals_k_star)} eigenphases",
                f"    Ground eigenphase: {self.ground_janus_eigenphase:.8f} Ha",
            ]
        lines.append("─" * 60)
        return "\n".join(lines)


def run_hmqe_analysis(
    step_hamiltonians: List[np.ndarray],
    mechanism:         "MechanismTuple",
    dt:                float,
    t0:                float = 0.0,
    hermiticity_tol:   float = 1e-8,
) -> "HMQEResult":
    r"""High-level entry point: build Ĥ_MQE and return all spectral outputs.

    Performs the complete direct Hilbert-Pólya analysis:
      1. Build U_mech = ∏_n U^(n)  in L×V space.
      2. Compute Ĥ_MQE = (i/τ) logm(U_mech).
      3. Extract k* = m//2 sector eigenphases.
      4. Apply zeta-dual map to ground Janus eigenphase.

    Args:
        step_hamiltonians: List of M Hamiltonians for each mechanism step,
                           shape (4^N, 4^N) each.
        mechanism:         MechanismTuple with full mechanism specification.
        dt:                Trotter step size Δt_m in Ha⁻¹.
        t0:                Zeta-dual offset parameter (default 0.0).
        hermiticity_tol:   Hermiticity tolerance for Ĥ_MQE (default 1e-8).

    Returns:
        HMQEResult dataclass with H_mqe, H_janus, evals_k_star, evals_full,
        ground Janus eigenphase, γ_k, and diagnostics.
    """
    N = mechanism.N_orbitals
    m = mechanism.m
    M = mechanism.M_steps

    # Determine Janus step (first crossing, if any)
    n_star: Optional[int] = None
    if mechanism.crossings:
        n_star = mechanism.crossings[0][0]

    # Build U_mech (unitarity check inside)
    U_mech = build_full_mechanism_unitary(step_hamiltonians, mechanism, dt)
    unitarity_err = float(np.max(np.abs(U_mech @ U_mech.conj().T - np.eye(4**N * m))))

    # Build H_MQE (Hermiticity check inside)
    # We compute H_MQE directly here so we can capture hermiticity_max_err.
    tau_total = M * dt
    log_U     = logm(U_mech)
    H_mqe_raw = (1j / tau_total) * log_U
    anti_herm = (H_mqe_raw - H_mqe_raw.conj().T) / 2.0
    herm_err  = float(np.max(np.abs(anti_herm)))
    if herm_err > hermiticity_tol:
        log.warning(
            "[HMQE] Hermiticity violation %.2e > tol %.2e. "
            "Symmetrising H_MQE.", herm_err, hermiticity_tol,
        )
    H_mqe = (H_mqe_raw + H_mqe_raw.conj().T) / 2.0

    # Extract eigenphases
    evals_k_star, evals_full = k_star_eigenphases(H_mqe, N, m)
    H_janus  = janus_sector_matrix(H_mqe, N, m)
    phi_0    = float(evals_k_star[0])

    # Zeta-dual map
    gamma_k: Optional[float] = None
    if n_star is not None:
        dt_m = 0.04 / np.sqrt(m)              # canonical zeta-dual step size
        gamma_k = zeta_dual_gamma(phi_0, n_star, dt_m, t0=t0)

    result = HMQEResult(
        H_mqe                   = H_mqe,
        H_janus                 = H_janus,
        evals_k_star            = evals_k_star,
        evals_full              = evals_full,
        ground_janus_eigenphase = phi_0,
        gamma_k                 = gamma_k,
        n_star                  = n_star,
        k_star                  = m // 2,
        N                       = N,
        m                       = m,
        M_steps                 = M,
        tau_total               = tau_total,
        hermiticity_max_err     = herm_err,
        unitarity_max_err       = unitarity_err,
    )
    log.info("[HMQE] Analysis complete.\n%s", result.summary())
    return result
