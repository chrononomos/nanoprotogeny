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
mqescf.py — Seed-free MO layer.
==============================
Core-Hamiltonian MO guess (J = K = 0, one generalised diagonalisation), the
frozen-core effective Hamiltonian, and the AO→MO integral transform.  No SCF
iteration.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import eigh as _scipy_eigh

log = logging.getLogger(__name__)

def build_core_ham_guess(
    h1_AO:             np.ndarray,
    S_AO:              np.ndarray,
    N_active:          int,
    N_electrons_total: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the generalised eigenvalue problem h1_AO C = S_AO C ε.

    This is the core-Hamiltonian guess with J = K = 0.  A single
    diagonalisation — no SCF iteration.

    MO selection
    ------------
    When ``N_electrons_total`` is given (recommended for large AO bases such
    as those produced by PySCF for transition-metal systems), the ``N_active``
    MOs are chosen centred on the Fermi level:

        n_occ  = N_electrons_total // 2          (closed-shell Aufbau)
        start  = max(0, n_occ − N_active // 2)
        window = [start, start + N_active)

    This ensures that the active space spans HOMO/LUMO-adjacent MOs (3d-derived
    for Fe₂S₂) rather than deep core orbitals.

    When ``N_electrons_total`` is None or the AO basis is no larger than
    N_active, the N_active lowest-energy MOs are returned (original behaviour,
    correct for the 4-AO analytical s-only path).

    Args:
        h1_AO             : (N_AO, N_AO) core Hamiltonian in the AO basis.
        S_AO              : (N_AO, N_AO) AO overlap matrix.
        N_active          : Number of active MOs to retain.
        N_electrons_total : Total electron count for Fermi-level centering.
                            Pass ``sum(Z_α)`` for the full molecule.

    Returns:
        C_0   : (N_AO, N_active) MO coefficient matrix (Stiefel manifold).
        eps_0 : (N_active,) orbital energies [Ha].
    """
    N_AO   = h1_AO.shape[0]
    N_keep = min(N_active, N_AO)
    eps_all, C_all = _scipy_eigh(h1_AO, S_AO)
    sorted_idx = np.argsort(eps_all)

    if N_electrons_total is not None and N_AO > N_keep:
        # Fermi-level window: centre the active space on the HOMO/LUMO gap.
        n_occ = N_electrons_total // 2          # doubly occupied orbitals (Aufbau)
        n_occ = min(n_occ, N_AO - 1)           # clamp to valid range
        # For a valence-only basis (e.g. 3d+3p only, no core), the total electron
        # count far exceeds N_AO.  In that case treat the basis as half-filled:
        # the frontier orbitals are at N_AO // 2.
        if n_occ >= N_AO - 1:
            n_occ = N_AO // 2
        start  = max(0, n_occ - N_keep // 2)
        start  = min(start, N_AO - N_keep)     # ensure full window fits
        window = sorted_idx[start : start + N_keep]
        log.info(
            f"[zetazero] Fermi-level active space: "
            f"N_elec={N_electrons_total}, n_occ_eff={n_occ}, N_AO={N_AO}, "
            f"window=[{start},{start+N_keep}), "
            f"eps=[{eps_all[window[0]]:.3f}, …, {eps_all[window[-1]]:.3f}] Ha"
        )
    else:
        window = sorted_idx[:N_keep]

    C_0   = C_all[:, window]
    eps_0 = eps_all[window]
    # Verify orthonormality: C^T S C = I (up to numerical noise).
    err   = np.max(np.abs(C_0.T @ S_AO @ C_0 - np.eye(N_keep)))
    if err > 1.0e-6:
        log.warning(f"[zetazero] C_0 orthonormality error = {err:.2e}")
    return C_0, eps_0


def build_frozen_core_ham(
    h1_AO:             np.ndarray,
    g_AO:              np.ndarray,
    S_AO:              np.ndarray,
    N_active:          int,
    N_electrons_total: int,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    r"""Frozen-core effective Hamiltonian for CAS(N_e, N_active).

    Partitions the core-Hamiltonian MOs (from a single h1_AO diagonalisation)
    into three sets:

    * **core**    — MOs below the Fermi-level active window, doubly occupied
                    in the Aufbau reference.
    * **active**  — ``N_active`` MOs centred on the Fermi level (same Fermi
                    window as :func:`build_core_ham_guess`).
    * **virtual** — remaining MOs above the active window (discarded).

    The frozen-core effective integrals are (chemist notation ``(pq|rs)``):

    .. math::

        h^{\text{eff}}_{pq} = h_{pq}
            + \sum_{i \in \text{core}} \bigl(2(pq|ii) - (pi|qi)\bigr)

        E_{\text{core}} = \sum_{i \in \text{core}} 2 h_{ii}
            + \sum_{i,j \in \text{core}} \bigl(2(ii|jj) - (ij|ji)\bigr)

    The total energy then decomposes as::

        E_total = E_elec_active + E_core + E_nuc

    where ``E_elec_active`` is the CAS FCI energy on ``h1_eff`` and
    ``g_active`` (the active–active–active–active ERI block).

    Args:
        h1_AO             : (N_AO, N_AO) core Hamiltonian T + V [Ha].
        g_AO              : (N_AO, N_AO, N_AO, N_AO) ERI tensor, chemist (μν|λσ).
        S_AO              : (N_AO, N_AO) overlap matrix.
        N_active          : Number of active spatial orbitals.
        N_electrons_total : Total electron count for Fermi-level centering
                            (pass ``Σ Z_α`` over the molecule).

    Returns:
        h1_eff    : (N_active, N_active) Fock-screened 1e integrals [Ha].
        g_active  : (N_active, N_active, N_active, N_active) 2e ERIs [Ha].
        E_core    : frozen-core energy [Ha] — add to CAS E_elec + E_nuc.
        C_active  : (N_AO, N_active) MO coefficient matrix for the active space.
        eps_active: (N_active,) core-Ham orbital energies for the active MOs [Ha].
        C_core    : (N_AO, N_core) MO coefficient matrix for frozen-core MOs.
                    Shape is (N_AO, 0) when N_core == 0.  Returned so callers
                    can build the AO-basis Fock screening (_build_h1_AO_eff)
                    without a second diagonalisation.
    """
    N_AO   = h1_AO.shape[0]
    N_keep = min(N_active, N_AO)

    # Full diagonalisation of the core Hamiltonian (J = K = 0).
    eps_all, C_all = _scipy_eigh(h1_AO, S_AO)
    sorted_idx = np.argsort(eps_all)

    # Fermi-level active window — mirrors build_core_ham_guess.
    n_occ = N_electrons_total // 2
    n_occ = min(n_occ, N_AO - 1)
    if n_occ >= N_AO - 1:
        n_occ = N_AO // 2
    start = max(0, n_occ - N_keep // 2)
    start = min(start, N_AO - N_keep)

    core_idx   = sorted_idx[:start]                  # MOs below active window
    active_idx = sorted_idx[start : start + N_keep]  # active window

    N_core     = len(core_idx)
    C_active   = C_all[:, active_idx]    # (N_AO, N_active)
    eps_active = eps_all[active_idx]
    C_core     = C_all[:, core_idx] if N_core > 0 else np.empty((N_AO, 0), dtype=C_all.dtype)

    log.info(
        f"[zetazero] frozen-core: N_AO={N_AO}, N_core={N_core}, "
        f"N_active={N_keep}, N_virtual={N_AO - N_core - N_keep}, "
        f"active window=[{start},{start + N_keep}), "
        f"eps_active=[{eps_active[0]:.3f}, …, {eps_active[-1]:.3f}] Ha"
    )

    if N_core == 0:
        # No frozen-core MOs — identical to bare core-Ham path.
        h1_MO, g_MO = transform_integrals(C_active, h1_AO, g_AO)
        return h1_MO, g_MO, 0.0, C_active, eps_active, C_core

    # ── Combined (core + active) MO coefficient matrix ────────────────────
    # Ordering: core block first (indices 0..N_core-1), then active.
    ca_idx = np.concatenate([core_idx, active_idx])
    C_ca   = C_all[:, ca_idx]           # (N_AO, N_core+N_active)
    N_ca   = N_core + N_keep
    nc     = N_core                     # shorthand for slice offset

    # 1e transform to the combined basis.
    h1_ca = C_ca.T @ h1_AO @ C_ca      # (N_ca, N_ca)

    # 4-index transform to the combined basis.
    # Only the (N_ca)^4 subspace is needed; g_AO has shape (N_AO)^4.
    tmp  = np.einsum("up,uvlm->pvlm", C_ca, g_AO)
    tmp  = np.einsum("vq,pvlm->pqlm", C_ca, tmp)
    tmp  = np.einsum("lr,pqlm->pqrm", C_ca, tmp)
    g_ca = np.einsum("ms,pqrm->pqrs", C_ca, tmp)   # (N_ca, N_ca, N_ca, N_ca)

    # ── E_core: energy of doubly-occupied core MOs ────────────────────────
    # E_core = Σ_{i∈core} [2 h[i,i] + Σ_{j∈core} (2(ii|jj) − (ij|ji))]
    E_core = 0.0
    for i in range(nc):
        E_core += 2.0 * float(h1_ca[i, i])
        for j in range(nc):
            E_core += 2.0 * float(g_ca[i, i, j, j]) - float(g_ca[i, j, j, i])

    log.info(f"[zetazero] frozen-core: E_core = {E_core:.6f} Ha")

    # ── h1_eff: Fock-screened 1e integrals for the active space ──────────
    # h1_eff[p,q] = h1_ca[p+nc, q+nc]
    #             + Σ_{i∈core} (2·g_ca[p+nc, q+nc, i, i]  [Coulomb]
    #                             − g_ca[p+nc, i, q+nc, i]) [Exchange]
    h1_eff = h1_ca[nc:, nc:].copy()                # (N_active, N_active)
    for i in range(nc):
        h1_eff += 2.0 * g_ca[nc:, nc:, i, i]      # 2*(pq|ii)
        h1_eff -= g_ca[nc:, i, nc:, i]             # (pi|qi) exchange

    # ── g_active: active–active–active–active ERI block ──────────────────
    g_active = g_ca[nc:, nc:, nc:, nc:].copy()     # (N_active,)^4

    return h1_eff, g_active, E_core, C_active, eps_active, C_core


def transform_integrals(
    C_0:   np.ndarray,
    h1_AO: np.ndarray,
    g_AO:  np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """AO→MO integral transformation.

    h1_MO[p,q]     = Σ_{μν} C_{μp} h1_AO_{μν} C_{νq}
    g_MO[p,q,r,s]  = Σ_{μνλσ} C_{μp} C_{νq} g_AO_{μνλσ} C_{λr} C_{σs}

    Args:
        C_0   : (N_AO, N_MO) coefficient matrix.
        h1_AO : (N_AO, N_AO) core Hamiltonian.
        g_AO  : (N_AO, N_AO, N_AO, N_AO) ERI tensor.

    Returns:
        h1_MO : (N_MO, N_MO)
        g_MO  : (N_MO, N_MO, N_MO, N_MO)
    """
    h1_MO = C_0.T @ h1_AO @ C_0
    # Four-index transform: g_MO = C^T C^T g_AO C C (using einsum)
    tmp   = np.einsum("up,uvlm->pvlm", C_0, g_AO)
    tmp   = np.einsum("vq,pvlm->pqlm", C_0, tmp)
    tmp   = np.einsum("lr,pqlm->pqrm", C_0, tmp)
    g_MO  = np.einsum("ms,pqrm->pqrs", C_0, tmp)
    return h1_MO, g_MO
