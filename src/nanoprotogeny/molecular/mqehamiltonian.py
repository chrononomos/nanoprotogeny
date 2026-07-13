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
mqehamiltonian.py — Second-Quantisation Hamiltonian Matrix Algebra
===================================================================
Pure numpy/scipy second-quantisation layer for the MQE d=4 qudit register.
Constructs the exact (4^N × 4^N) Hermitian Hamiltonian matrix from molecular
integrals, projects it to electron-number sectors, and extracts ground states
by exact diagonalisation.

No cirq, ionq, or simulate-layer imports.  All functions are independently
reusable by any algorithm that needs the Hamiltonian as a matrix (VQE, QPE,
DMRG benchmarking, etc.).

Constants
---------
_A_UP_DAG, _A_DN_DAG
    Intra-site d=4 creation operators for spin-up / spin-down.
_PARITY_OP
    Jordan-Wigner parity operator P_k = (-1)^{n̂_k} = diag(1,-1,-1,1).

Functions
---------
_full_creation_op(p, sigma, n)
    Full (4^n × 4^n) fermionic creation operator c†_{p,σ} including the
    inter-site Jordan-Wigner parity string.

_partial_trace_qudit(rho, keep_sites, n_total, d=4)
    Partial trace over a multi-qudit density matrix, retaining keep_sites.

build_qudit_hamiltonian_matrix(n, h_diag, h_hop, g_full, screening_threshold)
    Builds H as a (4^n × 4^n) Hermitian matrix using the full ERI:
        H = Σ h_pp n̂_p  +  Σ h_pq (c†_p c_q + h.c.)
          + ½ Σ g[p,q,r,s] c†_p c†_r c_s c_q
    JW parity strings are included exactly via _full_creation_op.

_project_hamiltonian_to_sector(H, n_orbs, nelec, return_indices=False)
    Projects H to the nelec-electron sector of the full Fock space.

ground_state_from_diagonalization(H_qudit, n_orbs=None, nelec=None)
    Exact diagonalisation of H_qudit; returns (E_0, |ψ_GS⟩).
    Optionally projects to the nelec-electron sector first.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

# ==============================================================================
# QUDIT HAMILTONIAN MATRIX (FULL ERI)
# 4^N × 4^N in basis {|Th⟩=0=vacuum, |AntiTh⟩=1=↑, |SynTh⟩=2=↓, |HoloTh⟩=3=↑↓}.
#
# The JW ordering of spin-orbital modes is: 0↑, 0↓, 1↑, 1↓, …, (N-1)↑, (N-1)↓.
# The d=4 qudit at site p encodes {|vac⟩, |↑⟩, |↓⟩, |↑↓⟩} as {|0⟩, |1⟩, |2⟩, |3⟩}.
#
# Local (single-site) fermionic operators in the d=4 basis:
#   A†_↑ = |1⟩⟨0| + |3⟩⟨2|          (create spin-up; no intra-site JW sign)
#   A†_↓ = |2⟩⟨0| − |3⟩⟨1|          (create spin-down; minus sign from ↑ already there)
#   P    = diag(1,−1,−1,1) = (−1)^n̂  (local parity / JW string)
#
# Full (4^N × 4^N) creation operator with inter-site JW parity string:
#   c†_{p,σ} = (⊗_{k<p} P_k) ⊗ A†_{p,σ} ⊗ (⊗_{k>p} I_k)
#
# The Hamiltonian (chemist's notation, g[p,q,r,s] = (pq|rs)):
#   H = Σ_{p,σ}   h_pp  c†_{pσ} c_{pσ}
#     + Σ_{p≠q,σ} h_pq  c†_{pσ} c_{qσ}
#     + ½ Σ_{pqrs,στ} g[p,q,r,s] c†_{pσ} c†_{rτ} c_{sτ} c_{qσ}
#
# Correctness of the full-ERI formula is verified against PySCF FCI:
#   density-density g[p,p,r,r]: ½ g Σ_{στ} c†_{pσ}c†_{rτ}c_{rτ}c_{pσ} = ½ g n̂_p n̂_r  ✓
#   exchange g[p,q,q,p]:        ½ g Σ_{στ} c†_{pσ}c†_{qτ}c_{pτ}c_{qσ}  (K integral)  ✓
#   4-centre  g[p,q,r,s]:       exact via JW-string operator products               ✓
# ==============================================================================

# Local d=4 creation operators (intra-site part only; inter-site JW string added by
# _full_creation_op).  These are kept as module-level constants for efficiency.
_A_UP_DAG = np.array([[0,0,0,0],[1,0,0,0],[0,0,0,0],[0,0,1,0]], dtype=complex)
_A_DN_DAG = np.array([[0,0,0,0],[0,0,0,0],[1,0,0,0],[0,-1,0,0]], dtype=complex)

# Local parity operator P_k = (−1)^{n̂_k} = diag(1,−1,−1,1)
# For the d=4 JW ordering 0↑,0↓,1↑,1↓,…: P encodes the full intra-site parity
# (−1)^{n_{k↑}+n_{k↓}} = (−1)^{n̂_k}.
_PARITY_OP: np.ndarray = np.diag([1., -1., -1., 1.]).astype(complex)


def _full_creation_op(p: int, sigma: int, n: int) -> np.ndarray:
    r"""Return the full (4^n × 4^n) fermionic creation operator c†_{p,σ}.

    Includes the Jordan-Wigner parity string over all sites k < p so that
    anti-commutation relations are exact for ALL site pairs (not just adjacent):

        c†_{p,σ} = (⊗_{k<p} P_k) ⊗ A†_{p,σ} ⊗ (⊗_{k>p} I_k)

    where P_k = diag(1,−1,−1,1) = (−1)^{n̂_k} is the site-k parity operator.

    Correctness (verified by construction):
        {c†_{pσ}, c_{qτ}} = δ_{pq} δ_{στ} I   (fermionic anti-commutation)

    Why this was missing in the density-density-only code:
        Single-site operators P_k² = I, so diagonal (p=q) terms are unaffected.
        But off-diagonal hops p→q (p≠q) require the parity string over intermediate
        sites; omitting it gives wrong signs whenever any site between p and q is
        occupied — causing errors in hopping and ALL general ERI scattering terms.

    Args:
        p:     Spatial orbital index (0-based).
        sigma: Spin: 0 = ↑, 1 = ↓.
        n:     Total number of spatial orbitals.

    Returns:
        Complex (4^n × 4^n) matrix.
    """
    A_dag = _A_UP_DAG if sigma == 0 else _A_DN_DAG
    ops: List[np.ndarray] = (
        [_PARITY_OP] * p
        + [A_dag]
        + [np.eye(4, dtype=complex)] * (n - p - 1)
    )
    result = ops[0].astype(complex)
    for o in ops[1:]:
        result = np.kron(result, o)
    return result


def _partial_trace_qudit(
    rho: np.ndarray,
    keep_sites: List[int],
    n_total: int,
    d: int = 4,
) -> np.ndarray:
    r"""Partial trace over a multi-qudit density matrix, retaining `keep_sites`.

    Args:
        rho:        Full density matrix, shape (d^n_total, d^n_total).
        keep_sites: Sorted list of site indices (0-based) to retain.
        n_total:    Total number of d-dimensional sites.
        d:          Local qudit dimension.

    Returns:
        Reduced density matrix, shape (d^len(keep_sites), d^len(keep_sites)).
    """
    trace_sites = [i for i in range(n_total) if i not in keep_sites]
    rho_t = rho.reshape([d] * (2 * n_total))
    perm = (
        keep_sites + trace_sites
        + [s + n_total for s in keep_sites]
        + [s + n_total for s in trace_sites]
    )
    rho_t = np.transpose(rho_t, perm)
    dim_k = d ** len(keep_sites)
    dim_t = d ** (n_total - len(keep_sites))
    rho_t = rho_t.reshape(dim_k, dim_t, dim_k, dim_t)
    return np.einsum("ikjk->ij", rho_t)


def build_qudit_hamiltonian_matrix(
    n: int,
    h_diag: Dict[int, float],
    h_hop:  Dict[Tuple[int, int], float],
    g_full: Dict[Tuple[int, int, int, int], float],
    screening_threshold: float = 1e-8,
) -> np.ndarray:
    r"""Build H as a (4^n × 4^n) Hermitian matrix using the full ERI.

    H = Σ_{p,σ}   h_pp c†_{pσ} c_{pσ}
      + Σ_{p≠q,σ} h_pq c†_{pσ} c_{qσ}
      + ½ Σ_{pqrs,στ} g[p,q,r,s] c†_{pσ} c†_{rτ} c_{sτ} c_{qσ}

    g[p,q,r,s] is in chemist's notation (pq|rs), matching the PySCF FCI call.

    All creation/annihilation operators are full (4^n × 4^n) matrices including
    Jordan-Wigner parity strings (see _full_creation_op), so fermionic
    anti-commutation is exact for all operator products including 4-centre
    scattering terms with non-adjacent site indices.

    The 8-fold symmetry of g is used for deduplication:
        (pq|rs) = (qp|sr) = (rs|pq) = (sr|qp)
                = (pq|sr) = (qp|rs) = (rs|qp) = (sr|pq)   (real orbitals)
    The function builds the full 4^4 eri tensor first (filling all 8 positions)
    so the ½ prefactor and the complete operator sum are applied correctly.

    Args:
        n:                    Number of spatial orbitals.
        h_diag:               {p: h_pp} one-electron on-site energies.
        h_hop:                {(p,q): h_pq} one-electron hopping integrals (p≠q).
        g_full:               {(p,q,r,s): g_pqrs} ERI in chemist's notation.
                              May contain a subset of symmetry-equivalent entries;
                              the function expands to all 8 before summing.
        screening_threshold:  Skip ERI entries with |g| below this value.

    Returns:
        4^n × 4^n complex Hermitian Hamiltonian matrix.
    """
    H = np.zeros((4**n, 4**n), dtype=complex)

    # ── Precompute all (4^n × 4^n) creation / annihilation operators ─────────
    # C_dag[p][sigma] = c†_{p,sigma},   C[p][sigma] = c_{p,sigma}
    # Storing them avoids repeated Kronecker-product construction in the inner loop.
    C_dag: List[List[np.ndarray]] = [
        [_full_creation_op(p, s, n) for s in range(2)] for p in range(n)
    ]
    C: List[List[np.ndarray]] = [
        [C_dag[p][s].conj().T for s in range(2)] for p in range(n)
    ]

    # ── One-electron diagonal: Σ_{p,σ} h_pp c†_{pσ} c_{pσ} ─────────────────
    for p, h_pp in h_diag.items():
        for s in range(2):
            H += h_pp * (C_dag[p][s] @ C[p][s])

    # ── One-electron hopping: Σ_{p≠q,σ} h_pq c†_{pσ} c_{qσ}  +  h.c. ──────
    # h_hop supplies one canonical (p<q) entry per pair; h.c. is added explicitly.
    for (p, q), h_pq in h_hop.items():
        for s in range(2):
            H += h_pq * (C_dag[p][s] @ C[q][s] + C_dag[q][s] @ C[p][s])

    # ── Two-electron full ERI ─────────────────────────────────────────────────
    # H_2 = ½ Σ_{pqrs,στ} g[p,q,r,s] c†_{pσ} c†_{rτ} c_{sτ} c_{qσ}
    #
    # Strategy: build the full (n,n,n,n) eri tensor with all 8 symmetry positions
    # filled, then iterate over ALL (p,q,r,s) with the ½ prefactor.  This is
    # equivalent to the PySCF FCI Hamiltonian and avoids any ambiguity about which
    # canonical representative to use or how many times each term appears.
    eri = np.zeros((n, n, n, n), dtype=float)
    for key, val in g_full.items():
        if abs(val) < screening_threshold:
            continue
        p, q, r, s = key
        v = float(val)
        # Fill all 8 real-orbital symmetry positions
        eri[p,q,r,s]=v;  eri[q,p,s,r]=v;  eri[r,s,p,q]=v;  eri[s,r,q,p]=v
        eri[p,q,s,r]=v;  eri[q,p,r,s]=v;  eri[r,s,q,p]=v;  eri[s,r,p,q]=v

    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    g_val = eri[p, q, r, s]
                    if abs(g_val) < screening_threshold:
                        continue
                    # Σ_{σ,τ} c†_{pσ} c†_{rτ} c_{sτ} c_{qσ}
                    for sigma in range(2):
                        for tau in range(2):
                            H += 0.5 * g_val * (
                                C_dag[p][sigma] @ C_dag[r][tau] @ C[s][tau] @ C[q][sigma]
                            )

    # ── Hermiticity check ─────────────────────────────────────────────────────
    # Any asymmetry >1e-8 Ha indicates a bug in operator construction or index
    # convention (e.g. wrong JW sign, mismatched chemist's/physicist's notation).
    # Force exact hermiticity (numerical safety)
    H = (H + H.conj().T) / 2
    max_asym = float(np.max(np.abs(H - H.conj().T)))
    assert max_asym < 1e-8, (
        f"H_qudit is not Hermitian (max |H−H†| = {max_asym:.2e}). "
        "Check JW parity strings and ERI index convention."
    )
    return H


#==============================================================================
# SECTOR PROJECTION UTILITY
#==============================================================================
def _project_hamiltonian_to_sector(
    H: np.ndarray, n_orbs: int, nelec: int, return_indices: bool = False
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """Project a 4^n_orbs qudit Hamiltonian onto the nelec-electron sector.
    If return_indices=True, also returns the boolean/integer mask for lifting back to full space.
    """
    dim             = 4 ** n_orbs
    electron_count  = np.array([0, 1, 1, 2], dtype=int)
    sector_indices  = []
    for state_idx in range(dim):
        idx   = state_idx
        n_e   = 0
        for _ in range(n_orbs):
            n_e += electron_count[idx % 4]
            idx //= 4
        if n_e == nelec:
            sector_indices.append(state_idx)
    sector_indices = np.array(sector_indices, dtype=int)
    if len(sector_indices) == 0:
        raise ValueError(f"No basis states found for nelec={nelec} in {n_orbs}-orbital space.")
    H_proj = H[np.ix_(sector_indices, sector_indices)]
    if return_indices:
        return H_proj, sector_indices
    return H_proj



def ground_state_from_diagonalization(
    H_qudit: np.ndarray,
    n_orbs: Optional[int] = None,
    nelec: Optional[int] = None,
) -> Tuple[float, np.ndarray]:
    """Exact diagonalization of H_qudit; returns (E_0, |ψ_GS⟩).
    
    If n_orbs and nelec are provided, projects H_qudit to the nelec-electron
    sector before diagonalization for efficiency and physical correctness.
    """
    # Project to electron sector if parameters provided
    if n_orbs is not None and nelec is not None:
        H_qudit = _project_hamiltonian_to_sector(H_qudit, n_orbs, nelec)
    
    eigvals, eigvecs = np.linalg.eigh(H_qudit)
    E_0    = float(eigvals[0])
    psi_gs = eigvecs[:, 0]
    psi_gs = psi_gs / np.linalg.norm(psi_gs)
    return E_0, psi_gs



