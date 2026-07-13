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
mqeseedtensors.py — Full-MO tensor seed, algebraic slicing, and tower pipeline.
=============================================================================
Step 0 of sec:algebraic_tower: one diagonalisation → full MO-basis integral
tensors; algebraic CASCI slicing to active set A_k; seed-tensor .npz I/O; and
the end-to-end algebraic Iwasawa tower pipeline (single FCI solve at k=2).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.linalg import eigh as _scipy_eigh

from nanoprotogeny.molecular.mqeconstants import _EPS_MILLI_HA, _K_BASE
from nanoprotogeny.molecular.mqeaointegrals import _pack_g_mo, _unpack_g_mo
from nanoprotogeny.molecular.mqefci import solve_cas
from nanoprotogeny.molecular.mqescf import transform_integrals
from nanoprotogeny.molecular.mqetower import compute_tower, padicinterp_energy

log = logging.getLogger(__name__)

def build_full_mo_tensors(
    h1_AO:        np.ndarray,
    g_AO:         Optional[np.ndarray],
    S_AO:         np.ndarray,
    E_nuc:        float,
    N_frozen:     int,
    N_elec_total: int,
    N_total_orbs: int                       = 0,
    localize:     bool                      = False,
    screened:     bool                      = False,
    shells:       Optional[list]            = None,
    norms:        Optional[list]            = None,
    sph_C:        Optional[np.ndarray]      = None,
    schwarz_thr:  float                     = 1.0e-10,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    r"""Compute h1_MO∈ℝ^{N×N} and g_MO∈ℝ^{N^4} over ALL N MOs.

    Implements Step~0 of sec:algebraic_tower: one diagonalisation of h1_AO
    w.r.t. S_AO yields the canonical MO coefficient matrix C∈ℝ^{N_AO×N}.
    The full MO-basis integral tensors are then:

        h1_MO[p,q]      = (C^T h1_AO C)[p,q]
        g_MO[p,q,r,s]   = Σ_{μνλσ} C[μ,p] C[ν,q] C[λ,r] C[σ,s] g_AO[μ,ν,λ,σ]

    The frozen-core energy (eq:ecore) is fixed from the N_frozen lowest MOs:

        E_core = E_nuc
               + Σ_{f<N_frozen} [ 2 h1_MO[f,f]
                 + Σ_{g<N_frozen} (2 g_MO[f,f,g,g] − g_MO[f,g,g,f]) ]

    No further quantum-chemistry call is needed after this function.

    Args:
        h1_AO        : (N_AO, N_AO) core Hamiltonian (T + V_ne) [Ha].
        g_AO         : (N_AO, N_AO, N_AO, N_AO) ERI tensor, chemist (μν|λσ).
                       Required when ``screened=False``; pass ``None`` when
                       ``screened=True`` (ERIs are computed on-the-fly).
        S_AO         : (N_AO, N_AO) overlap matrix.
        E_nuc        : Nuclear repulsion energy [Ha].
        N_frozen     : Number of frozen-core MOs (lowest eigenvalue MOs of h1_AO).
        N_elec_total : Total electron count for Fermi-level MO ordering.
        N_total_orbs : Number of MOs to keep.  0 (default) = all N_AO MOs.
        localize     : If True, apply Boys orbital localization to the canonical
                       MO coefficient matrix before the 4-index transform.
                       Requires ``shells`` and ``norms``.
        screened     : If True, use the Schwarz-screened integral-direct
                       AO→MO transform instead of the dense einsum.  Avoids
                       materialising g_AO; requires ``shells`` and ``norms``.
                       Incompatible with ``g_AO`` being pre-supplied (pass None).
        shells       : Cartesian shell list (from ``_build_basis_shells``).
                       Required when ``localize=True`` or ``screened=True``.
        norms        : Contracted norms (one per shell in *shells*).
                       Required when ``localize=True`` or ``screened=True``.
        sph_C        : Cartesian→spherical transform (N_cart, N_sph) returned
                       by ``build_ao_integrals_with_shells``.  Used only when
                       ``screened=True`` and the basis is spherical; the
                       combined coefficient matrix C_cart = sph_C @ C_sph is
                       passed to the screened transform.
        schwarz_thr  : Schwarz screening threshold for the direct transform.

    Returns:
        h1_MO  : (N, N) 1e integrals in the MO basis [Ha].
        g_MO   : (N, N, N, N) ERIs in the MO basis [Ha] (chemist notation).
        E_core : frozen-core energy [Ha]; add to active-space E_elec only.
                 E_core = E_nuc + Σ_f[2·h1[f,f] + Σ_g(2·g[f,f,g,g]−g[f,g,g,f])],
                 so E_nuc is already included — do NOT add E_nuc separately.
        C      : (N_AO, N) MO coefficient matrix (S-orthonormal columns).
    """
    N_AO = h1_AO.shape[0]
    if N_total_orbs <= 0 or N_total_orbs > N_AO:
        N_total_orbs = N_AO

    # ── One diagonalisation: all N_AO eigenvectors ───────────────────────────
    eps_all, C_all = _scipy_eigh(h1_AO, S_AO)          # S-orthonormal columns
    sorted_idx = np.argsort(eps_all)

    # The N_frozen lowest MOs always come from the very bottom of the spectrum.
    # Strategy: take sorted_idx[0 : N_total_orbs] so that frozen core indices
    # are 0..N_frozen-1 and active/virtual are N_frozen..N_total_orbs-1.
    mo_idx = sorted_idx[:N_total_orbs]
    C      = C_all[:, mo_idx]                           # (N_AO, N_total_orbs)
    N      = N_total_orbs

    log.info(
        f"[full_mo] N_AO={N_AO}, N_total_orbs={N}, N_frozen={N_frozen}, "
        f"eps range=[{eps_all[mo_idx[0]]:.3f}, {eps_all[mo_idx[-1]]:.3f}] Ha"
    )

    # Guard: N_frozen must fit strictly inside the MO window.
    # If N_frozen >= N, the active window (N_frozen : N_frozen+block) is
    # entirely outside g_MO, which causes an IndexError in the E_core loop.
    # Typical cause: --n-total-orbs too small relative to N_elec_total.
    if N_frozen >= N:
        raise ValueError(
            f"[full_mo] N_frozen={N_frozen} >= N_total_orbs={N}: the active "
            f"space window is outside the MO tensor.  Increase --n-total-orbs "
            f"to at least {N_frozen + 4} "
            f"(N_elec_total/2={N_frozen + 2} occupied MOs + 2 active + "
            f"2 virtual at minimum; current N_AO={N_AO})."
        )

    # ── Step 4 (optional): Boys orbital localization ─────────────────────────
    if localize:
        if shells is None or norms is None:
            raise ValueError(
                "[full_mo] localize=True requires shells and norms; "
                "call build_ao_integrals_with_shells instead of build_ao_integrals."
            )
        from nanoprotogeny.molecular.mqelocalize import (
            build_dipole_ao_matrices,
            boys_localize,
        )
        log.info(f"[full_mo] Boys localization (N_AO={N_AO}, N={N})…")
        # Dipole matrices in the spherical AO basis if sph_C was provided,
        # else in the Cartesian AO basis.  Since shells are always Cartesian,
        # we build Cartesian dipole matrices and transform if needed.
        r_cart = build_dipole_ao_matrices(shells, norms)
        if sph_C is not None:
            r_ao = tuple(sph_C.T @ r @ sph_C for r in r_cart)
        else:
            r_ao = r_cart
        C, _U = boys_localize(C, r_ao)
        log.info(f"[full_mo] Boys localization complete.")

    # ── 1e transform: h1_MO = C^T h1_AO C  (N×N) ───────────────────────────
    h1_MO = C.T @ h1_AO @ C                            # (N, N)

    # ── Step 5 (optional): Screened integral-direct AO→MO transform ──────────
    if screened:
        if shells is None or norms is None:
            raise ValueError(
                "[full_mo] screened=True requires shells and norms; "
                "call build_ao_integrals_with_shells instead of build_ao_integrals."
            )
        from nanoprotogeny.molecular.mqeaointegrals import screened_direct_ao_to_mo
        log.info(
            f"[full_mo] Screened direct AO→MO transform "
            f"(N_AO={N_AO}, N={N}, threshold={schwarz_thr:.1e})…"
        )
        # When the basis is spherical, the MO coefficients C are in the
        # spherical AO space but the primitive shells are Cartesian.
        # The combined coefficient matrix maps Cartesian AOs → MOs.
        if sph_C is not None:
            C_cart = sph_C @ C          # (N_cart, N) = Cartesian→MO
        else:
            C_cart = C                  # already Cartesian
        g_MO = screened_direct_ao_to_mo(
            shells, norms, C_cart, threshold=schwarz_thr,
        )
        log.info(f"[full_mo] Screened transform complete. g_MO shape={g_MO.shape}")
    else:
        # ── Dense 4-index transform via sequential einsum ────────────────────
        # Each step contracts one AO index against the coefficient matrix.
        # Total cost: O(N_AO^4 · N).
        if g_AO is None:
            raise ValueError(
                "[full_mo] g_AO is None but screened=False; "
                "either pass g_AO or set screened=True."
            )
        log.info(f"[full_mo] Starting 4-index ERI transform (N_AO={N_AO}, N={N})…")
        tmp   = np.einsum("up,uvlm->pvlm", C, g_AO, optimize=True)
        tmp   = np.einsum("vq,pvlm->pqlm", C, tmp,  optimize=True)
        tmp   = np.einsum("lr,pqlm->pqrm", C, tmp,  optimize=True)
        g_MO  = np.einsum("ms,pqrm->pqrs", C, tmp,  optimize=True)   # (N,N,N,N)
        log.info(f"[full_mo] 4-index transform complete. g_MO shape={g_MO.shape}")

    # ── E_core from N_frozen lowest MOs (eq:ecore, fixed for all tower k) ───
    # E_core = E_nuc
    #        + Σ_{f<N_frozen} [ 2 h1_MO[f,f]
    #          + Σ_{g<N_frozen} (2 g_MO[f,f,g,g] − g_MO[f,g,g,f]) ]
    E_core = E_nuc
    for f in range(N_frozen):
        E_core += 2.0 * float(h1_MO[f, f])
        for g in range(N_frozen):
            E_core += 2.0 * float(g_MO[f, f, g, g]) - float(g_MO[f, g, g, f])

    log.info(f"[full_mo] E_core = {E_core:.6f} Ha (N_frozen={N_frozen})")
    return h1_MO, g_MO, E_core, C


def slice_active_hamiltonian(
    h1_MO:    np.ndarray,
    g_MO:     np.ndarray,
    N_frozen: int,
    k:        int,
    block:    int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Algebraically slice H^(k) from the full MO integral tensors.

    Implements eq:h1eff and eq:H_casci restricted to active set A_k.

    Per subsec:orbital_ladder:
        A_k = { N_frozen, …, N_frozen + block*(k−1) − 1 }   (size = block*(k−1))
        F   = { 0, …, N_frozen − 1 }                         (frozen core)

    The effective 1e integrals (eq:h1eff):

        h̃^(1,k)[p,q] = h1_MO[A_k[p], A_k[q]]
                      + Σ_{f∈F} ( 2·g_MO[A_k[p], A_k[q], f, f]   [Coulomb]
                                  − g_MO[A_k[p], f, A_k[q], f] )  [Exchange]

    The 2e integrals restricted to the active space:

        h^(2)|_{A_k}[p,q,r,s] = g_MO[A_k[p], A_k[q], A_k[r], A_k[s]]

    This is a pure array-indexing operation; no eigenvalue problem is solved.
    The largest array touched is h1_MO (N×N) and g_MO slices — at most N×N×N_f.

    Args:
        h1_MO    : (N, N) full MO-basis 1e integrals [Ha].
        g_MO     : (N, N, N, N) full MO-basis ERI tensor [Ha] (chemist).
        N_frozen : Number of frozen-core MOs (indices 0..N_frozen-1).
        k        : Tower level (k=2 → CAS(4,4), k=3 → CAS(8,8), …).
        block    : Orbitals added per tower step (always 4 per theory).

    Returns:
        h1_eff  : (n_act, n_act) Fock-screened 1e integrals [Ha].
        g_slice : (n_act, n_act, n_act, n_act) 2e ERIs [Ha].

    where n_act = block * (k − 1).

    Raises:
        ValueError : If A_k would extend beyond the available MO range.
    """
    n_act  = block * (k - 1)
    N      = h1_MO.shape[0]
    a_start = N_frozen
    a_end   = N_frozen + n_act

    if a_end > N:
        raise ValueError(
            f"[slice_active] Tower level k={k} requires {n_act} active MOs "
            f"(indices {a_start}..{a_end-1}) but only {N - N_frozen} MOs are "
            f"available above the frozen core (N_total_orbs={N}, "
            f"N_frozen={N_frozen}).  Increase n_total_orbs or reduce k."
        )

    # Active-set MO indices (contiguous range above frozen core).
    A_k = np.arange(a_start, a_end, dtype=int)   # shape (n_act,)

    # ── h1 slice: (n_act, n_act) subblock of h1_MO ──────────────────────────
    h1_slice = h1_MO[np.ix_(A_k, A_k)].copy()    # (n_act, n_act)

    # ── Frozen-core Fock screening (eq:h1eff) ────────────────────────────────
    # h1_eff[p,q] += Σ_{f<N_frozen} (2·g[A_k[p],A_k[q],f,f] − g[A_k[p],f,A_k[q],f])
    if N_frozen > 0:
        F = np.arange(N_frozen, dtype=int)
        # Coulomb: 2 * g[A_k, A_k, f, f] summed over f
        # Shape: g_MO[A_k[:,None], A_k[None,:], F, F] — (n_act, n_act, N_frozen)
        coulomb  = 2.0 * g_MO[np.ix_(A_k, A_k, F, F)].sum(axis=(2, 3))
        # Exchange: g[A_k, f, A_k, f] — need diagonal over f
        # g_MO[A_k[p], f, A_k[q], f] for each p,q,f
        # = g_MO[A_k[:,None,None], F[None,:,None], A_k[None,None,:], F[None,:,None]]
        # Reshape trick: g_MO[ix_(A_k, F, A_k, F)].diagonal over (1,3)
        exchange = np.einsum(
            "pfqf->pq",
            g_MO[np.ix_(A_k, F, A_k, F)],
            optimize=True,
        )
        h1_slice += coulomb - exchange

    # ── g slice: (n_act, n_act, n_act, n_act) active-active ERI block ────────
    g_slice = g_MO[np.ix_(A_k, A_k, A_k, A_k)].copy()

    log.debug(
        f"[slice_active] k={k}, n_act={n_act}, "
        f"A_k=[{A_k[0]}..{A_k[-1]}], "
        f"h1_eff range=[{h1_slice.min():.3f},{h1_slice.max():.3f}] Ha"
    )
    return h1_slice, g_slice


def save_seed_tensors(
    path:     "Union[str, Path]",
    h1_MO:   np.ndarray,
    g_MO:    np.ndarray,
    E_core:  float,
    E_nuc:   float,
    N_frozen: int,
    meta:    Optional[Dict] = None,
) -> None:
    """Write full MO integral tensors to a compressed .npz file.

    The file is the "write to disk once" artefact of sec:algebraic_tower.
    All subsequent tower levels are assembled by algebraic slicing of these
    tensors without any further AO integral computation.

    Args:
        path     : Output file path (will be created/overwritten; .npz suffix
                   added automatically if absent).
        h1_MO    : (N, N) 1e MO integrals [Ha].
        g_MO     : (N, N, N, N) 2e MO ERIs [Ha].
        E_core   : Frozen-core energy [Ha].
        E_nuc    : Nuclear repulsion energy [Ha].
        N_frozen : Number of frozen-core MOs.
        meta     : Optional dict of scalar metadata (mechanism name, basis, …).
                   Values must be JSON-serialisable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_json = json.dumps(meta or {})
    # Pack g_MO using 8-fold permutation symmetry (sec:hamiltonian_from_zeros).
    # Storage: Np*(Np+1)/2 entries vs N^4, where Np = N*(N+1)/2.
    # For N=76: ~4.3M × 8 bytes ≈ 34 MB vs 76^4 × 8 ≈ 255 MB.
    g_packed, g_N = _pack_g_mo(g_MO)
    np.savez_compressed(
        str(path),
        h1_MO    = h1_MO,
        g_packed = g_packed,
        g_N      = np.array([g_N], dtype=int),
        E_core   = np.array([E_core]),
        E_nuc    = np.array([E_nuc]),
        N_frozen = np.array([N_frozen], dtype=int),
        meta_json= np.array([meta_json]),
    )
    log.info(
        f"[seed_tensors] Saved: {path}  "
        f"h1_MO={h1_MO.shape}, g_packed={g_packed.shape} (N={g_N}), "
        f"E_core={E_core:.6f} Ha, N_frozen={N_frozen}"
    )


def load_seed_tensors(
    path: "Union[str, Path]",
) -> Tuple[np.ndarray, np.ndarray, float, float, int, Dict]:
    """Load seed tensors from a .npz file written by :func:`save_seed_tensors`.

    Returns:
        h1_MO    : (N, N) 1e MO integrals [Ha].
        g_MO     : (N, N, N, N) 2e MO ERIs [Ha].
        E_core   : Frozen-core energy [Ha].
        E_nuc    : Nuclear repulsion energy [Ha].
        N_frozen : Number of frozen-core MOs.
        meta     : Metadata dict (may be empty).
    """
    path = Path(path)
    if not path.exists():
        # Try appending .npz in case the caller omitted the suffix.
        path = path.with_suffix(".npz")
    data     = np.load(str(path), allow_pickle=False)
    h1_MO    = data["h1_MO"]
    E_core   = float(data["E_core"][0])
    E_nuc    = float(data["E_nuc"][0])
    N_frozen = int(data["N_frozen"][0])
    meta     = json.loads(str(data["meta_json"][0]))
    # Support both packed (new) and full 4D (legacy) storage formats.
    if "g_packed" in data:
        g_N   = int(data["g_N"][0])
        g_MO  = _unpack_g_mo(data["g_packed"], g_N)
    else:
        g_MO  = data["g_MO"]
    log.info(
        f"[seed_tensors] Loaded: {path}  "
        f"h1_MO={h1_MO.shape}, g_MO={g_MO.shape}, "
        f"E_core={E_core:.6f} Ha, N_frozen={N_frozen}"
    )
    return h1_MO, g_MO, E_core, E_nuc, N_frozen, meta


# ── Tower helpers ──────────────────────────────────────────────────────────

# Maximum FCI determinant count for dense solve; Kummer fallback above this.
# CAS(8,8) → comb(8,4)^2 = 4 900 ✓; CAS(12,12) → 854 K → Kummer fallback.
_MAX_FCI_DIM: int = 50_000


def _recompute_ecore(
    h1_MO:    np.ndarray,
    g_MO:     np.ndarray,
    E_nuc:    float,
    N_frozen: int,
) -> float:
    """Vectorised frozen-core energy for an arbitrary frozen-set size.

    E_core = E_nuc
           + 2·Σ_{f<N} h1_MO[f,f]
           + Σ_{f,g<N} (2·g_MO[f,f,g,g] − g_MO[f,g,g,f])

    Matches the scalar loop in :func:`build_full_mo_tensors`.
    """
    if N_frozen <= 0:
        return float(E_nuc)
    E = float(E_nuc)
    E += 2.0 * float(np.trace(h1_MO[:N_frozen, :N_frozen]))
    g_ff = g_MO[:N_frozen, :N_frozen, :N_frozen, :N_frozen]
    E += 2.0 * float(np.einsum("iijj->", g_ff))
    E -= float(np.einsum("ijji->", g_ff))
    return E


def _fci_det_count(N_e: int, N_orb: int) -> int:
    """Alpha×beta determinant count for CAS(N_e, N_orb) with balanced spin."""
    n_alpha = N_e // 2
    n_beta  = N_e - n_alpha
    return math.comb(N_orb, min(n_alpha, N_orb)) * math.comb(N_orb, min(n_beta, N_orb))


def _janus_electron_distributions(
    K: int,
    N_e_total: int,
    m: int,
    nu_n: int,
    max_ne_per_block: int = 8,
) -> List[Tuple[int, ...]]:
    """Enumerate valid Janus block electron distributions (carry-bus constraint).

    Implements eq:block_local_coupling: after each block b the virtual register
    updates as j_V^(b+1) = (j_V^(b) + nu_n × N_e^(b)) mod m.  At every
    internal boundary b=0,...,K-2 the result must equal m//2 (Janus projector
    Π_{k*}, eq:kstar_projector_construction).  The last block (b=K-1) is
    unconstrained.  Recursion starts at j_V^(0)=0.

    Group B (Fe₄S₄): m=4, nu_n=2 → N_e^(0) odd, N_e^(b≥1,internal) even.
    Group A (nitrogenase): m=8, nu_n=2 → N_e^(0) ≡ 2 (mod 4), interior ≡ 0 (mod 4).

    Returns list of tuples (N_e^(0), …, N_e^(K-1)) summing to N_e_total.
    """
    k_star = m // 2
    result: List[Tuple[int, ...]] = []

    def _rec(b: int, j_v: int, n_rem: int, dist: List[int]) -> None:
        if b == K:
            if n_rem == 0:
                result.append(tuple(dist))
            return
        is_last = (b == K - 1)
        for ne in range(min(max_ne_per_block, n_rem) + 1):
            j_v_new = (j_v + nu_n * ne) % m
            if not is_last and j_v_new != k_star:
                continue
            dist.append(ne)
            _rec(b + 1, j_v_new, n_rem - ne, dist)
            dist.pop()

    _rec(0, 0, N_e_total, [])
    return result


def _block_sequential_janus_energy(
    h1_k: np.ndarray,
    g_k: np.ndarray,
    K: int,
    N_e_total: int,
    m: int,
    nu_n: int,
    E_core_k: float,
) -> float:
    """Block-sequential Janus eigenvalue (prop:coupling_factorization).

    For each Janus-valid electron distribution (N_e^(0),…,N_e^(K-1)):
    1. Build block-local effective h1 with mean-field Fock correction from
       all other blocks (doubly-occupied lowest N_e^(b')//2 orbitals of b').
    2. Solve dense block FCI for each block  — dim ≤ C(8, 4) = 70, trivially fast.
    3. Accumulate E_core_k + Σ_b E_FCI^(b).
    Return the minimum over all valid distributions (eq:janus_eigenvalue).

    The cross-block 2e correlation enters only via the mean-field Fock channel.
    This is the block-additive Janus approximation of sec:block_sequential.
    """
    dists = _janus_electron_distributions(K, N_e_total, m, nu_n)
    if not dists:
        raise ValueError(
            f"[block_seq] No Janus distributions: "
            f"K={K}, N_e={N_e_total}, m={m}, nu_n={nu_n}"
        )

    N_ORB_B = 4  # spatial orbitals per block (always 4 per theory)

    E_best = float("inf")
    for dist in dists:
        # Number of doubly-occupied orbitals per block (closed-shell HF guess)
        n_occ = [ne // 2 for ne in dist]

        # Build Fock-corrected h1 for each block
        block_h1: List[np.ndarray] = []
        for b in range(K):
            p_idx = np.arange(N_ORB_B * b, N_ORB_B * (b + 1))
            h1_b = h1_k[np.ix_(p_idx, p_idx)].copy()  # (4, 4)

            for bp in range(K):
                if bp == b or n_occ[bp] == 0:
                    continue
                r_idx = np.arange(N_ORB_B * bp, N_ORB_B * bp + n_occ[bp])

                # Coulomb: 2 Σ_r g[p,q,r,r]  (chemist notation: (pq|rr))
                g_J = g_k[np.ix_(p_idx, p_idx, r_idx, r_idx)]  # (4,4,n_occ,n_occ)
                h1_b += 2.0 * np.einsum("pqrr->pq", g_J, optimize=True)

                # Exchange: Σ_r g[p,r,q,r]  (chemist notation: (pr|qr))
                g_K = g_k[np.ix_(p_idx, r_idx, p_idx, r_idx)]  # (4,n_occ,4,n_occ)
                h1_b -= np.einsum("prqr->pq", g_K, optimize=True)

            block_h1.append(h1_b)

        # Sum of block FCI ground-state energies
        E_sum = E_core_k
        failed = False
        for b in range(K):
            ne_b = dist[b]
            if ne_b == 0:
                continue
            p_idx = np.arange(N_ORB_B * b, N_ORB_B * (b + 1))
            g_b = g_k[np.ix_(p_idx, p_idx, p_idx, p_idx)]  # (4,4,4,4)
            try:
                E_elec_b, _, _ = solve_cas(
                    h1_MO=block_h1[b], g_MO=g_b,
                    N_e=ne_b, N_orb=N_ORB_B, E_nuc=0.0,
                )
            except Exception as exc:
                log.warning(
                    f"[block_seq] b={b}, ne={ne_b}, dist={dist}: FCI failed ({exc})"
                )
                failed = True
                break
            E_sum += float(E_elec_b)

        if not failed and E_sum < E_best:
            E_best = E_sum

    if E_best == float("inf"):
        raise RuntimeError(
            f"[block_seq] All Janus distributions failed. "
            f"K={K}, N_e={N_e_total}, m={m}, nu_n={nu_n}"
        )
    return E_best


def run_algebraic_tower_pipeline(
    h1_AO:         np.ndarray,
    g_AO:          Optional[np.ndarray],
    S_AO:          np.ndarray,
    E_nuc:         float,
    E_inf:         float,
    N_elec_total:  int,
    N_frozen:      int,
    N_e_seed:      int,
    tower_p:       int,
    m:             int                = 4,
    nu_n:          int                = 2,
    eps_thresh:    float              = _EPS_MILLI_HA,
    K_max:         int                = 12,
    N_total_orbs:  int                = 0,
    seed_tensors_path: Optional[str]  = None,
    save_tensors_to:   Optional[str]  = None,
    localize:          bool           = False,
    screened:          bool           = False,
    shells:            Optional[list] = None,
    norms:             Optional[list] = None,
    sph_C:             Optional[np.ndarray] = None,
    schwarz_thr:       float          = 1.0e-10,
    screen_frozen:     bool           = True,
    cstar:             bool           = False,
    cstar_max_iter:    int            = 300,
    cstar_step_size:   float          = 5e-3,
) -> Tuple[np.ndarray, np.ndarray, float, float, float, int, List]:
    r"""Full algebraic Iwasawa tower from AO integrals (or pre-computed seed).

    When ``screen_frozen=False`` (zetazeros seed-only mode):
    - E_core is set to E_nuc (nuclear repulsion only); no frozen-core 2e
      contribution is added.  This avoids the catastrophic Fock-screening
      artefact that occurs with non-SCF core-Ham MOs on heavy-atom systems.
    - The CAS(4,4) active window is taken by direct array-slice at indices
      [N_frozen : N_frozen+4] without Fock screening.
    - The tower expansion loop is skipped; tower = [(k_BASE, E_seed, δ)].
    The returned E_seed = E_seed_elec + E_nuc (total, no frozen-core 2e).


    Implements sec:algebraic_tower Steps 0–5 via block-sequential Janus CASCI
    (prop:coupling_factorization), without DMRG or Kummer extrapolation:

    1. Build (or load) h1_MO∈ℝ^{N×N} and g_MO∈ℝ^{N^4} from AO integrals.
    2. Compute E_core from N_frozen lowest MOs (seed level, k=2).
    3. Slice to k=2 active set A_2 (CAS(4,4)) via :func:`slice_active_hamiltonian`
       and solve FCI → E_seed = E_seed_elec + E_core.
    4. Balanced expansion loop (subsec:orbital_ladder): for k=3,4,…
       promote 2 occupied MOs per level (N_frozen_k = N_frozen_seed − 2·(k−2)),
       recompute E_core_k, slice CAS(4(k-1), 4(k-1)), and run
       :func:`_block_sequential_janus_energy` (K=k-1 blocks of 4 orbitals each).
       This enumerates carry-bus-valid electron distributions and solves dense
       block FCI (dim ≤ 70) per block with mean-field Fock correction.
    5. Stop when Cauchy step |E_Janus^(k) − E_Janus^(k-1)| < eps_thresh.

    The Janus projector Π_{k*} = I_L ⊗ |m/2⟩⟨m/2|_V is enforced via the
    carry-bus constraint at internal block boundaries (eq:kstar_projector_construction).
    k* is self-consistently discovered, not predicted from any geometric formula.

    Args:
        h1_AO           : (N_AO, N_AO) core Hamiltonian [Ha].
        g_AO            : (N_AO, N_AO, N_AO, N_AO) AO ERI tensor [Ha].
        S_AO            : (N_AO, N_AO) overlap matrix.
        E_nuc           : Nuclear repulsion [Ha].
        E_inf           : Algebraic Janus energy E_∞ [Ha] (used for initial
                          delta_seed logging only; not used for convergence).
        N_elec_total    : Total electron count (for Fermi-level MO centering).
        N_frozen        : Frozen-core MOs (fixed for all tower levels).
        N_e_seed        : Active electrons for the CAS(4,4) FCI (= 4).
        tower_p         : Tower prime p (kept for API compatibility; not used
                          in block-sequential computation).
        m               : Mechanism order (ℤ_m virtual register dimension).
                          Group B: 4, Group A: 8, Group C: 4, Group D: 12.
        nu_n            : Uniform shift parameter from mech.nu_shifts.
                          Group B: 2, Group A: 2, Group C: 1, Group D: 2.
        eps_thresh      : Cauchy convergence threshold [Ha] (default 1.6 mHa).
        K_max           : Maximum extra tower levels computed beyond seed.
        N_total_orbs    : Total MOs to include in h1_MO, g_MO (0 = all N_AO).
        seed_tensors_path : If given, load pre-computed .npz instead of
                            re-running the 4-index transform.
        save_tensors_to   : If given, save h1_MO, g_MO, E_core to this path.

    Returns:
        h1_MO      : (N, N) full MO-basis 1e integrals [Ha].
        g_MO       : (N, N, N, N) full MO-basis ERIs [Ha].
        E_core     : Frozen-core energy [Ha].
        E_seed     : CAS(4,4) total energy = E_seed_elec + E_core [Ha].
                     (E_core already includes E_nuc; see build_full_mo_tensors.)
        E_seed_elec: CAS(4,4) electronic energy [Ha].
        k_0        : Required tower level (first k with Δ_k < eps_thresh).
        tower      : List of (k, E_k, Δ_k) from the Kummer formula.
    """
    # ── Step 1: build or load full MO integral tensors ─────────────────────
    _C: Optional[np.ndarray] = None   # MO coefficient matrix; set only in build branch
    if seed_tensors_path is not None:
        h1_MO, g_MO, E_core, _E_nuc_stored, N_frozen_stored, _meta = \
            load_seed_tensors(seed_tensors_path)
        # N_frozen from file takes precedence (it was used when E_core was
        # computed); warn if caller supplied a different value.
        if N_frozen_stored != N_frozen:
            log.warning(
                f"[algebraic_tower] N_frozen in file ({N_frozen_stored}) differs "
                f"from caller argument ({N_frozen}). Using file value."
            )
            N_frozen = N_frozen_stored
        # E_nuc may differ if geometry changed; use caller's value.
        E_core_adj = E_core - _E_nuc_stored + E_nuc
        E_core = E_core_adj
    else:
        h1_MO, g_MO, E_core, _C = build_full_mo_tensors(
            h1_AO        = h1_AO,
            g_AO         = g_AO,
            S_AO         = S_AO,
            E_nuc        = E_nuc,
            N_frozen     = N_frozen,
            N_elec_total = N_elec_total,
            N_total_orbs = N_total_orbs,
            localize     = localize,
            screened     = screened,
            shells       = shells,
            norms        = norms,
            sph_C        = sph_C,
            schwarz_thr  = schwarz_thr,
        )

    # ── screen_frozen=False: override E_core to nuclear-only ──────────────────
    # When building from non-SCF core-Ham MOs (zetazeros seed-only path) the
    # frozen-core Fock screening is unphysical for heavy-atom systems: the
    # inner-shell MOs are unscreened → 2e Coulomb sum = +11 kHa.  Replace the
    # computed E_core with E_nuc only; the frontier-MO CAS(4,4) slice below
    # uses direct indexing and skips Fock screening entirely.
    if not screen_frozen:
        E_core = float(E_nuc)
        log.info(
            f"[algebraic_tower] screen_frozen=False: overriding E_core={E_core:.6f} Ha "
            f"(= E_nuc; frozen-core 2e contribution suppressed for non-SCF MOs)"
        )

    # ── S4b (optional): Hilbert–Pólya C* orbital optimisation ──────────────
    # Only valid when:
    #   - cstar=True requested
    #   - screen_frozen=False (non-SCF MOs — avoids Fock-screening artifact)
    #   - MOs were built here (seed_tensors_path is None → _C is not None)
    #   - g_AO is available (not a screened/localized path)
    if cstar and not screen_frozen and seed_tensors_path is None and \
            g_AO is not None and _C is not None:
        from nanoprotogeny.molecular.mqecstar import hilbert_polya_cstar_optimize
        _C_active = _C[:, N_frozen:N_frozen + 4]
        log.info(
            f"[algebraic_tower] S4b: C* optimisation (screen_frozen=False, "
            f"N_frozen={N_frozen}, N_active=4, E_∞={E_inf:.6f} Ha)…"
        )
        _C_star, _delta_star, _k0_star, _n_cstar, _hist = hilbert_polya_cstar_optimize(
            h1_AO_eff  = h1_AO,       # bare core-Ham (no Fock screen) — correct here
            g_AO       = g_AO,
            S_AO       = S_AO,
            E_inf      = E_inf,
            N_active   = 4,
            N_e        = N_e_seed,
            E_core     = 0.0,          # not added to objective (Fix 1)
            E_nuc      = 0.0,          # not added to objective (Fix 1)
            C_init     = _C_active,
            max_iter   = cstar_max_iter,
            step_size  = cstar_step_size,
        )
        # Splice optimised active columns back into the full coefficient matrix.
        _C_updated = _C.copy()
        _C_updated[:, N_frozen:N_frozen + 4] = _C_star
        h1_MO, g_MO = transform_integrals(_C_updated, h1_AO, g_AO)
        _cstar_reduction = _hist[0] / max(_delta_star, 1e-15)
        log.info(
            f"[algebraic_tower] C*: Δ₀ {_hist[0]:.6e} → {_delta_star:.6e} Ha "
            f"(×{_cstar_reduction:.2f} reduction), k₀={_k0_star}, iters={_n_cstar}"
        )

    if save_tensors_to is not None:
        save_seed_tensors(
            path     = save_tensors_to,
            h1_MO    = h1_MO,
            g_MO     = g_MO,
            E_core   = E_core,
            E_nuc    = E_nuc,
            N_frozen = N_frozen,
        )

    # ── Step 2: slice to CAS(4,4) seed (k=2) ──────────────────────────────
    if screen_frozen:
        # Standard path: slice_active_hamiltonian applies Fock screening from
        # N_frozen frozen MOs — correct when MOs come from SCF.
        h1_seed, g_seed = slice_active_hamiltonian(
            h1_MO    = h1_MO,
            g_MO     = g_MO,
            N_frozen = N_frozen,
            k        = _K_BASE,          # k=2 → CAS(4,4), n_act=4
        )
    else:
        # Non-SCF path: take frontier block [N_frozen:N_frozen+4] by direct
        # array index WITHOUT Fock screening.  N_frozen here is used only as
        # an orbital-selection offset (HOMO-LUMO window in the core-Ham
        # eigenvalue ordering); it does NOT contribute to E_core or h1_eff.
        _a, _b = N_frozen, N_frozen + 4
        if _b > h1_MO.shape[0]:
            raise ValueError(
                f"[algebraic_tower] screen_frozen=False: frontier window "
                f"[{_a}:{_b}] exceeds h1_MO size {h1_MO.shape[0]}. "
                f"Increase n_total_orbs (currently {N_total_orbs})."
            )
        h1_seed = h1_MO[_a:_b, _a:_b].copy()
        g_seed  = g_MO [_a:_b, _a:_b, _a:_b, _a:_b].copy()
    log.info(
        f"[algebraic_tower] CAS(4,4) seed: "
        f"h1_seed shape={h1_seed.shape}, g_seed shape={g_seed.shape}, "
        f"h1_seed diag={np.diag(h1_seed).tolist()}"
    )

    # ── Step 3: FCI at CAS(4,4) — the only eigenvalue solve ───────────────
    N_e_use = min(N_e_seed, 2 * h1_seed.shape[0] - 1)
    E_seed_elec, _, _psi = solve_cas(
        h1_MO = h1_seed,
        g_MO  = g_seed,
        N_e   = N_e_use,
        N_orb = h1_seed.shape[0],
        E_nuc = 0.0,                 # E_core already contains E_nuc
    )
    # E_core = E_nuc + frozen_contributions (standard), or E_nuc only
    # (screen_frozen=False).  Either way, E_nuc is not added again.
    E_seed = float(E_seed_elec) + E_core
    log.info(
        f"[algebraic_tower] E_seed_elec={E_seed_elec:.6f} Ha, "
        f"E_core={E_core:.6f} Ha, E_nuc={E_nuc:.6f} Ha, "
        f"E_seed={E_seed:.6f} Ha  (screen_frozen={screen_frozen})"
    )

    # ── screen_frozen=False: seed-only, skip tower expansion ──────────────
    # The zetazeros seed-only workflow must NOT attempt tower convergence
    # (the user explicitly requires this).  Return immediately with a single
    # seed entry so callers can populate the step files and manifest.
    if not screen_frozen:
        delta_seed = abs(float(E_seed_elec) - E_inf)
        tower: List = [(_K_BASE, E_seed, delta_seed)]
        log.info(
            f"[algebraic_tower] seed-only mode: δ_seed={delta_seed*1000:.3f} mHa "
            f"(|E_seed_elec − E_∞|), k_0={_K_BASE}, no tower computed"
        )
        return h1_MO, g_MO, E_core, E_seed, float(E_seed_elec), _K_BASE, tower

    # ── Steps 4–5: balanced CAS expansion — block-sequential Janus CASCI ──────
    # At seed (k=2), N_frozen_seed MOs are frozen.  At each subsequent level k,
    # 2 occupied MOs are promoted into the active space (balanced expansion per
    # subsec:orbital_ladder): N_frozen_k = N_frozen_seed − 2·(k − k_BASE).
    # E_core is recomputed for each N_frozen_k.  The Janus energy is computed
    # via _block_sequential_janus_energy (K=k-1 blocks, 4 orbitals each).
    # Convergence: Cauchy criterion |E_Janus^(k) − E_Janus^(k-1)| < eps_thresh.
    _N_frozen_seed: int = N_frozen   # alias — never modified below

    delta_seed = abs(E_seed - E_inf)
    tower: List[Tuple[int, float, float]] = [(_K_BASE, E_seed, delta_seed)]
    k_0 = _K_BASE
    E_janus_prev = E_seed

    log.info(
        f"[algebraic_tower] k={_K_BASE} CAS(4,4) [seed]: "
        f"E={E_seed:.6f} Ha  delta_seed={delta_seed * 1000:.3f} mHa  "
        f"(E_∞={E_inf:.6f} Ha  [eigenphase frame, for reference only])"
    )

    if delta_seed >= eps_thresh:
        for k in range(_K_BASE + 1, _K_BASE + K_max + 1):
            # Balanced expansion: promote 2 occ MOs per level above seed
            n_frozen_k = _N_frozen_seed - 2 * (k - _K_BASE)
            if n_frozen_k < 0:
                log.info(
                    f"[algebraic_tower] k={k}: frozen core exhausted "
                    f"(n_frozen_k={n_frozen_k}), stopping tower"
                )
                break

            # Recompute E_core with the reduced frozen set
            E_core_k = _recompute_ecore(h1_MO, g_MO, E_nuc, n_frozen_k)

            # Algebraic slice: CAS(4*(k-1), 4*(k-1)) centred on HOMO-LUMO gap
            try:
                h1_k, g_k = slice_active_hamiltonian(
                    h1_MO    = h1_MO,
                    g_MO     = g_MO,
                    N_frozen = n_frozen_k,
                    k        = k,
                )
            except ValueError as exc:
                log.warning(
                    f"[algebraic_tower] k={k}: cannot slice — {exc}. "
                    f"Stopping tower at k={k - 1}."
                )
                break

            N_e_k   = N_elec_total - 2 * n_frozen_k   # = 4*(k-1) for balanced
            n_orb_k = h1_k.shape[0]                   # = 4*(k-1)
            K_blocks = k - 1                           # one 4-orbital block per level above seed

            # Block-sequential Janus CASCI (prop:coupling_factorization)
            # Enumerates carry-bus-valid distributions, solves block FCI ≤ C(8,4)=70.
            try:
                E_janus_k = _block_sequential_janus_energy(
                    h1_k=h1_k, g_k=g_k,
                    K=K_blocks, N_e_total=N_e_k,
                    m=m, nu_n=nu_n,
                    E_core_k=E_core_k,
                )
            except Exception as exc:
                log.warning(
                    f"[algebraic_tower] k={k}: block-sequential failed ({exc}). "
                    f"Stopping tower at k={k - 1}."
                )
                break

            # Cauchy convergence: compare consecutive Janus energies (no E_∞ needed)
            dE = abs(E_janus_k - E_janus_prev)
            tower.append((k, E_janus_k, dE))
            log.info(
                f"[algebraic_tower] k={k} CAS({N_e_k},{n_orb_k}) [BlockSeq K={K_blocks}]: "
                f"E={E_janus_k:.6f} Ha  dE={dE * 1000:.3f} mHa"
            )

            if dE < eps_thresh:
                k_0 = k
                break

            k_0 = k
            E_janus_prev = E_janus_k

    return h1_MO, g_MO, E_core, E_seed, float(E_seed_elec), k_0, tower
