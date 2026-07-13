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
mqefci.py — CAS / FCI solver.
============================
Full configuration-interaction in the spin-orbital basis via Slater–Condon
rules.  Inputs are MO-basis 1e/2e integral tensors; no mechanism coupling.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import List, Tuple

import numpy as np
from numpy.linalg import eigh as _eigh

log = logging.getLogger(__name__)

# ===========================================================================
# SECTION 6 — CAS(4,4) FCI DIAGONALISATION
# ===========================================================================
# Work in the spin-orbital basis with ordering α₀β₀α₁β₁...α_{N-1}β_{N-1}.
# Spin-orbital index: 2p → p_α, 2p+1 → p_β.
# FCI matrix uses Slater-Condon rules with antisymmetrised two-electron
# integrals ⟨pq||rs⟩ = ⟨pq|rs⟩ − ⟨pq|sr⟩ in physicist's notation.

def _spatial_idx(i: int) -> int:
    """Spatial orbital index from spin-orbital index (0-based)."""
    return i // 2


def _spin(i: int) -> int:
    """Spin: 0 = α, 1 = β."""
    return i % 2


def _h1_so(h1_MO: np.ndarray, p: int, q: int) -> float:
    """h_SO[p,q] in spin-orbital basis."""
    if _spin(p) != _spin(q):
        return 0.0
    return h1_MO[_spatial_idx(p), _spatial_idx(q)]


def _g_so(g_MO: np.ndarray, p: int, q: int, r: int, s: int) -> float:
    """⟨pq|rs⟩_SO in physicist's notation = chemist's (pr|qs).

    g_AO/g_MO is stored in chemist's notation (μν|λσ).
    Physicist ⟨pq|rs⟩ = chemist (pr|qs), so indices map as
        spatial(p), spatial(r), spatial(q), spatial(s).
    """
    if _spin(p) != _spin(r) or _spin(q) != _spin(s):
        return 0.0
    return g_MO[
        _spatial_idx(p), _spatial_idx(r),
        _spatial_idx(q), _spatial_idx(s),
    ]


def _g_so_anti(g_MO: np.ndarray, p: int, q: int, r: int, s: int) -> float:
    """⟨pq||rs⟩ = ⟨pq|rs⟩ − ⟨pq|sr⟩."""
    return _g_so(g_MO, p, q, r, s) - _g_so(g_MO, p, q, s, r)


def _parity(occ: List[int], p: int, q: int) -> int:
    """Sign from moving orbital p (creation) and q (annihilation) to front of occ."""
    occ = list(occ)
    # Count transpositions to bring q to position 0, then p.
    pos_q = occ.index(q)
    sign = (-1) ** pos_q
    occ.pop(pos_q)
    pos_p = 0  # p goes to the front of the remaining list
    sign *= (-1) ** pos_p
    return sign


def _slater_condon(
    occ_I: Tuple[int, ...],
    occ_J: Tuple[int, ...],
    h1_MO: np.ndarray,
    g_MO:  np.ndarray,
    n_so:  int,
) -> float:
    """⟨I|H|J⟩ via Slater-Condon rules in the spin-orbital basis."""
    set_I = set(occ_I)
    set_J = set(occ_J)
    only_I = sorted(set_I - set_J)  # in I but not J (creation needed)
    only_J = sorted(set_J - set_I)  # in J but not I (annihilation needed)
    common = sorted(set_I & set_J)

    n_diff = len(only_I)

    if n_diff == 0:
        # Diagonal element.
        e = sum(_h1_so(h1_MO, a, a) for a in occ_I)
        for i, a in enumerate(occ_I):
            for b in occ_I[i + 1:]:
                e += _g_so_anti(g_MO, a, b, a, b)
        return e

    if n_diff == 1:
        # Single excitation: I→J via m→p (m∈only_I, p∈only_J).
        m = only_I[0]
        p = only_J[0]
        # Parity: number of occupied spin-orbitals between m and p in |I⟩ and |J⟩.
        occ_I_list = list(occ_I)
        sign = (-1) ** (
            sum(1 for x in occ_I_list if min(m, p) < x < max(m, p))
        )
        val = _h1_so(h1_MO, p, m)
        for b in common:
            val += _g_so_anti(g_MO, p, b, m, b)
        return sign * val

    if n_diff == 2:
        # Double excitation: m,n→p,q  (m<n, p<q from sorted only_I/only_J).
        m, n = only_I   # both sorted ascending
        p, q = only_J
        # Canonical sign: count transpositions to pull m then n out of
        # sorted(occ_I), and to pull p then q out of sorted(occ_J).
        # phase = (pos of m in sorted_I) + (pos of n in sorted_I) - 1
        #       + (pos of p in sorted_J) + (pos of q in sorted_J) - 1
        # The "−1" for n (and q) accounts for the shift after m (p) is removed.
        occ_I_s = sorted(occ_I)
        occ_J_s = sorted(occ_J)
        pm = occ_I_s.index(m)
        pn = occ_I_s.index(n) - 1   # m already removed → shift by 1
        pp = occ_J_s.index(p)
        pq = occ_J_s.index(q) - 1   # p already removed → shift by 1
        sign = (-1) ** (pm + pn + pp + pq)
        return sign * _g_so_anti(g_MO, p, q, m, n)

    return 0.0


def build_fci_matrix(
    h1_MO: np.ndarray,
    g_MO:  np.ndarray,
    N_e:   int,
    N_orb: int,
) -> np.ndarray:
    """Build the full FCI Hamiltonian matrix in the spin-orbital basis.

    Args:
        h1_MO : (N_orb, N_orb) 1e integrals in MO basis.
        g_MO  : (N_orb, N_orb, N_orb, N_orb) 2e ERI in MO basis (physicist).
        N_e   : Number of electrons.
        N_orb : Number of spatial orbitals (2*N_orb spin-orbitals).

    Returns:
        H_FCI : (dim, dim) real symmetric FCI matrix.
    """
    n_so = 2 * N_orb
    dets = list(combinations(range(n_so), N_e))
    dim  = len(dets)
    log.debug(f"[zetazero] FCI dim = {dim} ({N_e}e/{N_orb}orb)")

    H = np.zeros((dim, dim))
    for I, occ_I in enumerate(dets):
        for J in range(I, dim):
            occ_J = dets[J]
            val   = _slater_condon(occ_I, occ_J, h1_MO, g_MO, n_so)
            H[I, J] = val
            H[J, I] = val
    return H


def compute_rdms(
    psi_0: np.ndarray,
    N_e:   int,
    N_orb: int,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Compute the 1-RDM and 2-RDM from the FCI ground state.

    Works in the spin-orbital basis (2*N_orb spin-orbitals, same ordering
    as :func:`build_fci_matrix`) and returns spin-summed spatial-orbital
    density matrices.

    Definitions
    -----------
    1-RDM (spatial)::

        γ[p,q] = ⟨ψ₀|a†_{pα}a_{qα} + a†_{pβ}a_{qβ}|ψ₀⟩

    2-RDM (spatial, antisymmetric chemist convention)::

        Γ[p,q,r,s] = ⟨ψ₀|a†_{pα}a†_{rα}a_{sα}a_{qα} + cross-spin terms|ψ₀⟩

    stored as Γ[p,q,r,s] = Σ_{σσ'} ⟨ψ₀|a†_{pσ}a†_{rσ'}a_{sσ'}a_{qσ}|ψ₀⟩.

    These are the RDMs needed for the Hellmann–Feynman gradient of E_seed w.r.t.
    orbital rotation:

        ∂E_seed/∂C_{μ,p} = 2 Σ_q γ[p,q] h1_eff[μ,ν] C[ν,q]
                           + 2 Σ_{qrs} Γ[p,q,r,s] g_AO[μ,ν,λ,σ] C[ν,q] C[λ,r] C[σ,s]

    Args:
        psi_0 : (dim,) ground-state CI vector from :func:`solve_cas`.
        N_e   : Number of electrons.
        N_orb : Number of spatial orbitals.

    Returns:
        gamma : (N_orb, N_orb) spin-summed 1-RDM.
        Gamma : (N_orb, N_orb, N_orb, N_orb) spin-summed 2-RDM.
    """
    n_so = 2 * N_orb
    dets = list(combinations(range(n_so), N_e))
    dim  = len(dets)

    gamma = np.zeros((N_orb, N_orb))
    Gamma = np.zeros((N_orb, N_orb, N_orb, N_orb))

    for I, occ_I in enumerate(dets):
        ci_I = psi_0[I]
        if abs(ci_I) < 1e-15:
            continue
        set_I = set(occ_I)

        for J in range(dim):
            ci_J = psi_0[J]
            if abs(ci_J) < 1e-15:
                continue
            occ_J = dets[J]
            set_J = set(occ_J)

            only_I = sorted(set_I - set_J)
            only_J = sorted(set_J - set_I)
            n_diff = len(only_I)

            if n_diff == 0:
                # ⟨I|a†_p a_q|I⟩ — diagonal: q must be in occ_I
                for so_p in occ_I:
                    p = _spatial_idx(so_p)
                    gamma[p, p] += ci_I * ci_J
                # 2-RDM diagonal: Γ[p,q,r,s] += c_I² for all (p,r)∈occ_I, swap (q,s)
                for idx_a, so_p in enumerate(occ_I):
                    for so_r in occ_I[idx_a + 1:]:
                        p = _spatial_idx(so_p)
                        r = _spatial_idx(so_r)
                        val = ci_I * ci_J
                        # Γ[p,p,r,r] and antisymmetric permutations
                        Gamma[p, p, r, r] += val
                        Gamma[r, r, p, p] += val
                        Gamma[p, r, r, p] -= val
                        Gamma[r, p, p, r] -= val

            elif n_diff == 1:
                # Single: I differs from J by one spin-orbital  m→p
                so_m = only_I[0]   # annihilated from I
                so_p = only_J[0]   # created  in  I to get J
                # Parity: transpositions to bring so_m out of sorted(occ_I)
                occ_I_s = sorted(occ_I)
                occ_J_s = sorted(occ_J)
                pm = occ_I_s.index(so_m)
                pp = occ_J_s.index(so_p)
                sign = (-1) ** (pm + pp)
                p_sp = _spatial_idx(so_p)
                m_sp = _spatial_idx(so_m)
                if _spin(so_p) == _spin(so_m):
                    gamma[p_sp, m_sp] += sign * ci_I * ci_J

                # 2-RDM single-excitation contribution:
                # Γ[p,q,m,q] for q in common (same-spin coupling)
                common = sorted(set_I & set_J)
                for so_q in common:
                    q_sp = _spatial_idx(so_q)
                    # Sign for removing so_q from both sides (it's common)
                    occ_I_noq = [x for x in occ_I_s if x != so_q]
                    occ_J_noq = [x for x in occ_J_s if x != so_q]
                    pm2 = occ_I_noq.index(so_m) if so_m in occ_I_noq else 0
                    pp2 = occ_J_noq.index(so_p) if so_p in occ_J_noq else 0
                    # position of so_q
                    pq_I = occ_I_s.index(so_q)
                    pq_J = occ_J_s.index(so_q)
                    sign2 = (-1) ** (pm2 + pp2 + pq_I + pq_J)
                    if _spin(so_p) == _spin(so_m):
                        Gamma[p_sp, q_sp, m_sp, q_sp] += sign2 * ci_I * ci_J
                        Gamma[q_sp, p_sp, q_sp, m_sp] += sign2 * ci_I * ci_J
                        Gamma[p_sp, q_sp, q_sp, m_sp] -= sign2 * ci_I * ci_J
                        Gamma[q_sp, p_sp, m_sp, q_sp] -= sign2 * ci_I * ci_J

    return gamma, Gamma


def solve_cas(
    h1_MO: np.ndarray,
    g_MO:  np.ndarray,
    N_e:   int   = 4,
    N_orb: int   = 4,
    E_nuc: float = 0.0,
) -> Tuple[float, float, np.ndarray]:
    """Diagonalise the CAS(N_e, N_orb) FCI matrix.

    Args:
        h1_MO : (N_orb, N_orb) 1e integrals in MO basis.
        g_MO  : (N_orb, N_orb, N_orb, N_orb) 2e ERI in MO basis (physicist).
        N_e   : Number of electrons.
        N_orb : Number of spatial orbitals.
        E_nuc : Nuclear repulsion energy [Ha] to add to the electronic energy.

    Returns:
        E_elec : Electronic ground-state energy [Ha] (no E_nuc).
        E_total: E_elec + E_nuc [Ha] — comparable to PySCF CASCI total energy.
        psi_0  : (dim,) ground-state CI vector.
    """
    H = build_fci_matrix(h1_MO, g_MO, N_e, N_orb)
    vals, vecs = _eigh(H)
    E_elec = float(vals[0])
    return E_elec, E_elec + E_nuc, vecs[:, 0]
