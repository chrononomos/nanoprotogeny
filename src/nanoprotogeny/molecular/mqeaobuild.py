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
mqeaobuild.py — AO integral matrix assembly and dispatch.
========================================================
Builds (S_AO, h1_AO, g_AO, E_nuc) for a geometry.  Routes to the s-only Boys
path, the McMurchie–Davidson p/d path, or the explicit PySCF fallback, and
computes the nuclear repulsion energy.  Sits one layer above
:mod:`mqeaointegrals`.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

import nanoprotogeny.basis.mqebasis as _mb
from nanoprotogeny.molecular.mqeconstants import (
    _BOHR_PER_ANG,
    _HAS_D_BASIS,
    _HAS_P_BASIS,
    _NUCLEAR_CHARGES,
)
from nanoprotogeny.molecular.mqeaointegrals import (
    _build_1e_contracted,
    _build_1e_contracted_cart,
    _build_basis_shells,
    _build_eri_batch_by_type,
    _contracted_norm,
    _contracted_norm_cart,
    _eri_contracted,
)
import nanoprotogeny.basis.mqebasisloader as _mbl

log = logging.getLogger(__name__)


def count_ao_basis(
    atoms:         List[Tuple[str, float, float, float]],
    basis_spec:    Optional[Dict[str, str]] = None,
    d_single_zeta: bool                     = True,
    full_shells:   bool                     = False,
) -> int:
    """Return N_AO for *atoms* without computing any integrals (fast shell count).

    Calls :func:`mqeaointegrals._build_basis_shells` which only walks the basis
    tables — no ERI primitives are evaluated.  Suitable for pre-flight checks
    before an expensive AO integral build.
    """
    shells = _build_basis_shells(
        atoms,
        d_single_zeta = d_single_zeta,
        basis_spec    = basis_spec,
        full_shells   = full_shells,
    )
    return len(shells)


def _effective_charges(
    atoms: List[Tuple[str, float, float, float]],
    basis_spec: Optional[Dict[str, str]],
) -> Tuple[List[float], list]:
    """Return (z_eff per atom, ecp_centres).

    For an atom whose ``basis_spec`` basis carries an ECP, the effective
    nuclear charge is ``Z − n_core`` and an ECP centre
    ``(centre_bohr, n_core, blocks)`` is recorded.  Non-ECP atoms keep the full
    ``Z`` and contribute no ECP centre.
    """
    z_eff: List[float] = []
    centres: list = []
    for sym, x, y, z in atoms:
        Z = _NUCLEAR_CHARGES.get(sym, 1.0)
        n_core, blocks = 0, []
        bname = (basis_spec or {}).get(sym)
        if bname is not None:
            try:
                n_core, blocks = _mbl.load_bse_ecp(bname, sym)
            except (FileNotFoundError, KeyError):
                n_core, blocks = 0, []
        z_eff.append(Z - n_core)
        if n_core > 0 and blocks:
            centres.append((np.array([x, y, z]) * _BOHR_PER_ANG, n_core, blocks))
    return z_eff, centres


def _basis_for(symbol: str) -> Tuple[List[float], List[float]]:
    """Return (alphas, s_coeffs) for the outermost valence s-shell of *symbol*.

    Used by the s-only integral path.  Reads from the authoritative
    :mod:`mqebasis` module (EMSL STO-3G, Hehre/Stewart/Pople 1969–1983).
    For SP shells the s-contraction coefficients are returned; the p-component
    is handled separately in :func:`mqeaointegrals._build_basis_shells`.
    """
    for stype, exps, c1, _c2 in reversed(_mb.get_shells(symbol)):
        if stype in ("S", "SP"):
            return list(exps), list(c1)
    # Fallback: single diffuse Gaussian (should never be reached for known elements)
    return [1.0000000, 0.3000000, 0.1000000], [-0.1000000, 0.2000000, 0.9000000]

def _build_ao_integrals_cart(
    atoms:         List[Tuple[str, float, float, float]],
    basis_spec:    Optional[Dict[str, str]] = None,
    d_single_zeta: bool                    = True,
    full_shells:   bool                    = False,
    spherical:     bool                    = False,
    ecp:           bool                    = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Analytical AO integrals for a mixed s/p/d/f/g basis via McMurchie-Davidson.

    Called automatically by ``build_ao_integrals`` when the atom list contains
    elements in ``_HAS_D_BASIS`` or ``_HAS_P_BASIS``.  Not called directly.

    With ``full_shells=True`` the genuine untruncated basis (all contractions,
    all angular momenta s..g) is used.  With ``spherical=True`` the Cartesian
    integrals are contracted to pure spherical harmonics (d/f/g contaminants
    removed) — matching the def2/cc spherical definition.  With ``ecp=True``
    (default), atoms whose basis carries an ECP use the reduced nuclear charge
    ``Z_eff = Z − n_core`` and the numerical ECP matrix ``V^ECP`` is added to
    ``h1`` (no-op for all-electron atoms).
    """
    shells  = _build_basis_shells(atoms, basis_spec=basis_spec,
                                  d_single_zeta=d_single_zeta, full_shells=full_shells)
    N_sh    = len(shells)

    # Effective nuclear charges (Z_eff for ECP atoms) and ECP centres.
    z_eff, ecp_centres = _effective_charges(atoms, basis_spec if ecp else None)
    nuclear = [
        (z_eff[i], np.array([x, y, z]) * _BOHR_PER_ANG)
        for i, (sym, x, y, z) in enumerate(atoms)
    ]

    # Contracted norms
    norms = []
    for (_, ang, alphas, coeffs) in shells:
        norms.append(_contracted_norm_cart(alphas, coeffs, *ang))

    S  = np.zeros((N_sh, N_sh))
    h1 = np.zeros((N_sh, N_sh))

    # One-electron matrices (nuclear attraction uses Z_eff)
    for mu, (A, la, amu, dmu) in enumerate(shells):
        for nu, (B, lb, anu, dnu) in enumerate(shells):
            S_raw, T_raw, V_raw = _build_1e_contracted_cart(
                amu, dmu, A, la, anu, dnu, B, lb, nuclear)
            n_mn = norms[mu] * norms[nu]
            S[mu, nu]  = S_raw / n_mn
            h1[mu, nu] = (T_raw + V_raw) / n_mn

    # ── ECP core potential added to h1 (no-op when no ECP atoms) ─────────────
    if ecp_centres:
        from nanoprotogeny.molecular.mqeecp import ecp_matrix
        h1 = h1 + ecp_matrix(shells, norms, ecp_centres)

    # Two-electron repulsion integrals — vectorised batch path.
    # _build_eri_batch_by_type groups unique (μ≥ν, λ≥σ, μν≥λσ) quartets by
    # angular-momentum type and processes all K primitives simultaneously via
    # numpy/scipy.  Falls back to the scalar loop for s-only bases (N_sh small).
    g = _build_eri_batch_by_type(shells)

    # ── Cartesian → spherical (pure harmonic) contraction ────────────────────
    # Drops the d/f/g lower-l contaminants so the basis matches def2/cc's
    # spherical definition.  S_sph = Cᵀ S C ; g_sph via the 4-index C-transform.
    if spherical:
        from nanoprotogeny.molecular.mqeaointegrals import cart_to_sph_transform
        C  = cart_to_sph_transform(shells)
        S  = C.T @ S @ C
        h1 = C.T @ h1 @ C
        g  = np.einsum("pi,qj,pqrs,rk,sl->ijkl", C, C, g, C, C, optimize=True)
        N_sh = C.shape[1]

    # Nuclear repulsion uses Z_eff so ECP cores repel by their reduced charge.
    E_nuc = _nuclear_repulsion(atoms, charges=z_eff)
    log.info(
        f"[zetazero] Cart AO integrals: N_AO={N_sh}, E_nuc={E_nuc:.6f} Ha, "
        f"shells={[(s[1], len(s[2])) for s in shells[:8]]}…"
    )
    return S, h1, g, E_nuc


# ===========================================================================
# SECTION 4 — BUILD AO INTEGRAL MATRICES
# ===========================================================================

def _nuclear_repulsion(
    atoms: List[Tuple[str, float, float, float]],
    charges: Optional[List[float]] = None,
) -> float:
    """E_nuc = Σ_{α<β} Z_α Z_β / |R_α − R_β|  [Ha].

    Args:
        atoms:   List of (symbol, x, y, z) in Angstroms.
        charges: Optional per-atom charges (use Z_eff for ECP atoms); defaults
                 to the full nuclear charges from ``_NUCLEAR_CHARGES``.
    """
    coords_bohr = [
        np.array([x, y, z]) * _BOHR_PER_ANG for (_, x, y, z) in atoms
    ]
    if charges is None:
        charges = [_NUCLEAR_CHARGES.get(sym, 1.0) for (sym, *_) in atoms]
    E_nuc = 0.0
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n):
            r = float(np.linalg.norm(coords_bohr[i] - coords_bohr[j]))
            if r > 1.0e-10:
                E_nuc += charges[i] * charges[j] / r
    return E_nuc


def _has_d_basis_atoms(
    atoms: List[Tuple[str, float, float, float]],
) -> bool:
    """Return True if any atom in *atoms* is in ``_HAS_D_BASIS``."""
    return any(sym in _HAS_D_BASIS for (sym, *_) in atoms)


def _build_ao_integrals_pyscf(
    atoms: List[Tuple[str, float, float, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """PySCF-based AO integrals using the full STO-3G basis (s + p + d shells).

    **Not called by the seed-free pipeline.**  This is an explicit-use utility
    for callers that intentionally want PySCF integrals (e.g. the hybrid path).
    ``build_ao_integrals`` always uses the analytical s-only path.

    The returned matrices are in the full AO basis (dimension = number of
    contracted basis functions), NOT pre-truncated to N_active MOs.
    The downstream ``build_core_ham_guess`` + ``transform_integrals`` pipeline
    handles orbital selection as usual.

    Args:
        atoms : List of (symbol, x, y, z) in Angstroms.

    Returns:
        S_AO  : (N_AO, N_AO) overlap matrix
        h1_AO : (N_AO, N_AO) core Hamiltonian T + V
        g_AO  : (N_AO, N_AO, N_AO, N_AO) ERI tensor (μν|λσ) chemist notation
        E_nuc : nuclear repulsion energy [Ha]

    Raises:
        ImportError : if pyscf is not installed.
    """
    from pyscf import gto

    mol_str = "; ".join(f"{sym} {x} {y} {z}" for (sym, x, y, z) in atoms)
    mol = gto.Mole()
    mol.atom   = mol_str
    mol.basis  = "sto-3g"
    mol.unit   = "Angstrom"
    mol.charge = 0
    mol.spin   = 0
    mol.verbose = 0
    mol.build()

    S_AO   = mol.intor("int1e_ovlp")
    T_AO   = mol.intor("int1e_kin")
    V_AO   = mol.intor("int1e_nuc")
    h1_AO  = T_AO + V_AO
    g_AO   = mol.intor("int2e")     # shape (N, N, N, N), chemist (μν|λσ)
    E_nuc  = mol.energy_nuc()

    log.info(
        f"[zetazero] PySCF AO integrals: N_AO={S_AO.shape[0]}, "
        f"E_nuc={E_nuc:.6f} Ha"
    )
    return S_AO, h1_AO, g_AO, E_nuc


def build_ao_integrals(
    atoms:         List[Tuple[str, float, float, float]],
    basis_spec:    Optional[Dict[str, str]] = None,
    d_single_zeta: bool                    = True,
    full_shells:   bool                    = False,
    spherical:     bool                    = False,
    ecp:           bool                    = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute (S_AO, h1_AO, g_AO, E_nuc) analytically for the given geometry.

    Routing:
    * If *basis_spec* overrides any element, or if any atom is in
      ``_HAS_D_BASIS`` / ``_HAS_P_BASIS``, delegates to
      ``_build_ao_integrals_cart`` (McMurchie-Davidson p/d path with the
      shells resolved from either the BSE mirror or :mod:`mqebasis`).
    * Otherwise uses the s-only Boys path (one contracted s-function per atom,
      valence exponents from :mod:`mqebasis`).

    No PySCF, no SCF iteration.  ``_build_ao_integrals_pyscf`` is a separate
    explicit-use utility and is never called automatically.

    Args:
        atoms:         List of (symbol, x, y, z) in Angstroms.
        basis_spec:    Optional per-element basis override dict, e.g.
                       ``{"Fe": "def2-TZVP", "S": "def2-TZVP"}``.  Elements not
                       listed fall back to STO-3G.  ``None`` means STO-3G for all.
                       Choose the basis so that N_AO ≥ n_total_orbs: for small
                       models def2-TZVP with ``--full-basis`` may be needed to
                       reach the required orbital count; for large molecules
                       def2-SVP suffices and is cheaper.
        d_single_zeta: If True (default), use only the most diffuse single
                       primitive of the outermost D contraction for each TM atom
                       (fast, O(N⁴) cost).  Set False to include all contracted
                       D shells from the BSE basis (larger basis → more MOs,
                       needed when n_total_orbs > 20 for tower building).

    Returns:
        S_AO  : (N_AO, N_AO) overlap matrix
        h1_AO : (N_AO, N_AO) core Hamiltonian T + V
        g_AO  : (N_AO, N_AO, N_AO, N_AO) electron repulsion integrals (μν|λσ) chemist
        E_nuc : nuclear repulsion energy [Ha]
    """
    # Route to Cartesian p/d/f/g path when any BSE override is active, when
    # full_shells is requested, or when the atom list contains TMs or S that
    # need p/d functions.
    if spherical or full_shells or basis_spec or any(
        sym in _HAS_D_BASIS or sym in _HAS_P_BASIS for sym, *_ in atoms
    ):
        return _build_ao_integrals_cart(
            atoms, basis_spec=basis_spec,
            d_single_zeta=d_single_zeta, full_shells=full_shells,
            spherical=spherical, ecp=ecp,
        )

    N = len(atoms)
    coords = [
        np.array([x, y, z]) * _BOHR_PER_ANG
        for (_, x, y, z) in atoms
    ]
    bases = [_basis_for(sym) for (sym, *_) in atoms]
    nuclear = [
        (_NUCLEAR_CHARGES.get(sym, 1.0), coords[i])
        for i, (sym, *_) in enumerate(atoms)
    ]

    # Normalise each contracted basis function.
    norms = [_contracted_norm(*bases[i]) for i in range(N)]

    S   = np.zeros((N, N))
    h1  = np.zeros((N, N))
    g   = np.zeros((N, N, N, N))

    for mu in range(N):
        a_mu, d_mu = bases[mu]
        R_mu = coords[mu]
        n_mu = norms[mu]
        for nu in range(N):
            a_nu, d_nu = bases[nu]
            R_nu = coords[nu]
            n_nu = norms[nu]
            S_raw, T_raw, V_raw = _build_1e_contracted(
                a_mu, d_mu, R_mu,
                a_nu, d_nu, R_nu,
                nuclear,
            )
            S[mu, nu]  = S_raw / (n_mu * n_nu)
            h1[mu, nu] = (T_raw + V_raw) / (n_mu * n_nu)

    for mu in range(N):
        a_mu, d_mu = bases[mu]
        R_mu = coords[mu]
        n_mu = norms[mu]
        for nu in range(N):
            a_nu, d_nu = bases[nu]
            R_nu = coords[nu]
            n_nu = norms[nu]
            for lam in range(N):
                a_lam, d_lam = bases[lam]
                R_lam = coords[lam]
                n_lam = norms[lam]
                for sig in range(N):
                    a_sig, d_sig = bases[sig]
                    R_sig = coords[sig]
                    n_sig = norms[sig]
                    g[mu, nu, lam, sig] = (
                        _eri_contracted(
                            a_mu, d_mu, R_mu,
                            a_nu, d_nu, R_nu,
                            a_lam, d_lam, R_lam,
                            a_sig, d_sig, R_sig,
                        )
                        / (n_mu * n_nu * n_lam * n_sig)
                    )

    E_nuc = _nuclear_repulsion(atoms)
    return S, h1, g, E_nuc


# ===========================================================================
# SECTION 5 — SHELL-EXPOSING VARIANT FOR SCREENED / LOCALIZED PATH (STEP 5)
# ===========================================================================

def build_ao_integrals_with_shells(
    atoms:         List[Tuple[str, float, float, float]],
    basis_spec:    Optional[Dict[str, str]] = None,
    d_single_zeta: bool                    = True,
    full_shells:   bool                    = False,
    spherical:     bool                    = False,
    ecp:           bool                    = True,
) -> Tuple[np.ndarray, np.ndarray, float, list, list, Optional[np.ndarray]]:
    """Like ``build_ao_integrals`` but returns shell data instead of g_AO.

    Used by the screened integral-direct path (Step 5) and the Boys
    localization path (Step 4).  The g_AO tensor is **not** built; callers
    are expected to call :func:`~mqeaointegrals.screened_direct_ao_to_mo`
    directly.

    Returns:
        S_AO      : (N_AO, N_AO) overlap matrix.
        h1_AO     : (N_AO, N_AO) core Hamiltonian [Ha].
        E_nuc     : Nuclear repulsion energy [Ha].
        shells    : Cartesian shell list from ``_build_basis_shells``.
                    In the spherical case these are the *Cartesian* shells
                    that the screened path uses to compute primitives.
        norms     : Contracted norms (one per shell in *shells*).
        sph_C     : If ``spherical=True``, the (N_cart, N_sph) Cartesian →
                    spherical transform matrix; ``None`` otherwise.
                    Pass as the first factor in ``C_combined = sph_C.T @ C_mo``
                    to map the screened primitive ERIs (Cartesian basis) to
                    the MO basis in the spherical representation.
    """
    from nanoprotogeny.molecular.mqeaointegrals import (
        _build_basis_shells,
        _contracted_norm_cart,
        cart_to_sph_transform,
    )

    # ── Build shells and 1e matrices (same as _build_ao_integrals_cart) ───────
    shells = _build_basis_shells(
        atoms, basis_spec=basis_spec,
        d_single_zeta=d_single_zeta, full_shells=full_shells,
    )
    N_sh = len(shells)

    z_eff, ecp_centres = _effective_charges(atoms, basis_spec if ecp else None)
    nuclear = [
        (z_eff[i], np.array([x, y, z]) * _BOHR_PER_ANG)
        for i, (sym, x, y, z) in enumerate(atoms)
    ]

    norms = [_contracted_norm_cart(s[2], s[3], *s[1]) for s in shells]

    S  = np.zeros((N_sh, N_sh))
    h1 = np.zeros((N_sh, N_sh))
    for mu, (A, la, amu, dmu) in enumerate(shells):
        for nu, (B, lb, anu, dnu) in enumerate(shells):
            S_raw, T_raw, V_raw = _build_1e_contracted_cart(
                amu, dmu, A, la, anu, dnu, B, lb, nuclear)
            n_mn = norms[mu] * norms[nu]
            S[mu, nu]  = S_raw / n_mn
            h1[mu, nu] = (T_raw + V_raw) / n_mn

    if ecp_centres:
        from nanoprotogeny.molecular.mqeecp import ecp_matrix
        h1 = h1 + ecp_matrix(shells, norms, ecp_centres)

    E_nuc = _nuclear_repulsion(atoms, charges=z_eff)

    # ── Spherical transform of 1e matrices (g stays Cartesian for screened) ──
    sph_C: Optional[np.ndarray] = None
    if spherical:
        sph_C = cart_to_sph_transform(shells)   # (N_cart, N_sph)
        S  = sph_C.T @ S  @ sph_C
        h1 = sph_C.T @ h1 @ sph_C

    log.info(
        f"[build_with_shells] N_cart={N_sh}, "
        f"N_AO={S.shape[0]}, "
        f"E_nuc={E_nuc:.6f} Ha, "
        f"spherical={spherical}"
    )
    return S, h1, E_nuc, shells, norms, sph_C
