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
mqeecp.py — Numerical effective-core-potential (ECP) integrals
==============================================================
Adds the one-electron ECP matrix ``V^ECP`` to ``h1_AO`` for atoms whose basis
is a small-core ECP set (def2-TZVP/TZVPP on Mo, Ag, …).  The analytical engine
is otherwise all-electron and uses the *full* nuclear charge; with an ECP atom
the nuclear charge is reduced to ``Z_eff = Z − n_core`` (done in mqeaobuild) and
``V^ECP`` supplies the core potential.

Scalar (spin-orbit-averaged) semilocal ECP:

    U_ECP(r) = U_L(r) + Σ_{l=0}^{L-1} [U_l(r) − U_L(r)] · P_l
    U_l(r)   = Σ_k d_{lk} · r^{n_{lk}-2} · exp(−ζ_{lk} r²)

with P_l the projector onto angular momentum l about the ECP centre, and U_L
(highest l) the local part.

Method: **numerical quadrature** on an ECP-centred grid (radial Becke transform
× product angular grid).  The angular projectors P_l reuse the harmonic
polynomials of :func:`mqeaointegrals._harmonic_null` (no spherical-harmonic
coefficient tables).  Recommended over analytic Type-2 recursions because the
engine's basis quality makes high-accuracy ECP integrals unnecessary
(see docs/ecp-support-scope.md).

Self-consistency (validatable without an external reference): with every U set
to 1, the local form reproduces the overlap S, and Σ_l P_l = 1 reproduces S —
both checked in tests/test_ecp.py.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from nanoprotogeny.molecular.mqeaointegrals import (
    _CART_COMPS,
    _contracted_norm_cart,
    _harmonic_null,
    _norm_cart,
)

Shell    = Tuple[np.ndarray, Tuple[int, int, int], List[float], List[float]]
EcpTerm  = Tuple[int, float, float]              # (n, zeta, d)
EcpBlock = Tuple[int, List[EcpTerm]]             # (l, terms)
EcpCentre = Tuple[np.ndarray, int, List[EcpBlock]]  # (centre_bohr, n_core, blocks)


# ── grids ──────────────────────────────────────────────────────────────────

def _radial_grid(n: int, rm: float) -> Tuple[np.ndarray, np.ndarray]:
    """Nodes/weights for ∫₀^∞ dr r² f(r) ≈ Σ w_k f(r_k) (Becke map of Gauss–Legendre)."""
    x, w = np.polynomial.legendre.leggauss(n)        # x ∈ (-1,1)
    r  = rm * (1.0 + x) / (1.0 - x)                  # → (0, ∞)
    dr = rm * 2.0 / (1.0 - x) ** 2
    return r, w * dr * r ** 2


def _angular_grid(ntheta: int, nphi: int) -> Tuple[np.ndarray, np.ndarray]:
    """Product Gauss–Legendre(cosθ) × uniform(φ); Σ w = 4π, exact for low-l Yₗₘ."""
    ct, wct = np.polynomial.legendre.leggauss(ntheta)
    st = np.sqrt(np.clip(1.0 - ct ** 2, 0.0, 1.0))
    phi = np.arange(nphi) * (2.0 * np.pi / nphi)
    dphi = 2.0 * np.pi / nphi
    dirs = []
    w = []
    for i in range(ntheta):
        for j in range(nphi):
            dirs.append((st[i] * np.cos(phi[j]), st[i] * np.sin(phi[j]), ct[i]))
            w.append(wct[i] * dphi)
    return np.asarray(dirs), np.asarray(w)


# ── basis-function evaluation on a point cloud ───────────────────────────────

def _eval_shell(shell: Shell, norm: float, pts: np.ndarray) -> np.ndarray:
    """Contracted normalised Cartesian GTO χ(r) evaluated on pts (P,3) → (P,)."""
    A, (lx, ly, lz), alphas, coeffs = shell
    d  = pts - A
    r2 = np.einsum("pi,pi->p", d, d)
    poly = (d[:, 0] ** lx) * (d[:, 1] ** ly) * (d[:, 2] ** lz)
    rad = np.zeros(len(pts))
    for a, c in zip(alphas, coeffs):
        rad += c * _norm_cart(a, lx, ly, lz) * np.exp(-a * r2)
    return poly * rad / norm


def _sph_on_grid(l: int, dirs: np.ndarray, wang: np.ndarray) -> np.ndarray:
    """Orthonormal degree-l harmonics on the unit-sphere grid → (2l+1, Ndir),
    with Σ_a w_a Yᵢ(a) Yⱼ(a) = δᵢⱼ (reuses the harmonic polynomials)."""
    H = _harmonic_null(l)                       # (n_cart, 2l+1) monomial coeffs
    comps = _CART_COMPS[l]
    mon = np.array([dirs[:, 0] ** cx * dirs[:, 1] ** cy * dirs[:, 2] ** cz
                    for (cx, cy, cz) in comps])  # (n_cart, Ndir)
    Y = H.T @ mon                                # (2l+1, Ndir) raw harmonics
    G = (Y * wang) @ Y.T                         # angular Gram
    w_, U = np.linalg.eigh(G)
    return (U @ np.diag(1.0 / np.sqrt(w_)) @ U.T) @ Y


# ── the ECP matrix ───────────────────────────────────────────────────────────

def _U(terms: Sequence[EcpTerm], r: np.ndarray) -> np.ndarray:
    out = np.zeros_like(r)
    for n, z, d in terms:
        out += d * r ** (n - 2) * np.exp(-z * r ** 2)
    return out


def ecp_matrix(
    shells: List[Shell],
    norms: Sequence[float],
    ecp_centres: List[EcpCentre],
    nr: int = 60,
    ntheta: int = 18,
    nphi: int = 36,
    rm: float = 1.0,
    _unit_potential: bool = False,
) -> np.ndarray:
    """Numerical ECP one-electron matrix V^ECP (N_AO × N_AO), summed over centres.

    ``_unit_potential=True`` replaces every U_l by 1 (validation: the local form
    then reproduces the overlap S, and Σ_l P_l = 1 also reproduces S).
    """
    N = len(shells)
    V = np.zeros((N, N))
    r, wr = _radial_grid(nr, rm)
    dirs, wang = _angular_grid(ntheta, nphi)
    ndir = len(dirs)

    for C, _ncore, blocks in ecp_centres:
        if not blocks:
            continue
        L = max(l for l, _ in blocks)
        block_of = {l: terms for l, terms in blocks}

        # evaluate every basis function on the C-centred (nr × ndir) grid
        pts = C[None, None, :] + r[:, None, None] * dirs[None, :, :]   # (nr,ndir,3)
        flat = pts.reshape(-1, 3)
        chi = np.array([_eval_shell(shells[i], norms[i], flat) for i in range(N)])
        chi = chi.reshape(N, nr, ndir)

        UL = np.ones(nr) if _unit_potential else _U(block_of[L], r)

        # local part: ∫ χ_μ U_L χ_ν d³r
        for k in range(nr):
            ck = chi[:, k, :]                              # (N, ndir)
            V += wr[k] * UL[k] * ((ck * wang) @ ck.T)

        # semilocal: Σ_{l<L} (U_l − U_L) P_l
        for l, terms in blocks:
            if l == L:
                continue
            Yl = _sph_on_grid(l, dirs, wang)               # (2l+1, ndir)
            dU = (np.ones(nr) if _unit_potential else _U(terms, r)) - UL
            for k in range(nr):
                ck = chi[:, k, :]                          # (N, ndir)
                proj = (ck * wang) @ Yl.T                  # (N, 2l+1) = ∫dΩ Y χ
                V += wr[k] * dU[k] * (proj @ proj.T)

    return V
