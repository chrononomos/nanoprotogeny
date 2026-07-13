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
mqelocalize.py — Boys orbital localization.
===========================================
Computes position-operator AO matrices using McMurchie-Davidson
E-coefficients (the t=1 entry that the overlap integral leaves unused)
and applies Jacobi-sweep Foster-Boys localization to the canonical MO
coefficient matrix.

Localization is an optional Step 4 in the full-Hamiltonian build scope:
   canonical C  →  C_loc = C @ U
where U maximises F = Σ_i Σ_κ ⟨i|r_κ|i⟩².

No dependency on mechanism, Riemann, or tower logic.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import numpy as np

from nanoprotogeny.molecular.mqeaointegrals import (
    _CART_COMPS,
    _build_e1d,
    _contracted_norm_cart,
)

log = logging.getLogger(__name__)


# ===========================================================================
# SECTION 1 — PRIMITIVE DIPOLE INTEGRAL (McMurchie-Davidson)
# ===========================================================================

def _dipole_prim_cart(
    alpha: float,
    A: np.ndarray,
    la: Tuple[int, int, int],
    beta: float,
    B: np.ndarray,
    lb: Tuple[int, int, int],
    comp: int,
) -> float:
    r"""⟨G(α,A,la)|r_κ|G(β,B,lb)⟩ for two Cartesian GTOs.

    From the McMurchie-Davidson factorisation, the 1D dipole integral in
    component κ is:

        ⟨Ω^{la_κ} | r_κ | Ω^{lb_κ}⟩_{x_κ}
            = [ P_κ · E^{la_κ,lb_κ}_{0} + (1/(2p)) · E^{la_κ,lb_κ}_{1} ]
              × √(π/p)

    The other two 1D factors are the usual overlap integrals
    E^{la_ι,lb_ι}_0 × √(π/p).  Combining all three dimensions:

        ⟨G_a|r_κ|G_b⟩ = (π/p)^{3/2}
                         × [P_κ·E^{la_κ,lb_κ}_0 + (1/(2p))·E^{la_κ,lb_κ}_1]
                         × E^{la_y,lb_y}_0  (if κ≠y)
                         × E^{la_z,lb_z}_0  (if κ≠z)

    The full Gaussian pre-factor exp(-αβ/p·|AB|²) is placed on the
    x-dimension E-coefficient, exactly as in _overlap_prim_cart.

    Args:
        alpha, A, la : Exponent, centre (Bohr), angular indices of bra.
        beta,  B, lb : Exponent, centre (Bohr), angular indices of ket.
        comp         : Component to evaluate (0=x, 1=y, 2=z).

    Returns:
        ⟨G_a|r_comp|G_b⟩ (un-normalised primitive integral, in Bohr).
    """
    p      = alpha + beta
    P      = (alpha * A + beta * B) / p
    XPA    = P - A
    XPB    = P - B
    inv_2p = 0.5 / p
    # Full Gaussian overlap factor attached to dimension 0
    exp_ab = math.exp(-alpha * beta / p * float(np.dot(A - B, A - B)))

    result = (math.pi / p) ** 1.5
    for k in range(3):
        E = _build_e1d(
            la[k], lb[k],
            float(XPA[k]), float(XPB[k]),
            inv_2p,
            exp_ab if k == 0 else 1.0,
        )
        if k == comp:
            # Dipole: P_κ·E_0 + (1/(2p))·E_1
            # E^{la,lb}_t = 0 for t > la+lb; guard when la+lb = 0 (s-s pair).
            e1 = E[la[k], lb[k], 1] if la[k] + lb[k] >= 1 else 0.0
            result *= P[k] * E[la[k], lb[k], 0] + inv_2p * e1
        else:
            # Overlap: E_0
            result *= E[la[k], lb[k], 0]
    return result


# ===========================================================================
# SECTION 2 — CONTRACTED AO DIPOLE MATRICES
# ===========================================================================

def build_dipole_ao_matrices(
    shells: list,
    norms:  list,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contracted AO position-operator matrices ⟨μ|r_κ|ν⟩, κ = x, y, z.

    Each element is the sum over all primitive pairs, divided by the
    contracted norms, giving the properly normalised matrix element.

    Args:
        shells : Shell list from ``_build_basis_shells``.
                 Each entry: (center_bohr, (lx,ly,lz), alphas, coeffs).
        norms  : Contracted norms from ``_contracted_norm_cart``,
                 one float per shell.

    Returns:
        (x_ao, y_ao, z_ao) — each an (N_sh, N_sh) float64 array [Bohr].
    """
    N = len(shells)
    r_ao = [np.zeros((N, N)) for _ in range(3)]

    for mu, (A, la, amu, dmu) in enumerate(shells):
        n_mu = norms[mu]
        for nu, (B, lb, anu, dnu) in enumerate(shells):
            n_nu = norms[nu]
            scale = 1.0 / (n_mu * n_nu)
            for comp in range(3):
                val = 0.0
                for a, d_a in zip(amu, dmu):
                    for b, d_b in zip(anu, dnu):
                        val += d_a * d_b * _dipole_prim_cart(
                            a, A, la, b, B, lb, comp
                        )
                r_ao[comp][mu, nu] = val * scale

    x_ao, y_ao, z_ao = r_ao
    return x_ao, y_ao, z_ao


# ===========================================================================
# SECTION 3 — JACOBI-SWEEP BOYS LOCALIZATION
# ===========================================================================

def boys_localize(
    C:           np.ndarray,
    r_ao_list:   Tuple[np.ndarray, np.ndarray, np.ndarray],
    max_iter:    int   = 200,
    tol:         float = 1.0e-8,
    active_only: int   = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Foster-Boys localization via Jacobi sweeps.

    Maximises the Boys functional
        F = Σ_i Σ_κ ⟨ψ_i|r_κ|ψ_i⟩²
    by iterating 2×2 Jacobi rotations on the MO window [0, n_loc).

    For each pair (i, j) the optimal rotation angle is

        θ = (1/4) arctan2(B_{ij}, A_{ij})

    where
        A_{ij} = Σ_κ [ (r_κ)_{ii} − (r_κ)_{jj} ]² − 4(r_κ)_{ij}²
        B_{ij} = 4 Σ_κ [ (r_κ)_{ii} − (r_κ)_{jj} ] (r_κ)_{ij}

    The rotation is applied in-place to the MO-basis dipole matrices and
    accumulated in the running unitary U (C_loc = C @ U).

    Args:
        C          : (N_AO, N) canonical MO coefficient matrix.
        r_ao_list  : (x_ao, y_ao, z_ao) AO dipole matrices, each
                     (N_AO, N_AO) [Bohr].  Can also be AO-basis matrices
                     in the spherical representation.
        max_iter   : Maximum number of full Jacobi sweeps.
        tol        : Convergence on |ΔF| per sweep.
        active_only: If > 0, localize only MOs 0..active_only-1; the rest
                     pass through unchanged.  0 = localize all N MOs.

    Returns:
        C_loc : (N_AO, N) localized MO coefficient matrix.
        U     : (N, N) unitary s.t. C_loc = C @ U (columns of U are the
                rotation vectors in the canonical MO basis).
    """
    N_AO, N = C.shape
    n_loc = active_only if (0 < active_only <= N) else N

    # MO-basis dipole matrices for the localized window
    C_win = C[:, :n_loc]
    r_mo  = [C_win.T @ r @ C_win for r in r_ao_list]

    # Full unitary accumulator
    U = np.eye(N)

    F_prev = sum(float(np.dot(np.diag(r), np.diag(r))) for r in r_mo)
    log.info(
        f"[boys_localize] n_loc={n_loc}, N={N}, "
        f"initial F={F_prev:.10f}"
    )

    dF = float("inf")
    for sweep in range(max_iter):
        for i in range(n_loc - 1):
            for j in range(i + 1, n_loc):
                # Compute A_ij, B_ij
                Aij = 0.0
                Bij = 0.0
                for r in r_mo:
                    dii_jj = r[i, i] - r[j, j]
                    rij    = r[i, j]
                    Aij   += dii_jj * dii_jj - 4.0 * rij * rij
                    Bij   += dii_jj * rij
                Bij *= 4.0

                if abs(Bij) < 1.0e-15 and abs(Aij) < 1.0e-15:
                    continue

                theta = 0.25 * math.atan2(Bij, Aij)
                if abs(theta) < 1.0e-15:
                    continue

                cos_t = math.cos(theta)
                sin_t = math.sin(theta)

                # Update each r_mo[κ] via similarity transform G^T r G
                # where G = [[cos θ, -sin θ], [sin θ, cos θ]] acts on (i,j)
                for r in r_mo:
                    ri = r[i, :].copy()
                    rj = r[j, :].copy()
                    r[i, :] =  cos_t * ri + sin_t * rj
                    r[j, :] = -sin_t * ri + cos_t * rj
                    ci = r[:, i].copy()
                    cj = r[:, j].copy()
                    r[:, i] =  cos_t * ci + sin_t * cj
                    r[:, j] = -sin_t * ci + cos_t * cj

                # Accumulate rotation into U (columns i, j of localized block)
                Ui = U[:, i].copy()
                Uj = U[:, j].copy()
                U[:, i] =  cos_t * Ui + sin_t * Uj
                U[:, j] = -sin_t * Ui + cos_t * Uj

        F = sum(float(np.dot(np.diag(r), np.diag(r))) for r in r_mo)
        dF = abs(F - F_prev)
        log.debug(
            f"[boys_localize] sweep {sweep + 1:4d}: F={F:.10f}, ΔF={dF:.3e}"
        )
        F_prev = F
        if dF < tol:
            log.info(
                f"[boys_localize] converged at sweep {sweep + 1}: "
                f"F={F:.10f}, ΔF={dF:.3e}"
            )
            break
    else:
        log.warning(
            f"[boys_localize] did not converge in {max_iter} sweeps "
            f"(last ΔF={dF:.3e})"
        )

    C_loc = C @ U
    return C_loc, U
