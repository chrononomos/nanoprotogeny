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
mqeaointegrals.py — Analytical AO integral engine (Boys + McMurchie–Davidson).
============================================================================
Leaf-level quantum-chemistry integral library: Boys functions, the
McMurchie–Davidson Cartesian GTO path (overlap, kinetic, nuclear attraction,
ERI), the vectorised angular-type batch ERI path, the s-only Boys primitives,
basis-shell construction, and 8-fold ERI packing helpers.  No dependency on
mechanism, Riemann, or tower logic.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

import nanoprotogeny.basis.mqebasis as _mb
import nanoprotogeny.basis.mqebasisloader as _mbl
from nanoprotogeny.molecular.mqeconstants import (
    _BOHR_PER_ANG,
    _CART_COMPS,
    _HAS_D_BASIS,
    _HAS_P_BASIS,
)

log = logging.getLogger(__name__)

# ===========================================================================
# SECTION 1 — BOYS FUNCTION AND GAUSSIAN PRIMITIVES
# ===========================================================================

def _boys_f0(x: float) -> float:
    r"""F_0(x) = (√π / (2√x)) erf(√x).

    F_0(0) = 1 by L'Hôpital.  Uses math.erf for x > 1e-8.
    """
    if x < 1.0e-8:
        return 1.0 - x / 3.0 + x * x / 10.0
    sqrt_x = math.sqrt(x)
    return (math.sqrt(math.pi) / (2.0 * sqrt_x)) * math.erf(sqrt_x)


def _boys_fn(n: int, x: float) -> float:
    r"""Generalized Boys function F_n(x) = ∫₀¹ t^{2n} exp(−x t²) dt.

    F_0 is handled by _boys_f0.  For n ≥ 1 the regularized lower incomplete
    gamma function is used:

        F_n(x) = Γ(n + ½) · P(n + ½, x) / (2 · x^{n + ½})

    where P(a, x) = γ(a, x) / Γ(a) is scipy's ``gammainc``.

    For x < 1e-8 a Taylor series is used:

        F_n(x) ≈ Σ_{k≥0} (−x)^k / (k! · (2n + 2k + 1))

    Supports n = 0, 1, 2, 3, 4 (sufficient for (dd|dd) ERIs).
    """
    if n == 0:
        return _boys_f0(x)
    if x < 1.0e-8:
        # Taylor: F_n(x) = 1/(2n+1) - x/(2n+3) + x²/(2(2n+5)) - ...
        val   = 0.0
        xpow  = 1.0
        fac   = 1.0
        denom = 2 * n + 1.0
        for k in range(20):
            term = xpow / (fac * denom)
            val += (term if k % 2 == 0 else -term)
            xpow  *= x
            fac   *= (k + 1)
            denom += 2.0
            if abs(xpow / fac) < 1.0e-16:
                break
        return val
    from scipy.special import gammainc as _gammainc, gamma as _gamma
    n_half = n + 0.5
    # gammainc returns the *regularized* lower incomplete gamma P(a,x) ∈ [0,1].
    return 0.5 * _gamma(n_half) * _gammainc(n_half, x) / (x ** n_half)


def _norm_s(alpha: float) -> float:
    r"""Normalisation constant for a primitive s-type GTO N(α).

    N(α) = (2α/π)^{3/4}
    """
    return (2.0 * alpha / math.pi) ** 0.75


def _contracted_norm(
    alphas: List[float],
    coeffs: List[float],
) -> float:
    """Self-overlap ⟨χ|χ⟩^{1/2} for a contracted s-type GTO.

    Used to normalise contracted basis functions.
    """
    s = 0.0
    for i, (a, d_i) in enumerate(zip(alphas, coeffs)):
        for j, (b, d_j) in enumerate(zip(alphas, coeffs)):
            s += d_i * d_j * _norm_s(a) * _norm_s(b) * (
                math.pi / (a + b)
            ) ** 1.5
    return math.sqrt(s)


# ===========================================================================
# SECTION 1B — CARTESIAN GTO BASIS (p/d) — McMURCHIE-DAVIDSON INTEGRALS
# ===========================================================================


def _dbl_fact(n: int) -> float:
    """Double factorial n!! with the convention (-1)!! = 1."""
    if n <= 0:
        return 1.0
    r = 1.0
    while n > 0:
        r *= n
        n -= 2
    return r


def _norm_cart(alpha: float, lx: int, ly: int, lz: int) -> float:
    r"""Normalisation constant for a primitive Cartesian GTO.

    N(α, lx, ly, lz) = 1 / sqrt(⟨φ|φ⟩) where the self-overlap along axis i is
      ∫ t^{2li} exp(-2α t²) dt = (2li−1)!! / (4α)^li · √(π/(2α)).
    """
    def _s1d(n: int) -> float:
        return _dbl_fact(2 * n - 1) / (4.0 * alpha) ** n * math.sqrt(math.pi / (2.0 * alpha))
    return 1.0 / math.sqrt(_s1d(lx) * _s1d(ly) * _s1d(lz))


def _build_e1d(
    la: int, lb: int,
    XPA: float, XPB: float, inv_2p: float, exp_ab: float,
) -> np.ndarray:
    r"""McMurchie-Davidson E_t^{la,lb} coefficients for one Cartesian axis.

    Recurrence (E[0,0,0] = exp(-μ (A_i−B_i)²) = ``exp_ab``):
      E[ia+1, ib, t] = XPA·E[ia,ib,t] + inv_2p·E[ia,ib,t-1] + (t+1)·E[ia,ib,t+1]
      E[ia, ib+1, t] = XPB·E[ia,ib,t] + inv_2p·E[ia,ib,t-1] + (t+1)·E[ia,ib,t+1]

    Returns E[0..la, 0..lb, 0..la+lb].
    """
    tmax = la + lb
    E = np.zeros((la + 2, lb + 2, tmax + 2))
    E[0, 0, 0] = exp_ab

    # Vertical recurrence: build E[1..la, 0, *]
    for ia in range(la):
        for t in range(ia + 2):
            v = XPA * (E[ia, 0, t] if t <= ia else 0.0)
            if t > 0:
                v += inv_2p * (E[ia, 0, t - 1] if t - 1 <= ia else 0.0)
            v += (t + 1) * (E[ia, 0, t + 1] if t + 1 <= ia else 0.0)
            E[ia + 1, 0, t] = v

    # Horizontal recurrence: build E[*, 1..lb, *]
    for ib in range(lb):
        for ia in range(la + 1):
            mt = ia + ib
            for t in range(mt + 2):
                v = XPB * (E[ia, ib, t] if t <= mt else 0.0)
                if t > 0:
                    v += inv_2p * (E[ia, ib, t - 1] if t - 1 <= mt else 0.0)
                v += (t + 1) * (E[ia, ib, t + 1] if t + 1 <= mt else 0.0)
                E[ia, ib + 1, t] = v

    return E[:la + 1, :lb + 1, :tmax + 1]


def _build_R(
    p: float,
    Px: float, Py: float, Pz: float,
    Cx: float, Cy: float, Cz: float,
    t_max: int, u_max: int, v_max: int,
) -> np.ndarray:
    r"""Hermite Coulomb integrals R[t, u, v, n].

    Base: R[0,0,0,n] = (-2p)^n F_n(p |P-C|²).
    Recurrences (vectorised over the n dimension via numpy slices):
      R[t+1, u, v, :N] = (Px-Cx) R[t, u, v, 1:N+1]  + t R[t-1, u, v, 1:N+1]
    Similarly for u and v.

    Used for nuclear attraction (p=α+β, C=R_nuc) and ERI (p=ρ, C=Q_cd).

    Returns R[0..t_max, 0..u_max, 0..v_max, 0..n_max].
    """
    PCx = Px - Cx; PCy = Py - Cy; PCz = Pz - Cz
    eta   = p * (PCx * PCx + PCy * PCy + PCz * PCz)
    n_max = t_max + u_max + v_max

    R = np.zeros((t_max + 1, u_max + 1, v_max + 1, n_max + 1))
    twop = -2.0 * p
    for n in range(n_max + 1):
        R[0, 0, 0, n] = twop ** n * _boys_fn(n, eta)

    # t direction — vectorised over n
    for t in range(t_max):
        nt = n_max - t - 1
        if nt <= 0:
            break
        R[t + 1, 0, 0, :nt] = PCx * R[t, 0, 0, 1:nt + 1]
        if t > 0:
            R[t + 1, 0, 0, :nt] += t * R[t - 1, 0, 0, 1:nt + 1]

    # u direction — vectorised over n
    for u in range(u_max):
        for t in range(t_max + 1):
            nu = n_max - t - u - 1
            if nu <= 0:
                continue
            R[t, u + 1, 0, :nu] = PCy * R[t, u, 0, 1:nu + 1]
            if u > 0:
                R[t, u + 1, 0, :nu] += u * R[t, u - 1, 0, 1:nu + 1]

    # v direction — vectorised over n
    for v in range(v_max):
        for t in range(t_max + 1):
            for u in range(u_max + 1):
                nv = n_max - t - u - v - 1
                if nv <= 0:
                    continue
                R[t, u, v + 1, :nv] = PCz * R[t, u, v, 1:nv + 1]
                if v > 0:
                    R[t, u, v + 1, :nv] += v * R[t, u, v - 1, 1:nv + 1]

    return R


def _overlap_prim_cart(
    alpha: float, A: np.ndarray, la: Tuple[int, int, int],
    beta:  float, B: np.ndarray, lb: Tuple[int, int, int],
) -> float:
    """Normalised primitive overlap ⟨φ_a|φ_b⟩ for Cartesian GTOs (any l ≤ 2)."""
    p      = alpha + beta
    mu     = alpha * beta / p
    P      = (alpha * A + beta * B) / p
    inv_2p = 1.0 / (2.0 * p)
    E0 = 1.0
    for dim in range(3):
        E = _build_e1d(la[dim], lb[dim],
                       P[dim] - A[dim], P[dim] - B[dim],
                       inv_2p,
                       math.exp(-mu * (A[dim] - B[dim]) ** 2))
        E0 *= E[la[dim], lb[dim], 0]
    return _norm_cart(alpha, *la) * _norm_cart(beta, *lb) * (math.pi / p) ** 1.5 * E0


def _kinetic_prim_cart(
    alpha: float, A: np.ndarray, la: Tuple[int, int, int],
    beta:  float, B: np.ndarray, lb: Tuple[int, int, int],
) -> float:
    r"""Normalised primitive kinetic energy ⟨φ_a|T|φ_b⟩ for Cartesian GTOs.

    T_AB = -(1/2) Σ_i [T1d_i · Π_{j≠i} S1d_j]
    T1d_i = lbi(lbi-1) S1d(la_i,lbi-2) - 2β(2lbi+1) S1d(la_i,lbi) + 4β² S1d(la_i,lbi+2)
    """
    p      = alpha + beta
    mu     = alpha * beta / p
    P      = (alpha * A + beta * B) / p
    inv_2p = 1.0 / (2.0 * p)
    pref   = (math.pi / p) ** 1.5

    S1d: List[float] = []
    T1d: List[float] = []
    for dim in range(3):
        la_d, lb_d = la[dim], lb[dim]
        exp_ab = math.exp(-mu * (A[dim] - B[dim]) ** 2)
        # Extend to lb+2 to cover the +2 term
        E = _build_e1d(la_d, lb_d + 2, P[dim] - A[dim], P[dim] - B[dim], inv_2p, exp_ab)
        s0      = E[la_d, lb_d, 0]
        s_plus  = E[la_d, lb_d + 2, 0]
        s_minus = E[la_d, lb_d - 2, 0] if lb_d >= 2 else 0.0
        S1d.append(s0)
        T1d.append(
            lb_d * (lb_d - 1) * s_minus
            - 2.0 * beta * (2 * lb_d + 1) * s0
            + 4.0 * beta * beta * s_plus
        )

    T = (T1d[0] * S1d[1] * S1d[2]
         + S1d[0] * T1d[1] * S1d[2]
         + S1d[0] * S1d[1] * T1d[2])
    return _norm_cart(alpha, *la) * _norm_cart(beta, *lb) * (-0.5) * pref * T


def _nuclear_prim_cart(
    alpha: float, A: np.ndarray, la: Tuple[int, int, int],
    beta:  float, B: np.ndarray, lb: Tuple[int, int, int],
    Z: float, C_nuc: np.ndarray,
) -> float:
    """Normalised primitive nuclear attraction ⟨φ_a|−Z/|r−C||φ_b⟩."""
    p      = alpha + beta
    mu     = alpha * beta / p
    P      = (alpha * A + beta * B) / p
    inv_2p = 1.0 / (2.0 * p)
    t_max  = la[0] + lb[0]; u_max = la[1] + lb[1]; v_max = la[2] + lb[2]

    Edim = []
    for dim in range(3):
        exp_ab = math.exp(-mu * (A[dim] - B[dim]) ** 2)
        Edim.append(_build_e1d(la[dim], lb[dim],
                               P[dim] - A[dim], P[dim] - B[dim],
                               inv_2p, exp_ab))

    R = _build_R(p, P[0], P[1], P[2], C_nuc[0], C_nuc[1], C_nuc[2],
                 t_max, u_max, v_max)
    ex = Edim[0][la[0], lb[0], :t_max + 1]
    ey = Edim[1][la[1], lb[1], :u_max + 1]
    ez = Edim[2][la[2], lb[2], :v_max + 1]
    A   = np.einsum('i,j,k->ijk', ex, ey, ez)
    val = float(np.sum(A * R[:t_max + 1, :u_max + 1, :v_max + 1, 0]))

    return (_norm_cart(alpha, *la) * _norm_cart(beta, *lb)
            * (-Z) * (2.0 * math.pi / p) * val)


def _eri_prim_cart(
    alpha: float, A: np.ndarray, la: Tuple[int, int, int],
    beta:  float, B: np.ndarray, lb: Tuple[int, int, int],
    gamma: float, C: np.ndarray, lc: Tuple[int, int, int],
    delta: float, D: np.ndarray, ld: Tuple[int, int, int],
) -> float:
    r"""Normalised primitive ERI (ab|cd) via McMurchie-Davidson.

    (ab|cd) = (2π^{5/2}/(pq√(p+q))) Σ_{tuv,t'u'v'} E_t^{ab} E_u^{ab} E_v^{ab}
              · (-1)^{t'+u'+v'} E_{t'}^{cd} E_{u'}^{cd} E_{v'}^{cd} · R_{t+t',u+u',v+v'}^{(0)}
    """
    p   = alpha + beta;  P = (alpha * A + beta * B) / p
    q   = gamma + delta; Q = (gamma * C + delta * D) / q
    rho = p * q / (p + q)

    inv_2p = 1.0 / (2.0 * p); inv_2q = 1.0 / (2.0 * q)
    mu_ab  = alpha * beta / p;  mu_cd = gamma * delta / q

    Eab = []; Ecd = []
    for dim in range(3):
        Eab.append(_build_e1d(
            la[dim], lb[dim], P[dim] - A[dim], P[dim] - B[dim],
            inv_2p, math.exp(-mu_ab * (A[dim] - B[dim]) ** 2)))
        Ecd.append(_build_e1d(
            lc[dim], ld[dim], Q[dim] - C[dim], Q[dim] - D[dim],
            inv_2q, math.exp(-mu_cd * (C[dim] - D[dim]) ** 2)))

    t_max  = la[0]+lb[0]; u_max  = la[1]+lb[1]; v_max  = la[2]+lb[2]
    tp_max = lc[0]+ld[0]; up_max = lc[1]+ld[1]; vp_max = lc[2]+ld[2]

    R = _build_R(rho, P[0], P[1], P[2], Q[0], Q[1], Q[2],
                 t_max + tp_max, u_max + up_max, v_max + vp_max)

    # Apply (-1)^{t'+u'+v'} sign to Ecd via per-axis sign vectors.
    # (-1)^{t'+u'+v'} = (-1)^t' * (-1)^u' * (-1)^v', so absorb into each vector.
    def _signed(vec: np.ndarray) -> np.ndarray:
        s = vec.copy()
        s[1::2] *= -1.0
        return s

    ex_ab = Eab[0][la[0], lb[0], :t_max  + 1]
    ey_ab = Eab[1][la[1], lb[1], :u_max  + 1]
    ez_ab = Eab[2][la[2], lb[2], :v_max  + 1]
    ex_cd = _signed(Ecd[0][lc[0], ld[0], :tp_max + 1])
    ey_cd = _signed(Ecd[1][lc[1], ld[1], :up_max + 1])
    ez_cd = _signed(Ecd[2][lc[2], ld[2], :vp_max + 1])

    # Outer product tensors A[t,u,v] and B[tp,up,vp] (numpy, no Python loop)
    A = np.einsum('i,j,k->ijk', ex_ab, ey_ab, ez_ab)   # (T, U, V)
    B = np.einsum('i,j,k->ijk', ex_cd, ey_cd, ez_cd)   # (Tp, Up, Vp)

    # Build R0[t+tp, u+up, v+vp] indexed as R_ri[t, tp, u, up, v, vp]
    # using numpy fancy indexing — avoids all Python loops.
    T  = t_max  + 1; U  = u_max  + 1; V  = v_max  + 1
    Tp = tp_max + 1; Up = up_max + 1; Vp = vp_max + 1
    t_idx = np.arange(T)[:, None] + np.arange(Tp)[None, :]   # (T,Tp)
    u_idx = np.arange(U)[:, None] + np.arange(Up)[None, :]   # (U,Up)
    v_idx = np.arange(V)[:, None] + np.arange(Vp)[None, :]   # (V,Vp)
    R_ri  = R[t_idx[:, :, None, None, None, None],
               u_idx[None, None, :, :, None, None],
               v_idx[None, None, None, None, :, :],
               0]                                              # (T,Tp,U,Up,V,Vp)

    # Contract: val = Σ A[t,u,v]*B[tp,up,vp]*R_ri[t,tp,u,up,v,vp]
    A_bc = A[:, np.newaxis, :, np.newaxis, :, np.newaxis]     # (T,1,U,1,V,1)
    B_bc = B[np.newaxis, :, np.newaxis, :, np.newaxis, :]     # (1,Tp,1,Up,1,Vp)
    val  = float(np.sum(A_bc * B_bc * R_ri))

    N_abcd = (_norm_cart(alpha, *la) * _norm_cart(beta, *lb)
              * _norm_cart(gamma, *lc) * _norm_cart(delta, *ld))
    return N_abcd * 2.0 * math.pi ** 2.5 / (p * q * math.sqrt(p + q)) * val


# ===========================================================================
# SECTION 1C — VECTORISED BATCH ERI (Boys + E1d + R over K quartets)
# ===========================================================================

def _boys_fn_arr(n: int, x: np.ndarray) -> np.ndarray:
    r"""Vectorised Boys function F_n(x) for a 1-D float64 array x.

    Uses scipy.special.gammainc for x ≥ 1e-8 (one C-level call for all K
    elements) and a Taylor series for x < 1e-8.  Returns shape (K,).
    """
    from scipy.special import gammainc as _ginc, gamma as _gam
    out = np.empty_like(x)
    small = x < 1.0e-8
    # Small-x Taylor: F_n(x) ≈ 1/(2n+1) - x/(2n+3) + ...
    if np.any(small):
        xs = x[small]
        val   = np.zeros_like(xs)
        xpow  = np.ones_like(xs)
        fac   = 1.0
        denom = float(2 * n + 1)
        for k in range(20):
            term = xpow / (fac * denom)
            val  += term if k % 2 == 0 else -term
            xpow *= xs
            fac  *= (k + 1)
            denom += 2.0
            if np.max(np.abs(xpow / fac)) < 1.0e-16:
                break
        out[small] = val
    # Large-x: regularised incomplete gamma
    if np.any(~small):
        xl = x[~small]
        if n == 0:
            from scipy.special import erf as _erf
            sqrt_xl = np.sqrt(xl)
            out[~small] = (np.sqrt(math.pi) / 2.0) * _erf(sqrt_xl) / sqrt_xl
        else:
            n_half = n + 0.5
            out[~small] = (0.5 * _gam(n_half) * _ginc(n_half, xl) / xl ** n_half)
    return out


def _build_e1d_batch(
    la: int, lb: int,
    XPA: np.ndarray, XPB: np.ndarray,
    inv_2p: np.ndarray, exp_ab: np.ndarray,
) -> np.ndarray:
    r"""McMurchie-Davidson E coefficients over K primitive pairs simultaneously.

    Args:
        la, lb    : Angular quantum numbers (integer).
        XPA       : P_i − A_i for each of K pairs, shape (K,).
        XPB       : P_i − B_i, shape (K,).
        inv_2p    : 1/(2p) for each pair, shape (K,).
        exp_ab    : exp(−μ_ab (A_i−B_i)²), shape (K,).

    Returns:
        E : shape (K, la+1, lb+1, la+lb+1) — same semantics as _build_e1d but
            with a leading K-batch axis.
    """
    tmax = la + lb
    K    = len(XPA)
    E    = np.zeros((K, la + 2, lb + 2, tmax + 2))
    E[:, 0, 0, 0] = exp_ab

    # Vertical recurrence: build E[:, 1..la, 0, :]
    for ia in range(la):
        for t in range(ia + 2):
            v = XPA * (E[:, ia, 0, t] if t <= ia else 0.0)
            if t > 0:
                v = v + inv_2p * (E[:, ia, 0, t - 1] if t - 1 <= ia else 0.0)
            v = v + (t + 1) * (E[:, ia, 0, t + 1] if t + 1 <= ia else 0.0)
            E[:, ia + 1, 0, t] = v

    # Horizontal recurrence: build E[:, *, 1..lb, :]
    for ib in range(lb):
        for ia in range(la + 1):
            mt = ia + ib
            for t in range(mt + 2):
                v = XPB * (E[:, ia, ib, t] if t <= mt else 0.0)
                if t > 0:
                    v = v + inv_2p * (E[:, ia, ib, t - 1] if t - 1 <= mt else 0.0)
                v = v + (t + 1) * (E[:, ia, ib, t + 1] if t + 1 <= mt else 0.0)
                E[:, ia, ib + 1, t] = v

    return E[:, :la + 1, :lb + 1, :tmax + 1]


def _build_R_batch_k(
    rho: np.ndarray,
    PCx: np.ndarray, PCy: np.ndarray, PCz: np.ndarray,
    t_max: int, u_max: int, v_max: int,
) -> np.ndarray:
    r"""Hermite Coulomb integrals R[t,u,v,n] over K primitive quartets.

    Args:
        rho         : shape (K,) — reduced exponent ρ = pq/(p+q).
        PCx/y/z     : shape (K,) — P_xyz − Q_xyz (P is bra weighted centre,
                      Q is ket weighted centre, so PC = P − C = P − Q here).
        t_max, u_max, v_max : maximum t, u, v indices needed.

    Returns:
        R : shape (K, t_max+1, u_max+1, v_max+1, n_max+1).
    """
    n_max = t_max + u_max + v_max
    K     = len(rho)
    R     = np.zeros((K, t_max + 1, u_max + 1, v_max + 1, n_max + 1))
    twop  = -2.0 * rho                               # (K,)

    eta   = rho * (PCx * PCx + PCy * PCy + PCz * PCz)   # (K,)

    # Base: R[:,0,0,0,n] = (-2ρ)^n · F_n(η)
    twop_n = np.ones(K)
    for n in range(n_max + 1):
        R[:, 0, 0, 0, n] = twop_n * _boys_fn_arr(n, eta)
        twop_n = twop_n * twop

    # t-direction (vectorised over n)
    for t in range(t_max):
        nt = n_max - t - 1
        if nt <= 0:
            break
        R[:, t + 1, 0, 0, :nt] = PCx[:, None] * R[:, t, 0, 0, 1:nt + 1]
        if t > 0:
            R[:, t + 1, 0, 0, :nt] += t * R[:, t - 1, 0, 0, 1:nt + 1]

    # u-direction
    for u in range(u_max):
        for t in range(t_max + 1):
            nu = n_max - t - u - 1
            if nu <= 0:
                continue
            R[:, t, u + 1, 0, :nu] = PCy[:, None] * R[:, t, u, 0, 1:nu + 1]
            if u > 0:
                R[:, t, u + 1, 0, :nu] += u * R[:, t, u - 1, 0, 1:nu + 1]

    # v-direction
    for v in range(v_max):
        for t in range(t_max + 1):
            for u in range(u_max + 1):
                nv = n_max - t - u - v - 1
                if nv <= 0:
                    continue
                R[:, t, u, v + 1, :nv] = PCz[:, None] * R[:, t, u, v, 1:nv + 1]
                if v > 0:
                    R[:, t, u, v + 1, :nv] += v * R[:, t, u, v - 1, 1:nv + 1]

    return R


def _eri_prim_cart_batch(
    la: Tuple[int, int, int], lb: Tuple[int, int, int],
    lc: Tuple[int, int, int], ld: Tuple[int, int, int],
    alpha: np.ndarray, A: np.ndarray,
    beta:  np.ndarray, B: np.ndarray,
    gamma: np.ndarray, C: np.ndarray,
    delta: np.ndarray, D: np.ndarray,
) -> np.ndarray:
    r"""Normalised primitive ERI (ab|cd) for K quartets with same angular types.

    All angular-type arguments are (lx, ly, lz) tuples (fixed for the batch).
    Exponent/center arrays have shape (K,) and (K, 3) respectively.

    Returns shape (K,) float64.
    """
    K = len(alpha)

    p   = alpha + beta                              # (K,)
    q   = gamma + delta                             # (K,)
    pq  = p + q
    rho = p * q / pq                                # (K,)

    inv_2p = 1.0 / (2.0 * p)
    inv_2q = 1.0 / (2.0 * q)
    mu_ab  = alpha * beta / p
    mu_cd  = gamma * delta / q

    # Weighted centres P, Q
    P = (alpha[:, None] * A + beta[:, None]  * B) / p[:, None]   # (K, 3)
    Q = (gamma[:, None] * C + delta[:, None] * D) / q[:, None]   # (K, 3)

    # Build E1d for bra (3 axes) — shape (K, la_d+1, lb_d+1, tmax_d+1)
    Eab = []
    for dim in range(3):
        PA  = P[:, dim] - A[:, dim]
        PB  = P[:, dim] - B[:, dim]
        AB2 = (A[:, dim] - B[:, dim]) ** 2
        exp_ab = np.exp(-mu_ab * AB2)
        Eab.append(_build_e1d_batch(la[dim], lb[dim], PA, PB, inv_2p, exp_ab))

    # Build E1d for ket (3 axes)
    Ecd = []
    for dim in range(3):
        QC  = Q[:, dim] - C[:, dim]
        QD  = Q[:, dim] - D[:, dim]
        CD2 = (C[:, dim] - D[:, dim]) ** 2
        exp_cd = np.exp(-mu_cd * CD2)
        Ecd.append(_build_e1d_batch(lc[dim], ld[dim], QC, QD, inv_2q, exp_cd))

    t_max  = la[0] + lb[0]; u_max  = la[1] + lb[1]; v_max  = la[2] + lb[2]
    tp_max = lc[0] + ld[0]; up_max = lc[1] + ld[1]; vp_max = lc[2] + ld[2]

    # Build R_batch: P − Q is the P→C vector (Q is the "C" centre here)
    PCx = P[:, 0] - Q[:, 0]
    PCy = P[:, 1] - Q[:, 1]
    PCz = P[:, 2] - Q[:, 2]
    R = _build_R_batch_k(rho, PCx, PCy, PCz,
                         t_max + tp_max, u_max + up_max, v_max + vp_max)
    # R shape: (K, T+Tp+1, U+Up+1, V+Vp+1, n_max+1)

    # Extract E slices and apply (−1)^{t'+u'+v'} sign to ket
    T  = t_max + 1;  U  = u_max + 1;  V  = v_max + 1
    Tp = tp_max + 1; Up = up_max + 1; Vp = vp_max + 1

    ex_ab = Eab[0][:, la[0], lb[0], :T]   # (K, T)
    ey_ab = Eab[1][:, la[1], lb[1], :U]   # (K, U)
    ez_ab = Eab[2][:, la[2], lb[2], :V]   # (K, V)

    ex_cd = Ecd[0][:, lc[0], ld[0], :Tp].copy()   # (K, Tp)
    ey_cd = Ecd[1][:, lc[1], ld[1], :Up].copy()   # (K, Up)
    ez_cd = Ecd[2][:, lc[2], ld[2], :Vp].copy()   # (K, Vp)
    ex_cd[:, 1::2] *= -1.0
    ey_cd[:, 1::2] *= -1.0
    ez_cd[:, 1::2] *= -1.0

    # Outer products: A_b[k,t,u,v] and B_b[k,tp,up,vp]
    A_b = (ex_ab[:, :, None, None]
           * ey_ab[:, None, :, None]
           * ez_ab[:, None, None, :])          # (K, T, U, V)
    B_b = (ex_cd[:, :, None, None]
           * ey_cd[:, None, :, None]
           * ez_cd[:, None, None, :])          # (K, Tp, Up, Vp)

    # R0 fancy-indexed to (K, T, Tp, U, Up, V, Vp)
    t_idx = (np.arange(T)[:, None] + np.arange(Tp)[None, :])     # (T, Tp)
    u_idx = (np.arange(U)[:, None] + np.arange(Up)[None, :])     # (U, Up)
    v_idx = (np.arange(V)[:, None] + np.arange(Vp)[None, :])     # (V, Vp)

    R0 = R[:, t_idx[:, :, None, None, None, None],
              u_idx[None, None, :, :, None, None],
              v_idx[None, None, None, None, :, :],
              0]                                                   # (K,T,Tp,U,Up,V,Vp)

    AB = (A_b[:, :, None, :, None, :, None]
          * B_b[:, None, :, None, :, None, :])                    # (K,T,Tp,U,Up,V,Vp)

    val = np.sum(AB * R0, axis=(1, 2, 3, 4, 5, 6))                # (K,)

    # Normalisation factors
    def _nc(exp_arr: np.ndarray, l: Tuple[int, int, int]) -> np.ndarray:
        """Vectorised _norm_cart over K exponents."""
        def _s1d(n: int) -> np.ndarray:
            return _dbl_fact(2 * n - 1) / (4.0 * exp_arr) ** n * np.sqrt(
                math.pi / (2.0 * exp_arr))
        return 1.0 / np.sqrt(_s1d(l[0]) * _s1d(l[1]) * _s1d(l[2]))

    N_ab = _nc(alpha, la) * _nc(beta, lb) * _nc(gamma, lc) * _nc(delta, ld)  # (K,)

    prefac = 2.0 * math.pi ** 2.5 / (p * q * np.sqrt(pq))         # (K,)
    return N_ab * prefac * val


def _build_eri_batch_by_type(
    shells: List[Tuple[np.ndarray, Tuple[int, int, int], List[float], List[float]]],
    K_batch: int = 10_000,
) -> np.ndarray:
    """Vectorised AO ERI tensor using angular-type batching.

    Groups unique shell quartets by (la, lb, lc, ld) type.  Within each group
    ALL K shell quartets are expanded over their primitive contractions and
    processed simultaneously via :func:`_eri_prim_cart_batch`.  K_batch limits
    the number of primitive quartets per numpy call to bound peak memory.

    Replaces the nested Python loop in :func:`_build_ao_integrals_cart` and
    gives ~100–1000× wall-time improvement for contracted d-shells on M1.

    Returns:
        g : (N_sh, N_sh, N_sh, N_sh) float64 AO ERI tensor.
    """
    from collections import defaultdict

    N_sh  = len(shells)
    norms = [_contracted_norm_cart(s[2], s[3], *s[1]) for s in shells]
    g     = np.zeros((N_sh, N_sh, N_sh, N_sh))

    # Enumerate unique quartets: mu>=nu, lam>=sig, compound(mu,nu)>=compound(lam,sig)
    # Group by angular-momentum type tuple (la, lb, lc, ld)
    type_to_quartets: dict = defaultdict(list)
    for mu in range(N_sh):
        _, la, _, _ = shells[mu]
        for nu in range(mu + 1):
            _, lb, _, _ = shells[nu]
            mn = mu * (mu + 1) // 2 + nu
            for lam in range(N_sh):
                _, lc, _, _ = shells[lam]
                for sig in range(lam + 1):
                    ls = lam * (lam + 1) // 2 + sig
                    if mn < ls:
                        continue
                    _, ld, _, _ = shells[sig]
                    type_to_quartets[(la, lb, lc, ld)].append((mu, nu, lam, sig))

    for (la, lb, lc, ld), qlist in type_to_quartets.items():
        # Expand shell quartets → primitive quartets
        prim_recs: List[Tuple[int, int, int, int, float, float, float, float,
                               float, float, float, float, float, float, float,
                               float, float, float, float, float]] = []
        # Each record: (mu, nu, lam, sig, alpha, Ax,Ay,Az, beta, Bx,By,Bz,
        #               gamma, Cx,Cy,Cz, delta, Dx,Dy,Dz, pref)
        for mu, nu, lam, sig in qlist:
            A_c, _, amu, dmu = shells[mu]
            B_c, _, anu, dnu = shells[nu]
            C_c, _, alm, dlm = shells[lam]
            D_c, _, asg, dsg = shells[sig]
            n_mu = norms[mu]; n_nu = norms[nu]
            n_lm = norms[lam]; n_sg = norms[sig]
            norm_prod = n_mu * n_nu * n_lm * n_sg
            for a, da in zip(amu, dmu):
                for b, db in zip(anu, dnu):
                    for c, dc in zip(alm, dlm):
                        for d_, dd_ in zip(asg, dsg):
                            w = da * db * dc * dd_ / norm_prod
                            prim_recs.append((
                                mu, nu, lam, sig,
                                a,  float(A_c[0]), float(A_c[1]), float(A_c[2]),
                                b,  float(B_c[0]), float(B_c[1]), float(B_c[2]),
                                c,  float(C_c[0]), float(C_c[1]), float(C_c[2]),
                                d_, float(D_c[0]), float(D_c[1]), float(D_c[2]),
                                w,
                            ))

        if not prim_recs:
            continue

        # Process in K_batch chunks to limit peak memory
        for start in range(0, len(prim_recs), K_batch):
            chunk = prim_recs[start:start + K_batch]
            K = len(chunk)

            arr = np.array(chunk, dtype=np.float64)  # (K, 21)
            mu_idx   = arr[:, 0].astype(np.intp)
            nu_idx   = arr[:, 1].astype(np.intp)
            lam_idx  = arr[:, 2].astype(np.intp)
            sig_idx  = arr[:, 3].astype(np.intp)
            alpha_k  = arr[:, 4]
            A_k      = arr[:, 5:8]
            beta_k   = arr[:, 8]
            B_k      = arr[:, 9:12]
            gamma_k  = arr[:, 12]
            C_k      = arr[:, 13:16]
            delta_k  = arr[:, 16]
            D_k      = arr[:, 17:20]
            w_k      = arr[:, 20]

            vals = _eri_prim_cart_batch(
                la, lb, lc, ld,
                alpha_k, A_k, beta_k, B_k,
                gamma_k, C_k, delta_k, D_k,
            )
            contrib = vals * w_k          # (K,)

            # Scatter-add to shell quartet indices and fill 8 symmetry copies
            np.add.at(g,
                (mu_idx,  nu_idx,  lam_idx, sig_idx), contrib)
            np.add.at(g,
                (nu_idx,  mu_idx,  lam_idx, sig_idx), contrib)
            np.add.at(g,
                (mu_idx,  nu_idx,  sig_idx, lam_idx), contrib)
            np.add.at(g,
                (nu_idx,  mu_idx,  sig_idx, lam_idx), contrib)
            np.add.at(g,
                (lam_idx, sig_idx, mu_idx,  nu_idx),  contrib)
            np.add.at(g,
                (sig_idx, lam_idx, mu_idx,  nu_idx),  contrib)
            np.add.at(g,
                (lam_idx, sig_idx, nu_idx,  mu_idx),  contrib)
            np.add.at(g,
                (sig_idx, lam_idx, nu_idx,  mu_idx),  contrib)

    return g


# ---------------------------------------------------------------------------
# Helpers for 8-fold packed g_MO storage (Fix B: sec:hamiltonian_from_zeros)
# ---------------------------------------------------------------------------

def _pack_g_mo(g_MO: np.ndarray) -> Tuple[np.ndarray, int]:
    """Pack (N,N,N,N) ERI tensor using 8-fold symmetry into 1-D float64 array.

    Returns:
        packed : 1-D array of length Np*(Np+1)/2, Np = N*(N+1)/2.
        N      : orbital dimension (needed for unpacking).
    """
    N    = g_MO.shape[0]
    tril = np.tril_indices(N)        # (p,q) pairs with p >= q, length Np
    Np   = len(tril[0])
    # Build Np × Np matrix of unique symmetry elements: g2[i,j] = g[p_i,q_i,r_j,s_j]
    g2   = g_MO[tril[0][:, None], tril[1][:, None],
                tril[0][None, :], tril[1][None, :]]   # (Np, Np)
    tril2 = np.tril_indices(Np)
    return g2[tril2[0], tril2[1]], N


def _unpack_g_mo(packed: np.ndarray, N: int) -> np.ndarray:
    """Reconstruct (N,N,N,N) ERI tensor from 1-D packed form.

    Inverse of :func:`_pack_g_mo`.  Uses vectorised fancy indexing — no
    Python loops.
    """
    tril  = np.tril_indices(N)
    Np    = len(tril[0])
    # Build compound-index map M: (N,N) → int in 0..Np-1
    M     = np.zeros((N, N), dtype=np.intp)
    M[tril[0], tril[1]] = np.arange(Np)
    M[tril[1], tril[0]] = np.arange(Np)   # make symmetric
    # Reconstruct symmetric Np×Np matrix
    tril2 = np.tril_indices(Np)
    g2    = np.zeros((Np, Np))
    g2[tril2[0], tril2[1]] = packed
    g2[tril2[1], tril2[0]] = packed       # symmetrize (diagonal written twice, fine)
    # Expand to (N,N,N,N) via compound-index lookup
    return g2[M[:, :, None, None], M[None, None, :, :]]


def _contracted_norm_cart(
    alphas: List[float], coeffs: List[float], lx: int, ly: int, lz: int,
) -> float:
    """sqrt(⟨χ|χ⟩) for a contracted Cartesian GTO (all primitives have the same angular type)."""
    la = (lx, ly, lz)
    O  = np.zeros(3)  # placed at origin for self-overlap
    s  = 0.0
    for a, da in zip(alphas, coeffs):
        for b, db in zip(alphas, coeffs):
            s += da * db * _overlap_prim_cart(a, O, la, b, O, la)
    return math.sqrt(max(s, 0.0))


def _build_1e_contracted_cart(
    alphas_mu: List[float], coeffs_mu: List[float],
    A: np.ndarray, la: Tuple[int, int, int],
    alphas_nu: List[float], coeffs_nu: List[float],
    B: np.ndarray, lb: Tuple[int, int, int],
    nuclear_charges: List[Tuple[float, np.ndarray]],
) -> Tuple[float, float, float]:
    """Contracted S, T, V integrals for a pair of Cartesian CGTOs (l ≤ 2)."""
    S_val = T_val = V_val = 0.0
    for a, da in zip(alphas_mu, coeffs_mu):
        for b, db in zip(alphas_nu, coeffs_nu):
            pref   = da * db
            S_val += pref * _overlap_prim_cart(a, A, la, b, B, lb)
            T_val += pref * _kinetic_prim_cart(a, A, la, b, B, lb)
            for Z, R_nuc in nuclear_charges:
                V_val += pref * _nuclear_prim_cart(a, A, la, b, B, lb, Z, R_nuc)
    return S_val, T_val, V_val


def _eri_contracted_cart(
    alphas_mu: List[float], coeffs_mu: List[float],
    A: np.ndarray, la: Tuple[int, int, int],
    alphas_nu: List[float], coeffs_nu: List[float],
    B: np.ndarray, lb: Tuple[int, int, int],
    alphas_lm: List[float], coeffs_lm: List[float],
    C: np.ndarray, lc: Tuple[int, int, int],
    alphas_sg: List[float], coeffs_sg: List[float],
    D: np.ndarray, ld: Tuple[int, int, int],
) -> float:
    """Contracted (μν|λσ) ERI for four Cartesian CGTOs (l ≤ 2)."""
    val = 0.0
    for a, da in zip(alphas_mu, coeffs_mu):
        for b, db in zip(alphas_nu, coeffs_nu):
            for c, dc in zip(alphas_lm, coeffs_lm):
                for d_, dd_ in zip(alphas_sg, coeffs_sg):
                    val += da * db * dc * dd_ * _eri_prim_cart(
                        a, A, la, b, B, lb, c, C, lc, d_, D, ld)
    return val


#: Shell-type string → angular momentum l (for the full-shell builder).
_TYPE_L: Dict[str, int] = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4}


def _build_basis_shells(
    atoms:         List[Tuple[str, float, float, float]],
    d_single_zeta: bool                       = True,
    basis_spec:    Optional[Dict[str, str]]   = None,
    full_shells:   bool                       = False,
) -> List[Tuple[np.ndarray, Tuple[int, int, int], List[float], List[float]]]:
    """Build the AO shell list for a mixed s/p/d(/f/g) basis.

    Shells are sourced from two places depending on *basis_spec*:

    * **Default (STO-3G)**: shells read from :mod:`mqebasis` (authoritative
      EMSL ``sto-3g.dat`` parse) — same as the previous behaviour.
    * **Per-element BSE override**: if *basis_spec* maps the element symbol to
      a basis name (e.g. ``{"Fe": "def2-TZVP"}``), shells are loaded from the
      local Basis Set Exchange mirror via :mod:`mqebasisloader`.  Elements not
      listed in *basis_spec* continue to use STO-3G.

    Two shell-selection regimes:

    * ``full_shells=True`` — the **untruncated full basis**: every contracted
      shell of every angular momentum (s, p, d, f, g) is emitted as Cartesian
      shell groups.  This is the genuine def2-TZVP/cc-pVTZ basis (no truncation),
      required for climbing to the real full active space.  (Cartesian d/f/g
      include lower-l contaminants; spherical-harmonic contraction — removing
      them — is a separate refinement.)
    * ``full_shells=False`` (default, legacy) — the truncated path used by the
      validated runs:

      - TM with a D shell: six Cartesian d-shells (single-zeta when
        ``d_single_zeta=True``);
      - S / SP-type: outermost s + three p;
      - other: outermost s.

    Returns:
        List of (center_bohr, (lx, ly, lz), alphas, coeffs).
    """
    shells = []
    for sym, x, y, z in atoms:
        center = np.array([x, y, z]) * _BOHR_PER_ANG

        # ── Resolve shell source ──────────────────────────────────────────
        bse_override = (basis_spec or {}).get(sym)
        if bse_override is not None:
            # Load from BSE mirror; _mbl.load_bse_basis uses variable-length
            # tuples so list() on each works identically to the STO-3G path.
            try:
                basis_shells = _mbl.load_bse_basis(bse_override, sym)
            except (FileNotFoundError, KeyError) as exc:
                raise ValueError(
                    f"[zetazero] Cannot load BSE basis '{bse_override}' for "
                    f"element '{sym}': {exc}"
                ) from exc
            # The d-only branch is reserved for transition metals — elements
            # whose primary valence is d-type.  _HAS_D_BASIS encodes exactly
            # this set (same as STO-3G's HAS_D: Fe, Mo, Mn, V, Cu, Ti, …).
            # Do NOT use "any D shell in the BSE basis" here — main-group
            # elements such as S have D polarisation shells in def2-TZVP but
            # still need their s/p valence as the primary basis.
            has_d = sym in _HAS_D_BASIS
            # For the p-shell decision: check both the static _HAS_P_BASIS
            # (STO-3G sulfur) and whether the BSE shells contain SP or P.
            has_p = sym in _HAS_P_BASIS or any(
                s[0] in ("P", "SP") for s in basis_shells
            )
        else:
            basis_shells = _mb.get_shells(sym)
            has_d = sym in _HAS_D_BASIS
            has_p = sym in _HAS_P_BASIS

        # ── Full untruncated basis: emit every shell at full angular momentum ─
        if full_shells:
            for sh in basis_shells:
                stype  = sh[0]
                alphas = list(sh[1])
                if stype == "SP":
                    shells.append((center, (0, 0, 0), alphas, list(sh[2])))
                    if len(sh) > 3 and sh[3] is not None:
                        for ang in _CART_COMPS[1]:
                            shells.append((center, ang, alphas, list(sh[3])))
                    continue
                l = _TYPE_L.get(stype)
                if l is None or l not in _CART_COMPS:
                    raise ValueError(
                        f"[zetazero] full-basis: unsupported shell type "
                        f"'{stype}' for element '{sym}' (l>4 not implemented)."
                    )
                coeffs = list(sh[2])
                for ang in _CART_COMPS[l]:
                    shells.append((center, ang, alphas, coeffs))
            continue

        # ── Build Cartesian shells (legacy truncated path) ────────────────
        if has_d:
            all_d = [s for s in basis_shells if s[0] == "D"]
            if d_single_zeta:
                # Most diffuse single primitive: outermost D shell (last in
                # list), last exponent.  Correct angular character, single
                # GTO keeps ERI cost O(N⁴).  Works for both STO-3G (1 D
                # shell, 3 exps → use exp[-1]) and multi-contracted bases
                # (last D is the most diffuse uncontracted function).
                outermost = all_d[-1]
                alphas = [list(outermost[1])[-1]]
                coeffs = [1.0]
                for ang in _CART_COMPS[2]:
                    shells.append((center, ang, alphas, coeffs))
            else:
                # Full contracted basis: one group of 6 Cartesian d-shells
                # per D contraction.  For STO-3G this is one group; for
                # multi-contracted bases (e.g. def2-TZVP) each contracted D
                # function adds another 6 shells.
                for d_entry in all_d:
                    alphas = list(d_entry[1])
                    coeffs = list(d_entry[2])
                    for ang in _CART_COMPS[2]:
                        shells.append((center, ang, alphas, coeffs))

        else:
            outer = next(
                (s for s in reversed(basis_shells) if s[0] in ("S", "SP")), None
            )
            if outer is not None:
                alphas_s = list(outer[1])
                coeffs_s = list(outer[2])
                shells.append((center, (0, 0, 0), alphas_s, coeffs_s))
                if has_p and outer[0] == "SP" and outer[3] is not None:
                    # STO-3G / SP-type: s and p share exponents.
                    coeffs_p = list(outer[3])
                    for ang in _CART_COMPS[1]:
                        shells.append((center, ang, alphas_s, coeffs_p))
                elif has_p and outer[0] != "SP":
                    # Separate P shells (e.g. def2-TZVP): outermost P
                    # contraction → three Cartesian p-shells.
                    outer_p = next(
                        (s for s in reversed(basis_shells) if s[0] == "P"), None
                    )
                    if outer_p is not None:
                        alphas_p = list(outer_p[1])
                        coeffs_p = list(outer_p[2])
                        for ang in _CART_COMPS[1]:
                            shells.append((center, ang, alphas_p, coeffs_p))

    return shells


# ===========================================================================
# SECTION 1D — CARTESIAN → SPHERICAL (PURE) HARMONIC CONTRACTION
# ===========================================================================
# A Cartesian GTO shell of angular momentum l has (l+1)(l+2)/2 components, but
# only 2l+1 are pure-l spherical harmonics; the remainder are lower-l
# "contaminants" (d: 6→5 drops one s; f: 10→7 drops three p; g: 15→9 drops
# 5 d + 1 s).  The contaminant-free subspace is exactly the space of *harmonic*
# degree-l polynomials = kernel of the polynomial Laplacian.  We build it by
# null-space (no spherical-harmonic coefficient tables to transcribe) and
# orthonormalise in the actual GTO overlap metric, which is self-verifying:
# the resulting per-shell block satisfies Tᵀ S₀ T = I exactly.  Any orthonormal
# basis of the harmonic subspace is physically equivalent (energies are
# invariant to in-shell rotations), so canonical m-ordering is unnecessary.

_HARMONIC_CACHE: Dict[int, np.ndarray] = {}


def _harmonic_null(l: int) -> np.ndarray:
    """Harmonic degree-l polynomials (columns) in the _CART_COMPS[l] monomial
    basis.  dim = 2l+1.  l<2 → identity (no contaminant)."""
    if l in _HARMONIC_CACHE:
        return _HARMONIC_CACHE[l]
    comps = _CART_COMPS[l]
    if l < 2:
        out = np.eye(len(comps))
        _HARMONIC_CACHE[l] = out
        return out
    lower = _CART_COMPS[l - 2]
    lidx  = {c: i for i, c in enumerate(lower)}
    Lap   = np.zeros((len(lower), len(comps)))
    for j, (lx, ly, lz) in enumerate(comps):
        if lx >= 2: Lap[lidx[(lx - 2, ly, lz)], j] += lx * (lx - 1)
        if ly >= 2: Lap[lidx[(lx, ly - 2, lz)], j] += ly * (ly - 1)
        if lz >= 2: Lap[lidx[(lx, ly, lz - 2)], j] += lz * (lz - 1)
    # Right null space of Lap (degree-l polys annihilated by the Laplacian).
    _, s, vt = np.linalg.svd(Lap)
    rank = int(np.sum(s > 1e-9))
    out  = vt[rank:].T                      # (n_cart, 2l+1)
    _HARMONIC_CACHE[l] = out
    return out


def _cart_self_overlap(alphas, coeffs, comps) -> np.ndarray:
    """Normalised-Cartesian self-overlap block for a contracted shell at origin."""
    O = np.zeros(3)
    norms = [_contracted_norm_cart(alphas, coeffs, *c) for c in comps]
    n = len(comps)
    S0 = np.zeros((n, n))
    for i, ci in enumerate(comps):
        for j, cj in enumerate(comps):
            v = 0.0
            for a, da in zip(alphas, coeffs):
                for b, db in zip(alphas, coeffs):
                    v += da * db * _overlap_prim_cart(a, O, ci, b, O, cj)
            S0[i, j] = v / (norms[i] * norms[j])
    return S0


def _shell_sph_T(l: int, alphas, coeffs) -> np.ndarray:
    """(n_cart, 2l+1) transform: normalised-Cartesian → orthonormal pure-l
    spherical functions for one contracted shell.  Satisfies Tᵀ S₀ T = I."""
    comps = _CART_COMPS[l]
    if l < 2:
        return np.eye(len(comps))
    H     = _harmonic_null(l)                              # (n_cart, 2l+1)
    norms = np.array([_contracted_norm_cart(alphas, coeffs, *c) for c in comps])
    W     = H / norms[:, None]                             # monomial→norm-cart
    S0    = _cart_self_overlap(alphas, coeffs, comps)
    G     = W.T @ S0 @ W
    w, U  = np.linalg.eigh(G)                              # Löwdin orthonormalise
    return W @ (U @ np.diag(1.0 / np.sqrt(w)) @ U.T)


def cart_to_sph_transform(
    shells: List[Tuple[np.ndarray, Tuple[int, int, int], List[float], List[float]]],
) -> np.ndarray:
    """Block-diagonal Cartesian→spherical transform C (N_cart × N_sph).

    Groups the per-component shell list back into contracted shells (consecutive
    entries sharing centre/alphas/coeffs and enumerating _CART_COMPS[l]) and
    stacks each shell's :func:`_shell_sph_T`.  Apply as ``S_sph = Cᵀ S_cart C``,
    ``g_sph = einsum('pi,qj,pqrs,rk,sl->ijkl', C, C, g, C, C)``.
    """
    n_cart = len(shells)
    blocks: List[Tuple[int, np.ndarray]] = []   # (start_row, T)
    i = 0
    while i < n_cart:
        center, ang0, alphas, coeffs = shells[i]
        l = sum(ang0)
        ncl = len(_CART_COMPS[l])
        # the builder emits the ncl components of this shell consecutively
        T = _shell_sph_T(l, alphas, coeffs)
        blocks.append((i, T))
        i += ncl
    n_sph = sum(T.shape[1] for _, T in blocks)
    C = np.zeros((n_cart, n_sph))
    col = 0
    for start, T in blocks:
        nc, ns = T.shape
        C[start:start + nc, col:col + ns] = T
        col += ns
    return C


# ===========================================================================
# SECTION 2 — ONE-ELECTRON AO INTEGRALS
# ===========================================================================

def _overlap_prim(a: float, A: np.ndarray, b: float, B: np.ndarray) -> float:
    """⟨s_A|s_B⟩ for two primitive s-type GTOs centred at A, B."""
    AB2 = float(np.dot(A - B, A - B))
    return (
        _norm_s(a) * _norm_s(b)
        * (math.pi / (a + b)) ** 1.5
        * math.exp(-a * b / (a + b) * AB2)
    )


def _kinetic_prim(a: float, A: np.ndarray, b: float, B: np.ndarray) -> float:
    """⟨s_A|T|s_B⟩ kinetic energy integral for primitive s-type GTOs.

    T_AB = (a·b/(a+b)) [3 − 2ab/(a+b)·|A−B|²] · S_AB
    """
    AB2  = float(np.dot(A - B, A - B))
    p    = a + b
    q    = a * b / p
    S    = _overlap_prim(a, A, b, B)
    return q * (3.0 - 2.0 * q * AB2) * S


def _nuclear_prim(
    a: float, A: np.ndarray,
    b: float, B: np.ndarray,
    Z_alpha: float, R_alpha: np.ndarray,
) -> float:
    """⟨s_A|−Z_α/|r−R_α||s_B⟩ nuclear attraction for one nucleus α.

    V_AB = −Z_α · N_a N_b · (2π/(a+b)) · exp(−Q_AB) · F_0(p·|P−R|²)
    where P = (a·A + b·B)/(a+b), Q_AB = ab/(a+b)·|A−B|², p = a+b.
    """
    p    = a + b
    P    = (a * A + b * B) / p
    AB2  = float(np.dot(A - B, A - B))
    Q_AB = a * b / p * AB2
    PR2  = float(np.dot(P - R_alpha, P - R_alpha))
    return (
        -Z_alpha
        * _norm_s(a) * _norm_s(b)
        * (2.0 * math.pi / p)
        * math.exp(-Q_AB)
        * _boys_f0(p * PR2)
    )


def _build_1e_contracted(
    alphas_mu: List[float], coeffs_mu: List[float], A: np.ndarray,
    alphas_nu: List[float], coeffs_nu: List[float], B: np.ndarray,
    nuclear_charges: List[Tuple[float, np.ndarray]],
) -> Tuple[float, float, float]:
    """Contracted S, T, V integrals for a pair of s-type CGTOs.

    Returns (S_μν, T_μν, V_μν) where V includes all nuclei.
    """
    S_val = T_val = V_val = 0.0
    for a, d_a in zip(alphas_mu, coeffs_mu):
        for b, d_b in zip(alphas_nu, coeffs_nu):
            pref = d_a * d_b
            S_val += pref * _overlap_prim(a, A, b, B)
            T_val += pref * _kinetic_prim(a, A, b, B)
            for Z, R in nuclear_charges:
                V_val += pref * _nuclear_prim(a, A, b, B, Z, R)
    return S_val, T_val, V_val


def _eri_prim(
    a: float, A: np.ndarray,
    b: float, B: np.ndarray,
    c: float, C: np.ndarray,
    d: float, D: np.ndarray,
) -> float:
    r"""(ss|ss) ERI for four primitive s-type GTOs.

    (AB|CD) = N_a N_b N_c N_d · (2π^{5/2}/(pq√(p+q)))
              · exp(−Q_AB − Q_CD) · F_0(η|P−Q|²)

    where p=a+b, q=c+d, η=pq/(p+q), P=(aA+bB)/p, Q=(cC+dD)/q.
    """
    p   = a + b
    q   = c + d
    P   = (a * A + b * B) / p
    Q   = (c * C + d * D) / q
    AB2 = float(np.dot(A - B, A - B))
    CD2 = float(np.dot(C - D, C - D))
    PQ2 = float(np.dot(P - Q, P - Q))
    Q_AB = a * b / p * AB2
    Q_CD = c * d / q * CD2
    eta  = p * q / (p + q)
    pref = (
        _norm_s(a) * _norm_s(b) * _norm_s(c) * _norm_s(d)
        * 2.0 * math.pi ** 2.5
        / (p * q * math.sqrt(p + q))
        * math.exp(-Q_AB - Q_CD)
    )
    return pref * _boys_f0(eta * PQ2)


def _eri_contracted(
    alphas_mu:    List[float], coeffs_mu:    List[float], A: np.ndarray,
    alphas_nu:    List[float], coeffs_nu:    List[float], B: np.ndarray,
    alphas_lam:   List[float], coeffs_lam:   List[float], C: np.ndarray,
    alphas_sig:   List[float], coeffs_sig:   List[float], D: np.ndarray,
) -> float:
    """Contracted (μν|λσ) ERI for four s-type CGTOs."""
    val = 0.0
    for a, d_a in zip(alphas_mu, coeffs_mu):
        for b, d_b in zip(alphas_nu, coeffs_nu):
            for c, d_c in zip(alphas_lam, coeffs_lam):
                for d_, d_d in zip(alphas_sig, coeffs_sig):
                    val += d_a * d_b * d_c * d_d * _eri_prim(
                        a, A, b, B, c, C, d_, D
                    )
    return val


# ===========================================================================
# SECTION 3 — SCHWARZ SCREENING AND INTEGRAL-DIRECT AO→MO TRANSFORM (STEP 5)
# ===========================================================================
# These functions implement the screened integral-direct AO→MO transform
# described in sec:full_hamiltonian Step 5.  They avoid materialising the
# O(N_AO^4) AO ERI tensor g_AO by:
#   1. Pre-computing Schwarz bounds K[P,Q] = sqrt(|(PQ|PQ)|).
#   2. Skipping shell quartets where K[P,Q]·K[R,S] < threshold.
#   3. Projecting significant ERIs directly into a half-transformed
#      intermediate H[m,n,r,s] and then contracting with C to get g_MO.
# Memory: O(N_sh^2·N^2) for H vs O(N_sh^4) for g_AO.


def schwarz_bounds(
    shells: List[Tuple[np.ndarray, Tuple[int, int, int], List[float], List[float]]],
    norms:  List[float],
) -> np.ndarray:
    """Schwarz upper bounds K[P,Q] = sqrt(|(PQ|PQ)|) for all shell pairs.

    K[P,Q] is computed from the contracted diagonal ERI (PQ|PQ) using the
    same scalar primitive loop as :func:`_eri_contracted`.  Cost O(N_sh²·K⁴)
    where K is the maximum contraction length.

    Args:
        shells : Shell list from ``_build_basis_shells``.
        norms  : Contracted norms, one per shell.

    Returns:
        K : (N_sh, N_sh) float64 array of Schwarz bounds.
    """
    N_sh = len(shells)
    K_sw = np.zeros((N_sh, N_sh))

    for P in range(N_sh):
        A_c, la, alf_P, cof_P = shells[P]
        n_P = norms[P]
        for Q in range(P + 1):
            B_c, lb, alf_Q, cof_Q = shells[Q]
            n_Q = norms[Q]
            norm_diag = n_P * n_Q * n_P * n_Q  # (PQ|PQ) uses norms P,Q,P,Q
            val = 0.0
            for a, da in zip(alf_P, cof_P):
                for b, db in zip(alf_Q, cof_Q):
                    for c, dc in zip(alf_P, cof_P):
                        for d, dd in zip(alf_Q, cof_Q):
                            val += da * db * dc * dd * _eri_prim_cart(
                                a, A_c, la,
                                b, B_c, lb,
                                c, A_c, la,
                                d, B_c, lb,
                            )
            K_sw[P, Q] = K_sw[Q, P] = math.sqrt(abs(val / norm_diag))

    return K_sw


def screened_direct_ao_to_mo(
    shells:    List[Tuple[np.ndarray, Tuple[int, int, int], List[float], List[float]]],
    norms:     List[float],
    C:         np.ndarray,
    threshold: float = 1.0e-10,
    K_batch:   int   = 10_000,
) -> np.ndarray:
    """Integral-direct Schwarz-screened AO→MO 4-index transform.

    Computes g_MO = Σ_{μνλσ} C[μ,p] C[ν,q] C[λ,r] C[σ,s] g_AO[μ,ν,λ,σ]
    without materialising the O(N_sh⁴) AO ERI tensor.

    Algorithm
    ---------
    1. Compute Schwarz bounds K[P,Q] (see :func:`schwarz_bounds`).
    2. Enumerate canonical shell quartets (μ≥ν, λ≥σ, (μν)≥(λσ)) and skip
       those where K[μ,ν]·K[λ,σ] < ``threshold``.
    3. For each batch of B significant quartets of the same angular type,
       compute their ERIs via :func:`_eri_prim_cart_batch`.
    4. Accumulate into the half-transform intermediate
       H[m,n,r,s] = Σ_{λ,σ} g_AO[m,n,λ,σ] C[λ,r] C[σ,s]
       using the same 8-fold scatter-add pattern as
       :func:`_build_eri_batch_by_type`.
    5. Contract: g_MO = einsum("mp,nq,mnrs->pqrs", C, C, H).

    Memory peak: O(N_sh²·N²) for H, O(N·N_sh·N²) for the contraction
    intermediate — far below the O(N_sh⁴) of the dense path.

    Args:
        shells    : Shell list from ``_build_basis_shells``.
        norms     : Contracted norms, one per shell.
        C         : (N_sh, N) MO coefficient matrix (or combined sph-C @ C_mo).
        threshold : Skip quartet (PQ|RS) when K[P,Q]·K[R,S] < threshold.
        K_batch   : Maximum primitive quartets per numpy call (memory limit).

    Returns:
        g_MO : (N, N, N, N) float64 MO ERI tensor [Ha], chemist notation.
    """
    from collections import defaultdict

    N_sh = len(shells)
    N    = C.shape[1]

    # ── Step 1: Schwarz bounds ──────────────────────────────────────────────
    log.info(f"[screened_direct] Computing Schwarz bounds (N_sh={N_sh})…")
    K_sw = schwarz_bounds(shells, norms)

    # ── Step 2: Enumerate canonical quartets, filter by K_PQ·K_RS ──────────
    log.info(f"[screened_direct] Building screened quartet list (threshold={threshold:.1e})…")
    type_to_quartets: dict = defaultdict(list)
    n_total = 0
    n_screened = 0

    for mu in range(N_sh):
        _, la, _, _ = shells[mu]
        for nu in range(mu + 1):
            K_mn = K_sw[mu, nu]
            if K_mn == 0.0:
                continue
            _, lb, _, _ = shells[nu]
            mn = mu * (mu + 1) // 2 + nu
            for lam in range(N_sh):
                _, lc, _, _ = shells[lam]
                for sig in range(lam + 1):
                    ls = lam * (lam + 1) // 2 + sig
                    if mn < ls:
                        continue
                    n_total += 1
                    if K_mn * K_sw[lam, sig] < threshold:
                        n_screened += 1
                        continue
                    _, ld, _, _ = shells[sig]
                    type_to_quartets[(la, lb, lc, ld)].append((mu, nu, lam, sig))

    n_sig = n_total - n_screened
    log.info(
        f"[screened_direct] {n_sig}/{n_total} quartets significant "
        f"({100.0*n_sig/max(n_total,1):.1f}%)"
    )

    # ── Step 3+4: Compute ERIs and accumulate into H ─────────────────────────
    # H[m,n,r,s] = Σ_{λσ} g_AO[m,n,λ,σ] C[λ,r] C[σ,s]
    # 8-fold symmetry: same scatter pattern as _build_eri_batch_by_type.
    H = np.zeros((N_sh, N_sh, N, N))

    for (la, lb, lc, ld), qlist in type_to_quartets.items():
        # Expand shell quartets to primitive records (identical to
        # _build_eri_batch_by_type, including the same weight formula).
        prim_recs: list = []
        for mu, nu, lam, sig in qlist:
            A_c, _, amu, dmu = shells[mu]
            B_c, _, anu, dnu = shells[nu]
            C_c, _, alm, dlm = shells[lam]
            D_c, _, asg, dsg = shells[sig]
            n_mu = norms[mu]; n_nu = norms[nu]
            n_lm = norms[lam]; n_sg = norms[sig]
            norm_prod = n_mu * n_nu * n_lm * n_sg
            for a, da in zip(amu, dmu):
                for b, db in zip(anu, dnu):
                    for c, dc in zip(alm, dlm):
                        for d_, dd_ in zip(asg, dsg):
                            w = da * db * dc * dd_ / norm_prod
                            prim_recs.append((
                                mu, nu, lam, sig,
                                a,  float(A_c[0]), float(A_c[1]), float(A_c[2]),
                                b,  float(B_c[0]), float(B_c[1]), float(B_c[2]),
                                c,  float(C_c[0]), float(C_c[1]), float(C_c[2]),
                                d_, float(D_c[0]), float(D_c[1]), float(D_c[2]),
                                w,
                            ))

        if not prim_recs:
            continue

        for start in range(0, len(prim_recs), K_batch):
            chunk = prim_recs[start:start + K_batch]
            arr     = np.array(chunk, dtype=np.float64)   # (B, 21)
            mu_idx  = arr[:, 0].astype(np.intp)
            nu_idx  = arr[:, 1].astype(np.intp)
            lam_idx = arr[:, 2].astype(np.intp)
            sig_idx = arr[:, 3].astype(np.intp)
            alpha_k = arr[:, 4];  A_k = arr[:, 5:8]
            beta_k  = arr[:, 8];  B_k = arr[:, 9:12]
            gamma_k = arr[:, 12]; C_k = arr[:, 13:16]
            delta_k = arr[:, 16]; D_k = arr[:, 17:20]
            w_k     = arr[:, 20]

            vals    = _eri_prim_cart_batch(
                la, lb, lc, ld,
                alpha_k, A_k, beta_k, B_k,
                gamma_k, C_k, delta_k, D_k,
            )
            contrib = vals * w_k   # (B,)

            # Accumulate into H with the same 8 scatter-adds used in
            # _build_eri_batch_by_type — preserves exact numerical identity
            # with the dense-g → einsum path (threshold = 0).
            #
            # Each scatter-add: H[i0[b], i1[b], :, :] += contrib[b] * outer(C[i2[b],:], C[i3[b],:])
            # Combined as: outer_b[b,r,s] = contrib[b]*C[i2[b],r]*C[i3[b],s]
            #              np.add.at(H, (i0, i1), outer_b)
            for i0, i1, i2, i3 in (
                (mu_idx,  nu_idx,  lam_idx, sig_idx),
                (nu_idx,  mu_idx,  lam_idx, sig_idx),
                (mu_idx,  nu_idx,  sig_idx, lam_idx),
                (nu_idx,  mu_idx,  sig_idx, lam_idx),
                (lam_idx, sig_idx, mu_idx,  nu_idx),
                (sig_idx, lam_idx, mu_idx,  nu_idx),
                (lam_idx, sig_idx, nu_idx,  mu_idx),
                (sig_idx, lam_idx, nu_idx,  mu_idx),
            ):
                outer_b = np.einsum("b,br,bs->brs", contrib,
                                    C[i2, :], C[i3, :])  # (B, N, N)
                np.add.at(H, (i0, i1), outer_b)

    # ── Step 5: Final half-transform ─────────────────────────────────────────
    # g_MO[p,q,r,s] = Σ_{mn} H[m,n,r,s] C[m,p] C[n,q]
    log.info(f"[screened_direct] Half-transform contraction (N_sh={N_sh}, N={N})…")
    T    = np.einsum("mp,mnrs->pnrs", C, H,   optimize=True)   # (N, N_sh, N, N)
    g_MO = np.einsum("nq,pnrs->pqrs", C, T,   optimize=True)   # (N, N, N, N)
    log.info(f"[screened_direct] Done. g_MO shape={g_MO.shape}")
    return g_MO
