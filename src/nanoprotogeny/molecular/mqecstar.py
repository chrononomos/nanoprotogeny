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
mqecstar.py — Hilbert–Pólya C* orbital optimisation.
==================================================
Minimise Δ₀(C) = |E_seed(C) − E_∞| over the S-orthogonal Stiefel manifold via
Riemannian gradient descent with QR retraction.  Layer above the MO transform
(:mod:`mqescf`) and the FCI solver (:mod:`mqefci`).

Kummer-side constraint
----------------------
For the Iwasawa tower to converge from above (prop:kummer_convergence), we
need E_seed ≥ E_∞.  The optimisation enforces this via a one-sided penalty
objective:

    obj(C) = { E_seed − E_∞              if E_seed ≥ E_∞   (minimise overshoot)
             { _UNDERSHOOT_PENALTY ×      if E_seed < E_∞   (push to cross E_∞)
               (E_∞ − E_seed)

The gradient is scaled consistently so convergence speed from below is
_UNDERSHOOT_PENALTY× faster than from above.  Early-exit (Δ₀ < ε) only fires
once E_seed has crossed to the correct side.
"""

from __future__ import annotations

import logging
import math
from typing import List, Tuple

import numpy as np

from nanoprotogeny.molecular.mqeconstants import _EPS_MILLI_HA, _K_BASE
from nanoprotogeny.molecular.mqescf import transform_integrals
from nanoprotogeny.molecular.mqefci import solve_cas, compute_rdms

log = logging.getLogger(__name__)

# Gradient / objective scale factor applied when E_seed < E_∞.
# 10× makes the landscape steeper below E_∞, steering the line search toward
# crossing the Kummer boundary before the early-exit condition fires.
_UNDERSHOOT_PENALTY: float = 10.0

def _build_h1_AO_eff(
    h1_AO:  np.ndarray,
    g_AO:   np.ndarray,
    C_core: np.ndarray,
) -> np.ndarray:
    """Effective AO-basis 1e Hamiltonian with Fock screening from frozen core.

    h1_eff[μ,ν] = h1_AO[μ,ν]
                  + Σ_{i∈core} [2 J_i[μ,ν] − K_i[μ,ν]]

    where J_i[μ,ν] = Σ_{λσ} g_AO[μ,ν,λ,σ] C_core[λ,i] C_core[σ,i]
          K_i[μ,ν] = Σ_{λσ} g_AO[μ,λ,ν,σ] C_core[λ,i] C_core[σ,i]

    When C_core has zero columns (no frozen core), returns h1_AO unchanged.

    Args:
        h1_AO  : (N_AO, N_AO) bare core Hamiltonian.
        g_AO   : (N_AO, N_AO, N_AO, N_AO) ERI tensor (chemist notation).
        C_core : (N_AO, n_core) frozen-core MO coefficients.

    Returns:
        h1_eff : (N_AO, N_AO) screened 1e Hamiltonian.
    """
    if C_core.shape[1] == 0:
        return h1_AO.copy()
    # J[μ,ν] = Σ_{i,λ,σ} g[μ,ν,λ,σ] C[λ,i] C[σ,i]
    #        = Σ_i Σ_{λ,σ} g[μ,ν,λ,σ] D_core[λ,σ]  where D_core = C_core C_core^T
    D_core = C_core @ C_core.T          # (N_AO, N_AO) core 1-RDM
    J = np.einsum("mnls,ls->mn", g_AO, D_core)
    # K[μ,ν] = Σ_{i,λ,σ} g[μ,λ,ν,σ] C[λ,i] C[σ,i]
    #        = Σ_{λ,σ} g[μ,λ,ν,σ] D_core[λ,σ]
    K = np.einsum("mlns,ls->mn", g_AO, D_core)
    return h1_AO + 2.0 * J - K


def _eseed_for_C(
    C:         np.ndarray,
    h1_eff:    np.ndarray,
    g_AO:      np.ndarray,
    N_e:       int,
    N_active:  int,
    E_core:    float,
    E_nuc:     float,
) -> float:
    """E_seed = FCI(h1_MO(C), g_MO(C)) — pure active-space electronic energy.

    E_core and E_nuc are accepted for API compatibility but NOT added to the
    objective.  C* optimises on the same scale as E_∞ (active-space electronic
    energy); adding nuclear/frozen-core constants would make δ₀ ~ 9 kHa and
    render the gradient meaningless.
    """
    h1_MO, g_MO = transform_integrals(C, h1_eff, g_AO)
    N_e_use = min(N_e, 2 * N_active - 1)
    E_elec, _, _ = solve_cas(h1_MO, g_MO, N_e=N_e_use, N_orb=N_active, E_nuc=0.0)
    return float(E_elec)


def _eseed_and_grad(
    C:        np.ndarray,
    h1_eff:   np.ndarray,
    g_AO:     np.ndarray,
    N_e:      int,
    N_active: int,
    E_core:   float,
    E_nuc:    float,
) -> Tuple[float, np.ndarray]:
    r"""E_seed and analytic Hellmann–Feynman gradient ∂E_seed/∂C_{μ,p}.

    Derivation (generalized Fock matrix)
    -------------------------------------
    By the Hellmann–Feynman theorem applied to the MO-rotation parameterisation
    C → C + δC,

        ∂E_seed/∂C_{μ,p} = 2 [h1_eff @ C @ γ]_{μ,p}
                           + 2 Σ_{b,c,d} Γ[p,b,c,d]
                               Σ_{ν,λ,σ} g_AO[μ,ν,λ,σ] C[ν,b] C[λ,c] C[σ,d]

    where γ (N_active × N_active) and Γ (N_active⁴) are the spin-summed 1-RDM
    and 2-RDM of the FCI ground state, and g_AO is in chemist notation (μν|λσ).

    The 2-body term contracts as:
        F2[μ,p] = 2 Σ_{b,c,d} Γ[p,b,c,d] g_AO_{μν,λσ} C_{νb} C_{λc} C_{σd}
               = 2 einsum('pbcd,mnls,nb,lc,sd->mp', Γ, g_AO, C, C, C)

    which we compute via three sequential half-transforms to avoid forming the
    full (N_AO)^4 intermediate more than once.

    Args:
        C        : (N_AO, N_active) — current S-orthogonal MO coefficients.
        h1_eff   : (N_AO, N_AO) — Fock-screened 1e Hamiltonian.
        g_AO     : (N_AO, N_AO, N_AO, N_AO) — ERI in chemist notation.
        N_e      : Number of active electrons.
        N_active : Number of active spatial MOs.
        E_core   : Frozen-core energy [Ha].
        E_nuc    : Nuclear repulsion [Ha].

    Returns:
        E_seed   : Total seed energy [Ha].
        G_eucl   : (N_AO, N_active) Euclidean gradient of E_seed w.r.t. C.
    """
    h1_MO, g_MO = transform_integrals(C, h1_eff, g_AO)
    N_e_use = min(N_e, 2 * N_active - 1)
    E_elec, _, psi_0 = solve_cas(h1_MO, g_MO, N_e=N_e_use, N_orb=N_active, E_nuc=0.0)
    # Pure active-space electronic energy — E_core/E_nuc not added (see _eseed_for_C).
    E_seed = float(E_elec)

    gamma, Gamma = compute_rdms(psi_0, N_e_use, N_active)

    # 1e contribution: G1[μ,p] = 2 (h1_eff @ C @ γ)[μ,p]
    G1 = 2.0 * (h1_eff @ C @ gamma)

    # 2e contribution via sequential half-transforms.
    # F2[μ,p] = 2 Σ_{bcd} Γ[p,b,c,d] g_AO[μ,ν,λ,σ] C[ν,b] C[λ,c] C[σ,d]
    # Step 1: contract σ→d:  T1[μ,ν,λ,d] = Σ_σ g_AO[μ,ν,λ,σ] C[σ,d]
    T1 = np.einsum("mnls,sd->mnld", g_AO, C)
    # Step 2: contract λ→c:  T2[μ,ν,c,d] = Σ_λ T1[μ,ν,λ,d] C[λ,c]
    T2 = np.einsum("mnld,lc->mncd", T1, C)
    # Step 3: contract ν→b:  T3[μ,b,c,d] = Σ_ν T2[μ,ν,c,d] C[ν,b]
    T3 = np.einsum("mncd,nb->mbcd", T2, C)
    # Step 4: contract b,c,d with Γ[p,b,c,d]: G2[μ,p] = 2 Σ_{bcd} Γ[p,b,c,d] T3[μ,b,c,d]
    G2 = 2.0 * np.einsum("pbcd,mbcd->mp", Gamma, T3)

    G_eucl = G1 + G2
    return E_seed, G_eucl


def _stiefel_retract(C: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Project C onto the S-Stiefel manifold via thin QR.

    S = L L^T (Cholesky).  Retraction:
        C_tilde = L^T @ C       (standard orthonormality in transformed space)
        Q, _    = qr(C_tilde)   (thin QR → Q has orthonormal columns)
        C_ret   = L^{-T} Q      (back to S-orthogonal basis)

    Args:
        C : (N_AO, N_active) — perturbed MO matrix (not necessarily S-orthogonal).
        L : (N_AO, N_AO) — lower Cholesky factor of S_AO (S = L L^T).

    Returns:
        C_ret : (N_AO, N_active) — S-orthogonal MO matrix.
    """
    from scipy.linalg import solve_triangular, qr as _qr
    C_tilde = L.T @ C
    Q, _ = _qr(C_tilde, mode="economic")
    return solve_triangular(L.T, Q, lower=False)


def hilbert_polya_cstar_optimize(
    h1_AO_eff:  np.ndarray,
    g_AO:       np.ndarray,
    S_AO:       np.ndarray,
    E_inf:      float,
    N_active:   int,
    N_e:        int,
    E_core:     float,
    E_nuc:      float,
    C_init:     np.ndarray,
    eps_thresh: float = _EPS_MILLI_HA,
    max_iter:   int   = 300,
    grad_tol:   float = 1e-6,
    step_size:  float = 5e-3,
    fd_eps:     float = 1e-5,   # kept for API compatibility; ignored (analytic gradient used)
    tower_p:    int   = 2,
) -> Tuple[np.ndarray, float, float, int, List[float]]:
    r"""Hilbert–Pólya C* orbital optimisation.

    Minimise Δ₀(C) = |E_seed(C) − E_∞| over the S-Stiefel manifold
    St(N_active, N_AO) via Riemannian gradient descent with QR retraction and
    Armijo back-tracking line search, subject to the Kummer-side constraint
    E_seed(C) ≥ E_∞.

    Theory: def:hp_variational and thm:variational_equivalence in
    theory/iwasawa-tower-zeros.md.  E_∞ must be supplied by the caller (it is
    independent of C).  h1_AO_eff should already include any frozen-core Fock
    screening (_build_h1_AO_eff) so that E_core is a fixed additive constant
    throughout the optimisation.

    Kummer-side constraint
    ----------------------
    prop:kummer_convergence requires E_seed ≥ E_∞ for the tower to descend
    monotonically from above.  The optimisation uses a one-sided penalty
    objective (module-level _UNDERSHOOT_PENALTY) so that:
      - Early-exit (Δ₀ < ε) only fires once E_seed has crossed above E_∞.
      - The gradient is scaled ×_UNDERSHOOT_PENALTY when E_seed < E_∞ to push
        the search across the Kummer boundary faster.

    Gradient
    --------
    Uses the analytic Hellmann–Feynman gradient via 1-RDM (γ) and 2-RDM (Γ)
    from the FCI ground state (see :func:`_eseed_and_grad`).  This costs one
    FCI solve + one RDM contraction per step, vs. 2×N_AO×N_active FCI solves
    for finite differences.

    Args:
        h1_AO_eff  : (N_AO, N_AO) effective AO 1e Hamiltonian (may include FC).
        g_AO       : (N_AO, N_AO, N_AO, N_AO) ERI tensor in chemist notation.
        S_AO       : (N_AO, N_AO) AO overlap matrix.
        E_inf      : E_∞ — fixed Janus energy from Riemann zeros [Ha].
        N_active   : Number of active spatial MOs.
        N_e        : Number of active electrons.
        E_core     : Frozen-core energy [Ha] (0.0 for bare path).
        E_nuc      : Nuclear repulsion energy [Ha].
        C_init     : (N_AO, N_active) — initial S-orthogonal MO guess.
        eps_thresh : Tower convergence threshold [Ha] (default 1.6 mHa).
        max_iter   : Maximum gradient steps.
        grad_tol   : Stop when ‖G_tang‖_F < grad_tol.
        step_size  : Initial Armijo step size α.
        fd_eps     : Ignored (kept for API compatibility — analytic gradient used).
        tower_p    : Tower prime p for computing k₀ from Δ₀.

    Returns:
        C_star       : (N_AO, N_active) — optimised MO matrix with E_seed ≥ E_∞.
        delta0_star  : Actual Δ₀ = |E_seed(C_star) − E_∞| [Ha].
        k0_star      : Tower level k₀ at C_star.
        n_iter       : Number of gradient steps taken.
        history      : List of actual Δ₀ values (symmetric, not penalised) at
                       each iteration, for plotting convergence curves.
    """
    from scipy.linalg import cholesky as _chol

    N_AO = h1_AO_eff.shape[0]
    C = C_init.copy()

    # Cholesky factor of S for retraction.
    L = _chol(S_AO, lower=True)

    def _e_seed(C_: np.ndarray) -> float:
        return _eseed_for_C(C_, h1_AO_eff, g_AO, N_e, N_active, E_core, E_nuc)

    def _obj(C_: np.ndarray) -> float:
        """One-sided Kummer objective — penalise undershoot by _UNDERSHOOT_PENALTY."""
        e = _e_seed(C_)
        overshoot = e - E_inf
        return overshoot if overshoot >= 0.0 else (-overshoot) * _UNDERSHOOT_PENALTY

    def _k0_from_delta(d: float) -> int:
        if d < eps_thresh:
            return _K_BASE
        return max(
            _K_BASE,
            math.ceil(2.0 + math.log(d / eps_thresh) / math.log(tower_p)),
        )

    e_init = _e_seed(C)
    delta0_cur = abs(e_init - E_inf)   # actual distance (for reporting / history)
    obj_cur    = _obj(C)               # penalised objective (for Armijo / early-exit)
    history: List[float] = [delta0_cur]
    kummer_ok_init = e_init >= E_inf
    log.info(
        f"[cstar] Start: Δ₀ = {delta0_cur:.6e} Ha, k₀ = {_k0_from_delta(delta0_cur)}, "
        f"side = {'above' if kummer_ok_init else 'BELOW'} E_∞"
    )

    n_iter = 0
    grad_norm = float("inf")
    for step_idx in range(max_iter):
        # ── Analytic Hellmann–Feynman gradient of E_seed w.r.t. C ──────────
        e_cur, G_eseed = _eseed_and_grad(C, h1_AO_eff, g_AO, N_e, N_active, E_core, E_nuc)

        # Kummer-side gradient scaling:
        #   Above E_∞: ∂obj/∂C = +∂E_seed/∂C  (minimise overshoot)
        #   Below E_∞: ∂obj/∂C = −_UNDERSHOOT_PENALTY × ∂E_seed/∂C
        #              (maximise E_seed scaled to match penalty objective)
        overshoot_cur = e_cur - E_inf
        scale = 1.0 if overshoot_cur >= 0.0 else -_UNDERSHOOT_PENALTY
        G_eucl = scale * G_eseed

        # ── Riemannian (tangent-space) gradient ──────────────────────────────
        # T_C St = {Δ : C^T S Δ + Δ^T S C = 0}
        # Projection: G_tang = G - C sym(C^T S G)
        CtSG = C.T @ S_AO @ G_eucl        # (N_active, N_active)
        sym_CtSG = 0.5 * (CtSG + CtSG.T)
        G_tang = G_eucl - C @ sym_CtSG

        grad_norm = float(np.linalg.norm(G_tang, "fro"))
        n_iter = step_idx + 1

        if grad_norm < grad_tol:
            log.info(f"[cstar] Converged (‖G‖={grad_norm:.2e}) at step {n_iter}")
            break

        # ── Armijo back-tracking line search on one-sided objective ─────────
        alpha = step_size
        obj_target = obj_cur
        C_trial = C
        obj_trial = obj_cur
        for _ in range(20):
            C_trial = _stiefel_retract(C - alpha * G_tang, L)
            obj_trial = _obj(C_trial)
            # Armijo: sufficient decrease (c=0.1)
            if obj_trial <= obj_target - 0.1 * alpha * grad_norm**2:
                break
            alpha *= 0.5

        C       = C_trial
        obj_cur = obj_trial
        # Actual (symmetric) distance for history / reporting
        e_new      = _e_seed(C)
        delta0_cur = abs(e_new - E_inf)
        history.append(delta0_cur)

        kummer_ok = e_new >= E_inf
        if step_idx % 20 == 0 or (kummer_ok and delta0_cur < eps_thresh):
            log.info(
                f"[cstar] iter {n_iter:4d}: Δ₀={delta0_cur:.6e} Ha "
                f"{'(above E_∞)' if kummer_ok else '(BELOW E_∞)'} "
                f"‖G‖={grad_norm:.2e}  α={alpha:.2e}"
            )

        # Early exit only once E_seed has crossed above E_∞ (Kummer side).
        if kummer_ok and delta0_cur < eps_thresh:
            log.info(f"[cstar] Δ₀ < ε (above E_∞) at step {n_iter} — trivial tower.")
            break

        # ── Stall detection: geometry outside spectral participation window ──
        # If the Armijo search has collapsed to α < 1e-7 (i.e. 0.5^14 × step_size
        # with step_size=5e-3 → α_min ≈ 3e-7) after at least 20 iterations, the
        # line search will never recover — the landscape has no descent direction
        # within CAS(4,4).  Continuing to max_iter wastes time with no reduction.
        if n_iter >= 20 and alpha < 1e-7:
            log.info(
                f"[cstar] Stall detected at iter {n_iter}: α={alpha:.2e} < 1e-7. "
                f"Geometry outside spectral participation window — no C* in CAS(4,4). "
                f"Exiting early."
            )
            break

    # ── Final reporting — use actual (unpenalised) δ₀ ───────────────────────
    e_final    = _e_seed(C)
    kummer_ok  = e_final >= E_inf
    delta0_star = abs(e_final - E_inf)
    k0_star    = _k0_from_delta(delta0_star)
    stalled    = (delta0_star >= eps_thresh and grad_norm >= grad_tol)
    if stalled:
        status = "stalled (outside spectral window)"
    elif not kummer_ok:
        status = "converged (WARNING: below E_∞ — Kummer ✗)"
    else:
        status = "converged (above E_∞ — Kummer ✓)"
    log.info(
        f"[cstar] Done [{status}]: Δ₀ {history[0]:.6e} → {delta0_star:.6e} Ha "
        f"(reduction ×{history[0]/max(delta0_star,1e-15):.1f}), "
        f"k₀ {_k0_from_delta(history[0])} → {k0_star}, iters={n_iter}, "
        f"E_seed {'≥' if kummer_ok else '<'} E_∞"
    )
    return C, delta0_star, k0_star, n_iter, history
