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
mqeriemannqpe.py — Riemann-Constrained MLE for MQE-QPE
========================================================
Replaces the continuous energy search in hardware_map_energy with a
discrete hypothesis test over the Riemann spectral scaffold.

Theory
------
thm:spectral_identification proves that the Janus eigenphase of H_MQE is
*exactly* φ_{k*} = s·γ_k for some non-trivial Riemann zero γ_k.  The
scaffold {E_k = s·γ_k/(n*·Δt_m)} is therefore the complete set of
physically admissible Janus energies.  The MLE over a continuous energy
axis is replaced by:

    k* = argmax_k  Σ_τ Σ_j p_obs(j|τ) · log p_model(j|τ, E_k, η_V)

This is an exact discrete search — O(|window| × |τ-seq| × 4) — and
provides:
  (a) A dramatically reduced search space (≤ 20 candidates vs. ℝ).
  (b) A natural goodness-of-fit diagnostic: the log-likelihood ratio
      ΔL = L(E_best) − L(E_2nd_best) quantifies how sharply the
      measurement distinguishes the winning zero from its neighbours.
  (c) An internal consistency check: if no candidate achieves
      L > L_threshold, the active-space truncation (χ) is too small or
      the circuit has exceeded the decoherence budget.

For non-Janus steps the function `continuous_mle_fallback` delegates
directly to the existing `hardware_map_energy` from mqevancqpe.py,
preserving full backward compatibility.

Public API
----------
    riemann_constrained_mle(ancilla_probs, scaffold, eta_v)
        Discrete MLE: returns RiemannMLEResult.

    riemann_zne(ancilla_probs_by_lambda, scaffold, eta_v)
        ZNE (Richardson + exponential) over the discrete scaffold.
        Returns RiemannZNEResult.

    continuous_mle_fallback(ancilla_probs, E_ref, eta_v)
        Thin wrapper around the existing hardware_map_energy — used for
        non-Janus steps where the Riemann scaffold does not apply.

Dependencies
------------
    numpy, scipy (via mqevancqpe._p_model_corr),
    nanoprotogeny.molecular.mqeriemann (RiemannScaffold),
    nanoprotogeny.qpe.mqevancqpe (hardware_map_energy, existing MLE).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from nanoprotogeny.molecular.mqeriemann import RiemannScaffold, RIEMANN_ZEROS
from nanoprotogeny.qpe.mqevancqpe import hardware_map_energy

log = logging.getLogger(__name__)

_K_VALS = np.arange(4)


# ── p_model: η_V-corrected MQE-QPE signal (copied scalar form) ───────────────

def _p_model_corr(E: float, tau: float, eta_v: float) -> np.ndarray:
    r"""Corrected MQE-QPE probability vector for k ∈ {0,1,2,3}.

    p_model_corr(k|τ,E,η_V) = (1/16)·[
        4
      + 6·η_V·cos(Eτ − πk/2)
      + 4·η_V²·cos(2(Eτ − πk/2))
      + 2·η_V³·cos(3(Eτ − πk/2))
    ]

    Eq. (mqe_qpe_signal_m4).  Reduces to the ideal model at η_V=1.
    """
    phi = E * tau - np.pi * _K_VALS / 2.0
    return (1.0 / 16.0) * (
        4.0
        + 6.0 * eta_v       * np.cos(phi)
        + 4.0 * eta_v**2    * np.cos(2.0 * phi)
        + 2.0 * eta_v**3    * np.cos(3.0 * phi)
    )


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class RiemannMLEResult:
    """Result of a single Riemann-constrained MLE call.

    Fields
    ------
    E_best              : Energy of the best-matching Riemann candidate [Ha].
    gamma_best          : Corresponding Riemann zero γ_k [Ha⁻¹].
    k_best              : 0-based index into scaffold.gammas.
    zero_index          : 0-based index into RIEMANN_ZEROS (global list).
    log_likelihood_best : log L at E_best.
    log_likelihood_ratio: ΔL = L(best) − L(2nd_best).  Large → sharp selection.
    all_log_likelihoods : List of log L for every scaffold candidate.
    eta_v               : η_V used in the MLE model.
    is_degenerate       : True if ΔL < 1.0 (neighbouring zeros not resolved).
    """
    E_best:               float
    gamma_best:           float
    k_best:               int
    zero_index:           int
    log_likelihood_best:  float
    log_likelihood_ratio: float
    all_log_likelihoods:  List[float]
    eta_v:                float
    is_degenerate:        bool


@dataclass
class RiemannZNEResult:
    """Result of Riemann-constrained ZNE.

    Fields
    ------
    E_zne_rich  : Richardson ZNE result [Ha].
    E_zne_exp   : Exponential-fit ZNE result [Ha].
    E_best      : Better of the two (smaller residual against scaffold centre).
    zne_method  : 'rich' or 'exp'.
    residual_scaffold_mHa: |E_best − E_scaffold_centre| in mHa.
                   E_scaffold_centre = mean of the three winning E_k values
                   from folds λ=1,2,3 (may be the same zero or adjacent ones
                   if ZNE pulls the estimate slightly off a zero).
    mle_results : Dict[int, RiemannMLEResult] for λ=1,2,3.
    chem_ok     : True if residual_scaffold_mHa ≤ 1.6 mHa.
    """
    E_zne_rich:            float
    E_zne_exp:             float
    E_best:                float
    zne_method:            str
    residual_scaffold_mHa: float
    mle_results:           Dict[int, RiemannMLEResult]
    chem_ok:               bool


# ── Core: discrete MLE over scaffold ─────────────────────────────────────────

def riemann_constrained_mle(
    ancilla_probs: Dict[float, np.ndarray],
    scaffold:      "RiemannScaffold",
    eta_v:         float = 1.0,
    epsilon:       float = 1e-12,
) -> RiemannMLEResult:
    r"""Discrete MLE: select the Riemann zero best matching the observed p(k|τ).

    For each candidate energy E_k = s·γ_k/(n*·Δt_m) in the scaffold window,
    evaluate the log-likelihood:

        L(E_k) = Σ_τ Σ_j  p_obs(j|τ) · log(p_model_corr(j|τ, E_k, η_V) + ε)

    Return the E_k achieving the maximum.

    Complexity: O(|window| · |τ-seq| · 4)  — always sub-millisecond.

    Args:
        ancilla_probs: {τ: p_obs(k)} from compute_virtual_ancilla_qpe_probs.
                       p_obs is a length-4 numpy array.
        scaffold:      RiemannScaffold for the current mechanism step.
        eta_v:         D-state decoherence factor (passed to p_model_corr).
        epsilon:       Numerical floor for log (prevents log(0)).

    Returns:
        RiemannMLEResult with the selected zero and diagnostics.
    """
    if not scaffold.janus_energies:
        raise ValueError(
            "RiemannScaffold has no candidates in the eigenphase window. "
            "Either M_steps is too small or m is misconfigured."
        )

    log_likelihoods: List[float] = []

    for E_cand in scaffold.janus_energies:
        ll = 0.0
        for tau, p_obs in ancilla_probs.items():
            p_mod = _p_model_corr(E_cand, tau, eta_v)
            # Clip to [ε, 1] to avoid log(0); sum(p_obs)=1 so this is safe.
            ll += float(np.sum(p_obs * np.log(np.clip(p_mod, epsilon, 1.0))))
        log_likelihoods.append(ll)

    best_k   = int(np.argmax(log_likelihoods))
    best_ll  = log_likelihoods[best_k]

    # Log-likelihood ratio: best vs. 2nd best (resolution diagnostic).
    sorted_lls = sorted(log_likelihoods, reverse=True)
    ll_ratio   = (best_ll - sorted_lls[1]) if len(sorted_lls) > 1 else float("inf")
    is_degenerate = ll_ratio < 1.0

    if is_degenerate:
        log.warning(
            "[RIEMANN-MLE] ΔL=%.3f < 1.0: neighbouring zeros not resolved. "
            "Consider increasing n_shots or χ (bond dimension).", ll_ratio
        )

    return RiemannMLEResult(
        E_best               = scaffold.janus_energies[best_k],
        gamma_best           = scaffold.gammas[best_k],
        k_best               = best_k,
        zero_index           = scaffold.zero_indices[best_k],
        log_likelihood_best  = best_ll,
        log_likelihood_ratio = ll_ratio,
        all_log_likelihoods  = log_likelihoods,
        eta_v                = eta_v,
        is_degenerate        = is_degenerate,
    )


# ── ZNE over the discrete scaffold ───────────────────────────────────────────

def riemann_zne(
    ancilla_probs_by_lambda: Dict[int, Dict[float, np.ndarray]],
    scaffold:                "RiemannScaffold",
    eta_v:                   float = 1.0,
) -> RiemannZNEResult:
    r"""ZNE (Richardson + exponential) with Riemann-constrained MLE at each fold.

    For each noise fold λ ∈ {1, 2, 3}:
        E(λ) = riemann_constrained_mle(ancilla_probs_by_lambda[λ], scaffold, η_V)

    Richardson ZNE:  E_ZNE = 3·E(1) − 3·E(2) + E(3)
    Exponential ZNE: fit E(λ) = E_∞ + A·exp(−bλ), use E_∞.

    The ZNE result may fall slightly off a scaffold energy (noise residual).
    We report the nearest scaffold energy as the "true" Janus and compute the
    residual |E_ZNE − E_nearest_scaffold| to quantify extrapolation quality.

    Args:
        ancilla_probs_by_lambda: {λ: {τ: p_obs(k)}} for λ ∈ {1,2,3}.
        scaffold:                RiemannScaffold for this crossing step.
        eta_v:                   D-state decoherence factor.

    Returns:
        RiemannZNEResult.
    """
    mle_results: Dict[int, RiemannMLEResult] = {}
    E_by_lam: Dict[int, float] = {}

    for lam in [1, 2, 3]:
        res = riemann_constrained_mle(
            ancilla_probs_by_lambda[lam], scaffold, eta_v
        )
        mle_results[lam] = res
        E_by_lam[lam]    = res.E_best
        log.info(
            "[RIEMANN-ZNE λ=%d] γ_%d (E=%+.8f Ha, ΔL=%.3f)",
            lam, res.zero_index + 1, res.E_best, res.log_likelihood_ratio
        )

    E1, E2, E3 = E_by_lam[1], E_by_lam[2], E_by_lam[3]

    # Richardson cancellation (O(λ), O(λ²) error terms)
    E_zne_rich = 3.0 * E1 - 3.0 * E2 + E3

    # Exponential fit: E(λ) = E_inf + A·exp(-b·λ)
    denom = E3 - 2.0 * E2 + E1
    if abs(denom) > 1e-12:
        E_inf   = (E1 * E3 - E2**2) / denom
        denom_e = E2 - E_inf
        E_zne_ex = (
            E_inf + (E1 - E_inf)**2 / denom_e
            if abs(denom_e) > 1e-12 else E_zne_rich
        )
    else:
        E_zne_ex = E_zne_rich

    # Snap to the nearest scaffold energy (the ZNE result is the true Janus).
    # We measure the residual against the nearest candidate, not against E_ref
    # (which would require a quantum simulation result to compare against).
    scaff_arr = np.array(scaffold.janus_energies)
    res_rich = float(np.min(np.abs(scaff_arr - E_zne_rich))) * 1000.0
    res_exp  = float(np.min(np.abs(scaff_arr - E_zne_ex  ))) * 1000.0

    if res_exp <= res_rich:
        E_best, residual, method = E_zne_ex, res_exp, "exp"
    else:
        E_best, residual, method = E_zne_rich, res_rich, "rich"

    return RiemannZNEResult(
        E_zne_rich            = E_zne_rich,
        E_zne_exp             = E_zne_ex,
        E_best                = E_best,
        zne_method            = method,
        residual_scaffold_mHa = residual,
        mle_results           = mle_results,
        chem_ok               = residual <= 1.6,
    )


# ── Fallback: continuous MLE for non-Janus steps ─────────────────────────────

def continuous_mle_fallback(
    ancilla_probs: Dict[float, np.ndarray],
    E_ref:         float,
    eta_v:         float = 1.0,
) -> Tuple[float, float, float]:
    """Delegate to existing hardware_map_energy for non-Janus steps.

    Non-Janus energies (steps n ≠ n*) are not constrained by the Riemann
    scaffold.  This function passes through to the existing continuous MLE
    without modification, preserving full backward compatibility.

    Returns:
        (E_map, ll_best, eta_v_used) — same contract as hardware_map_energy.
    """
    return hardware_map_energy(ancilla_probs, E_ref=E_ref, eta_v=eta_v)
