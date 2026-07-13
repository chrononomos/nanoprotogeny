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
"""
mqestoichiometry.py — Stoichiometric Invariance Verification
=============================================================
Implements the four structural guarantees of Theorem 2
(Universal Stoichiometric Invariance) from the MQE framework paper.

Separated from simulate/mqe.py because the verification logic is pure
mathematics — it depends only on:

  * nanoprotogeny.molecular.mqemolecules  (MechanismTuple)
  * nanoprotogeny.molecular.mqephasetracker (ZmPhaseTracker)
  * nanoprotogeny.molecular.mqehamiltonian  (_partial_trace_qudit)
  * numpy

No cirq, no IonQ, no pipeline machinery is required.  The class can
therefore be used to verify mechanism correctness independently of any
simulation run (e.g. from ``mqe validate``).

Public API
----------
StoichiometricVerifier
    Full four-condition invariance suite.  Construct with a
    MechanismTuple and η threshold, then call ``full_report()``
    once a simulation step has completed.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from nanoprotogeny.molecular.mqemolecules import MechanismTuple
from nanoprotogeny.molecular.mqephasetracker import ZmPhaseTracker
from nanoprotogeny.molecular.mqehamiltonian import _partial_trace_qudit


# ==============================================================================
# STOICHIOMETRIC VERIFIER
# ==============================================================================

class StoichiometricVerifier:
    r"""Full stoichiometric invariance verification suite.

    Implements the four structural guarantees of Theorem 2 (Universal
    Stoichiometric Invariance) and dynamically routes the spin-parity
    check based on the mechanism's S_target.

    Conditions verified
    -------------------
    (i)   Net electron flux conservation:  <N_e>_net == expected
    (ii)  Net-flux phase closure:          Σ(ν−ν†) ≡ 0 (mod m)
    (iii) Trace preservation:              Tr(ρ_final) = 1
    (iv)  Spin-parity holding:             min_p ω_p ≥ η

    Usage::

        verifier = StoichiometricVerifier(mechanism, eta=0.90)
        all_ok, checks = verifier.full_report(
            tracker    = phase_tracker,
            final_rho  = rho,
            n_orbitals = N,
            S_target   = mechanism.S_target,
        )
    """

    def __init__(self, mechanism: MechanismTuple, eta: float):
        self._mechanism = mechanism
        self._eta       = eta

    # ── Individual condition checks ───────────────────────────────────────────

    def verify_phase_closure(self, tracker: ZmPhaseTracker) -> Dict:
        """Verify condition (ii): net k_total ≡ 0 (mod m)."""
        result = tracker.verify(self._mechanism)
        return {
            "condition": "(ii) Net-flux phase closure",
            "passed":    result["phase_ok"],
            "k_total":   result["actual_k_total"],
            "k_mod_m":   result["actual_k_mod_m"],
            "m":         result["m"],
            "detail": (
                f"Σ(ν−ν†)={result['actual_k_total']} mod {result['m']} "
                f"= {result['actual_k_mod_m']} (expected 0)"
            ),
        }

    def verify_electron_count(self, tracker: ZmPhaseTracker) -> Dict:
        """Verify condition (i): net electron flux == mechanism.total_net_electrons."""
        result   = tracker.verify(self._mechanism)
        expected = result["expected_net_electrons"]
        actual   = result["actual_net_electrons"]
        return {
            "condition": "(i) Net electron flux conservation",
            "passed":    result["electron_ok"],
            "expected":  expected,
            "actual":    actual,
            "injected":  result["actual_electrons"],
            "ejected":   result["actual_electrons_ejected"],
            "detail": (
                f"<N_e>_net = {actual} "
                f"(injected {result['actual_electrons']} "
                f"− ejected {result['actual_electrons_ejected']}) "
                f", expected {expected}"
            ),
        }

    def verify_trace_preservation(self, final_rho: np.ndarray) -> Dict:
        """Verify condition (iii): Tr(ρ_final) = 1.

        Tolerance is relaxed to 1e-5 to accommodate minor numerical noise
        from non-trace-preserving Kraus projectors.
        """
        tr     = float(np.real(np.trace(final_rho)))
        passed = abs(tr - 1.0) < 1e-5
        return {
            "condition": "(iii) Trace preservation",
            "passed":    passed,
            "trace":     tr,
            "detail":    f"Tr(ρ_final) = {tr:.8f} (expected 1.0)",
        }

    def verify_spin_parity(
        self,
        final_rho:  np.ndarray,
        n_orbitals: int,
        S_target:   float,
    ) -> Dict:
        """Verify condition (iv): target spin sector population ≥ η.

        Projector selection
        -------------------
        S_target ≥ 1.0  → high-spin: AntiTh (|1⟩) + SynTh (|2⟩)
        S_target < 1.0  → singlet:   HoloTh (|3⟩)
        """
        if S_target >= 1.0:
            P_target = np.zeros((4, 4), dtype=complex)
            P_target[1, 1] = 1.0
            P_target[2, 2] = 1.0
            label = "AntiTh+SynTh (high-spin)"
        else:
            P_target = np.zeros((4, 4), dtype=complex)
            P_target[3, 3] = 1.0
            label = "HoloTh (singlet)"

        # Normalize to guard against non-trace-preserving projectors
        trace_norm     = float(np.real(np.trace(final_rho)))
        normalized_rho = final_rho / trace_norm if trace_norm > 1e-12 else final_rho

        warrants = []
        for p in range(n_orbitals):
            rho_p = _partial_trace_qudit(normalized_rho, [p], n_orbitals, d=4)
            warrants.append(float(np.real(np.trace(P_target @ rho_p))))

        all_ok = all(w >= self._eta - 1e-6 for w in warrants)
        min_w  = min(warrants) if warrants else 0.0

        return {
            "condition":   "(iv) Spin-parity holding",
            "passed":      all_ok,
            "warrants":    warrants,
            "min_warrant": min_w,
            "eta":         self._eta,
            "detail":      f"min ω = {min_w:.4f} vs η={self._eta} [{label}]",
        }

    # ── Combined report ───────────────────────────────────────────────────────

    def full_report(
        self,
        tracker:    ZmPhaseTracker,
        final_rho:  np.ndarray,
        n_orbitals: int,
        S_target:   float = 0.0,
    ) -> Tuple[bool, List[Dict]]:
        """Run all four verification conditions and return (all_passed, checks).

        Returns
        -------
        all_passed : bool
            True iff every condition passed.
        checks : List[Dict]
            One dict per condition with keys: condition, passed, detail, …
        """
        checks = [
            self.verify_electron_count(tracker),
            self.verify_phase_closure(tracker),
            self.verify_trace_preservation(final_rho),
            self.verify_spin_parity(final_rho, n_orbitals, S_target),
        ]
        return all(c["passed"] for c in checks), checks
