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
mqephasetracker.py — ℤ_m Stoichiometric Phase Tracker
======================================================
Classical implementation of the net-flux ℤ_m phase recurrence from
Theorem 2 (Net-Flux Stoichiometric Invariance) of the MQE theory.

    k^{(n)} ≡ (k^{(n-1)} + ν_n − ν_n^{dec}) mod m,   k^{(0)} = 0.

ZmPhaseTracker is the classical shadow of the quantum virtual phase
register: it runs in parallel with the circuit, tracking the same
arithmetic deterministically and enabling stoichiometric verification
without quantum measurement.

The tracker is the dynamic counterpart of MechanismTuple.phase_closure_satisfied
(which checks closure statically from the mechanism spec).  Together they
form the specification-and-verification layer for ℤ_m stoichiometry.

Module-level dependencies: stdlib (typing), MechanismTuple from mqemolecules.
The compensation_gate() method carries a lazy import of GeneralizedVirtualShiftGate
from nanoprotogeny.ionq.ionqmqegates — this fires only at call time, so
mqephasetracker.py itself has no hard ionq or cirq dependency at import.
"""

from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING

from nanoprotogeny.molecular.mqemolecules import MechanismTuple

if TYPE_CHECKING:
    from nanoprotogeny.ionq.ionqmqegates import (
        GeneralizedVirtualShiftGate,
        CompositeVirtualShiftGate,
    )

class ZmPhaseTracker:
    r"""Deterministic Z_m stoichiometric phase bookkeeper — net-flux edition.

    Implements the generalised phase index recurrence from Theorem 2
    (Net-Flux Stoichiometric Invariance):

        k^{(n)} ≡ (k^{(n-1)} + nu_n − nu_decouple_n) mod m,   k^{(0)} = 0.

    Tracks the *net* virtual phase shift for each cofactor register across
    the M catalytic steps, accounting for both forward coupling (nu) and
    inverse decoupling (nu_decouple). On completion, verifies:

        net phase closure:  Σ(nu_n − nu_decouple_n) ≡ 0 (mod m)
        net electron flux:  <N_e>_net = Σ(|A_n| − |A_n_eject|)

    The original forward-only interface is fully preserved:
        tracker.step(n, nu, n_injected)   ← nu_decouple=0 by default

    Args:
        m: Virtual register modulus.
    """

    def __init__(self, m: int):
        self._m                   = m
        self._k_step:       int   = 0   # Current net phase index k^{(n)} (mod m)
        self._k_total:      int   = 0   # Cumulative net shift (before mod)
        self._n_electrons:  int   = 0   # Cumulative electrons injected (forward)
        self._n_ejected:    int   = 0   # Cumulative electrons ejected (reverse)
        self._n_absorbed:   int   = 0   # Cumulative photons absorbed
        self._n_emitted:    int   = 0   # Cumulative photons emitted
        self._step_log: List[Dict] = []

    def step(
        self,
        n:                    int,
        nu:                   int,
        n_electrons_injected: int,
        nu_decouple:          int = 0,
        n_electrons_ejected:  int = 0,
        n_photons_absorbed:   int = 0,
        n_photons_emitted:    int = 0,
    ) -> int:
        r"""Advance one catalytic step, accounting for both forward/inverse and photon flux.

        Args:
            n:                    Step index (0-based).
            nu:                   Forward cofactor shift (coupling) for this step.
            n_electrons_injected: |A_n|, electrons injected at step n.
            nu_decouple:          Inverse cofactor shift (decoupling) for this step.
                                  Subtracted from the phase index. Default 0.
            n_electrons_ejected:  |A_n_eject|, electrons ejected at step n.
                                  Subtracted from net electron count. Default 0.
            n_photons_absorbed:   |Γ_n_abs|, photons absorbed at step n. Default 0.
            n_photons_emitted:    |Γ_n_emit|, photons emitted at step n. Default 0.

        Returns:
            k^{(n)}: net phase index after this step (reduced mod m).
        """
        net_nu             = nu - nu_decouple
        self._k_total     += net_nu
        self._k_step       = self._k_total % self._m
        self._n_electrons += n_electrons_injected
        self._n_ejected   += n_electrons_ejected
        self._n_absorbed  += n_photons_absorbed
        self._n_emitted   += n_photons_emitted
        self._step_log.append({
            "step":                 n,
            "nu":                   nu,
            "nu_decouple":          nu_decouple,
            "net_nu":               net_nu,
            "k_step":               self._k_step,
            "k_total_raw":          self._k_total,
            "electrons_injected":   n_electrons_injected,
            "electrons_ejected":    n_electrons_ejected,
            "cumulative_injected":  self._n_electrons,
            "cumulative_ejected":   self._n_ejected,
            "cumulative_net":       self._n_electrons - self._n_ejected,
            "photons_absorbed":     n_photons_absorbed,
            "photons_emitted":      n_photons_emitted,
            "cumulative_absorbed":  self._n_absorbed,
            "cumulative_emitted":   self._n_emitted,
            "cumulative_net_photons": self._n_absorbed - self._n_emitted,
            # Legacy keys preserved for backward-compatible log consumers:
            "electrons":            n_electrons_injected,
            "cumulative_electrons":  self._n_electrons,
        })
        return self._k_step

    @property
    def k_total(self) -> int:
        """Net cumulative cofactor shift (forward − inverse, before modular reduction)."""
        return self._k_total

    @property
    def phase_closed(self) -> bool:
        """True iff net k_total ≡ 0 (mod m): net-flux cycle closure condition."""
        if self._m == 1:
            return True
        return (self._k_total % self._m) == 0

    @property
    def total_electrons(self) -> int:
        """Cumulative electrons injected (forward only, kept for backward compat)."""
        return self._n_electrons

    @property
    def total_electrons_ejected(self) -> int:
        """Cumulative electrons ejected (reverse pathway)."""
        return self._n_ejected

    @property
    def net_electrons(self) -> int:
        """Net electron flux: injected − ejected. Zero for a closed cycle."""
        return self._n_electrons - self._n_ejected

    @property
    def total_photons_absorbed(self) -> int:
        """Cumulative photons absorbed across all completed steps."""
        return self._n_absorbed

    @property
    def total_photons_emitted(self) -> int:
        """Cumulative photons emitted across all completed steps."""
        return self._n_emitted

    @property
    def net_photons(self) -> int:
        """Net photon flux: absorbed − emitted. Zero for a closed optical cycle."""
        return self._n_absorbed - self._n_emitted

    def verify(self, mechanism: MechanismTuple) -> Dict:
        r"""Run full net-flux stoichiometric verification against mechanism spec.

        Checks:
          (i)  Net electron flux:  <N_e>_net == mechanism.total_net_electrons
          (ii) Net phase closure:  k_total ≡ 0 (mod m)

        For purely forward mechanisms (no ejection fields populated) the net
        values equal the cumulative forward values, preserving full backward
        compatibility with all existing callers.

        Returns:
            Dict with 'passed', 'electron_ok', 'phase_ok', and details.
        """
        expected_net_e = mechanism.total_net_electrons
        actual_net_e   = self.net_electrons
        e_ok           = (actual_net_e == expected_net_e)
        phi_ok         = self.phase_closed

        return {
            "passed":                   e_ok and phi_ok,
            "electron_ok":              e_ok,
            "phase_ok":                 phi_ok,
            # Forward counts (legacy keys preserved)
            "expected_electrons":       mechanism.total_electrons,
            "actual_electrons":         self._n_electrons,
            # Net-flux counts (new)
            "expected_net_electrons":   expected_net_e,
            "actual_net_electrons":     actual_net_e,
            "actual_electrons_ejected": self._n_ejected,
            "expected_k_mod_m":         0,
            "actual_k_total":           self._k_total,
            # Photon balance (informational)
            "total_photons_absorbed":   self._n_absorbed,
            "total_photons_emitted":    self._n_emitted,
            "net_photons":              self._n_absorbed - self._n_emitted,
            "actual_k_mod_m":           self._k_total % max(self._m, 1),
            "m":                        self._m,
            "step_log":                 self._step_log,
        }

    def report(self, mechanism: MechanismTuple) -> str:
        v     = self.verify(mechanism)
        is_rv = mechanism.is_reversible_cycle
        lines = [
            f"\n[STOICHIOMETRY] ℤ_{self._m} Net-Flux Phase Closure & Electron Count",
            f"  e⁻ inject : {v['actual_electrons']} | "
            f"e⁻ eject: {v['actual_electrons_ejected']} | "
            f"Net: {v['actual_net_electrons']} "
            f"(expected {v['expected_net_electrons']}) "
            f"{'[✓]' if v['electron_ok'] else '[✗]'}",
            f"  Phase:    Σ(ν−ν†)={v['actual_k_total']}, "
            f"mod {self._m} = {v['actual_k_mod_m']} (expected 0) "
            f"{'[✓]' if v['phase_ok'] else '[✗]'}",
        ]
        if is_rv:
            lines.append("  [Reversible cycle — net-flux invariants active]")
        if v["total_photons_absorbed"] > 0 or v["total_photons_emitted"] > 0:
            lines.append(
                f"  Photon balance: abs={v['total_photons_absorbed']} "
                f"emit={v['total_photons_emitted']} "
                f"net={v['net_photons']}"
            )
        lines.append("  Step log:")
        for entry in v["step_log"]:
            nu_str = (
                f"ν={entry['nu']}/ν†={entry['nu_decouple']}"
                if entry.get("nu_decouple", 0) != 0
                else f"ν={entry['nu']}"
            )
            lines.append(
                f"    n={entry['step']:02d}: {nu_str}, "
                f"k^(n)={entry['k_step']}, "
                f"Σe_net={entry['cumulative_net']}"
            )
        lines.append(
            f"  Overall: {'[✓] PASSED' if v['passed'] else '[✗] FAILED'}"
        )
        return "\n".join(lines)

    def compensation_gate(self):
        r"""Return the U_comp gate that restores the virtual register before logical re-entry.

        Computes the group-theoretic inverse of the accumulated phase shift
        k^{(n)} in ℤ_m, i.e. the gate whose power is (m − k^{(n)}) mod m.
        Applying this gate immediately after step n returns the virtual register
        to |0⟩ before the next logical manifold re-entry.

        Gate selection by modulus:

        m = 1           : identity — returns GeneralizedVirtualShiftGate(m=1, power=0).
        m = 4  (r=1)    : GeneralizedVirtualShiftGate(m=4, power=...) — hardware-native
                          d=4 shift on a single VirtualQudit.
        m > 1 (all)     : CompositeVirtualShiftGate(m=m, power=...) — covers every
                          modulus via the unified composite register.  For r=1 acts
                          on V₁ alone; for r>1 acts on (V₁, V_aux).  The modular
                          correction C_m is built into the gate for non-4r m.

        Returns:
            GeneralizedVirtualShiftGate or CompositeVirtualShiftGate.
        """
        power = (self._m - self._k_step) % self._m if self._m > 1 else 0
        if self._m > 1:
            # All m > 1: composite gate on the unified V₁ (+ V_aux) register.
            # For m = 1: trivial — returns a zero-power gate (identity).
            from nanoprotogeny.ionq.ionqmqegates import CompositeVirtualShiftGate
            return CompositeVirtualShiftGate(m=self._m, power=power)
        else:
            from nanoprotogeny.ionq.ionqmqegates import GeneralizedVirtualShiftGate
            return GeneralizedVirtualShiftGate(m=1, power=0)

