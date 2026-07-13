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
mqemolecules.py — MQE Mechanism Descriptors and Predefined Mechanism Registry
==============================================================================
Standalone module containing:

  • MechanismTuple — the formal 7-component mechanism data tuple
    M = (N, M, m, {H_{n,1},H_{n,2}}, {A_n,B_n,P_n,ν_n,...}, crossings, S_target)
    extracted from mqe.py so that molecular-layer code can use it without
    importing the full quantum simulation stack.

  • _make_uniform_sets — orbital distribution helper for demo mechanisms.

  • _build_*_spec functions — builders for all predefined catalytic mechanisms
    (nitrogenase LT variants, PSII, hydrogenase, CYP450, etc.).

  • build_predefined_mechanisms — registry dict {name: MechanismTuple}.

Dependencies: numpy, dataclasses, typing, stdlib only.
No cirq, ionq, or simulate-layer imports.

mqe.py imports from here:
    from nanoprotogeny.molecular.mqemolecules import (
        MechanismTuple,
        build_predefined_mechanisms,
    )
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

@dataclass
class MechanismTuple:
    r"""Formal 7-component mechanism data tuple M, extended with inverse fields.

    Implements Definition 2 (Fermionic Catalytic Mechanism Tuple) from the
    article Section "Universal Hardware-Isomorphic Framework", generalised
    to a fully reversible bidirectional catalytic cycle.

    Fields mirror the extended tuple:
        M = (N, M_steps, m, {H_{n,1}, H_{n,2}},
             {A_n, B_n, P_n, nu_n,
              A_n_eject, P_n_eject, B_n_decouple, nu_decouple_n,
              Gamma_n_abs, Gamma_n_emit, phi_photon_n},
             {(n_i, p_i, q_i, delta_CI_i)}, S_target)

    ── FORWARD FIELDS (injection / coupling) ──────────────────────────────────
        name:          Human-readable mechanism identifier.
        N_orbitals:    Number of active spatial orbitals.
        M_steps:       Number of discrete mechanistic steps.
        m:             Virtual register modulus (Z_m phase group).
        S_target:      Target spin quantum number (half-integer).
        electron_sets: List[List[int]] of length M_steps. A_n — orbitals
                       receiving one electron via U_R (reduction/injection).
        proton_sets:   List[List[int]] of length M_steps. P_n — orbitals
                       receiving proton phase rotation Z_Clock(phi).
        cofactor_sets: List[List[int]] of length M_steps. B_n — virtual
                       register indices for forward cofactor coupling.
        nu_shifts:     List[int] of length M_steps. nu_n — cofactor shift
                       per step; B_n receives (U_R^{V,m})^{nu_n}.
        crossings:     List of (step_idx, orbital_p, orbital_q, delta_CI)
                       specifying conical intersections. Empty for adiabatic.
        phi_proton:    Proton phase rotation angle (radians). Default pi/2.
        phi_photon:    Photon phase per step (radians). Default pi/2. Used as
                       the phi argument to PhotonAbsorptionGate/PhotonEmissionGate
                       when the per-step lists below are non-empty.
        dock_orbitals: List[List[int]] of length M_steps. D_n — local
                       conformational neighbourhood orbitals for S_dock.
        description:   Free-text description of the physical mechanism.

    ── PHOTON FIELDS (absorption / emission — photo-driven pathways) ──────────
        photon_absorb_sets: List[List[int]] of length M_steps. Γ_n_abs —
                            orbital indices receiving a photon via
                            PhotonAbsorptionGate(phi_photon). Empty = no
                            absorption at step n.
        photon_emit_sets:   List[List[int]] of length M_steps. Γ_n_emit —
                            orbital indices emitting a photon via
                            PhotonEmissionGate(phi_photon). Empty = no
                            emission at step n. Exact inverse of absorption.

    ── REVERSE FIELDS (ejection / decoupling — bidirectional PCET) ───────────
        electron_eject_sets:    List[List[int]] of length M_steps. A_n_eject
                                — orbitals losing one electron via U_R†
                                (oxidation/ejection). Empty list = no ejection.
        proton_eject_sets:      List[List[int]] of length M_steps. P_n_eject
                                — orbitals undergoing deprotonation via
                                Z_Clock(−phi). Empty list = no deprotonation.
        cofactor_decouple_sets: List[List[int]] of length M_steps. B_n_decouple
                                — virtual register indices for inverse coupling
                                U_coupling†. Empty list = no decoupling.
        nu_decouple_shifts:     List[int] of length M_steps. nu_decouple_n —
                                inverse cofactor shift magnitude subtracted from
                                the phase index at each step.

    ── STOICHIOMETRIC INVARIANTS ─────────────────────────────────────────────
    Forward phase closure (legacy):
        sum_{n=0}^{M-1} nu_n ≡ 0  (mod m)

    Net-flux phase closure (Theorem 2 extension, bidirectional):
        sum_{n=0}^{M-1} (nu_n − nu_decouple_n) ≡ 0  (mod m)

    Net electron flux:
        sum_{n=0}^{M-1} (|A_n| − |A_n_eject|)  — equals zero for closed cycles.

    Photon balance (informational, not a hard closure condition):
        sum_{n=0}^{M-1} (|Γ_n_abs| − |Γ_n_emit|)  — net absorbed photons.
        For a photo-driven machine this is the driving photon count;
        for a closed-loop emitter it should equal zero.
    """
    name:          str
    N_orbitals:    int
    M_steps:       int
    m:             int          # Virtual register modulus
    S_target:      float
    electron_sets: List[List[int]]          # A_n: orbitals receiving e⁻
    proton_sets:   List[List[int]]          # P_n: orbitals receiving H⁺
    cofactor_sets: List[List[int]]          # B_n: virtual cofactor registers
    nu_shifts:     List[int]                # nu_n: cofactor shift per step (forward)

    # ── REVERSE PATHWAY FIELDS (bidirectional PCET / catalytic reset) ────────
    # Default to empty lists so all existing callers are backward-compatible.
    electron_eject_sets:    List[List[int]] = field(default_factory=list)
    # A_n_eject: orbitals losing e⁻ via U_R† (oxidation)
    proton_eject_sets:      List[List[int]] = field(default_factory=list)
    # P_n_eject: orbitals deprotonated via Z_Clock(−phi)
    cofactor_decouple_sets: List[List[int]] = field(default_factory=list)
    # B_n_decouple: virtual registers for inverse coupling U_coupling†
    nu_decouple_shifts:     List[int]       = field(default_factory=list)
    # nu_decouple_n: inverse cofactor shift subtracted from phase index

    # ── PHOTON PATHWAY FIELDS (photo-excitation / emission) ─────────────────
    photon_absorb_sets: List[List[int]] = field(default_factory=list)
    # Γ_n_abs: orbitals receiving a photon via PhotonAbsorptionGate(phi_photon)
    photon_emit_sets:   List[List[int]] = field(default_factory=list)
    # Γ_n_emit: orbitals emitting a photon via PhotonEmissionGate(phi_photon)

    crossings:     List[Tuple[int,int,int,float]] = field(default_factory=list)
    phi_proton:    float = np.pi / 2        # Proton phase angle
    phi_photon:    float = np.pi / 2        # Photon phase angle (abs + emit)
    dock_orbitals: Optional[List[List[int]]] = None
    description:   str = ""

    def __post_init__(self):
        assert len(self.electron_sets) == self.M_steps, \
            f"electron_sets must have M_steps={self.M_steps} entries"
        assert len(self.proton_sets) == self.M_steps, \
            f"proton_sets must have M_steps={self.M_steps} entries"
        assert len(self.cofactor_sets) == self.M_steps, \
            f"cofactor_sets must have M_steps={self.M_steps} entries"
        assert len(self.nu_shifts) == self.M_steps, \
            f"nu_shifts must have M_steps={self.M_steps} entries"
        if self.dock_orbitals is not None:
            assert len(self.dock_orbitals) == self.M_steps

        # ── Auto-fill reverse fields to M_steps empty lists if omitted ────────
        # This preserves full backward compatibility: callers that never set
        # these fields get a semantically correct "no reverse pathway" default.
        if not self.electron_eject_sets:
            object.__setattr__(self, "electron_eject_sets",
                               [[] for _ in range(self.M_steps)])
        if not self.proton_eject_sets:
            object.__setattr__(self, "proton_eject_sets",
                               [[] for _ in range(self.M_steps)])
        if not self.cofactor_decouple_sets:
            object.__setattr__(self, "cofactor_decouple_sets",
                               [[] for _ in range(self.M_steps)])
        if not self.nu_decouple_shifts:
            object.__setattr__(self, "nu_decouple_shifts",
                               [0] * self.M_steps)

        # ── Auto-fill photon fields ───────────────────────────────────────────
        if not self.photon_absorb_sets:
            object.__setattr__(self, "photon_absorb_sets",
                               [[] for _ in range(self.M_steps)])
        if not self.photon_emit_sets:
            object.__setattr__(self, "photon_emit_sets",
                               [[] for _ in range(self.M_steps)])

        # ── Validate lengths of supplied reverse fields ───────────────────────
        assert len(self.electron_eject_sets) == self.M_steps, \
            f"electron_eject_sets must have M_steps={self.M_steps} entries"
        assert len(self.proton_eject_sets) == self.M_steps, \
            f"proton_eject_sets must have M_steps={self.M_steps} entries"
        assert len(self.cofactor_decouple_sets) == self.M_steps, \
            f"cofactor_decouple_sets must have M_steps={self.M_steps} entries"
        assert len(self.nu_decouple_shifts) == self.M_steps, \
            f"nu_decouple_shifts must have M_steps={self.M_steps} entries"

        # ── Validate photon field lengths ─────────────────────────────────────
        assert len(self.photon_absorb_sets) == self.M_steps, \
            f"photon_absorb_sets must have M_steps={self.M_steps} entries"
        assert len(self.photon_emit_sets) == self.M_steps, \
            f"photon_emit_sets must have M_steps={self.M_steps} entries"

    @property
    def total_electrons(self) -> int:
        """Sum of |A_n| over all steps: cumulative forward electron injection."""
        return sum(len(A) for A in self.electron_sets)

    @property
    def total_electrons_ejected(self) -> int:
        """Sum of |A_n_eject| over all steps: cumulative reverse electron ejection."""
        return sum(len(A) for A in self.electron_eject_sets)

    @property
    def total_net_electrons(self) -> int:
        """Net electron flux: Σ(|A_n| − |A_n_eject|).

        Zero for a closed catalytic cycle (catalyst fully regenerated).
        """
        return self.total_electrons - self.total_electrons_ejected

    @property
    def total_cofactor_shift(self) -> int:
        """Sum of nu_n over all steps (cumulative forward shift)."""
        return sum(self.nu_shifts)

    @property
    def total_cofactor_decouple_shift(self) -> int:
        """Sum of nu_decouple_n over all steps (cumulative inverse shift)."""
        return sum(self.nu_decouple_shifts)

    @property
    def total_net_cofactor_shift(self) -> int:
        """Net cofactor phase shift: Σ(nu_n − nu_decouple_n).

        Must equal 0 (mod m) for a stoichiometrically closed cycle.
        """
        return self.total_cofactor_shift - self.total_cofactor_decouple_shift

    @property
    def phase_closure_satisfied(self) -> bool:
        """Net-flux phase closure: Σ(nu_n − nu_decouple_n) ≡ 0 (mod m).

        Generalises the original cumulative condition to bidirectional cycles.
        Always True when m=1 (trivial Z_1 group).
        """
        if self.m == 1:
            return True
        return (self.total_net_cofactor_shift % self.m) == 0

    @property
    def total_photons_absorbed(self) -> int:
        """Total number of photon absorption events across all steps: Σ|Γ_n_abs|."""
        return sum(len(g) for g in self.photon_absorb_sets)

    @property
    def total_photons_emitted(self) -> int:
        """Total number of photon emission events across all steps: Σ|Γ_n_emit|."""
        return sum(len(g) for g in self.photon_emit_sets)

    @property
    def net_photons(self) -> int:
        """Net photon count: absorbed − emitted. Zero for a closed optical cycle."""
        return self.total_photons_absorbed - self.total_photons_emitted

    @property
    def is_photo_driven(self) -> bool:
        """True when any photon absorption or emission field is non-trivial."""
        return (
            any(len(g) > 0 for g in self.photon_absorb_sets)
            or any(len(g) > 0 for g in self.photon_emit_sets)
        )

    @property
    def is_reversible_cycle(self) -> bool:
        """True when any reverse (ejection/decoupling/emission) field is non-trivial."""
        return (
            any(len(A) > 0 for A in self.electron_eject_sets)
            or any(len(P) > 0 for P in self.proton_eject_sets)
            or any(len(B) > 0 for B in self.cofactor_decouple_sets)
            or any(nu > 0 for nu in self.nu_decouple_shifts)
            or self.is_photo_driven
        )

    @property
    def n_crossings(self) -> int:
        return len(self.crossings)

    def summary(self) -> str:
        rev_tag   = " [REVERSIBLE]" if self.is_reversible_cycle else ""
        photo_tag = " [PHOTO-DRIVEN]" if self.is_photo_driven else ""
        lines = [
            f"  Mechanism : {self.name}{rev_tag}{photo_tag}",
            f"  N orbitals: {self.N_orbitals} | Steps M={self.M_steps} | "
            f"Virtual modulus m={self.m} (ℤ_{self.m})",
            f"  S_target  : {self.S_target}",
            f"  e⁻ inject : {self.total_electrons} | "
            f"e⁻ eject: {self.total_electrons_ejected} | "
            f"Net e⁻: {self.total_net_electrons}",
            f"  Σν (fwd)  : {self.total_cofactor_shift} | "
            f"Σν (inv): {self.total_cofactor_decouple_shift} | "
            f"Net Σν: {self.total_net_cofactor_shift}",
            f"  Phase closure ≡ 0 (mod {self.m}): "
            f"{'[✓] SATISFIED' if self.phase_closure_satisfied else '[✗] VIOLATED'}",
            f"  Photons abs: {self.total_photons_absorbed} | "
            f"emit: {self.total_photons_emitted} | "
            f"net: {self.net_photons} | "
            f"phi_photon={self.phi_photon:.4f} rad",
            f"  Non-adiabatic crossings: {self.n_crossings}",
            f"  Description: {self.description}",
        ]
        return "\n".join(lines)



# ==============================================================================
# PREDEFINED MECHANISM BUILDERS
# ==============================================================================

def _make_uniform_sets(n_orbitals: int, m_steps: int) -> Tuple[
    List[List[int]], List[List[int]], List[List[int]]
]:
    """Helper: distribute orbitals uniformly across steps for demo purposes."""
    block = max(1, n_orbitals // m_steps)
    A_sets, P_sets, B_sets = [], [], []
    for step in range(m_steps):
        start = (step * block) % n_orbitals
        A_sets.append([start % n_orbitals])
        P_sets.append([(start + 1) % n_orbitals])
        B_sets.append([(start + 2) % n_orbitals])
    return A_sets, P_sets, B_sets


def _build_nitrogenase_lt_m8_spec(n_orbitals: int) -> MechanismTuple:
    """Nitrogenase LT variant: M=8, m=8 (ℤ₈).

    Tests finer phase granularity (d=8 virtual register) while keeping
    identical sequential injection (1e⁻/step) and total cofactor shift (8).
    Validates CompositeVirtualShiftGate for m=8 (r=2, Case III: C_8=I) and
    assesses impact on free-energy landscape for δ_ATP minimization.

    Phase design:
        nu_n=1 per step (coprime with 8) → orbit size = 8/gcd(1,8) = 8.
        All 8 distinct ℤ₈ states are visited: k^(n)=1,2,3,4,5,6,7,0.
        Σν = 8 ≡ 0 (mod 8). Using nu=2 would collapse orbit to size 4,
        making this variant equivalent to ℤ₄ — defeating its purpose.

    Janus crossing:
        Phase condition k^(n*)=m/2=4 is satisfied after step n*=3
        (k accumulates as 1,2,3,4 after steps 0..3; phase_tracker.step()
        runs after the crossing check, so the crossing at n=3 is the step
        whose nu brings k to 4). Geometric degeneracy is placed at n=3.
    """
    N = n_orbitals
    M_steps = 8

    # PCET Fields (8e⁻, 8H⁺, 8 ATP)
    A_sets = [[n % N] for n in range(M_steps)]
    P_sets = [[(n + 1) % N] for n in range(M_steps)]
    B_sets = [[(n + 2) % N] for n in range(M_steps)]
    nu_shifts = [1] * M_steps  # nu=1 (coprime with 8) → full ℤ₈ orbit: k^(n)=1,2,3,4,5,6,7,0

    # Janus Crossing at Step 3: k^(3)=4=m/2 ✓ (phase condition satisfied)
    crossings = [(3, 0, 1, 1.6e-3)]

    return MechanismTuple(
        name          = "nitrogenase_lt_m8",
        N_orbitals    = N,
        M_steps       = M_steps,
        m             = 8,                  # ℤ₈ phase group
        S_target      = 1.5,                # S=3/2 resting spin sector
        electron_sets = A_sets,
        proton_sets   = P_sets,
        cofactor_sets = B_sets,
        nu_shifts     = nu_shifts,
        crossings     = crossings,
        phi_proton    = np.pi / 2,
        dock_orbitals = None,
        description   = (
            "Nitrogenase LT variant with ℤ₈ virtual phase group. "
            "nu_n=1 (coprime with 8) exercises all 8 ℤ₈ states: k^(n)=1,2,3,4,5,6,7,0. "
            "Janus crossing at step 3 (k^(3)=4=m/2, all three conditions satisfied). "
            "Phase closure: Σν=8 ≡ 0 (mod 8). Net e⁻: 8."
        ),
    )


def _build_nitrogenase_lt_parallel_spec(n_orbitals: int) -> MechanismTuple:
    """Nitrogenase LT variant: M=4, m=4 (ℤ₄).
    
    Tests parallel electron injection (2e⁻/step over 4 steps) vs sequential.
    Compresses the catalytic trajectory while maintaining identical total 
    stoichiometry (8e⁻, 8H⁺, 16ATP) to evaluate gate-depth scaling and 
    free-energy profile for δ_ATP minimization.
    """
    N = n_orbitals
    M_steps = 4
    
    # Parallel injection: 2 orbitals per step
    A_sets = [[n % N, (n + 2) % N] for n in range(M_steps)]
    P_sets = [[(n + 1) % N, (n + 3) % N] for n in range(M_steps)]
    # B_sets offset by +1 from A_sets: A={n,(n+2)}, B={(n+1),(n+3)} → A∩B=∅ for all N=4 steps
    B_sets = [[(n + 1) % N, (n + 3) % N] for n in range(M_steps)]
    nu_shifts = [2] * M_steps  # nu=2 → Σν=8 ≡ 0 (mod 4); k^(n)=2,0,2,0

    # Janus Crossing at Step 2: k^(2)=6%4=2=m/2 ✓
    crossings = [(2, 0, 1, 1.6e-3)]

    return MechanismTuple(
        name          = "nitrogenase_lt_parallel",
        N_orbitals    = N,
        M_steps       = M_steps,
        m             = 4,                  # ℤ₄ phase group
        S_target      = 1.5,                # S=3/2 resting spin sector
        electron_sets = A_sets,
        proton_sets   = P_sets,
        cofactor_sets = B_sets,
        nu_shifts     = nu_shifts,
        crossings     = crossings,
        phi_proton    = np.pi / 2,
        dock_orbitals = None,
        description   = (
            "Nitrogenase LT variant with parallel injection (2e⁻/step over 4 steps). "
            "nu_n=2, k^(n)=2,0,2,0 — ℤ₄ cycles between |0⟩ and |2⟩ (intentional). "
            "A_n and B_n disjoint: no qudit double-acted in same step. "
            "Phase closure: Σν=8 ≡ 0 (mod 4). Net e⁻: 8."
        ),
    )

def _build_nitrogenase_group_a_spec(n_orbitals: int) -> MechanismTuple:
    """Group A spectral proxy: M=8, m=8 (ℤ₈), ν=2 → n*=3, s=0.04090.

    Phase: k_acc = 2,4,6,8,10,12,14,16 → phase_index = 2,4,6,0,2,4,6,0 (mod 8).
    Closure: 16 ≡ 0 (mod 8) ✓.
    Janus at n=3: k_acc=8=m (first full ℤ₈ revolution). n*=m/ν-1=3 ✓.

    Covers catalog entries 1 (Mo-nitrogenase), 7 (assimilatory NR),
    14 (photocatalytic N₂ fix) — all Group A.
    """
    N = n_orbitals
    M_steps = 8
    A_sets  = [[n % N] for n in range(M_steps)]
    P_sets  = [[(n + 1) % N] for n in range(M_steps)]
    B_sets  = [[(n + 2) % N] for n in range(M_steps)]
    nu_shifts = [2] * M_steps  # ν=2 → Group A; n*=8/2-1=3
    crossings = [(3, 0, 1, 1.6e-3)]  # Janus at n=3 (k_acc=8=m)

    return MechanismTuple(
        name          = "nitrogenase_group_a",
        N_orbitals    = N,
        M_steps       = M_steps,
        m             = 8,
        S_target      = 1.5,
        electron_sets = A_sets,
        proton_sets   = P_sets,
        cofactor_sets = B_sets,
        nu_shifts     = nu_shifts,
        crossings     = crossings,
        phi_proton    = np.pi / 2,
        dock_orbitals = None,
        description   = (
            "Group A proxy: m=8, ν=2, n*=3, s=0.04090. "
            "Covers entries 1 (Mo-nitrogenase), 7 (assimilatory NR), "
            "14 (photocatalytic N₂ fix, M=8). Phase closure: 16≡0(mod 8)."
        ),
    )


def _build_nitrogenase_group_d_spec(n_orbitals: int) -> MechanismTuple:
    """Group D spectral proxy: M=12, m=12 (ℤ₁₂), ν=2 → n*=5, s=0.02743.

    Phase: k_acc = 2,4,...,24 → phase_index = 2,4,6,8,10,0,2,4,6,8,10,0 (mod 12).
    Closure: 24 ≡ 0 (mod 12) ✓.
    Janus at n=5: k_acc=12=m (first full ℤ₁₂ revolution). n*=m/ν-1=5 ✓.

    Covers catalog entries 3 (V-nitrogenase) and 13 (Cu CO₂RR) —
    cross-domain Group D spectral degeneracy.
    Use --tower-p 3 for the Iwasawa climb (m=12=4×3).
    """
    N = n_orbitals
    M_steps = 12
    A_sets  = [[n % N] for n in range(M_steps)]
    P_sets  = [[(n + 1) % N] for n in range(M_steps)]
    B_sets  = [[(n + 2) % N] for n in range(M_steps)]
    nu_shifts = [2] * M_steps  # ν=2 → Group D; n*=12/2-1=5
    crossings = [(5, 0, 1, 1.6e-3)]  # Janus at n=5 (k_acc=12=m)

    return MechanismTuple(
        name          = "nitrogenase_group_d",
        N_orbitals    = N,
        M_steps       = M_steps,
        m             = 12,
        S_target      = 1.5,
        electron_sets = A_sets,
        proton_sets   = P_sets,
        cofactor_sets = B_sets,
        nu_shifts     = nu_shifts,
        crossings     = crossings,
        phi_proton    = np.pi / 2,
        dock_orbitals = None,
        description   = (
            "Group D proxy: m=12, ν=2, n*=5, s=0.02743. "
            "Covers entries 3 (V-nitrogenase, winding (1,2)) and "
            "13 (Cu CO₂RR, winding (1,0)). Phase closure: 24≡0(mod 12)."
        ),
    )


# ---------------------------------------------------------------------------
# Individual specs — catalog entries 1, 7, 14 (Group A) and 3, 13 (Group D)
# ---------------------------------------------------------------------------

def _build_mo_nitrogenase_spec(n_orbitals: int) -> MechanismTuple:
    """Catalog entry 1 (Mo-nitrogenase, Group A): M=8, m=8, ν=2, n*=3, s=0.04090.

    Fe-Mo-S₂ proxy (72e, charge=0, spin_2S=4). Bond 2.700→2.620 Å.
    Winding (1,2): N_e=8, M_cof=16 (ATP). Janus at n=3. --tower-p 2.
    """
    N = n_orbitals
    M_steps = 8
    A_sets    = [[n % N] for n in range(M_steps)]
    P_sets    = [[(n + 1) % N] for n in range(M_steps)]
    B_sets    = [[(n + 2) % N] for n in range(M_steps)]
    nu_shifts = [2] * M_steps
    crossings = [(3, 0, 1, 1.6e-3)]

    return MechanismTuple(
        name          = "mo_nitrogenase",
        N_orbitals    = N,
        M_steps       = M_steps,
        m             = 8,
        S_target      = 1.5,
        electron_sets = A_sets,
        proton_sets   = P_sets,
        cofactor_sets = B_sets,
        nu_shifts     = nu_shifts,
        crossings     = crossings,
        phi_proton    = np.pi / 2,
        dock_orbitals = None,
        description   = (
            "Entry 1 (Mo-nitrogenase, Group A): m=8, ν=2, n*=3, s=0.04090. "
            "Fe-Mo-S₂ proxy, bond 2.700→2.620 Å. Winding (1,2), M_cof=16."
        ),
    )


def _build_femon2_trimer_spec(n_orbitals: int) -> MechanismTuple:
    """Fe–Mo–N₂ trimer proxy (Group B, m=4): M=8, m=4, ν=2, n*=1, Janus at n=4.

    4-atom convergent proxy for the N₂ activation spectral sector (Group B,
    s=0.08115) targeted by nitrogenase_femoco (22 atoms, ROHF non-convergent).
    Fe(26) + Mo(ECP28→14val) + 2×N(7) = 54e, charge=0, spin_2S=4.
    N–N elongation 1.10→1.52 Å; Fe–Mo fixed at 2.700 Å, Mo–N at 2.000 Å.
    Janus at n=4: k^(4)=10 mod 4=2=m/2 ✓ (antipodal, same as nitrogenase_femoco).
    Stoichiometry: N_e=8, M_cof=16. Phase closure: 16≡0(mod 4) ✓. --tower-p 2.
    """
    N = n_orbitals
    M_steps = 8
    A_sets    = [[n % N] for n in range(M_steps)]
    P_sets    = [[(n + 1) % N] for n in range(M_steps)]
    B_sets    = [[(n + 2) % N] for n in range(M_steps)]
    nu_shifts = [2] * M_steps
    # Janus at n=4: k_acc=10, k^(4)=10 mod 4=2=m/2 ✓ (antipodal condition)
    crossings = [(4, 0, 1, 1.6e-3)]

    return MechanismTuple(
        name          = "femon2_trimer",
        N_orbitals    = N,
        M_steps       = M_steps,
        m             = 4,
        S_target      = 1.5,
        electron_sets = A_sets,
        proton_sets   = P_sets,
        cofactor_sets = B_sets,
        nu_shifts     = nu_shifts,
        crossings     = crossings,
        phi_proton    = np.pi / 2,
        dock_orbitals = None,
        description   = (
            "Fe–Mo–N₂ trimer proxy (Group B): m=4, ν=2, n*=1, Janus at n=4 "
            "(k=m/2=2, antipodal). Convergent 4-atom proxy for N₂ activation. "
            "Targets Group B spectral sector (s=0.08115). N_e=8, M_cof=16."
        ),
    )


def _build_assimilatory_nr_spec(n_orbitals: int) -> MechanismTuple:
    """Catalog entry 7 (assimilatory nitrate reductase, Group A): M=8, m=8, ν=2, n*=3, s=0.04090.

    Mo-S₂-O₂ pterin-dithiolate proxy (62e, charge=0, spin_2S=4). Bond 2.420→2.340 Å.
    Winding (1,0): N_e=8, M_cof=0. Janus at n=3. --tower-p 2.
    """
    N = n_orbitals
    M_steps = 8
    A_sets    = [[n % N] for n in range(M_steps)]
    P_sets    = [[(n + 1) % N] for n in range(M_steps)]
    B_sets    = [[(n + 2) % N] for n in range(M_steps)]
    nu_shifts = [2] * M_steps
    crossings = [(3, 0, 1, 1.6e-3)]

    return MechanismTuple(
        name          = "assimilatory_nr",
        N_orbitals    = N,
        M_steps       = M_steps,
        m             = 8,
        S_target      = 1.5,
        electron_sets = A_sets,
        proton_sets   = P_sets,
        cofactor_sets = B_sets,
        nu_shifts     = nu_shifts,
        crossings     = crossings,
        phi_proton    = np.pi / 2,
        dock_orbitals = None,
        description   = (
            "Entry 7 (assimilatory NR, Group A): m=8, ν=2, n*=3, s=0.04090. "
            "Mo-S₂-O₂ pterin-dithiolate proxy, bond 2.420→2.340 Å. Winding (1,0), M_cof=0."
        ),
    )


def _build_photocatalytic_n2_spec(n_orbitals: int) -> MechanismTuple:
    """Catalog entry 14 (photocatalytic N₂ fixation, Group A): M=8, m=8, ν=2, n*=3, s=0.04090.

    Ti₂N₂ proxy (58e, charge=0, spin_2S=4). Ti-N bond 1.900→1.556 Å.
    Winding (1,1): N_e=8, M_cof=8 (photon). Janus at n=3. --tower-p 2.
    """
    N = n_orbitals
    M_steps = 8
    A_sets    = [[n % N] for n in range(M_steps)]
    P_sets    = [[(n + 1) % N] for n in range(M_steps)]
    B_sets    = [[(n + 2) % N] for n in range(M_steps)]
    nu_shifts = [2] * M_steps
    crossings = [(3, 0, 1, 1.6e-3)]

    return MechanismTuple(
        name          = "photocatalytic_n2",
        N_orbitals    = N,
        M_steps       = M_steps,
        m             = 8,
        S_target      = 1.5,
        electron_sets = A_sets,
        proton_sets   = P_sets,
        cofactor_sets = B_sets,
        nu_shifts     = nu_shifts,
        crossings     = crossings,
        phi_proton    = np.pi / 2,
        dock_orbitals = None,
        description   = (
            "Entry 14 (photocatalytic N₂ fix, Group A): m=8, ν=2, n*=3, s=0.04090. "
            "Ti₂N₂ proxy, Ti-N bond 1.900→1.556 Å. Winding (1,1), M_cof=8."
        ),
    )


def _build_v_nitrogenase_spec(n_orbitals: int) -> MechanismTuple:
    """Catalog entry 3 (V-nitrogenase, Group D): M=12, m=12, ν=2, n*=5, s=0.02743.

    V₂S₂ FeVco proxy (78e, charge=0, spin_2S=4). Bond 2.350→2.258 Å.
    Winding (1,2): N_e=12, M_cof=24. Janus at n=5. --tower-p 3.
    """
    N = n_orbitals
    M_steps = 12
    A_sets    = [[n % N] for n in range(M_steps)]
    P_sets    = [[(n + 1) % N] for n in range(M_steps)]
    B_sets    = [[(n + 2) % N] for n in range(M_steps)]
    nu_shifts = [2] * M_steps
    crossings = [(5, 0, 1, 1.6e-3)]

    return MechanismTuple(
        name          = "v_nitrogenase",
        N_orbitals    = N,
        M_steps       = M_steps,
        m             = 12,
        S_target      = 1.5,
        electron_sets = A_sets,
        proton_sets   = P_sets,
        cofactor_sets = B_sets,
        nu_shifts     = nu_shifts,
        crossings     = crossings,
        phi_proton    = np.pi / 2,
        dock_orbitals = None,
        description   = (
            "Entry 3 (V-nitrogenase, Group D): m=12, ν=2, n*=5, s=0.02743. "
            "V₂S₂ FeVco proxy, bond 2.350→2.258 Å. Winding (1,2), M_cof=24."
        ),
    )


def _build_cu_co2rr_spec(n_orbitals: int) -> MechanismTuple:
    """Catalog entry 13 (Cu CO₂ electroreduction, Group D): M=12, m=12, ν=2, n*=5, s=0.02743.

    Cu₃⁻ trimer proxy (88e, charge=−1, spin_2S=0). Bond 2.550→2.458 Å.
    Winding (1,0): N_e=12, M_cof=0. Janus at n=5. Cross-domain Group D degeneracy.
    --tower-p 3 (m=12=4×3).
    """
    N = n_orbitals
    M_steps = 12
    A_sets    = [[n % N] for n in range(M_steps)]
    P_sets    = [[(n + 1) % N] for n in range(M_steps)]
    B_sets    = [[(n + 2) % N] for n in range(M_steps)]
    nu_shifts = [2] * M_steps
    crossings = [(5, 0, 1, 1.6e-3)]

    return MechanismTuple(
        name          = "cu_co2rr",
        N_orbitals    = N,
        M_steps       = M_steps,
        m             = 12,
        S_target      = 0.0,
        electron_sets = A_sets,
        proton_sets   = P_sets,
        cofactor_sets = B_sets,
        nu_shifts     = nu_shifts,
        crossings     = crossings,
        phi_proton    = np.pi / 2,
        dock_orbitals = None,
        description   = (
            "Entry 13 (Cu CO₂RR, Group D): m=12, ν=2, n*=5, s=0.02743. "
            "Cu₃⁻ trimer proxy, bond 2.550→2.458 Å. Winding (1,0), M_cof=0."
        ),
    )


def _build_haber_bosch_spec(n_orbitals: int) -> MechanismTuple:
    """Haber-Bosch N₂ activation on Fe₂S₂N₂ proxy: M=8, m=4 (ℤ₄), Group B.

    Stoichiometry: N₂ + 8H → 2NH₃ + H₂  (Case III topological lift;
    same argument as nitrogenase — 4∤6 electrons, so lift to N_e=8 with
    surplus 2H expelled as H₂).  M_cof=0: no ATP — thermal drive (high T, P).
    Winding: (N_e/m, M_cof/m) = (2, 0).

    ν_n = 2 per step → Σν = 16 ≡ 0 (mod 4) ✓.
    Janus at n=4: N–N bond length 1.34 Å, the dissociative-chemisorption
    transition-state midpoint.  Phase at n=4: k^(4) = 2·5 mod 4 = 2 = m/2 ✓.
    n* = m/ν − 1 = 4/2 − 1 = 1  →  Group B  (s = 0.08115).
    Tower prime p = 2  (m = 4 = 2²).
    """
    N = n_orbitals
    M_steps  = 8
    m        = 4

    A_sets    = [[n % N]       for n in range(M_steps)]
    P_sets    = [[(n + 1) % N] for n in range(M_steps)]
    B_sets    = [[(n + 2) % N] for n in range(M_steps)]
    nu_shifts = [2] * M_steps   # ν=2; n* = m/ν − 1 = 1 → Group B

    # Janus at n=4: k^(4) = 2·5 mod 4 = 2 = k* = m/2
    crossings = [(4, 0, 1, 1.6e-3)]

    return MechanismTuple(
        name          = "haber_bosch",
        N_orbitals    = N,
        M_steps       = M_steps,
        m             = m,
        S_target      = 1.5,               # high-spin Fe₂S₂N₂, S = 3/2
        electron_sets = A_sets,
        proton_sets   = P_sets,
        cofactor_sets = B_sets,
        nu_shifts     = nu_shifts,
        crossings     = crossings,
        phi_proton    = np.pi / 2,
        dock_orbitals = None,
        description   = (
            "Haber-Bosch N₂ activation on Fe₂S₂N₂ proxy (Case III, m=4, Group B). "
            "ν=2 per step; Σν=16≡0 (mod 4). Janus at n=4 (N–N=1.34 Å TS). "
            "n*=1; Group B spectral class (s=0.08115). Tower prime p=2."
        ),
    )

def _build_nitrogenase_fe4s4_spec(n_orbitals: int) -> MechanismTuple:
    """Nitrogenase Fe4S4 mechanism: M=8, m=4 (Z4), 1 Janus at n=4.
    
    Scales the LT cycle validation to a chemically realistic [Fe4S4] cubane.
    Geometry sequence simulates expansion of the cubane core upon progressive reduction.
    """
    N = n_orbitals
    M_steps = 8
    
    # LT-style electron/proton/cofactor assignment
    A_sets = [[n % N] for n in range(M_steps)]
    P_sets = [[(n + 1) % N] for n in range(M_steps)]
    B_sets = [[(n + 2) % N] for n in range(M_steps)]
    nu_shifts = [2] * M_steps  # 2 ATP equivalents per step
    
    # Janus crossing at E4→E5 (step index 4)
    crossings = [(4, 0, 1, 1.6e-3)]
    
    return MechanismTuple(
        name          = "nitrogenase_fe4s4",
        N_orbitals    = N,
        M_steps       = M_steps,
        m             = 4,
        S_target      = 0.0,
        electron_sets = A_sets,
        proton_sets   = P_sets,
        cofactor_sets = B_sets,
        nu_shifts     = nu_shifts,
        crossings     = crossings,
        phi_proton    = np.pi / 2,
        dock_orbitals = None,
        description   = (
            "Lowe-Thorneley nitrogenase LT cycle scaled to [Fe4S4] cubane. "
            "E0→E8 via 8 sequential PCET steps. Janus non-adiabatic "
            "crossing at E4→E5. Phase closure: Σν=16 ≡ 0 (mod 4)."
        ),
    )


def _build_nitrogenase_femoco_spec(n_orbitals: int) -> MechanismTuple:
    """Full Nitrogenase FeMo-cofactor Mechanism: M=8, m=4 (Z4), Janus at n=4.
    
    Aligned with algebraic PCET framework. No physical H reservoir required.
    Models the complete E0→E8 trajectory on the Fe7MoS9C cluster proxy with
    progressive N2 bond elongation (1.10 → 1.52 Å). 
    """
    N = n_orbitals
    M_steps = 8

    # PCET Fields (8e⁻, 8H⁺, 16 ATP)
    A_sets = [[n % N] for n in range(M_steps)]
    P_sets = [[(n + 1) % N] for n in range(M_steps)]
    B_sets = [[(n + 2) % N] for n in range(M_steps)]
    nu_shifts = [2] * M_steps  # 2 ATP equivalents per step

    # Janus Crossing at Step 4 (E4→E5: H2 release / N2 binding onset)
    crossings = [(4, 0, 1, 1.6e-3)]

    return MechanismTuple(
        name          = "nitrogenase_femoco",
        N_orbitals    = N,
        M_steps       = M_steps,
        m             = 4,                  # ℤ₄ phase group
        S_target      = 1.5,                # S=3/2 resting spin sector
        electron_sets = A_sets,
        proton_sets   = P_sets,
        cofactor_sets = B_sets,
        nu_shifts     = nu_shifts,
        crossings     = crossings,
        phi_proton    = np.pi / 2,
        dock_orbitals = None,
        description   = (
            "Full FeMo-cofactor LT cycle with N2 substrate. "
            "Stoichiometry enforced algebraically via A_n/P_n fields. "
            "N2 activation modeled via progressive bond elongation. "
            "Phase closure: Σν=16 ≡ 0 (mod 4). Net e⁻: 8."
        ),
    )


def _build_ethylene_epoxidation_spec(n_orbitals: int) -> MechanismTuple:
    """Ethylene Epoxidation on Ag3 Proxy: M=4, m=1 (Z1), adiabatic.
    
    Models the surface-mediated addition of an oxygen adatom to ethylene.
    The sequence follows: Reactants -> Early TS -> Oxametallacycle -> Ethylene Oxide.
    
    Trivial Z_1 phase group (adiabatic structural evolution without external 
    electron injection). Explicitly tracks surface coordination via B_n.
    """
    N = n_orbitals
    M_steps = 4
    
    # Adiabatic evolution: no external electron/proton injection per step.
    electron_sets = [[] for _ in range(M_steps)]
    proton_sets   = [[] for _ in range(M_steps)]
    
    # FIX: Explicit indexing matches mqe_datasets_dev.py surface coordination tracking
    cofactor_sets = [[n % N] for n in range(M_steps)]
    
    nu_shifts     = [2] * M_steps  # ν=2, Group B: n*=4/2-1=1

    return MechanismTuple(
        name          = "ethylene_epoxidation",
        N_orbitals    = N,
        M_steps       = M_steps,
        m             = 4,              # Z₄ — Group B spectral class
        S_target      = 0.5,            # doublet (81e, odd count with ECP28)
        electron_sets = electron_sets,
        proton_sets   = proton_sets,
        cofactor_sets = cofactor_sets,
        nu_shifts     = nu_shifts,
        crossings     = [(1, 0, 1, 1.6e-3)],  # Janus at n=1: oxametallacycle TS
        phi_proton    = np.pi / 2,
        dock_orbitals = None,
        description   = (
            "Ethylene epoxidation on Ag3 proxy cluster. "
            "Simulates Oxametallacycle formation and closure to EO. "
            "Trivial Z_1 phase. Designed for CAS(4,4) validation of surface chemistry. "
            "Tracks surface coordination via B_n."
        ),
    )

def _build_thymine_dimer_spec(n_orbitals: int) -> MechanismTuple:
    """Thymine Dimer Proxy ([2+2] cycloaddition): M=6, m=1 (Z1), 1 Janus at n=3.
    
    Models the photochemical dimerization of two stacked ethylene molecules.
    Tracks inter-planar distance from 3.4 Å (vdW) to 1.5 Å (cyclobutane).
    Closed system (no external e⁻/H⁺ injection), singlet manifold (S=0).
    Conical intersection proxy flagged at step 3 (d ≈ 2.2 Å).
    """
    N = n_orbitals
    M_steps = 6
    
    # Closed photochemical system: no external charge/cofactor transfer
    A_sets = [[] for _ in range(M_steps)]
    P_sets = [[] for _ in range(M_steps)]
    B_sets = [[] for _ in range(M_steps)]
    nu_shifts = [2] * M_steps  # ν=2, algebraic π→π* excitation depth; Group B n*=1

    # Janus crossing proxy at step index 3
    crossings = [(3, 0, 1, 1.6e-3)]

    return MechanismTuple(
        name          = "thymine_dimer_proxy",
        N_orbitals    = N,
        M_steps       = M_steps,
        m             = 4,                  # Z₄ — Group B spectral class
        S_target      = 0.0,                # Singlet manifold (S₁ → S₀)
        electron_sets = A_sets,
        proton_sets   = P_sets,
        cofactor_sets = B_sets,
        nu_shifts     = nu_shifts,
        crossings     = crossings,
        phi_proton    = np.pi / 2,
        dock_orbitals = None,
        description   = (
            "Thymine dimer [2+2] cycloaddition proxy. Two stacked ethylene molecules. "
            "Tracks approach from 3.4Å to 1.5Å. Features a photochemical conical "
            "intersection at step 3. Designed for CAS(4,4) validation."
        ),
    )

def _build_anammox_proxy_spec(n_orbitals: int) -> MechanismTuple:
    """Anammox Hydrazine Synthase Proxy: M=4, m=4 (ℤ₄), oxidative PCET.
    
    Models oxidative N-N coupling of two NH₂ fragments on a single Fe center 
    to form hydrazine (N₂H₄). Tests framework handling of oxidative pathways 
    where electrons are extracted from the substrate into the cofactor.
    
    Phase closure: Σ ν_n = 4 ≡ 0 (mod 4) ✓
    Net electron flux: -4 (oxidative extraction) ✓
    """
    N = n_orbitals
    M_steps = 4
    m = 4

    # ── Forward fields remain empty for oxidative pathway ─────────────
    electron_sets = [[] for _ in range(M_steps)]
    proton_sets   = [[] for _ in range(M_steps)]
    cofactor_sets = [[(n + 2) % N] for n in range(M_steps)]
    nu_shifts     = [1] * M_steps

    # ── Reverse fields handle the actual electron/proton ejection ─────
    electron_eject_sets    = [[n % N] for n in range(M_steps)]
    proton_eject_sets      = [[(n + 1) % N] for n in range(M_steps)]
    cofactor_decouple_sets = [[] for _ in range(M_steps)]
    nu_decouple_shifts     = [0] * M_steps

    return MechanismTuple(
        name                     = "anammox_proxy",
        N_orbitals               = N,
        M_steps                  = M_steps,
        m                        = m,
        S_target                 = 1.5,            # high-spin Fe(II), S=2 (2S=4)
        electron_sets            = electron_sets,
        proton_sets              = proton_sets,
        cofactor_sets            = cofactor_sets,
        nu_shifts                = nu_shifts,
        electron_eject_sets      = electron_eject_sets,
        proton_eject_sets        = proton_eject_sets,
        cofactor_decouple_sets   = cofactor_decouple_sets,
        nu_decouple_shifts       = nu_decouple_shifts,
        crossings                = [(2, 0, 1, 1.6e-3)],  # Janus at n=2: N–N midpoint
        phi_proton               = np.pi / 2,
        dock_orbitals            = None,
        description              = (
            "Anammox proxy: Oxidative N-N coupling on Fe center. "
            "Simulates hydrazine formation. 4-step oxidative PCET. "
            "Tracks 4e⁻/4H⁺ ejection via reverse fields. "
            "Phase closure: Σν=4 ≡ 0 (mod 4). Net e⁻ flux: -4."
        ),
    )


def _build_atp_hydrolysis_proxy_spec(n_orbitals: int) -> MechanismTuple:
    """ATP Hydrolysis Proxy: M=4, m=4 (ℤ₄), Group C.

    Models H₂O nucleophilic attack on H₂PO₄H (phosphoric acid proxy).
    Tracks P–O(water) distance 3.0→1.8 Å through the trigonal bipyramidal TS.

    M=5 (original) is incompatible with Z_m (M=5 prime; gcd(5,4)=1).
    Changed to M=4. nu_n=1: one P–O bond event per step (internal proton relay;
    mol.charge=0 throughout). Group C: n*=4/1−1=3, s=0.04135.
    Janus at n=2 (~2.2 Å): trigonal bipyramidal TS.
    Phase closure: 4×1=4 ≡ 0 (mod 4) ✓.
    """
    N = n_orbitals
    M_steps = 4

    A_sets = [[] for _ in range(M_steps)]             # no external electron injection
    P_sets = [[n % N] for n in range(M_steps)]        # protonation of Pi leaving group
    B_sets = [[(n + 1) % N] for n in range(M_steps)]  # phosphate coordination

    nu_shifts = [1] * M_steps   # ν=1, Group C: n*=4/1-1=3

    return MechanismTuple(
        name          = "atp_hydrolysis_proxy",
        N_orbitals    = N,
        M_steps       = M_steps,
        m             = 4,              # Z₄ — Group C spectral class
        S_target      = 0.0,            # singlet (50e, charge=0)
        electron_sets = A_sets,
        proton_sets   = P_sets,
        cofactor_sets = B_sets,
        nu_shifts     = nu_shifts,
        crossings     = [(2, 0, 1, 1.6e-3)],  # Janus at n=2: TBP TS, P–O=2.2 Å
        phi_proton    = np.pi / 2,
        dock_orbitals = None,
        description   = (
            "ATP hydrolysis proxy: H₂O + H₂PO₄H → TBP-TS → H₃PO₄. "
            "Group C: m=4, ν=1, n*=3, Janus at n=2 (P–O=2.2 Å). "
            "Internal proton relay; mol charge fixed at 0."
        ),
    )

def _build_cyp450_metabolism_spec(n_orbitals: int) -> MechanismTuple:
    """Cytochrome P450 Metabolism: M=6, m=2 (Z2).
    
    CATEGORY B (Active PCET Simulation):
    Models the formation of the reactive ferryl-oxo 'Compound I' (Fe⁴⁺=O)
    and the subsequent oxygen insertion (hydroxylation) of a substrate.
    
    CORRECTED 2e⁻ / 2H⁺ Stoichiometry:
    0. Fe(III) + e⁻ → Fe(II)                     (1st Reduction)
    1. Fe(II) + O2 → Fe(II)-O2                   (O2 Binding)
    2. Fe(II)-O2 + e⁻ + H⁺ → Fe(III)-OOH         (2nd Reduction & 1st Protonation)
    3. Fe(III)-OOH + H⁺ → Fe(IV)=O + H2O         (2nd Protonation / O-O Cleavage)
    4. Fe(IV)=O + Sub → Fe(IV)=O···Sub           (Reaction Prep)
    5. Fe(IV)=O···Sub → Fe(III) + Sub-OH         (Rebound & Regeneration)
    """
    N = n_orbitals
    M_steps = 6

    # ── Forward PCET Fields ──────────────────────────────────────
    A_sets = []
    P_sets = []
    B_sets = []
    nu_shifts = []

    for n in range(M_steps):
        # Electron injection at steps 0 and 2
        a_n = [n % N] if n in [0, 2] else []
        A_sets.append(a_n)

        # Proton injection at steps 2 and 3
        p_n = [(n + 1) % N] if n in [2, 3] else []
        P_sets.append(p_n)

        # Cofactor coupling at all steps
        b_n = [(n + 2) % N]
        B_sets.append(b_n)

        # Phase shift at steps 0 and 3 (nu=2 each for Z₄ closure: Σ=4≡0 mod 4)
        nu = 2 if n in [0, 3] else 0
        nu_shifts.append(nu)

    # ── Reverse PCET Fields (Cycle Reset at Step 5) ──────────────
    A_eject_sets = [[] for _ in range(M_steps)]
    P_eject_sets = [[] for _ in range(M_steps)]
    B_decouple_sets = [[] for _ in range(M_steps)]
    nu_decouple_shifts = [0] * M_steps

    # At step 5: Eject 2e- and 2H+ to close the cycle.
    # Slicing [:N] handles cases where N is small (e.g., N=1, N=2) to prevent
    # duplicate indices (e.g., if N=2, [0, 2%2] becomes [0, 0] -> sliced to [0]).
    idx = 5
    A_eject_sets[idx] = [0, 2 % N][:N]
    P_eject_sets[idx] = [3 % N, 4 % N][:N]
    B_decouple_sets[idx] = [(idx + 2) % N]
    
    # Reset phase by 4 (Σnu_couple=4) to restore Z₄ accumulator to 0.
    nu_decouple_shifts[idx] = 4

    # ── Non-Adiabatic Crossing ───────────────────────────────────
    # O-O cleavage at step 3 (Janus crossing).
    crossings = [(3, 0, 1, 2.1e-3)]

    return MechanismTuple(
        name                    = "cyp450_metabolism",
        N_orbitals              = N,
        M_steps                 = M_steps,
        m                       = 4,                  # Z₄ — 4|m required for spectral selectivity
        S_target                = 1.0,                # Triplet surface (Compound I)
        electron_sets           = A_sets,
        proton_sets             = P_sets,
        cofactor_sets           = B_sets,
        nu_shifts               = nu_shifts,
        electron_eject_sets     = A_eject_sets,
        proton_eject_sets       = P_eject_sets,
        cofactor_decouple_sets  = B_decouple_sets,
        nu_decouple_shifts      = nu_decouple_shifts,
        crossings               = crossings,
        phi_proton              = np.pi / 2,
        dock_orbitals           = None,
        description             = (
            "Cytochrome P450 metabolism proxy. "
            "Category B: Active PCET. Accurately tracks 2e⁻/2H⁺ flux. "
            "Simulates Compound I formation via proximal thiolate push. "
            "Verification: Stoichiometry exactly net-zeros at cycle end. "
            "Janus crossing at O-O cleavage (step 3)."
        ),
    )

def _build_hydrogenase_oxidation_spec(n_orbitals: int) -> MechanismTuple:
    """Hydrogenase Oxidation (H2 → 2H⁺ + 2e⁻): M=2, m=1 (Z1 trivial).
    
    CATEGORY B (Active PCET Simulation):
    Models the oxidative pathway. By populating the `_eject` fields, we trigger
    the inverse gate compilation in the MQE pipeline. This mechanism effectively 
    reverses the electron flux compared to the `hydrogenase` reduction spec.
    """
    N = n_orbitals
    M_steps = 2
    
    # Forward fields remain empty (oxidative pathway has no net injection)
    electron_sets = [[] for _ in range(M_steps)]
    proton_sets   = [[] for _ in range(M_steps)]
    cofactor_sets = [[] for _ in range(M_steps)]
    nu_shifts     = [0] * M_steps
    
    # ── REVERSE FIELDS (Active Oxidation Flux) ────────────────────────
    # Explicitly map ejection to match dataset's A_n_eject and P_n_eject
    electron_eject_sets    = [[n % N] for n in range(M_steps)]
    proton_eject_sets      = [[(n + 1) % N] for n in range(M_steps)]
    cofactor_decouple_sets = [[] for _ in range(M_steps)]
    nu_decouple_shifts     = [0] * M_steps
    
    return MechanismTuple(
        name                    = "hydrogenase_oxidation",
        N_orbitals              = N,
        M_steps                 = M_steps,
        m                       = 1,                  # ℤ₁ trivial phase group
        S_target                = 0.0,
        electron_sets           = electron_sets,
        proton_sets             = proton_sets,
        cofactor_sets           = cofactor_sets,
        nu_shifts               = nu_shifts,
        electron_eject_sets     = electron_eject_sets,
        proton_eject_sets       = proton_eject_sets,
        cofactor_decouple_sets  = cofactor_decouple_sets,
        nu_decouple_shifts      = nu_decouple_shifts,
        crossings               = [],                 # Adiabatic oxidation
        phi_proton              = np.pi / 2,
        dock_orbitals           = None,
        description             = (
            "[FeFe]-hydrogenase: H2 → 2H⁺ + 2e⁻ (Oxidation). "
            "Category B: Active PCET Simulation. "
            "Algebraically tracks oxidative flux via A_n_eject/P_n_eject. "
            "Symmetric to the reduction mechanism. Net e⁻ flux: -2."
        ),
    )

def _build_rnr_radical_proxy_spec(n_orbitals: int) -> MechanismTuple:
    """Ribonucleotide Reductase (RNR) Radical PCET Proxy: M=4, m=4 (ℤ₄), Group C.

    Models thiyl radical H-atom transfer (HAT) from C3'-H to Cys-S• on a
    ribose-like scaffold, followed by β-elimination (C-O cleavage).
    nu_n=1 tracks the radical electron hop per step algebraically.
    mol.charge=0 throughout (internal HAT; no external injection).
    Group C: n*=4/1−1=3. Janus at n=1: S···H···C TS (S–H=1.6 Å).
    Phase closure: 4×1=4 ≡ 0 (mod 4) ✓.
    """
    N = n_orbitals
    M_steps = 4

    electron_sets = [[] for _ in range(M_steps)]  # no external injection
    proton_sets   = [[] for _ in range(M_steps)]
    cofactor_sets = [[] for _ in range(M_steps)]
    nu_shifts     = [1] * M_steps   # ν=1, Group C: n*=3

    return MechanismTuple(
        name          = "rnr_radical_proxy",
        N_orbitals    = N,
        M_steps       = M_steps,
        m             = 4,                  # Z₄ — Group C spectral class
        S_target      = 0.5,                # open-shell doublet (63e, 2S=1)
        electron_sets = electron_sets,
        proton_sets   = proton_sets,
        cofactor_sets = cofactor_sets,
        nu_shifts     = nu_shifts,
        crossings     = [(1, 0, 1, 1.6e-3)],  # Janus at n=1: S···H···C HAT TS
        phi_proton    = np.pi / 2,
        dock_orbitals = None,
        description   = (
            "RNR radical PCET proxy: thiyl HAT on ribose scaffold. "
            "Group C: m=4, ν=1, n*=3, Janus at n=1 (S–H=1.6 Å). "
            "Internal radical migration; mol charge fixed at 0."
        ),
    )


def _build_psii_photo_spec(n_orbitals: int) -> MechanismTuple:
    """Photosystem II Kok S-state cycle with explicit photon absorption fields.
    
    UPGRADED: Category B (Active PCET Simulation).
    Models the 4-step Kok cycle (S0→S1→S2→S3) driven by P680 photon absorption.
    Explicitly tracks photon flux (Γ_n_abs) alongside sequential e⁻/H⁺ extraction.
    """
    N = n_orbitals
    M_steps = 4
    m = 4
    phi_P680 = 0.067 * 0.02  # P680 photon energy (Ha) × dt (Ha⁻¹)

    # ── Explicit indexing matching mqe_datasets_dev.py exactly ─────────────
    # Guarantees orbital routing parity for ALL N, not just N % 4 == 0
    electron_sets = [[n % N] for n in range(M_steps)]
    proton_sets   = [[(n + 1) % N] for n in range(M_steps)]
    cofactor_sets = [[(n + 2) % N] for n in range(M_steps)]
    nu_shifts     = [1] * M_steps

    # Photon absorption at the same orbital receiving the electron extraction
    photon_absorb_sets = [[electron_sets[n][0]] for n in range(M_steps)]
    photon_emit_sets   = [[] for _ in range(M_steps)]

    return MechanismTuple(
        name                = "psii_photo",
        N_orbitals          = N,
        M_steps             = M_steps,
        m                   = m,
        S_target            = 0.0,
        electron_sets       = electron_sets,
        proton_sets         = proton_sets,
        cofactor_sets       = cofactor_sets,
        nu_shifts           = nu_shifts,
        photon_absorb_sets  = photon_absorb_sets,
        photon_emit_sets    = photon_emit_sets,
        phi_photon          = phi_P680,
        crossings           = [(2, 0, 1, 1.6e-3)],  # Janus at S2: Mn(III)/Mn(IV) inversion
        phi_proton          = np.pi / 2,
        dock_orbitals       = None,
        description         = (
            "Photosystem II Kok S-state cycle with explicit P680 photon absorption. "
            "4 sequential photooxidations of Mn₄CaO₅. Each step: one P680 photon "
            f"(phi={phi_P680:.5f} rad, ~1.82 eV) absorbed, one e⁻ extracted, "
            "one H⁺ released. Phase closure: Σν=4 ≡ 0 (mod 4). "
            "Photon balance: 4 absorbed, 0 emitted (pump-only cycle)."
        ),
    )


def _build_nitrogenase_closed_loop_spec(n_orbitals: int) -> MechanismTuple:
    """Nitrogenase LT closed-loop: M=8 forward + M=8 reverse = 16-step full cycle.
    
    Demonstrates the complete catalytic machine: 8 forward PCET steps reduce
    N2 (E0→E8), then 8 reverse steps regenerate the resting state (E8→E0).
    The net electron flux and net phase shift are both zero — the catalyst
    is fully recovered and no virtual register state is leaked.

    Stoichiometric invariants (Theorem 2 net-flux extension):
        Σ(ν_n − ν†_n) = 16 − 16 = 0  ≡ 0 (mod 4)  ✓
        Σ(|A_n| − |A_n_eject|) = 8 − 8 = 0           ✓

    This is the definitive "closed-loop test case" described in the
    reversibility theory: initialize S0, forward-transform to S1,
    then apply exact inverses to recover S0' ≈ S0.
    """
    N       = n_orbitals
    M_fwd   = 8
    M_total = 16  # 8 forward + 8 reverse

    # Forward half: LT-style 8-step PCET (same as nitrogenase_fe4s4)
    A_fwd   = [[n % N]           for n in range(M_fwd)]
    P_fwd   = [[(n + 1) % N]     for n in range(M_fwd)]
    B_fwd   = [[(n + 2) % N]     for n in range(M_fwd)]
    nu_fwd  = [2] * M_fwd

    # Reverse half: exact inverses applied in reverse order (steps 8-15)
    # A_eject mirrors A_fwd; forward orbitals are ejected
    # Matches dataset's rn_rev = 7 - (n - 8) exactly
    A_rev   = [A_fwd[M_fwd - 1 - i] for i in range(M_fwd)]
    P_rev   = [P_fwd[M_fwd - 1 - i] for i in range(M_fwd)]
    B_rev   = [B_fwd[M_fwd - 1 - i] for i in range(M_fwd)]
    nu_rev  = [2] * M_fwd

    # Concatenate forward + reverse into a 16-step mechanism
    electron_sets           = A_fwd + [[] for _ in range(M_fwd)]
    proton_sets             = P_fwd + [[] for _ in range(M_fwd)]
    cofactor_sets           = B_fwd + [[] for _ in range(M_fwd)]
    nu_shifts               = nu_fwd + [0] * M_fwd

    electron_eject_sets     = [[] for _ in range(M_fwd)] + A_rev
    proton_eject_sets       = [[] for _ in range(M_fwd)] + P_rev
    cofactor_decouple_sets  = [[] for _ in range(M_fwd)] + B_rev
    nu_decouple_shifts      = [0] * M_fwd + nu_rev

    # FIX: Janus crossings at n=4 (forward E4→E5) AND n=11 (reverse E5→E4)
    # Microscopic reversibility requires the reverse crossing to undo the forward one
    crossings = [(4, 0, 1, 1.6e-3), (11, 0, 1, 1.6e-3)]

    return MechanismTuple(
        name                    = "nitrogenase_closed_loop",
        N_orbitals              = N,
        M_steps                 = M_total,
        m                       = 4,
        S_target                = 1.5,                # FIX: S=3/2 matches Fe-S cluster physics
        electron_sets           = electron_sets,
        proton_sets             = proton_sets,
        cofactor_sets           = cofactor_sets,
        nu_shifts               = nu_shifts,
        electron_eject_sets     = electron_eject_sets,
        proton_eject_sets       = proton_eject_sets,
        cofactor_decouple_sets  = cofactor_decouple_sets,
        nu_decouple_shifts      = nu_decouple_shifts,
        crossings               = crossings,
        phi_proton              = np.pi / 2,
        dock_orbitals           = None,
        description             = (
            "Nitrogenase LT full-cycle closed-loop: 8 forward (E0→E8) + "
            "8 reverse (E8→E0) PCET steps. Net e⁻ flux = 0. "
            "Net phase shift = 0 (mod 4). Definitive catalyst regeneration "
            "benchmark — validates circuit uncomputation and microscopic "
            "reversibility of the MQE gate set G(M). "
            "Janus crossings at steps 4 (fwd) and 11 (rev, undoes step 4)."
        ),
    )


def _build_reversible_quinone_spec(n_orbitals: int) -> MechanismTuple:
    """Reversible Quinone/QH2 Redox Buffer: M=6, m=2 (Z2).
    
    Demonstrates full bidirectional PCET: Q → QH₂ → Q.
    Tests net-flux closure: Σe_in = Σe_out, Σν_couple = Σν_decouple.
    Forward steps (0-2) inject electrons; reverse steps (3-5) eject them.
    """
    N = n_orbitals
    M_steps = 6
    m = 4  # Z₄ — Group B: n*=4/2-1=1, Janus at n=3 (bidirectional turnaround)

    # Initialize lists
    A_sets = []
    P_sets = []
    B_sets = []
    nu_shifts = []

    A_eject_sets = []
    P_eject_sets = []
    B_decouple_sets = []
    nu_decouple_shifts = []

    for n in range(M_steps):
        if n < 3:
            # Forward: Reduction + Cofactor Coupling; nu=2 for Z₄ phase accumulation
            A_sets.append([n % N])
            P_sets.append([(n + 1) % N])
            B_sets.append([(n + 2) % N])
            nu_shifts.append(2)

            A_eject_sets.append([])
            P_eject_sets.append([])
            B_decouple_sets.append([])
            nu_decouple_shifts.append(0)
        else:
            # Reverse: Oxidation + Decoupling; nu_decouple=2 restores k_acc→0
            A_sets.append([])
            P_sets.append([])
            B_sets.append([])
            nu_shifts.append(0)

            A_eject_sets.append([n % N])
            P_eject_sets.append([(n + 1) % N])
            B_decouple_sets.append([(n + 2) % N])
            nu_decouple_shifts.append(2)

    return MechanismTuple(
        name          = "reversible_quinone",
        N_orbitals    = N,
        M_steps       = M_steps,
        m             = m,
        S_target      = 0.0,
        electron_sets = A_sets,
        proton_sets   = P_sets,
        cofactor_sets = B_sets,
        nu_shifts     = nu_shifts,
        electron_eject_sets     = A_eject_sets,
        proton_eject_sets       = P_eject_sets,
        cofactor_decouple_sets  = B_decouple_sets,
        nu_decouple_shifts      = nu_decouple_shifts,
        crossings     = [(3, 0, 1, 1.6e-3)],  # Janus at n=3: bidirectional turnaround
        phi_proton    = np.pi / 2,
        dock_orbitals = None,
        description   = (
            "Reversible Quinone/QH2 cycle. Full bidirectional PCET. "
            "Net electron/phase flux returns to zero. "
            "Tests CofactorDecoupling and ElectronEject dispatch."
        ),
    )

def _build_methanogenesis_proxy_spec(n_orbitals: int) -> MechanismTuple:
    """Methanogenesis Ni-CO-H2 proxy: M=8, m=4 (Z4), Group B."""
    N = n_orbitals
    M_steps = 8
    A_sets = [[n % N] for n in range(M_steps)]
    P_sets = [[(n + 1) % N] for n in range(M_steps)]
    B_sets = [[(n + 2) % N] for n in range(M_steps)]
    nu_shifts = [2] * M_steps
    crossings = [(4, 0, 1, 1.6e-3)]
    
    return MechanismTuple(
        name="methanogenesis_proxy",
        N_orbitals=N, 
        M_steps=M_steps, 
        m=4, 
        S_target=0.0,
        electron_sets=A_sets, 
        proton_sets=P_sets, 
        cofactor_sets=B_sets,
        nu_shifts=nu_shifts, 
        crossings=crossings, 
        phi_proton=np.pi/2,
        description="Methanogenesis Ni-CO-H2 proxy. 8-step PCET."
    )

def _build_mo_nitrogenase_m4_spec(n_orbitals: int) -> MechanismTuple:
    """Catalog entry 2 (Mo-nitrogenase oldform, Group B): M=8, m=4, ν=2, n*=1, s=0.08115.
    Same Fe-Mo-S₂ proxy as entry 1 but at coarser ℤ₄ register.
    Winding (2,4): N_e=8, M_cof=16. Janus at n=1. --tower-p 2.
    """
    N = n_orbitals
    M_steps = 8
    A_sets    = [[n % N] for n in range(M_steps)]
    P_sets    = [[(n + 1) % N] for n in range(M_steps)]
    B_sets    = [[(n + 2) % N] for n in range(M_steps)]
    nu_shifts = [2] * M_steps
    crossings = [(1, 0, 1, 1.6e-3)]  # Janus at n=1 (coarser register)
    return MechanismTuple(
        name="mo_nitrogenase_m4", 
        N_orbitals=N, 
        M_steps=M_steps, 
        m=4,
        S_target=1.5, 
        electron_sets=A_sets, 
        proton_sets=P_sets,
        cofactor_sets=B_sets, 
        nu_shifts=nu_shifts, 
        crossings=crossings,
        phi_proton=np.pi/2, 
        dock_orbitals=None,
        description=(
            "Entry 2 (Mo-nitrogenase oldform, Group B): m=4, ν=2, n*=1, s=0.08115. "
            "Same Fe-Mo-S₂ proxy as entry 1 at coarser ℤ₄ register. "
            "Winding (2,4), M_cof=16. Janus at n=1."
        ),
    )

def _build_v_nitrogenase_m4_spec(n_orbitals: int) -> MechanismTuple:
    """Catalog entry 4 (V-nitrogenase oldform, Group B): M=12, m=4, ν=2, n*=1, s=0.08115.
    Same V₂S₂ FeVco proxy as entry 3 but at coarser ℤ₄ register.
    Winding (3,6): N_e=12, M_cof=24. Janus at n=1. --tower-p 2.
    """
    N = n_orbitals
    M_steps = 12
    A_sets    = [[n % N] for n in range(M_steps)]
    P_sets    = [[(n + 1) % N] for n in range(M_steps)]
    B_sets    = [[(n + 2) % N] for n in range(M_steps)]
    nu_shifts = [2] * M_steps
    crossings = [(1, 0, 1, 1.6e-3)]
    return MechanismTuple(
        name="v_nitrogenase_m4", 
        N_orbitals=N, 
        M_steps=M_steps, 
        m=4,
        S_target=1.5, 
        electron_sets=A_sets, 
        proton_sets=P_sets,
        cofactor_sets=B_sets, 
        nu_shifts=nu_shifts, 
        crossings=crossings,
        phi_proton=np.pi/2, 
        dock_orbitals=None,
        description=(
            "Entry 4 (V-nitrogenase oldform, Group B): m=4, ν=2, n*=1, s=0.08115. "
            "Same V₂S₂ FeVco proxy as entry 3 at coarser ℤ₄ register. "
            "Winding (3,6), M_cof=24. Janus at n=1."
        ),
    )

def _build_assimilatory_nr_m4_spec(n_orbitals: int) -> MechanismTuple:
    """Catalog entry 8 (assimilatory NR oldform, Group B): M=8, m=4, ν=2, n*=1, s=0.08115.
    Same Mo-S₂-O₂ pterin-dithiolate proxy as entry 7 at coarser ℤ₄ register.
    Winding (2,0): N_e=8, M_cof=0. Janus at n=1 (NO₂⁻ intermediate). --tower-p 2.
    """
    N = n_orbitals
    M_steps = 8
    A_sets    = [[n % N] for n in range(M_steps)]
    P_sets    = [[(n + 1) % N] for n in range(M_steps)]
    B_sets    = [[(n + 2) % N] for n in range(M_steps)]
    nu_shifts = [2] * M_steps
    crossings = [(1, 0, 1, 1.6e-3)]
    return MechanismTuple(
        name="assimilatory_nr_m4", 
        N_orbitals=N, 
        M_steps=M_steps, 
        m=4,
        S_target=1.5, 
        electron_sets=A_sets, 
        proton_sets=P_sets,
        cofactor_sets=B_sets, 
        nu_shifts=nu_shifts, 
        crossings=crossings,
        phi_proton=np.pi/2, 
        dock_orbitals=None,
        description=(
            "Entry 8 (assimilatory NR oldform, Group B): m=4, ν=2, n*=1, s=0.08115. "
            "Same Mo-S₂-O₂ proxy as entry 7 at coarser ℤ₄ register. "
            "Winding (2,0), M_cof=0. Janus at n=1 (NO₂⁻ intermediate)."
        ),
    )

def _build_cu_co2rr_m4_spec(n_orbitals: int) -> MechanismTuple:
    """Catalog entry 12 (Cu CO₂RR oldform, Group B): M=12, m=4, ν=2, n*=1, s=0.08115.
    Same Cu₃⁻ trimer proxy as entry 13 at coarser ℤ₄ register.
    Winding (3,0): N_e=12, M_cof=0. Janus at n=1 (*CO intermediate). --tower-p 2.
    """
    N = n_orbitals
    M_steps = 12
    A_sets    = [[n % N] for n in range(M_steps)]
    P_sets    = [[(n + 1) % N] for n in range(M_steps)]
    B_sets    = [[(n + 2) % N] for n in range(M_steps)]
    nu_shifts = [2] * M_steps
    crossings = [(1, 0, 1, 1.6e-3)]
    return MechanismTuple(
        name="cu_co2rr_m4", 
        N_orbitals=N, 
        M_steps=M_steps, 
        m=4,
        S_target=0.0, 
        electron_sets=A_sets, 
        proton_sets=P_sets,
        cofactor_sets=B_sets, 
        nu_shifts=nu_shifts, 
        crossings=crossings,
        phi_proton=np.pi/2, 
        dock_orbitals=None,
        description=(
            "Entry 12 (Cu CO₂RR oldform, Group B): m=4, ν=2, n*=1, s=0.08115. "
            "Same Cu₃⁻ trimer proxy as entry 13 at coarser ℤ₄ register. "
            "Winding (3,0), M_cof=0. Janus at n=1 (*CO intermediate)."
        ),
    )

def _build_photocatalytic_n2_m4_spec(n_orbitals: int) -> MechanismTuple:
    """Catalog entry 15 (photocatalytic N₂ fix oldform, Group B): M=8, m=4, ν=2, n*=1, s=0.08115.
    Same Ti₂N₂ proxy as entry 14 at coarser ℤ₄ register.
    Winding (2,2): N_e=8, M_cof=8. Janus at n=1 (diimide *HN=NH intermediate). --tower-p 2.
    """
    N = n_orbitals
    M_steps = 8
    A_sets    = [[n % N] for n in range(M_steps)]
    P_sets    = [[(n + 1) % N] for n in range(M_steps)]
    B_sets    = [[(n + 2) % N] for n in range(M_steps)]
    nu_shifts = [2] * M_steps
    crossings = [(1, 0, 1, 1.6e-3)]
    return MechanismTuple(
        name="photocatalytic_n2_m4", 
        N_orbitals=N, 
        M_steps=M_steps, 
        m=4,
        S_target=1.5, 
        electron_sets=A_sets, 
        proton_sets=P_sets,
        cofactor_sets=B_sets, 
        nu_shifts=nu_shifts, 
        crossings=crossings,
        phi_proton=np.pi/2, 
        dock_orbitals=None,
        description=(
            "Entry 15 (photocatalytic N₂ fix oldform, Group B): m=4, ν=2, n*=1, s=0.08115. "
            "Same Ti₂N₂ proxy as entry 14 at coarser ℤ₄ register. "
            "Winding (2,2), M_cof=8. Janus at n=1 (diimide intermediate)."
        ),
    )


def _build_complex_i_spec(n_orbitals: int) -> MechanismTuple:
    """Catalog entry 6 (Complex I, Group C): M=4, m=4, ν=1, n*=3, s=0.04135.
    NADH:ubiquinone oxidoreductase — respiratory chain proton pump.
    [2Fe-2S] proxy representing terminal ET from N2 cluster to quinone.
    Winding (1,0): N_e=4, M_cof=0 (proton pumping is internal, not cofactor).
    Phase closure: Σν = 4 ≡ 0 (mod 4) ✓.
    """
    N = n_orbitals
    M_steps = 4

    # PCET Fields: 1 e⁻/step, 4 steps → 4 e⁻ total (Q → QH₂)
    A_sets = [[n % N] for n in range(M_steps)]
    P_sets = [[(n + 1) % N] for n in range(M_steps)]
    B_sets = [[(n + 2) % N] for n in range(M_steps)]
    nu_shifts = [1] * M_steps   # ν=1 → Group C; n*=4/1−1=3

    # Janus at n=2: TYKY conformational checkpoint (midpoint of 4-step cycle)
    # After 2 e⁻ transferred, electron sits on N6a/N6b; 2 of 4 H⁺ pumped.
    crossings = [(2, 0, 1, 1.6e-3)]

    return MechanismTuple(
        name          = "complex_i",
        N_orbitals    = N,
        M_steps       = M_steps,
        m             = 4,                  # ℤ₄ phase group
        S_target      = 0.0,                # Singlet (even-electron reduced chain)
        electron_sets = A_sets,
        proton_sets   = P_sets,
        cofactor_sets = B_sets,
        nu_shifts     = nu_shifts,
        crossings     = crossings,
        phi_proton    = np.pi / 2,
        dock_orbitals = None,
        description   = (
            "Entry 6 (Complex I, Group C): m=4, ν=1, n*=3, s=0.04135. "
            "[2Fe-2S] proxy for terminal ET to quinone. "
            "Winding (1,0), M_cof=0. 4e⁻ reduce Q to QH₂. "
            "Janus at n=2: TYKY conformational checkpoint. "
            "Phase closure: Σν=4 ≡ 0 (mod 4)."
        ),
    )

def _build_codh_acs_spec(n_orbitals: int) -> MechanismTuple:
    """Catalog entry 9 (CODH/ACS, Group C): M=8, m=4, ν=1, n*=3, s=0.04135.
    Ni₂S₂ proxy for the C-cluster/A-cluster interface.
    Winding (2,1): N_e=8, M_cof=4 (Na⁺). Janus at n=1 (Ni_p(I)·CO).
    
    The (2,1) winding is unique in the catalog: sub-unitary cofactor
    coupling (w_2 < w_1). The Na⁺ gradient drives the mechanism at half
    the rate of electron transfer.
    """
    N = n_orbitals
    M_steps = 8
    
    A_sets = [[n % N] for n in range(M_steps)]
    P_sets = [[(n + 1) % N] for n in range(M_steps)]
    B_sets = [[(n + 2) % N] for n in range(M_steps)]
    nu_shifts = [1] * M_steps  # ν=1 → Group C; n*=4/1-1=3
    
    # Janus at n=1: Ni_p(I)·CO A-cluster state formation
    crossings = [(1, 0, 1, 1.6e-3)]
    
    return MechanismTuple(
        name          = "codh_acs",
        N_orbitals    = N,
        M_steps       = M_steps,
        m             = 4,                  # ℤ₄ phase group
        S_target      = 0.0,                # Singlet (88e, even)
        electron_sets = A_sets,
        proton_sets   = P_sets,
        cofactor_sets = B_sets,
        nu_shifts     = nu_shifts,
        crossings     = crossings,
        phi_proton    = np.pi / 2,
        dock_orbitals = None,
        description   = (
            "Entry 9 (CODH/ACS, Group C): m=4, ν=1, n*=3, s=0.04135. "
            "Ni₂S₂ proxy for C-cluster/A-cluster interface, Ni-S 2.30→2.19 Å. "
            "Winding (2,1), M_cof=4. Janus at n=1: Ni_p(I)·CO state. "
            "Unique sub-unitary cofactor coupling in the catalog."
        ),
    )

def _build_cyt_bd_oxidase_spec(n_orbitals: int) -> MechanismTuple:
    """Catalog entry 10 (Cyt bd oxidase, Group C): M=4, m=4, ν=1, n*=3, s=0.04135.
    Fe₂O₂ proxy for the binuclear heme center (b558 + b595 + heme d simplified).
    Winding (1,0): N_e=4, M_cof=0 (non-pumping terminal oxidase).
    Janus at n=2: two-electron-reduced state (b558-Fe(II) + b595-Fe(II), d-Fe(III)).
    Phase closure: Σν = 4 ≡ 0 (mod 4) ✓.
    Spectrally degenerate with PSII (Entry 5) and Complex I (Entry 6 at M=8).
    """
    N = n_orbitals
    M_steps = 4

    # PCET Fields: 1 e⁻/step, 4 steps → 4 e⁻ total (2 quinol → 2 Q + 4 e⁻)
    A_sets = [[n % N] for n in range(M_steps)]
    P_sets = [[(n + 1) % N] for n in range(M_steps)]
    B_sets = [[(n + 2) % N] for n in range(M_steps)]
    nu_shifts = [1] * M_steps   # ν=1 → Group C; n*=4/1−1=3

    # Janus at n=2: two-electron-reduced state (midpoint of 4-step cycle)
    # After 2 e⁻ transferred, hemes b558 + b595 are Fe(II), heme d still Fe(III)
    crossings = [(2, 0, 1, 1.6e-3)]

    return MechanismTuple(
        name          = "cyt_bd_oxidase",
        N_orbitals    = N,
        M_steps       = M_steps,
        m             = 4,                  # ℤ₄ phase group
        S_target      = 0.0,                # Singlet (two-electron-reduced state)
        electron_sets = A_sets,
        proton_sets   = P_sets,
        cofactor_sets = B_sets,
        nu_shifts     = nu_shifts,
        crossings     = crossings,
        phi_proton    = np.pi / 2,
        dock_orbitals = None,
        description   = (
            "Entry 10 (Cyt bd oxidase, Group C): m=4, ν=1, n*=3, s=0.04135. "
            "Fe₂O₂ proxy for binuclear heme center. "
            "Winding (1,0), M_cof=0 (non-pumping). 4e⁻ from 2 quinol. "
            "Janus at n=2: two-electron-reduced state. "
            "Phase closure: Σν=4 ≡ 0 (mod 4). "
            "Spectrally degenerate with PSII (Entry 5)."
        ),
    )

def _build_cyt_c_oxidase_spec(n_orbitals: int) -> MechanismTuple:
    """Catalog entry 11 (Cyt c oxidase, Group C): M=4, m=4, ν=1, n*=3, s=0.04135.
    Fe-Cu-N-O binuclear proxy for heme a3-CuB center.
    Winding (1,0): N_e=4, M_cof=0 (proton pumping is internal).
    Janus at n=2: R state (heme a3-Fe(II), CuB-Cu(I)). --tower-p 2.
    """
    N = n_orbitals
    M_steps = 4
    A_sets    = [[n % N] for n in range(M_steps)]
    P_sets    = [[(n + 1) % N] for n in range(M_steps)]
    B_sets    = [[(n + 2) % N] for n in range(M_steps)]
    nu_shifts = [1] * M_steps
    crossings = [(2, 0, 1, 1.6e-3)]
    return MechanismTuple(
        name="cyt_c_oxidase", 
        N_orbitals=N, 
        M_steps=M_steps, 
        m=4,
        S_target=0.0, 
        electron_sets=A_sets, 
        proton_sets=P_sets,
        cofactor_sets=B_sets, 
        nu_shifts=nu_shifts, 
        crossings=crossings,
        phi_proton=np.pi/2, 
        dock_orbitals=None,
        description=(
            "Entry 11 (Cyt c oxidase, Group C): m=4, ν=1, n*=3, s=0.04135. "
            "Fe-Cu-N-O binuclear proxy for heme a3-CuB center. "
            "Winding (1,0), M_cof=0. Janus at n=2: R state."
        ),
    )

def build_predefined_mechanisms(n_orbitals: int) -> Dict[str, MechanismTuple]:
    r"""Build all five predefined mechanisms from the article table.

    Uses n_orbitals from the loaded molecular integral dataset. For a
    production implementation, each step would carry its own geometry-
    specific Hamiltonian H_n. Here, the same integral set is reused at
    each step to validate the pipeline structure and stoichiometric
    bookkeeping — not the molecular energetics.

    Mechanisms:
      1. nitrogenase_lt  — LT cycle, M=8, m=4 (ℤ₄), 16 ATP, 1 Janus
      2. psii            — Kok S-state, M=4, m=4 (ℤ₄), 4 photons
      3. hydrogenase     — 2H⁺+2e⁻→H₂, M=2, m=1 (trivial)
      4. z3_cofactor     — 3-cofactor reaction, M=3, m=3 (ℤ₃)
      5. z5_cofactor     — 5-cofactor reaction, M=5, m=5 (ℤ₅)

    Returns:
        Dict mapping mechanism name string → MechanismTuple.
    """
    N = n_orbitals

    # ──────────────────────────────────────────────────────────────────────────
    # 1. NITROGENASE LT CYCLE
    # N₂ + 8H⁺ + 8e⁻ + 16ATP → 2NH₃ + H₂ + 16ADP + 16Pᵢ
    # M=8, m=4, ν_n=2 (2 ATP per step → 16 total), n_cross=1 at n=4.
    # Phase closure: Σν = 16 ≡ 0 (mod 4) ✓
    # ──────────────────────────────────────────────────────────────────────────
    
    # FIX: Explicit indexing matches _build_nitrogenase_lt_spec exactly
    A_lt = [[n % N] for n in range(8)]
    P_lt = [[(n + 1) % N] for n in range(8)]
    B_lt = [[(n + 2) % N] for n in range(8)]
    
    nitrogenase_lt = MechanismTuple(
        name          = "nitrogenase_lt",
        N_orbitals    = N,
        M_steps       = 8,
        m             = 4,
        S_target      = 1.5,           # S = 3/2 resting spin sector
        electron_sets = A_lt,          # 1 e⁻ injected per step → 8 total
        proton_sets   = P_lt,          # 1 H⁺ per step → 8 total
        cofactor_sets = B_lt,          # 1 B_n register per step
        nu_shifts     = [2] * 8,       # 2 ATP/step → Σ=16 ≡ 0 (mod 4)
        crossings     = [(4, 0, 1, 1.6e-3)],  # Janus at n=4, orbitals 0,1
        phi_proton    = np.pi / 2,
        dock_orbitals = [[(i+2) % N] for i in range(8)],
        description   = (
            "Lowe-Thorneley nitrogenase catalytic cycle. "
            "E₀→E₈ via 8 sequential e⁻/H⁺ transfers, 16 ATP. "
            "Janus non-adiabatic crossing at E₄→E₅. "
            "Fully reversible framework ready for back-reaction modeling."
        ),
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 2. PHOTOSYSTEM II (KOK S-STATE CYCLE)
    # 2H₂O → O₂ + 4H⁺ + 4e⁻ (4 photons, adiabatic)
    # M=4, m=4, ν_n=1 (1 photon per step → 4 total), n_cross=0.
    # Phase closure: Σν = 4 ≡ 0 (mod 4) ✓
    # ──────────────────────────────────────────────────────────────────────────
    
    # FIX: Explicit modular indexing matches _build_psii_spec exactly
    A_ps = [[n % N] for n in range(4)]
    P_ps = [[(n + 1) % N] for n in range(4)]
    B_ps = [[(n + 2) % N] for n in range(4)]
    
    psii = MechanismTuple(
        name          = "psii",
        N_orbitals    = N,
        M_steps       = 4,
        m             = 4,
        S_target      = 0.0,          # Singlet product O₂ (triplet virtual)
        electron_sets = A_ps,
        proton_sets   = P_ps,
        cofactor_sets = B_ps,
        nu_shifts     = [1] * 4,      # 1 photon/step → Σ=4 ≡ 0 (mod 4)
        crossings     = [(2, 0, 1, 1.6e-3)],  # Janus at S2: max-oxidised state
        phi_proton    = np.pi / 2,
        dock_orbitals = None,
        description   = (
            "Photosystem II Kok S-state water oxidation cycle. "
            "4 sequential photooxidations of Mn₄CaO₅ cluster. "
            "Janus at n=2 (S₂ state, max oxidation before O₂ release). "
            "Net e⁻ extraction: 4. Phase closure: 4 ≡ 0 (mod 4)."
        ),
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 3. HYDROGENASE
    # 2H⁺ + 2e⁻ → H₂ (no cofactor, adiabatic)
    # M=2, m=1 (trivial phase group), n_cross=0.
    # Phase closure: trivially 0 ≡ 0 (mod 1) ✓
    # ──────────────────────────────────────────────────────────────────────────
    
    # FIX: Explicit indexing matches _build_hydrogenase_spec exactly
    A_hyd = [[n % N] for n in range(2)]
    P_hyd = [[(n + 1) % N] for n in range(2)]
    B_hyd = [[] for _ in range(2)]
    nu_hyd = [0] * 2
    
    hydrogenase = MechanismTuple(
        name          = "hydrogenase",
        N_orbitals    = N,
        M_steps       = 2,
        m             = 1,             # Trivial Z_1: no phase bookkeeping
        S_target      = 0.0,
        electron_sets = A_hyd,
        proton_sets   = P_hyd,
        cofactor_sets = B_hyd,
        nu_shifts     = nu_hyd,        # No cofactor → Σ=0 ≡ 0 (mod 1)
        crossings     = [],
        phi_proton    = np.pi / 2,
        dock_orbitals = None,
        description   = (
            "[FeFe]-hydrogenase: 2H⁺ + 2e⁻ → H₂. "
            "Two-step reduction, no cofactor consumption. "
            "Trivial ℤ₁ phase group (H_V serves as coherence buffer only)."
        ),
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 4. GENERIC 3-COFACTOR REACTION (ℤ₃)
    # 3 cofactor units consumed across M=3 steps.
    # m=3, ν_n=1 per step, Σν=3 ≡ 0 (mod 3) ✓
    # Requires d=3 virtual clock — generalizes beyond ℤ₄.
    # ──────────────────────────────────────────────────────────────────────────
    A_z3, P_z3, B_z3 = _make_uniform_sets(N, 3)
    z3_cofactor = MechanismTuple(
        name          = "z3_cofactor",
        N_orbitals    = N,
        M_steps       = 3,
        m             = 3,             # ℤ₃ virtual clock, d=3 virtual qudits
        S_target      = 0.5,
        electron_sets = A_z3,
        proton_sets   = P_z3,
        cofactor_sets = B_z3,
        nu_shifts     = [1, 1, 1],     # Σν=3 ≡ 0 (mod 3)
        crossings     = [],
        phi_proton    = np.pi / 2,
        dock_orbitals = None,
        description   = (
            "Generic 3-cofactor reaction: ℤ₃ virtual phase group. "
            "Validates GeneralizedVirtualShiftGate for m=3. "
            "Phase closure: 3 ≡ 0 (mod 3)."
        ),
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 5. GENERIC 5-COFACTOR REACTION (ℤ₅)
    # 5 cofactor units consumed across M=5 steps.
    # m=5, ν_n=1 per step, Σν=5 ≡ 0 (mod 5) ✓
    # Requires d=5 virtual clock — prime modulus, irreducible over ℤ₄.
    # ──────────────────────────────────────────────────────────────────────────
    A_z5, P_z5, B_z5 = _make_uniform_sets(N, 5)
    z5_cofactor = MechanismTuple(
        name          = "z5_cofactor",
        N_orbitals    = N,
        M_steps       = 5,
        m             = 5,             # ℤ₅ virtual clock, d=5 virtual qudits
        S_target      = 0.5,
        electron_sets = A_z5,
        proton_sets   = P_z5,
        cofactor_sets = B_z5,
        nu_shifts     = [1, 1, 1, 1, 1],  # Σν=5 ≡ 0 (mod 5)
        crossings     = [],
        phi_proton    = np.pi / 2,
        dock_orbitals = None,
        description   = (
            "Generic 5-cofactor reaction: ℤ₅ virtual phase group. "
            "Validates GeneralizedVirtualShiftGate for m=5 (prime modulus). "
            "Phase closure: 5 ≡ 0 (mod 5)."
        ),
    )

    

    return {
        "nitrogenase_lt":       nitrogenase_lt,
        "nitrogenase_lt_m8":    _build_nitrogenase_lt_m8_spec(n_orbitals),
        "nitrogenase_lt_parallel": _build_nitrogenase_lt_parallel_spec(n_orbitals),
        "psii":                 psii,
        "hydrogenase":          hydrogenase,
        "hydrogenase_oxidation": _build_hydrogenase_oxidation_spec(n_orbitals),
        "z3_cofactor":          z3_cofactor,
        "z5_cofactor":          z5_cofactor,
        "haber_bosch":          _build_haber_bosch_spec(n_orbitals),
        "nitrogenase_fe4s4":    _build_nitrogenase_fe4s4_spec(n_orbitals),
        "nitrogenase_femoco":   _build_nitrogenase_femoco_spec(n_orbitals),
        "femon2_trimer":        _build_femon2_trimer_spec(n_orbitals),
        "nitrogenase_group_a":  _build_nitrogenase_group_a_spec(n_orbitals),
        "nitrogenase_group_d":  _build_nitrogenase_group_d_spec(n_orbitals),
        "mo_nitrogenase":       _build_mo_nitrogenase_spec(n_orbitals),
        "assimilatory_nr":      _build_assimilatory_nr_spec(n_orbitals),
        "photocatalytic_n2":    _build_photocatalytic_n2_spec(n_orbitals),
        "v_nitrogenase":        _build_v_nitrogenase_spec(n_orbitals),
        "cu_co2rr":             _build_cu_co2rr_spec(n_orbitals),
        "ethylene_epoxidation": _build_ethylene_epoxidation_spec(n_orbitals),
        "thymine_dimer_proxy":  _build_thymine_dimer_spec(n_orbitals),
        "anammox_proxy":        _build_anammox_proxy_spec(n_orbitals),
        "atp_hydrolysis_proxy": _build_atp_hydrolysis_proxy_spec(n_orbitals),
        "cyp450_metabolism": _build_cyp450_metabolism_spec(n_orbitals),
        "reversible_quinone": _build_reversible_quinone_spec(n_orbitals),
        "rnr_radical_proxy":    _build_rnr_radical_proxy_spec(n_orbitals),
        # ── Reversible closed-loop benchmark ────────────────────────────────
        "nitrogenase_closed_loop": _build_nitrogenase_closed_loop_spec(n_orbitals),
        # ── Photo-driven mechanism benchmarks ───────────────────────────────
        "psii_photo":              _build_psii_photo_spec(n_orbitals),
        # ──Methanogenesis ──────────────────────────────────────────────────────
        "methanogenesis_proxy": _build_methanogenesis_proxy_spec(n_orbitals),

        # ── Oldform lifts (Group B) ─────────────────────────────────────────
        "mo_nitrogenase_m4":       _build_mo_nitrogenase_m4_spec(n_orbitals),
        "v_nitrogenase_m4":        _build_v_nitrogenase_m4_spec(n_orbitals),
        "assimilatory_nr_m4":      _build_assimilatory_nr_m4_spec(n_orbitals),
        "cu_co2rr_m4":             _build_cu_co2rr_m4_spec(n_orbitals),
        "photocatalytic_n2_m4":    _build_photocatalytic_n2_m4_spec(n_orbitals),

       # ── New unique entries ──────────────────────────────────────────────
        "complex_i":               _build_complex_i_spec(n_orbitals),
        "codh_acs":                _build_codh_acs_spec(n_orbitals),
        "cyt_bd_oxidase":          _build_cyt_bd_oxidase_spec(n_orbitals),
        "cyt_c_oxidase":           _build_cyt_c_oxidase_spec(n_orbitals),
    }




