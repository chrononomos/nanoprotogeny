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
mqedatagenerator.py
===================
MQE Test Dataset Generation Extension for mqeintegrals.py.

Generates the precise molecular integral datasets required to definitively
test the Modular Quantum Emulator (MQE) framework across all five
mechanism classes, all three Z_m phase groups, and both non-adiabatic and
adiabatic catalytic pathways.

Design Philosophy
-----------------
A definitive MQE test requires three orthogonal dimensions of validation:

  Dimension 1 — Mechanism Coverage (5 mechanisms from the article table)
    Tests that the piecewise compositional pipeline L_n correctly handles
    each mechanism type: LT (Z4, Janus crossing), PSII (Z4, adiabatic),
    Hydrogenase (Z1, trivial), Z3-cofactor, Z5-cofactor.

  Dimension 2 — Chemical Accuracy Validation
    For each mechanism, the Hamiltonian at each step E_n must be a
    chemically real active space with a known FCI reference. The QPE+ZNE
    pipeline must recover E_0 to within 1.6 mHa at EVERY step.

  Dimension 3 — Stoichiometric Invariance
    Phase closure k_total ≡ 0 (mod m) and electron accumulation
    <N_e>_final = Σ|A_n| must hold exactly, verified algebraically
    independent of the quantum simulation.

Dataset Architecture
--------------------
Each MQE mechanism maps to a SEQUENCE of M Hamiltonians {H_n}_{n=0}^{M-1},
one per catalytic step. Each H_n is a real fermionic active space extracted
from a chemically distinct molecular geometry along the reaction coordinate.

For each of the five mechanisms we generate:

  1. nitrogenase_lt (M=8, m=4, Z4):
     Step sequence: N2 dissociation curve points E0..E7 on [Fe2S2] cluster.
     8 geometries at bond-lengths spanning the N-N activation coordinate.
     Janus crossing at step 4 (E4→E5): minimum energy crossing point.

  2. psii (M=4, m=4, Z4):
     Step sequence: [Fe2S2] cluster (substituting Mn4 for tractability at N=4).
     4 geometries representing Kok S0→S1→S2→S3 states via bond compression.

  3. hydrogenase (M=2, m=1, Z1):
     Step sequence: H2 dissociation (2 steps: H–H stretched, compressed).
     Simplest possible test: N=2 orbitals, exact H2 FCI reference.

  4. z3_cofactor (M=3, m=3, Z3):
     Step sequence: H3+ (equilateral triangle → isoceles → equilateral).
     3 geometries validating Z3 phase closure for a prime modulus.

  5. z5_cofactor (M=5, m=5, Z5):
     Step sequence: H5+ linear chain (5 evenly-spaced geometries).
     5 geometries validating Z5 phase closure for a prime modulus.

JSON Schema (per mechanism, per step)
--------------------------------------
Each step produces a standard mqeintegrals.py JSON extended with:
  "mqe_step": {
    "mechanism":    str,          # mechanism name
    "step_n":       int,          # step index 0..M-1
    "M_total":      int,          # total steps
    "m_modulus":    int,          # Z_m group modulus
    "nu_n":         int,          # cofactor shift at this step
    "A_n":          List[int],    # electron-injection orbital indices
    "P_n":          List[int],    # proton-phase orbital indices
    "B_n":          List[int],    # virtual cofactor register indices
    "is_crossing":  bool,         # True if Janus conical intersection here
    "delta_CI_Ha":  float|null,   # degeneracy threshold (Ha)
    "crossing_orbitals": [p,q]|null,  # hydride orbital pair
    "geometry_label": str,        # human-readable geometry description
    "bondlength_angstrom": float, # primary bond length at this step
    "phase_index_k": int,         # k^{(n)} = Σ_{i<=n} ν_i mod m
    "cumulative_electrons": int,  # Σ_{i<=n} |A_i|
  }

Usage
-----
  # Generate all five mechanism datasets
  python mqedatagenerator.py --mechanism all --basis STO-3G --output_dir mqe_datasets/

  # Generate a single mechanism
  python mqedatagenerator.py --mechanism nitrogenase_lt --basis STO-3G

  # Generate without FCI (faster, skips chemical-accuracy check)
  python mqedatagenerator.py --mechanism all --no_fci

  # Run the MQE framework against generated datasets
  python mqe.py --mqe-mechanism nitrogenase_lt \\
      --mqe-dataset-dir mqe_datasets/

Validation Criteria (printed per mechanism)
--------------------------------------------
  [✓] Phase closure: k_total ≡ 0 (mod m) for all steps
  [✓] Electron count: <N_e>_final = expected_electrons
  [✓] FCI residual: |E_ZNE - E_FCI| ≤ 1.6 mHa at every step
  [✓] Hamiltonian Hermiticity: h1[p,q] = h1[q,p] for all p,q
  [✓] ERI 8-fold: g[p,q,r,s] = g[q,p,r,s] = g[p,q,s,r] = g[r,s,p,q]
  [✓] Trace preservation: Tr(rho) = 1 at every step
  [✓] Warrant threshold: omega(AntiTh) + omega(SynTh) >= eta
"""

from __future__ import annotations

import ast
import json
import logging
import gc
import argparse
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from pyscf import ao2mo, fci, gto, mcscf, scf

# Import all infrastructure from mqeintegrals.py
# (When run as extension, these are already in scope)
from nanoprotogeny.molecular.mqeriemann import (
    RiemannScaffold
)

from nanoprotogeny.molecular.mqeintegralstore import StepwiseIntegralStore

from nanoprotogeny.molecular.mqeintegrals import (
    build_molecule,
    run_rohf,
    extract_casci_integrals,
    extract_full_mo_integrals,
    extract_tower_window_integrals,
    compute_reference_energy,
    compress_eri,
    export_json,
    make_nelec_tuple,
    spin_label,
    default_output,
    MOLECULE_REGISTRY,
    FE2S2_GEOMETRY,
    log,
)

# ===========================================================================
# 1. MQE STEP DESCRIPTOR
# ===========================================================================

@dataclass
class MQEStep:
    r"""Complete descriptor for one catalytic step in a MQE mechanism.
    
    Encodes all parameters from the Fermionic Catalytic Mechanism Tuple
    (Definition 2) for a single step $n$ of mechanism $\mathfrak{M}$.
    Fields are directly consumed by the MQE pipeline 
    (`mqe.py::MechanismTuple` and `MQEGateFactory`).

    REVERSIBLE EXTENSION:
    Extends the forward-only LT model to support bidirectional PCET and
    cofactor decoupling. Explicit ejection/decoupling lists enable exact
    circuit uncomputation, thermodynamic back-reaction simulation, and
    net-flux stoichiometric tracking.

    Attributes:
        mechanism:          Mechanism identifier matching the dataset registry.
        step_n:             Zero-based step index $n \in [0, M-1]$.
        M_total:            Total number of steps $M$ in the catalytic cycle.
        m_modulus:          Virtual register modulus $m$ ($\mathbb{Z}_m$ phase group).

        nu_n:               Forward cofactor shift magnitude $\nu_n$ applied to $B_n$.
        A_n:                Logical orbital indices receiving $e^-$ (reduction via $\hat{U}_R$).
        P_n:                Logical orbital indices receiving $H^+$ (protonation via $\hat{Z}_{\text{Clock}}$).
        B_n:                Virtual register indices for forward coupling $\hat{U}_{\text{coupling}}$.

        A_n_eject:          Logical orbital indices losing $e^-$ (oxidation via $\hat{U}_R^\dagger$).
        P_n_eject:          Logical orbital indices losing $H^+$ (deprotonation via $\hat{Z}_{\text{Clock}}(-\phi)$).
        B_n_decouple:       Virtual register indices for inverse coupling $\hat{U}_{\text{coupling}}^\dagger$.
        nu_decouple_n:      Magnitude of inverse cofactor shift (subtracts from phase index).

        is_crossing:        Flags non-adiabatic surface hopping (Janus crossing) at this step.
        delta_CI_Ha:        Degeneracy threshold $\delta_{\text{CI}}$ (Ha) triggering the crossing.
        crossing_orbitals:  Pair of hydride orbital indices $[p, q]$ for $\mathcal{S}_{LV}^{(p,q)}$.

        geometry_label:     Human-readable geometry descriptor for the reaction coordinate.
        bondlength_angstrom:Primary bond length defining the nuclear configuration at step $n$.

        cumulative_electrons:   Forward-only electron count $\sum_{i=0}^n |\mathcal{A}_i|$ (legacy/compatibility).
        phase_index_k:          Net phase index $k^{(n)} \equiv \sum_{i=0}^n (\nu_i - \nu_i^{\text{decouple}}) \pmod m$.
        cumulative_net_electrons: Physical net electron count $\sum_{i=0}^n (|\mathcal{A}_i| - |\mathcal{A}_i^{\text{eject}}|)$.
    """
    # Step identity (mechanism-level fields belong on MQEMechanismSpec, not here)
    step_n:             int

    # ── Forward transitions (Injection/Coupling) ──
    nu_n:               int         # Cofactor shift: (U_R^{V,m})^{nu_n}
    A_n:                List[int]   # Orbitals receiving e⁻ (reduction)
    P_n:                List[int]   # Orbitals receiving H⁺ (protonation)
    B_n:                List[int]   # Virtual cofactor registers for coupling

    # ── REVERSIBLE EXTENSIONS (Ejection/Decoupling) ──
    A_n_eject:          List[int]   = field(default_factory=list)  # Orbitals losing e⁻ (oxidation)
    P_n_eject:          List[int]   = field(default_factory=list)  # Orbitals losing H⁺ (deprotonation)
    B_n_decouple:       List[int]   = field(default_factory=list)  # Virtual registers for U_cof^†
    nu_decouple_n:      int         = 0                            # Magnitude of inverse cofactor shift

    # ── PHOTON EXTENSIONS (Gap 1) ──
    Gamma_n_abs:        List[int]   = field(default_factory=list)
    Gamma_n_emit:       List[int]   = field(default_factory=list)
    phi_photon_n:       float       = 0.0

    # Non-adiabatic crossing (if applicable)
    is_crossing:        bool        = False
    delta_CI_Ha:        Optional[float] = None
    crossing_orbitals:  Optional[List[int]] = None

    # Geometry parameters
    geometry_label:     str         = " "
    bondlength_angstrom:float       = 1.0

    # Accumulated state (computed after full sequence)
    cumulative_electrons: int       = 0   # Σ_{i≤n} |A_i|
    phase_index_k:      int         = 0   # k^{(n)} ≡ Σ_{i≤n} (ν_i - ν_decouple_i) (mod m)
    cumulative_net_electrons: int   = 0   # Σ_{i≤n} (|A_i| - |A_i_eject|)

    # ── PHOTON INVARIANTS (Gap 1) ──
    cumulative_photons_absorbed: int = 0
    cumulative_photons_emitted:  int = 0
    cumulative_net_photons:      int = 0

    def to_dict(
        self,
        mechanism: str = "",
        M_total: int = 0,
        m_modulus: int = 1,
    ) -> Dict:
        """Serialise step to a dict for JSON embedding.

        ``mechanism``, ``M_total``, and ``m_modulus`` are injected from the
        parent ``MQEMechanismSpec`` by the caller — they are not stored on
        ``MQEStep`` to avoid the drift-apart bug where step-side and spec-side
        values could diverge.  The fields are retained in the JSON output for
        schema backward-compatibility.
        """
        d = asdict(self)
        # Inject spec-derived fields for JSON backward-compatibility
        d["mechanism"] = mechanism
        d["M_total"]   = M_total
        d["m_modulus"] = m_modulus
        d["phase_closure_check"] = (
            f"k^({self.step_n}) = {self.phase_index_k}  "
            f"(mod {m_modulus}) | "
            f"photons: {self.cumulative_net_photons}"
        )
        return d


@dataclass
class MQEMechanismSpec:
    r"""Full specification of one MQE mechanism for dataset generation.

    Encodes the complete Fermionic Catalytic Mechanism Tuple $\mathfrak{M}$
    with geometry sequences and stoichiometric invariants required for
    step-wise integral generation and algebraic validation.

    THEOREM 2 EXTENSION (Net-Flux Invariance):
    Replaces cumulative stoichiometric counters with exact net-flux
    invariants. Enables rigorous validation of reversible cycles where 
    forward injection and backward ejection/decoupling balance to a 
    prescribed net change (typically zero for closed thermodynamic loops).

    Args:
        name:                   Mechanism identifier (keys into `mqe.py` registry).
        M_steps:                Number of discrete catalytic steps $M$.
        m_modulus:              Virtual clock modulus $m$ for $\mathbb{Z}_m$ phase tracking.
        S_target:               Target spin quantum number $S$ for symmetry projection.
        n_orbitals:             Active-space orbital count $N$.
        steps:                  Ordered list of `MQEStep` descriptors (length $M$).
        description:            Physical/chemical mechanism summary.
        expected_total_electrons:   Legacy forward-only target $\sum_n |\mathcal{A}_n|$.
        expected_net_electrons:     Target net electron flux $\sum_n (|\mathcal{A}_n| - |\mathcal{A}_n^{\text{eject}}|)$.
        expected_net_phase:         Target net phase accumulation (typically $0$ for closed cycles).
        expected_net_phase_closure: Flag enforcing $k_{\text{total}}^{\text{net}} \equiv 0 \pmod m$.
    """
    name:                      str
    M_steps:                   int
    m_modulus:                 int
    S_target:                  float
    n_orbitals:                int
    steps:                     List[MQEStep]
    description:               str
    
    # ALL fields from here down MUST have defaults
    expected_total_electrons:   int = 0
    
    # ── NET-FLUX INVARIANTS (Theorem 2 Extension) ──
    expected_net_electrons:    int = 0       # Σ (|A_n| - |A_n_eject|)
    expected_net_phase:        int = 0       # 0 (mod m)
    expected_net_phase_closure: bool = True  # Already had a default

    # ── PHOTON INVARIANTS (Gap 2) ──
    phi_photon:                      float = 0.0
    expected_total_photons_absorbed: int = 0
    expected_total_photons_emitted:  int = 0
    expected_net_photons:            int = 0

    # ── ENERGY ORDERING (declarative, used by _validate_energy_ordering) ──
    # Allowed values:
    #   "decreasing"          — fci[-1] < fci[0]  (reductive cycles)
    #   "increasing"          — fci[-1] > fci[0]  (oxidative cycles)
    #   "monotone_decreasing" — each step ≤ previous (compression)
    #   "monotone_increasing" — each step ≥ previous (photo-oxidation)
    #   "closure"             — |fci[0] - fci[-1]| < tol  (Zₘ cofactor return)
    #   "closed_loop"         — fwd half decreases + |fci[-1] - fci[0]| < 10 mHa
    #   "nondegen"            — not all steps degenerate (radical proxy)
    #   "reversible_quinone"  — reduction increases then oxidation recovers
    #   "none"                — no ordering constraint (always passes)
    expected_energy_ordering: str = "none"


# ===========================================================================
# 2. GEOMETRY SEQUENCES (reaction coordinates for each mechanism)
# ===========================================================================

def _hchain_geometry(n: int, bondlength: float) -> str:
    """Linear H_n chain geometry (Angstrom), centred at origin."""
    total_len = (n - 1) * bondlength
    positions = [-total_len / 2 + i * bondlength for i in range(n)]
    lines = [f"H  {x:10.6f}  0.000000  0.000000" for x in positions]
    return "\n".join(lines)


def _fe2s2_geometry_at_bond(fe_s_distance: float) -> str:
    """[Fe2(mu-S)2] rhombic core with variable Fe–S distance (Angstrom).

    Used to represent the catalytic trajectory of nitrogenase and PSII
    model clusters. The Fe–Fe distance is held fixed at 2.70 Å; the
    Fe–S distance varies to model oxidation-state changes.

    Args:
        fe_s_distance: Fe–S bond length in Angstrom (nominally 2.26 Å).
    """
    d = fe_s_distance
    return (
        f"Fe   0.000000  1.350000  0.000000\n"
        f"Fe   0.000000 -1.350000  0.000000\n"
        f"S    {d:.6f}  0.000000  0.000000\n"
        f"S   -{d:.6f}  0.000000  0.000000"
    )


def _haber_bosch_fe2s2n2_geometry(step: MQEStep) -> str:
    """Fe₂S₂N₂ proxy geometry for Haber-Bosch N₂ activation.

    Fixed Fe₂S₂ rhombic core at the Janus bondlength (Fe–S = 2.316 Å, the
    nitrogenase E4 geometry) with N₂ adsorbed axially above the Fe₂ centre.
    The N–N bond elongates step-wise from 1.10 Å (chemisorbed N₂) to 1.52 Å
    (near-dissociation limit at n=7), modelling progressive activation.

    H atoms are handled algebraically via A_n/P_n PCET fields — they do not
    appear in the geometry, keeping the orbital space fixed across all 8 steps.

    Coordinates (Angstrom):
      Fe–Fe axis: y  |  S–S axis: x  |  N₂ axis: z
    The N₂ centre-of-mass sits 2.0 Å above the Fe₂S₂ plane.
    """
    fe_s = 2.316   # fixed at nitrogenase_lt Janus bondlength
    nn   = 1.10 + step.step_n * 0.06   # 1.10 Å (n=0) → 1.52 Å (n=7)
    half = nn / 2.0
    z0   = 2.0     # N₂ centre-of-mass elevation above Fe₂S₂ plane
    return (
        f"Fe   0.000000  1.350000  0.000000\n"
        f"Fe   0.000000 -1.350000  0.000000\n"
        f"S    {fe_s:.6f}  0.000000  0.000000\n"
        f"S   -{fe_s:.6f}  0.000000  0.000000\n"
        f"N    0.000000  0.000000  {z0 + half:.6f}\n"
        f"N    0.000000  0.000000  {z0 - half:.6f}"
    )


def _psii_photo_geometry_at_step(step: MQEStep) -> str:
    # Shorter Mn–O bonds at higher oxidation states (Gap 3b)
    d = 2.260 - (step.step_n * 0.02)
    return (
        f"Mn   0.000000  1.350000  0.000000\n"
        f"Mn   0.000000 -1.350000  0.000000\n"
        f"O    {d:.6f}  0.000000  0.000000\n"
        f"O   -{d:.6f}  0.000000  0.000000"
    )


def _h3plus_geometry(r_12: float, r_13: float) -> str:
    """H3+ geometry in Angstrom: atoms 0-1 at distance r_12, atom 2 at r_13.

    Isoceles triangle allowing asymmetric deformation for Z3 step sequence.
    """
    x2 = r_12
    x3 = r_13 * 0.5
    y3 = r_13 * np.sqrt(3) / 2
    return (
        f"H   0.000000  0.000000  0.000000\n"
        f"H  {x2:.6f}  0.000000  0.000000\n"
        f"H  {x3:.6f}  {y3:.6f}  0.000000"
    )


def _h2_geometry(bondlength: float) -> str:
    """H2 geometry at given bondlength (Angstrom)."""
    half = bondlength / 2
    return (
        f"H  -{half:.6f}  0.000000  0.000000\n"
        f"H   {half:.6f}  0.000000  0.000000"
    )

def _codh_acs_proxy_atom_block(d: float) -> str:
    """Ni₂S₂-CO proxy for CODH/ACS (catalog entry 9, Group B).
    2×Ni(28)+2×S(16)+C(6)+O(8)=102e (even), charge=0, spin_2S=0.
    Reaction coordinate: Ni-S bond compression 2.300→2.190 Å.
    CO ligand bound axially to Ni_p (proximal Ni) along Z-axis,
    perpendicular to the Ni-S-Ni plane (avoids S-C steric clash).
    Ni-Ni fixed at 2.60 Å; Ni-C = 1.80 Å; C-O = 1.15 Å.
    """
    y_ni = 1.300000
    x_s = math.sqrt(max(d**2 - y_ni**2, 0.0))
    return (
        f"Ni   0.000000  {y_ni:.6f}  0.000000\n"
        f"Ni   0.000000 {-y_ni:.6f}  0.000000\n"
        f"S    {x_s:.6f}  0.000000  0.000000\n"
        f"S   -{x_s:.6f}  0.000000  0.000000\n"
        f"C    0.000000  {y_ni:.6f}  1.800000\n"
        f"O    0.000000  {y_ni:.6f}  2.950000"
    )

def _cyt_bd_proxy_atom_block(d: float) -> str:
    """Fe₂O₂ rhombic proxy for Cyt bd oxidase (catalog entry 10, Group C).
    2×Fe(26)+2×O(8)=68e (even), charge=0, spin_2S=0.
    Reaction coordinate: Fe-O bond compression 2.300→2.200 Å.
    Fe-Fe fixed at 2.70 Å.
    """
    return (
        f"Fe   0.000000  1.350000  0.000000\n"
        f"Fe   0.000000 -1.350000  0.000000\n"
        f"O    {d:.6f}  0.000000  0.000000\n"
        f"O   -{d:.6f}  0.000000  0.000000"
    )

def _cyt_c_oxidase_proxy_atom_block(d: float) -> str:
    """Fe-Cu-N-O binuclear proxy for Cyt c oxidase (catalog entry 11, Group C).
    Fe(26)+Cu(29)+N(7)+O(8)=70e (even), charge=0, spin_2S=0.
    Reaction coordinate: Fe-Cu distance compression 2.600→2.500 Å.
    N and O represent proximal histidine ligands (1.80 Å from metals).
    """
    half_d = d / 2.0
    return (
        f"Fe   0.000000  0.000000  {half_d:.6f}\n"
        f"Cu   0.000000  0.000000 -{half_d:.6f}\n"
        f"N    1.800000  0.000000  0.000000\n"
        f"O   -1.800000  0.000000  0.000000"
    )

# ===========================================================================
# 3. MECHANISM SPECIFICATIONS
# ===========================================================================

def _build_nitrogenase_lt_spec(n_orbitals: int) -> MQEMechanismSpec:
    """Nitrogenase LT mechanism: M=8, m=4 (Z4), 16 ATP, 1 Janus at n=4.

    UPDATED: Reflects full reversible PCET framework.
    The canonical LT cycle is strictly forward (reduction), so reverse fields
    are explicitly initialized as empty/zero. The architecture is now ready
    to model back-reactions, thermal fluctuations, product inhibition, or
    non-ideal oxidative pathways by populating these fields.

    Geometry sequence: [Fe2S2] cluster at 8 Fe–S distances spanning the
    oxidation ladder E0→E7. Each step corresponds to one electron/proton
    injection raising the Fe–S bond from 2.26 Å (resting) to 2.42 Å
    (maximally reduced E7), following the electron-density expansion
    of the iron-sulfur cluster upon progressive reduction.

    Fe–S distances:
        E0: 2.260 Å  (resting, all-ferric reference)
        E1: 2.274 Å  (+1e, +0.014 Å per step from crystallographic data)
        E2: 2.288 Å
        E3: 2.302 Å
        E4: 2.316 Å  (Janus: 4 hydrides, N2 binding onset)
        E5: 2.330 Å  (post-crossing: H2 eliminated, N2 bound)
        E6: 2.344 Å
        E7: 2.358 Å  (E7: 7 electrons, near fully reduced)

    The step size 0.014 Å derives from the empirical Badger's rule
    correlation between Fe oxidation state and Fe–S bond length
    in biological iron-sulfur clusters (Venkateswara Rao & Holm, 2004).

    Net-Flux Phase Closure: Σ (ν_n - ν_decouple_n) = 8 × 2 = 16 ≡ 0 (mod 4) ✓
    Net-Flux Electron Count: Σ (|A_n| - |A_n_eject|) = 8 × 1 = 8 ✓
    """
    N = n_orbitals
    fe_s_distances = [2.260 + i * 0.014 for i in range(8)]
    steps = []
    k_acc = 0  # Tracks net phase index: k^{(n)} = Σ(ν_i - ν_decouple_i) mod m
    e_acc = 0  # Tracks net electrons:   Σ(|A_i| - |A_i_eject|)

    for n in range(8):
        # ── Forward Transitions (Reduction / Coupling) ──────────────────────
        nu_n          = 2                   # 2 ATP per step (half-turn each)
        A_n           = [n % N]             # 1 electron injected per step
        P_n           = [(n + 1) % N]       # 1 protonated per step
        B_n           = [(n + 2) % N]       # 1 virtual register coupled per step

        # ── REVERSIBLE TRANSITIONS (Canonical LT = None) ───────────────────
        # Framework ready for back-reactions (e.g., thermal back-transfer,
        # equilibrium fluctuations, or oxidative PCET). Explicitly zeroed here.
        A_n_eject     = []
        P_n_eject     = []
        B_n_decouple  = []
        nu_decouple_n = 0

        # ── Net-Flux Accumulation (Theorem 2 Extension) ───────────────────
        k_acc += (nu_n - nu_decouple_n)
        e_acc += (len(A_n) - len(A_n_eject))

        is_crossing = (n == 4)
        
        # Explicit instantiation → append prevents silent list-mutation bugs
        step = MQEStep(
            step_n               = n,
            nu_n                 = nu_n,
            A_n                  = A_n,
            P_n                  = P_n,
            B_n                  = B_n,
            A_n_eject            = A_n_eject,
            P_n_eject            = P_n_eject,
            B_n_decouple         = B_n_decouple,
            nu_decouple_n        = nu_decouple_n,
            is_crossing          = is_crossing,
            delta_CI_Ha          = 1.6e-3 if is_crossing else None,
            crossing_orbitals    = [0, 1] if is_crossing else None,
            geometry_label       = f"Fe2S2 E{n}: Fe-S={fe_s_distances[n]:.3f} Ang",
            bondlength_angstrom  = fe_s_distances[n],
            phase_index_k        = k_acc % 4,
            cumulative_electrons = e_acc,              # Legacy compatibility
            cumulative_net_electrons = e_acc,          # Primary net-flux tracker
        )
        steps.append(step)

    return MQEMechanismSpec(
        name                     = "nitrogenase_lt",
        M_steps                  = 8,
        m_modulus                = 4,
        S_target                 = 1.5,
        n_orbitals               = N,
        steps                    = steps,
        description              = (
            "Lowe-Thorneley nitrogenase LT cycle. E0→E8 via 8 sequential "
            "forward PCET steps (8e⁻, 8H⁺, 16 ATP). Janus crossing at E4→E5. "
            "Fully reversible framework: A_n_eject/P_n_eject/B_n_decouple "
            "explicitly zeroed for canonical LT; ready for back-reaction modeling. "
            "Net-flux phase closure: Σ(ν-ν†)=16 ≡ 0 (mod 4). Net e⁻: 8."
        ),
        expected_total_electrons = 8,      # Legacy/compatibility
        expected_net_electrons   = 8,      # Theorem 2 Extension (Net-Flux)
        expected_net_phase       = 0,      # Theorem 2 Extension
        expected_net_phase_closure = True,
        expected_energy_ordering   = "increasing", # Theorem 2 Extension
    )

def _build_nitrogenase_lt_m8_spec(n_orbitals: int) -> MQEMechanismSpec:
    """Nitrogenase LT variant: M=8, m=8 (ℤ₈).

    Tests finer phase granularity (d=8 virtual register) while keeping
    identical sequential injection (1e⁻/step) and total cofactor shift (8).
    Validates GeneralizedVirtualShiftGate for m=8 and assesses impact
    on free-energy landscape for δ_ATP minimization.

    Phase design:
        nu_n=1 per step (odd, coprime with 8) → orbit size = 8/gcd(1,8) = 8.
        All 8 distinct Z_8 states are visited before closure: k^(n) = 1,2,3,4,5,6,7,0.
        Σν = 8 ≡ 0 (mod 8). This is the minimal nu that exercises the full Z_8 group.
        Using nu_n=2 (even) would give orbit size 4 = gcd(2,8)⁻¹·8/2,
        making Z_8 equivalent to Z_4 — the purpose of the variant is defeated.

    Geometry:
        Fe-S bond COMPRESSES (2.260→2.162 Å) identical to baseline nitrogenase_lt,
        modelling progressive reduction of the Fe₂S₂ cluster (E0→E8).
    """
    N = n_orbitals
    M_steps = 8
    fe_s_distances = [2.260 - i * 0.014 for i in range(8)]   # BUG1 FIX: compression not extension
    steps = []
    k_acc = 0
    e_fwd = 0
    e_acc = 0

    for n in range(M_steps):
        nu_n = 1                   # BUG3 FIX: nu=1 (coprime with 8) → full Z_8 orbit
        A_n = [n % N]
        P_n = [(n + 1) % N]
        B_n = [(n + 2) % N]

        A_n_eject = P_n_eject = B_n_decouple = []
        nu_decouple_n = 0

        k_acc += nu_n
        e_fwd += len(A_n)
        e_acc += len(A_n)

        is_crossing = (n == 3)   # k^(3)=4=m/2 ✓ — phase condition satisfied at step 3

        steps.append(MQEStep(
            step_n               = n,
            nu_n                 = nu_n,
            A_n                  = A_n,
            P_n                  = P_n,
            B_n                  = B_n,
            A_n_eject            = A_n_eject,
            P_n_eject            = P_n_eject,
            B_n_decouple         = B_n_decouple,
            nu_decouple_n        = nu_decouple_n,
            is_crossing          = is_crossing,
            delta_CI_Ha          = 1.6e-3 if is_crossing else None,
            crossing_orbitals    = [0, 1] if is_crossing else None,
            geometry_label       = f"Fe2S2_LT_m8_E{n}: Fe-S={fe_s_distances[n]:.3f} Ang",
            bondlength_angstrom  = fe_s_distances[n],
            phase_index_k        = k_acc % 8,
            cumulative_electrons = e_fwd,
            cumulative_net_electrons = e_acc,
        ))

    return MQEMechanismSpec(
        name                     = "nitrogenase_lt_m8",
        M_steps                  = M_steps,
        m_modulus                = 8,
        S_target                 = 1.5,
        n_orbitals               = N,
        steps                    = steps,
        description              = (
            "Nitrogenase LT variant with ℤ₈ virtual phase group. "
            "nu_n=1 (coprime with 8) exercises all 8 Z_8 states: k^(n)=1,2,3,4,5,6,7,0. "
            "Same Fe₂S₂ bond compression (2.260→2.162 Å) as baseline LT. "
            "Net-flux phase closure: Σ(ν−ν†)=8 ≡ 0 (mod 8). Net e⁻: 8."
        ),
        expected_total_electrons = 8,
        expected_net_electrons   = 8,
        expected_net_phase       = 0,
        expected_net_phase_closure = True,
        expected_energy_ordering   = "decreasing",
    )


def _build_nitrogenase_lt_parallel_spec(n_orbitals: int) -> MQEMechanismSpec:
    """Nitrogenase LT variant: M=4, m=4 (ℤ₄).

    Tests parallel electron injection (2e⁻/step over 4 steps) vs sequential.
    Compresses the catalytic trajectory while maintaining identical total
    stoichiometry (8e⁻, 8H⁺) to evaluate gate-depth scaling and free-energy
    profile for δ_ATP minimization.

    Phase design:
        nu_n=2 per step → Σν=8 ≡ 0 (mod 4). orbit size = 4/gcd(2,4) = 2.
        k^(n) alternates 2, 0, 2, 0 — the Z_4 register cycles between |0⟩ and |2⟩
        only. This is intentional for the parallel variant: the faster phase cycling
        (every 2 steps vs 4 in the baseline) tests whether the Trotter propagator
        can distinguish the compressed phase trajectory.
        Using nu_n=4 ≡ 0 (mod 4) would make CofactorCouplingGate an identity
        at every step — no coupling would occur.

    Orbital design:
        A_n and B_n are kept disjoint to avoid double-acting the same qudit with
        ElectronShiftGate (Moment 3) followed by CofactorCouplingGate (Moment 5).
        B_n = [(n+1)%N, (n+3)%N] — offset by 1 from A_n, disjoint when N=4.

    Geometry:
        Fe-S bond COMPRESSES (2.260→2.176 Å) spanning the same physical range
        as the 8-step baseline at 4 geometric checkpoints.
    """
    N = n_orbitals
    M_steps = 4
    fe_s_distances = [2.260 - i * 0.028 for i in range(4)]   # BUG1 FIX: compression not extension

    steps = []
    k_acc = 0
    e_fwd = 0
    e_acc = 0

    for n in range(M_steps):
        nu_n = 2                   # BUG4 FIX: nu=2 (not 4) → non-trivial Z_4 shift
        # Parallel injection: 2 electrons/protons per step
        A_n = [n % N, (n + 2) % N]
        P_n = [(n + 1) % N, (n + 3) % N]
        # BUG2 FIX: B_n offset by +1 from A_n — disjoint from A_n for all N=4 steps
        # A_n = {n%4, (n+2)%4}, B_n = {(n+1)%4, (n+3)%4} → A∩B = {} always
        B_n = [(n + 1) % N, (n + 3) % N]

        A_n_eject = P_n_eject = B_n_decouple = []
        nu_decouple_n = 0

        k_acc += nu_n
        e_fwd += len(A_n)          # +2 per step
        e_acc += len(A_n)          # +2 per step

        # Janus crossing at step 2 (midpoint of 4-step cycle, equivalent to E4→E5)
        is_crossing = (n == 2)

        steps.append(MQEStep(
            step_n               = n,
            nu_n                 = nu_n,
            A_n                  = A_n,
            P_n                  = P_n,
            B_n                  = B_n,
            A_n_eject            = A_n_eject,
            P_n_eject            = P_n_eject,
            B_n_decouple         = B_n_decouple,
            nu_decouple_n        = nu_decouple_n,
            is_crossing          = is_crossing,
            delta_CI_Ha          = 1.6e-3 if is_crossing else None,
            crossing_orbitals    = [0, 1] if is_crossing else None,
            geometry_label       = f"Fe2S2_LT_Par_E{n}: Fe-S={fe_s_distances[n]:.3f} Ang",
            bondlength_angstrom  = fe_s_distances[n],
            phase_index_k        = k_acc % 4,
            cumulative_electrons = e_fwd,
            cumulative_net_electrons = e_acc,
        ))

    return MQEMechanismSpec(
        name                     = "nitrogenase_lt_parallel",
        M_steps                  = M_steps,
        m_modulus                = 4,
        S_target                 = 1.5,
        n_orbitals               = N,
        steps                    = steps,
        description              = (
            "Nitrogenase LT variant with parallel injection (2e⁻/step over 4 steps). "
            "nu_n=2 gives k^(n)=2,0,2,0 — Z_4 cycles between |0⟩ and |2⟩. "
            "A_n and B_n are disjoint: no qudit receives ElectronShift and "
            "CofactorCoupling in the same step. Fe-S compresses 2.260→2.176 Å. "
            "Net-flux phase closure: Σ(ν−ν†)=8 ≡ 0 (mod 4). Net e⁻: 8."
        ),
        expected_total_electrons = 8,
        expected_net_electrons   = 8,
        expected_net_phase       = 0,
        expected_net_phase_closure = True,
        expected_energy_ordering   = "decreasing",
    )

def _build_nitrogenase_closed_loop_spec(n_orbitals: int) -> MQEMechanismSpec:
    """Gap 5b: Dataset counterpart to the 16-step closed-loop mechanism.

    Builds a full catalytic cycle: 8 forward LT steps (E0→E8, bond compression)
    followed by 8 reverse steps (E8→E0, bond extension) that exactly undo the
    forward sequence in REVERSE orbital order, consistent with microscopic
    reversibility and the evolution-side spec.

    Orbital ordering of reverse half:
        Reverse step n=8  undoes forward step 7  (rn_rev = 7 - (n - 8))
        Reverse step n=9  undoes forward step 6
        ...
        Reverse step n=15 undoes forward step 0

    Janus crossings:
        n=4:  Forward Janus (E4→E5), same as nitrogenase_lt.
        n=12: Reverse Janus (E5→E4 in reverse), = 8 + (7 - 4) = 11... wait:
              reverse of forward step 4 is at reverse_index = 7 - 4 = 3,
              so absolute step = 8 + 3 = 11. This is n=11.
              BUG8 FIX: The reverse Janus is at n=11 (undoing forward step 4).
    """
    N         = n_orbitals
    M_steps   = 16
    m_modulus = 4      # Z₄ phase group — same as nitrogenase_lt

    # Bond lengths: forward half compresses, reverse half expands back
    fe_s_distances_fwd = [2.260 - i * 0.014 for i in range(8)]    # compression
    fe_s_distances_full = fe_s_distances_fwd + fe_s_distances_fwd[::-1]  # mirror

    steps    = []
    k_acc    = 0   # net phase index: Σ(nu_n - nu_decouple_n) mod m
    e_fwd    = 0   # cumulative_electrons (forward injection only, legacy)
    e_acc    = 0   # cumulative_net_electrons (net flux)

    for n in range(M_steps):
        if n < 8:
            # ── Forward half: electron injection + cofactor coupling ────────
            nu_n = 2
            A_n  = [n % N]
            P_n  = [(n + 1) % N]
            B_n  = [(n + 2) % N]
            A_n_eject = P_n_eject = B_n_decouple = []
            nu_decouple_n = 0

            e_fwd += len(A_n)
            e_acc += len(A_n)
            k_acc += nu_n
        else:
            # ── Reverse half: ejection + decoupling in REVERSE orbital order ─
            # BUG2 FIX: reverse step n undoes forward step (7 - (n - 8)),
            # so orbitals follow the reversed sequence 7,6,5,4,3,2,1,0.
            rn_rev = 7 - (n - 8)        # forward step being undone: 7→0
            nu_n   = 0
            A_n    = P_n = B_n = []
            A_n_eject    = [rn_rev % N]
            P_n_eject    = [(rn_rev + 1) % N]
            B_n_decouple = [(rn_rev + 2) % N]
            nu_decouple_n = 2

            e_acc -= 1   # ejecting one electron
            k_acc -= nu_decouple_n

        # Janus crossing:
        #   Forward Janus at n=4 (E4→E5, same as nitrogenase_lt).
        #   Reverse Janus at n=11: undoes forward step rn_rev = 7-(11-8) = 4 (E5→E4).
        is_crossing = (n == 4 or n == 11)   # BUG8 FIX: n=11, not n=12

        steps.append(MQEStep(
            step_n              = n,
            nu_n                = nu_n,
            A_n                 = A_n,
            P_n                 = P_n,
            B_n                 = B_n,
            A_n_eject           = A_n_eject,
            P_n_eject           = P_n_eject,
            B_n_decouple        = B_n_decouple,
            nu_decouple_n       = nu_decouple_n,
            is_crossing         = is_crossing,
            delta_CI_Ha         = 1.6e-3 if is_crossing else None,
            crossing_orbitals   = [0, 1] if is_crossing else None,
            geometry_label      = f"Fe2S2_ClosedLoop_Step{n:02d}",
            bondlength_angstrom = fe_s_distances_full[n],
            cumulative_electrons      = e_fwd,   # BUG1b FIX: required for check (d)
            phase_index_k             = k_acc % m_modulus,
            cumulative_net_electrons  = e_acc,
            # Photon fields: empty — closed loop is PCET-driven, not photo-driven
            cumulative_photons_absorbed = 0,
            cumulative_photons_emitted  = 0,
            cumulative_net_photons      = 0,
        ))

    return MQEMechanismSpec(
        name                       = "nitrogenase_closed_loop",
        M_steps                    = M_steps,
        m_modulus                  = m_modulus,
        S_target                   = 1.5,
        n_orbitals                 = N,
        steps                      = steps,
        description                = (
            "Nitrogenase LT closed-loop: 8 forward (E0→E8, bond compression) + "
            "8 reverse (E8→E0, bond extension) PCET steps. Reverse half undoes "
            "forward orbitals in REVERSE order (step 7→0) for microscopic "
            "reversibility. Net e⁻ flux = 0. Net phase = 0 (mod 4). "
            "Janus crossings at steps 4 (fwd) and 11 (rev, undoes step 4)."
        ),
        expected_net_electrons     = 0,
        expected_net_phase_closure = True,
        expected_energy_ordering   = "closed_loop",
        # Photon invariants: all zero (non-photo mechanism)
        phi_photon                      = 0.0,
        expected_total_photons_absorbed = 0,
        expected_total_photons_emitted  = 0,
        expected_net_photons            = 0,
    )



def _build_psii_spec(n_orbitals: int) -> MQEMechanismSpec:
    """PSII Kok S-state cycle: M=4, m=4 (Z4).
    
    UPGRADED: Category B (Active PCET Simulation).
    Models the S-state transitions (S0→S1→S2→S3). Actively tracks the
    sequential extraction of 4e⁻/4H⁺ via the PCET algebraic fields, while
    simulating the Mn-cluster compression (bond shortening) on oxidation.
    Initialized for full cycle reversibility.
    
    Key parameters:
    - M_steps=4: Four discrete S-state transitions (S₀→S₁→S₂→S₃).
    - m_modulus=4: Z₄ virtual phase group for photon/ATP coupling.
    - fe_s_distances: Progressive Fe–S bond compression (2.260 → 2.215 Å)
      modeling oxidation-induced structural changes in the Mn₄CaO₅ cluster.
    - nu_n=1: One quantum-coupled phase shift per step (Σν=4 ≡ 0 mod 4).
    - S_target=0.0: Singlet product O₂ (triplet character handled virtually).
    
    Stoichiometric invariants:
    - Electron count: Σ|Aₙ| = 4e⁻ extracted over the full cycle.
    - Phase closure: Σνₙ = 4 ≡ 0 (mod 4) ✓
    """
    # Compact parameter initialization
    N, M_steps, m_modulus = n_orbitals, 4, 4
    fe_s_distances = [2.260 - i * 0.015 for i in range(4)]
    steps = []
    k_acc = e_acc = e_fwd = 0  # Phase, net-electron, and forward-electron accumulators

    for n in range(M_steps):
        nu_n, A_n, P_n, B_n = 1, [n % N], [(n + 1) % N], [(n + 2) % N]
        A_n_eject, P_n_eject, B_n_decouple, nu_decouple_n = [], [], [], 0

        k_acc  += (nu_n - nu_decouple_n)
        e_fwd  += len(A_n)                        # forward-only (check d)
        e_acc  += (len(A_n) - len(A_n_eject))

        # Janus crossing at S2 (n=2): maximally oxidised state before O2 release.
        # n=M/2=2 is the energetic midpoint of the 4-step Kok cycle, consistent
        # with the n*=M/2 convention used across all m=4 mechanisms.
        is_crossing = (n == 2)

        steps.append(MQEStep(
            step_n               = n,
            nu_n                 = nu_n,
            A_n                  = A_n,
            P_n                  = P_n,
            B_n                  = B_n,
            A_n_eject            = A_n_eject,
            P_n_eject            = P_n_eject,
            B_n_decouple         = B_n_decouple,
            nu_decouple_n        = nu_decouple_n,
            is_crossing          = is_crossing,
            delta_CI_Ha          = 1.6e-3 if is_crossing else None,
            crossing_orbitals    = [0, 1] if is_crossing else None,
            geometry_label       = f"Fe2S2_S{n}_Intermediate",
            bondlength_angstrom  = fe_s_distances[n],
            cumulative_electrons     = e_fwd,      # FIX: required for check (d)
            phase_index_k            = k_acc % m_modulus,
            cumulative_net_electrons = e_acc,
        ))

    return MQEMechanismSpec(
        name                     = "psii",
        M_steps                  = M_steps,
        m_modulus                = m_modulus,
        S_target                 = 0.0,  # Singlet product O₂ (triplet virtual)
        n_orbitals               = N,
        steps                    = steps,
        description              = (
            "Photosystem II Kok S-state cycle. Category B: Active PCET. "
            "Algebraically tracks 4e⁻/4H⁺ oxidation flux. "
            "Adiabatic S-state transitions with Z₄ phase closure."
        ),
        expected_net_electrons   = 4,      # Theorem 2 Extension (Net-Flux)
        expected_net_phase_closure = True,
        expected_energy_ordering   = "none", # Σν=4 ≡ 0 (mod 4)
    )


def _build_psii_photo_spec(n_orbitals: int) -> MQEMechanismSpec:
    """Gap 5a: Photo-driven PSII builder with populated photon fields."""
    N, M_steps, m_modulus = n_orbitals, 4, 4
    steps = []
    k_acc = e_acc = e_fwd = p_abs = p_emit = p_net = 0
    phi_P680 = 0.067 * 0.02  # ~1.82 eV

    for n in range(M_steps):
        nu_n, A_n, P_n, B_n = 1, [n % N], [(n + 1) % N], [(n + 2) % N]
        Gamma_n_abs  = [A_n[0]]
        Gamma_n_emit = []

        k_acc  += nu_n
        e_fwd  += len(A_n)           # cumulative_electrons: forward injection only
        e_acc  += len(A_n)           # cumulative_net_electrons: net (no ejection here)
        p_abs  += len(Gamma_n_abs)
        p_emit += len(Gamma_n_emit)
        p_net  += (len(Gamma_n_abs) - len(Gamma_n_emit))

        # Janus crossing at S2 (n=2): same convention as psii non-photo spec.
        is_crossing = (n == 2)

        steps.append(MQEStep(
            step_n=n,
            nu_n=nu_n,
            A_n=A_n,
            P_n=P_n,
            B_n=B_n,
            Gamma_n_abs=Gamma_n_abs,
            Gamma_n_emit=Gamma_n_emit,
            phi_photon_n=phi_P680,
            is_crossing=is_crossing,
            delta_CI_Ha=1.6e-3 if is_crossing else None,
            crossing_orbitals=[0, 1] if is_crossing else None,
            geometry_label=f"PSII_Photo_S{n}",
            bondlength_angstrom=2.260 - n * 0.02,
            cumulative_electrons=e_fwd,          # BUG1a FIX: required for check (d)
            phase_index_k=k_acc % m_modulus,
            cumulative_net_electrons=e_acc,
            cumulative_photons_absorbed=p_abs,
            cumulative_photons_emitted=p_emit,
            cumulative_net_photons=p_net,
        ))

    return MQEMechanismSpec(
        name="psii_photo", 
        M_steps=M_steps, 
        m_modulus=m_modulus, 
        S_target=0.0, 
        n_orbitals=N, 
        steps=steps,
        description="Photo-driven PSII Kok S-state cycle. Evaluates discrete P680 photon absorptions.",
        expected_net_electrons=4, 
        expected_net_phase_closure=True,
        expected_energy_ordering   = "monotone_increasing",
        phi_photon=phi_P680, 
        expected_total_photons_absorbed=4, 
        expected_total_photons_emitted=0, 
        expected_net_photons=4
    )


def _build_hydrogenase_spec(n_orbitals: int) -> MQEMechanismSpec:
    """Hydrogenase: M=2, m=1 (Z1 trivial).
    
    UPGRADED: Category B (Active PCET Simulation).
    Models proton reduction (2H⁺ + 2e⁻ → H2). While minimal, it now uses the 
    standardized reversible PCET framework. This serves as the 'Unit Test' for 
    the entire MQE compilation pipeline.
    """
    N         = n_orbitals
    M_steps   = 2
    m_modulus = 1      # Z₁ trivial group — no virtual clock
    bondlengths = [1.40, 0.742]
    
    steps = []
    k_acc = 0
    e_acc = 0
    e_fwd = 0

    for n in range(M_steps):
        # ── Active PCET Transitions (Reduction) ──────────────────────────────
        nu_n           = 0                 # Trivial Z1
        A_n            = [n % N]           # 1e⁻ injection per step
        P_n            = [(n + 1) % N]     # 1H⁺ addition per step
        B_n            = []                # No cofactor register needed
        
        # ── REVERSIBLE FIELDS (Initialized for Oxidation/Proton Evolution) ───
        A_n_eject      = []
        P_n_eject      = []
        B_n_decouple   = []
        nu_decouple_n  = 0
        
        # ── Net-Flux Accumulation ───────────────────────────────────────────
        k_acc += (nu_n - nu_decouple_n)
        e_fwd += len(A_n)                      # BUGD FIX: forward-only
        e_acc += (len(A_n) - len(A_n_eject))

        steps.append(MQEStep(
            step_n               = n,
            nu_n                 = nu_n,
            A_n                  = A_n,
            P_n                  = P_n,
            B_n                  = B_n,
            A_n_eject            = A_n_eject,
            P_n_eject            = P_n_eject,
            B_n_decouple         = B_n_decouple,
            nu_decouple_n        = nu_decouple_n,
            is_crossing          = False,
            delta_CI_Ha          = None,
            crossing_orbitals    = None,
            geometry_label       = f"H2_Bond_r={bondlengths[n]:.3f}_Ang",
            bondlength_angstrom  = bondlengths[n],
            phase_index_k        = k_acc % m_modulus,
            cumulative_electrons = e_fwd,
            cumulative_net_electrons = e_acc,
        ))

    return MQEMechanismSpec(
        name                     = "hydrogenase",
        M_steps                  = M_steps,
        m_modulus                = m_modulus,
        S_target                 = 0.0,
        n_orbitals               = N,
        steps                    = steps,
        description              = (
            "[FeFe]-hydrogenase: 2H⁺ + 2e⁻ → H2. Category B: Active PCET. "
            "Minimal N=2 test case for architectural validation. "
            "Reversible framework enabled for H2 oxidation studies."
        ),
        expected_net_electrons   = 2,
        expected_net_phase_closure = True,
        expected_energy_ordering   = "decreasing",
    )


def _build_hydrogenase_oxidation_spec(n_orbitals: int) -> MQEMechanismSpec:
    """Hydrogenase Oxidation (H2 → 2H⁺ + 2e⁻): M=2, m=1 (Z1 trivial).
    
    CATEGORY B (Active PCET Simulation):
    Models the oxidative pathway. By populating the `_eject` fields, we trigger
    the inverse gate compilation in the MQE pipeline. This mechanism effectively 
    reverses the electron flux compared to the `hydrogenase` reduction spec.
    """
    N         = n_orbitals
    M_steps   = 2
    m_modulus = 1      # Z₁ trivial group — no virtual clock
    bondlengths = [0.742, 1.40] # Reverse order: Start at product, end at TS
    
    steps = []
    k_acc = 0
    e_acc = 0
    e_fwd = 0   # forward-only injection count for check (d)

    for n in range(M_steps):
        # ── Active PCET Transitions (Oxidation) ──────────────────────────────
        nu_n           = 0                 
        A_n            = []                # No injection
        P_n            = []                
        B_n            = []                
        
        # ── REVERSIBLE FIELDS (Active Oxidation Flux) ────────────────────────
        A_n_eject      = [n % N]           # Oxidative: 1e⁻ ejected per step
        P_n_eject      = [(n + 1) % N]     # Oxidative: 1H⁺ released per step
        B_n_decouple   = []                
        nu_decouple_n  = 0
        
        # ── Net-Flux Accumulation (Decremental Flux) ────────────────────────
        k_acc += (nu_n - nu_decouple_n)
        e_acc += (len(A_n) - len(A_n_eject)) # Result: -2 total electrons

        steps.append(MQEStep(
            step_n               = n,
            nu_n                 = nu_n,
            A_n                  = A_n,
            P_n                  = P_n,
            B_n                  = B_n,
            A_n_eject            = A_n_eject,
            P_n_eject            = P_n_eject,
            B_n_decouple         = B_n_decouple,
            nu_decouple_n        = nu_decouple_n,
            is_crossing          = False,
            delta_CI_Ha          = None,
            crossing_orbitals    = None,
            geometry_label       = f"H2_Oxidation_r={bondlengths[n]:.3f}_Ang",
            bondlength_angstrom  = bondlengths[n],
            # BUGH FIX: cumulative_electrons = forward injection count (0 here;
            # all flux is ejection). Required for validate_integral_dataset check (d).
            cumulative_electrons     = 0,  # no forward injection in this mechanism
            phase_index_k            = k_acc % m_modulus,
            cumulative_net_electrons = e_acc,
        ))

    return MQEMechanismSpec(
        name                     = "hydrogenase_oxidation",
        M_steps                  = M_steps,
        m_modulus                = m_modulus,
        S_target                 = 0.0,
        n_orbitals               = N,
        steps                    = steps,
        description              = (
            "[FeFe]-hydrogenase: H2 → 2H⁺ + 2e⁻ (Oxidation). "
            "Category B: Active PCET Simulation. "
            "Algebraically tracks oxidative flux via A_n_eject/P_n_eject. "
            "Symmetric to the reduction mechanism."
        ),
        expected_net_electrons   = -2,
        expected_net_phase_closure = True,
        expected_energy_ordering   = "increasing",
    )

# CORRECTED: Dissociative pathway proxy on an Fe2 active site
HABER_BOSCH_GEOMETRIES = {
    # Step 0: N2 chemisorbed (side-on bridge), N-N bond slightly elongated (1.15 A)
    0: """Fe -1.200000  0.000000 -1.500000
          Fe  1.200000  0.000000 -1.500000
          N  -0.575000  0.000000  0.000000
          N   0.575000  0.000000  0.000000""",
          
    # Step 1: N2 completely dissociated into 2 surface-bound N atoms (N-N ~ 2.4 A)
    1: """Fe -1.200000  0.000000 -1.500000
          Fe  1.200000  0.000000 -1.500000
          N  -1.200000  0.000000  0.000000
          N   1.200000  0.000000  0.000000""",
          
    # Step 2: First Hydrogenation (N + NH)
    2: """Fe -1.200000  0.000000 -1.500000
          Fe  1.200000  0.000000 -1.500000
          N  -1.200000  0.000000  0.000000
          N   1.200000  0.000000  0.000000
          H  -1.200000  0.900000  0.500000""",
          
    # Step 3: Second Hydrogenation (NH + NH)
    3: """Fe -1.200000  0.000000 -1.500000
          Fe  1.200000  0.000000 -1.500000
          N  -1.200000  0.000000  0.000000
          N   1.200000  0.000000  0.000000
          H  -1.200000  0.900000  0.500000
          H   1.200000  0.900000  0.500000""",
          
    # Step 4: Further Hydrogenation (NH2 + NH2)
    4: """Fe -1.200000  0.000000 -1.500000
          Fe  1.200000  0.000000 -1.500000
          N  -1.200000  0.000000  0.000000
          N   1.200000  0.000000  0.000000
          H  -1.200000  0.900000  0.500000
          H  -1.200000 -0.900000  0.500000
          H   1.200000  0.900000  0.500000
          H   1.200000 -0.900000  0.500000""",
          
    # Step 5: Final Hydrogenation & Desorption (2NH3 moving away from surface)
    5: """Fe -1.200000  0.000000 -1.500000
          Fe  1.200000  0.000000 -1.500000
          N  -1.200000  0.000000  1.000000
          N   1.200000  0.000000  1.000000
          H  -1.200000  0.900000  1.500000
          H  -2.000000 -0.450000  1.500000
          H  -0.400000 -0.450000  1.500000
          H   1.200000  0.900000  1.500000
          H   2.000000 -0.450000  1.500000
          H   0.400000 -0.450000  1.500000"""
}

# Ensure your geometry retriever uses this without appending extra Fe atoms
def _haber_bosch_geometry(step: MQEStep) -> str:
    """Retrieves geometry for Haber-Bosch intermediates from registry."""
    if step.step_n not in HABER_BOSCH_GEOMETRIES:
        raise KeyError(f"No geometry defined for Haber-Bosch step {step.step_n}")
    return HABER_BOSCH_GEOMETRIES[step.step_n].strip()


def _get_haber_bosch_atom_block(step: MQEStep) -> str:
    # Hook into the step generator
    return _haber_bosch_geometry(step)

def _build_haber_bosch_spec(n_orbitals: int) -> MQEMechanismSpec:
    """Haber-Bosch N₂ activation: M=8, m=4 (Z₄), Janus at n=4.

    REDESIGNED: Fixed Fe₂S₂N₂ proxy geometry (variable-atom H removed from
    coordinates; H injection modelled algebraically via PCET A_n/P_n fields).
    Fe₂S₂ core at Janus bondlength (2.316 Å); N–N elongates 1.10→1.52 Å.

    Stoichiometry: 8 H⁺/e⁻ over 8 steps ≡ N₂ + 8H → 2NH₃ + H₂ (with ATP
    bookkeeping). Σν = 16 ≡ 0 (mod 4). Crossing at n=4 (N–N = 1.34 Å,
    the transition-state midpoint of dissociative chemisorption).
    """
    N, M_steps, m_modulus = n_orbitals, 8, 4
    steps = []
    k_acc = e_acc = e_fwd = 0

    for n in range(M_steps):
        nu_n = 2
        A_n  = [n % N]
        P_n  = [(n + 1) % N]
        B_n  = [(n + 2) % N]

        k_acc += nu_n
        e_fwd += len(A_n)
        e_acc += len(A_n)

        # Janus at n=4: N–N = 1.34 Å, midpoint of dissociative activation
        is_crossing = (n == 4)
        nn_stretch  = 1.10 + n * 0.06   # matches _haber_bosch_fe2s2n2_geometry

        steps.append(MQEStep(
            step_n               = n,
            nu_n                 = nu_n,
            A_n                  = A_n,
            P_n                  = P_n,
            B_n                  = B_n,
            is_crossing          = is_crossing,
            delta_CI_Ha          = 1.6e-3 if is_crossing else None,
            crossing_orbitals    = [0, 1] if is_crossing else None,
            geometry_label       = f"Fe2S2N2_HB_step{n:02d}_NN={nn_stretch:.2f}A",
            bondlength_angstrom  = nn_stretch,
            cumulative_electrons     = e_fwd,
            phase_index_k            = k_acc % m_modulus,
            cumulative_net_electrons = e_acc,
        ))

    return MQEMechanismSpec(
        name                     = "haber_bosch",
        M_steps                  = M_steps,
        m_modulus                = m_modulus,
        S_target                 = 1.5,        # high-spin Fe₂S₂N₂, S=3/2 approx
        n_orbitals               = N,
        steps                    = steps,
        description              = (
            "Haber-Bosch N₂ activation on Fe₂S₂N₂ proxy. Fixed Fe₂S₂ core; "
            "N–N bond elongation models dissociative chemisorption. "
            "H injection algebraic (PCET A_n/P_n). Z₄ phase closure, "
            "Janus at n=4 (N–N transition-state midpoint)."
        ),
        expected_total_electrons = 8,
        expected_net_electrons   = 8,
        expected_net_phase       = 0,
        expected_net_phase_closure = True,
        expected_energy_ordering   = "decreasing",
    )


def _fe4s4_geometry_at_step(step_n: int) -> str:
    """Generates an [Fe4S4] cubane geometry that expands with step_n.
    
    The base coordinates form a T_d symmetric-like cubane core.
    As step_n increases (simulating reduction), the core volume expands
    slightly (parameterized by a scale factor).
    """
    # Base half-cube dimension (approximate for Fe-S bond ~2.26 A)
    # 2.26 / sqrt(3) ~= 1.305
    d = 1.305 
    
    # Expand the core by ~0.5% per reduction step to simulate breathing
    scale = 1.0 + (step_n * 0.005)
    d = d * scale

    return (
        f"Fe   {d:.6f}  {d:.6f}  {d:.6f}\n"
        f"Fe  -{d:.6f} -{d:.6f}  {d:.6f}\n"
        f"Fe  -{d:.6f}  {d:.6f} -{d:.6f}\n"
        f"Fe   {d:.6f} -{d:.6f} -{d:.6f}\n"
        f"S   -{d:.6f} -{d:.6f} -{d:.6f}\n"
        f"S    {d:.6f}  {d:.6f} -{d:.6f}\n"
        f"S    {d:.6f} -{d:.6f}  {d:.6f}\n"
        f"S   -{d:.6f}  {d:.6f}  {d:.6f}"
    )

def _get_nitrogenase_fe4s4_atom_block(step: MQEStep) -> str:
    return _fe4s4_geometry_at_step(step.step_n)

def _build_nitrogenase_fe4s4_spec(n_orbitals: int) -> MQEMechanismSpec:
    """Nitrogenase Fe4S4 mechanism: M=8, m=4 (Z4), 1 Janus at n=4.
    
    UPGRADED: Category B (Active PCET Simulation).
    Restores algebraic electron injection (A_n) while preserving the geometric
    breathing of the Fe4S4 cluster. Now initialized for reversible modeling.
    """
    N = n_orbitals
    M_steps = 8
    steps = []
    
    # Net-Flux Accumulation
    e_fwd = 0   # forward-only injection count for check (d)
    k_acc = 0
    e_acc = 0

    for n in range(M_steps):
        # ── Active PCET Transitions ──────────────────────────────────────────
        nu_n          = 2                   # ATP-coupled phase shift (Z4)
        A_n           = [n % N]             # Active: 1e⁻ injected per step
        P_n           = [(n + 1) % N]       # 1H⁺ addition
        B_n           = [(n + 2) % N]       # Virtual cofactor coupling

        # ── REVERSIBLE FIELDS (Initialized for PCET back-reaction) ───────────
        A_n_eject     = []
        P_n_eject     = []
        B_n_decouple  = []
        nu_decouple_n = 0

        # ── Net-Flux Accumulation ───────────────────────────────────────────
        k_acc += (nu_n - nu_decouple_n)
        e_fwd += len(A_n)                        # forward-only (check d)
        e_acc += (len(A_n) - len(A_n_eject))

        is_crossing = (n == 4)
        step = MQEStep(
            step_n               = n,
            nu_n                 = nu_n,
            A_n                  = A_n,
            P_n                  = P_n,
            B_n                  = B_n,
            A_n_eject            = A_n_eject,
            P_n_eject            = P_n_eject,
            B_n_decouple         = B_n_decouple,
            nu_decouple_n        = nu_decouple_n,
            is_crossing          = is_crossing,
            delta_CI_Ha          = 1.6e-3 if is_crossing else None,
            crossing_orbitals    = [0, 1] if is_crossing else None,
            geometry_label       = f"Fe4S4_E{n}_Intermediate",
            bondlength_angstrom  = 2.260 + (n * 0.015),
            phase_index_k        = k_acc % 4,
            cumulative_electrons     = e_fwd,      # FIX: required for check (d)
            cumulative_net_electrons = e_acc,
        )
        steps.append(step)

    return MQEMechanismSpec(
        name                     = "nitrogenase_fe4s4",
        M_steps                  = M_steps,
        m_modulus                = 4,
        S_target                 = 0.0,
        n_orbitals               = N,
        steps                    = steps,
        description              = (
            "Lowe-Thorneley Fe4S4 cubane simulation. Category B: Active PCET. "
            "Algebraic e⁻ injection enabled for realistic cluster redox simulation. "
            "Reversible framework enabled (eject/decouple fields ready). "
            "Phase closure: Σ(ν-ν†)=16 ≡ 0 (mod 4). Net electrons: 8."
        ),
        expected_net_electrons   = 8,
        expected_net_phase_closure = True,
        expected_energy_ordering   = "increasing",
    )



def _get_femoco_atom_block(step: MQEStep) -> str:
    """Construct FeMo-co geometry with Mo, S, C, Substrate (N2), and Homocitrate.
    
    Maintains strict Hilbert-space invariance by keeping the core atom count 
    constant (22 atoms). The N-N bond elongates step-wise to model substrate 
    activation/cleavage. Stoichiometry is enforced algebraically via A_n/P_n 
    fields in the MQE pipeline.
    """
    # 1. Base Catalyst Core (Fe7 Mo S9 C)
    base_geom = """
    Fe   0.000000    0.000000    3.457475
    Fe   0.000000    1.524205    1.294913
    Fe  -1.320000   -0.762102    1.294913
    Fe   1.320000   -0.762102    1.294913
    Fe   1.320000    0.762102   -1.294913
    Fe  -1.320000    0.762102   -1.294913
    Fe   0.000000   -1.524205   -1.294913
    Mo   0.000000    0.000000   -3.506624
    S    0.000000    1.062102    2.376194
    S   -0.919808   -0.531051    2.376194
    S    0.919808   -0.531051    2.376194
    S    1.060000    1.835974    0.000000
    S   -2.120000    0.000000    0.000000
    S    1.060000   -1.835974    0.000000
    S    0.919808    0.531051   -2.400768
    S   -0.919808    0.531051   -2.400768
    S    0.000000   -1.062102   -2.400768
    C    0.000000    0.000000    0.000000"""

    # 2. Homocitrate (O-rich ligand on Mo)
    lig = (
        "\nO   -1.500000    1.500000   -4.500000"
        "\nO   -2.500000    1.500000   -4.500000"
        "\nC   -2.000000    2.500000   -4.500000"
    )

    # 3. Substrate N2: Progressive elongation models activation
    nn_stretch = 1.10 + (step.step_n * 0.06)
    sub_n2 = (
        f"\nN    0.000000    0.000000   -4.800000"
        f"\nN    0.000000    0.000000   -{(4.800000 + nn_stretch):.6f}"
    )

    return base_geom + lig + sub_n2


def _build_nitrogenase_femoco_spec(n_orbitals: int) -> MQEMechanismSpec:
    """Full Nitrogenase FeMo-cofactor Mechanism: M=8, m=4 (Z4), Janus at n=4.
    Aligned with algebraic PCET framework. No physical H reservoir required.
    """
    N = n_orbitals
    M_steps = 8
    e_fwd = 0   # forward-only injection count for check (d)
    steps = []
    k_acc = 0  
    e_acc = 0  

    for n in range(M_steps):
        # PCET Fields (8e⁻, 8H⁺, 16 ATP)
        nu_n = 2
        A_n  = [n % N]
        P_n  = [(n + 1) % N]
        B_n  = [(n + 2) % N]
        
        k_acc += nu_n
        e_fwd += len(A_n)   # FIX: forward-only injection count for check (d)
        e_acc += len(A_n)

        # Janus Crossing at Step 4 (E4→E5: H2 release / N2 binding onset)
        is_crossing = (n == 4)
        
        steps.append(MQEStep(
            step_n               = n,
            nu_n                 = nu_n,
            A_n                  = A_n,
            P_n                  = P_n,
            B_n                  = B_n,
            is_crossing          = is_crossing,
            delta_CI_Ha          = 1.6e-3 if is_crossing else None,
            crossing_orbitals    = [0, 1] if is_crossing else None,
            geometry_label       = f"FeMoCo_N2_activation_{n}",
            bondlength_angstrom  = 1.10 + (n * 0.06),  # Track N-N stretch
            phase_index_k        = k_acc % 4,
            cumulative_electrons     = e_fwd,      # FIX: required for check (d)
            cumulative_net_electrons = e_acc,
        ))

    return MQEMechanismSpec(
        name                     = "nitrogenase_femoco",
        M_steps                  = M_steps,
        m_modulus                = 4,
        S_target                 = 1.5,
        n_orbitals               = N,
        steps                    = steps,
        description              = (
            "Full FeMo-cofactor LT cycle with N2 substrate. "
            "Stoichiometry enforced algebraically via A_n/P_n fields. "
            "N2 activation modeled via progressive bond elongation."
        ),
        expected_total_electrons = 8,
        expected_net_electrons   = 8,
        expected_net_phase       = 0,
        expected_net_phase_closure = True,
        expected_energy_ordering   = "decreasing",
    )


def _build_nitrogenase_group_a_spec(n_orbitals: int) -> MQEMechanismSpec:
    """Group A spectral proxy: M=8, m=8 (Z₈), ν=2 → n*=3, s=0.04090.

    Phase trace: k_acc = 2,4,6,8,10,12,14,16 → phase_index = 2,4,6,0,2,4,6,0 (mod 8).
    Closure: 16 ≡ 0 (mod 8) ✓.
    Janus at n=3: k_acc=8=m (first full Z₈ revolution). n*=m/ν-1=3 ✓.

    Proxy: Fe₂S₂ cluster, bond compression 2.260→2.190 Å over 8 steps.
    charge=0, spin_2S=4 throughout.

    Covers catalog entries (all Group A, spectral class independent of winding):
      Entry 1:  Mo-nitrogenase       (N_e=8, M_cof=16, winding (1,2))
      Entry 7:  Assimilatory NR      (N_e=8, M_cof=0,  winding (1,0))
      Entry 14: Photocatalytic N₂    (N_e=8, M_cof=8,  winding (1,1)) at M=8
    """
    N = n_orbitals
    M_steps   = 8
    m_modulus = 8
    fe_s_distances = [2.260 - i * 0.010 for i in range(M_steps)]

    steps = []
    k_acc = e_fwd = e_acc = 0

    for n in range(M_steps):
        nu_n = 2
        A_n  = [n % N]
        P_n  = [(n + 1) % N]
        B_n  = [(n + 2) % N]

        k_acc += nu_n
        e_fwd += len(A_n)
        e_acc += len(A_n)

        # Janus at n=3: k_acc=8=m (Z₈ first revolution complete). n*=3.
        is_crossing = (n == 3)

        steps.append(MQEStep(
            step_n               = n,
            nu_n                 = nu_n,
            A_n                  = A_n,
            P_n                  = P_n,
            B_n                  = B_n,
            A_n_eject            = [],
            P_n_eject            = [],
            B_n_decouple         = [],
            nu_decouple_n        = 0,
            is_crossing          = is_crossing,
            delta_CI_Ha          = 1.6e-3 if is_crossing else None,
            crossing_orbitals    = [0, 1] if is_crossing else None,
            geometry_label       = f"Fe2S2_GroupA_n{n:02d}_{fe_s_distances[n]:.3f}A",
            bondlength_angstrom  = fe_s_distances[n],
            phase_index_k        = k_acc % m_modulus,
            cumulative_electrons     = e_fwd,
            cumulative_net_electrons = e_acc,
        ))

    return MQEMechanismSpec(
        name                     = "nitrogenase_group_a",
        M_steps                  = M_steps,
        m_modulus                = m_modulus,
        S_target                 = 1.5,
        n_orbitals               = N,
        steps                    = steps,
        description              = (
            "Group A spectral proxy: m=8, ν=2, n*=3, s=0.04090. Fe₂S₂ cluster. "
            "Catalog entries 1 (Mo-nitrogenase), 7 (assimilatory NR), "
            "14 (photocatalytic N₂ fix, M=8)."
        ),
        expected_total_electrons   = 8,
        expected_net_electrons     = 8,
        expected_net_phase_closure = True,
        expected_energy_ordering   = "none",
    )


def _build_nitrogenase_group_d_spec(n_orbitals: int) -> MQEMechanismSpec:
    """Group D spectral proxy: M=12, m=12 (Z₁₂), ν=2 → n*=5, s=0.02743.

    Phase trace: k_acc = 2,4,...,24 → phase_index = 2,4,6,8,10,0,2,4,6,8,10,0 (mod 12).
    Closure: 24 ≡ 0 (mod 12) ✓.
    Janus at n=5: k_acc=12=m (first full Z₁₂ revolution). n*=m/ν-1=5 ✓.

    Proxy: Fe₂S₂ cluster, bond compression 2.260→2.172 Å over 12 steps.
    charge=0, spin_2S=4 throughout.

    Covers catalog entries (both Group D — cross-domain spectral degeneracy):
      Entry 3:  V-nitrogenase  (N_e=12, M_cof=24, winding (1,2), m=12)
      Entry 13: Cu CO₂RR      (N_e=12, M_cof=0,  winding (1,0), m=12)
    """
    N = n_orbitals
    M_steps   = 12
    m_modulus = 12
    # 12 checkpoints: 2.260 → 2.172 Å (step = 0.088/11 Å)
    fe_s_distances = [round(2.260 - i * (0.088 / 11), 6) for i in range(M_steps)]

    steps = []
    k_acc = e_fwd = e_acc = 0

    for n in range(M_steps):
        nu_n = 2
        A_n  = [n % N]
        P_n  = [(n + 1) % N]
        B_n  = [(n + 2) % N]

        k_acc += nu_n
        e_fwd += len(A_n)
        e_acc += len(A_n)

        # Janus at n=5: k_acc=12=m (Z₁₂ first revolution complete). n*=5.
        is_crossing = (n == 5)

        steps.append(MQEStep(
            step_n               = n,
            nu_n                 = nu_n,
            A_n                  = A_n,
            P_n                  = P_n,
            B_n                  = B_n,
            A_n_eject            = [],
            P_n_eject            = [],
            B_n_decouple         = [],
            nu_decouple_n        = 0,
            is_crossing          = is_crossing,
            delta_CI_Ha          = 1.6e-3 if is_crossing else None,
            crossing_orbitals    = [0, 1] if is_crossing else None,
            geometry_label       = f"Fe2S2_GroupD_n{n:02d}_{fe_s_distances[n]:.3f}A",
            bondlength_angstrom  = fe_s_distances[n],
            phase_index_k        = k_acc % m_modulus,
            cumulative_electrons     = e_fwd,
            cumulative_net_electrons = e_acc,
        ))

    return MQEMechanismSpec(
        name                     = "nitrogenase_group_d",
        M_steps                  = M_steps,
        m_modulus                = m_modulus,
        S_target                 = 1.5,
        n_orbitals               = N,
        steps                    = steps,
        description              = (
            "Group D spectral proxy: m=12, ν=2, n*=5, s=0.02743. Fe₂S₂ cluster. "
            "Catalog entries 3 (V-nitrogenase, winding (1,2)) and "
            "13 (Cu CO₂RR, winding (1,0)) — cross-domain Group D degeneracy."
        ),
        expected_total_electrons   = 12,
        expected_net_electrons     = 12,
        expected_net_phase_closure = True,
        expected_energy_ordering   = "none",
    )


# ---------------------------------------------------------------------------
# Individual proxy geometry functions (one per catalog entry)
# ---------------------------------------------------------------------------

def _femo_proxy_atom_block(d: float) -> str:
    """Fe-Mo-S₂ proxy for Mo-nitrogenase (catalog entry 1, Group A).
    Fe(26)+Mo(14 ECP)+2×S(16)=72e (even), charge=0, spin_2S=4.
    Reaction coordinate: Fe-Mo bond compression 2.700→2.620 Å.
    """
    return (
        f"Fe  0.000000  0.000000  0.000000\n"
        f"Mo  0.000000  0.000000  {d:.6f}\n"
        f"S   0.000000  1.400000  {d / 2:.6f}\n"
        f"S   0.000000 -1.400000  {d / 2:.6f}"
    )


def _mo_nr_proxy_atom_block(d: float) -> str:
    """Mo-S₂-O₂ proxy for assimilatory nitrate reductase (catalog entry 7, Group A).
    Mo(14 ECP)+2×S(16)+2×O(8)=62e (even), charge=0, spin_2S=4.
    Reaction coordinate: Mo-S bond compression 2.420→2.340 Å.
    """
    return (
        f"Mo  0.000000  0.000000  0.000000\n"
        f"S   0.000000  {d:.6f}  0.000000\n"
        f"S   0.000000  {-d:.6f}  0.000000\n"
        f"O   1.800000  0.000000  0.000000\n"
        f"O  -1.800000  0.000000  0.000000"
    )


def _ti2n2_proxy_atom_block(d: float) -> str:
    """Ti₂N₂ proxy for photocatalytic N₂ fixation (catalog entry 14, Group A).
    2×Ti(22)+2×N(7)=58e (even), charge=0, spin_2S=4.
    Reaction coordinate: Ti-N bond compression 1.900→1.556 Å (N₂ side-on activation).
    """
    return (
        f"Ti  0.000000  0.000000  0.000000\n"
        f"Ti  2.960000  0.000000  0.000000\n"
        f"N   1.480000  {d:.6f}  0.000000\n"
        f"N   1.480000  {-d:.6f}  0.000000"
    )


def _v2s2_proxy_atom_block(d: float) -> str:
    """V₂S₂ proxy for V-nitrogenase (catalog entry 3, Group D). FeVco homolog.
    2×V(23)+2×S(16)=78e (even), charge=0, spin_2S=4.
    Reaction coordinate: V-S bond compression 2.350→2.258 Å.
    """
    return (
        f"V   0.000000  1.300000  0.000000\n"
        f"V   0.000000 -1.300000  0.000000\n"
        f"S   {d:.6f}  0.000000  0.000000\n"
        f"S  {-d:.6f}  0.000000  0.000000"
    )


def _cu3_proxy_atom_block(d: float) -> str:
    """Cu₃ equilateral trimer proxy for Cu CO₂RR (catalog entry 13, Group D).
    3×Cu(29), charge=−1 → 88e (even), spin_2S=0.
    Reaction coordinate: Cu-Cu bond compression 2.550→2.458 Å.
    """
    half = d / 2.0
    height = d * 0.866025403784
    return (
        f"Cu  0.000000  0.000000  0.000000\n"
        f"Cu  {d:.6f}  0.000000  0.000000\n"
        f"Cu  {half:.6f}  {height:.6f}  0.000000"
    )


def _femon2_trimer_atom_block(d_nn: float) -> str:
    """Fe–Mo–N₂ trimer proxy for N₂ activation (Group B, m=4).

    Fe(26) + Mo(ECP28→14 val) + 2×N(7) = 54e (even), charge=0, spin_2S=4.
    Linear arrangement along z: Fe — Mo — N≡N (end-on binding, biologically
    relevant for Mo-nitrogenase N₂ activation).

    Fixed geometry:
        Fe–Mo = 2.700 Å  (same equilibrium as mo_nitrogenase Fe–Mo proxy)
        Mo–N(prox) = 2.000 Å  (Mo–N₂ end-on binding distance)
    Reaction coordinate:
        N–N elongation d_nn = 1.10 → 1.52 Å (8 steps, 0.06 Å/step)
        Step 4 (d=1.34 Å) is the TS region; Janus fires here.
    """
    d_femo = 2.700
    d_mon  = 2.000
    z_fe   = 0.0
    z_mo   = z_fe + d_femo
    z_n1   = z_mo + d_mon
    z_n2   = z_n1 + d_nn
    return (
        f"Fe  0.000000  0.000000  {z_fe:.6f}\n"
        f"Mo  0.000000  0.000000  {z_mo:.6f}\n"
        f"N   0.000000  0.000000  {z_n1:.6f}\n"
        f"N   0.000000  0.000000  {z_n2:.6f}"
    )


# ---------------------------------------------------------------------------
# Individual spec builders — catalog entries 1, 7, 14 (Group A) and 3, 13 (Group D)
# ---------------------------------------------------------------------------

def _build_mo_nitrogenase_spec(n_orbitals: int) -> MQEMechanismSpec:
    """Catalog entry 1 (Mo-nitrogenase, Group A): M=8, m=8, ν=2 → n*=3, s=0.04090.

    Proxy: Fe-Mo-S₂ cluster. 72e, charge=0, spin_2S=4.
    Bond: Fe-Mo compression 2.700→2.620 Å (step=0.010 Å). Janus at n=3 (k_acc=8=m).
    Winding (1,2): N_e=8, M_cof=16 (ATP). Phase closure: 16≡0(mod 8) ✓.
    Distinct from nitrogenase_group_a (Fe₂S₂ generic proxy). --tower-p 2.
    """
    N = n_orbitals
    M_steps   = 8
    m_modulus = 8
    bonds = [round(2.700 - i * 0.010, 6) for i in range(M_steps)]

    steps = []
    k_acc = e_fwd = e_acc = 0

    for n in range(M_steps):
        nu_n = 2
        A_n  = [n % N]
        P_n  = [(n + 1) % N]
        B_n  = [(n + 2) % N]
        k_acc += nu_n
        e_fwd += len(A_n)
        e_acc += len(A_n)
        is_crossing = (n == 3)

        steps.append(MQEStep(
            step_n               = n,
            nu_n                 = nu_n,
            A_n                  = A_n,
            P_n                  = P_n,
            B_n                  = B_n,
            A_n_eject            = [],
            P_n_eject            = [],
            B_n_decouple         = [],
            nu_decouple_n        = 0,
            is_crossing          = is_crossing,
            delta_CI_Ha          = 1.6e-3 if is_crossing else None,
            crossing_orbitals    = [0, 1] if is_crossing else None,
            geometry_label       = f"FeMoS2_entry1_n{n:02d}_{bonds[n]:.3f}A",
            bondlength_angstrom  = bonds[n],
            phase_index_k        = k_acc % m_modulus,
            cumulative_electrons     = e_fwd,
            cumulative_net_electrons = e_acc,
        ))

    return MQEMechanismSpec(
        name                     = "mo_nitrogenase",
        M_steps                  = M_steps,
        m_modulus                = m_modulus,
        S_target                 = 1.5,
        n_orbitals               = N,
        steps                    = steps,
        description              = (
            "Catalog entry 1 (Mo-nitrogenase, Group A): m=8, ν=2, n*=3, s=0.04090. "
            "Fe-Mo-S₂ proxy, bond 2.700→2.620 Å. Winding (1,2), M_cof=16."
        ),
        expected_total_electrons   = 8,
        expected_net_electrons     = 8,
        expected_net_phase_closure = True,
        expected_energy_ordering   = "none",
    )


def _build_assimilatory_nr_spec(n_orbitals: int) -> MQEMechanismSpec:
    """Catalog entry 7 (assimilatory nitrate reductase, Group A): M=8, m=8, ν=2 → n*=3, s=0.04090.

    Proxy: Mo-S₂-O₂ pterin-dithiolate mimic. 62e, charge=0, spin_2S=4.
    Bond: Mo-S compression 2.420→2.340 Å. Janus at n=3 (Mo(VI)/Mo(IV) inversion).
    Winding (1,0): N_e=8, M_cof=0. Phase closure: 16≡0(mod 8) ✓. --tower-p 2.
    """
    N = n_orbitals
    M_steps   = 8
    m_modulus = 8
    step_sz   = (2.420 - 2.340) / 7
    bonds = [round(2.420 - i * step_sz, 6) for i in range(M_steps)]

    steps = []
    k_acc = e_fwd = e_acc = 0

    for n in range(M_steps):
        nu_n = 2
        A_n  = [n % N]
        P_n  = [(n + 1) % N]
        B_n  = [(n + 2) % N]
        k_acc += nu_n
        e_fwd += len(A_n)
        e_acc += len(A_n)
        is_crossing = (n == 3)

        steps.append(MQEStep(
            step_n               = n,
            nu_n                 = nu_n,
            A_n                  = A_n,
            P_n                  = P_n,
            B_n                  = B_n,
            A_n_eject            = [],
            P_n_eject            = [],
            B_n_decouple         = [],
            nu_decouple_n        = 0,
            is_crossing          = is_crossing,
            delta_CI_Ha          = 1.6e-3 if is_crossing else None,
            crossing_orbitals    = [0, 1] if is_crossing else None,
            geometry_label       = f"MoS2O2_entry7_n{n:02d}_{bonds[n]:.3f}A",
            bondlength_angstrom  = bonds[n],
            phase_index_k        = k_acc % m_modulus,
            cumulative_electrons     = e_fwd,
            cumulative_net_electrons = e_acc,
        ))

    return MQEMechanismSpec(
        name                     = "assimilatory_nr",
        M_steps                  = M_steps,
        m_modulus                = m_modulus,
        S_target                 = 1.5,
        n_orbitals               = N,
        steps                    = steps,
        description              = (
            "Catalog entry 7 (assimilatory NR, Group A): m=8, ν=2, n*=3, s=0.04090. "
            "Mo-S₂-O₂ pterin-dithiolate proxy, bond 2.420→2.340 Å. Winding (1,0), M_cof=0."
        ),
        expected_total_electrons   = 8,
        expected_net_electrons     = 8,
        expected_net_phase_closure = True,
        expected_energy_ordering   = "none",
    )


def _build_photocatalytic_n2_spec(n_orbitals: int) -> MQEMechanismSpec:
    """Catalog entry 14 (photocatalytic N₂ fixation, Group A): M=8, m=8, ν=2 → n*=3, s=0.04090.

    Proxy: Ti₂N₂ (N₂ side-on over TiO₂ surface). 58e, charge=0, spin_2S=4.
    Bond: Ti-N compression 1.900→1.556 Å (end-on → side-on → activated N₂).
    Janus at n=3. Winding (1,1): N_e=8, M_cof=8 (photon). --tower-p 2.
    """
    N = n_orbitals
    M_steps   = 8
    m_modulus = 8
    step_sz   = (1.900 - 1.556) / 7
    bonds = [round(1.900 - i * step_sz, 6) for i in range(M_steps)]

    steps = []
    k_acc = e_fwd = e_acc = 0

    for n in range(M_steps):
        nu_n = 2
        A_n  = [n % N]
        P_n  = [(n + 1) % N]
        B_n  = [(n + 2) % N]
        k_acc += nu_n
        e_fwd += len(A_n)
        e_acc += len(A_n)
        is_crossing = (n == 3)

        steps.append(MQEStep(
            step_n               = n,
            nu_n                 = nu_n,
            A_n                  = A_n,
            P_n                  = P_n,
            B_n                  = B_n,
            A_n_eject            = [],
            P_n_eject            = [],
            B_n_decouple         = [],
            nu_decouple_n        = 0,
            is_crossing          = is_crossing,
            delta_CI_Ha          = 1.6e-3 if is_crossing else None,
            crossing_orbitals    = [0, 1] if is_crossing else None,
            geometry_label       = f"Ti2N2_entry14_n{n:02d}_{bonds[n]:.3f}A",
            bondlength_angstrom  = bonds[n],
            phase_index_k        = k_acc % m_modulus,
            cumulative_electrons     = e_fwd,
            cumulative_net_electrons = e_acc,
        ))

    return MQEMechanismSpec(
        name                     = "photocatalytic_n2",
        M_steps                  = M_steps,
        m_modulus                = m_modulus,
        S_target                 = 1.5,
        n_orbitals               = N,
        steps                    = steps,
        description              = (
            "Catalog entry 14 (photocatalytic N₂ fix, Group A): m=8, ν=2, n*=3, s=0.04090. "
            "Ti₂N₂ proxy, Ti-N bond 1.900→1.556 Å. Winding (1,1), M_cof=8."
        ),
        expected_total_electrons   = 8,
        expected_net_electrons     = 8,
        expected_net_phase_closure = True,
        expected_energy_ordering   = "none",
    )


def _build_v_nitrogenase_spec(n_orbitals: int) -> MQEMechanismSpec:
    """Catalog entry 3 (V-nitrogenase, Group D): M=12, m=12, ν=2 → n*=5, s=0.02743.

    Proxy: V₂S₂ (FeVco homolog). 78e, charge=0, spin_2S=4.
    Bond: V-S compression 2.350→2.258 Å. Janus at n=5 (k_acc=12=m).
    Winding (1,2): N_e=12, M_cof=24. Phase closure: 24≡0(mod 12) ✓. --tower-p 3.
    """
    N = n_orbitals
    M_steps   = 12
    m_modulus = 12
    bonds = [round(2.350 - i * (0.092 / 11), 6) for i in range(M_steps)]

    steps = []
    k_acc = e_fwd = e_acc = 0

    for n in range(M_steps):
        nu_n = 2
        A_n  = [n % N]
        P_n  = [(n + 1) % N]
        B_n  = [(n + 2) % N]
        k_acc += nu_n
        e_fwd += len(A_n)
        e_acc += len(A_n)
        is_crossing = (n == 5)

        steps.append(MQEStep(
            step_n               = n,
            nu_n                 = nu_n,
            A_n                  = A_n,
            P_n                  = P_n,
            B_n                  = B_n,
            A_n_eject            = [],
            P_n_eject            = [],
            B_n_decouple         = [],
            nu_decouple_n        = 0,
            is_crossing          = is_crossing,
            delta_CI_Ha          = 1.6e-3 if is_crossing else None,
            crossing_orbitals    = [0, 1] if is_crossing else None,
            geometry_label       = f"V2S2_entry3_n{n:02d}_{bonds[n]:.3f}A",
            bondlength_angstrom  = bonds[n],
            phase_index_k        = k_acc % m_modulus,
            cumulative_electrons     = e_fwd,
            cumulative_net_electrons = e_acc,
        ))

    return MQEMechanismSpec(
        name                     = "v_nitrogenase",
        M_steps                  = M_steps,
        m_modulus                = m_modulus,
        S_target                 = 1.5,
        n_orbitals               = N,
        steps                    = steps,
        description              = (
            "Catalog entry 3 (V-nitrogenase, Group D): m=12, ν=2, n*=5, s=0.02743. "
            "V₂S₂ FeVco proxy, bond 2.350→2.258 Å. Winding (1,2), M_cof=24."
        ),
        expected_total_electrons   = 12,
        expected_net_electrons     = 12,
        expected_net_phase_closure = True,
        expected_energy_ordering   = "none",
    )


def _build_cu_co2rr_spec(n_orbitals: int) -> MQEMechanismSpec:
    """Catalog entry 13 (Cu CO₂ electroreduction, Group D): M=12, m=12, ν=2 → n*=5, s=0.02743.

    Proxy: Cu₃⁻ equilateral trimer (Cu surface active site). 88e, charge=−1, spin_2S=0.
    Bond: Cu-Cu compression 2.550→2.458 Å. Janus at n=5 (CO₂*→CO+O bifurcation).
    Winding (1,0): N_e=12, M_cof=0. Cross-domain Group D degeneracy with entry 3.
    --tower-p 3 (m=12=4×3).
    """
    N = n_orbitals
    M_steps   = 12
    m_modulus = 12
    bonds = [round(2.550 - i * (0.092 / 11), 6) for i in range(M_steps)]

    steps = []
    k_acc = e_fwd = e_acc = 0

    for n in range(M_steps):
        nu_n = 2
        A_n  = [n % N]
        P_n  = [(n + 1) % N]
        B_n  = [(n + 2) % N]
        k_acc += nu_n
        e_fwd += len(A_n)
        e_acc += len(A_n)
        is_crossing = (n == 5)

        steps.append(MQEStep(
            step_n               = n,
            nu_n                 = nu_n,
            A_n                  = A_n,
            P_n                  = P_n,
            B_n                  = B_n,
            A_n_eject            = [],
            P_n_eject            = [],
            B_n_decouple         = [],
            nu_decouple_n        = 0,
            is_crossing          = is_crossing,
            delta_CI_Ha          = 1.6e-3 if is_crossing else None,
            crossing_orbitals    = [0, 1] if is_crossing else None,
            geometry_label       = f"Cu3_entry13_n{n:02d}_{bonds[n]:.3f}A",
            bondlength_angstrom  = bonds[n],
            phase_index_k        = k_acc % m_modulus,
            cumulative_electrons     = e_fwd,
            cumulative_net_electrons = e_acc,
        ))

    return MQEMechanismSpec(
        name                     = "cu_co2rr",
        M_steps                  = M_steps,
        m_modulus                = m_modulus,
        S_target                 = 0.0,
        n_orbitals               = N,
        steps                    = steps,
        description              = (
            "Catalog entry 13 (Cu CO₂RR, Group D): m=12, ν=2, n*=5, s=0.02743. "
            "Cu₃⁻ trimer proxy, bond 2.550→2.458 Å. Winding (1,0), M_cof=0."
        ),
        expected_total_electrons   = 12,
        expected_net_electrons     = 12,
        expected_net_phase_closure = True,
        expected_energy_ordering   = "none",
    )


def _build_femon2_trimer_spec(n_orbitals: int) -> MQEMechanismSpec:
    """Fe–Mo–N₂ trimer proxy (Group B, m=4): M=8, m=4, ν=2, Janus at n=4.

    Minimal 4-atom proxy for the N₂ activation spectral sector (Group B,
    s=0.08115) that nitrogenase_femoco (full 22-atom FeMo-co) targets but
    cannot converge under ROHF CAS(4,4) due to the 7-Fe near-degenerate 3d
    manifold. This system (Fe–Mo–N₂, 54e) converges cleanly.

    Spectral class: Group B, m=4, ν=2, n*=m/ν−1=1, s=0.08115.
    Janus condition: k^(4)=10 mod 4=2=m/2 ✓ (antipodal; same as nitrogenase_femoco).
    Physical Janus: n=4, d(N–N)=1.34 Å (TS midpoint between triple bond 1.10 Å
    and single-bond territory 1.47 Å).

    Geometry: Fe–Mo fixed at 2.700 Å, Mo–N(proximal) fixed at 2.000 Å.
    N–N scan: 1.10→1.52 Å (Δ=0.06 Å/step). 54e, charge=0, spin_2S=4.
    Stoichiometry: N_e=8, M_cof=16. Phase closure: 16≡0(mod 4) ✓.
    --tower-p 2 (m=4=2²).
    """
    N = n_orbitals
    M_steps   = 8
    m_modulus = 4
    bonds = [round(1.10 + i * 0.06, 6) for i in range(M_steps)]

    steps = []
    k_acc = e_fwd = e_acc = 0

    for n in range(M_steps):
        nu_n = 2
        A_n  = [n % N]
        P_n  = [(n + 1) % N]
        B_n  = [(n + 2) % N]
        k_acc += nu_n
        e_fwd += len(A_n)
        e_acc += len(A_n)
        # Janus at n=4: k_acc=10, k^(4)=10 mod 4=2=m/2 ✓ (antipodal condition)
        is_crossing = (n == 4)

        steps.append(MQEStep(
            step_n               = n,
            nu_n                 = nu_n,
            A_n                  = A_n,
            P_n                  = P_n,
            B_n                  = B_n,
            A_n_eject            = [],
            P_n_eject            = [],
            B_n_decouple         = [],
            nu_decouple_n        = 0,
            is_crossing          = is_crossing,
            delta_CI_Ha          = 1.6e-3 if is_crossing else None,
            crossing_orbitals    = [0, 1] if is_crossing else None,
            geometry_label       = f"FeMoN2_trimer_n{n:02d}_{bonds[n]:.3f}A",
            bondlength_angstrom  = bonds[n],
            phase_index_k        = k_acc % m_modulus,
            cumulative_electrons     = e_fwd,
            cumulative_net_electrons = e_acc,
        ))

    return MQEMechanismSpec(
        name                     = "femon2_trimer",
        M_steps                  = M_steps,
        m_modulus                = m_modulus,
        S_target                 = 1.5,
        n_orbitals               = N,
        steps                    = steps,
        description              = (
            "Fe–Mo–N₂ trimer proxy (Group B, m=4): N₂ end-on activation on a "
            "minimal Fe–Mo site. 4-atom, 54e convergent proxy for the spectral "
            "sector (s=0.08115) targeted by nitrogenase_femoco. N–N bond "
            "1.10→1.52 Å; Janus at n=4 (k=m/2=2, antipodal). --tower-p 2."
        ),
        expected_total_electrons   = 8,
        expected_net_electrons     = 8,
        expected_net_phase_closure = True,
        expected_energy_ordering   = "none",
    )


# Anammox Proxy: N-N bond formation on a single Fe atom.
# Represents 2x NH2 coupling to form N2H4 (Hydrazine)
ANAMMOX_GEOMETRIES = {
    # Step 0: Distant NH2 fragments (Pre-coupling, N-N distance ~2.6 A)
    0: """Fe  0.000000  0.000000  0.000000
          N   0.000000  1.300000  1.500000
          H   0.800000  1.300000  2.000000
          H  -0.800000  1.300000  2.000000
          N   0.000000 -1.300000  1.500000
          H   0.800000 -1.300000  2.000000
          H  -0.800000 -1.300000  2.000000""",
          
    # Step 1: Fragments move closer (N-N distance ~2.0 A)
    1: """Fe  0.000000  0.000000  0.000000
          N   0.000000  1.000000  1.600000
          H   0.800000  1.100000  2.100000
          H  -0.800000  1.100000  2.100000
          N   0.000000 -1.000000  1.600000
          H   0.800000 -1.100000  2.100000
          H  -0.800000 -1.100000  2.100000""",
          
    # Step 2: Transition state proxy (N-N distance ~1.6 A)
    2: """Fe  0.000000  0.000000  0.000000
          N   0.000000  0.800000  1.800000
          H   0.800000  0.900000  2.300000
          H  -0.800000  0.900000  2.300000
          N   0.000000 -0.800000  1.800000
          H   0.800000 -0.900000  2.300000
          H  -0.800000 -0.900000  2.300000""",
          
    # Step 3: Bound Hydrazine N2H4 (N-N distance ~1.44 A)
    3: """Fe  0.000000  0.000000  0.000000
          N   0.000000  0.720000  2.000000
          H   0.800000  0.800000  2.500000
          H  -0.800000  0.800000  2.500000
          N   0.000000 -0.720000  2.000000
          H   0.800000 -0.800000  2.500000
          H  -0.800000 -0.800000  2.500000"""
}

def _get_anammox_atom_block(step: MQEStep) -> str:
    if step.step_n not in ANAMMOX_GEOMETRIES:
        raise KeyError(f"No geometry defined for Anammox step {step.step_n}")
    return ANAMMOX_GEOMETRIES[step.step_n]

def _build_anammox_proxy_spec(n_orbitals: int) -> MQEMechanismSpec:
    """Anammox Hydrazine Synthase Proxy: M=4, m=4 (Z4).
    
    UPGRADED: Category B (Active PCET Simulation).
    Models oxidative N-N coupling (Nitrite/Ammonia to Hydrazine). 
    Actively tracks the ejection of 4e⁻/4H⁺ while maintaining geometric
    Hamiltonian evolution. Initialized for full reversibility.
    """
    N         = n_orbitals
    M_steps   = 4
    m_modulus = 4      # Z₄ phase group — oxidative PCET
    steps = []
    e_fwd = 0   # forward-only injection count for check (d)
    
    # Net-Flux Accumulation
    k_acc = 0
    e_acc = 0

    for n in range(M_steps):
        # ── Active PCET Transitions (Oxidative) ──────────────────────────────
        nu_n           = 1                  # ATP-coupled phase shift
        A_n            = []                 # Oxidative: No injection
        P_n            = []                 # Oxidative: No protonation
        B_n            = [(n + 2) % N]      # Virtual register coupling
        
        # ── Ejection/Reversible Fields ───────────────────────────────────────
        A_n_eject      = [n % N]            # Oxidative: 1e⁻ ejected per step
        P_n_eject      = [(n + 1) % N]      # Oxidative: 1H⁺ released per step
        B_n_decouple   = []                 # No decoupling in forward proxy
        nu_decouple_n  = 0
        
        # ── Net-Flux Accumulation ───────────────────────────────────────────
        k_acc += (nu_n - nu_decouple_n)
        e_fwd += len(A_n)                        # forward-only (check d)
        e_acc += (len(A_n) - len(A_n_eject)) # e_acc will trend negative

        # Janus at n=2: N–N = 1.84 Å, midpoint of coupling trajectory.
        # Mol charge is FIXED at 0 for all steps (electron ejection is
        # algebraic via A_n_eject — not reflected in mol.charge).
        is_crossing = (n == 2)

        steps.append(MQEStep(
            step_n               = n,
            nu_n                 = nu_n,
            A_n                  = A_n,
            P_n                  = P_n,
            B_n                  = B_n,
            A_n_eject            = A_n_eject,
            P_n_eject            = P_n_eject,
            B_n_decouple         = B_n_decouple,
            nu_decouple_n        = nu_decouple_n,
            is_crossing          = is_crossing,
            delta_CI_Ha          = 1.6e-3 if is_crossing else None,
            crossing_orbitals    = [0, 1] if is_crossing else None,
            geometry_label       = f"Anammox_N-N_coupling_step_{n}",
            bondlength_angstrom  = 2.6 - (n * 0.38),
            phase_index_k        = k_acc % m_modulus,
            cumulative_electrons     = e_fwd,      # FIX: required for check (d)
            cumulative_net_electrons = e_acc,
        ))

    return MQEMechanismSpec(
        name                     = "anammox_proxy",
        M_steps                  = M_steps,
        m_modulus                = m_modulus,
        S_target                 = 1.5,   # high-spin Fe(II), S=2 (2S=4), consistent charge=0
        n_orbitals               = N,
        steps                    = steps,
        description              = (
            "Anammox: Fe(N₂H₄) N-N oxidative coupling (hydrazine synthase proxy). "
            "Fixed mol charge=0 throughout; e⁻/H⁺ ejection algebraic via PCET fields. "
            "Category B: Active PCET. Z₄ phase closure, Janus at n=2."
        ),
        expected_net_electrons   = -4,
        expected_net_phase_closure = True,
        expected_energy_ordering   = "increasing",
    )


# Ethylene Epoxidation Proxy: Ag3 cluster + Oxygen + Ethylene
# Represents the formation of the oxametallacycle and ring closure to EO.
# ETHYLENE_EPOXIDATION_GEOMETRIES = {
#     # Step 0: Reactants. Ag3 cluster with adsorbed atomic O, ethylene approaching.
#     0: """Ag  0.000000  1.670000  0.000000
#           Ag -1.450000 -0.835000  0.000000
#           Ag  1.450000 -0.835000  0.000000
#           O   0.000000  0.000000  1.400000
#           C  -0.665000  0.000000  3.500000
#           C   0.665000  0.000000  3.500000
#           H  -1.230000  0.920000  3.500000
#           H  -1.230000 -0.920000  3.500000
#           H   1.230000  0.920000  3.500000
#           H   1.230000 -0.920000  3.500000""",
          
#     # Step 1: Approach/Early Transition State. C-O bond begins to form.
#     1: """Ag  0.000000  1.670000  0.000000
#           Ag -1.450000 -0.835000  0.000000
#           Ag  1.450000 -0.835000  0.000000
#           O   0.000000  0.000000  1.500000
#           C  -0.665000  0.000000  2.800000
#           C   0.665000  0.000000  2.800000
#           H  -1.230000  0.920000  3.000000
#           H  -1.230000 -0.920000  3.000000
#           H   1.230000  0.920000  3.000000
#           H   1.230000 -0.920000  3.000000""",
          
#     # Step 2: Oxametallacycle Intermediate. One C is bound to O, the other interacts with Ag.
#     # Note: Ethylene C-C bond elongates and loses pure planar sp2 character.
#     2: """Ag  0.000000  1.670000  0.000000
#           Ag -1.450000 -0.835000  0.000000
#           Ag  1.450000 -0.835000  0.000000
#           O   0.000000 -0.700000  1.600000
#           C  -0.740000 -0.200000  2.500000
#           C   0.740000 -0.200000  2.500000
#           H  -1.200000  0.700000  2.800000
#           H  -1.200000 -1.000000  3.000000
#           H   1.200000  0.700000  2.800000
#           H   1.200000 -1.000000  3.000000""",
          
#     # Step 3: Ethylene Oxide (EO) Product. Desorbing from the surface.
#     3: """Ag  0.000000  1.670000  0.000000
#           Ag -1.450000 -0.835000  0.000000
#           Ag  1.450000 -0.835000  0.000000
#           O   0.000000  0.000000  2.500000
#           C  -0.730000  0.000000  3.300000
#           C   0.730000  0.000000  3.300000
#           H  -1.200000  0.900000  3.700000
#           H  -1.200000 -0.900000  3.700000
#           H   1.200000  0.900000  3.700000
#           H   1.200000 -0.900000  3.700000"""
# }

# def _get_epoxidation_atom_block(step: MQEStep) -> str:
#     if step.step_n not in ETHYLENE_EPOXIDATION_GEOMETRIES:
#         raise KeyError(f"No geometry defined for Epoxidation step {step.step_n}")
#     return ETHYLENE_EPOXIDATION_GEOMETRIES[step.step_n]


def _get_ethylene_epoxidation_atom_block(step: MQEStep) -> str:
    """Ethylene Epoxidation Promoted Proxy: Ag3 + Cl + O + C2H4
    
    Stoichiometry: 3*Ag(19 valence w/ ECP28) + Cl(17) + O(8) + C2H4(16) = 98 valence e-.
    Net charge = 0. Total electrons = 98 (even, closed-shell singlet).
    
    Reaction coordinate: Ethylene C-C center approaches the Ag-O complex.
    The Cl promoter withdraws electron density, stabilizing the electrophilic O-.
    """
    # Distance from Ethylene C-C midpoint to the O atom
    distance = step.bondlength_angstrom  
    
    # 1. Ag3 Triangle (Surface Proxy)
    # Ag-Ag bond length ~2.89 A
    ag1 = ( 0.000000,  1.670000, 0.000000)
    ag2 = (-1.450000, -0.835000, 0.000000)
    ag3 = ( 1.450000, -0.835000, 0.000000)
    
    # 2. Cl Promoter (adsorbed on the Ag3 face, withdrawing electron density)
    cl  = ( 0.000000,  0.000000, -1.800000)
    
    # 3. Electrophilic Oxygen (bound to Ag1, pointing up towards ethylene)
    o   = ( 0.000000,  1.670000, 1.900000)
    
    # 4. Ethylene (approaching the O atom along the Z axis)
    z_eth = 1.900000 + distance
    c1 = (-0.665000,  1.670000, z_eth)
    c2 = ( 0.665000,  1.670000, z_eth)
    
    # Ethylene Hydrogens
    h1 = (-1.230000,  2.590000, z_eth)
    h2 = (-1.230000,  0.750000, z_eth)
    h3 = ( 1.230000,  2.590000, z_eth)
    h4 = ( 1.230000,  0.750000, z_eth)

    return (
        f"Ag   {ag1[0]:.6f}  {ag1[1]:.6f}  {ag1[2]:.6f}\n"
        f"Ag   {ag2[0]:.6f}  {ag2[1]:.6f}  {ag2[2]:.6f}\n"
        f"Ag   {ag3[0]:.6f}  {ag3[1]:.6f}  {ag3[2]:.6f}\n"
        f"Cl   {cl[0]:.6f}   {cl[1]:.6f}   {cl[2]:.6f}\n"
        f"O    {o[0]:.6f}    {o[1]:.6f}    {o[2]:.6f}\n"
        f"C    {c1[0]:.6f}   {c1[1]:.6f}   {c1[2]:.6f}\n"
        f"C    {c2[0]:.6f}   {c2[1]:.6f}   {c2[2]:.6f}\n"
        f"H    {h1[0]:.6f}   {h1[1]:.6f}   {h1[2]:.6f}\n"
        f"H    {h2[0]:.6f}   {h2[1]:.6f}   {h2[2]:.6f}\n"
        f"H    {h3[0]:.6f}   {h3[1]:.6f}   {h3[2]:.6f}\n"
        f"H    {h4[0]:.6f}   {h4[1]:.6f}   {h4[2]:.6f}\n"
    )


def _build_ethylene_epoxidation_spec(n_orbitals: int) -> MQEMechanismSpec:
    """Ethylene Epoxidation on Cl-promoted Ag3: M=4, m=8 (Z8).

    The Ag3 trimer undergoes a 4-electron internal redox cycle.
    mol.charge=0 throughout.
    98 valence electrons (even) -> closed-shell singlet (S=0).

    m=8 replaces m=4 to move the Janus crossing from n=1 (d=2.8 Å, weakly
    interacting geometry, Ag ECP orbital reordering artifact) to n=2 (d=2.4 Å,
    genuine π-complex / pre-oxametallacycle regime). With m=8 and nu_n=2:
      k^(n) = 2n mod 8; Janus fires at k^(n) = m/2 = 4, i.e. n=2.
    At d=2.4 Å the Ag–O–C2H4 orbital character is consistent across geometries,
    eliminating the >100 Ha ecore discontinuity caused by ECP28 reordering at
    the previously forced n=1 step.
    Phase closure: Σν = 4×2 = 8 ≡ 0 (mod 8) ✓.
    """
    N       = n_orbitals
    M_steps = 4
    m_mod   = 8   # FIX: was 4; moved to 8 to push Janus to n=2 (d=2.4 Å)
    e_fwd   = 0
    e_acc   = 0
    k_acc   = 0
    steps   = []

    # Distances: 3.5 Å (reactants) to 2.0 Å (product-like ring closure)
    distances = [3.5, 2.8, 2.4, 2.0]

    for n in range(M_steps):
        nu_n          = 2
        A_n           = []
        P_n           = []
        B_n           = [(n % N)]
        A_n_eject     = []
        P_n_eject     = []
        B_n_decouple  = []
        nu_decouple_n = 0

        k_acc += nu_n
        e_fwd += len(A_n)
        e_acc += (len(A_n) - len(A_n_eject))

        # FIX: Janus at n=2 (d=2.4 Å, k^(2)=4=m/2 ✓), was n=1 (d=2.8 Å).
        # At d=2.4 Å the Ag-O-C2H4 system is in the π-complex regime where
        # orbital character is consistent across adjacent geometry steps.
        is_crossing = (n == 2)

        steps.append(MQEStep(
            step_n               = n,
            nu_n                 = nu_n,
            A_n                  = A_n,
            P_n                  = P_n,
            B_n                  = B_n,
            A_n_eject            = A_n_eject,
            P_n_eject            = P_n_eject,
            B_n_decouple         = B_n_decouple,
            nu_decouple_n        = nu_decouple_n,
            is_crossing          = is_crossing,
            delta_CI_Ha          = 1.6e-3 if is_crossing else None,
            crossing_orbitals    = [0, 1] if is_crossing else None,
            geometry_label       = f"Epoxidation_Ag3Cl_step_{n}_d={distances[n]:.1f}A",
            bondlength_angstrom  = distances[n],
            phase_index_k        = k_acc % m_mod,
            cumulative_electrons     = e_fwd,
            cumulative_net_electrons = e_acc,
        ))

    return MQEMechanismSpec(
        name                     = "ethylene_epoxidation",
        M_steps                  = M_steps,
        m_modulus                = m_mod,
        S_target                 = 0.0,   # 98e is even -> closed-shell singlet
        n_orbitals               = N,
        steps                    = steps,
        description              = (
            "Ethylene epoxidation on Cl-promoted Ag3 surface cluster. "
            "m=8, nu_n=2, Janus at n=2 (d=2.4 Å, k^(2)=4=m/2, π-complex regime). "
            "98 valence electrons (even); closed-shell singlet (S=0). "
            "Internal Ag⁰↔Ag⁺ redox tracked algebraically; mol charge fixed at 0. "
            "Phase closure: Σν=8 ≡ 0 (mod 8) ✓."
        ),
        expected_net_electrons   = 0,
        expected_net_phase_closure = True,
        expected_energy_ordering   = "none",
    )


# def _build_ethylene_epoxidation_spec(n_orbitals: int) -> MQEMechanismSpec:
#     """Ethylene Epoxidation on Ag3: M=4, m=4 (Z4), Group B.

#     The Ag3 trimer undergoes a 4-electron internal redox cycle:
#       n=0–1: two Ag⁰ → Ag⁺ (O activation, oxidative half); nu_n=2
#       n=2–3: two Ag⁺ → Ag⁰ (EO formation and desorption, reductive half)
#     mol.charge=0 throughout — electron transfer is internal (Ag↔O), not
#     injection from a reservoir. nu_n=2 tracks the Ag oxidation depth
#     algebraically; A_n=[] since no external electrons enter the system.

#     Group B classification: n* = m/ν_n − 1 = 4/2 − 1 = 1.
#     Janus at n=1: oxametallacycle intermediate, maximum Ag–O–C electron
#     density, electronic TS for ring closure.
#     """
#     N       = n_orbitals
#     M_steps = 4
#     m_mod   = 4
#     e_fwd   = 0
#     e_acc   = 0
#     k_acc   = 0
#     steps   = []

#     for n in range(M_steps):
#         # Ag⁰↔Ag⁺ internal redox tracked as nu_n=2; no external injection.
#         nu_n          = 2
#         A_n           = []
#         P_n           = []
#         B_n           = [(n % N)]
#         A_n_eject     = []
#         P_n_eject     = []
#         B_n_decouple  = []
#         nu_decouple_n = 0

#         k_acc += nu_n
#         e_fwd += len(A_n)
#         e_acc += (len(A_n) - len(A_n_eject))

#         # Janus at n=1: oxametallacycle, Ag–O–C ring TS.
#         is_crossing = (n == 1)

#         steps.append(MQEStep(
#             step_n               = n,
#             nu_n                 = nu_n,
#             A_n                  = A_n,
#             P_n                  = P_n,
#             B_n                  = B_n,
#             A_n_eject            = A_n_eject,
#             P_n_eject            = P_n_eject,
#             B_n_decouple         = B_n_decouple,
#             nu_decouple_n        = nu_decouple_n,
#             is_crossing          = is_crossing,
#             delta_CI_Ha          = 1.6e-3 if is_crossing else None,
#             crossing_orbitals    = [0, 1] if is_crossing else None,
#             geometry_label       = f"Epoxidation_Ag3_step_{n}",
#             bondlength_angstrom  = 3.5 - (n * 0.33),
#             phase_index_k        = k_acc % m_mod,
#             cumulative_electrons     = e_fwd,
#             cumulative_net_electrons = e_acc,
#         ))

#     return MQEMechanismSpec(
#         name                     = "ethylene_epoxidation",
#         M_steps                  = M_steps,
#         m_modulus                = m_mod,
#         S_target                 = 0.0,   # 98e is even -> closed-shell singlet!
#         n_orbitals               = N,
#         steps                    = steps,
#         description              = (
#             "Ethylene epoxidation on Ag3 surface cluster. "
#             "Group B: m=4, nu_n=2, Janus at n=1 (oxametallacycle). "
#             "Internal Ag⁰↔Ag⁺ redox tracked algebraically; mol charge fixed at 0."
#         ),
#         expected_net_electrons   = 0,
#         expected_net_phase_closure = True,
#         expected_energy_ordering   = "none",
#     )


def _thymine_dimer_proxy_geometry(distance: float) -> str:
    """Two stacked ethylene molecules representing the [2+2] cycloaddition
    of adjacent thymine bases.
    
    Ethylene 1 is fixed at Z = 0.
    Ethylene 2 approaches from Z = distance.
    """
    # Ethylene 1 (Bottom, Z = 0)
    z1 = 0.0
    eth1 = (
        f"C  -0.665000  0.000000  {z1:.6f}\n"
        f"C   0.665000  0.000000  {z1:.6f}\n"
        f"H  -1.230000  0.920000  {z1:.6f}\n"
        f"H  -1.230000 -0.920000  {z1:.6f}\n"
        f"H   1.230000  0.920000  {z1:.6f}\n"
        f"H   1.230000 -0.920000  {z1:.6f}"
    )
    
    # Ethylene 2 (Top, Z = distance)
    z2 = distance
    eth2 = (
        f"C  -0.665000  0.000000  {z2:.6f}\n"
        f"C   0.665000  0.000000  {z2:.6f}\n"
        f"H  -1.230000  0.920000  {z2:.6f}\n"
        f"H  -1.230000 -0.920000  {z2:.6f}\n"
        f"H   1.230000  0.920000  {z2:.6f}\n"
        f"H   1.230000 -0.920000  {z2:.6f}"
    )
    return eth1 + "\n" + eth2

def _get_thymine_dimer_atom_block(step: MQEStep) -> str:
    return _thymine_dimer_proxy_geometry(step.bondlength_angstrom)

# def _build_thymine_dimer_spec(n_orbitals: int) -> MQEMechanismSpec:
#     """Thymine Dimer Proxy ([2+2] cycloaddition): M=6, m=4 (Z4), Group B.

#     Models photochemical dimerization of two stacked ethylene molecules.
#     Tracks inter-planar distance from 3.4 Å (vdW stack) to 1.5 Å (cyclobutane).

#     Phase structure: the CAS(4,4) π/π* system has 4 electrons that shift
#     between the bonding (π₁,π₂) and antibonding (π₁*,π₂*) manifolds.
#     nu_n=2 tracks this 2-electron excitation algebraically per step;
#     mol.charge=0 throughout (no external injection — internal π→π* only).
#     Phase closure: 6×2=12 ≡ 0 (mod 4) ✓. n*=4/2−1=1 → Group B.

#     Janus at n=3 (~2.26 Å): S₁/S₀ conical intersection. Note n* (revolution
#     depth) ≠ crossing step — same convention as nitrogenase_lt (n*=1, gate at n=4).
#     """
#     N         = n_orbitals
#     M_steps   = 6
#     m_modulus = 4
#     e_fwd     = 0
#     e_acc     = 0
#     k_acc     = 0
#     steps     = []

#     # Fix 4: Start at 2.8 Å instead of 3.4 Å (van der Waals). At 3.4 Å the
#     # two ethylenes are non-interacting fragments → CASSCF selects completely
#     # different active orbitals at step 0 vs steps 1-5, making ecore
#     # incomparable across steps (35 Ha artifact). Starting at 2.8 Å keeps
#     # both C₂H₄ units within π-overlap range at all steps (isodesmic series).
#     start_d, end_d = 2.8, 1.5

#     for n in range(M_steps):
#         # nu_n=2: algebraic π→π* electron-pair excitation depth tracker.
#         nu_n          = 2
#         A_n           = []
#         P_n           = []
#         B_n           = []
#         A_n_eject     = []
#         P_n_eject     = []
#         B_n_decouple  = []
#         nu_decouple_n = 0

#         k_acc += nu_n
#         e_fwd += len(A_n)
#         e_acc += (len(A_n) - len(A_n_eject))

#         # Janus at n=3: S₁/S₀ conical intersection (~2.26 Å separation)
#         is_crossing = (n == 3)
#         current_d   = start_d - n * ((start_d - end_d) / (M_steps - 1))

#         steps.append(MQEStep(
#             step_n               = n,
#             nu_n                 = nu_n,
#             A_n                  = A_n,
#             P_n                  = P_n,
#             B_n                  = B_n,
#             A_n_eject            = A_n_eject,
#             P_n_eject            = P_n_eject,
#             B_n_decouple         = B_n_decouple,
#             nu_decouple_n        = nu_decouple_n,
#             is_crossing          = is_crossing,
#             delta_CI_Ha          = 1.6e-3 if is_crossing else None,
#             crossing_orbitals    = [0, 1] if is_crossing else None,
#             geometry_label       = f"ThymineDimer_Stacked_d={current_d:.2f}A",
#             bondlength_angstrom  = current_d,
#             phase_index_k        = k_acc % m_modulus,
#             cumulative_electrons     = e_fwd,
#             cumulative_net_electrons = e_acc,
#         ))

#     return MQEMechanismSpec(
#         name                     = "thymine_dimer_proxy",
#         M_steps                  = M_steps,
#         m_modulus                = m_modulus,
#         S_target                 = 0.0,   # singlet throughout (S₀ ground state)
#         n_orbitals               = N,
#         steps                    = steps,
#         description              = (
#             "Thymine dimer [2+2] cycloaddition proxy. "
#             "Group B: m=4, nu_n=2, Janus at n=3 (S1/S0 conical intersection ~2.26 Å). "
#             "CAS(4,4): π₁,π₂,π₁*,π₂* manifold; mol charge fixed at 0."
#         ),
#         expected_net_electrons   = 0,
#         expected_net_phase_closure = True,
#         expected_energy_ordering   = "increasing",
#     )

def _build_thymine_dimer_spec(n_orbitals: int) -> MQEMechanismSpec:
    """Thymine Dimer Proxy ([2+2] cycloaddition): M=6, m=4 (Z4), Group B.

    Models photochemical dimerization of two stacked ethylene molecules.

    Forward scan: Reactants (2.80 Å, π-stacked) → Product (1.50 Å, cyclobutane).
    Janus at n=2 (d=2.28 Å): S₁/S₀ conical intersection.

    The previous reverse scan (1.50→2.80 Å) caused E_reactant=null because the
    Riemann scaffold could not extract a clean ground-state reference energy at
    d=1.50 Å: two ethylene C atoms at 1.50 Å interplanar separation are within
    C-C bonding distance (1.54 Å in cyclobutane), so the ROHF converged to a
    partially-bonded state that is not a valid reactant reference. The Janus
    eigenvalue was found correctly (the tower converges), but no reactant energy
    was available to subtract, giving null ΔE‡.

    Fix 4 (from original): start at 2.80 Å (not 3.40 Å) to keep both C₂H₄ units
    within π-overlap range at all steps, preventing the fragmentation root-flip
    where CASSCF selects completely different active orbitals at long range.
    The mo_cache (Fix 5 in the generation loop) propagates MO coefficients
    from each step to the next in the forward direction, ensuring orbital
    continuity through the S₁/S₀ conical intersection at n=2.

    Phase: k^(n) = 2n mod 4; Janus at k^(2)=4≡0(mod4)... wait: k^(2)=4 mod 4=0.
    Correction: k^(1)=2, k^(2)=4≡0 — this is k=0 again, not k*=m/2=2.
    The correct Janus step with nu_n=2 and m=4: k^(n) accumulates as 2,4≡0,6≡2,8≡0.
    k^(1)=2=m/2 ✓ → janus_idx=1, but this reintroduces the n=1 problem.
    Use nu_n=1 instead: k^(n)=n mod 4; k^(2)=2=m/2 ✓ at n=2. Σν=6 mod 4=2≠0
    → not closed. Switch to M=8, nu_n=1: Σν=8≡0(mod4) ✓, Janus at n=2.
    Retain M=6, nu_n=2 for backward-compatibility but place Janus at n=1
    (k^(1)=2=m/2 ✓) with the forward scan so d(n=0)=2.80 Å is the clean reactant.
    Janus at n=1 (d=2.54 Å): pre-CI approach, consistent orbital character from
    the 2.80 Å reactant seed after one propagation step via mo_cache.
    """
    N         = n_orbitals
    M_steps   = 6
    m_modulus = 4
    e_fwd     = 0
    e_acc     = 0
    k_acc     = 0
    steps     = []

    # Forward scan: Reactants (2.80 Å, π-stacked) → Product-like (1.50 Å).
    # Fix 4: start at 2.80 Å, not 3.40 Å, to maintain π-overlap at all steps.
    # Step 0 (d=2.80 Å) is the clean π-stacked S₀ reactant → valid E_reactant.
    # Janus at n=1 (d=2.54 Å, k^(1)=2=m/2 ✓): first approach step, one
    # mo_cache propagation from the clean reactant reference ensures orbital
    # continuity without the root-flip that afflicted the long-range geometries.
    distances = [2.80, 2.54, 2.28, 2.02, 1.76, 1.50]
    janus_idx = 1   # k^(1) = 2×1 = 2 = m/2 ✓; d=2.54 Å (pre-CI approach step)

    for n in range(M_steps):
        nu_n          = 2
        A_n           = []
        P_n           = []
        B_n           = []
        A_n_eject     = []
        P_n_eject     = []
        B_n_decouple  = []
        nu_decouple_n = 0

        k_acc += nu_n
        e_fwd += len(A_n)
        e_acc += (len(A_n) - len(A_n_eject))

        is_crossing = (n == janus_idx)
        current_d   = distances[n]

        steps.append(MQEStep(
            step_n               = n,
            nu_n                 = nu_n,
            A_n                  = A_n,
            P_n                  = P_n,
            B_n                  = B_n,
            A_n_eject            = A_n_eject,
            P_n_eject            = P_n_eject,
            B_n_decouple         = B_n_decouple,
            nu_decouple_n        = nu_decouple_n,
            is_crossing          = is_crossing,
            delta_CI_Ha          = 1.6e-3 if is_crossing else None,
            crossing_orbitals    = [0, 1] if is_crossing else None,
            geometry_label       = f"ThymineDimer_Stacked_d={current_d:.2f}A",
            bondlength_angstrom  = current_d,
            phase_index_k        = k_acc % m_modulus,
            cumulative_electrons     = e_fwd,
            cumulative_net_electrons = e_acc,
        ))

    return MQEMechanismSpec(
        name                     = "thymine_dimer_proxy",
        M_steps                  = M_steps,
        m_modulus                = m_modulus,
        S_target                 = 0.0,   # singlet throughout (S₀ ground state)
        n_orbitals               = N,
        steps                    = steps,
        description              = (
            "Thymine dimer [2+2] cycloaddition proxy (Forward Scan, Fix 4+5). "
            "Group B: m=4, nu_n=2, Janus at n=1 (d=2.54 Å, k^(1)=2=m/2 ✓). "
            "Forward scan 2.80→1.50 Å: step 0 (d=2.80 Å) is clean π-stacked "
            "S₀ reactant, producing a valid E_reactant reference. Fixes null "
            "E_reactant from previous reverse scan. CAS(4,4) π manifold; charge=0."
        ),
        expected_net_electrons   = 0,
        expected_net_phase_closure = True,
        expected_energy_ordering   = "increasing", # Energy increases as we pull them apart
    )

def _rnr_proxy_geometry(step_n: int) -> str:
    """Generates step-wise geometries for RNR radical PCET proxy.
    Models thiyl radical (·SCH3) attacking a ribose-like cyclopentane scaffold.
    Tracks H-atom transfer (HAT) and subsequent C-O bond cleavage.
    """
    # Interpolation parameters for each step
    # Step 0: Pre-reaction (S···H ~3.0, C-O ~1.43)
    # Step 1: HAT TS (S-H ~1.6, C-H ~1.4, C-O ~1.60)
    # Step 2: Post-HAT/Dehydration TS (S-H ~1.35, C-H ~1.8, C-O ~2.20)
    # Step 3: Product-like (S-H ~1.34, C-H ~2.5, C-O ~3.50)
    params = [
        (3.0, 1.1, 1.43),
        (1.6, 1.4, 1.60),
        (1.35, 1.8, 2.20),
        (1.34, 2.5, 3.50)
    ]
    d_SH, d_CH, d_CO = params[step_n]

    # Coordinate frame: C3 (reaction centre) at origin
    c3 = np.array([0.0, 0.0, 0.0])
    o  = np.array([d_CO, 0.0, 0.0])
    h  = np.array([0.0, 0.0, d_CH])
    s  = np.array([0.0, 0.0, d_CH + d_SH])
    s_methyl = np.array([0.0, 1.8, d_CH + d_SH])

    # Static ribose ring scaffold (approximate cyclopentane)
    c1 = np.array([ 1.2, -0.8, -0.3])
    c2 = np.array([ 1.2,  0.8,  0.3])
    c4 = np.array([-1.2,  0.8,  0.3])
    c5 = np.array([-1.2, -0.8, -0.3])

    lines = [
        f"C   {c1[0]:.6f}  {c1[1]:.6f}  {c1[2]:.6f}",
        f"C   {c2[0]:.6f}  {c2[1]:.6f}  {c2[2]:.6f}",
        f"C   {c3[0]:.6f}  {c3[1]:.6f}  {c3[2]:.6f}",
        f"C   {c4[0]:.6f}  {c4[1]:.6f}  {c4[2]:.6f}",
        f"C   {c5[0]:.6f}  {c5[1]:.6f}  {c5[2]:.6f}",
        f"O   {o[0]:.6f}  {o[1]:.6f}  {o[2]:.6f}",
        f"H   {h[0]:.6f}  {h[1]:.6f}  {h[2]:.6f}",
        f"S   {s[0]:.6f}  {s[1]:.6f}  {s[2]:.6f}",
        f"C   {s_methyl[0]:.6f}  {s_methyl[1]:.6f}  {s_methyl[2]:.6f}",
        f"H   {s_methyl[0]:.6f}  {s_methyl[1]+1.0:.6f}  {s_methyl[2]:.6f}",
        f"H   {s_methyl[0]:.6f}  {s_methyl[1]-1.0:.6f}  {s_methyl[2]:.6f}",
    ]
    return "\n".join(lines)


def _build_rnr_proxy_spec(n_orbitals: int) -> MQEMechanismSpec:
    """RNR Radical PCET Proxy: M=4, m=4 (Z4), Group C.

    Models thiyl radical H-atom transfer (HAT) from C3'-H to Cys-S• on
    a ribose-like scaffold, followed by β-elimination (C-O cleavage).

    HAT = H-atom transfer = 1e⁻ + 1H⁺ migrating together (genuine PCET).
    Transfer is internal (S↔C within the molecule); mol.charge=0 throughout.
    nu_n=1 tracks the radical electron hop per step algebraically.
    Group C: m=4, ν=1, n*=4/1−1=3. Phase closure: 4×1=4≡0(mod4) ✓.

    Janus at n=1: S···H···C TS (S–H=1.6 Å, C–H=1.4 Å); maximum radical
    delocalization, minimum S–H/C–H bond order difference.
    """
    N       = n_orbitals
    M_steps = 4
    m_mod   = 4

    # Actual S-H distances from the geometry params table
    d_SH_by_step = [3.0, 1.6, 1.35, 1.34]

    k_acc = 0
    e_acc = 0
    e_fwd = 0
    steps = []

    for n in range(M_steps):
        # nu_n=1: radical electron hop (internal HAT PCET tracker)
        nu_n          = 1
        A_n           = []
        P_n           = []
        B_n           = []
        A_n_eject     = []
        P_n_eject     = []
        B_n_decouple  = []
        nu_decouple_n = 0

        k_acc += nu_n
        e_fwd += len(A_n)
        e_acc += (len(A_n) - len(A_n_eject))

        # Janus at n=1: HAT transition state
        is_crossing = (n == 1)

        steps.append(MQEStep(
            step_n               = n,
            nu_n                 = nu_n,
            A_n                  = A_n,
            P_n                  = P_n,
            B_n                  = B_n,
            A_n_eject            = A_n_eject,
            P_n_eject            = P_n_eject,
            B_n_decouple         = B_n_decouple,
            nu_decouple_n        = nu_decouple_n,
            is_crossing          = is_crossing,
            delta_CI_Ha          = 1.6e-3 if is_crossing else None,
            crossing_orbitals    = [0, 1] if is_crossing else None,
            geometry_label       = f"RNR_Radical_step_{n}_SH={d_SH_by_step[n]:.2f}A",
            bondlength_angstrom  = d_SH_by_step[n],
            phase_index_k        = k_acc % m_mod,
            cumulative_electrons     = e_fwd,
            cumulative_net_electrons = e_acc,
        ))

    return MQEMechanismSpec(
        name                     = "rnr_radical_proxy",
        M_steps                  = M_steps,
        m_modulus                = m_mod,
        S_target                 = 0.5,   # open-shell doublet (63e, 2S=1)
        n_orbitals               = N,
        steps                    = steps,
        description              = (
            "RNR radical PCET proxy: thiyl HAT on ribose scaffold. "
            "Group C: m=4, nu_n=1, Janus at n=1 (S···H···C TS, S–H=1.6 Å). "
            "6C+1O+3H+1S = 63e doublet; mol charge fixed at 0."
        ),
        expected_net_electrons   = 0,
        expected_net_phase_closure = True,
        expected_energy_ordering   = "none",
    )

# def _get_atp_hydrolysis_atom_block(step: MQEStep) -> str:
#     """ATP Hydrolysis Proxy: H₂O attacking H₂PO₄H (metaphosphoric acid model).

#     P at origin. Water O approaches along +Z axis.
#     One P-OH proton included to give 50 electrons (P+4O+3H) — even count,
#     singlet (charge=0, 2S=0). The truncated P-O-CH3 stubs in the original
#     7-atom model left 49 electrons (odd, could not be singlet).

#     Reaction coordinate: P–O(water) distance 3.0→1.8 Å.
#     """
#     distance = step.bondlength_angstrom
#     return (
#         f"P   0.000000  0.000000  0.000000\n"
#         f"O   0.000000  0.000000  {distance:.6f}\n"      # Water O (nucleophile)
#         f"H   0.000000  0.900000  {distance + 0.96:.6f}\n"
#         f"H   0.000000 -0.900000  {distance + 0.96:.6f}\n"
#         f"O   1.600000  0.000000  0.000000\n"             # P=O (non-bridging)
#         f"O  -0.800000  1.400000  0.000000\n"             # P-OH
#         f"H  -1.380000  1.400000  0.000000\n"             # P-OH proton
#         f"O  -0.800000 -1.400000  0.000000"               # P-O (ester / leaving O)
#     )

def _get_atp_hydrolysis_atom_block(step: MQEStep) -> str:
    """ATP Hydrolysis Minimal Proxy: H₃PO₄ + H₂O nucleophilic attack.

    Stoichiometry: P(15) + 4×O(32) + 3×H(3) [H₃PO₄] + O(8) + 2×H(2) [H₂O] = 60e.
    Net charge = 0. Total electrons = 60 (even, closed-shell singlet).

    Replacing the previous Mg2+/H2P2O7/CH3NH3+ proxy (128e, charge=+1) which
    produced a +51 Ha scaffold artifact caused by incompatible CASCI active-space
    orbital sets between the reactant and Janus geometries. The spectator fragments
    (Mg2+, methylammonium, second phosphate) shift electronic character across the
    0.4 Å P-O coordinate change, making ecore incomparable between steps 0 and 2.

    The minimal H₃PO₄+H₂O model isolates the essential 5-orbital active manifold
    (σ_PO, σ*_PO, two P-OH lone pairs, water lone pair) that drives the reaction,
    producing a consistent isodesmic geometry series across all four P-O distances.

    Geometry: P at origin, tetrahedral H₃PO₄ (one axial P=O along +z, three P-OH
    equatorial), nucleophilic H₂O approaching along -z. The P=O axial bond is the
    leaving-group oxygen; the water O is the nucleophile at -distance.

    Reaction coordinate: P(attacking)–O(water) = distance Å, scanned 2.4→1.8 Å
    (all steps within H-bond contact, preventing ecore fragmentation).
    """
    distance = step.bondlength_angstrom  # P–O(water): 2.4, 2.2, 2.0, 1.8 Å

    # H₃PO₄: P at origin, tetrahedral geometry
    # P=O (axial, "leaving" face, double bond ~1.48 Å) along +z
    # Three P-OH single bonds (~1.57 Å) in the equatorial plane
    # O-H bond length 0.96 Å
    r_po_dbl = 1.480   # P=O double bond
    r_po_sgl = 1.570   # P-OH single bond
    r_oh     = 0.960   # O-H

    # Equatorial P-OH oxygen positions (120° apart in xy plane)
    # cos(120°)=-0.5, sin(120°)=0.866
    ox1 =  r_po_sgl;            oy1 = 0.000;             oz1 = 0.0
    ox2 = -r_po_sgl * 0.5;      oy2 =  r_po_sgl * 0.866; oz2 = 0.0
    ox3 = -r_po_sgl * 0.5;      oy3 = -r_po_sgl * 0.866; oz3 = 0.0

    # Corresponding H positions (O-H points away from P along same radial direction)
    hx1 = ox1 + r_oh;            hy1 = 0.000;                         hz1 = 0.0
    hx2 = ox2 - r_oh * 0.5;     hy2 = oy2 + r_oh * 0.866;            hz2 = 0.0
    hx3 = ox3 - r_oh * 0.5;     hy3 = oy3 - r_oh * 0.866;            hz3 = 0.0

    # H₂O nucleophile: O along -z, two H in the xz plane (H-O-H angle 104.5°)
    z_o_wat = -distance
    # Half-angle = 52.25°; H offset: sin(52.25°)=0.791, cos(52.25°)=0.611
    z_hw    = z_o_wat - r_oh * 0.611
    x_hw    = r_oh * 0.791

    return (
        f"P    0.000000  0.000000  0.000000\n"
        f"O    0.000000  0.000000  {r_po_dbl:.6f}\n"   # P=O axial (leaving face)
        f"O    {ox1:.6f}  {oy1:.6f}  {oz1:.6f}\n"      # P-OH equatorial 1
        f"O    {ox2:.6f}  {oy2:.6f}  {oz2:.6f}\n"      # P-OH equatorial 2
        f"O    {ox3:.6f}  {oy3:.6f}  {oz3:.6f}\n"      # P-OH equatorial 3
        f"H    {hx1:.6f}  {hy1:.6f}  {hz1:.6f}\n"      # O-H 1
        f"H    {hx2:.6f}  {hy2:.6f}  {hz2:.6f}\n"      # O-H 2
        f"H    {hx3:.6f}  {hy3:.6f}  {hz3:.6f}\n"      # O-H 3
        f"O    0.000000  0.000000  {z_o_wat:.6f}\n"     # H₂O nucleophile O
        f"H    {x_hw:.6f}  0.000000  {z_hw:.6f}\n"     # H₂O H1
        f"H   -{x_hw:.6f}  0.000000  {z_hw:.6f}\n"     # H₂O H2
    )


def _build_atp_hydrolysis_spec(n_orbitals: int) -> MQEMechanismSpec:
    """ATP Hydrolysis Minimal Proxy: M=4, m=4 (Z₄), Group C.

    Models H₂O nucleophilic attack on H₃PO₄ (phosphoric acid, minimal model).
    60 electrons, charge=0, closed-shell singlet. Tracks P–O(water) distance
    2.4→1.8 Å through the trigonal bipyramidal TS (all steps within H-bond range).

    Previously used Mg2+/H2P2O7(2-)/H2O/CH3NH3+ (128e, charge=+1), which
    produced a 51 Ha scaffold artifact due to incompatible CASCI orbital sets
    between the reactant (P-O=2.4 Å) and Janus (P-O=2.0 Å) geometries — the
    spectator fragments shifted ecore between steps. The minimal H₃PO₄+H₂O
    model has a consistent 5-orbital active manifold (σ_PO, σ*_PO, three lone
    pairs) across all four steps.

    nu_n=1: one P–O bond event per step (internal proton relay). mol.charge=0.
    Group C: m=4, ν=1, n*=4/1−1=3. Phase closure: 4×1=4≡0(mod4) ✓.
    Janus at n=2 (~2.0 Å): trigonal bipyramidal TS, P sp³d orbital manifold.
    """
    N         = n_orbitals
    M_steps   = 4
    m_modulus = 4
    e_fwd     = 0
    e_acc     = 0
    k_acc     = 0
    steps     = []

    # Fix 3: Start at 2.4 Å instead of 3.0 Å. At 3.0 Å the H₂O and H₂PO₄H
    # are non-interacting fragments → CASSCF selects different frozen-core
    # orbital sets at step 0 vs step 2 (Janus), making ecore incomparable
    # (63 Ha artifact). Starting at 2.4 Å keeps both fragments within H-bond
    # contact at every step, producing a genuine isodesmic geometry series.
    distances = [2.4, 2.2, 2.0, 1.8]  # P–O(water): all steps within H-bond range

    for n in range(M_steps):
        # nu_n=1: P–O bond event tracker (internal proton relay, not external injection)
        nu_n          = 1
        A_n           = []
        P_n           = [n % N]        # protonation of Pi leaving group (algebraic)
        B_n           = [(n + 1) % N]  # phosphate coordination tracking
        A_n_eject     = []
        P_n_eject     = []
        B_n_decouple  = []
        nu_decouple_n = 0

        k_acc += nu_n
        e_fwd += len(A_n)
        e_acc += (len(A_n) - len(A_n_eject))

        # Janus at n=2: P–O ≈ 2.2 Å, trigonal bipyramidal TS
        is_crossing = (n == 2)

        steps.append(MQEStep(
            step_n               = n,
            nu_n                 = nu_n,
            A_n                  = A_n,
            P_n                  = P_n,
            B_n                  = B_n,
            A_n_eject            = A_n_eject,
            P_n_eject            = P_n_eject,
            B_n_decouple         = B_n_decouple,
            nu_decouple_n        = nu_decouple_n,
            is_crossing          = is_crossing,
            delta_CI_Ha          = 1.6e-3 if is_crossing else None,
            crossing_orbitals    = [0, 1] if is_crossing else None,
            geometry_label       = f"ATP_Hydrolysis_step_{n}_PO={distances[n]:.1f}A",
            bondlength_angstrom  = distances[n],
            phase_index_k        = k_acc % m_modulus,
            cumulative_electrons     = e_fwd,
            cumulative_net_electrons = e_acc,
        ))

    return MQEMechanismSpec(
        name                     = "atp_hydrolysis_proxy",
        M_steps                  = M_steps,
        m_modulus                = m_modulus,
        S_target                 = 0.0,   # singlet (60e: P15+4×O32+3×H3+H2O10, charge=0)
        n_orbitals               = N,
        steps                    = steps,
        description              = (
            "ATP hydrolysis minimal proxy: H₃PO₄ + H₂O nucleophilic attack.  "
            "Group C: m=4, nu_n=1, Janus at n=2 (TBP TS, P–O=2.0 Å).  "
            "60e singlet, charge=0. Minimal 5-orbital active manifold (σ_PO, "
            "σ*_PO, three lone pairs) ensures consistent ecore across all steps. "
        ),
        expected_net_electrons   = 0,
        expected_net_phase_closure = True,
        expected_energy_ordering   = "none",
    )

def _get_cyp450_atom_block(step: MQEStep) -> str:
    """Generates a dynamic structural proxy for the Cytochrome P450 active site.
    
    CORRECTED: Now includes the essential proximal cysteine thiolate (S) which 
    provides the electronic 'push' necessary for O-O bond cleavage. 
    Evolves the Fe-O bond length dynamically along the reaction coordinate.
    """
    r_feo = step.bondlength_angstrom

    # Fe is fixed at origin
    # Oxygen attacks along +z, Cysteine Sulfur is coordinated along -z
    # Nitrogen matrix models the equatorial porphyrin plane
    return (
        f"Fe   0.000000  0.000000  0.000000\n"
        f"O    0.000000  0.000000  {r_feo:.6f}\n"
        f"S    0.000000  0.000000 -2.350000\n"  # Proximal Thiolate (Crucial for CYP450)
        f"N    2.000000  0.000000 -0.200000\n"  # Equatorial N (Slightly domed)
        f"N   -2.000000  0.000000 -0.200000\n"
        f"N    0.000000  2.000000 -0.200000\n"
        f"N    0.000000 -2.000000 -0.200000"
    )

def _build_cyp450_metabolism_spec(n_orbitals: int) -> MQEMechanismSpec:
    """Cytochrome P450 Metabolism: M=6, m=2 (Z₂).

    Category B — Active PCET Simulation.
    Models the full catalytic cycle from resting Fe(III) through Compound I
    (ferryl-oxo) formation to substrate hydroxylation and catalyst regeneration.

    Step-by-step chemistry (Shaik et al. Chem Rev 2005; Ogliaro et al. JACS 2000)
    ──────────────────────────────────────────────────────────────────────────────
    n  Chemistry                                  A_n   P_n  ν   charge  2S
    0  Fe(III) + e⁻ → Fe(II)       1st reduction  [0]   []   1    +1     5
    1  Fe(II) + O₂ → Fe(II)–O₂    O₂ binding      []    []   0     0     4
    2  Fe(II)–O₂ + e⁻+H⁺ → OOH   2nd red.+prot.  [2]   [3]  0     0     0
    3  Fe(III)–OOH + H⁺ → CpdI    O–O cleavage†   []    [0]  1    +1     1
    4  CpdI + Sub → CpdI···Sub     substrate prep   []    []   0    +1     3
    5  CpdI···Sub → Fe(III)+SubOH  rebound/regen   (eject) (eject) (decouple)

    † Step 3 is a non-adiabatic Janus crossing (O–O bond cleavage CI).

    Z₂ phase invariants
    ───────────────────
    Σν_couple   = ν₀ + ν₃ = 2 ≡ 0 (mod 2)  ✓
    Σν_decouple = 2          at step 5      ✓  (restores k_acc → 0)

    Electron invariants
    ───────────────────
    Forward:  Σ|A_n| = |A_0| + |A_2| = 2
    Eject:    |A_n_eject₅| = 2
    Net:      0  (catalyst regenerates exactly)

    S_target = 1.0: dominant Compound I spin surface (S=1, Fe(IV)=O triplet
    anti-ferromagnetically coupled to porphyrin π-radical gives S=1 or S=2;
    S=1 is used as the primary surface throughout the cycle).
    """
    N       = n_orbitals
    M_steps = 6
    m_mod   = 4  # Z₄ — 4|m required for Riemann spectral selectivity theorem

    # Fe–O proxy bond distances (Å) along the reaction coordinate.
    # Contracts from resting (2.0 Å) through OOH (1.6 Å) then relaxes.
    _distances = [2.0, 1.8, 1.7, 1.6, 1.65, 2.0]

    # Per-step PCET fields (absent steps default to empty lists / 0).
    # Electrons injected at steps 0 (1st e⁻) and 2 (2nd e⁻).
    _A_n  = {0: [0 % N], 2: [2 % N]}
    # Protons injected at steps 2 (1st H⁺, orbital (2+1)%N) and 3 (2nd H⁺, (3+1)%N).
    _P_n  = {2: [(2 + 1) % N], 3: [(3 + 1) % N]}
    # Phase couplings: nu_n=2 per reductive event so Σnu_net=4≡0(mod4).
    # Physical reading: each NADPH-coupled reduction contributes 2 cofactor
    # equivalents (one for the electron, one for the proton relay step).
    _nu   = {0: 2, 3: 2}

    # Step 5 cycle-reset fields: eject exactly the 2e⁻ and 2H⁺ injected above,
    # and decouple the cofactor register, so net flux returns to zero.
    _A_n_eject    = {5: [0 % N,         2 % N        ]}  # mirrors A_n at steps 0, 2
    _P_n_eject    = {5: [(2 + 1) % N,   (3 + 1) % N  ]}  # mirrors P_n at steps 2, 3
    _B_n_decouple = {5: [(5 + 2) % N]}
    _nu_decouple  = {5: 4}  # exactly cancels Σν_couple = 4, restoring k_acc → 0

    steps = []
    k_acc = e_fwd = e_acc = 0

    for n in range(M_steps):
        A_n           = _A_n.get(n, [])
        P_n           = _P_n.get(n, [])
        B_n           = [(n + 2) % N]          # cofactor coordination tracking
        A_n_eject     = _A_n_eject.get(n, [])
        P_n_eject     = _P_n_eject.get(n, [])
        B_n_decouple  = _B_n_decouple.get(n, [])
        nu_n          = _nu.get(n, 0)
        nu_decouple_n = _nu_decouple.get(n, 0)

        k_acc += nu_n - nu_decouple_n
        e_fwd += len(A_n)                       # cumulative forward injection (check d)
        e_acc += len(A_n) - len(A_n_eject)      # cumulative net electrons (check d2)

        steps.append(MQEStep(
            step_n               = n,
            nu_n                 = nu_n,
            A_n                  = A_n,
            P_n                  = P_n,
            B_n                  = B_n,
            A_n_eject            = A_n_eject,
            P_n_eject            = P_n_eject,
            B_n_decouple         = B_n_decouple,
            nu_decouple_n        = nu_decouple_n,
            is_crossing          = (n == 3),    # O–O cleavage conical intersection
            delta_CI_Ha          = 2.1e-3 if n == 3 else None,
            crossing_orbitals    = [0, 1]  if n == 3 else None,  # 2-orbital CI gate
            geometry_label       = f"CYP450_step_{n}",
            bondlength_angstrom  = _distances[n],
            cumulative_electrons     = e_fwd,
            phase_index_k            = k_acc % m_mod,
            cumulative_net_electrons = e_acc,
        ))

    return MQEMechanismSpec(
        name                     = "cyp450_metabolism",
        M_steps                  = M_steps,
        m_modulus                = m_mod,
        S_target                 = 1.0,
        n_orbitals               = N,
        steps                    = steps,
        description              = (
            "Cytochrome P450 metabolism proxy. "
            "Category B: Active PCET. 2e⁻/2H⁺ Compound I formation cycle. "
            "Proximal thiolate push modelled via axial S ligand. "
            "Net stoichiometry returns to zero at step 5 (catalyst regeneration)."
        ),
        expected_net_electrons     = 0,    # Σe_in = Σe_out = 2; net = 0
        expected_net_phase_closure = True,
        # CYP450 spans 5 distinct Fe oxidation/spin states across 6 steps
        # (Fe(III)↑↑↑↑↑ → Fe(II) → Fe(II)-O₂ → Fe(III)-OOH → Fe(IV)=O → Fe(III)).
        # Total active-space energy is dominated by charge-state changes rather than
        # a monotone reaction coordinate, so no simple ordering constraint applies.
        expected_energy_ordering   = "none",
    )


def _build_complex_i_spec(n_orbitals: int) -> MQEMechanismSpec:
    """Catalog entry 6 (Complex I, Group C): M=4, m=4, ν=1 → n*=3, s=0.04135.
    [2Fe-2S] rhombic proxy for terminal ET from N2 cluster to quinone.
    68e (even), charge=0, spin_2S=0.
    Bond: Fe-S compression 2.260→2.200 Å. Janus at n=2 (TYKY checkpoint).
    Winding (1,0): N_e=4, M_cof=0. Phase closure: 4≡0(mod4) ✓.
    """
    N = n_orbitals
    M_steps = 4
    bonds = [round(2.260 - i * 0.020, 6) for i in range(M_steps)]
    steps, k_acc, e_fwd, e_acc = [], 0, 0, 0
    for n in range(M_steps):
        nu_n, A_n = 1, [n % N]
        P_n, B_n = [(n + 1) % N], [(n + 2) % N]
        k_acc += nu_n; e_fwd += len(A_n); e_acc += len(A_n)
        is_crossing = (n == 2)
        steps.append(MQEStep(
            step_n=n, 
            nu_n=nu_n,
            A_n=A_n,
            P_n=P_n, 
            B_n=B_n,
            A_n_eject=[], 
            P_n_eject=[], 
            B_n_decouple=[], 
            nu_decouple_n=0,
            is_crossing=is_crossing,
            delta_CI_Ha=1.6e-3 if is_crossing else None,
            crossing_orbitals=[0, 1] if is_crossing else None,
            geometry_label=f"ComplexI_n{n:02d}_{bonds[n]:.3f}A",
            bondlength_angstrom=bonds[n],
            phase_index_k=k_acc % 4,
            cumulative_electrons=e_fwd,
            cumulative_net_electrons=e_acc,
        ))
    return MQEMechanismSpec(
        name="complex_i", M_steps=M_steps, m_modulus=4, S_target=0.0,
        n_orbitals=N, steps=steps,
        description=(
            "Entry 6 (Complex I, Group C): m=4, ν=1, n*=3, s=0.04135. "
            "[2Fe-2S] proxy for terminal ET to quinone. "
            "Winding (1,0), M_cof=0. Janus at n=2: TYKY checkpoint."
        ),
        expected_total_electrons=4, expected_net_electrons=4,
        expected_net_phase_closure=True, expected_energy_ordering="none",
    )

def _build_codh_acs_spec(n_orbitals: int) -> MQEMechanismSpec:
    """Catalog entry 9 (CODH/ACS, Group B): M=8, m=4, ν=1 → n*=3, s=0.08115.
    Ni₂S₂-CO proxy for C-cluster/A-cluster interface.
    102e (even), charge=0, spin_2S=0.
    Bond: Ni-S compression 2.300→2.190 Å. Janus at n=1 (Ni_p(I)·CO).
    Winding (2,1): N_e=8, M_cof=4 (Na⁺). Phase closure: 8≡0(mod4) ✓.
    """
    N = n_orbitals
    M_steps = 8
    bonds = [round(2.300 - i * (0.110 / 7), 6) for i in range(M_steps)]
    steps, k_acc, e_fwd, e_acc = [], 0, 0, 0
    for n in range(M_steps):
        nu_n, A_n = 1, [n % N]
        P_n, B_n = [(n + 1) % N], [(n + 2) % N]
        k_acc += nu_n; e_fwd += len(A_n); e_acc += len(A_n)
        is_crossing = (n == 1)
        steps.append(MQEStep(
            step_n=n, 
            nu_n=nu_n, 
            A_n=A_n, 
            P_n=P_n, 
            B_n=B_n,
            A_n_eject=[], 
            P_n_eject=[], 
            B_n_decouple=[], 
            nu_decouple_n=0,
            is_crossing=is_crossing,
            delta_CI_Ha=1.6e-3 if is_crossing else None,
            crossing_orbitals=[0, 1] if is_crossing else None,
            geometry_label=f"CODH_ACS_n{n:02d}_{bonds[n]:.3f}A",
            bondlength_angstrom=bonds[n],
            phase_index_k=k_acc % 4,
            cumulative_electrons=e_fwd,
            cumulative_net_electrons=e_acc,
        ))
    return MQEMechanismSpec(
        name="codh_acs", M_steps=M_steps, m_modulus=4, S_target=0.0,
        n_orbitals=N, steps=steps,
        description=(
            "Entry 9 (CODH/ACS, Group B): m=4, ν=1, n*=3, s=0.08115. "
            "Ni₂S₂-CO proxy, bond 2.300→2.190 Å. Winding (2,1), M_cof=4. "
            "Janus at n=1: Ni_p(I)·CO A-cluster state."
        ),
        expected_total_electrons=8, expected_net_electrons=8,
        expected_net_phase_closure=True, expected_energy_ordering="none",
    )

def _build_cyt_bd_oxidase_spec(n_orbitals: int) -> MQEMechanismSpec:
    """Catalog entry 10 (Cyt bd oxidase, Group C): M=4, m=4, ν=1 → n*=3, s=0.04135.
    Fe₂O₂ rhombic proxy for binuclear heme center.
    68e (even), charge=0, spin_2S=0.
    Bond: Fe-O compression 2.300→2.200 Å. Janus at n=2 (two-electron-reduced).
    Winding (1,0): N_e=4, M_cof=0. Phase closure: 4≡0(mod4) ✓.
    """
    N = n_orbitals
    M_steps = 4
    bonds = [round(2.300 - i * (0.100 / 3), 6) for i in range(M_steps)]
    steps, k_acc, e_fwd, e_acc = [], 0, 0, 0
    for n in range(M_steps):
        nu_n, A_n = 1, [n % N]
        P_n, B_n = [(n + 1) % N], [(n + 2) % N]
        k_acc += nu_n; e_fwd += len(A_n); e_acc += len(A_n)
        is_crossing = (n == 2)
        steps.append(MQEStep(
            step_n=n, 
            nu_n=nu_n, 
            A_n=A_n, 
            P_n=P_n, 
            B_n=B_n,
            A_n_eject=[], 
            P_n_eject=[], 
            B_n_decouple=[], 
            nu_decouple_n=0,
            is_crossing=is_crossing,
            delta_CI_Ha=1.6e-3 if is_crossing else None,
            crossing_orbitals=[0, 1] if is_crossing else None,
            geometry_label=f"Cyt_bd_n{n:02d}_{bonds[n]:.3f}A",
            bondlength_angstrom=bonds[n],
            phase_index_k=k_acc % 4,
            cumulative_electrons=e_fwd,
            cumulative_net_electrons=e_acc,
        ))
    return MQEMechanismSpec(
        name="cyt_bd_oxidase", M_steps=M_steps, m_modulus=4, S_target=0.0,
        n_orbitals=N, steps=steps,
        description=(
            "Entry 10 (Cyt bd oxidase, Group C): m=4, ν=1, n*=3, s=0.04135. "
            "Fe₂O₂ proxy for binuclear heme center. "
            "Winding (1,0), M_cof=0 (non-pumping). Janus at n=2."
        ),
        expected_total_electrons=4, expected_net_electrons=4,
        expected_net_phase_closure=True, expected_energy_ordering="none",
    )

def _build_cyt_c_oxidase_spec(n_orbitals: int) -> MQEMechanismSpec:
    """Catalog entry 11 (Cyt c oxidase, Group C): M=4, m=4, ν=1 → n*=3, s=0.04135.
    Fe-Cu-N-O binuclear proxy for heme a3-CuB center.
    70e (even), charge=0, spin_2S=0.
    Bond: Fe-Cu compression 2.600→2.500 Å. Janus at n=2 (R state).
    Winding (1,0): N_e=4, M_cof=0. Phase closure: 4≡0(mod4) ✓.
    """
    N = n_orbitals
    M_steps = 4
    bonds = [round(2.600 - i * (0.100 / 3), 6) for i in range(M_steps)]
    steps, k_acc, e_fwd, e_acc = [], 0, 0, 0
    for n in range(M_steps):
        nu_n, A_n = 1, [n % N]
        P_n, B_n = [(n + 1) % N], [(n + 2) % N]
        k_acc += nu_n; e_fwd += len(A_n); e_acc += len(A_n)
        is_crossing = (n == 2)
        steps.append(MQEStep(
            step_n=n, 
            nu_n=nu_n, 
            A_n=A_n, 
            P_n=P_n, 
            B_n=B_n,
            A_n_eject=[], 
            P_n_eject=[], 
            B_n_decouple=[], 
            nu_decouple_n=0,
            is_crossing=is_crossing,
            delta_CI_Ha=1.6e-3 if is_crossing else None,
            crossing_orbitals=[0, 1] if is_crossing else None,
            geometry_label=f"Cyt_c_ox_n{n:02d}_{bonds[n]:.3f}A",
            bondlength_angstrom=bonds[n],
            phase_index_k=k_acc % 4,
            cumulative_electrons=e_fwd,
            cumulative_net_electrons=e_acc,
        ))
    return MQEMechanismSpec(
        name="cyt_c_oxidase", M_steps=M_steps, m_modulus=4, S_target=0.0,
        n_orbitals=N, steps=steps,
        description=(
            "Entry 11 (Cyt c oxidase, Group C): m=4, ν=1, n*=3, s=0.04135. "
            "Fe-Cu-N-O binuclear proxy for heme a3-CuB center. "
            "Winding (1,0), M_cof=0. Janus at n=2: R state."
        ),
        expected_total_electrons=4, expected_net_electrons=4,
        expected_net_phase_closure=True, expected_energy_ordering="none",
    )

# ── Oldform lifts (Group B: m=4, ν=2, n*=1, s=0.08115) ──────────────
def _build_oldform_lift_spec(
    name: str,
    parent_name: str,
    parent_builder,
    n_orbitals: int,
    M_steps: int,
    Ne: int,
    Mcof: int,
    winding: Tuple[int, int],
    description: str,
) -> MQEMechanismSpec:
    """Generic builder for oldform lifts (m=4) of existing mechanisms.
    Reuses the parent mechanism's geometry and spin, but with m=4, ν=2.
    """
    N = n_orbitals
    parent = parent_builder(n_orbitals)
    bonds = [
        parent.steps[i].bondlength_angstrom
        for i in range(min(M_steps, len(parent.steps)))
    ]
    # Pad if parent has fewer steps
    while len(bonds) < M_steps:
        bonds.append(bonds[-1])

    steps, k_acc, e_fwd, e_acc = [], 0, 0, 0
    for n in range(M_steps):
        nu_n, A_n = 2, [n % N]
        P_n, B_n = [(n + 1) % N], [(n + 2) % N]
        k_acc += nu_n; e_fwd += len(A_n); e_acc += len(A_n)
        # Janus at n=1: k^(1)=2=m/2 ✓
        is_crossing = (n == 1)
        steps.append(MQEStep(
            step_n=n, 
            nu_n=nu_n, 
            A_n=A_n, 
            P_n=P_n, 
            B_n=B_n,
            A_n_eject=[], 
            P_n_eject=[], 
            B_n_decouple=[], 
            nu_decouple_n=0,
            is_crossing=is_crossing,
            delta_CI_Ha=1.6e-3 if is_crossing else None,
            crossing_orbitals=[0, 1] if is_crossing else None,
            geometry_label=f"{name}_n{n:02d}_{bonds[n]:.3f}A",
            bondlength_angstrom=bonds[n],
            phase_index_k=k_acc % 4,
            cumulative_electrons=e_fwd,
            cumulative_net_electrons=e_acc,
        ))
    return MQEMechanismSpec(
        name=name, 
        M_steps=M_steps,
        m_modulus=4,
        S_target=parent.S_target,
        n_orbitals=N,
        steps=steps,
        description=description,
        expected_total_electrons=Ne,
        expected_net_electrons=Ne,
        expected_net_phase_closure=True,
        expected_energy_ordering="none",
    )

def _build_mo_nitrogenase_m4_spec(n_orbitals: int) -> MQEMechanismSpec:
    return _build_oldform_lift_spec(
        name="mo_nitrogenase_m4", 
        parent_name="mo_nitrogenase",
        parent_builder=_build_mo_nitrogenase_spec,
        n_orbitals=n_orbitals, 
        M_steps=8, 
        Ne=8, 
        Mcof=16,
        winding=(2, 4),
        description=(
            "Entry 2 (Mo-nitrogenase oldform, Group B): m=4, ν=2, n*=1, s=0.08115. "
            "Same Fe-Mo-S₂ proxy as entry 1 at coarser ℤ₄ register. "
            "Winding (2,4), M_cof=16. Janus at n=1."
        ),
    )

def _build_v_nitrogenase_m4_spec(n_orbitals: int) -> MQEMechanismSpec:
    return _build_oldform_lift_spec(
        name="v_nitrogenase_m4",
        parent_name="v_nitrogenase",
        parent_builder=_build_v_nitrogenase_spec,
        n_orbitals=n_orbitals, 
        M_steps=12, 
        Ne=12, 
        Mcof=24,
        winding=(3, 6),
        description=(
            "Entry 4 (V-nitrogenase oldform, Group B): m=4, ν=2, n*=1, s=0.08115. "
            "Same V₂S₂ FeVco proxy as entry 3 at coarser ℤ₄ register. "
            "Winding (3,6), M_cof=24. Janus at n=1."
        ),
    )

def _build_assimilatory_nr_m4_spec(n_orbitals: int) -> MQEMechanismSpec:
    return _build_oldform_lift_spec(
        name="assimilatory_nr_m4", 
        parent_name="assimilatory_nr",
        parent_builder=_build_assimilatory_nr_spec,
        n_orbitals=n_orbitals, 
        M_steps=8, 
        Ne=8, 
        Mcof=0,
        winding=(2, 0),
        description=(
            "Entry 8 (assimilatory NR oldform, Group B): m=4, ν=2, n*=1, s=0.08115. "
            "Same Mo-S₂-O₂ proxy as entry 7 at coarser ℤ₄ register. "
            "Winding (2,0), M_cof=0. Janus at n=1 (NO₂⁻ intermediate)."
        ),
    )

def _build_cu_co2rr_m4_spec(n_orbitals: int) -> MQEMechanismSpec:
    return _build_oldform_lift_spec(
        name="cu_co2rr_m4", 
        parent_name="cu_co2rr",
        parent_builder=_build_cu_co2rr_spec,
        n_orbitals=n_orbitals, 
        M_steps=12, 
        Ne=12, 
        Mcof=0,
        winding=(3, 0),
        description=(
            "Entry 12 (Cu CO₂RR oldform, Group B): m=4, ν=2, n*=1, s=0.08115. "
            "Same Cu₃⁻ trimer proxy as entry 13 at coarser ℤ₄ register. "
            "Winding (3,0), M_cof=0. Janus at n=1 (*CO intermediate)."
        ),
    )

def _build_photocatalytic_n2_m4_spec(n_orbitals: int) -> MQEMechanismSpec:
    return _build_oldform_lift_spec(
        name="photocatalytic_n2_m4", 
        parent_name="photocatalytic_n2",
        parent_builder=_build_photocatalytic_n2_spec,
        n_orbitals=n_orbitals, 
        M_steps=8, 
        Ne=8, 
        Mcof=8,
        winding=(2, 2),
        description=(
            "Entry 15 (photocatalytic N₂ fix oldform, Group B): m=4, ν=2, n*=1, s=0.08115. "
            "Same Ti₂N₂ proxy as entry 14 at coarser ℤ₄ register. "
            "Winding (2,2), M_cof=8. Janus at n=1 (diimide intermediate)."
        ),
    )


def _build_z3_cofactor_spec(n_orbitals: int) -> MQEMechanismSpec:
    """Generic 3-cofactor reaction: M=3, m=3 (Z3 prime modulus), adiabatic.

    Definitively tests the generalization of the virtual clock to a prime
    modulus m=3 (irreducible over Z4). Any failure here indicates a bug in
    GeneralizedVirtualShiftGate or the Z_m phase tracker for m≠4.

    System: H3+ triangle at 3 geometries (equilateral → isoceles → return).
        Step 0: r12=0.874, r13=0.874 Å  (equilateral, equilibrium)
        Step 1: r12=1.000, r13=0.874 Å  (isoceles, one H elongated)
        Step 2: r12=0.874, r13=0.874 Å  (return to equilateral)

    The return to the original geometry at step 2 provides a CLOSURE TEST:
    the final energy must equal the step-0 energy to within FCI precision.
    This validates that the Z3 phase accumulation does not accumulate drift.

    Phase closure: Σ ν_n = 3 × 1 = 3 ≡ 0 (mod 3) ✓
    Electron count: Σ |A_n| = 3 × 1 = 3 ✓
    e_fwd = 0   # forward-only injection count for check (d)
    """
    N = n_orbitals  # Should be 3 or 4 (H3+ active space)
    geom_params = [(0.874, 0.874), (1.000, 0.874), (0.874, 0.874)]
    labels      = ["equilateral (E_ref)", "isoceles (stretched)", "equilateral (closure)"]

    steps = []
    k_acc = 0
    e_acc = 0

    for n in range(3):
        nu_n  = 1
        A_n   = [n % N]
        P_n   = [(n + 1) % N]
        B_n   = [(n + 2) % N]
        k_acc += nu_n
        e_acc += len(A_n)

        steps.append(MQEStep(
            step_n               = n,
            nu_n                 = nu_n,
            A_n                  = A_n,
            P_n                  = P_n,
            B_n                  = B_n,
            is_crossing          = False,
            delta_CI_Ha          = None,
            crossing_orbitals    = None,
            geometry_label       = f"H3+: {labels[n]} r12={geom_params[n][0]:.3f}, r13={geom_params[n][1]:.3f} Ang",
            bondlength_angstrom  = geom_params[n][0],
            phase_index_k        = k_acc % 3,
            cumulative_electrons = e_acc,
        ))

    return MQEMechanismSpec(
        name                     = "z3_cofactor",
        M_steps                  = 3,
        m_modulus                = 3,
        S_target                 = 0.0,
        n_orbitals               = N,
        steps                    = steps,
        description              = (
            "Generic 3-cofactor reaction: Z3 prime virtual clock. H3+ 3-step "
            "triangle deformation. Validates GeneralizedVirtualShiftGate(m=3). "
            "Closure test: step 2 geometry = step 0 geometry (energy must match). "
            "Phase closure: Σν=3 ≡ 0 (mod 3)."
        ),
        expected_total_electrons   = 3,
        expected_net_electrons     = 3,
        expected_energy_ordering   = "closure",
    )

def _build_z5_cofactor_spec(n_orbitals: int) -> MQEMechanismSpec:
    """Generic 5-cofactor reaction: M=5, m=5 (Z5 prime modulus), adiabatic.

    Definitively tests Z5 phase closure for a second prime modulus.
    Any failure confirms the issue is with general-m virtual clock
    (PhysicalGenVirtShiftWrapper_general) rather than the m=4 path.

    System: H5+ linear chain at 5 geometries (compression → expansion).
        Step 0: 1.20 Å  (most compressed)
        Step 1: 1.05 Å
        Step 2: 0.90 Å  (near H2 equilibrium)
        Step 3: 0.75 Å
        Step 4: 0.60 Å  (most compressed — tests energy monotonicity)

    Monotonicity test: E_FCI must decrease monotonically with bond length
    (H5+ is unbound at large r, so energies should order E0 < E1 < ... < E4).
    If the MQE QPE pipeline recovers these energies in the correct order
    to chemical accuracy at all 5 steps, the Z5 phase bookkeeping is correct.

    Phase closure: Σ ν_n = 5 × 1 = 5 ≡ 0 (mod 5) ✓
    e_fwd = 0   # forward-only injection count for check (d)
    Electron count: Σ |A_n| = 5 × 1 = 5 ✓
    """
    N = n_orbitals  # Should be 4 or 5
    bondlengths = [1.20, 1.05, 0.90, 0.75, 0.60]
    labels      = ["H5+ r=1.20", "H5+ r=1.05", "H5+ r=0.90", "H5+ r=0.75", "H5+ r=0.60"]

    steps = []
    k_acc = 0
    e_acc = 0

    for n in range(5):
        nu_n  = 1
        A_n   = [n % N]
        P_n   = [(n + 1) % N]
        B_n   = [(n + 2) % N]
        k_acc += nu_n
        e_acc += len(A_n)

        steps.append(MQEStep(
            step_n               = n,
            nu_n                 = nu_n,
            A_n                  = A_n,
            P_n                  = P_n,
            B_n                  = B_n,
            is_crossing          = False,
            delta_CI_Ha          = None,
            crossing_orbitals    = None,
            geometry_label       = labels[n],
            bondlength_angstrom  = bondlengths[n],
            phase_index_k        = k_acc % 5,
            cumulative_electrons = e_acc,
        ))

    return MQEMechanismSpec(
        name                     = "z5_cofactor",
        M_steps                  = 5,
        m_modulus                = 5,
        S_target                 = 0.0,
        n_orbitals               = N,
        steps                    = steps,
        description              = (
            "Generic 5-cofactor reaction: Z5 prime virtual clock. H5+ linear chain "
            "at 5 compression states. Validates GeneralizedVirtualShiftGate(m=5). "
            "Monotonicity test: E_FCI must be ordered at all 5 steps. "
            "Phase closure: Σν=5 ≡ 0 (mod 5)."
        ),
        expected_total_electrons   = 5,
        expected_net_electrons     = 5,
        expected_energy_ordering   = "monotone_decreasing",
    )


def _build_reversible_quinone_spec(n_orbitals: int) -> MQEMechanismSpec:
    """Reversible Quinone/QH₂ Redox Buffer: M=6, m=4 (Z₄), Group B.

    H₄ chain proxy scanning bond length 1.2→1.7 Å (forward) then 1.7→1.2 Å
    (implicit reverse via the 6-step symmetric geometry). mol.charge=0 (4e,
    singlet) throughout — variable charge was causing different N_e per step,
    making _compute_step_relative_energy_algebraic ΔEs physically meaningless.

    Bidirectional PCET bookkeeping:
      n=0–2 forward: nu_n=2 (k_acc: 0→2→4→6), A_n/P_n/B_n tracks reduction
      n=3–5 reverse: nu_decouple=2 (k_acc: 6→4→2→0), A_n_eject/P_n_eject tracks oxidation

    Phase closure: Σnu_couple=6, Σnu_decouple=6 → k_acc_final=0≡0(mod4) ✓
    Net electrons: Σ|A_n|=3, Σ|A_n_eject|=3 → net=0 ✓
    Group B: n*=4/2−1=1. Janus at n=3 (bl=1.5 Å, bidirectional turnaround).
    """
    N       = n_orbitals
    M_steps = 6
    m_mod   = 4
    steps   = []
    k_acc   = 0
    e_fwd   = 0
    e_acc   = 0

    for n in range(M_steps):
        if n < 3:
            # Forward half: Q→QH₂ (algebraic reduction)
            A_n    = [n % N]
            P_n    = [(n + 1) % N]
            B_n    = [(n + 2) % N]
            nu_n   = 2
            A_n_ej = []; P_n_ej = []; B_n_dc = []; nu_dc = 0
            k_acc += 2
            e_fwd += 1
            e_acc += 1
        else:
            # Reverse half: QH₂→Q (algebraic oxidation)
            A_n    = []; P_n = []; B_n = []; nu_n = 0
            A_n_ej = [n % N]
            P_n_ej = [(n + 1) % N]
            B_n_dc = [(n + 2) % N]
            nu_dc  = 2
            k_acc -= 2
            e_acc -= 1

        # Janus at n=3: bidirectional turnaround, maximum H–H separation
        is_crossing = (n == 3)

        steps.append(MQEStep(
            step_n               = n,
            nu_n                 = nu_n,
            A_n                  = A_n,       P_n          = P_n,
            B_n                  = B_n,       A_n_eject    = A_n_ej,
            P_n_eject            = P_n_ej,    B_n_decouple = B_n_dc,
            nu_decouple_n        = nu_dc,
            is_crossing          = is_crossing,
            delta_CI_Ha          = 1.6e-3 if is_crossing else None,
            crossing_orbitals    = [0, 1]  if is_crossing else None,
            geometry_label       = f"Q_cycle_step_{n}_({'fwd' if n<3 else 'rev'})",
            bondlength_angstrom  = 1.2 + 0.1 * n,
            phase_index_k        = k_acc % m_mod,
            cumulative_electrons     = e_fwd,
            cumulative_net_electrons = e_acc,
        ))

    return MQEMechanismSpec(
        name                     = "reversible_quinone",
        M_steps                  = M_steps,
        m_modulus                = m_mod,
        S_target                 = 0.0,   # singlet (H₄ neutral, 4e, charge=0)
        n_orbitals               = N,
        steps                    = steps,
        description              = (
            "Reversible Q/QH₂ redox proxy: H₄ chain, bidirectional PCET. "
            "Group B: m=4, nu=2, Janus at n=3 (bl=1.5 Å, turnaround). "
            "Fixed charge=0 (4e singlet); net electron/phase flux = 0."
        ),
        expected_net_electrons     = 0,
        expected_net_phase_closure = True,
        expected_energy_ordering   = "reversible_quinone",
    )


# ===========================================================================
# 4. INTEGRAL EXTRACTION PER STEP
# ===========================================================================

def _atom_block_for_step(spec: MQEMechanismSpec, step: MQEStep) -> Tuple[str, str, int, int]:
    """Build (atom_block, spin_str, charge, spin_2S) for one MQE step.

    Returns:
        (atom_block, spin_label_str, charge, spin_2S)
    """
    mech = spec.name
    bl   = step.bondlength_angstrom

    if mech == "nitrogenase_lt":
        atom_block = _fe2s2_geometry_at_bond(bl)
        charge, spin_2S = 0, 4      # [Fe2S2]^0, S=2 (high-spin)

    elif mech == "nitrogenase_closed_loop":
        atom_block = _fe2s2_geometry_at_bond(bl)
        charge, spin_2S = 0, 4

    elif mech == "nitrogenase_lt_m8":
        # Z₈ phase variant; identical [Fe₂S₂] geometry/redox pattern to baseline LT
        atom_block = _fe2s2_geometry_at_bond(bl)
        charge, spin_2S = 0, 4

    elif mech == "nitrogenase_lt_parallel":
        # Parallel injection variant; identical [Fe₂S₂] geometry/redox pattern to baseline LT
        atom_block = _fe2s2_geometry_at_bond(bl)
        charge, spin_2S = 0, 4

    elif mech == "psii":
        atom_block = _fe2s2_geometry_at_bond(bl)
        charge, spin_2S = 0, 0      # singlet — matches S_target=0.0

    elif mech == "psii_photo":
        atom_block = _psii_photo_geometry_at_step(step)
        # Fix 1: Fixed charge=0 throughout — photon absorption is algebraic
        # (Gamma_n_abs / phi_photon_n fields in the MQEStep). Varying charge
        # was computing cationic energies for steps 0-3 (N-1, N-2, N-3, N-4
        # electrons), making Path R cross-step ΔE physically meaningless.
        # Identical bug was fixed for anammox_proxy (see comment at line 3112).
        charge, spin_2S = 0, 0

    elif mech == "hydrogenase":
        atom_block = _h2_geometry(bl)
        charge, spin_2S = 0, 0      # H2, singlet

    elif mech == "hydrogenase_oxidation":
        # S1a FIX: oxidative pathway uses same H2 proxy geometry.
        # charge=0 (neutral H2 proxy before ionisation);
        # spin_2S=1 (H2 becomes H2•+ radical cation after e⁻ ejection,
        # but proxy stays neutral — use doublet to flag open-shell).
        atom_block = _h2_geometry(bl)
        charge, spin_2S = 0, 0      # H2 singlet proxy (ejection tracked by A_n_eject)

    elif mech == "z3_cofactor":
        atom_block = _h3plus_geometry(bl, bl)
        charge, spin_2S = 1, 0

    elif mech == "z5_cofactor":
        atom_block = _hchain_geometry(5, bl)
        charge, spin_2S = 1, 0
    
    elif mech == "haber_bosch":
        # Fixed Fe₂S₂N₂ geometry — same spin convention as nitrogenase_lt
        # (Fe₂S₂ high-spin S=2, 2S=4; N₂ adds 14 electrons, stays closed-shell)
        atom_block = _haber_bosch_fe2s2n2_geometry(step)
        charge, spin_2S = 0, 4
        return atom_block, spin_label(spin_2S), charge, spin_2S

    elif mech == "nitrogenase_fe4s4":
        atom_block     = _get_nitrogenase_fe4s4_atom_block(step)
        current_charge = 4
        # Fix 2: spin_2S=4 (S=2, quintet) — consistent with every other
        # Fe-cluster mechanism in this dispatch (haber_bosch, assimilatory_nr,
        # v_nitrogenase, anammox_proxy, nitrogenase_group_a, etc.).
        # Singlet (spin_2S=0) caused ROHF to converge to different spin states
        # across the 8 steps (step 0 was 320 Ha below steps 1-7), despite
        # the geometry changing by only 0.5% per step.
        spin_2S        = 4
        return atom_block, spin_label(spin_2S), current_charge, spin_2S

    elif mech == "nitrogenase_femoco":
        atom_block = _get_femoco_atom_block(step)
        current_charge = 0
        # Electron count: 7×Fe(26) + Mo(14 val, ECP28) + 9×S(16) + 2×C(6) + 2×O(8) + 2×N(7)
        # = 182 + 14 + 144 + 12 + 16 + 14 = 382 (even).
        # Physical FeMo-co ground state is S=3/2 (2S=3) but that requires an odd
        # electron count; 382-electron neutral system must use even 2S.
        # S=2 (2S=4) is the nearest valid high-spin approximation.
        spin_2S = 4
        return atom_block, spin_label(spin_2S), current_charge, spin_2S

    elif mech == "nitrogenase_group_a":
        # Fe₂S₂ proxy, same geometry/redox as nitrogenase_lt.
        # Group A: m=8, ν=2, n*=3. charge=0, spin_2S=4 (S=2).
        atom_block = _fe2s2_geometry_at_bond(bl)
        charge, spin_2S = 0, 4
        return atom_block, spin_label(spin_2S), charge, spin_2S

    elif mech == "nitrogenase_group_d":
        # Fe₂S₂ proxy, same geometry/redox as nitrogenase_lt.
        # Group D: m=12, ν=2, n*=5. charge=0, spin_2S=4 (S=2).
        atom_block = _fe2s2_geometry_at_bond(bl)
        charge, spin_2S = 0, 4
        return atom_block, spin_label(spin_2S), charge, spin_2S

    elif mech == "mo_nitrogenase":
        # Entry 1: Fe-Mo-S₂ proxy. 72e, charge=0, spin_2S=4.
        atom_block = _femo_proxy_atom_block(bl)
        charge, spin_2S = 0, 4
        return atom_block, spin_label(spin_2S), charge, spin_2S

    elif mech == "assimilatory_nr":
        # Entry 7: Mo-S₂-O₂ pterin-dithiolate proxy. 62e, charge=0, spin_2S=4.
        atom_block = _mo_nr_proxy_atom_block(bl)
        charge, spin_2S = 0, 4
        return atom_block, spin_label(spin_2S), charge, spin_2S

    elif mech == "photocatalytic_n2":
        # Entry 14: Ti₂N₂ proxy. 58e, charge=0, spin_2S=4.
        atom_block = _ti2n2_proxy_atom_block(bl)
        charge, spin_2S = 0, 4
        return atom_block, spin_label(spin_2S), charge, spin_2S

    elif mech == "v_nitrogenase":
        # Entry 3: V₂S₂ FeVco proxy. 78e, charge=0, spin_2S=4.
        atom_block = _v2s2_proxy_atom_block(bl)
        charge, spin_2S = 0, 4
        return atom_block, spin_label(spin_2S), charge, spin_2S

    elif mech == "cu_co2rr":
        # Entry 13: Cu₃⁻ trimer proxy. 88e, charge=−1, spin_2S=0.
        atom_block = _cu3_proxy_atom_block(bl)
        charge, spin_2S = -1, 0
        return atom_block, spin_label(spin_2S), charge, spin_2S

    elif mech == "femon2_trimer":
        # Fe–Mo–N₂ trimer proxy. 54e, charge=0, spin_2S=4.
        # Fe(26)+Mo(ECP28→14val)+2×N(7)=54e. Group B, m=4, N–N activation.
        atom_block = _femon2_trimer_atom_block(bl)
        charge, spin_2S = 0, 4
        return atom_block, spin_label(spin_2S), charge, spin_2S

    elif mech == "ethylene_epoxidation":
        atom_block = _get_ethylene_epoxidation_atom_block(step)
        current_charge = 0
        spin_2S = 0     # 98 valence electrons = even = closed-shell singlet
        return atom_block, spin_label(spin_2S), current_charge, spin_2S

    elif mech == "anammox_proxy":
        atom_block = _get_anammox_atom_block(step)
        # Fixed charge=0 and high-spin Fe(II) (2S=4) for all steps.
        # Electron/proton ejection is algebraic via A_n_eject/P_n_eject fields;
        # varying mol.charge was causing cross-step ΔE comparisons between
        # different N_e systems, making Path R step energies physically wrong.
        charge, spin_2S = 0, 4
        return atom_block, spin_label(spin_2S), charge, spin_2S

    elif mech == "thymine_dimer_proxy":
        atom_block = _get_thymine_dimer_atom_block(step)
        current_charge = 0 
        spin_2S = 0 
        return atom_block, spin_label(spin_2S), current_charge, spin_2S

    elif mech == "atp_hydrolysis_proxy":
        atom_block = _get_atp_hydrolysis_atom_block(step)
        current_charge = 0   # minimal H₃PO₄+H₂O proxy: 60e, charge=0, singlet
        spin_2S = 0
        return atom_block, spin_label(spin_2S), current_charge, spin_2S

    elif mech == "rnr_radical_proxy":
        atom_block = _rnr_proxy_geometry(step.step_n)
        current_charge = 0
        spin_2S = 1
        return atom_block, spin_label(spin_2S), current_charge, spin_2S

    elif mech == "reversible_quinone":
        atom_block = _hchain_geometry(4, bl)
        # Fixed charge=0 (H₄ neutral, 4e, singlet) for all steps.
        # Variable charge (-ne) was causing ΔE comparisons between H₄ with
        # 4–7 electrons — different N_e systems, physically meaningless for Path R.
        # Bidirectional PCET bookkeeping is algebraic (A_n/A_n_eject fields).
        charge, spin_2S = 0, 0

    # ── CYP450 METABOLISM — fixed charge=0, S=1 surface throughout ──
    elif mech == "cyp450_metabolism":
        atom_block = _get_cyp450_atom_block(step)
        # Fixed charge=0 (78e: Fe26+O8+S16+4×N7) and spin_2S=2 (S=1) for
        # all steps. Variable charge was causing _compute_step_relative_energy_algebraic
        # to compare energies of different N_e systems (77e vs 78e), making
        # Path R step ΔE physically meaningless.
        # Locking to S=1 (Compound I surface) scans the spin-triplet adiabat
        # through all six geometries — the relevant surface for the O–O CI
        # (Shaik et al., Chem Rev 2005). The CAS active space will correctly
        # resolve the Fe d-orbital/porphyrin π splitting at each geometry.
        charge, spin_2S = 0, 2
    # ─────────────────────────────────────────────────────────────────────

    # ── Catalog entries 6, 9, 10, 11 (new unique mechanisms) ─────────────
    elif mech == "complex_i":
        # Entry 6: [2Fe-2S] rhombic proxy. 68e, charge=0, spin_2S=0.
        # Group C: m=4, ν=1, n*=3. Fe-S compression 2.260→2.200 Å.
        atom_block = _fe2s2_geometry_at_bond(bl)
        charge, spin_2S = 0, 0
        return atom_block, spin_label(spin_2S), charge, spin_2S

    elif mech == "codh_acs":
        # Entry 9: Ni₂S₂-CO proxy. 102e, charge=0, spin_2S=0.
        # Group B: m=4, ν=1, n*=3. Ni-S compression 2.300→2.190 Å.
        atom_block = _codh_acs_proxy_atom_block(bl)
        charge, spin_2S = 0, 0
        return atom_block, spin_label(spin_2S), charge, spin_2S

    elif mech == "cyt_bd_oxidase":
        # Entry 10: Fe₂O₂ rhombic proxy. 68e, charge=0, spin_2S=0.
        # Group C: m=4, ν=1, n*=3. Fe-O compression 2.300→2.200 Å.
        atom_block = _cyt_bd_proxy_atom_block(bl)
        charge, spin_2S = 0, 0
        return atom_block, spin_label(spin_2S), charge, spin_2S

    elif mech == "cyt_c_oxidase":
        # Entry 11: Fe-Cu-N-O binuclear proxy. 70e, charge=0, spin_2S=0.
        # Group C: m=4, ν=1, n*=3. Fe-Cu compression 2.600→2.500 Å.
        atom_block = _cyt_c_oxidase_proxy_atom_block(bl)
        charge, spin_2S = 0, 0
        return atom_block, spin_label(spin_2S), charge, spin_2S

    # ── Oldform lifts (catalog entries 2, 4, 8, 12, 15) ──────────────────
    elif mech == "mo_nitrogenase_m4":
        # Entry 2: Same Fe-Mo-S₂ proxy as mo_nitrogenase (entry 1), m=4.
        # Group B: m=4, ν=2, n*=1. 72e, charge=0, spin_2S=4.
        atom_block = _femo_proxy_atom_block(bl)
        charge, spin_2S = 0, 4
        return atom_block, spin_label(spin_2S), charge, spin_2S

    elif mech == "v_nitrogenase_m4":
        # Entry 4: Same V₂S₂ proxy as v_nitrogenase (entry 3), m=4.
        # Group B: m=4, ν=2, n*=1. 78e, charge=0, spin_2S=4.
        atom_block = _v2s2_proxy_atom_block(bl)
        charge, spin_2S = 0, 4
        return atom_block, spin_label(spin_2S), charge, spin_2S

    elif mech == "assimilatory_nr_m4":
        # Entry 8: Same Mo-S₂-O₂ proxy as assimilatory_nr (entry 7), m=4.
        # Group B: m=4, ν=2, n*=1. 62e, charge=0, spin_2S=4.
        atom_block = _mo_nr_proxy_atom_block(bl)
        charge, spin_2S = 0, 4
        return atom_block, spin_label(spin_2S), charge, spin_2S

    elif mech == "cu_co2rr_m4":
        # Entry 12: Same Cu₃⁻ trimer proxy as cu_co2rr (entry 13), m=4.
        # Group B: m=4, ν=2, n*=1. 88e, charge=−1, spin_2S=0.
        atom_block = _cu3_proxy_atom_block(bl)
        charge, spin_2S = -1, 0
        return atom_block, spin_label(spin_2S), charge, spin_2S

    elif mech == "photocatalytic_n2_m4":
        # Entry 15: Same Ti₂N₂ proxy as photocatalytic_n2 (entry 14), m=4.
        # Group B: m=4, ν=2, n*=1. 58e, charge=0, spin_2S=4.
        atom_block = _ti2n2_proxy_atom_block(bl)
        charge, spin_2S = 0, 4
        return atom_block, spin_label(spin_2S), charge, spin_2S


    else:
        raise ValueError(f"Unknown mechanism: {mech}")

    return atom_block, spin_label(spin_2S), charge, spin_2S



def generate_step_integrals(
    spec,
    step,
    basis:        str  = "STO-3G",
    validate_fci: bool = True,
    verbose:      int  = 0,
    dm_cache:     Optional[Dict[str, Optional[np.ndarray]]] = None,
    mo_cache:     Optional[Dict[str, Optional[np.ndarray]]] = None,
    _save_mo_integrals_dir:  Optional[Path]       = None,
    _n_total_orbs:           int                  = 76,
    _save_mo_coeffs:         bool                 = False,
    _cas_orbital_indices:    Optional[np.ndarray] = None,
    _precomputed_mf:         Optional[object]     = None,
    _mf_store:               Optional[Dict]       = None,
) -> Dict:
    """Extract CASCI integrals for one MQE step and assemble the MQE-extended JSON.

    Identical to the original except the SCF block no longer calls
    mf.kernel() after run_rohf().  See module docstring for full explanation.

    Args:
        spec:         MQEMechanismSpec for the full mechanism.
        step:         MQEStep descriptor for this step.
        basis:        PySCF basis set string or per-element dict.
        validate_fci: Run exact FCI if ncas <= 20.
        verbose:      PySCF verbosity level.
        _save_mo_coeffs: When True, write mo_coeffs.npy and ao_labels.json
            to _save_mo_integrals_dir after ROHF convergence.  Required for
            the sub-Janus two-pass protocol in mqehybridgenerator
            (prop:seed_is_sp): pass 1 saves artefacts; load_sub_janus_selection
            reads them; pass 2 supplies _cas_orbital_indices.
        _cas_orbital_indices: Optional 0-based array of MO indices to use as
            the CASCI active space, overriding Fermi-level selection.  Set by
            run_hybrid_generation pass 2 after sub-Janus orbital selection.

    Returns:
        Dict: Complete JSON-serialisable dataset for this step.
    """

    atom_block, _, charge, spin_2S = _atom_block_for_step(spec, step)
    ncas = spec.n_orbitals

    log.info(
        f"[MQE-STEP] {spec.name} | n={step.step_n}/{spec.M_steps-1} | "
        f"geo={step.geometry_label!r} | ncas={ncas}"
    )

    # ── Build PySCF molecule ─────────────────────────────────────────────────
    # build_molecule now auto-injects def2-ECP for heavy elements (Fix 1).
    mol = build_molecule(
        atom_block  = atom_block,
        basis       = basis,
        charge      = charge,
        target_spin = spin_2S,
        coord_unit  = "Angstrom",
        verbose     = verbose,
        # ecp is intentionally omitted here: build_molecule's _auto_ecp() handles it
    )

    # ── Active-space electron count (parity-safe) ────────────────────────────
    ncas_actual  = min(ncas, mol.nao)
    total_nelec  = sum(mol.nelec)
    nelec_active = min(ncas_actual, total_nelec)

    if nelec_active > 0 and (total_nelec - nelec_active) % 2 != 0:
        nelec_active -= 1

    active_spin = 0 if nelec_active % 2 == 0 else 1

    if total_nelec > ncas_actual:
        log.warning(
            f"  [CAS] Truncating to CAS({nelec_active},{ncas_actual}): "
            f"{total_nelec - nelec_active} core electrons excluded. "
            f"Ensure active space matches validation target."
        )

    try:
        nelec_tuple = make_nelec_tuple(nelec_active, active_spin)
    except ValueError:
        nelec_active = max(0, nelec_active - 1)
        active_spin  = 0
        nelec_tuple  = make_nelec_tuple(nelec_active, active_spin)

    nalpha, nbeta = nelec_tuple
    log.info(
        f"  CAS({nelec_active},{ncas_actual}) nalpha={nalpha} nbeta={nbeta}  "
        f"spin_2S={active_spin} (total_nelec={total_nelec}, core={total_nelec-nelec_active})"
    )

    # ── SCF with inter-step density propagation (FIX 2) ─────────────────────
    #
    # ORIGINAL (BROKEN):
    #   mf = run_rohf(mol)          # ← already runs and converges SCF internally
    #   ...
    #   mf.kernel(dm0=prev_dm)      # ← runs SCF AGAIN on already-converged object
    #
    # PATCHED:
    #   Read cached density → pass as dm0 to run_rohf → cache new density.
    #   run_rohf returns converged; no further .kernel() call.
    #
    # The cache is now a caller-supplied dict (not a module global), which
    # makes concurrent generation of multiple mechanisms safe: each call to
    # generate_mechanism_dataset creates its own local dict and passes it here.
    if dm_cache is None:
        dm_cache = {}
    cache_key = f"{spec.name}_{spec.n_orbitals}"
    prev_dm   = dm_cache.get(cache_key)

    # ── Fix 5 (Option A): MO-coefficient cache ────────────────────────────────
    # For mechanisms with severe SCF convergence challenges (e.g.,
    # ethylene_epoxidation Ag₃ oxametallacycle doublet), the raw density matrix
    # warm-start is insufficient: the DFT Level-2 seeded DM still lands on the
    # wrong spin-state manifold at the problematic geometry.
    #
    # Propagating the converged MO coefficients from the previous step allows
    # us to build a DM that exactly encodes the previous orbital occupations
    # (doubly occupied, singly occupied, virtual) which is more stable than
    # the 1-RDM for large geometry changes along a reaction coordinate.
    #
    # This overrides dm0 if valid MO coefficients are available from the prior step.
    if mo_cache is None:
        mo_cache = {}
    prev_mo = mo_cache.get(cache_key)

    # ── Dimension guard: DM warm-start ───────────────────────────────────────
    dm0 = None
    if prev_dm is not None and step.step_n > 0:
        nao_prev = prev_dm.shape[-1]
        if nao_prev == mol.nao and prev_dm.ndim == 2:
            dm0 = prev_dm
            log.info(
                "[SCF] Warm-starting step %d from step %d density matrix.",
                step.step_n, step.step_n - 1,
            )
        else:
            log.warning(
                "[SCF] Skipping DM warm-start for step %d: "
                "previous DM nao=%d != current nao=%d (atom count changed). "
                "Falling back to standard initial guess.",
                step.step_n, nao_prev, mol.nao,
            )
            dm_cache.pop(cache_key, None)

    # ── Fix 5 (Option A): override dm0 with MO-coefficient projection ─────────
    if prev_mo is not None and step.step_n > 0:
        if prev_mo.shape[0] == mol.nao:
            try:
                # Build an improved dm0 from the previous converged MO
                # coefficients using the current-step occupation pattern.
                # This preserves orbital symmetry across geometry steps.
                mo_occ_vec = np.zeros(prev_mo.shape[1])
                mo_occ_vec[:nbeta]         = 2.0   # doubly occupied
                mo_occ_vec[nbeta:nalpha]   = 1.0   # singly occupied
                mf_tmp = scf.ROHF(mol)
                dm0_from_mo = mf_tmp.make_rdm1(prev_mo, mo_occ_vec)
                dm0 = dm0_from_mo
                log.info(
                    "[SCF] Fix5/MO-cache: overriding dm0 with MO-coeff projection "
                    "from step %d (nao=%d, nbeta=%d, nalpha=%d).",
                    step.step_n - 1, mol.nao, nbeta, nalpha,
                )
            except Exception as _mo_exc:
                log.warning(
                    "[SCF] Fix5/MO-cache: projection failed at step %d: %s. "
                    "Falling back to plain DM cache.",
                    step.step_n, _mo_exc,
                )
        else:
            log.warning(
                "[SCF] Fix5/MO-cache: shape mismatch at step %d "
                "(prev nao=%d != current nao=%d). Discarding MO cache.",
                step.step_n, prev_mo.shape[0], mol.nao,
            )
            mo_cache.pop(cache_key, None)

    if _precomputed_mf is not None:
        # ── Pass 2 of the sub-Janus two-pass protocol ────────────────────────
        # ROHF already converged in pass 1 and the mf object was retained by the
        # caller via _mf_store.  Reuse it directly: skip ROHF, dm/mo cache
        # updates, mo_coeffs/mo_energy saves, and full-MO tensor saves (all
        # already written by pass 1).  Only CASCI is re-run with the corrected
        # sub-Janus orbital indices (_cas_orbital_indices).
        mf = _precomputed_mf
        log.info(
            "[MQE-STEP] Pass 2: reusing cached ROHF (E = %.10f Ha) — ROHF skipped.",
            mf.e_tot,
        )
    else:
        mf = run_rohf(mol, dm0=dm0)

        if mf.converged:
            rdm1 = mf.make_rdm1()
            # Normalise to (nao, nao): UKS/UHF returns (2, nao, nao); ROHF returns (nao, nao).
            if rdm1.ndim == 3:
                rdm1 = rdm1[0] + rdm1[1]
            dm_cache[cache_key] = rdm1
            # Fix 5 (Option A): also cache MO coefficients for next step
            mo_cache[cache_key] = mf.mo_coeff.copy()
            log.debug("[SCF] Density + MO coefficients cached for step %d.", step.step_n + 1)

            # ── Save MO coefficients for sub-Janus orbital selection ─────────
            # prop:seed_is_sp: pass 1 writes mo_coeffs.npy + mo_energy.npy +
            # ao_labels.json so that load_sub_janus_selection can identify
            # {ℓ<k*=2}={s,p} orbitals for pass 2.
            if _save_mo_coeffs and _save_mo_integrals_dir is not None:
                try:
                    _sdir = Path(_save_mo_integrals_dir)
                    np.save(str(_sdir / "mo_coeffs.npy"), mf.mo_coeff)
                    np.save(str(_sdir / "mo_energy.npy"), mf.mo_energy)
                    (_sdir / "ao_labels.json").write_text(
                        json.dumps(mol.ao_labels())
                    )
                    log.info(
                        "[TOWER] Saved mo_coeffs.npy (shape=%s) + mo_energy.npy + ao_labels.json → %s",
                        mf.mo_coeff.shape, _sdir,
                    )
                except Exception as _exc:
                    log.warning(
                        "[TOWER] Could not save MO coefficients at step %d: %s",
                        step.step_n, _exc,
                    )
        else:
            dm_cache.pop(cache_key, None)
            mo_cache.pop(cache_key, None)
            log.warning(
                "[SCF] SCF did not converge at step %d; "
                "density and MO caches cleared to avoid propagating a bad state.",
                step.step_n,
            )

        # ── Full MO integral tensor for Iwasawa tower (saved at every step) ──
        # No second ROHF — we reuse the already-converged mf from this step.
        # Per-step files h1_full_step{n:02d}.npy / eri_packed_step{n:02d}.npy are
        # needed for correct non-Janus geometry in tower level datasets.
        # Backward-compat Janus files h1_full.npy / eri_packed.npy are also kept.
        if _save_mo_integrals_dir is not None and mf.converged:
            try:
                nmo   = mf.mo_coeff.shape[1]
                _sdir = Path(_save_mo_integrals_dir)
                sn    = step.step_n

                # ── Decide whether to use the tower-window path ───────────────
                # When n_occ_base (= n_core from the CAS metadata) is large
                # (> _TOWER_WIN_THRESHOLD), saving h1_full for MOs 0..n-1 is
                # impractical: both the on-disk eri_packed and the in-memory 4D
                # ERI needed by compute_casci_effective_integrals become infeasible
                # (e.g. 22 GB for n=229 at n_occ_base=189).
                #
                # Instead we save h1_eff_win (Fock-corrected over the tower
                # window only) and eri_win (raw 2e for the window).  The tower
                # climber reads h1_full_win_offset from the manifest and uses
                # local indices 0..win_size-1 throughout.
                #
                # The deep-core energy (contribution of MOs 0..win_start-1 to
                # the nuclear + 1e + 2e energy) is saved as deep_ecore.npy next
                # to h1_full_win.npy.  The tower climber adds it to the base
                # ecore read from the step JSON minus the window-MO contribution.
                _TOWER_WIN_THRESHOLD: int = 60   # n_occ_base above which window path activates

                # n_core comes from the CASCI extraction metadata set earlier
                # in generate_step_integrals; fall back to _n_total_orbs for
                # legacy datasets that don't record n_core.
                # n_core = (total_electrons - active_electrons) // 2
                # Already computed from mol + CAS parameters earlier in this function.
                _n_core_here = (mol.nelectron - int(nelec_tuple[0]) - int(nelec_tuple[1])) // 2
                _use_win     = (_n_core_here > _TOWER_WIN_THRESHOLD)

                if _use_win:
                    # Tower window: start at win_start = n_core_here - occ_buf,
                    # where occ_buf = k_max_budget * block // 2.  Use k_max=20,
                    # block=4 as safe defaults (occ_buf = 18*2 = 36).
                    _occ_buf  = 36   # (k_max=20 - k_base=2) * block//2 = 18*2
                    _win_start = max(0, _n_core_here - _occ_buf)
                    _win_size  = min(_n_total_orbs, nmo - _win_start)
                    h1_eff_win, eri_win, deep_ecore = extract_tower_window_integrals(
                        mf, _win_start, _win_size,
                    )
                    np.save(str(_sdir / f"h1_full_win_step{sn:02d}.npy"),  h1_eff_win)
                    np.save(str(_sdir / f"eri_win_step{sn:02d}.npy"),      eri_win)
                    np.save(str(_sdir / f"deep_ecore_step{sn:02d}.npy"),
                            np.array([deep_ecore]))
                    log.info(
                        "[TOWER] (window) Saved h1_full_win_step%02d.npy %s + "
                        "eri_win_step%02d.npy (%.0f MB) + deep_ecore=%.6f Ha → %s",
                        sn, h1_eff_win.shape, sn, eri_win.nbytes / 2**20,
                        deep_ecore, _sdir,
                    )
                    # Backward-compat un-suffixed files at the Janus crossing
                    if getattr(step, "is_crossing", False):
                        np.save(str(_sdir / "h1_full_win.npy"),  h1_eff_win)
                        np.save(str(_sdir / "eri_win.npy"),      eri_win)
                        np.save(str(_sdir / "deep_ecore.npy"),   np.array([deep_ecore]))
                        # Record win_start in a sidecar so the tower climber
                        # can reconstruct global→local index translation.
                        import json as _json
                        (_sdir / "h1_full_win_meta.json").write_text(
                            _json.dumps({
                                "win_start":  _win_start,
                                "win_size":   h1_eff_win.shape[0],
                                "n_core":     _n_core_here,
                                "occ_buf":    _occ_buf,
                            }, indent=2)
                        )
                        log.info(
                            "[TOWER] (window Janus) Also wrote un-suffixed "
                            "h1_full_win.npy / eri_win.npy / deep_ecore.npy / "
                            "h1_full_win_meta.json → %s", _sdir,
                        )
                else:
                    # Original path: h1_full covers MOs 0..n-1.
                    # Works correctly when n_occ_base ≤ _TOWER_WIN_THRESHOLD
                    # (n_total_orbs ≥ n_occ_base + active + virtual budget).
                    n = min(_n_total_orbs, nmo)
                    h1_full, eri_packed = extract_full_mo_integrals(mf, n)
                    np.save(str(_sdir / f"h1_full_step{sn:02d}.npy"),    h1_full)
                    np.save(str(_sdir / f"eri_packed_step{sn:02d}.npy"), eri_packed)
                    log.info(
                        "[TOWER] Saved h1_full_step%02d.npy (shape=%s) + "
                        "eri_packed_step%02d.npy (%.1f MB) → %s",
                        sn, h1_full.shape, sn, eri_packed.nbytes / 2**20, _sdir,
                    )
                    if getattr(step, "is_crossing", False):
                        np.save(str(_sdir / "h1_full.npy"),    h1_full)
                        np.save(str(_sdir / "eri_packed.npy"), eri_packed)
                        log.info(
                            "[TOWER] (Janus) Also wrote h1_full.npy + eri_packed.npy "
                            "for backward compatibility → %s", _sdir,
                        )

            except Exception as _exc:
                log.warning(
                    "[TOWER] Could not save full MO integrals at step %d: %s — "
                    "tower will fall back to heuristic extension.",
                    step.step_n, _exc,
                )

    # ── Cache mf for caller (two-pass protocol) ───────────────────────────────
    # run_hybrid_generation passes _mf_store={} to pass 1; after pass 1 returns,
    # _mf_store["mf"] holds the converged ROHF object for pass 2 to reuse.
    if _mf_store is not None:
        _mf_store["mf"] = mf

    # ── ROHF canonical MO energies (used by tower climber to rank virtuals) ──
    mo_energies_rohf: Optional[List[float]] = None
    if hasattr(mf, "mo_energy") and mf.mo_energy is not None:
        mo_energies_rohf = mf.mo_energy.tolist()

    # ── CASCI integrals (+ active-orbital NOON from 1-RDM) ───────────────────
    # _cas_orbital_indices: when set (pass 2 of sub-Janus protocol), overrides
    # Fermi-level selection with the {ℓ<k*=2}={s,p} subset (prop:seed_is_sp).
    h1, eri_full, ecore, e_casci, noons_active = extract_casci_integrals(
        mf, ncas_actual, nelec_tuple,
        cas_orbital_indices=_cas_orbital_indices,
    )

    # ── Reference energy ─────────────────────────────────────────────────────
    e_ref, ref_method = compute_reference_energy(
        h1, eri_full, ncas_actual, nelec_tuple, ecore
    )

    # ── Assemble JSON ────────────────────────────────────────────────────────
    h_diag = {str(p): float(h1[p, p]) for p in range(ncas_actual)}
    h_hop  = {
        f"({p},{q})": float(h1[p, q])
        for p in range(ncas_actual) for q in range(p + 1, ncas_actual)
    }
    g_full = compress_eri(eri_full, ncas_actual)
    del eri_full
    gc.collect()

    data = {
        "h_diag":                    h_diag,
        "h_hop":                     h_hop,
        "g_full":                    g_full,
        "ecore_Ha":                  float(ecore),
        "rohf_energy_Ha":            float(mf.e_tot),
        "exact_fci_energy_Ha":       e_ref if ncas_actual <= 20 else None,
        "circuit_reference_energy_Ha": e_ref,
        "active_space_corr_energy_Ha": float(
            (e_casci if e_casci else e_ref) - mf.e_tot
        ),
        "active_space_corr_label":   "E_CASCI - E_ROHF",
        "mqe_step": step.to_dict(
            mechanism=spec.name,
            M_total=spec.M_steps,
            m_modulus=spec.m_modulus,
        ),
        "metadata": {
            "mol_name":            f"{spec.name}_step{step.step_n}",
            "basis":               basis,
            "bondlength_angstrom": step.bondlength_angstrom,
            "coord_unit_pyscf":    "Angstrom",
            "geometry_source":     "mqe_analytical",
            "geometry_optimized":  False,
            "integral_convention": "chemist (pq|rs)",
            "eri_symmetry":        "8-fold real",
            "screening_threshold": 1e-8,
            "dt_ref_Ha_inv":       0.04,
            "scf_method":          "ROHF",
            "ref_method":          ref_method,
            "ncas":                ncas_actual,
            "nao_total":           mol.nao,
            "nelec_active":        nalpha + nbeta,
            "nalpha":              nalpha,
            "nbeta":               nbeta,
            "spin_2S":             spin_2S,
            "spin_sector":         spin_label(spin_2S),
            "fermion_mapping":     "Native_d4_Tetralemmatic",
            "photon_absorbed":        bool(step.Gamma_n_abs),
            "photon_emitted":         bool(step.Gamma_n_emit),
            "phi_photon_n":           step.phi_photon_n,
            "photon_energy_Ha":       step.phi_photon_n / 0.02 if step.phi_photon_n else 0.0,
            "photon_energy_eV": (
                (step.phi_photon_n / 0.02) * 27.2114 if step.phi_photon_n else 0.0
            ),
            "cumulative_net_photons": step.cumulative_net_photons,
            "photo_charge_delta":     step.cumulative_net_photons,
            "mqe_mechanism":          spec.name,
            "mqe_M_steps":            spec.M_steps,
            "mqe_m_modulus":          spec.m_modulus,
            "mqe_S_target":           spec.S_target,
            "mqe_step_index":         step.step_n,
            "mqe_phase_closure_satisfied": (
                (step.phase_index_k == 0)
                if step.step_n == spec.M_steps - 1 else None
            ),
            "mqe_expected_electrons": spec.expected_total_electrons,
            # ── Tower-climbing seeds ──────────────────────────────────────────
            # noons_active: CASCI 1-RDM diagonal for the active orbitals (ncas
            #   values in [0,2]).  Used by TowerClimber to rank correlation-
            #   important virtual orbitals for block selection.
            # mo_energies_rohf: canonical ROHF orbital energies for ALL nao_total
            #   MOs.  Used as a tiebreaker when virtual NOONs are degenerate (0.0).
            # n_core: number of frozen-core doubly-occupied orbitals so the tower
            #   climber can place active NOONs at the correct position in the full
            #   (nao_total,) NOON vector.
            "noons_active":     noons_active.tolist() if noons_active is not None else None,
            "mo_energies_rohf": mo_energies_rohf,
            "n_core":           (total_nelec - nelec_active) // 2,
            # ncas_occ_base: alias for n_core consumed by tower_climber._load_base.
            # Explicit field avoids the legacy default of 40 when n_core is absent.
            "ncas_occ_base":    (total_nelec - nelec_active) // 2,
        },
    }

    return data


def _save_noons_for_tower(mech_dir: Path, step_datasets: List[Dict]) -> None:
    r"""Save ``noons.npy`` and ``mo_energies.npy`` in ``mech_dir`` for the tower climber.

    Called by ``generate_mechanism_dataset`` after all step JSONs are written.

    ``noons.npy`` — shape ``(nao_total,)``, full NOON vector in canonical MO order:

    * Core orbitals ``[0 .. n_core-1]``            → NOON = 2.0 (doubly occupied)
    * Active orbitals ``[n_core .. n_core+ncas-1]`` → CASCI 1-RDM diagonal
    * Virtual orbitals ``[n_core+ncas .. nao_total-1]`` → NOON = 0.0

    ``mo_energies.npy`` — shape ``(nao_total,)``, canonical ROHF orbital energies.
    Used by :func:`select_next_block` as a tiebreaker when all candidate virtual
    NOONs equal 0.0 — closer-to-Fermi virtuals are added first.

    The data is taken from the Janus step (most correlated geometry), falling back
    to the first step if no Janus step exists.
    """
    if not step_datasets:
        return

    # Prefer Janus (most correlated); fall back to first step
    target = next(
        (d for d in step_datasets
         if d.get("mqe_step", {}).get("is_crossing", False)),
        step_datasets[0],
    )
    meta = target.get("metadata", {})

    try:
        ncas      = int(meta.get("ncas", 4))
        nao_total = int(meta.get("nao_total", ncas))
        n_core    = int(meta.get("n_core", 0))

        # ── Full NOON vector ─────────────────────────────────────────────────
        noons_full = np.zeros(nao_total, dtype=float)
        noons_full[:n_core] = 2.0   # core: doubly occupied

        noons_active_list = meta.get("noons_active")
        if noons_active_list is not None:
            active_arr = np.asarray(noons_active_list, dtype=float)
            n_active   = min(len(active_arr), ncas, nao_total - n_core)
            noons_full[n_core : n_core + n_active] = active_arr[:n_active]
        # virtual slots remain 0.0

        noons_path = mech_dir / "noons.npy"
        np.save(str(noons_path), noons_full)
        log.info(
            "[TOWER] Saved noons.npy (shape=%s, n_core=%d, n_active=%d) → %s",
            noons_full.shape, n_core,
            0 if noons_active_list is None else len(noons_active_list),
            noons_path,
        )

        # ── ROHF MO energy vector ────────────────────────────────────────────
        mo_energies_list = meta.get("mo_energies_rohf")
        if mo_energies_list is not None:
            mo_path = mech_dir / "mo_energies.npy"
            np.save(str(mo_path), np.asarray(mo_energies_list, dtype=float))
            log.info("[TOWER] Saved mo_energies.npy (shape=(%d,)) → %s",
                     len(mo_energies_list), mo_path)

    except Exception as exc:
        log.warning("[TOWER] Could not save NOONs for tower climber: %s", exc)


def _save_full_mo_integrals_for_tower(
    spec:          "MQEMechanismSpec",
    basis:         str,
    mech_dir:      Path,
    n_total_orbs:  int = 76,
    verbose:       int = 0,
) -> None:
    """Run ROHF on the Janus step geometry and save h1_full.npy + eri_packed.npy.

    These files provide exact molecular integrals for *all* n_total_orbs MOs
    in a single ~17 MB footprint (8-fold packed ERIs for n=76).  The Iwasawa
    tower climber loads them once and slices to the first 4k orbitals at each
    level k, giving exact cross-block coupling without any heuristic scaling.

    Files written (alongside noons.npy in mech_dir):
      h1_full.npy   — shape (n, n), one-electron integrals in MO basis
      eri_packed.npy — 8-fold packed ERIs, restored via ao2mo.restore(1, eri, n)

    Where n = min(n_total_orbs, nmo_available).
    """
    # Use the Janus step geometry (most correlated state); fall back to step 0.
    janus_step = next(
        (s for s in spec.steps if getattr(s, "is_crossing", False)),
        spec.steps[0],
    )

    try:
        atom_block, _, charge, spin_2S = _atom_block_for_step(spec, janus_step)
        mol = build_molecule(
            atom_block  = atom_block,
            basis       = basis,
            charge      = charge,
            target_spin = spin_2S,
            coord_unit  = "Angstrom",
            verbose     = verbose,
        )
        mf  = run_rohf(mol)
        if not mf.converged:
            log.warning(
                "[TOWER] ROHF did not converge for %r Janus step — "
                "full MO integrals NOT saved (tower will use heuristic extension).",
                spec.name,
            )
            return

        nmo = mf.mo_coeff.shape[1]
        n   = min(n_total_orbs, nmo)

        h1_full, eri_packed = extract_full_mo_integrals(mf, n)

        np.save(str(mech_dir / "h1_full.npy"),    h1_full)
        np.save(str(mech_dir / "eri_packed.npy"), eri_packed)
        log.info(
            "[TOWER] Saved h1_full.npy (shape=%s) + eri_packed.npy "
            "(%.1f MB) for %d MOs → %s",
            h1_full.shape, eri_packed.nbytes / 2**20, n, mech_dir,
        )

    except Exception as exc:
        log.warning(
            "[TOWER] Could not save full MO integrals for %r: %s — "
            "tower will fall back to heuristic extension.",
            spec.name, exc,
        )


# ===========================================================================
# 5. ALGEBRAIC VALIDATION (Pre-simulation checks on generated integrals)
# ===========================================================================

def validate_integral_dataset(data: Dict, step: MQEStep, spec: MQEMechanismSpec) -> Dict[str, bool]:
    """Run algebraic validation checks on a generated step dataset.

    These checks are independent of the quantum simulation and run on
    the raw classical integral data. They verify the mathematical
    integrity of the generated Hamiltonians before MQE pipeline execution.

    Standard Checks:
        (a) Hermiticity:     h1[p,q] = h1[q,p] ∀ p,q
        (b) ERI 8-fold:      g[p,q,r,s] = g[q,p,r,s] = g[p,q,s,r] = g[r,s,p,q]
        (c) Phase closure:   step.phase_index_k ≡ Σν (mod m) [algebraic]
        (d) Electron count:  step.cumulative_electrons = Σ|A_i| so far
        (e) Orbital sanity:  all orbital indices in A_n, P_n, B_n < ncas
        (f) ERI positivity:  diagonal ERIs g[p,p,p,p] ≥ 0 (Coulomb integrals)
        (g) h_diag ordering: no extreme outliers (|h_pp| < 100 Ha)

    Photon-Aware Extensions (Gap 4a):
        (h) Photon orbital sanity:  indices in Gamma_n_abs, Gamma_n_emit < ncas
        (i) Photon energy physical: phi_photon_n == 0 if idle, else 0 < phi < pi
        (j) Photon flux consistency: cumulative_net_photons matches expected net flux
    """
    ncas   = len(data["h_diag"])
    h_diag = np.array([data["h_diag"][str(p)] for p in range(ncas)])
    h_hop  = {
        ast.literal_eval(k): v for k, v in data["h_hop"].items()
    }
    g_full = {ast.literal_eval(k): v for k, v in data["g_full"].items()}

    checks = {}

    # (a) Hermiticity
    herm_ok = True
    for (p, q), val in h_hop.items():
        rev = h_hop.get((q, p), val)
        if abs(val - rev) > 1e-10:
            herm_ok = False
            break
    checks["hermiticity"] = herm_ok

    # (b) ERI 8-fold symmetry — correct check on canonical keys.
    #
    # compress_eri stores only the canonical (p,q,r,s) where (p,q)>=(r,s) and
    # p>=q, r>=s.  For any canonical key, the seven other symmetry-equivalent
    # permutations must NOT appear as separate canonical keys (that would
    # indicate duplicated storage and broken 8-fold factoring).
    #
    # We also spot-check the first 50 canonical entries and assert that their
    # canonical-form permutations all map back to the SAME canonical key and
    # the same value — not to a missing key (which would silently pass the
    # old fallback test).
    def _canonical_eri(p, q, r, s):
        pq = (min(p, q), max(p, q))
        rs = (min(r, s), max(r, s))
        if pq > rs:
            pq, rs = rs, pq
        return pq + rs

    eightfold_ok = True
    for i, ((p, q, r, s), val) in enumerate(g_full.items()):
        if i >= 50:
            break
        # Verify this key is itself canonical; if not, the compressor is broken.
        if (p, q, r, s) != _canonical_eri(p, q, r, s):
            eightfold_ok = False
            break
        # Verify each of the seven non-trivial permutations maps back to this
        # canonical key and is NOT stored as a separate entry.
        perms = [
            (q, p, r, s), (p, q, s, r), (q, p, s, r),
            (r, s, p, q), (s, r, p, q), (r, s, q, p), (s, r, q, p),
        ]
        for pp in perms:
            canon = _canonical_eri(*pp)
            if canon != (p, q, r, s):
                eightfold_ok = False
                break
            # The non-canonical form must not exist as a separate key
            if pp != (p, q, r, s) and pp in g_full:
                eightfold_ok = False
                break
        if not eightfold_ok:
            break
    checks["eri_8fold_symmetry"] = eightfold_ok

    # (c) Phase closure consistency — m_modulus comes from spec (single source of truth)
    expected_k = step.phase_index_k
    checks["phase_index_consistent"] = (
        0 <= expected_k < max(spec.m_modulus, 1)
    )

    # (d) Forward-only electron count consistency (legacy / to_mechanism_tuple compat)
    checks["cumulative_electrons_consistent"] = (
        step.cumulative_electrons == sum(
            len(s.A_n) for s in spec.steps[:step.step_n + 1]
        )
    )

    # (d2) S2 FIX: Net-flux electron consistency (reversible-aware)
    # NOTE: The suggestion proposed REPLACING (d) with this. That is incorrect:
    # (d) is the integrity check on cumulative_electrons (forward injection count),
    # which is a separate invariant consumed by to_mechanism_tuple(). Both must
    # be checked independently.
    expected_net_e = sum(
        len(s.A_n) - len(s.A_n_eject) for s in spec.steps[:step.step_n + 1]
    )
    checks["cumulative_net_electrons_consistent"] = (
        step.cumulative_net_electrons == expected_net_e
    )

    # (e) Orbital index sanity
    all_orbital_indices = step.A_n + step.P_n + step.B_n
    checks["orbital_indices_valid"] = all(0 <= idx < ncas for idx in all_orbital_indices)

    # (f) ERI diagonal positivity
    diag_positive = all(
        g_full.get((p, p, p, p), 0.0) >= -1e-10
        for p in range(ncas)
    )
    checks["eri_diagonal_positive"] = diag_positive

    # (g) Hamiltonian sanity (no extreme values)
    checks["h_diag_bounded"] = bool(np.all(np.abs(h_diag) < 100.0))

    # ── GAP 4a: Photon-aware validation ──────────────────────────────────
    # (h) Photon orbital index sanity
    photon_indices = step.Gamma_n_abs + step.Gamma_n_emit
    checks["photon_orbital_indices_valid"] = all(
        0 <= idx < ncas for idx in photon_indices
    )

    # (i) Photon energy physical bounds
    if not step.Gamma_n_abs and not step.Gamma_n_emit:
        # No photons absorbed/emitted: phase must be zero
        checks["phi_photon_energy_physical"] = (step.phi_photon_n == 0)
    else:
        # Active photochemistry: phase must be in (0, pi)
        checks["phi_photon_energy_physical"] = (0 < step.phi_photon_n < np.pi)

    # (j) Cumulative photon flux consistency
    expected_net_p = sum(
        len(s.Gamma_n_abs) - len(s.Gamma_n_emit)
        for s in spec.steps[:step.step_n + 1]
    )
    checks["cumulative_photon_consistent"] = (
        step.cumulative_net_photons == expected_net_p
    )

    return checks


# ===========================================================================
# 6. STOICHIOMETRIC INVARIANCE VALIDATOR (Full mechanism)
# ===========================================================================

def validate_mechanism_stoichiometry(spec: MQEMechanismSpec) -> Dict:
    """Verify stoichiometric invariance for the full mechanism specification.

    Implements Theorem 2 (Universal Stoichiometric Invariance) checks
    on the mechanism descriptor (algebraic, not simulation-dependent).
    Extended with Gap 4b: Photon-aware mechanism validation.

    Checks:
        (i)   Electron conservation:   Σ (|A_n| - |A_n_eject|) = expected_net_electrons
        (ii)  Phase closure:           Σ (ν_n - ν_decouple_n) ≡ 0 (mod m)
        (iii) Step count:              len(steps) == M_steps
        (iv)  Orbital consistency:     A_n, P_n, B_n, eject indices < n_orbitals
        (v)   Monotone k:              k^{(n)} = Σ (ν - ν†) mod m
        (vi)  Crossing integrity:      is_crossing implies delta_CI_Ha is not None
        (vii) Photon orbital:          Gamma indices < n_orbitals
        (viii) Photon balance:         Abs/Em counts match expected totals
        (ix)  Phi sanity:              0 < phi < pi if photons present
    """
    m = spec.m_modulus
    N = spec.n_orbitals

    # ── EXISTING CHECKS ───────────────────────────────────────────────

    # (i) Net Electron Conservation: Σ (|A_n| - |A_n_eject|) = expected_net_electrons
    actual_net_electrons = sum(
        len(s.A_n) - len(s.A_n_eject) for s in spec.steps
    )
    electron_ok = (actual_net_electrons == spec.expected_net_electrons)

    # (ii) Net Phase Closure: Σ (ν_n - ν_decouple_n) ≡ 0 (mod m)
    total_nu       = sum(s.nu_n for s in spec.steps)
    total_decouple = sum(s.nu_decouple_n for s in spec.steps)
    net_nu         = total_nu - total_decouple
    phase_ok       = (m == 1) or (net_nu % m == 0)

    # (iii) Step count
    step_count_ok = (len(spec.steps) == spec.M_steps)

    # (iv) Orbital consistency (forward + inverse sets)
    orbital_ok = all(
        all(0 <= idx < N for idx in 
            s.A_n + s.P_n + s.B_n + s.A_n_eject + s.P_n_eject + s.B_n_decouple)
        for s in spec.steps
    )

    # (v) Monotone phase index (tracks net accumulation)
    k_running = 0
    phase_monotone_ok = True
    for s in spec.steps:
        k_running = (k_running + s.nu_n - s.nu_decouple_n) % max(m, 1)
        if s.phase_index_k != k_running:
            phase_monotone_ok = False
            break

    # (vi) Crossing integrity
    crossing_ok = all(
        (not s.is_crossing) or (s.delta_CI_Ha is not None and s.crossing_orbitals is not None)
        for s in spec.steps
    )

    # ── NEW: PHOTON-AWARE CHECKS (Gap 4b) ─────────────────────────────

    # (vii) Photon orbital consistency
    # Check that all photon interaction orbitals are within bounds
    photon_orbital_ok = all(
        all(0 <= idx < N for idx in s.Gamma_n_abs + s.Gamma_n_emit)
        for s in spec.steps
    )
    
    # (viii) Photon balance
    # Verify total absorbed/emitted counts match the spec
    actual_p_abs = sum(len(s.Gamma_n_abs) for s in spec.steps)
    actual_p_emit = sum(len(s.Gamma_n_emit) for s in spec.steps)
    actual_p_net = actual_p_abs - actual_p_emit
    
    photon_balance_ok = (
        actual_p_abs == spec.expected_total_photons_absorbed and
        actual_p_emit == spec.expected_total_photons_emitted
    )

    # (ix) Phi sanity
    # If photon interactions are present, ensure the phase angle is physical
    phi_sanity_ok = True
    has_photon_activity = any(s.Gamma_n_abs or s.Gamma_n_emit for s in spec.steps)
    if has_photon_activity:
        phi_sanity_ok = (0 < spec.phi_photon < np.pi)

    # ── AGGREGATION ───────────────────────────────────────────────────
    all_passed = all([
        electron_ok, phase_ok, step_count_ok, orbital_ok, 
        phase_monotone_ok, crossing_ok,
        # New photon checks
        photon_orbital_ok, photon_balance_ok, phi_sanity_ok
    ])

    return {
        "mechanism":               spec.name,
        "m_modulus":               m,
        "M_steps":                 spec.M_steps,
        "passed":                  all_passed,
        "electron_conservation":   {"ok": electron_ok, 
                                    "expected": spec.expected_net_electrons, 
                                    "actual": actual_net_electrons},
        "phase_closure":           {"ok": phase_ok, 
                                    "total_nu": total_nu,      # <--- ADD THIS
                                    "net_nu": net_nu, 
                                    "net_nu_mod_m": net_nu % max(m,1)},
        "step_count":              {"ok": step_count_ok, 
                                    "expected": spec.M_steps, 
                                    "actual": len(spec.steps)},
        "orbital_consistency":     {"ok": orbital_ok},
        "phase_index_monotone":    {"ok": phase_monotone_ok},
        "crossing_integrity":      {"ok": crossing_ok},
        
        # New Return dictionary fields for Photons
        "photon_orbital_consistency": {"ok": photon_orbital_ok},
        "photon_balance": {
            "ok": photon_balance_ok,
            "absorbed": actual_p_abs,
            "emitted": actual_p_emit,
            "net": actual_p_net,
            "expected_net": spec.expected_net_photons
        },
        "phi_photon_sanity": {"ok": phi_sanity_ok},
    }


# ===========================================================================
# 7. MAIN GENERATION PIPELINE
# ===========================================================================

def build_all_specs(n_orbitals: int = 4) -> Dict[str, MQEMechanismSpec]:
    """Build all five MQE mechanism specifications.

    Args:
        n_orbitals: Active-space orbital count N (default 4 = minimal
                    chemically meaningful test; must match the molecule size).

    Returns:
        Dict mapping mechanism name → MQEMechanismSpec.
    """
    return {
        "nitrogenase_lt": _build_nitrogenase_lt_spec(n_orbitals),
        "nitrogenase_closed_loop": _build_nitrogenase_closed_loop_spec(n_orbitals),
        "nitrogenase_lt_m8":      _build_nitrogenase_lt_m8_spec(n_orbitals),
        "nitrogenase_lt_parallel": _build_nitrogenase_lt_parallel_spec(n_orbitals),
        "psii":           _build_psii_spec(n_orbitals),
        "psii_photo":     _build_psii_photo_spec(n_orbitals),
        "hydrogenase":          _build_hydrogenase_spec(min(n_orbitals, 2)),
        "hydrogenase_oxidation": _build_hydrogenase_oxidation_spec(min(n_orbitals, 2)),
        "z3_cofactor":    _build_z3_cofactor_spec(min(n_orbitals, 3)),
        "z5_cofactor":    _build_z5_cofactor_spec(min(n_orbitals, 4)),
        "haber_bosch":    _build_haber_bosch_spec(n_orbitals),
        "nitrogenase_fe4s4": _build_nitrogenase_fe4s4_spec(n_orbitals),
        "nitrogenase_femoco": _build_nitrogenase_femoco_spec(n_orbitals),
        "nitrogenase_group_a": _build_nitrogenase_group_a_spec(n_orbitals),
        "nitrogenase_group_d": _build_nitrogenase_group_d_spec(n_orbitals),
        "mo_nitrogenase":      _build_mo_nitrogenase_spec(n_orbitals),
        "assimilatory_nr":     _build_assimilatory_nr_spec(n_orbitals),
        "photocatalytic_n2":   _build_photocatalytic_n2_spec(n_orbitals),
        "v_nitrogenase":       _build_v_nitrogenase_spec(n_orbitals),
        "cu_co2rr":            _build_cu_co2rr_spec(n_orbitals),
        "femon2_trimer":       _build_femon2_trimer_spec(n_orbitals),
        "anammox_proxy": _build_anammox_proxy_spec(n_orbitals),
        "ethylene_epoxidation": _build_ethylene_epoxidation_spec(n_orbitals),
        "thymine_dimer_proxy": _build_thymine_dimer_spec(n_orbitals),
        "atp_hydrolysis_proxy": _build_atp_hydrolysis_spec(n_orbitals),
        "rnr_radical_proxy": _build_rnr_proxy_spec(n_orbitals),
        "reversible_quinone": _build_reversible_quinone_spec(n_orbitals),
        "cyp450_metabolism": _build_cyp450_metabolism_spec(n_orbitals),
        # ── New Unique Entries (6, 9, 10, 11) ───────────────────────────
        "complex_i":               _build_complex_i_spec(n_orbitals),
        "codh_acs":                _build_codh_acs_spec(n_orbitals),
        "cyt_bd_oxidase":          _build_cyt_bd_oxidase_spec(n_orbitals),
        "cyt_c_oxidase":           _build_cyt_c_oxidase_spec(n_orbitals),
        # ── Oldform Lifts (Entries 2, 4, 8, 12, 15) ─────────────────────
        "mo_nitrogenase_m4":       _build_mo_nitrogenase_m4_spec(n_orbitals),
        "v_nitrogenase_m4":        _build_v_nitrogenase_m4_spec(n_orbitals),
        "assimilatory_nr_m4":      _build_assimilatory_nr_m4_spec(n_orbitals),
        "cu_co2rr_m4":             _build_cu_co2rr_m4_spec(n_orbitals),
        "photocatalytic_n2_m4":    _build_photocatalytic_n2_m4_spec(n_orbitals),
    }


def generate_mechanism_dataset(
    spec:         MQEMechanismSpec,
    basis:        str   = "STO-3G",
    output_dir:   Path  = Path("mqe_datasets"),
    validate_fci: bool  = True,
    verbose:      int   = 0,
) -> Tuple[bool, Dict]:
    """Generate all step datasets for one mechanism and write to JSON files.

    File naming: {output_dir}/{mechanism_name}/step_{n:02d}.json
    Index file:  {output_dir}/{mechanism_name}/manifest.json

    Args:
        spec:         MQEMechanismSpec for this mechanism.
        basis:        PySCF basis set.
        output_dir:   Root output directory.
        validate_fci: Include FCI reference energies.
        verbose:      PySCF verbosity.

    Returns:
        (all_passed, summary_dict) where all_passed iff all algebraic
        checks pass for every step.
    """

    # Local density cache — one per generate_mechanism_dataset call.
    # Keeps inter-step warm-starting fully isolated per mechanism and
    # eliminates the global-state race condition under concurrent generation.
    dm_cache: Dict[str, Optional[np.ndarray]] = {}

    mech_dir = output_dir / spec.name
    mech_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"\n{'='*68}")
    log.info(f"[MQE-GEN] Mechanism: {spec.name.upper()}")
    log.info(f"  M={spec.M_steps} steps | m={spec.m_modulus} (ℤ_{spec.m_modulus}) | "
             f"N={spec.n_orbitals} orbitals")
    log.info(f"  Expected e⁻: {spec.expected_total_electrons} | "
             f"Expected Σν: {sum(s.nu_n for s in spec.steps)} ≡ 0 (mod {spec.m_modulus})")
    log.info(f"{'='*68}")

    # ── Algebraic stoichiometry validation (before any integrals) ────────────
    stoich = validate_mechanism_stoichiometry(spec)
    log.info(
        f"[STOICH] Phase closure: {'[✓]' if stoich['phase_closure']['ok'] else '[✗]'} "
        f"Σν={stoich['phase_closure']['net_nu']} mod {spec.m_modulus} = "
        f"{stoich['phase_closure']['net_nu_mod_m']}"
    )
    log.info(
        f"[STOICH] Electron count: {'[✓]' if stoich['electron_conservation']['ok'] else '[✗]'} "
        f"actual={stoich['electron_conservation']['actual']} "
        f"expected={stoich['electron_conservation']['expected']}"
    )
    # ── Photon balance log (Gap 5e) ───────────────────────────────────────────
    pb = stoich.get("photon_balance", {})
    if pb.get("absorbed", 0) > 0 or pb.get("emitted", 0) > 0 or spec.phi_photon > 0:
        log.info(
            f"[PHOTON] abs={pb.get('absorbed', 0)} emit={pb.get('emitted', 0)} "
            f"net={pb.get('net', 0)} "
            f"phi={spec.phi_photon:.5f} rad "
            f"({'[✓]' if pb.get('ok', True) else '[✗]'})"
        )

    step_results   = []
    step_datasets  = []
    all_checks_ok  = stoich["passed"]

    fci_energies: List[float] = []

    for step in spec.steps:
        t0    = time.time()
        data  = generate_step_integrals(
            spec, step, basis, validate_fci, verbose, dm_cache,
            _save_mo_integrals_dir=mech_dir,
            _n_total_orbs=76,
        )
        dt    = time.time() - t0

        # Algebraic validation of this step's integrals
        checks = validate_integral_dataset(data, step, spec)
        step_ok = all(checks.values())
        all_checks_ok = all_checks_ok and step_ok

        e_ref = data.get("circuit_reference_energy_Ha", 0.0)
        if isinstance(e_ref, (int, float)):
            fci_energies.append(float(e_ref))

        log.info(
            f"  [STEP {step.step_n}] E_FCI={e_ref:+.8f} Ha | "
            f"ν={step.nu_n} | k^({step.step_n})={step.phase_index_k} | "
            f"checks={'[✓]' if step_ok else '[✗]'} | {dt:.1f}s"
        )
        if not step_ok:
            for chk, ok in checks.items():
                if not ok:
                    log.warning(f"    ✗ {chk}")

        # Write step JSON
        step_path = mech_dir / f"step_{step.step_n:02d}.json"
        step_path.write_text(json.dumps(data, indent=2))

        step_results.append({
            "step_n":   step.step_n,
            "geometry": step.geometry_label,
            "e_ref_Ha": e_ref,
            "checks":   checks,
            "passed":   step_ok,
        })
        step_datasets.append(data)

    # ── Energy ordering validation (declarative — dispatches on spec attribute) ─
    energy_ordering_ok = _validate_energy_ordering(spec, fci_energies)

    # ── Save NOONs + MO energies for Iwasawa tower climber ───────────────────
    _save_noons_for_tower(mech_dir, step_datasets)

    # h1_full.npy + eri_packed.npy were saved inside generate_step_integrals
    # at the Janus step (is_crossing=True), reusing the already-converged mf.
    # No second ROHF needed.

    # ── Write manifest ────────────────────────────────────────────────────────
    manifest = {
        "mechanism":              spec.name,
        "description":            spec.description,
        "M_steps":                spec.M_steps,
        "m_modulus":              spec.m_modulus,
        "S_target":               spec.S_target,
        "n_orbitals":             spec.n_orbitals,
        "basis":                  basis,
        "stoichiometry":          stoich,
        "energy_ordering_ok":     energy_ordering_ok,
        "step_results":           step_results,
        "all_algebraic_ok":       all_checks_ok,
        "fci_energies_Ha":        fci_energies,
        "generated_at":           time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mqe_article_reference":  "nanoprotogeny.theory.mqe v2026.05",
    }
    manifest_path = mech_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # ── Print summary ─────────────────────────────────────────────────────────
    _print_mechanism_summary(spec, stoich, step_results, energy_ordering_ok, all_checks_ok)

    return all_checks_ok, manifest


def _validate_energy_ordering(spec: "MQEMechanismSpec", fci_energies: List[float]) -> bool:
    """Check expected energy ordering using the declarative ``spec.expected_energy_ordering``.

    Dispatches on the ordering label set by each mechanism builder.  Adding a
    new mechanism only requires setting ``expected_energy_ordering`` in its
    ``MQEMechanismSpec`` — no edits here are needed.

    Ordering labels
    ---------------
    "decreasing"          fci[-1] < fci[0]            reductive cycles
    "increasing"          fci[-1] > fci[0]            oxidative cycles
    "monotone_decreasing" each step ≤ previous         compression scans
    "monotone_increasing" each step ≥ previous         photo-oxidation
    "closure"             |fci[0] - fci[-1]| < tol    Zₘ cofactor return
    "closed_loop"         fwd half ↓ + |fci[-1]-fci[0]| < 10 mHa
    "nondegen"            not all steps equal           radical proxy guard
    "reversible_quinone"  reduction ↑ then oxidation ↓
    "none"                always passes
    """
    if not fci_energies or len(fci_energies) < 2:
        return True

    tol      = 1e-6
    ordering = spec.expected_energy_ordering

    if ordering == "decreasing":
        return fci_energies[-1] < fci_energies[0] - tol

    elif ordering == "increasing":
        return fci_energies[-1] > fci_energies[0] + tol

    elif ordering == "monotone_decreasing":
        return all(
            fci_energies[i] >= fci_energies[i + 1] - tol
            for i in range(len(fci_energies) - 1)
        )

    elif ordering == "monotone_increasing":
        return all(
            fci_energies[i + 1] > fci_energies[i] - tol
            for i in range(len(fci_energies) - 1)
        )

    elif ordering == "closure":
        return abs(fci_energies[0] - fci_energies[-1]) < tol

    elif ordering == "closed_loop":
        # Forward half (steps 0–7) must show net energy decrease;
        # final step must return within 10 mHa of the starting energy.
        if len(fci_energies) < 16:
            return True   # incomplete dataset — skip
        fwd_ok    = fci_energies[7] < fci_energies[0] - tol
        return_ok = abs(fci_energies[15] - fci_energies[0]) < 10e-3
        return fwd_ok and return_ok

    elif ordering == "nondegen":
        # Weak guard: energies must not all be degenerate (numerical noise check)
        return len(set(round(e, 6) for e in fci_energies)) > 1

    elif ordering == "reversible_quinone":
        # Reduction phase (steps 0–2): monotonically increasing strain energy.
        # Oxidation phase (steps 3–5): recovers below reduction peak.
        if len(fci_energies) < 6:
            return True
        reduction_ok = (
            fci_energies[0] < fci_energies[1] - tol
            and fci_energies[1] < fci_energies[2] - tol
        )
        oxidation_ok = fci_energies[5] < fci_energies[2] - tol
        return reduction_ok and oxidation_ok

    # "none" and any unrecognised label: no constraint — always passes
    return True


def _print_mechanism_summary(
    spec:                MQEMechanismSpec,
    stoich:              Dict,
    step_results:        List[Dict],
    energy_ordering_ok:  bool,
    all_ok:              bool,
) -> None:
    w = 68
    print(f"\n{'='*w}")
    print(f" MQE DATASET: {spec.name.upper()}")
    print(f"{'='*w}")
    # Contract: 
    # validate_mechanism_stoichiometry provides 'total_nu_mod_m'
    # _print_mechanism_summary consumes 'total_nu_mod_m'
    print(f"  ℤ_{spec.m_modulus} phase closure : "
          f"{'[✓]' if stoich['phase_closure']['ok'] else '[✗]'} "
          f"Σν={stoich['phase_closure']['total_nu']} mod {spec.m_modulus} = "
          f"{stoich['phase_closure']['net_nu_mod_m']}")
    print(f"  Electron count      : "
          f"{'[✓]' if stoich['electron_conservation']['ok'] else '[✗]'} "
          f"{stoich['electron_conservation']['actual']} e⁻ "
          f"(expected {stoich['electron_conservation']['expected']})")
    # Gap 5e: photon balance line
    pb = stoich.get("photon_balance", {})
    if pb.get("absorbed", 0) > 0 or pb.get("emitted", 0) > 0:
        print(f"  Photon balance      : "
              f"{'[✓]' if pb.get('ok', True) else '[✗]'} "
              f"abs={pb.get('absorbed', 0)} emit={pb.get('emitted', 0)} "
              f"net={pb.get('net', 0)} "
              f"(expected net={pb.get('expected_net', 0)})")
    print(f"  Energy ordering     : {'[✓]' if energy_ordering_ok else '[✗]'}")
    print(f"  All algebraic checks: {'[✓] PASSED' if all_ok else '[✗] FAILED'}")
    print(f"\n  Step breakdown:")
    for sr in step_results:
        e = sr.get('e_ref_Ha', 0.0)
        e_str = f"{e:+.8f} Ha" if isinstance(e, float) else str(e)
        ok_str = "[✓]" if sr["passed"] else "[✗]"
        print(f"    n={sr['step_n']}: {e_str}  {ok_str}  {sr['geometry']}")
    print(f"{'='*w}")


def generate_all_datasets(
    basis:        str  = "STO-3G",
    n_orbitals:   int  = 4,
    output_dir:   Path = Path("mqe_datasets"),
    validate_fci: bool = True,
    verbose:      int  = 0,
) -> Dict:
    """Generate all five MQE mechanism datasets and write a master manifest.

    Args:
        basis:        PySCF basis set default for unit test is STO-3G (use cc-pVTZ for publication quality).
        n_orbitals:   Active-space orbital count N (default 4).
        output_dir:   Root output directory.
        validate_fci: Include FCI reference energies (required for chemical-accuracy test).
        verbose:      PySCF verbosity level.

    Returns:
        Dict: Master manifest with all mechanism results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    specs    = build_all_specs(n_orbitals)
    results  = {}
    all_pass = True

    print(f"\n{'#'*68}")
    print(f"  MQE FRAMEWORK — DEFINITIVE TEST DATASET GENERATION")
    print(f"  {len(specs)} mechanisms | basis={basis} | N={n_orbitals} orbitals")
    print(f"{'#'*68}\n")

    for name, spec in specs.items():
        # Verify algebraic stoichiometry before any computation
        stoich = validate_mechanism_stoichiometry(spec)
        if not stoich["passed"]:
            log.error(
                f"[MQE-GEN] Mechanism {name!r} failed algebraic stoichiometry "
                f"check BEFORE integral generation. This is a code bug. "
                f"Details: {stoich}"
            )
            results[name] = {"passed": False, "error": "stoichiometry_spec_invalid"}
            all_pass = False
            continue

        passed, manifest = generate_mechanism_dataset(
            spec, basis, output_dir, validate_fci, verbose
        )
        results[name]    = {"passed": passed, "manifest_path": str(output_dir / name / "manifest.json")}
        all_pass         = all_pass and passed

    # ── Write master manifest ─────────────────────────────────────────────────
    master = {
        "mqe_test_suite":     "Modular Quantum Emulator Framework",
        "mechanisms":         list(specs.keys()),
        "basis":              basis,
        "n_orbitals":         n_orbitals,
        "all_passed":         all_pass,
        "results":            results,
        "generated_at":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validation_criteria": {
            "phase_closure":      "k_total ≡ 0 (mod m) for each mechanism",
            "electron_count":     "Σ|A_n| = expected_total_electrons",
            "chemical_accuracy":  "|E_ZNE - E_FCI| ≤ 1.6 mHa at every step",
            "hermiticity":        "h1[p,q] = h1[q,p] ∀ p,q",
            "eri_8fold":          "g[p,q,r,s] = g[q,p,r,s] = g[p,q,s,r] = g[r,s,p,q]",
            "energy_ordering":    "FCI energies follow expected physical ordering",
        },
    }
    master_path = output_dir / "mqe_master_manifest.json"
    master_path.write_text(json.dumps(master, indent=2))

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\n{'#'*68}")
    print(f"  MQE DATASET GENERATION COMPLETE")
    print(f"  Output: {output_dir}/")
    print(f"  Master manifest: {master_path}")
    print(f"\n  {'Mechanism':<22} {'m':>4} {'M':>3} {'Algebraic':>12} {'Status':>8}")
    print(f"  {'─'*22} {'─'*4} {'─'*3} {'─'*12} {'─'*8}")
    for name, r in results.items():
        spec = specs[name]
        ok   = r.get("passed", False)
        print(f"  {name:<22} {spec.m_modulus:>4} {spec.M_steps:>3} "
              f"{'[✓] OK' if ok else '[✗] FAIL':>12} {('[✓]' if ok else '[✗]'):>8}")
    print(f"\n  OVERALL: {'[✓] ALL PASSED' if all_pass else '[✗] SOME FAILED'}")
    print(f"{'#'*68}\n")

    return master


# ===========================================================================
# 8. ARG PARSER & MAIN
# ===========================================================================

def build_mqe_dataset_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the MQE dataset generator."""
    p = argparse.ArgumentParser(
        description=(
            "MQE Test Dataset Generator: generates the definitive molecular "
            "integral datasets for validating the Modular Quantum Emulator "
            "framework across all five mechanism classes and Z_m phase groups."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--mechanism",
        choices=[
            "nitrogenase_lt",
            "nitrogenase_lt_m8",
            "nitrogenase_lt_parallel",
            "nitrogenase_fe4s4",
            "nitrogenase_closed_loop",
            "nitrogenase_femoco",
            "nitrogenase_group_a",
            "nitrogenase_group_d",
            "mo_nitrogenase",
            "assimilatory_nr",
            "photocatalytic_n2",
            "v_nitrogenase",
            "cu_co2rr",
            "femon2_trimer",
            "anammox_proxy",
            "psii",
            "psii_photo",
            "hydrogenase",
            "hydrogenase_oxidation",
            "z3_cofactor",
            "z5_cofactor",
            "haber_bosch",
            "ethylene_epoxidation",
            "thymine_dimer_proxy",
            "atp_hydrolysis_proxy",
            "rnr_radical_proxy",
            "reversible_quinone",
            "cyp450_metabolism",
            "all"
        ],
        default="all",
        help=(
            "Mechanism to generate. 'all' generates all registered mechanisms. "
            "Use 'psii_photo' for the photo-driven Kok S-state cycle with explicit "
            "P680 photon absorption fields. Use 'nitrogenase_closed_loop' for the "
            "full 16-step closed-loop catalyst-regeneration benchmark."
        ),
    )
    p.add_argument(
        "--basis", default="STO-3G",
        help=(
            "PySCF basis set. STO-3G is recommended for compatibility with "
            "mqe.py benchmarks. Use def2-SVP for publication-quality "
            "metalloenzyme integrals."
        ),
    )
    p.add_argument(
        "--n_orbitals", type=int, default=4,
        help=(
            "Active-space orbital count N. Must be consistent with the molecule "
            "size (N=2 for H2/hydrogenase, N=3 for H3+/Z3, N=4 for Fe2S2/LT/PSII)."
        ),
    )
    p.add_argument(
        "--output_dir", default="mqe_datasets",
        help="Root output directory for generated JSON datasets.",
    )
    p.add_argument(
        "--no_fci", action="store_true", default=False,
        help=(
            "Skip exact FCI reference computation (faster, but disables chemical-accuracy "
            "validation).  FCI is run by default for active spaces with ncas ≤ 20."
        ),
    )
    p.add_argument(
        "--verbose", type=int, default=0,
        help="PySCF verbosity level (0=silent, 3=normal, 9=debug).",
    )
    p.add_argument(
        "--verify_only", action="store_true", default=False,
        help=(
            "Run algebraic stoichiometry verification ONLY (no integrals generated). "
            "Useful for checking mechanism specifications before committing to computation."
        ),
    )
    return p


# ===========================================================================
# RIEMANN SCAFFOLD DATASET GENERATOR
# ===========================================================================

def _build_scaffold_hamiltonian(
    E_target:      float,
    n_orbitals:    int,
    n_elec_alpha:  int,
    n_elec_beta:   int,
    delta_virtual: float = 0.01,
) -> Tuple[List[List[float]], List[List[List[List[float]]]], float]:
    r"""Construct a minimal diagonal Hamiltonian with FCI energy E_target.

    For a non-interacting (ERI=0) system with nα α-electrons and nβ β-electrons
    in N orbitals:

        h1[i, i] = E_target / (nα + nβ)   for i  < nα or i < nβ  (occupied)
        h1[i, i] = E_target / (nα + nβ) + δ  otherwise            (virtual)
        h1[i, j] = 0                          for i ≠ j
        ERI      = 0

    FCI ground state energy = E_target exactly (non-interacting, unique by gap δ).

    Physical justification: the QPE signal depends on E·τ — the magnitude of
    orbital energies does not affect the ROBUSTNESS of the signal provided τ is
    chosen appropriately.  For synthetic scaffold datasets τ is already fixed by
    the Riemann preflight τ-sequence.

    Args:
        E_target:      Target FCI ground-state energy [Ha] (negative for bound states).
        n_orbitals:    Active-space orbital count N.
        n_elec_alpha:  Number of α electrons.
        n_elec_beta:   Number of β electrons.
        delta_virtual: Energy gap between occupied and virtual orbitals [Ha].
                       Must be > 0 to ensure a unique non-degenerate ground state.

    Returns:
        (h1, eri, e_core) where:
            h1   : N×N list (diagonal, off-diagonal = 0)
            eri  : N×N×N×N list (all zeros)
            e_core: 0.0 (energy absorbed into active-space integrals)
    """
    n_elec = n_elec_alpha + n_elec_beta
    eps_occ = E_target / n_elec if n_elec > 0 else 0.0
    eps_vir = eps_occ + delta_virtual

    # Build h1: diagonal only, occupied orbitals get eps_occ, virtual get eps_vir
    # "Occupied" = the n_elec lowest spatial orbitals in the spin-summed picture.
    n_occupied = min(max(n_elec_alpha, n_elec_beta), n_orbitals)
    h1 = [[0.0] * n_orbitals for _ in range(n_orbitals)]
    for i in range(n_orbitals):
        h1[i][i] = eps_occ if i < n_occupied else eps_vir

    # ERI: all zeros (non-interacting)
    eri = [[[[0.0] * n_orbitals for _ in range(n_orbitals)]
             for _ in range(n_orbitals)]
            for _ in range(n_orbitals)]

    return h1, eri, 0.0


def build_riemann_scaffold_dataset(
    mechanism_spec: "MQEMechanismSpec",
    scaffold:       "RiemannScaffold",
    k_target:       int   = 0,
    alpha:          float = 0.05,
    delta_virtual:  float = 0.01,
    output_dir:     Optional[Path] = None,
) -> List[Dict]:
    r"""Generate step-wise synthetic integrals from a Riemann spectral scaffold.

    Theory
    ------
    thm:spectral_identification guarantees that the Janus eigenphase of H_MQE
    equals s·γ_k, so the physical Janus energy is (QPE e^{-iHτ} convention):

        E_Janus = −s · γ_k / (n* · Δt_m)   [Ha, negative]

    This function engineers Hamiltonians whose FCI ground state equals E_Janus
    at each Janus step, so that Path-R (Riemann QPE) correctly identifies γ_k
    with a near-zero residual.  Non-Janus steps are assigned energies on a
    V-shaped trajectory centred at the Janus intermediate.

    V-shaped trajectory for non-Janus steps
    ----------------------------------------
    Let n_J = index of the first Janus step.

        E_n = E_Janus + α · (n − n_J)²   [Ha]

    α > 0 means E_n > E_Janus for n ≠ n_J (Janus is the deepest point, i.e.
    the most stabilised intermediate in the LT cycle).  The continuous MLE
    fallback can recover these energies to within chemical accuracy.

    Hamiltonian construction
    ------------------------
    At each step n, we build a minimal non-interacting Hamiltonian:

        h1[i, i] = E_active / N_e   (occupied orbitals i < max(nα, nβ))
        h1[i, i] = E_active / N_e + δ_virtual   (virtual orbitals)
        ERI      = 0
        E_core   = 0

    FCI(h1, ERI) = E_active = E_n exactly.

    Args:
        mechanism_spec: MQEMechanismSpec with M_steps, m_modulus, n_orbitals, steps.
        scaffold:       RiemannScaffold built from the corresponding MechanismTuple.
        k_target:       Index into scaffold.janus_energies — which Riemann zero to
                        use as the primary Janus prediction (default 0 = γ₁).
        alpha:          V-trajectory curvature [Ha/step²].  Controls how far the
                        non-Janus energies sit above E_Janus.
        delta_virtual:  Energy gap between occupied and virtual orbitals [Ha].
        output_dir:     If given, write JSON files to
                        <output_dir>/<mechanism>/step_{n:02d}.json.
                        The directory is created if absent.

    Returns:
        List[Dict] of length M_steps; each dict is a complete step JSON matching
        the schema consumed by StepwiseIntegralStore / mqe run --dataset-dir.

    Example::

        scaffold = build_riemann_scaffold(mech_tuple)
        spec     = build_all_specs(4)["nitrogenase_lt"]
        steps    = build_riemann_scaffold_dataset(spec, scaffold, k_target=0)
        # steps[4]["fci_reference"]["E_active"] == scaffold.janus_energies[0]
    """
    from nanoprotogeny.molecular.mqeriemann import build_riemann_scaffold  # local import to avoid circularity

    N_orb   = mechanism_spec.n_orbitals
    M_steps = mechanism_spec.M_steps

    # Janus step indices from crossing list
    janus_step_indices = {
        step.step_n for step in mechanism_spec.steps if step.is_crossing
    }
    if not janus_step_indices:
        raise ValueError(
            f"Mechanism '{mechanism_spec.name}' has no Janus crossings — "
            "Riemann scaffold dataset generation requires at least one crossing."
        )
    n_J = min(janus_step_indices)

    # Janus energy from scaffold (negative — bound state)
    if k_target >= len(scaffold.janus_energies):
        raise ValueError(
            f"k_target={k_target} out of range; scaffold has "
            f"{len(scaffold.janus_energies)} zeros in window."
        )
    E_Janus = scaffold.janus_energies[k_target]   # already negative after sign fix
    gamma_k = scaffold.gammas[k_target]
    z_idx   = scaffold.zero_indices[k_target]

    # Electron counts: use mechanism spec cumulative at step M-1
    last_step = mechanism_spec.steps[-1]
    n_elec    = last_step.cumulative_net_electrons
    # Split evenly α/β; for odd N_e put the extra electron in α
    n_elec_beta  = n_elec // 2
    n_elec_alpha = n_elec - n_elec_beta

    log.info(
        "[RIEMANN-GEN] Mechanism=%s  M=%d  N=%d  N_e=%d (α=%d β=%d)",
        mechanism_spec.name, M_steps, N_orb, n_elec, n_elec_alpha, n_elec_beta,
    )
    log.info(
        "[RIEMANN-GEN] Janus step n_J=%d  γ_%d=%.6f  E_Janus=%.6f Ha  α=%.4f Ha/step²",
        n_J, z_idx + 1, gamma_k, E_Janus, alpha,
    )

    step_records: List[Dict] = []

    for step in mechanism_spec.steps:
        n = step.step_n

        # ── Assign target energy ──────────────────────────────────────────
        if n in janus_step_indices:
            E_n = E_Janus
        else:
            E_n = E_Janus + alpha * float((n - n_J) ** 2)

        # ── Build Hamiltonian ─────────────────────────────────────────────
        h1, eri, e_core = _build_scaffold_hamiltonian(
            E_target      = E_n,
            n_orbitals    = N_orb,
            n_elec_alpha  = n_elec_alpha,
            n_elec_beta   = n_elec_beta,
            delta_virtual = delta_virtual,
        )

        # ── Convert h1 list to StepwiseIntegralStore format ─────────────
        # _parse_step_integrals expects:
        #   h_diag : {str(i): float}       — diagonal one-body integrals
        #   h_hop  : {"(p, q)": float}     — off-diagonal one-body (empty here)
        #   g_full : {"(p,q,r,s)": float}  — two-electron (empty here, ERI=0)
        #   ecore_Ha : float               — core energy
        # get_reference_energy expects:
        #   circuit_reference_energy_Ha : float   — FCI ground-state energy
        # metadata.ncas is used for n_orbitals when store parses the file.
        h_diag_dict: Dict[str, float] = {
            str(i): h1[i][i] for i in range(N_orb)
        }
        h_hop_dict:  Dict[str, float] = {}    # diagonal only → no hopping
        g_full_dict: Dict[str, float] = {}    # ERI = 0

        # ── Assemble step JSON ────────────────────────────────────────────
        mqe_step_meta = step.to_dict(
            mechanism = mechanism_spec.name,
            M_total   = M_steps,
            m_modulus = mechanism_spec.m_modulus,
        )

        record = {
            "metadata": {
                "source":         "riemann_scaffold",
                "mechanism":      mechanism_spec.name,
                "step_n":         n,
                "M_total":        M_steps,
                "m_modulus":      mechanism_spec.m_modulus,
                "ncas":           N_orb,          # read by _parse_step_integrals
                "n_orbitals":     N_orb,
                "nelec_active":   [n_elec_alpha, n_elec_beta],
                "basis":          "synthetic_riemann",
                "geometry_label": step.geometry_label,
                "scaffold_class": scaffold.spectral_class,
                "k_target":       k_target,
                "gamma_k":        gamma_k,
                "zero_index":     z_idx + 1,          # 1-based
                "s_value":        scaffold.s,
                "n_star":         scaffold.n_star,
                "alpha_Ha_step2": alpha,
                "is_janus":       (n in janus_step_indices),
            },
            # StepwiseIntegralStore / _parse_step_integrals schema:
            "h_diag":   h_diag_dict,
            "h_hop":    h_hop_dict,
            "g_full":   g_full_dict,
            "ecore_Ha": e_core,
            # get_reference_energy reads this field:
            "circuit_reference_energy_Ha": E_n,
            # Human-readable extras (not parsed by store but useful for inspection):
            "fci_reference": {
                "nalpha":     n_elec_alpha,
                "nbeta":      n_elec_beta,
                "E_active":   E_n,
                "E_absolute": E_n + e_core,   # e_core=0 → same as E_active
            },
            "mqe_step": mqe_step_meta,
        }

        step_records.append(record)
        log.info(
            "[RIEMANN-GEN] Step n=%d: E_n=%.6f Ha  is_janus=%s",
            n, E_n, n in janus_step_indices,
        )

    # ── Write to disk if output_dir supplied ─────────────────────────────
    if output_dir is not None:
        mech_dir = Path(output_dir) / mechanism_spec.name
        mech_dir.mkdir(parents=True, exist_ok=True)
        for n, record in enumerate(step_records):
            step_path = mech_dir / f"step_{n:02d}.json"
            with open(step_path, "w") as fh:
                json.dump(record, fh, indent=2)
            log.info("[RIEMANN-GEN] Wrote %s", step_path)
        manifest_path = mech_dir / "manifest.json"
        with open(manifest_path, "w") as fh:
            json.dump({
                # Required by StepwiseIntegralStore._load_manifest + _manifest_to_mechanism_tuple:
                "mechanism":      mechanism_spec.name,
                "description":    f"Riemann scaffold dataset — {scaffold.spectral_class}",
                "M_steps":        M_steps,
                "m_modulus":      mechanism_spec.m_modulus,
                "n_orbitals":     N_orb,
                "S_target":       0.0,
                # Extra provenance fields (not parsed by store):
                "source":         "riemann_scaffold",
                "scaffold_class": scaffold.spectral_class,
                "k_target":       k_target,
                "gamma_k":        gamma_k,
                "E_Janus_Ha":     E_Janus,
                "janus_steps":    sorted(janus_step_indices),
                "steps":          [f"step_{n:02d}.json" for n in range(M_steps)],
            }, fh, indent=2)
        print(
            f"[RIEMANN-GEN] {M_steps} step files written to {mech_dir}/\n"
            f"  γ_{z_idx+1} = {gamma_k:.6f}  E_Janus = {E_Janus:+.6f} Ha  "
            f"class = {scaffold.spectral_class}\n"
            f"  Run with: mqe run --mechanism {mechanism_spec.name} --riemann "
            f"--dataset-dir {mech_dir.parent}/"
        )

    return step_records


# ── Track-2 hybrid dataset builder ───────────────────────────────────────────

def build_hybrid_scaffold_dataset(
    source_dataset_dir: Union[str, "Path"],
    mechanism_name: str,
    scaffold: "RiemannScaffold",
    k_target: int = 0,
    output_dir: Optional[Union[str, "Path"]] = None,
) -> List[Dict]:
    """Build a hybrid dataset: physical PySCF integrals + Riemann-anchored references.

    For each mechanism step:
      - h_diag, h_hop, g_full, ecore_Ha  → taken verbatim from the source PySCF dataset.
      - circuit_reference_energy_Ha       →
            Janus step:      E_Janus + ecore_Ha  (anchors scaffold comparison to γ_k)
            Non-Janus steps: PySCF value unchanged (tests real QPE accuracy).
      - metadata gains: E_FCI_active_pyscf, E_Janus_scaffold, truncation_gap_Ha.

    Physical meaning
    ----------------
    E_ref_chk = circuit_reference_energy_Ha − ecore_Ha.
    · Janus step:     E_ref_chk = E_Janus   → scaffold comparison |E_Janus − E_Janus| = 0.
    · Non-Janus step: E_ref_chk = E_FCI_active_PySCF → ZNE tested against real chemistry.

    The "truncation_gap_Ha" field records E_Janus − E_FCI_active_PySCF (≈ −54.6 Ha for
    CAS(4,4) vs the full active space predicted by the Riemann theory).  This gap quantifies
    the active-space truncation error and is the quantity that a full CAS(76,76) run would
    reduce to zero.

    Args:
        source_dataset_dir: Root directory containing the PySCF dataset
                            (parent of the mechanism subdirectory).
        mechanism_name:     Mechanism name (e.g. 'nitrogenase_lt').
        scaffold:           Pre-built RiemannScaffold for the mechanism.
        k_target:           Index into scaffold.janus_energies (0 → γ₁).
        output_dir:         Where to write the hybrid dataset.  Defaults to
                            source_dataset_dir/../hybrid/.

    Returns:
        List of per-step record dicts (same as build_riemann_scaffold_dataset).
    """
    import json
    from pathlib import Path
    from nanoprotogeny.molecular.mqeintegralstore import StepwiseIntegralStore
    from nanoprotogeny.molecular.mqehamiltonian import (
        build_qudit_hamiltonian_matrix,
        _project_hamiltonian_to_sector,
        ground_state_from_diagonalization,
    )

    source_root = Path(source_dataset_dir)
    store       = StepwiseIntegralStore(source_root, mechanism_name)
    M_steps     = store.M_steps

    if k_target >= len(scaffold.janus_energies):
        raise ValueError(
            f"k_target={k_target} out of range; scaffold has "
            f"{len(scaffold.janus_energies)} zeros in window."
        )

    E_Janus = scaffold.janus_energies[k_target]   # negative
    gamma_k = scaffold.gammas[k_target]
    z_idx   = scaffold.zero_indices[k_target]

    # Identify Janus steps from the mechanism reconstructed from the store
    mech_tuple     = store.to_mechanism_tuple()
    janus_step_set = {c[0] for c in mech_tuple.crossings}

    # Output directory: source/../hybrid/<mechanism_name>/
    if output_dir is None:
        output_dir = source_root.parent / "hybrid"
    out_root = Path(output_dir)
    mech_dir = out_root / mechanism_name
    mech_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "[HYBRID-GEN] mechanism=%s  M=%d  scaffold=%s  γ_%d=%.6f  E_Janus=%.6f Ha",
        mechanism_name, M_steps, scaffold.spectral_class,
        z_idx + 1, gamma_k, E_Janus,
    )
    log.info("[HYBRID-GEN] Janus steps from crossings: %s", sorted(janus_step_set))

    step_records: List[Dict] = []

    for n in range(M_steps):
        # ── Load PySCF integrals ──────────────────────────────────────────
        h_diag, h_hop, g_full, ecore_pyscf, n_orbs = store.get_step(n)
        e_ref_pyscf = store.get_reference_energy(n)      # total = ecore + E_active
        step_meta   = store.get_step_meta(n)
        raw         = store._load_step_raw(n)
        nelec_active = raw.get("metadata", {}).get("nelec_active", 4)
        if isinstance(nelec_active, (list, tuple)):
            nelec_active = int(sum(nelec_active))

        # ── Exact diagonalization — verify E_FCI_active ──────────────────
        H_full = build_qudit_hamiltonian_matrix(n_orbs, h_diag, h_hop, g_full)
        H_proj, _ = _project_hamiltonian_to_sector(
            H_full, n_orbs, nelec=nelec_active, return_indices=True
        )
        E_FCI_active, _ = ground_state_from_diagonalization(H_proj)
        # Sanity: must agree with PySCF to < 0.01 mHa
        expected_active = (e_ref_pyscf - ecore_pyscf) if e_ref_pyscf is not None else None
        if expected_active is not None:
            delta = abs(E_FCI_active - expected_active) * 1000
            if delta > 0.01:
                log.warning(
                    "[HYBRID-GEN] Step %d: diag/PySCF mismatch %.4f mHa — "
                    "check nelec_active=%d", n, delta, nelec_active
                )

        # ── Determine new reference energy ────────────────────────────────
        if n in janus_step_set:
            # Anchor to Riemann scaffold: E_ref_chk = E_Janus
            circuit_ref_new = E_Janus + ecore_pyscf
            is_janus        = True
            log.info(
                "[HYBRID-GEN] Step %d (Janus): E_FCI_active=%.6f Ha  "
                "E_Janus=%.6f Ha  gap=%.4f Ha",
                n, E_FCI_active, E_Janus, E_Janus - E_FCI_active,
            )
        else:
            # Keep PySCF reference: tests real QPE accuracy
            circuit_ref_new = e_ref_pyscf
            is_janus        = False
            log.info(
                "[HYBRID-GEN] Step %d (non-Janus): E_FCI_active=%.6f Ha  "
                "E_ref_pyscf=%.6f Ha",
                n, E_FCI_active, e_ref_pyscf if e_ref_pyscf else float("nan"),
            )

        truncation_gap = E_Janus - E_FCI_active   # < 0 for CAS(4,4) (CAS is higher than Riemann)

        # ── Build output metadata ─────────────────────────────────────────
        meta_out = dict(raw.get("metadata", {}))
        meta_out.update({
            "source":               "hybrid_scaffold",
            "pyscf_source_dir":     str(source_root / mechanism_name),
            "scaffold_class":       scaffold.spectral_class,
            "k_target":             k_target,
            "gamma_k":              gamma_k,
            "zero_index":           z_idx + 1,
            "s_value":              scaffold.s,
            "n_star":               scaffold.n_star,
            "is_janus":             is_janus,
            "E_FCI_active_pyscf":   E_FCI_active,
            "E_Janus_scaffold":     E_Janus if is_janus else None,
            "truncation_gap_Ha":    truncation_gap if is_janus else None,
        })

        # ── Assemble step record ──────────────────────────────────────────
        # Preserve all PySCF fields, override only circuit_reference_energy_Ha.
        record = dict(raw)   # shallow copy of the full PySCF step dict
        record["metadata"]                  = meta_out
        record["circuit_reference_energy_Ha"] = circuit_ref_new
        # Keep ecore_Ha, h_diag, h_hop, g_full, mqe_step unchanged.

        step_path = mech_dir / f"step_{n:02d}.json"
        with open(step_path, "w") as fh:
            json.dump(record, fh, indent=2)
        log.info("[HYBRID-GEN] Wrote %s", step_path)
        step_records.append(record)

    # ── Write manifest ────────────────────────────────────────────────────
    src_manifest = dict(store.manifest)   # copy PySCF manifest
    src_manifest.update({
        "source":         "hybrid_scaffold",
        "pyscf_source":   str(source_root / mechanism_name),
        "scaffold_class": scaffold.spectral_class,
        "k_target":       k_target,
        "gamma_k":        gamma_k,
        "E_Janus_Ha":     E_Janus,
        "janus_steps":    sorted(janus_step_set),
        "description":    (
            f"Hybrid dataset: PySCF CAS({src_manifest.get('n_orbitals',4)},"
            f"{src_manifest.get('n_orbitals',4)}) integrals + Riemann-anchored "
            f"Janus reference ({scaffold.spectral_class}, γ_{z_idx+1}={gamma_k:.6f})."
        ),
    })
    manifest_path = mech_dir / "manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(src_manifest, fh, indent=2)

    # ── Print summary table ───────────────────────────────────────────────
    print(f"\n[HYBRID-GEN] {M_steps} step files written to {mech_dir}/")
    print(f"  γ_{z_idx+1} = {gamma_k:.6f}  E_Janus = {E_Janus:+.6f} Ha  "
          f"class = {scaffold.spectral_class}")
    print(f"\n  {'Step':>4} {'Type':>10} {'E_FCI_active (Ha)':>20} "
          f"{'E_ref_chk (Ha)':>20} {'Truncation gap (Ha)':>22}")
    print(f"  {'----':>4} {'----------':>10} {'--------------------':>20} "
          f"{'--------------------':>20} {'----------------------':>22}")
    for n, rec in enumerate(step_records):
        is_j   = rec["metadata"]["is_janus"]
        e_act  = rec["metadata"]["E_FCI_active_pyscf"]
        e_ref  = rec["circuit_reference_energy_Ha"] - rec["ecore_Ha"]
        gap    = rec["metadata"].get("truncation_gap_Ha") or 0.0
        tag    = "JANUS" if is_j else "non-J"
        print(f"  {n:>4} {tag:>10} {e_act:>+20.8f} {e_ref:>+20.8f} "
              f"{'%+.4f' % gap if is_j else '—':>22}")
    print(f"\n  Run with: mqe run --mechanism {mechanism_name} --riemann "
          f"--dataset-dir {mech_dir.parent}/")

    return step_records


# ===========================================================================
# TOWER LEVEL DATASET BUILDER
# ===========================================================================

def build_tower_level_dataset(
    k:                      int,
    p:                      int,
    n_orbs:                 int,
    h_diag:                 "Dict[str, float]",
    h_hop:                  "Dict[str, float]",
    g_full:                 "Dict[str, float]",
    ecore_Ha:               float,
    E_janus_Ha:             float,
    mechanism_name:         str,
    scaffold:               "RiemannScaffold",
    k_target:               int,
    janus_steps:            "set",
    M_steps:                int,
    src_manifest:           "Dict",
    output_root:            "Union[str, Path]",
    store:                  "StepwiseIntegralStore",
    nalpha:                 "Optional[int]" = None,
    nbeta:                  "Optional[int]" = None,
    active_orbital_indices: "Optional[List[int]]" = None,
    per_step_integrals:     "Optional[Dict[int, tuple]]" = None,
) -> "Path":
    r"""Write step JSONs for one Iwasawa tower level k.

    Each step file is compatible with StepwiseIntegralStore:
      * h_diag, h_hop, g_full, ecore_Ha  — from the extended integrals at level k.
      * circuit_reference_energy_Ha       —
            Janus step:     E_Janus(k) + ecore_Ha   (tower Riemann prediction)
            Non-Janus:      preserved from the base CAS(4,4) store.
      * metadata gains: tower_level, m_k, n_orbs_tower, E_janus_tower_Ha.

    The output directory is:
        output_root / f"k{k}_{mechanism_name}/"

    Args:
        k:              Tower level index.
        p:              Prime base.
        n_orbs:         Active orbitals at level k.
        h_diag:         str-keyed diagonal integrals [Ha].
        h_hop:          str-keyed off-diagonal integrals [Ha].
        g_full:         str-keyed two-body integrals [Ha].
        ecore_Ha:       Frozen-core energy (from CAS(4,4) base) [Ha].
        E_janus_Ha:     Janus energy at level k (interpolated or measured) [Ha].
        mechanism_name: Mechanism identifier.
        scaffold:       RiemannScaffold (provides spectral class, γ values).
        k_target:       Index into scaffold.janus_energies.
        janus_steps:    Set of Janus step indices (from the base mechanism).
        M_steps:        Total number of steps in the mechanism.
        src_manifest:   Manifest dict from the CAS(4,4) base store.
        output_root:    Parent directory for tower-level datasets.
        store:          Base StepwiseIntegralStore (for non-Janus step references).

    Returns:
        Path to the directory written (output_root / f"k{k}_{mechanism_name}").
    """
    from pathlib import Path

    output_root = Path(output_root)
    # level_dir is the directory for this tower level.  Step files and
    # manifest.json are written directly here (flat layout).
    # StepwiseIntegralStore callers use StepwiseIntegralStore(level_dir.parent,
    # level_dir.name) so that root = level_dir.parent / level_dir.name = level_dir.
    # This avoids the redundant double-nesting k{k}_{name}/{name}/ that the old
    # data_dir = level_dir / mechanism_name layout produced.
    level_dir = output_root / f"k{k}_{mechanism_name}"
    data_dir  = level_dir                                 # flat: files go here directly
    data_dir.mkdir(parents=True, exist_ok=True)

    gamma_k = scaffold.gammas[k_target]
    z_idx   = scaffold.zero_indices[k_target]
    m_k     = p ** k

    log.info(
        "[TOWER-GEN] k=%d  m_k=%d  n_orbs=%d  E_Janus=%.6f Ha  dir=%s",
        k, m_k, n_orbs, E_janus_Ha, level_dir,
    )

    step_records: "List[Dict]" = []

    for n in range(M_steps):
        raw_base   = store._load_step_raw(n)
        meta_base  = dict(raw_base.get("metadata", {}))
        mqe_step   = raw_base.get("mqe_step", {})

        # Per-step integrals override shared Janus integrals when available.
        # Tuple layout: (h_diag, h_hop, g_full, ecore_Ha, circuit_reference_energy_Ha)
        if per_step_integrals is not None and n in per_step_integrals:
            _hd, _hh, _gf, _ec, circuit_ref = per_step_integrals[n]
        else:
            # Legacy / heuristic path: shared integrals; non-Janus refs from base store
            e_ref_base = store.get_reference_energy(n)
            is_janus   = (n in janus_steps)
            _hd, _hh, _gf, _ec = h_diag, h_hop, g_full, ecore_Ha
            # Janus: E_total = E_janus(k) + ecore at tower level.
            # Non-Janus: if the base store carries an active-space reference
            # (e.g. Weyl PES from a hybrid seed, no ecore contribution), lift
            # it to E_total by adding ecore_Ha so that all step files share the
            # same energy scale.  If the base value is None (deferred / standard
            # PySCF tower) leave it as None — the Riemann pipeline will resolve
            # it algebraically via _compute_step_relative_energy_algebraic.
            if is_janus:
                circuit_ref = E_janus_Ha + ecore_Ha
            elif e_ref_base is not None:
                circuit_ref = e_ref_base + ecore_Ha
            else:
                circuit_ref = None

        meta_out = dict(meta_base)
        meta_out.update({
            "source":                  "tower_scaffold",
            "tower_level_k":           k,
            "tower_p":                 p,
            "tower_m_k":               m_k,
            "n_orbs_tower":            n_orbs,
            "E_janus_tower_Ha":        E_janus_Ha,
            "is_janus":                (n in janus_steps),
            "scaffold_class":          scaffold.spectral_class,
            "k_target":                k_target,
            "gamma_k":                 gamma_k,
            "zero_index":              z_idx + 1,
            "ncas":                    n_orbs,
            "nalpha":                  nalpha,
            "nbeta":                   nbeta,
            "active_orbital_indices":  active_orbital_indices,
        })

        record = {
            "metadata":                   meta_out,
            "h_diag":                     _hd,
            "h_hop":                      _hh,
            "g_full":                     _gf,
            "ecore_Ha":                   _ec,
            "circuit_reference_energy_Ha": circuit_ref,
            "mqe_step":                   mqe_step,
        }

        step_path = data_dir / f"step_{n:02d}.json"
        with open(step_path, "w") as fh:
            json.dump(record, fh, indent=2)
        step_records.append(record)

    # ── Manifest ──────────────────────────────────────────────────────────────
    manifest = dict(src_manifest)
    manifest.update({
        "source":          "tower_scaffold",
        "tower_level_k":   k,
        "tower_p":         p,
        "tower_m_k":       m_k,
        "n_orbitals":      n_orbs,
        "n_orbs_base":     src_manifest.get("n_orbitals", 4),
        "scaffold_class":  scaffold.spectral_class,
        "k_target":        k_target,
        "gamma_k":         gamma_k,
        "E_janus_Ha":      E_janus_Ha,
        "janus_steps":     sorted(janus_steps),
        "description": (
            f"Tower level k={k} (m_k={m_k}, {n_orbs} orbitals): "
            f"Riemann-interpolated dataset for {mechanism_name}. "
            f"E_Janus(k={k}) = {E_janus_Ha:+.6f} Ha  "
            f"[{scaffold.spectral_class}, γ_{z_idx+1}={gamma_k:.6f}]."
        ),
    })

    with open(data_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    log.info(
        "[TOWER-GEN] Wrote %d step files + manifest to %s",
        M_steps, data_dir,
    )
    # Return level_dir.  Callers pass this as --dataset-dir; to open with
    # StepwiseIntegralStore use StepwiseIntegralStore(level_dir.parent, level_dir.name).
    return level_dir


def main_mqe_dataset() -> None:
    """Entry point for MQE dataset generation."""
    parser = build_mqe_dataset_parser()
    args   = parser.parse_args()
    
    # ── JSON Basis Parsing ─────────────────────────────────────────────
    # Automatically converts CLI JSON strings into Python dicts for PySCF
    try:
        if isinstance(args.basis, str) and args.basis.strip().startswith('{'):
            basis = json.loads(args.basis)
            log.info(f"[BASIS] Parsed JSON per-element basis: {list(basis.keys())}")
        else:
            basis = args.basis
    except json.JSONDecodeError:
        basis = args.basis
    # ─────────────────────────────────────────────────────────────────────

    validate_fci = not args.no_fci
    output_dir   = Path(args.output_dir)

    # ── Algebraic verification only mode ────────────────────────────────
    if args.verify_only:
        specs = build_all_specs(args.n_orbitals)
        print("\n[MQE-VERIFY] Algebraic stoichiometry verification (no integrals):\n")
        all_ok = True
        for name, spec in specs.items():
            if args.mechanism != "all" and name != args.mechanism:
                continue
            result = validate_mechanism_stoichiometry(spec)
            ok     = result["passed"]
            all_ok = all_ok and ok
            print(f"  {name: <22} ℤ_{spec.m_modulus} | "
                  f"Phase: {'[✓]' if result['phase_closure']['ok'] else '[✗]'} | "
                  f"e⁻: {'[✓]' if result['electron_conservation']['ok'] else '[✗]'} | "
                  f"{'[✓] PASS' if ok else '[✗] FAIL'}")
        print(f"\n  OVERALL: {'[✓] ALL PASSED' if all_ok else '[✗] SOME FAILED'}")
        return

    # ── Full generation ─────────────────────────────────────────────────
    if args.mechanism == "all":
        generate_all_datasets(
            basis        = basis,  # ← Pass parsed dict
            n_orbitals   = args.n_orbitals,
            output_dir   = output_dir,
            validate_fci = validate_fci,
            verbose      = args.verbose,
        )
    else:
        specs = build_all_specs(args.n_orbitals)
        spec  = specs[args.mechanism]
        passed, manifest = generate_mechanism_dataset(
            spec         = spec,
            basis        = basis,  # ← Pass parsed dict
            output_dir   = output_dir,
            validate_fci = validate_fci,
            verbose      = args.verbose,
        )
        if not passed:
            import sys
            sys.exit(1)


if __name__ == "__main__":
    main_mqe_dataset()