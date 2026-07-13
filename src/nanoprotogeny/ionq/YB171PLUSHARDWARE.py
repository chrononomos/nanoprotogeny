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
nanoprotogeny.ionq.YB171PLUSHARDWARE
Refactored Physical Hardware Abstraction Layer for 171Yb+ Trapped-Ion Systems.

PHYSICAL CORRECTIONS APPLIED:
1. Virtual Manifold Mapping:
   - BEFORE: Mapped to P-state (369.5 nm / ~811 THz).
     ISSUE: P-state lifetime is ~ns, too short for phase storage.
   - AFTER:  Mapped to Metastable D-state via the S_1/2 -> D_3/2 shelving
     transition at 436 nm / ~688 THz (electric quadrupole clock transition).
     NOTE: 935.2 nm / ~320.6 THz is the D_3/2 -> [3/2]_1/2 REPUMP laser used
     to EXIT the D-state, NOT the frequency used to enter it. These are distinct.
     BENEFIT: D-state lifetime ~52 ms, suitable for coherent phase shielding.

2. Logical Manifold Clock States:
   - BEFORE: Applied linear Zeeman shift to all logical levels.
   - AFTER:  Removed linear Zeeman shift for logical levels (0-3).
     REASON: Logical states map to |F=0, mF=0> and |F=1, mF=0>, 
             which are insensitive to first-order magnetic fields.

ARCHITECTURAL SEPARATION:
- Logical Manifold (d=4): NomosIonQid. Indices 0-3 (Th, AntiTh, SynTh, HoloTh).
  Used for truth evaluation, semantic holding (⊨_τ), and algorithmic diffusion.
- Virtual Phase Register (d=4): VirtualQudit. Virtual indices 0-3 mapping to
  physical auxiliary levels 4-7 (D-state shelving manifold).
  Used for contextual routing, phase compensation, and cross-manifold algorithms.
"""
import numpy as np
import cirq
from enum import IntEnum
from typing import Dict, Tuple, Literal, Union

__all__ = [
    "NomosState", "IonManifold", "NomosIonQid", "VirtualQudit",
    "LOGICAL_LEVELS", "AUXILIARY_LEVELS", "VIRTUAL_OFFSET",
    "PHYS_TO_VIRTUAL_MAP", "VIRTUAL_TO_PHYS_MAP"
]

# ==============================================================================
# 1. BOUNDARY CONSTANTS & ROUTING MAPS
# ==============================================================================
LOGICAL_LEVELS = {0, 1, 2, 3}
AUXILIARY_LEVELS = {4, 5, 6, 7}
VIRTUAL_OFFSET = 1000  # Prevents hash/eq collision with NomosIonQid

# Explicit mapping between physical auxiliary levels and virtual phase register indices
PHYS_TO_VIRTUAL_MAP = {4: 0, 5: 1, 6: 2, 7: 3}
VIRTUAL_TO_PHYS_MAP = {0: 4, 1: 5, 2: 6, 3: 7}

# ==============================================================================
# 2. PHYSICAL STATE ENUM & PHASE REGISTER
# ==============================================================================
class NomosState(IntEnum):
    r"""Maps logical ontological corners to the ^{171}Yb+ hyperfine & auxiliary manifolds.
    Indices 0–3 constitute the Logical Manifold (d=4).
    Indices 4–7 constitute the Virtual Phase Register for routing & shielding.
    """
    Th       = 0  # Th   -> |00> (Logical)
    AntiTh   = 1  # Anti -> |11> (Logical)
    SynTh    = 2  # Syn  -> |Ψ+> (Logical)
    HoloTh   = 3  # Holo -> |Ψ-> (Logical / Base Boundary)
    
    # Virtual / Auxiliary Levels (Metastable D-State Shelving)
    HoloTh_F = 4  # F: Base Logical Boundary / U_R^0 on |Ψ->
    HoloTh_P = 5  # P: +π/2 Phase Accumulator / U_R^1
    HoloTh_M = 6  # M:  π Phase Accumulator   / U_R^2
    HoloTh_R = 7  # R: -π/2 Phase Accumulator / U_R^3

    @property
    def is_logical(self) -> bool:
        """True if index ∈ {0,1,2,3} (participates in logical gate unitaries)."""
        return self.value in LOGICAL_LEVELS

    @property
    def is_auxiliary(self) -> bool:
        """True if index ∈ {4,5,6,7} (physical routing/phase accumulation)."""
        return self.value in AUXILIARY_LEVELS

    @property
    def role(self) -> Literal['logical', 'vphase_F', 'vphase_P', 'vphase_M', 'vphase_R']:
        """Categorizes the physical function for the Holographic Routing Layer."""
        roles = {
            0: 'logical', 1: 'logical', 2: 'logical', 3: 'logical',
            4: 'vphase_F', 5: 'vphase_P', 6: 'vphase_M', 7: 'vphase_R'
        }
        return roles[self.value]

    @property
    def ur_phase(self) -> float:
        """Returns the accumulated U_R phase (in radians) for this auxiliary level."""
        phases = {4: 0.0, 5: np.pi/2, 6: np.pi, 7: -np.pi/2}
        return phases.get(self.value, 0.0)

# ==============================================================================
# 3. LOGICAL QUDIT CLASS (Computational Manifold)
# ==============================================================================
class NomosIonQid(cirq.Qid):
    """Logical d=4 qudit representing a single 171Yb+ ion in Cirq.
    Dimension is strictly 4. Levels 4–7 are physical infrastructure managed
    by the Holographic Routing Layer, not logical qudit states.
    """
    def __init__(self, index: int):
        self._index = index

    @property
    def dimension(self) -> int:
        return 4

    def _comparison_key(self) -> Tuple[int]:
        return (self._index,)

    def __repr__(self) -> str:
        return f"NomosIon({self._index})"

    def __eq__(self, other) -> bool:
        if isinstance(other, NomosIonQid):
            return self._index == other._index
        return NotImplemented

    def __hash__(self) -> int:
        return hash((type(self), self._index))

# ==============================================================================
# 4. VIRTUAL QUDIT CLASS (Phase Register)
# ==============================================================================
class VirtualQudit(cirq.Qid):
    """Virtual d=4 qudit representing the Holothesis Phase Register.
    Maps physical auxiliary levels (4-7) to a coherent algorithmic phase space.
    """
    def __init__(self, index: int):
        self._index = index

    @property
    def dimension(self) -> int:
        return 4

    def _comparison_key(self) -> Tuple[int]:
        return (self._index + VIRTUAL_OFFSET,)

    def __repr__(self) -> str:
        return f"VirtualQudit({self._index})"

    def __eq__(self, other) -> bool:
        return isinstance(other, VirtualQudit) and self._index == other._index

    def __hash__(self) -> int:
        return hash((type(self), self._index + VIRTUAL_OFFSET))

# Type alias for downstream pipeline convenience
AnyQudit = Union[NomosIonQid, VirtualQudit]

# ==============================================================================
# 5. PHYSICAL MANIFOLD CONSTANTS
# ==============================================================================
class IonManifold:
    """Physical constants and transition parameters for 171Yb+ trapped ions.

    Transition frequency reference guide for 171Yb+:
    ┌──────────────────────────────────────────────────────────────────────┐
    │ Transition              │ Wavelength │ Frequency   │ Role            │
    ├──────────────────────────────────────────────────────────────────────┤
    │ S_1/2  -> P_1/2 (cool) │  369.5 nm  │  ~811 THz   │ Doppler cooling │
    │ S_1/2  -> D_3/2 (E2)   │  436.0 nm  │  ~688 THz   │ SHELVING (in)   │
    │ D_3/2  -> [3/2]_1/2    │  935.2 nm  │  ~321 THz   │ REPUMP (out)    │
    │ S_1/2  -> F_7/2 (E3)   │  467.0 nm  │  ~642 THz   │ SHELVING (in)   │
    │ F_7/2  -> [5/2]_5/2    │  760.1 nm  │  ~394 THz   │ REPUMP (out)    │
    └──────────────────────────────────────────────────────────────────────┘

    Key distinction: shelving lasers drive the ion INTO a metastable state;
    repump lasers drive it OUT. Prior versions of this file assigned repump
    frequencies to shelving constants, which inverted the physical meaning.

    Sources:
    - HF_SPLITTING_HZ: Tamm et al. (1983); Chen et al. Opt. Lett. 50 (2025).
      Precise value: 12 642 812 118.4686(3) Hz.
    - D32_SHELVING_HZ: Peik et al. (2005), arXiv:physics/0504101.
      S_1/2(F=0) -> D_3/2(F=2) electric quadrupole transition at 436 nm / 688 THz.
    - D32_REPUMP_HZ: Olmschenk et al. (2007), arXiv:0708.0657.
      D_3/2 -> [3/2]_1/2 transition at 935.2 nm / ~321 THz.
    - F72_SHELVING_HZ: Huntemann et al. (2012), arXiv:1111.2446.
      S_1/2 -> F_7/2 electric octupole transition at 467 nm / ~642 THz.
    - F72_REPUMP_HZ: Huntemann et al. (2012).
      F_7/2 -> [5/2]_5/2 repump transition at 760 nm / ~394 THz.
    - Zeeman Insensitivity: Logical clock states map to |F=0, mF=0> and
      |F=1, mF=0>, which are insensitive to first-order magnetic fields.
    """

    # ------------------------------------------------------------------
    # Ground-state hyperfine splitting: S_1/2 |F=0,mF=0> <-> |F=1,mF=0>
    # Precise measured value: 12 642 812 118.4686(3) Hz
    # Ref: Chen et al., Opt. Lett. 50, 6024 (2025)
    # NOTE: Prior file had 12_642_821_000 — digit transposition (821 vs 812).
    # ------------------------------------------------------------------
    HF_SPLITTING_HZ = 12_642_812_118.4686

    # Zeeman sensitivity (approximate for non-clock states; zeroed for clock states below)
    G_FACTOR_MHZ_PER_G = 1.40

    # ------------------------------------------------------------------
    # D_3/2 manifold — SHELVING transition (INTO the metastable state)
    # S_1/2(F=0) -> D_3/2(F=2) electric quadrupole clock transition
    # Wavelength: ~436 nm  |  Frequency: ~688 THz
    # Ref: Peik et al., arXiv:physics/0504101 (2005)
    # ------------------------------------------------------------------
    D32_SHELVING_HZ = 688_358_979_309_000.0  # ~688 THz / 436 nm

    # ------------------------------------------------------------------
    # D_3/2 manifold — REPUMP transition (OUT of the metastable state)
    # D_3/2 -> [3/2]_1/2 transition; quickly returns ion to S_1/2 ground state
    # Wavelength: ~935.2 nm  |  Frequency: ~321 THz
    # Ref: Olmschenk et al., arXiv:0708.0657 (2007)
    # NOTE: Prior file used this repump frequency as the shelving frequency.
    # ------------------------------------------------------------------
    D32_REPUMP_HZ = 320_565_000_000_000.0   # ~321 THz / 935.2 nm

    # ------------------------------------------------------------------
    # F_7/2 manifold — SHELVING transition (INTO the metastable state)
    # S_1/2 -> F_7/2 electric octupole transition; ultra-narrow linewidth
    # Wavelength: ~467 nm  |  Frequency: ~642 THz
    # Ref: Huntemann et al., arXiv:1111.2446 (2012)
    # ------------------------------------------------------------------
    F72_SHELVING_HZ = 642_121_496_772_000.0  # ~642 THz / 467 nm

    # ------------------------------------------------------------------
    # F_7/2 manifold — REPUMP transition (OUT of the metastable state)
    # F_7/2 -> [5/2]_5/2 transition; returns ion to ground state for cooling
    # Wavelength: ~760.1 nm  |  Frequency: ~394 THz
    # Ref: Huntemann et al., arXiv:1111.2446 (2012)
    # NOTE: Prior file used this repump frequency as the shelving frequency.
    # ------------------------------------------------------------------
    F72_REPUMP_HZ = 394_700_000_000_000.0   # ~394 THz / 760.1 nm

    # ------------------------------------------------------------------
    # Virtual phase register sub-level offset within the D_3/2 manifold.
    # The D_3/2 hyperfine splitting (F=1 <-> F=2) in 171Yb+ is ~2.2 GHz.
    # The four virtual indices (4-7) are modeled as D_3/2 hyperfine
    # sub-levels and adjacent Zeeman sub-levels; a representative spacing
    # of 2.2 GHz is used here as a physically motivated placeholder.
    # This value requires experimental characterisation for precise work.
    # ------------------------------------------------------------------
    D32_HFS_OFFSET_HZ = 2_200_000_000.0  # ~2.2 GHz (D_3/2 hyperfine splitting)

    @staticmethod
    def energy_levels(B_field_gauss: float = 1.0) -> Dict[NomosState, float]:
        """Returns absolute transition frequencies (Hz) for all 8 levels.

        Logical levels 0-3 reference the S_1/2 hyperfine ground-state manifold.
        Virtual levels 4-7 reference the D_3/2 shelving manifold, entered via
        the 436 nm / ~688 THz electric quadrupole transition, with sub-level
        offsets approximated by the D_3/2 hyperfine splitting (~2.2 GHz).
        """
        # Logical states are clock states (mF=0): first-order field-insensitive.
        zeeman_hz = 0.0

        hf      = IonManifold.HF_SPLITTING_HZ
        d_state = IonManifold.D32_SHELVING_HZ
        d_hfs   = IonManifold.D32_HFS_OFFSET_HZ

        return {
            # Logical Manifold (S_1/2 hyperfine clock states)
            NomosState.Th:       0.0,
            NomosState.AntiTh:   hf,   # ~12.643 GHz above Th
            NomosState.SynTh:    hf,   # Bell-symmetric superposition; base freq same
            NomosState.HoloTh:   hf,   # Bell-antisymmetric superposition; base freq same

            # Virtual Phase Register (D_3/2 shelving manifold, entered at ~688 THz)
            # Sub-levels offset by D_3/2 hyperfine splitting (~2.2 GHz per step).
            NomosState.HoloTh_F: d_state,
            NomosState.HoloTh_P: d_state + 1 * d_hfs,
            NomosState.HoloTh_M: d_state + 2 * d_hfs,
            NomosState.HoloTh_R: d_state + 3 * d_hfs,
        }

# ==============================================================================
# VERIFICATION & USAGE DEMO
# ==============================================================================
if __name__ == "__main__":
    print("=== 171Yb+ Hardware Abstraction Layer Verification (8-Level) ===")
    print("\nNomosState Index Mapping & Routing Classification: ")
    for state in NomosState:
        phase_str = f"{state.ur_phase:+.3f}π" if state.is_auxiliary else "N/A"
        print(f"  {state.name:12} -> Index {state.value} | Logical: {state.is_logical:5} | U_R Phase: {phase_str} | Role: {state.role} ")

    B_field = 1.0
    energies = IonManifold.energy_levels(B_field)
    print(f"\nIon Manifold Energies (B = {B_field}G):  ")
    for state, freq in energies.items():
        suffix = "Hz (S_1/2 clock)" if state.is_logical else "Hz (D_3/2 shelving manifold)"
        print(f"  {state.name:12}: {freq:,.0f} {suffix} ")

    print("\nKey Transition Frequencies (for reference): ")
    print(f"  HF splitting (S_1/2 clock):    {IonManifold.HF_SPLITTING_HZ:>25,.4f} Hz  (~12.643 GHz)")
    print(f"  D_3/2 SHELVING  (436 nm, E2):  {IonManifold.D32_SHELVING_HZ:>25,.0f} Hz  (~688 THz)  [enter D-state]")
    print(f"  D_3/2 REPUMP    (935 nm):      {IonManifold.D32_REPUMP_HZ:>25,.0f} Hz  (~321 THz)  [exit  D-state]")
    print(f"  F_7/2 SHELVING  (467 nm, E3):  {IonManifold.F72_SHELVING_HZ:>25,.0f} Hz  (~642 THz)  [enter F-state]")
    print(f"  F_7/2 REPUMP    (760 nm):      {IonManifold.F72_REPUMP_HZ:>25,.0f} Hz  (~394 THz)  [exit  F-state]")
    print(f"  D_3/2 HFS offset (sub-levels): {IonManifold.D32_HFS_OFFSET_HZ:>25,.0f} Hz  (~2.2 GHz)")

    print("\nLogical vs Virtual Qudit Separation: ")
    q_log = NomosIonQid(0)
    q_virt = VirtualQudit(1)  # Maps to HoloTh_P
    print(f"  Logical:  {q_log} (dim={q_log.dimension}) | Hash: {hash(q_log)} ")
    print(f"  Virtual:  {q_virt} (dim={q_virt.dimension}) | Hash: {hash(q_virt)} ")
    print(f"  Collision Safe: {hash(q_log) != hash(q_virt)} ")

    print("\nRouting Map Verification: ")
    print(f"  Phys 5 (HoloTh_P) -> Virtual Index {PHYS_TO_VIRTUAL_MAP[5]} ")
    print(f"  Virtual Index 2   -> Phys Level {VIRTUAL_TO_PHYS_MAP[2]} ")

    print("\n✓ Hardware Abstraction Layer: frequencies corrected to peer-reviewed values.")
    print("  D-state shelving: 436 nm / ~688 THz (NOT the 935 nm repump).")
    print("  F-state shelving: 467 nm / ~642 THz (NOT the 760 nm repump).")
    print("  Ready for Holographic Routing. ")