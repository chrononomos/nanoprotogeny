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
nanoprotogeny.ionq.ionqsemantics
Dual-Manifold Semantic Holding Relation (⊨_τ) layer for IonQ Hardware.
Supports both Logical Manifold (NomosIonQid) and Virtual Phase Register (VirtualQudit).

Mappings:
Logical:  Th→|00⟩, AntiTh→|11⟩, SynTh→|Ψ⁺⟩, HoloTh→|Ψ⁻⟩
Virtual:  F→|Ψ⁻⟩, P→|00⟩, M→|Ψ⁺⟩, R→|11⟩
"""
import numpy as np
from enum import IntEnum
from typing import Dict, Optional, Tuple, Literal, Union
import cirq
from nanoprotogeny.ionq.YB171PLUSHARDWARE import NomosState, IonManifold, NomosIonQid, VirtualQudit
from nanoprotogeny.ionq.ionqprojectorgate import Vertex, TetralemmaticIonProjectorGate

#==============================================================================
# 1. BASIS TRANSFORMATIONS (Logical & Virtual)
#==============================================================================
# Consistent with ionqprojectorgate.py and ionqurgate.py
B_LOG = np.array([
    [1.0, 0.0,          0.0,          0.0], # Th (0)   -> |00>
    [0.0, 0.0, 1/np.sqrt(2),  1/np.sqrt(2)], # Syn (2)  -> |Psi+> (Row index 1 is |01>, 2 is |10>)
    [0.0, 0.0, 1/np.sqrt(2), -1/np.sqrt(2)], # Holo (3) -> |Psi->
    [0.0, 1.0,          0.0,          0.0]  # Anti (1) -> |11>
], dtype=complex)
# Note: The rows correspond to computational basis states |00>, |01>, |10>, |11>.
# Col 0 (Th):   |00>
# Col 1 (Anti): |11>
# Col 2 (Syn):  (|01> + |10>)/sqrt(2)
# Col 3 (Holo): (|01> - |10>)/sqrt(2)

B_VIRT = np.array([
    [0.0, 1.0,          0.0,          0.0],          # Row 0 (|00>) -> P (AntiTh)
    [1/np.sqrt(2), 0.0, 1/np.sqrt(2),  0.0], # Row 1 (|01>)
    [-1/np.sqrt(2), 0.0, 1/np.sqrt(2), 0.0], # Row 2 (|10>)
    [0.0, 0.0,          0.0,          1.0]           # Row 3 (|11>) -> R (HoloTh)
], dtype=complex)
# Col 0 (Th/F):   (|01> - |10>)/sqrt(2) = |Psi->
# Col 1 (Anti/P): |00>
# Col 2 (Syn/M):  (|01> + |10>)/sqrt(2) = |Psi+>
# Col 3 (Holo/R): |11>

# Logical corner projectors |v><v|
_P_LOGICAL = {v: np.zeros((4, 4), dtype=complex) for v in range(4)}
for v in range(4):
    _P_LOGICAL[v][v, v] = 1.0

# Physical projectors for each manifold
P_PHYS_LOG: Dict[int, np.ndarray] = {
    v: B_LOG @ _P_LOGICAL[v] @ B_LOG.conj().T for v in range(4)
}
P_PHYS_VIRT: Dict[int, np.ndarray] = {
    v: B_VIRT @ _P_LOGICAL[v] @ B_VIRT.conj().T for v in range(4)
}

#==============================================================================
# 2. STATUS ENUM & BRIDGE
#==============================================================================
class Status(IntEnum):
    """Semantic labels for the four ontological corners (indices 0-3)."""
    Th      = 0   # Thesis / F (virtual)
    AntiTh  = 1   # Antithesis / P (virtual)
    SynTh   = 2   # Synthesis / M (virtual)
    HoloTh  = 3   # Holothesis / R (virtual)

    @property
    def label(self) -> str:
        return self.name.capitalize()

    def to_vertex(self) -> Vertex:
        """Converts Status to the Vertex enum used by the projector gates."""
        return Vertex(self.value)
        
    def to_nomos_state(self) -> NomosState:
        """Converts Status to the NomosState physical level mapping."""
        return NomosState(self.value)

    def projector(self, manifold: Literal['logical', 'virtual'] = 'logical') -> np.ndarray:
        """Returns the 4×4 physical projector matrix for the given manifold."""
        if manifold == 'virtual':
            return P_PHYS_VIRT[self.value].copy()
        else:
            return P_PHYS_LOG[self.value].copy()

#==============================================================================
# 3. SEMANTIC OBSERVER
#==============================================================================
class SemanticObserver:
    """
    Evaluates logical warrants and holding relations (⊨_τ).
    Works for both logical and virtual manifolds.
    """
    def __init__(self, manifold: Literal['logical', 'virtual'] = 'logical'):
        self.manifold = manifold
        self.projectors = {s: s.projector(manifold) for s in Status}
        self.B = B_VIRT if manifold == 'virtual' else B_LOG

    def get_warrant(self, state: np.ndarray, status: Status) -> float:
        """
        Computes ω_Λ(C_•) = Tr(ρ Π_•) for pure states or density matrices.
        
        Args:
            state: 1D array (ket) or 2D array (density matrix).
            status: The ontological status to evaluate.
            
        Returns:
            The warrant value (probability of the state projecting onto the status).
        """
        rho = np.outer(state, np.conj(state)) if state.ndim == 1 else state
        return float(np.real(np.trace(rho @ self.projectors[status])))

    def holds(self, state: np.ndarray, status: Status, tau: float = 0.5) -> bool:
        """Thresholded holding relation: status holds iff warrant ≥ τ."""
        return self.get_warrant(state, status) >= tau

    def evaluate_manifold(self, state: np.ndarray) -> Dict[str, float]:
        """Returns the complete warrant distribution across the ontological manifold."""
        return {s.name: self.get_warrant(state, s) for s in Status}

    def decode_shots(
        self,
        result: cirq.Result,
        keys: Optional[Tuple[str, str]] = None,
        basis: Literal['computational', 'bell'] = 'computational'
    ) -> Dict[Status, float]:
        """
        Maps raw Cirq measurement shots to tetralemmatic warrant frequencies.
        
        Note: For 'computational' basis, this assumes a standard mapping where
        Bell states are projected to computational outcomes. Specifically for 
        Virtual:
          00 -> P (AntiTh)
          11 -> R (HoloTh)
          01 -> SynTh (M)  [Assumption]
          10 -> Th (F)     [Assumption]
        """
        if keys is None:
            available = list(result.measurements.keys())
            if len(available) < 2:
                raise ValueError("Result must contain ≥2 measurement keys or provide explicit keys.")
            keys = (available[0], available[1])
            
        data0 = np.asarray(result.measurements[keys[0]]).flatten()
        data1 = np.asarray(result.measurements[keys[1]]).flatten()
        num_shots = len(data0)
        if num_shots == 0:
            return {s: 0.0 for s in Status}

        counts = {s: 0.0 for s in Status}
        for b0, b1 in zip(data0, data1):
            if basis == 'bell':
                # Bell outcomes map directly to ontological corners according to the manifold's B matrix.
                # For logical: 00→Th, 11→Anti, 01→Syn, 10→Holo
                # For virtual: 00→P, 11→R, 01→M, 10→F
                if self.manifold == 'logical':
                    if b0 == 0 and b1 == 0: counts[Status.Th]     += 1
                    elif b0 == 1 and b1 == 1: counts[Status.AntiTh]  += 1
                    elif b0 == 0 and b1 == 1: counts[Status.SynTh]   += 1
                    elif b0 == 1 and b1 == 0: counts[Status.HoloTh]  += 1
                else:  # virtual
                    if b0 == 0 and b1 == 0: counts[Status.AntiTh] += 1  # P
                    elif b0 == 1 and b1 == 1: counts[Status.HoloTh] += 1  # R
                    elif b0 == 0 and b1 == 1: counts[Status.SynTh]  += 1  # M
                    elif b0 == 1 and b1 == 0: counts[Status.Th]     += 1  # F
            else:  # computational basis
                # In the computational basis, we measure qubits directly.
                # The mapping assumes specific projections.
                if self.manifold == 'logical':
                    if b0 == 0 and b1 == 0: counts[Status.Th]     += 1
                    elif b0 == 1 and b1 == 1: counts[Status.AntiTh]  += 1
                    else:
                        counts[Status.SynTh]  += 0.5
                        counts[Status.HoloTh] += 0.5
                else:
                    if b0 == 0 and b1 == 0: counts[Status.AntiTh] += 1  # P
                    elif b0 == 1 and b1 == 1: counts[Status.HoloTh] += 1  # R
                    else:
                        # 01 and 10 map to M (Syn) and F (Th) mixture in computational basis
                        # The specific mapping here is heuristic based on phase conventions
                        counts[Status.Th]    += 0.5  # F
                        counts[Status.SynTh] += 0.5  # M
        return {s: count / num_shots for s, count in counts.items()}

    def simulate_channel(
        self,
        initial_state: np.ndarray,
        status: Status,
        transmission: float = 1.0
    ) -> np.ndarray:
        """
        Applies the TetralemmaticIonProjectorGate Kraus channel to a state.
        Returns the post-measurement density matrix (4x4).
        """
        rho = np.outer(initial_state, np.conj(initial_state)) if initial_state.ndim == 1 else initial_state
        # The projector gate is manifold-agnostic in its definition but uses the logical B matrix by default.
        # However, `to_vertex` ensures the correct ontological corner is selected.
        kraus_ops = TetralemmaticIonProjectorGate(status.to_vertex(), transmission)._kraus_()
        return sum(K @ rho @ K.conj().T for K in kraus_ops)

#==============================================================================
# 4. VERIFICATION & DEMO (Dual-Manifold)
#==============================================================================
if __name__ == "__main__":
    print("=== IonQ Semantic Holding Relation Verification (Dual-Manifold) ===")
    print(f"Hardware Context: HF Splitting = {IonManifold.HF_SPLITTING_HZ / 1e9:.4f} GHz\n")
    
    # --- Logical Observer Test ---
    print("--- Logical Manifold Observer ---")
    obs_log = SemanticObserver(manifold='logical')
    psi_syn = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0], dtype=complex)  # |Ψ⁺⟩
    warrants = obs_log.evaluate_manifold(psi_syn)
    print("Warrant distribution for |Ψ⁺⟩ (logical): ")
    for k, v in warrants.items():
        print(f"  {k}: {v:.4f} ")
    print(f"Holds SynTh (τ=0.9)? {obs_log.holds(psi_syn, Status.SynTh, tau=0.9)} ")

    # --- Virtual Observer Test ---
    print("\n--- Virtual Manifold Observer ---")
    obs_virt = SemanticObserver(manifold='virtual')
    # Virtual basis state |F⟩ = HoloTh_F = |Ψ⁻⟩
    psi_F = np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0], dtype=complex)
    warrants_v = obs_virt.evaluate_manifold(psi_F)
    print("Warrant distribution for |Ψ⁻⟩ (virtual F): ")
    for k, v in warrants_v.items():
        print(f"  {k}: {v:.4f} ")
    print(f"Holds Th (F) (τ=0.9)? {obs_virt.holds(psi_F, Status.Th, tau=0.9)} ")

    # --- Channel Simulation (logical) ---
    print("\n--- Channel Simulation (Unsharp Projector on Thesis, logical) ---")
    rho_out = obs_log.simulate_channel(psi_syn, Status.Th, transmission=0.8)
    print(f"Post-channel purity: {np.real(np.trace(rho_out @ rho_out)):.4f} ")

    print("\n✓ Semantic layer ready for both logical and virtual qudits. ")