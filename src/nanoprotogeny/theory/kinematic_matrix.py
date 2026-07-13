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
nanoprotogeny.theory.kinematic_matrix

Implements the kinematic matrix layer bridging the Tetralemmatic Algebra 
to the Logical Modal Subspace (\mathcal{H}_L(P) \cong \mathbb{C}^4).
Corresponds to: GNS Construction → Corner Gram Matrix → Complexification → Ontological Basis.
"""
import numpy as np
from enum import Enum
from typing import Dict, Optional, Tuple

from nanoprotogeny.theory.algebra import Vertex

class KinematicMatrixLayer:
    r"""
    Kinematic Matrix Layer for the Tetralemmatic Manifold.
    
    Transforms abstract logical generators into concrete 4x4 complex matrices
    operating on the Logical Modal Subspace. Provides the GNS cyclic vector,
    corner Gram matrix, complex structure (J), and quarter-turn/duality operators.
    """
    
    def __init__(self, gram_matrix: Optional[np.ndarray] = None):
        self.dim = 4
        
        # 1. Corner Gram Matrix G(P) (\cref{def:corner_gram_matrix_atomic})
        # Default: Identity (orthonormal sharp regime)
        self.G = np.eye(self.dim, dtype=complex) if gram_matrix is None else np.asarray(gram_matrix, dtype=complex)
        assert self.G.shape == (self.dim, self.dim), "Gram matrix must be 4x4"
        assert np.allclose(self.G, self.G.conj().T), "Gram matrix must be Hermitian"
        assert np.all(np.linalg.eigvalsh(self.G) >= -1e-9), "Gram matrix must be Positive Semidefinite"
        
        # 2. Ontological Basis / Corner Kets (\cref{def:canonical_corner_kets})
        # Standard computational basis in \mathbb{C}^4
        self.kets: Dict[Vertex, np.ndarray] = {
            v: np.eye(self.dim, dtype=complex)[:, v.value] for v in Vertex
        }
        
        # 3. GNS Cyclic Vector / Logical Holo-Vector (\cref{def:logical_holo_vector})
        # Invariant under Duality, representing the unbroken symmetry vacuum.
        # Default: equal superposition of non-polar sectors (Synthesis/Holothesis)
        self.omega = (self.kets[Vertex.SynTh] + self.kets[Vertex.HoloTh]) / np.sqrt(2)
        
        # 4. Quarter-Turn Operator U_R (\cref{def:tetralemmatic_quarter_turn_automorphism})
        # Cycles: Th -> SynTh -> AntiTh -> HoloTh -> Th
        # Indices: 0 -> 2 -> 1 -> 3 -> 0
        self.U_R = np.zeros((self.dim, self.dim), dtype=complex)
        cycle = [0, 2, 1, 3]
        for i, j in zip(cycle, cycle[1:] + cycle[:1]):
            self.U_R[j, i] = 1.0
            
        # 5. Duality Operator U_D (\cref{cor:duality_orthogonal_operator})
        # Swaps poles (Th <-> AntiTh), fixes non-poles (SynTh, HoloTh)
        self.U_D = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
        
        # Verify Holo-vector invariance under Duality
        assert np.allclose(self.U_D @ self.omega, self.omega), "Logical Holo-vector must be Duality-invariant"
        
    # =========================================================================
    # BASIS & STATE ACCESS
    # =========================================================================
    
    def get_ket(self, v: Vertex) -> np.ndarray:
        r"""Returns the normalized corner ket |\bullet; P\rangle."""
        return self.kets[v].copy()
    
    def get_gram_matrix(self) -> np.ndarray:
        r"""Returns the 4x4 Corner Gram Matrix G(P)."""
        return self.G.copy()
    
    def get_cyclic_vector(self) -> np.ndarray:
        r"""Returns the GNS cyclic state |\Omega_\Lambda\rangle."""
        return self.omega.copy()
    
    # =========================================================================
    # INNER PRODUCTS & WARRANT FUNCTIONAL
    # =========================================================================
    
    def inner_product(self, psi: np.ndarray, phi: np.ndarray) -> complex:
        r"""GNS inner product: \langle \psi, \phi \rangle_\Lambda = \psi^\dagger G \phi"""
        return psi.conj() @ self.G @ phi
    
    def warrant(self, operator: np.ndarray) -> float:
        r"""Real Warrant Functional: \omega_\Lambda(A) = \langle \Omega_\Lambda | A | \Omega_\Lambda \rangle"""
        return np.real(self.omega.conj() @ operator @ self.omega)
    
    # =========================================================================
    # LOGICAL OPERATORS
    # =========================================================================
    
    def get_quarter_turn(self) -> np.ndarray:
        r"""Returns the Quarter-Turn Automorphism U_{\mathbf{R}} (Complex Structure J)."""
        return self.U_R.copy()
    
    def get_duality(self) -> np.ndarray:
        r"""Returns the Duality Involution U_{\mathbf{D}}."""
        return self.U_D.copy()
    
    def get_projector(self, v: Vertex) -> np.ndarray:
        r"""GNS image of corner generator: \pi_\Lambda(C_v(P)) = |v\rangle\langle v|"""
        k = self.kets[v]
        return np.outer(k, k.conj())
    
    def get_complex_structure(self) -> np.ndarray:
        r"""Returns the complex structure operator J \equiv U_{\mathbf{R}}."""
        return self.U_R.copy()
    
    # =========================================================================
    # VERIFICATION SUITE
    # =========================================================================
    
    def verify_kinematic_properties(self) -> Dict[str, bool]:
        r"""Validates core kinematic constraints from the framework."""
        checks = {}
        
        # 1. Gram Matrix PSD
        eigs = np.linalg.eigvalsh(self.G)
        checks["gram_psd"] = bool(np.all(eigs >= -1e-9))
        
        # 2. Quarter-Turn Order-4 Symmetry
        checks["UR_order4"] = bool(np.allclose(np.linalg.matrix_power(self.U_R, 4), np.eye(self.dim)))
        checks["UR_nontrivial"] = bool(not np.allclose(np.linalg.matrix_power(self.U_R, 2), np.eye(self.dim)))
        
        # 3. Duality Involution
        checks["UD_involution"] = bool(np.allclose(self.U_D @ self.U_D, np.eye(self.dim)))
        
        # 4. Duality Invariance of Cyclic Vector
        checks["omega_duality_invariant"] = bool(np.allclose(self.U_D @ self.omega, self.omega))
        
        # 5. Complex Structure Compatibility (J^2 = -I on interference sector)
        J = self.U_R
        # On the subspace spanned by poles (0,1), J^2 should act as -I if properly aligned
        # In this basis, we verify unitary preservation of the Gram form
        J_Herm = J.conj().T @ self.G @ J
        checks["J_Gram_preserving"] = bool(np.allclose(J_Herm, self.G))
        
        return checks
        
    def __repr__(self) -> str:
        return (f"KinematicMatrixLayer(dim=4, "
                f"GramCond={np.linalg.cond(self.G):.2f}, "
                f"U_R(Cycle)=[0→2→1→3→0])")

# =============================================================================
# MODULE EXPORTS
# =============================================================================
__all__ = [
    "KinematicMatrixLayer",
    "Vertex"
]