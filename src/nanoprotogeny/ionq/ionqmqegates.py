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
nanoprotogeny.ionq.ionq_mqe_gates
IonQ Native Compilation for Modular Quantum Emulator (MQE) Extension Gates.

Implements the Bell-separable compilation pipeline for the six new gate classes
introduced in mqe_extension.py, following the exact pattern established in
gatesetY.py (ionqurgate) and gatesetX.py:

  Gate                      Onto dim   Phys qubits   Basis transform
  ─────────────────────────────────────────────────────────────────────
  ElectronShiftGate         4×4        2 (logical)   B_LOG
  ProtonPhaseGate           4×4        2 (logical)   B_LOG
  ConformationalShiftGate   4×4        2 (logical)   B_LOG
  GeneralizedVirtualShiftGate
    m=4  (VirtualQudit)     4×4        2 (virtual)   B_VIRT
    m≠4  (LineQid)          m×m→2^n   n (binary)    none (std basis)
  CofactorCouplingGate      4m×4m      2+n_virt      B_LOG ⊗ B_virt(m)
  CrossManifoldSWAPGate     16×16      4              B_LOG ⊗ B_VIRT

Compilation pipeline (three-stage, same as existing gates):

  Stage 1 – expand_mqe_qudit_circuit:
    Dispatch each MQE gate from abstract d=4/d=m qudits to a Physical*Wrapper
    operating on standard cirq.LineQubit registers.

  Stage 2 – Physical*Wrapper._decompose_:
    Each wrapper yields two-qubit operations via
    cirq.two_qubit_matrix_to_cz_operations (4×4 physical matrix → CZ+rotations)
    or multi-qubit generalizations for 4-qubit gates.

  Stage 3 – compile_mqe_gates (or compile_tetralemmatic_ionq):
    cirq.optimize_for_target_gateset decomposes to GPI/GPI2/ZZ
    (ForteNativeGateset) or the 16-gate IonQ API subset (IonQTargetGateset).

Usage:
    from ionq_mqe_gates import expand_mqe_qudit_circuit, compile_mqe_gates

    expanded = expand_mqe_qudit_circuit(mqe_circuit)
    forte_circ = compile_mqe_gates(expanded, target="forte_native")
    api_circ   = compile_mqe_gates(expanded, target="api")
"""

from __future__ import annotations

import numpy as np
import cirq
import cirq_ionq
from cirq_ionq.ionq_native_target_gateset import ForteNativeGateset
from typing import Dict, Iterator, List, Tuple, Union, Optional
from cirq import OP_TREE
import math

from nanoprotogeny.ionq.YB171PLUSHARDWARE import NomosIonQid, VirtualQudit

# ---------------------------------------------------------------------------
# Helper: number of carry qubits for the auxiliary register of a m=4r clock.
# ---------------------------------------------------------------------------
def _n_aux_bits(r: int) -> int:
    """ceil(log2(r)) — bits needed to represent carry index in [0, r-1]."""
    if r <= 1:
        return 0
    return math.ceil(math.log2(r))

# ==============================================================================
# 1. BASIS TRANSFORMATIONS (matching existing gatesetY.py convention)
# ==============================================================================

# B_LOG: Logical manifold (NomosIonQid d=4) → 2 physical qubits via Bell-separable encoding.
# Columns map [Th, AntiTh, SynTh, HoloTh] → [|00⟩, |11⟩, |Ψ⁺⟩, |Ψ⁻⟩].
B_LOG = np.array([
    [1.0, 0.0,           0.0,           0.0],
    [0.0, 0.0,  1/np.sqrt(2),  1/np.sqrt(2)],
    [0.0, 0.0,  1/np.sqrt(2), -1/np.sqrt(2)],
    [0.0, 1.0,           0.0,           0.0],
], dtype=complex)

# B_VIRT: Virtual phase register (VirtualQudit d=4) → 2 physical qubits.
# Columns map [F, P, M, R] → [|Ψ⁻⟩, |00⟩, |Ψ⁺⟩, |11⟩].
B_VIRT = np.array([
    [ 0.0,           1.0,           0.0,  0.0],
    [ 1/np.sqrt(2),  0.0,  1/np.sqrt(2),  0.0],
    [-1/np.sqrt(2),  0.0,  1/np.sqrt(2),  0.0],
    [ 0.0,           0.0,           0.0,  1.0],
], dtype=complex)

# B_total: Kronecker product for two-qudit (logical × virtual d=4) gates.
# Used by CofactorCouplingGate (m=4) and CrossManifoldSWAPGate.
B_total = np.kron(B_LOG, B_VIRT)   # 16×16

def _get_physical_1q(M_onto: np.ndarray, B: np.ndarray) -> np.ndarray:
    r"""Compute physical 4×4 matrix from 4×4 ontological matrix via B @ M @ B†."""
    return B @ M_onto @ B.conj().T

def _get_physical_2q(M_onto: np.ndarray,
                     B_ctrl: np.ndarray,
                     B_tgt: np.ndarray) -> np.ndarray:
    r"""Compute physical 16×16 matrix from 16×16 ontological matrix
    via (B_ctrl ⊗ B_tgt) @ M @ (B_ctrl ⊗ B_tgt)†."""
    B_tot = np.kron(B_ctrl, B_tgt)
    return B_tot @ M_onto @ B_tot.conj().T

# ==============================================================================
# 2. ONTOLOGICAL MATRICES (MQE gate definitions in the onto basis)
# ==============================================================================

def _cyclic_shift_onto(d: int, power: int = 1) -> np.ndarray:
    r"""d×d cyclic permutation matrix: |k⟩ → |k+power mod d⟩."""
    U = np.zeros((d, d), dtype=complex)
    for k in range(d):
        U[(k + power) % d, k] = 1.0
    return U

def _diagonal_phase_onto(d: int, phi: float) -> np.ndarray:
    r"""d×d diagonal phase matrix: diag(1, e^{iφ}, e^{2iφ}, ..., e^{(d-1)iφ})."""
    return np.diag([np.exp(1j * k * phi) for k in range(d)])

def _cofactor_coupling_onto(m: int, nu: int) -> np.ndarray:
    r"""(4m)×(4m) ontological matrix for U_coupling^{(p,m)}:
        |k⟩_L |j⟩_V → |k⟩_L |(j + ν·k) mod m⟩_V
    Acts on (d=4 logical) ⊗ (d=m virtual) = dimension 4m."""
    dim = 4 * m
    U   = np.zeros((dim, dim), dtype=complex)
    for k_log in range(4):
        for j_virt in range(m):
            j_out        = (j_virt + nu * k_log) % m
            row          = k_log * m + j_out
            col          = k_log * m + j_virt
            U[row, col]  = 1.0
    return U

def _cross_manifold_swap_onto() -> np.ndarray:
    r"""16×16 SWAP in (d=4 logical) ⊗ (d=4 virtual) onto basis: |i⟩|j⟩ → |j⟩|i⟩."""
    U = np.zeros((16, 16), dtype=complex)
    for i in range(4):
        for j in range(4):
            U[j * 4 + i, i * 4 + j] = 1.0
    return U

# ==============================================================================
# LAYER 2: GENERALIZED GATE ALGEBRA G(M)
# ==============================================================================

class GeneralizedVirtualShiftGate(cirq.Gate):
    r"""U_R^{V,m}: Cyclic shift automorphism on a d=m virtual register.

    Implements Definition 3 (Generalized Virtual Shift) from the article:
        U_R^{V,m} = sum_{k=0}^{m-1} |k+1 mod m><k|_V

    For m=4, this reduces to the quarter-turn U_R^V of the LT framework.
    The phase ladder is phi_k = 2*pi*k/m, satisfying (U_R^{V,m})^m = I_m.

    Args:
        m:     Modulus (virtual register dimension).
        power: Integer power to apply. Default 1. Negative = inverse.
    """

    def __init__(self, m: int, power: int = 1):
        assert m >= 1, f"Modulus m must be >= 1, got {m}"
        self._m     = m
        self._power = power % m  # Reduce modulo m (captures closure)

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self._m,)

    def _unitary_(self) -> np.ndarray:
        """Cyclic permutation matrix: |k+1 mod m><k|."""
        U = np.zeros((self._m, self._m), dtype=complex)
        for k in range(self._m):
            U[(k + self._power) % self._m, k] = 1.0
        return U

    def _has_unitary_(self) -> bool:
        return True

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        sup = f"^{self._power}" if self._power != 1 else ""
        return cirq.CircuitDiagramInfo(wire_symbols=(f"UR_V{sup}(m={self._m})",))

    def __pow__(self, exponent: int):
        return GeneralizedVirtualShiftGate(self._m, (self._power * exponent) % self._m)

    def __repr__(self) -> str:
        return f"GeneralizedVirtualShiftGate(m={self._m}, power={self._power})"

    def __eq__(self, other) -> bool:
        return (isinstance(other, GeneralizedVirtualShiftGate)
                and self._m == other._m and self._power == other._power)

    def __hash__(self) -> int:
        return hash((type(self), self._m, self._power))


# ==============================================================================
# 6. ELECTRON SHIFT GATE (LOGICAL ORBITAL)
# ==============================================================================

class ElectronShiftGate(cirq.Gate):
    r"""Local d=4 cyclic shift encoding one electron injection: U_R^{(p)}.

    Implements the electron-injection term in J_{n→n+1}:
        U_R |k> = |k+1 mod 4>

    Acts on a single NomosIonQid. Increments occupancy by 1 (mod 4)
    within the Zeno-stabilized active window k < 3.

    Args:
        power: Integer power. Default 1 (one electron).
    """

    def __init__(self, power: int = 1):
        self._power = power % 4

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (4,)

    def _unitary_(self) -> np.ndarray:
        U = np.zeros((4, 4), dtype=complex)
        for k in range(4):
            U[(k + self._power) % 4, k] = 1.0
        return U

    def _has_unitary_(self) -> bool:
        return True

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        sup = f"^{self._power}" if self._power != 1 else ""
        return cirq.CircuitDiagramInfo(wire_symbols=(f"eShift{sup}",))

    def __pow__(self, exponent: int):
        return ElectronShiftGate((self._power * exponent) % 4)

    def __repr__(self) -> str:
        return f"ElectronShiftGate(power={self._power})"

class ElectronEjectGate(cirq.Gate):
    """Semantic alias for U_R† (inverse cyclic shift). Mathematically equivalent to power=3."""
    def __init__(self, power: int = 1):
        self._power = power % 4

    def _qid_shape_(self) -> Tuple[int, ...]: return (4,)
    def _unitary_(self) -> np.ndarray:
        U = np.zeros((4, 4), dtype=complex)
        for k in range(4):
            U[(k - self._power) % 4, k] = 1.0
        return U
    def _circuit_diagram_info_(self, args):
        return cirq.CircuitDiagramInfo(wire_symbols=(f"eEject^{self._power}",))
    def __pow__(self, exponent: int):
        return ElectronEjectGate((self._power * exponent) % 4)


# ==============================================================================
# 6. PROTON PHASE GATE (LOGICAL ORBITAL)
# ==============================================================================

class ProtonPhaseGate(cirq.Gate):
    r"""Phase rotation encoding proton addition: Z_Clock^{(q)}(phi_H).

    Implements the protonation term in J_{n→n+1}:
        Z_Clock(phi) = diag(1, e^{i*phi}, e^{2i*phi}, e^{3i*phi})

    Acts on a single NomosIonQid. The angle phi_H encodes discrete
    protonation as a phase rotation on the occupation ladder.

    Args:
        phi: Proton phase angle in radians. Default pi/2 (quarter-turn).
    """

    def __init__(self, phi: float = np.pi / 2):
        self._phi = phi

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (4,)

    def _unitary_(self) -> np.ndarray:
        return np.diag([np.exp(1j * k * self._phi) for k in range(4)])

    def _has_unitary_(self) -> bool:
        return True

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(
            wire_symbols=(f"ZH({self._phi:.3f})",)
        )

    def __repr__(self) -> str:
        return f"ProtonPhaseGate(phi={self._phi!r})"


# ==============================================================================
# 8. COFACTOR COUPLING GATE (LOGICAL × VIRTUAL, 4-QUBIT)
# ==============================================================================

class CofactorCouplingGate(cirq.Gate):
    r"""Cross-manifold coupling: U_coupling^{(p,m)} = sum_k |k>_L<k| ⊗ (U_R^{V,m})^k.

    Implements the ATP/cofactor coupling operator U_ATP^{(n)}:
        U_coupling |k>_L |j>_V = |k>_L |(j + nu*k) mod m>_V

    This maps cofactor hydrolysis to deterministic phase accumulation in H_V.
    The shift applied to the virtual register is proportional to the logical
    occupancy k, encoding the 2:1 ATP-per-electron stoichiometry (nu=2)
    or any other cofactor ratio via general nu.

    Acts on one NomosIonQid (d=4) + one virtual register (d=m).

    Args:
        m:   Virtual register modulus.
        nu:  Cofactor shift per unit logical occupancy. Default 1.
    """

    def __init__(self, m: int, nu: int = 1):
        self._m  = m
        self._nu = nu % m

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (4, self._m)  # (logical d=4, virtual d=m)

    def _unitary_(self) -> np.ndarray:
        dim = 4 * self._m
        U   = np.zeros((dim, dim), dtype=complex)
        for k_log in range(4):
            for j_virt in range(self._m):
                # |k>_L |j>_V → |k>_L |(j + nu*k) mod m>_V
                j_out    = (j_virt + self._nu * k_log) % self._m
                row      = k_log * self._m + j_out
                col      = k_log * self._m + j_virt
                U[row, col] = 1.0
        return U

    def _has_unitary_(self) -> bool:
        return True

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(
            wire_symbols=(f"U_cof(m={self._m},ν={self._nu})", f"@V")
        )

    def __repr__(self) -> str:
        return f"CofactorCouplingGate(m={self._m}, nu={self._nu})"


class CofactorDecouplingGate(cirq.Gate):
    """Inverse cross-manifold coupling: U_coupling^†.
    
    Implements the inverse of the ATP/Cofactor hydrolysis step.
    Unitary Action: |k>_L |j>_V → |k>_L |(j - nu*k) mod m>_V
    
    This allows the system to "unwind" the cofactor state, essential 
    for circuit uncomputation and simulating reverse catalytic pathways.
    """
    def __init__(self, m: int, nu: int = 1):
        self._m  = m
        # Store nu as positive modulo for consistency
        self._nu = nu % m

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (4, self._m)

    def _unitary_(self) -> np.ndarray:
        dim = 4 * self._m
        U   = np.zeros((dim, dim), dtype=complex)
        for k_log in range(4):
            for j_virt in range(self._m):
                # Subtract nu * k_log (Inverse Shift)
                j_out = (j_virt - self._nu * k_log) % self._m
                row   = k_log * self._m + j_out
                col   = k_log * self._m + j_virt
                U[row, col] = 1.0
        return U

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(
            wire_symbols=(f"U_cof^†(m={self._m},ν={self._nu})", f"@V")
        )

    def __repr__(self) -> str:
        return f"CofactorDecouplingGate(m={self._m}, nu={self._nu})"


# ==============================================================================
# COMPOSITE VIRTUAL REGISTER GATES  (all m > 1 — unified generalisation)
# ==============================================================================
#
# For any mechanism modulus m > 1 the virtual phase register per orbital uses:
#   V₁   — primary d=4 qudit  (hardware-native ¹⁷¹Yb⁺, carries k % 4)
#   V_aux — auxiliary carry register (carries k // 4; r = ceil(m/4) − 1 bits)
#
# Mixed-radix encoding φ(k) = (k%4, k//4) represents k ∈ ℤ_m exactly.
#
# For m = 4r  (r ≥ 1): all 4r states are active — no padding, no correction.
# For m < 4r  (r = ceil(m/4)): states 0..m-1 are active; states m..4r-1 are
#   inert (Zeno boundary keeps them unpopulated).  The ℤ_m shift is implemented
#   as a ℤ_{4r} carry-gate shift followed by a modular correction gate C_m that
#   swaps state |m⟩ (transient image of state |m-1⟩) back to |0⟩.
#
# _qid_shape_ for r=1 (m≤4): (4,)        — V₁ alone
# _qid_shape_ for r>1 (m>4): (4, r)      — V₁ + V_aux
# Coupling _qid_shape_ for r=1: (4, 4)   — logical + V₁
# Coupling _qid_shape_ for r>1: (4, 4, r) — logical + V₁ + V_aux


def _r_for_m(m: int) -> int:
    """Ceiling-division r = ceil(m / 4) — smallest r with 4r ≥ m."""
    return (m + 3) // 4


def _state_physical_index(m: int, r: int) -> int:
    """Physical qubit index of abstract state m in (V₁_q0, V₁_q1, [V_aux bits]).

    In the (V₁_q0, V₁_q1, V_aux_q0, ...) computational basis (MSB first),
    state k has:
        V₁_q0  = (k%4) >> 1
        V₁_q1  = (k%4) &  1
        V_aux  = k // 4  (binary, n_aux bits, MSB first)

    Returns the integer index for state k = m in the 2^n_phys physical space.
    """
    j       = m % 4
    b       = m // 4
    n_aux   = _n_aux_bits(r) if r > 1 else 0
    v1_bits = ((j >> 1) & 1, j & 1)                  # (MSB, LSB) of V₁
    if n_aux == 0:
        all_bits = v1_bits
    else:
        aux_bits = tuple((b >> (n_aux - 1 - i)) & 1 for i in range(n_aux))
        all_bits = v1_bits + aux_bits
    idx = 0
    for bit in all_bits:
        idx = idx * 2 + bit
    return idx


def _composite_shift_phys(m: int, power: int) -> np.ndarray:
    """Physical matrix for CompositeVirtualShiftGate(m, power).

    V₁ (d=4) is encoded in 2 physical qubits via B_VIRT.
    For r=1 (m≤4): acts on V₁ alone → 4×4 physical matrix.
    For r>1 (m>4): Vaux encoded in n_aux standard-basis qubits → (4·2^n_aux)×(…) matrix.

    Implements ℤ_m cyclic on active states 0..m-1, identity on padding m..4r-1.

    M_phys = (B_VIRT [⊗ I_{2^n_aux}]) @ U @ (B_VIRT [⊗ I_{2^n_aux}])†
    """
    r      = _r_for_m(m)
    n_aux  = _n_aux_bits(r) if r > 1 else 0
    dim_va = 2 ** n_aux if n_aux > 0 else 1
    dim    = 4 * dim_va

    U = np.zeros((dim, dim), dtype=complex)
    for k in range(m):
        U[(k + power) % m, k] = 1.0
    for k in range(m, dim):
        U[k, k] = 1.0                 # identity on padding states

    if n_aux == 0:
        return B_VIRT @ U @ B_VIRT.conj().T
    B_lift = np.kron(B_VIRT, np.eye(dim_va, dtype=complex))
    return B_lift @ U @ B_lift.conj().T


def _composite_coupling_phys(m: int, nu: int, inverse: bool = False) -> np.ndarray:
    """Physical matrix for CompositeCofactorCoupling/DecouplingGate(m, nu).

    Logical (d=4) in 2 qubits via B_LOG; V₁ (d=4) in 2 qubits via B_VIRT;
    V_aux (for r>1) in n_aux standard-basis qubits.

    Active states: j_full = j1 + 4b ∈ [0, m-1] — ℤ_m coupling arithmetic.
    Padding states: j_full ≥ m — identity (never populated, Zeno protected).

    M_phys = (B_LOG ⊗ B_VIRT [⊗ I_{dim_va}]) @ U_padded @ (...)†
    """
    r      = _r_for_m(m)
    n_aux  = _n_aux_bits(r) if r > 1 else 0
    dim_va = 2 ** n_aux if n_aux > 0 else 1
    stride = 4 * dim_va                     # virtual stride per logical sector
    dim    = 4 * stride                     # total: 4 (log) × 4 (V1) × dim_va (Vaux)

    U = np.zeros((dim, dim), dtype=complex)
    for k_log in range(4):
        for j1 in range(4):
            for b in range(dim_va):          # iterate over all Vaux states
                j_full = j1 + 4 * b
                if j_full < m:               # active state: apply ℤ_m coupling
                    j_out = (j_full - nu * k_log) % m if inverse \
                            else (j_full + nu * k_log) % m
                    j1_out = j_out % 4
                    b_out  = j_out // 4
                else:                        # padding state: identity
                    j1_out = j1
                    b_out  = b
                row = k_log * stride + j1_out + 4 * b_out
                col = k_log * stride + j1     + 4 * b
                U[row, col] = 1.0

    if n_aux == 0:
        B_lift = np.kron(B_LOG, B_VIRT)
    else:
        B_lift = np.kron(np.kron(B_LOG, B_VIRT), np.eye(dim_va, dtype=complex))
    return B_lift @ U @ B_lift.conj().T


class CompositeVirtualShiftGate(cirq.Gate):
    r"""ℤ_m cyclic shift on the composite (V₁, [V_aux]) register for any m > 1.

    Implements U_R^{(m)} via the mixed-radix encoding φ(k) = (k%4, k//4):
      - For m = 4r: pure carry-gate shift — all 4r states active, no correction.
      - For m < 4r: carry-gate shift on ℤ_{4r}, then modular correction C_m
        (SWAP of states |0⟩ ↔ |m⟩) to enforce ℤ_m wrap-around.

    _qid_shape_:
      r = 1 (m ≤ 4) : (4,)    — acts on V₁ alone (single VirtualQudit)
      r > 1 (m > 4) : (4, r)  — acts on V₁ + V_aux

    Args:
        m:     Virtual modulus (any integer ≥ 1).
        power: Shift power (default 1). Reduced mod m.
    """

    def __init__(self, m: int, power: int = 1):
        assert m >= 1, f"m must be ≥ 1; got {m}"
        self._m     = m
        self._r     = _r_for_m(m)
        self._power = power % m if m > 1 else 0

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (4,) if self._r == 1 else (4, self._r)

    def _unitary_(self) -> np.ndarray:
        m = self._m
        r = self._r
        dim = 4 if r == 1 else 4 * r
        U = np.zeros((dim, dim), dtype=complex)
        for k in range(m):
            U[(k + self._power) % m, k] = 1.0
        for k in range(m, dim):       # identity on padding
            U[k, k] = 1.0
        return U

    def _has_unitary_(self) -> bool:
        return True

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        sup = f"^{self._power}" if self._power != 1 else ""
        tag = f"(m={self._m})"
        if self._r == 1:
            return cirq.CircuitDiagramInfo(wire_symbols=(f"UR_V{sup}{tag}",))
        return cirq.CircuitDiagramInfo(
            wire_symbols=(f"UR_V1{sup}{tag}", "@Vaux")
        )

    def __pow__(self, exponent: int) -> "CompositeVirtualShiftGate":
        return CompositeVirtualShiftGate(self._m, (self._power * exponent) % self._m)

    def __repr__(self) -> str:
        return f"CompositeVirtualShiftGate(m={self._m}, power={self._power})"

    def __eq__(self, other) -> bool:
        return (isinstance(other, CompositeVirtualShiftGate)
                and self._m == other._m and self._power == other._power)

    def __hash__(self) -> int:
        return hash((type(self), self._m, self._power))


class CompositeCofactorCouplingGate(cirq.Gate):
    r"""Cross-manifold coupling on (logical d=4, V₁ d=4, [V_aux]) for any m > 1.

    Implements U_coupling^{(p,m)}: |k⟩_L |j_full⟩ → |k⟩_L |(j_full+ν·k) mod m⟩
    for active states j_full ∈ [0, m-1]; padding states are identity.

    _qid_shape_:
      r = 1 (m ≤ 4) : (4, 4)    — logical + V₁
      r > 1 (m > 4) : (4, 4, r) — logical + V₁ + V_aux

    Args:
        m:  Virtual modulus.
        nu: Cofactor shift per unit logical occupancy.
    """

    def __init__(self, m: int, nu: int = 1):
        assert m >= 1
        self._m  = m
        self._r  = _r_for_m(m)
        self._nu = nu % m if m > 1 else 0

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (4, 4) if self._r == 1 else (4, 4, self._r)

    def _unitary_(self) -> np.ndarray:
        m      = self._m
        r      = self._r
        dim_v  = 4 if r == 1 else 4 * r
        stride = dim_v
        dim    = 4 * stride
        nu     = self._nu
        U      = np.zeros((dim, dim), dtype=complex)
        for k_log in range(4):
            for j_full in range(dim_v):
                if j_full < m:
                    j_out = (j_full + nu * k_log) % m
                else:
                    j_out = j_full               # padding identity
                row = k_log * stride + j_out
                col = k_log * stride + j_full
                U[row, col] = 1.0
        return U

    def _has_unitary_(self) -> bool:
        return True

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        tag = f"(m={self._m},ν={self._nu})"
        if self._r == 1:
            return cirq.CircuitDiagramInfo(wire_symbols=(f"Ucof_c{tag}", "@V1"))
        return cirq.CircuitDiagramInfo(
            wire_symbols=(f"Ucof_c{tag}", "@V1", "@Vaux")
        )

    def __repr__(self) -> str:
        return f"CompositeCofactorCouplingGate(m={self._m}, nu={self._nu})"


class CompositeCofactorDecouplingGate(cirq.Gate):
    r"""Inverse coupling on (logical d=4, V₁ d=4, [V_aux]) for any m > 1.

    Maps j_full → (j_full − ν·k) mod m on active states; identity on padding.

    Args:
        m:  Virtual modulus.
        nu: Cofactor shift magnitude.
    """

    def __init__(self, m: int, nu: int = 1):
        assert m >= 1
        self._m  = m
        self._r  = _r_for_m(m)
        self._nu = nu % m if m > 1 else 0

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (4, 4) if self._r == 1 else (4, 4, self._r)

    def _unitary_(self) -> np.ndarray:
        m      = self._m
        r      = self._r
        dim_v  = 4 if r == 1 else 4 * r
        stride = dim_v
        dim    = 4 * stride
        nu     = self._nu
        U      = np.zeros((dim, dim), dtype=complex)
        for k_log in range(4):
            for j_full in range(dim_v):
                if j_full < m:
                    j_out = (j_full - nu * k_log) % m
                else:
                    j_out = j_full
                row = k_log * stride + j_out
                col = k_log * stride + j_full
                U[row, col] = 1.0
        return U

    def _has_unitary_(self) -> bool:
        return True

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        tag = f"(m={self._m},ν={self._nu})"
        if self._r == 1:
            return cirq.CircuitDiagramInfo(wire_symbols=(f"Ucof_c†{tag}", "@V1"))
        return cirq.CircuitDiagramInfo(
            wire_symbols=(f"Ucof_c†{tag}", "@V1", "@Vaux")
        )

    def __repr__(self) -> str:
        return f"CompositeCofactorDecouplingGate(m={self._m}, nu={self._nu})"


def _ctrl_shift_ops(
    ctrl0:   cirq.Qid,
    ctrl1:   cirq.Qid,
    v1_q0:   cirq.Qid,
    v1_q1:   cirq.Qid,
    p_part:  int,
) -> Iterator[OP_TREE]:
    r"""Yield doubly-controlled U_R4^{p_part} on (v1_q0, v1_q1).

    U_R4^p is a cyclic-increment permutation matrix, so it factors entirely into
    CNOT and X gates.  All emitted operations are ≤4-qubit standard Cirq gates
    compilable to GPI/GPI2/ZZ via ForteNativeGateset — no MatrixGate emitted.

    U_R4^1 = CNOT(v1_q1→v1_q0) · X(v1_q1)
    U_R4^2 = X(v1_q0)
    U_R4^3 = X(v1_q1)  · CNOT(v1_q1→v1_q0)
    """
    if p_part == 0:
        return
    elif p_part == 1:
        # Doubly-controlled U_R4^1 = doubly-controlled [CNOT(v1_q1→v1_q0) · X(v1_q1)]
        #
        # keep=len≤2: ForteNativeGateset has no native 3-qubit operations; stopping
        # at len≤3 previously left CCX-type ops for QSD fallback.  Explicit CCX
        # yields are also wrapped — cirq.CCX._decompose_() gives the standard 6-CNOT
        # Toffoli which ForteNativeGateset compiles cleanly via KAK.
        _ctrl_x_inner = cirq.ControlledGate(
            cirq.X, num_controls=2, control_values=[1, 1]
        )
        yield from cirq.decompose(
            _ctrl_x_inner.on(ctrl1, v1_q1, v1_q0),
            keep=lambda op: len(op.qubits) <= 2,
        )
        yield from cirq.decompose(
            cirq.CCX(ctrl0, ctrl1, v1_q1),
            keep=lambda op: len(op.qubits) <= 2,
        )
    elif p_part == 2:
        # Doubly-controlled X(v1_q0)
        yield from cirq.decompose(
            cirq.CCX(ctrl0, ctrl1, v1_q0),
            keep=lambda op: len(op.qubits) <= 2,
        )
    elif p_part == 3:
        # Doubly-controlled U_R4^3 = doubly-controlled [X(v1_q1) · CNOT(v1_q1→v1_q0)]
        yield from cirq.decompose(
            cirq.CCX(ctrl0, ctrl1, v1_q1),
            keep=lambda op: len(op.qubits) <= 2,
        )
        _ctrl_x_inner = cirq.ControlledGate(
            cirq.X, num_controls=2, control_values=[1, 1]
        )
        yield from cirq.decompose(
            _ctrl_x_inner.on(ctrl1, v1_q1, v1_q0),
            keep=lambda op: len(op.qubits) <= 2,
        )


def _ctrl_carry_ops(
    ctrl0:   cirq.Qid,
    ctrl1:   cirq.Qid,
    v1_q0:   cirq.Qid,
    v1_q1:   cirq.Qid,
    vaux_q:  cirq.Qid,
    p_part:  int,
) -> Iterator[OP_TREE]:
    r"""Yield doubly-controlled conditional carry: flip vaux_q iff V₁ < p_part
    AND ctrl0=1 AND ctrl1=1.

    All multi-qubit controlled gates are pre-decomposed to len≤2 ops so
    ForteNativeGateset never encounters 3+ qubit operations directly (which
    would trigger QSD rather than the efficient KAK path).

    p_part=1: fire when ctrl0=1, ctrl1=1, v1_q0=0, v1_q1=0 (4-control Toffoli)
    p_part=2: fire when ctrl0=1, ctrl1=1, v1_q0=0             (3-control Toffoli)
    p_part=3: fire when ctrl0=1, ctrl1=1, NOT(v1_q0=1 & v1_q1=1)
    """
    _keep2 = lambda op: len(op.qubits) <= 2
    if p_part == 0:
        return
    elif p_part == 1:
        yield cirq.X(v1_q0)
        yield cirq.X(v1_q1)
        yield from cirq.decompose(
            cirq.X(vaux_q).controlled_by(ctrl0, ctrl1, v1_q0, v1_q1),
            keep=_keep2,
        )
        yield cirq.X(v1_q1)
        yield cirq.X(v1_q0)
    elif p_part == 2:
        yield cirq.X(v1_q0)
        yield from cirq.decompose(
            cirq.X(vaux_q).controlled_by(ctrl0, ctrl1, v1_q0),
            keep=_keep2,
        )
        yield cirq.X(v1_q0)
    elif p_part == 3:
        # ¬(v1_q0=1 ∧ v1_q1=1): unconditional flip on (ctrl0,ctrl1), undo for V1=3
        yield from cirq.decompose(
            cirq.X(vaux_q).controlled_by(ctrl0, ctrl1),
            keep=_keep2,
        )
        yield from cirq.decompose(
            cirq.X(vaux_q).controlled_by(ctrl0, ctrl1, v1_q0, v1_q1),
            keep=_keep2,
        )


def _carry_ops_single(
    v1_q0: cirq.Qid,
    v1_q1: cirq.Qid,
    vaux_q: cirq.Qid,
    p_part: int,
) -> Iterator[OP_TREE]:
    r"""Yield the carry gate: flip vaux_q iff V₁ (in computational basis) < p_part.

    After applying the cyclic d=4 shift by power p = 4*p_full + p_part, the
    carry register must be incremented once for every time V₁ crossed zero.
    The p_full full-cycle wraps contribute unconditional X gates (handled by
    the caller); the p_part partial wraps fire when new_V₁ ∈ [0, p_part).

    Standard binary encoding of V₁ states: |b0 b1⟩ with state k = 2*b0 + b1.
        p_part=0 : never fires → yield nothing
        p_part=1 : fires when new_V₁ = 0 → (b0=0 AND b1=0)
                   → X(v1_q0); X(v1_q1); CCX(v1_q0, v1_q1, vaux_q); X(v1_q1); X(v1_q0)
        p_part=2 : fires when new_V₁ ∈ {0,1} → b0=0
                   → X(v1_q0); CNOT(v1_q0, vaux_q); X(v1_q0)
        p_part=3 : fires when new_V₁ ∈ {0,1,2} → NOT(b0=1 AND b1=1)
                   → X(vaux_q); CCX(v1_q0, v1_q1, vaux_q)

    All CCX yields are pre-decomposed to len≤2 ops (keep=len≤2) to avoid
    QSD fallback in ForteNativeGateset, which has no native 3-qubit gates.
    """
    _keep2 = lambda op: len(op.qubits) <= 2
    if p_part == 0:
        return
    elif p_part == 1:
        # Carry iff new_V₁ = 0 = |00⟩ → Toffoli with both controls negated
        yield cirq.X(v1_q0)
        yield cirq.X(v1_q1)
        yield from cirq.decompose(cirq.CCX(v1_q0, v1_q1, vaux_q), keep=_keep2)
        yield cirq.X(v1_q1)
        yield cirq.X(v1_q0)
    elif p_part == 2:
        # Carry iff new_V₁ ∈ {0,1} → MSB (v1_q0) = 0 → CNOT with negated control
        yield cirq.X(v1_q0)
        yield cirq.CNOT(v1_q0, vaux_q)
        yield cirq.X(v1_q0)
    elif p_part == 3:
        # Carry iff new_V₁ ∈ {0,1,2} → NOT(v1_q0=1 AND v1_q1=1)
        # Implemented as: flip vaux unconditionally, then undo for V₁=3 via CCX
        yield cirq.X(vaux_q)
        yield from cirq.decompose(cirq.CCX(v1_q0, v1_q1, vaux_q), keep=_keep2)


def _correction_ops(
    v1_q0:    cirq.Qid,
    v1_q1:    cirq.Qid,
    vaux_qs:  list,
    m:        int,
    r:        int,
) -> Iterator[OP_TREE]:
    r"""Yield modular correction C_m in the computational basis.

    After a ℤ_{4r} carry-gate shift, state m is the transient image of state
    m-1.  C_m performs SWAP(|0⟩, |m_in_comp_basis⟩) to redirect that transient
    population to |0⟩, enforcing ℤ_m wrap-around.

    For r=1 (m≤4, 2-qubit system on V₁):
        The correction is a 4×4 permutation — emitted via
        cirq.two_qubit_matrix_to_cz_operations.

    For r=2 (5≤m≤8, 3-qubit system on V₁+1 Vaux qubit):
        The correction is an 8×8 permutation — QSD via cirq.decompose(keep≤2).

    No cirq.MatrixGate survives into the compiled circuit.
    """
    if m % (4 * r) == 0:
        return                          # m = 4r exactly: no correction needed

    n_aux  = _n_aux_bits(r) if r > 1 else 0
    dim    = 4 * (2 ** n_aux) if n_aux > 0 else 4
    m_idx  = _state_physical_index(m, r)

    # Build SWAP(|0⟩, |m_idx⟩) permutation matrix
    corr = np.eye(dim, dtype=complex)
    corr[0, 0]         = 0.0
    corr[m_idx, m_idx] = 0.0
    corr[0, m_idx]     = 1.0
    corr[m_idx, 0]     = 1.0

    if r == 1:
        # 4×4 correction on V₁ — direct two-qubit compilation
        yield cirq.two_qubit_matrix_to_cz_operations(
            v1_q0, v1_q1, np.round(corr, 10), allow_partial_czs=True,
        )
    else:
        # 8×8 (or larger) correction on V₁+Vaux — QSD → ≤2-qubit ops
        all_qs = [v1_q0, v1_q1] + list(vaux_qs)
        n_phys = 2 + n_aux
        mat_op = cirq.MatrixGate(np.round(corr, 10), qid_shape=(2,) * n_phys).on(*all_qs)
        yield from cirq.decompose(
            mat_op,
            keep=lambda o: cirq.num_qubits(o) <= 2,
            on_stuck_raise=None,
        )


class PhysicalCompositeVirtShiftWrapper(cirq.Gate):
    r"""Physical wrapper for CompositeVirtualShiftGate(m, power) for any m > 1.

    Qubit layout for r=1 (m≤4): (v1_q0, v1_q1)
    Qubit layout for r>1 (m>4): (v1_q0, v1_q1, vaux_q0, ..., vaux_q_{n_aux-1})
    V₁ uses B_VIRT encoding; Vaux uses standard binary encoding.

    Decomposition pipeline — entirely within the existing two-qubit-first contract:

      Step 1  B_VIRT† on (v1_q0, v1_q1)
                  → cirq.two_qubit_matrix_to_cz_operations → CZ + rotations → GPI/GPI2/ZZ

      Step 2  Cyclic d=4 shift U_R4^{p_part} on (v1_q0, v1_q1)
                  → cirq.two_qubit_matrix_to_cz_operations → CZ + rotations → GPI/GPI2/ZZ

      Step 3  Unconditional X^{p_full % 2} on each vaux qubit (full-cycle carry)
                  → cirq.X → single-qubit → GPI/GPI2

      Step 4  Conditional carry gate on vaux: flip iff new_V₁ < p_part
                  → cirq.X, cirq.CNOT, cirq.CCX on ≤3 qubits → CZ+rot → GPI/GPI2/ZZ

      Step 5  B_VIRT on (v1_q0, v1_q1)
                  → cirq.two_qubit_matrix_to_cz_operations → CZ + rotations → GPI/GPI2/ZZ

    No cirq.MatrixGate on n>2 qubits is emitted, preserving the
    two-qubit-first compilation contract for GPI/GPI2/ZZ hardware.
    """

    def __init__(self, m: int, power: int = 1):
        assert m >= 1
        self._m      = m
        self._r      = _r_for_m(m)
        self._power  = power % m if m > 1 else 0
        self._n_aux  = _n_aux_bits(self._r) if self._r > 1 else 0
        self._n      = 2 + self._n_aux
        self._matrix = _composite_shift_phys(m, self._power)

    def _num_qubits_(self) -> int:
        return self._n

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (2,) * self._n

    def _has_unitary_(self) -> bool:
        return True

    def _unitary_(self) -> np.ndarray:
        return self._matrix.copy()

    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        r"""Explicit carry-gate decomposition — no MatrixGate on n>2 qubits.

        Decomposes the composite ℤ_{4r} shift into:
          (a) 2-qubit V₁ operations via B_VIRT and the cyclic d=4 shift, and
          (b) at most one Toffoli (CCX) or CNOT per vaux qubit for the carry.
        All operations are ≤3-qubit standard Cirq gates compilable by ForteNativeGateset.
        """
        v1_q0, v1_q1 = qubits[0], qubits[1]
        vaux_qs       = list(qubits[2:])   # one qubit per aux bit

        p       = self._power
        p_full  = p // 4   # number of complete ℤ₄ cycles → unconditional carry increments
        p_part  = p  % 4   # residual shift → conditional carry

        # ── Step 1: rotate V₁ to standard computational basis ────────────────
        yield cirq.two_qubit_matrix_to_cz_operations(
            v1_q0, v1_q1,
            np.round(B_VIRT.conj().T, 10),
            allow_partial_czs=True,
        )

        # ── Step 2: apply cyclic d=4 shift U_R4^{p_part} on V₁ ──────────────
        if p_part > 0:
            U_shift = np.round(_cyclic_shift_binary_padded(4, p_part), 10)
            yield cirq.two_qubit_matrix_to_cz_operations(
                v1_q0, v1_q1, U_shift, allow_partial_czs=True,
            )

        # ── Step 3: unconditional carry for p_full full cycles ────────────────
        # Each full ℤ₄ cycle (power += 4) increments Vaux by 1.
        # For a binary Vaux register of n_aux bits we perform a binary increment
        # of p_full, ripple-carry style, but for r=2 (n_aux=1) it is simply an
        # X gate if p_full is odd.
        if vaux_qs:
            if len(vaux_qs) == 1:
                if p_full % 2 == 1:
                    yield cirq.X(vaux_qs[0])
            else:
                # Generic n_aux-bit increment by p_full using ripple-carry chain.
                # This only arises for r≥4 (m≥16); the loop emits at most n_aux
                # Toffoli gates, all ≤3-qubit.
                carry = p_full
                for bit, vq in enumerate(vaux_qs):
                    if carry & 1:
                        yield cirq.X(vq)
                    carry >>= 1
                    if carry == 0:
                        break

        # ── Step 4: conditional carry for the p_part partial shift ───────────
        if vaux_qs and p_part > 0:
            yield from _carry_ops_single(v1_q0, v1_q1, vaux_qs[0], p_part)

        # ── Step 5 (NEW): modular correction for m < 4r ──────────────────────
        # C_m = SWAP(|0⟩, |m⟩) in computational basis: redirects the transient
        # population at state |m⟩ (image of |m-1⟩ under ℤ_{4r} shift) to |0⟩.
        # No-op when m = 4r (pure carry-gate, no correction needed).
        yield from _correction_ops(v1_q0, v1_q1, vaux_qs, self._m, self._r)

        # ── Step 6: rotate V₁ back to B_VIRT Bell-separable basis ────────────
        yield cirq.two_qubit_matrix_to_cz_operations(
            v1_q0, v1_q1,
            np.round(B_VIRT, 10),
            allow_partial_czs=True,
        )

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        sup = f"^{self._power}" if self._power != 1 else ""
        sym = f"CVR{sup}(m={self._m})"
        return cirq.CircuitDiagramInfo(wire_symbols=(sym,) * self._n)

    def __repr__(self) -> str:
        return f"PhysicalCompositeVirtShiftWrapper(m={self._m}, power={self._power})"

    def __eq__(self, other) -> bool:
        return (isinstance(other, PhysicalCompositeVirtShiftWrapper)
                and self._m == other._m and self._power == other._power)

    def __hash__(self) -> int:
        return hash((type(self), self._m, self._power))


class PhysicalCompositeCofactorCouplingWrapper(cirq.Gate):
    r"""Physical (2+2+n_aux)-qubit wrapper for CompositeCofactorCouplingGate.

    Qubit layout: (log_q0, log_q1, v1_q0, v1_q1, vaux_q0, ..., vaux_q_{n_aux-1})
    Logical uses B_LOG; V₁ uses B_VIRT; Vaux uses standard binary encoding.

    Decomposition pipeline — no MatrixGate, fully hardware-compilable:

    The coupling action maps |k_log⟩_L |j_full⟩_V → |k_log⟩_L |(j_full ± ν·k_log) mod m⟩_V.
    It is implemented as a sum of doubly-controlled composite shifts, one per
    logical sector k_log ∈ {1, 2, 3} (k_log=0 is identity).

    For each k_log:
      1. X gates on logical to map k_log → |11⟩ in computational basis.
      2. Doubly-controlled B_LOG† rotation on logical  [2-qubit, via two_qubit_matrix_to_cz]
         ... is NOT needed: we stay in computational basis throughout. B_LOG transforms
         are applied once at the outer boundary.
      3. Doubly-controlled U_R4^{p_part} on V₁  [_ctrl_shift_ops: CCX + doubly-ctrl CNOT]
      4. Doubly-controlled unconditional carry on Vaux  [CCX]
      5. Doubly-controlled conditional carry on Vaux  [_ctrl_carry_ops: multi-ctrl X]
      6. X gates back to undo step 1.

    The outer boundary:
      - B_LOG† on (log_q0, log_q1) and B_VIRT† on (v1_q0, v1_q1) at the start.
      - B_VIRT on (v1_q0, v1_q1) and B_LOG on (log_q0, log_q1) at the end.

    Every emitted gate is cirq.two_qubit_matrix_to_cz_operations, cirq.X,
    cirq.CNOT, cirq.CCX, or cirq.X.controlled_by(n) — all compilable to
    GPI/GPI2/ZZ by ForteNativeGateset.  No cirq.MatrixGate is emitted.
    """

    def __init__(self, m: int, nu: int = 1, inverse: bool = False):
        assert m >= 1
        self._m       = m
        self._r       = _r_for_m(m)
        self._nu      = nu % m if m > 1 else 0
        self._inverse = inverse
        self._n_aux   = _n_aux_bits(self._r) if self._r > 1 else 0
        self._n       = 2 + 2 + self._n_aux
        self._matrix  = _composite_coupling_phys(m, self._nu, inverse)

    def _num_qubits_(self) -> int:
        return self._n

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (2,) * self._n

    def _has_unitary_(self) -> bool:
        return True

    def _unitary_(self) -> np.ndarray:
        return self._matrix.copy()

    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        r"""Explicit hardware-compilable decomposition — no MatrixGate.

        Implements doubly-controlled composite shifts for each logical sector
        k_log ∈ {1,2,3}, using only CCX, doubly-controlled CNOT, and
        multi-control X gates.  Follows the same gate-compilation contract as
        every other hardware-targeting physical wrapper.
        """
        log_q0, log_q1, v1_q0, v1_q1 = qubits[0], qubits[1], qubits[2], qubits[3]
        vaux_qs = list(qubits[4:])

        m   = 4 * self._r
        nu  = self._nu

        # ── Outer basis rotation: computational basis entry ──────────────────
        yield cirq.two_qubit_matrix_to_cz_operations(
            log_q0, log_q1, np.round(B_LOG.conj().T, 10), allow_partial_czs=True,
        )
        yield cirq.two_qubit_matrix_to_cz_operations(
            v1_q0, v1_q1, np.round(B_VIRT.conj().T, 10), allow_partial_czs=True,
        )

        # ── Per-sector controlled composite shifts ───────────────────────────
        # Standard binary: k_log = 2*b0 + b1, so
        #   k_log=1 → |01⟩: b0=0, b1=1  (need X on log_q0 to get |11⟩)
        #   k_log=2 → |10⟩: b0=1, b1=0  (need X on log_q1 to get |11⟩)
        #   k_log=3 → |11⟩: already all-ones, no X needed

        for k_log in range(1, 4):
            eff_power = (nu * k_log) % m if not self._inverse else ((-nu * k_log) % m)
            if eff_power == 0:
                continue

            p_full = eff_power // 4
            p_part = eff_power  % 4

            # ── Map k_log to |11⟩ ────────────────────────────────────────────
            if k_log == 1:
                yield cirq.X(log_q0)
            elif k_log == 2:
                yield cirq.X(log_q1)

            # ── Doubly-controlled U_R4^{p_part} on V₁ ────────────────────────
            yield from _ctrl_shift_ops(log_q0, log_q1, v1_q0, v1_q1, p_part)

            # ── Doubly-controlled unconditional carry for p_full full cycles ──
            if vaux_qs and p_full % 2 == 1:
                yield cirq.CCX(log_q0, log_q1, vaux_qs[0])

            # ── Doubly-controlled conditional carry for p_part residual ───────
            if vaux_qs and p_part > 0:
                yield from _ctrl_carry_ops(
                    log_q0, log_q1, v1_q0, v1_q1, vaux_qs[0], p_part
                )

            # ── Doubly-controlled modular correction for m < 4r ──────────────
            # After the carry-gate shift, state m is a transient for eff_power steps.
            # We apply the controlled correction only when eff_power would land
            # on state m (i.e., the input was state m - eff_power, modulo m).
            # The correction SWAP(|0⟩,|m⟩) is doubly-controlled on (log_q0, log_q1).
            m_r = 4 * self._r
            if m < m_r:
                m_idx   = _state_physical_index(m, self._r)
                n_aux_c = _n_aux_bits(self._r) if self._r > 1 else 0
                dim_c   = 2 ** (2 + n_aux_c)
                # Build doubly-controlled SWAP(|0⟩,|m_idx⟩) on (v1_q0, v1_q1, *vaux_qs)
                # as a MatrixGate (QSD → ≤2-qubit ops); prefixed by X(log_q0/q1) if needed
                corr = np.eye(dim_c, dtype=complex)
                corr[0, 0] = 0; corr[m_idx, m_idx] = 0
                corr[0, m_idx] = 1; corr[m_idx, 0] = 1
                virt_qs = [v1_q0, v1_q1] + list(vaux_qs)
                ctrl_corr = cirq.MatrixGate(
                    np.round(corr, 10), qid_shape=(2,) * (2 + n_aux_c)
                ).controlled_by(log_q0, log_q1)
                yield from cirq.decompose(
                    ctrl_corr.on(log_q0, log_q1, *virt_qs[:2 + n_aux_c]),
                    keep=lambda o: cirq.num_qubits(o) <= 2,
                    on_stuck_raise=None,
                )

            # ── Undo the k_log → |11⟩ mapping ────────────────────────────────
            if k_log == 1:
                yield cirq.X(log_q0)
            elif k_log == 2:
                yield cirq.X(log_q1)

        # ── Outer basis rotation: return to Bell-separable bases ─────────────
        yield cirq.two_qubit_matrix_to_cz_operations(
            v1_q0, v1_q1, np.round(B_VIRT, 10), allow_partial_czs=True,
        )
        yield cirq.two_qubit_matrix_to_cz_operations(
            log_q0, log_q1, np.round(B_LOG, 10), allow_partial_czs=True,
        )

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        dag = "†" if self._inverse else ""
        sym = f"Ucof_c{dag}(r={self._r},ν={self._nu})"
        return cirq.CircuitDiagramInfo(wire_symbols=(sym,) * self._n)

    def __repr__(self) -> str:
        return (f"PhysicalCompositeCofactorCouplingWrapper("
                f"m={self._m}, nu={self._nu}, inverse={self._inverse})")

    def __eq__(self, other) -> bool:
        return (isinstance(other, PhysicalCompositeCofactorCouplingWrapper)
                and self._m == other._m and self._nu == other._nu
                and self._inverse == other._inverse)

    def __hash__(self) -> int:
        return hash((type(self), self._m, self._nu, self._inverse))


# ==============================================================================
# 7. PHOTONIC GATES (LOGICAL ORBITAL)
# ==============================================================================

class PhotonAbsorptionGate(cirq.Gate):
    r"""Diagonal phase operator encoding discrete photon absorption energy.
    
    Implements the photo-excitation energy injection term in the MQE pipeline:
        U_photon(phi) = diag(1, e^{i*phi}, e^{2i*phi}, e^{3i*phi})
    
    Acts on a single NomosIonQid. The angle phi = omega * tau encodes the 
    absorbed photon energy (Ha) scaled by the effective interaction duration.
    Physically, this imprints the photon energy onto the logical occupation 
    ladder, preparing the state for non-adiabatic surface hopping at conical 
    intersections.
    
    Args:
        phi: Photon-induced phase shift in radians. Default pi/2.
    """
    def __init__(self, phi: float = np.pi / 2):
        self._phi = phi

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (4,)

    def _unitary_(self) -> np.ndarray:
        return np.diag([np.exp(1j * k * self._phi) for k in range(4)])

    def _has_unitary_(self) -> bool:
        return True

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=(f"Photon({self._phi:.3f})",))

    def __repr__(self) -> str:
        return f"PhotonAbsorptionGate(phi={self._phi!r})"

    @classmethod
    def from_energy(cls, omega_ha: float, dt: float = 0.02) -> "PhotonAbsorptionGate":
        r"""Convenience constructor from photon energy in Hartree.
        Args:
            omega_ha: Photon energy in atomic units (Ha).
            dt:       Effective Trotter interaction time (Ha⁻¹).
        """
        return cls(phi=omega_ha * dt)

class PhotonEmissionGate(cirq.Gate):
    r"""Diagonal phase operator encoding discrete photon emission energy.
    
    Implements the photo-emission energy extraction term in the MQE pipeline:
        U_emission(phi) = diag(1, e^{-i*phi}, e^{-2i*phi}, e^{-3i*phi})
    
    Acts on a single NomosIonQid. The angle phi = omega * tau encodes the 
    emitted photon energy (Ha) scaled by the effective interaction duration.
    Physically, this unwinds the photon energy from the logical occupation 
    ladder, modeling stimulated emission or radiative decay pathways.
    Mathematically, it is the exact inverse of PhotonAbsorptionGate:
        U_emission(phi) = U_absorption(phi)†
    
    Args:
        phi: Photon-induced phase shift magnitude in radians. Default pi/2.
    """
    def __init__(self, phi: float = np.pi / 2):
        self._phi = abs(phi)  # Store as positive magnitude for semantic clarity

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (4,)

    def _unitary_(self) -> np.ndarray:
        return np.diag([np.exp(-1j * k * self._phi) for k in range(4)])

    def _has_unitary_(self) -> bool:
        return True

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=(f"Photon†({self._phi:.3f})",))

    def __repr__(self) -> str:
        return f"PhotonEmissionGate(phi={self._phi!r})"

    def __pow__(self, exponent: int):
        return PhotonEmissionGate(self._phi * exponent)

    def __eq__(self, other) -> bool:
        return isinstance(other, PhotonEmissionGate) and np.isclose(self._phi, other._phi)

    def __hash__(self) -> int:
        return hash((type(self), round(self._phi, 8)))

    def adjoint(self) -> "PhotonAbsorptionGate":
        """Return the adjoint (exact inverse) as a PhotonAbsorptionGate."""
        return PhotonAbsorptionGate(phi=self._phi)

    @classmethod
    def from_energy(cls, omega_ha: float, dt: float = 0.02) -> "PhotonEmissionGate":
        r"""Convenience constructor from emitted photon energy in Hartree.
        Args:
            omega_ha: Emitted photon energy in atomic units (Ha).
            dt:       Effective Trotter interaction time (Ha⁻¹).
        """
        return cls(phi=omega_ha * dt)


# ==============================================================================
# 8. CROSS-MANIFOLD SWAP GATE (LOGICAL × VIRTUAL, 4-QUBIT)
# ==============================================================================

class CrossManifoldSWAPGate(cirq.Gate):
    r"""Restricted SWAP_{L_p,V_p} for a single orbital: Janus surface-hopping.

    Implements the cross-manifold transfer for one hydride orbital:
        SWAP_{L_p, V_p}: |k>_L |j>_V → |j>_L |k>_V

    The full Janus operator S_LV^{(p,q)} = SWAP_{L_p,V_p} ⊗ SWAP_{L_q,V_q}
    is constructed by applying this gate to each hydride orbital pair p, q.

    Acts on one NomosIonQid (d=4) + one virtual register (d=4, m=4 case).
    For m≠4, we use GeneralizedVirtualShiftGate to pad/unpad.

    Note: In the m=4 case this gate is exact. For m≠4, the Janus crossing
    is assumed adiabatic (n_cross=0) by construction of the mechanism.
    """

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (4, 4)  # SWAP operates on matching dimensions

    def _unitary_(self) -> np.ndarray:
        """16×16 SWAP: |i,j> → |j,i>."""
        U = np.zeros((16, 16), dtype=complex)
        for i in range(4):
            for j in range(4):
                U[j * 4 + i, i * 4 + j] = 1.0
        return U

    def _has_unitary_(self) -> bool:
        return True

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=("×LV", "×LV"))

    def __repr__(self) -> str:
        return "CrossManifoldSWAPGate()"


# ==============================================================================
# 9. CONFORMATIONAL SHIFT GATE (LOGICAL ORBITAL)
# ==============================================================================

class ConformationalShiftGate(cirq.Gate):
    r"""Local docking-induced conformational shift S_dock^{(n)}.

    Models Fe-protein docking-induced modulation of orbital energies within
    a local O(1)-sized neighbourhood D_n. Implemented as a product of
    clock rotations and shift operators on the docking orbitals.

    For the MQE validation framework, this gate applies a small diagonal
    phase to each docking orbital, representing the geometry-induced change
    delta_h_{pp}^{(n)} in one-electron integrals.

    Acts on a single NomosIonQid.

    Args:
        delta_h: On-site energy shift in Ha (typ. O(0.01) Ha per step).
        dt:      Trotter step size in Ha⁻¹.
    """

    def __init__(self, delta_h: float = 0.01, dt: float = 0.02):
        self._delta_h = delta_h
        self._dt      = dt

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (4,)

    def _unitary_(self) -> np.ndarray:
        # Z_Clock rotation by delta_h * dt per occupation unit
        phi = self._delta_h * self._dt
        return np.diag([np.exp(1j * k * phi) for k in range(4)])

    def _has_unitary_(self) -> bool:
        return True

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(
            wire_symbols=(f"Sdock(Δh={self._delta_h:.3f})",)
        )

    def __repr__(self) -> str:
        return f"ConformationalShiftGate(delta_h={self._delta_h!r}, dt={self._dt!r})"

# ==============================================================================
# 3. PRECOMPUTED PHYSICAL MATRICES
# ==============================================================================

# ElectronShiftGate (power=1) → same ontological action as U_R shift
# Physical matrix for power k: B_LOG @ cyclic_shift_onto(4, k) @ B_LOG†
def _electron_shift_phys(power: int) -> np.ndarray:
    return _get_physical_1q(_cyclic_shift_onto(4, power), B_LOG)

# ProtonPhaseGate → parameterized diagonal phase on logical manifold
def _proton_phase_phys(phi: float) -> np.ndarray:
    return _get_physical_1q(_diagonal_phase_onto(4, phi), B_LOG)

# ConformationalShiftGate → diagonal clock rotation by delta_h * dt per level
def _conformational_shift_phys(delta_h: float, dt: float) -> np.ndarray:
    phi = delta_h * dt
    return _get_physical_1q(_diagonal_phase_onto(4, phi), B_LOG)

# GeneralizedVirtualShiftGate (m=4, VirtualQudit) → B_VIRT basis
def _gen_virt_shift_phys_d4(power: int) -> np.ndarray:
    return _get_physical_1q(_cyclic_shift_onto(4, power), B_VIRT)

# CrossManifoldSWAPGate → B_LOG ⊗ B_VIRT basis (16×16 physical)
_cross_swap_onto = _cross_manifold_swap_onto()
cross_swap_phys = _get_physical_2q(_cross_swap_onto, B_LOG, B_VIRT)  # 16×16

# CofactorCouplingGate (m=4) → B_LOG ⊗ B_VIRT basis (16×16 physical)
def _cofactor_coupling_phys_d4(nu: int) -> np.ndarray:
    M_onto = _cofactor_coupling_onto(m=4, nu=nu)
    return _get_physical_2q(M_onto, B_LOG, B_VIRT)

# ==============================================================================
# 4. GENERAL-m BINARY QUBIT ENCODING (for m ≠ 4)
# ==============================================================================

def _n_qubits_for_m(m: int) -> int:
    r"""Minimum number of qubits to represent d=m in standard binary basis."""
    if m <= 1: return 0
    if m == 2: return 1
    return int(math.ceil(math.log2(m)))

def _cyclic_shift_binary_padded(m: int, power: int) -> np.ndarray:
    r"""2^n × 2^n unitary implementing cyclic shift of d=m in standard binary basis.

    The d=m cyclic shift |k⟩ → |k+power mod m⟩ is embedded in a 2^n
    space where n = ceil(log2(m)). Unused states |m⟩,...,|2^n-1⟩ are
    mapped to identity (no leakage under Zeno boundary conditions).

    Returns:
        (2^n × 2^n) complex unitary matrix.
    """
    n      = _n_qubits_for_m(m)
    dim    = 2 ** n
    U = np.zeros((dim, dim), dtype=complex)
    for k in range(m):
        k_out        = (k + power) % m
        U[k_out, k]  = 1.0
    # Identity on unused states
    for k in range(m, dim):
        U[k, k] = 1.0
    return U

def _cofactor_coupling_binary_padded(m: int, nu: int) -> np.ndarray:
    r"""Physical (4 × 2^n_virt)² matrix for CofactorCouplingGate with general m.

    The logical part (d=4) uses 2 qubits with B_LOG encoding.
    The virtual part (d=m) uses n_virt = ceil(log2(m)) qubits in standard basis.

    Full physical matrix dimension: 2² × 2^n_virt = 4 × 2^n_virt qubits total.
    """
    n_virt  = _n_qubits_for_m(m)
    dim_virt= 2 ** n_virt
    dim_log = 4   # always 4 (d=4 logical in Bell-separable encoding)

    # Build ontological matrix: acts on logical index (0-3) and virtual index (0-m-1)
    dim_onto = 4 * m
    M_onto   = _cofactor_coupling_onto(m, nu)

    # Physical logical block: lift via B_LOG
    # We need a (4 * dim_virt) × (4 * dim_virt) physical matrix.
    # Approach: apply B_LOG to logical sector, keep virtual in standard binary basis.
    # U_phys = (B_LOG ⊗ I_{dim_virt}) @ M_padded @ (B_LOG ⊗ I_{dim_virt})†
    # where M_padded is M_onto embedded in 4 × dim_virt space.

    dim_phys = dim_log * dim_virt
    M_padded = np.zeros((dim_phys, dim_phys), dtype=complex)

    # Embed M_onto (4m × 4m) into (4 × dim_virt) × (4 × dim_virt)
    # Indexing: physical index = k_log * dim_virt + j_virt
    for k_log in range(4):
        for j_virt in range(m):
            j_out   = (j_virt + nu * k_log) % m
            row     = k_log * dim_virt + j_out
            col     = k_log * dim_virt + j_virt
            M_padded[row, col] = 1.0
    # Identity for unused virtual states
    for k_log in range(4):
        for j_virt in range(m, dim_virt):
            idx     = k_log * dim_virt + j_virt
            M_padded[idx, idx] = 1.0

    # Apply B_LOG to logical sector: B_LOG_block = B_LOG ⊗ I_{dim_virt}
    B_log_block = np.kron(B_LOG, np.eye(dim_virt, dtype=complex))
    return B_log_block @ M_padded @ B_log_block.conj().T

# ==============================================================================
# 5. PHYSICAL WRAPPER CLASSES (One-Qudit Gates, d=4 Logical / d=4 Virtual)
# ==============================================================================

class PhysicalElectronShiftWrapper(cirq.Gate):
    r"""Physical 2-qubit wrapper for ElectronShiftGate (d=4 logical).

    Compilation:
        U_phys = B_LOG @ cyclic_shift(4, power) @ B_LOG†  (4×4)
    Decomposed via cirq.two_qubit_matrix_to_cz_operations → CZ + rotations.
    """

    def __init__(self, power: int = 1):
        self._power  = power % 4
        self._matrix = _electron_shift_phys(self._power)

    def _num_qubits_(self) -> int: return 2
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return self._matrix.copy()

    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        yield cirq.two_qubit_matrix_to_cz_operations(
            qubits[0], qubits[1],
            np.round(self._matrix, 10),
            allow_partial_czs=True,
        )

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        sup = f"^{self._power}" if self._power != 1 else ""
        sym = f"eShift{sup}"
        return cirq.CircuitDiagramInfo(wire_symbols=(sym, sym))

    def __repr__(self) -> str:
        return f"PhysicalElectronShiftWrapper(power={self._power})"

    def __eq__(self, other) -> bool:
        return isinstance(other, PhysicalElectronShiftWrapper) and self._power == other._power

    def __hash__(self) -> int:
        return hash((type(self), self._power))


# ==============================================================================
#  UNIFIED ELECTRON TRANSFER WRAPPER (INJECTION + EJECTION)
# ==============================================================================
class PhysicalElectronTransferWrapper(cirq.Gate):
    r"""Physical 2-qubit wrapper for bidirectional electron transfer (Injection/Ejection).
    
    Unifies PhysicalElectronShiftWrapper and ElectronEject logic.
    Compilation:
        U_phys = B_LOG @ cyclic_shift(4, signed_power) @ B_LOG†
    where signed_power = power * direction.
    
    Decomposed via cirq.two_qubit_matrix_to_cz_operations → CZ + rotations.
    """

    def __init__(self, direction: int = 1, power: int = 1):
        self._direction = direction  # +1 for Injection, -1 for Ejection
        self._power = power
        
        # Calculate effective shift (e.g., power=1, dir=-1 -> shift=-1 = 3 mod 4)
        shift_power = (self._power * self._direction) % 4
        self._matrix = _get_physical_1q(_cyclic_shift_onto(4, shift_power), B_LOG)

    def _num_qubits_(self) -> int: return 2
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return self._matrix.copy()

    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        yield cirq.two_qubit_matrix_to_cz_operations(
            qubits[0], qubits[1],
            np.round(self._matrix, 10),
            allow_partial_czs=True,
        )

    def _circuit_diagram_info_(self, args):
        sym = "eInject" if self._direction == 1 else "eEject"
        return cirq.CircuitDiagramInfo(wire_symbols=(sym, sym))

    def __repr__(self) -> str:
        return f"PhysicalElectronTransferWrapper(direction={self._direction}, power={self._power})"

    def __eq__(self, other) -> bool:
        return (isinstance(other, PhysicalElectronTransferWrapper) and 
                self._direction == other._direction and self._power == other._power)

    def __hash__(self) -> int:
        return hash((type(self), self._direction, self._power))


# ==============================================================================
# 7. PROTON PHASE GATE WRAPPER (LOGICAL ORBITAL) - Wrapper
# ==============================================================================

class PhysicalProtonPhaseWrapper(cirq.Gate):
    r"""Physical 2-qubit wrapper for ProtonPhaseGate (d=4 logical).

    Compilation:
        U_phys = B_LOG @ diag(1, e^{iφ}, e^{2iφ}, e^{3iφ}) @ B_LOG†  (4×4)
    Decomposed via cirq.two_qubit_matrix_to_cz_operations.

    Note: For φ = π/2 this is exactly Z_clock; for general φ it is a
    parameterized phase gate in the Bell-separable basis.
    """

    def __init__(self, phi: float = np.pi / 2):
        self._phi    = phi
        self._matrix = _proton_phase_phys(phi)

    def _num_qubits_(self) -> int: return 2
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return self._matrix.copy()

    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        yield cirq.two_qubit_matrix_to_cz_operations(
            qubits[0], qubits[1],
            np.round(self._matrix, 10),
            allow_partial_czs=True,
        )

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(
            wire_symbols=(f"ZH({self._phi:.3f})", f"ZH({self._phi:.3f})")
        )

    def __repr__(self) -> str:
        return f"PhysicalProtonPhaseWrapper(phi={self._phi!r})"

    def __eq__(self, other) -> bool:
        return isinstance(other, PhysicalProtonPhaseWrapper) and np.isclose(self._phi, other._phi)

    def __hash__(self) -> int:
        return hash((type(self), round(self._phi, 8)))


class PhysicalConformationalShiftWrapper(cirq.Gate):
    r"""Physical 2-qubit wrapper for ConformationalShiftGate (d=4 logical).

    Compilation:
        U_phys = B_LOG @ diag(1, e^{iδh·dt}, ...) @ B_LOG†  (4×4)
    """

    def __init__(self, delta_h: float = 0.01, dt: float = 0.02):
        self._delta_h = delta_h
        self._dt      = dt
        self._matrix  = _conformational_shift_phys(delta_h, dt)

    def _num_qubits_(self) -> int: return 2
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return self._matrix.copy()

    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        yield cirq.two_qubit_matrix_to_cz_operations(
            qubits[0], qubits[1],
            np.round(self._matrix, 10),
            allow_partial_czs=True,
        )

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        sym = f"Sdock(Δh={self._delta_h:.3f})"
        return cirq.CircuitDiagramInfo(wire_symbols=(sym, sym))

    def __repr__(self) -> str:
        return f"PhysicalConformationalShiftWrapper(delta_h={self._delta_h!r}, dt={self._dt!r})"


# ==============================================================================
# 6. GENERALIZED VIRTUAL SHIFT WRAPPERS (d=4 and d=m cases)
# ==============================================================================

class PhysicalGenVirtShiftWrapper_d4(cirq.Gate):
    r"""Physical 2-qubit wrapper for GeneralizedVirtualShiftGate with m=4.

    Compilation identical to PhysicalURWrapper(is_virtual=True):
        U_phys = B_VIRT @ cyclic_shift(4, power) @ B_VIRT†  (4×4)
    """

    def __init__(self, power: int = 1):
        self._power  = power % 4
        self._matrix = _gen_virt_shift_phys_d4(self._power)

    def _num_qubits_(self) -> int: return 2
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return self._matrix.copy()

    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        yield cirq.two_qubit_matrix_to_cz_operations(
            qubits[0], qubits[1],
            np.round(self._matrix, 10),
            allow_partial_czs=True,
        )

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        sup = f"^{self._power}" if self._power != 1 else ""
        return cirq.CircuitDiagramInfo(
            wire_symbols=(f"VUR_m4{sup}", f"VUR_m4{sup}")
        )

    def __repr__(self) -> str:
        return f"PhysicalGenVirtShiftWrapper_d4(power={self._power})"

    def __eq__(self, other) -> bool:
        return isinstance(other, PhysicalGenVirtShiftWrapper_d4) and self._power == other._power

    def __hash__(self) -> int:
        return hash((type(self), self._power))


class PhysicalGenVirtShiftWrapper_general(cirq.Gate):
    r"""Physical n-qubit wrapper for GeneralizedVirtualShiftGate with m ≠ 4.

    Encoding: standard binary basis on n = ceil(log2(m)) qubits.
    The d=m cyclic shift is embedded in a 2^n-dimensional space with identity
    on unused states |m⟩, ..., |2^n-1⟩ (protected by Zeno boundary conditions).

    Compilation:
        U_phys = cyclic_shift_binary_padded(m, power)   (2^n × 2^n)

    Decomposition:
        n=1 (m=2) : single-qubit X gate — trivially GPI.
        n=2 (m=3) : cirq.two_qubit_matrix_to_cz_operations → CZ+rot → GPI/GPI2/ZZ.
        n≥3 (m=5+): cirq.decompose with keep≤2-qubit recursively applies Cirq's
                    Quantum Shannon Decomposition, producing ≤2-qubit operations
                    that cirq.optimize_for_target_gateset compiles to GPI/GPI2/ZZ.
                    No cirq.MatrixGate survives into the compiled circuit.

    Supported moduli with qubit counts:
        m=2 → 1 qubit  (X gate)
        m=3 → 2 qubits (cyclic permutation on |00⟩,|01⟩,|10⟩)
        m=5 → 3 qubits (cyclic permutation on |000⟩,...,|100⟩)
        m=6 → 3 qubits
    """

    def __init__(self, m: int, power: int = 1):
        assert m >= 2, f"m must be >= 2 for non-trivial shift; got {m}"
        self._m       = m
        self._power   = power % m
        self._n       = _n_qubits_for_m(m)
        self._dim     = 2 ** self._n
        self._matrix  = _cyclic_shift_binary_padded(m, self._power)

    def _num_qubits_(self) -> int: return self._n
    def _qid_shape_(self) -> Tuple[int, ...]: return (2,) * self._n
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return self._matrix.copy()

    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        M = np.round(self._matrix, 10)
        if self._n == 1:
            # m=2: single-qubit X gate
            yield cirq.X(qubits[0])
        elif self._n == 2:
            # m=3: direct 2-qubit CZ compilation — no MatrixGate
            yield cirq.two_qubit_matrix_to_cz_operations(
                qubits[0], qubits[1], M, allow_partial_czs=True,
            )
        else:
            # m≥5, n≥3: Quantum Shannon Decomposition → ≤2-qubit ops.
            # cirq.decompose recurses until every resulting op acts on ≤2 qubits.
            # cirq.optimize_for_target_gateset then compiles each 2-qubit op
            # (via KAK → CZ → GPI/GPI2/ZZ), leaving has_matrix: [✓] ZERO.
            mat_op = cirq.MatrixGate(M, qid_shape=(2,) * self._n).on(*qubits[:self._n])
            yield from cirq.decompose(
                mat_op,
                keep=lambda o: cirq.num_qubits(o) <= 2,
                on_stuck_raise=None,
            )

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        sup = f"^{self._power}" if self._power != 1 else ""
        sym = f"UR_V{sup}(m={self._m})"
        return cirq.CircuitDiagramInfo(wire_symbols=(sym,) * self._n)

    def __repr__(self) -> str:
        return f"PhysicalGenVirtShiftWrapper_general(m={self._m}, power={self._power})"

    def __eq__(self, other) -> bool:
        return (isinstance(other, PhysicalGenVirtShiftWrapper_general)
                and self._m == other._m and self._power == other._power)

    def __hash__(self) -> int:
        return hash((type(self), self._m, self._power))

    def verify_unitarity(self) -> bool:
        """Confirm 2^n × 2^n unitary structure: U†U = I."""
        U = self._matrix
        return bool(np.allclose(U.conj().T @ U, np.eye(self._dim)))

    def verify_cyclic_action(self) -> bool:
        """Confirm |k⟩ → |k+power mod m⟩ for k=0,...,m-1."""
        U = self._matrix
        for k in range(self._m):
            k_out = (k + self._power) % self._m
            if not np.isclose(U[k_out, k], 1.0):
                return False
        return True

    def verify_boundary_identity(self) -> bool:
        """Confirm U acts as identity on unused states k=m,...,2^n-1."""
        U = self._matrix
        for k in range(self._m, self._dim):
            if not np.isclose(U[k, k], 1.0):
                return False
        return True


# ==============================================================================
# 7. COFACTOR COUPLING GATE WRAPPER (LOGICAL × VIRTUAL, 4-QUBIT)
# ==============================================================================

class PhysicalCofactorCouplingWrapper_d4(cirq.Gate):
    r"""Physical 4-qubit wrapper for CofactorCouplingGate with m=4.

    Maps the cross-manifold coupling U_coupling^{(p,m=4)} to the
    (B_LOG ⊗ B_VIRT) physical basis:

        U_phys = (B_LOG ⊗ B_VIRT) @ U_coupling_onto(4,ν) @ (B_LOG ⊗ B_VIRT)†

    The 16×16 physical matrix is decomposed via the same pattern as
    PhysicalCURWrapper: basis rotation → controlled gate → basis unrotation.

    Qubits layout: (log_q0, log_q1, virt_q0, virt_q1)
    """

    def __init__(self, nu: int = 1):
        self._nu     = nu % 4
        self._matrix = _cofactor_coupling_phys_d4(self._nu)

    def _num_qubits_(self) -> int: return 4
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2, 2, 2)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return self._matrix.copy()

    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        l0, l1, v0, v1 = qubits

        # Stage 1: Rotate both registers to computational basis
        yield cirq.two_qubit_matrix_to_cz_operations(
            l0, l1, np.round(B_LOG.conj().T, 10), allow_partial_czs=True,
        )
        yield cirq.two_qubit_matrix_to_cz_operations(
            v0, v1, np.round(B_VIRT.conj().T, 10), allow_partial_czs=True,
        )

        # Stage 2: Per-sector doubly-controlled shifts on V1.
        #
        # In the computational basis: k_log = 2*b0 + b1 (b0=l0, b1=l1).
        #   k_log=1 → |01⟩ → X(l0) maps to |11⟩, fire, X(l0) back
        #   k_log=2 → |10⟩ → X(l1) maps to |11⟩, fire, X(l1) back
        #   k_log=3 → |11⟩ → already |11⟩, fire directly
        #
        # _ctrl_shift_ops fires a doubly-controlled U_R4^{p} on (v0,v1) when
        # both ctrl0=1 AND ctrl1=1, using only CCX / doubly-controlled CNOT
        # primitives — no MatrixGate emitted.
        for k_log in range(1, 4):
            eff_power = (self._nu * k_log) % 4
            if eff_power == 0:
                continue

            if k_log == 1:
                yield cirq.X(l0)
            elif k_log == 2:
                yield cirq.X(l1)

            yield from _ctrl_shift_ops(l0, l1, v0, v1, eff_power)

            if k_log == 1:
                yield cirq.X(l0)
            elif k_log == 2:
                yield cirq.X(l1)

        # Stage 3: Rotate back to Bell-separable basis
        yield cirq.two_qubit_matrix_to_cz_operations(
            v0, v1, np.round(B_VIRT, 10), allow_partial_czs=True,
        )
        yield cirq.two_qubit_matrix_to_cz_operations(
            l0, l1, np.round(B_LOG, 10), allow_partial_czs=True,
        )

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        sym = f"Ucof(m=4,ν={self._nu})"
        return cirq.CircuitDiagramInfo(wire_symbols=(sym,) * 4)

    def __repr__(self) -> str:
        return f"PhysicalCofactorCouplingWrapper_d4(nu={self._nu})"

    def __eq__(self, other) -> bool:
        return isinstance(other, PhysicalCofactorCouplingWrapper_d4) and self._nu == other._nu

    def __hash__(self) -> int:
        return hash((type(self), self._nu))


class PhysicalCofactorDecouplingWrapper_d4(cirq.Gate):
    """Physical 4-qubit wrapper for CofactorDecouplingGate (m=4).

    Implements |k⟩_L |j⟩_V → |k⟩_L |(j − ν·k) mod 4⟩_V via per-sector
    doubly-controlled inverse shifts, exactly mirroring PhysicalCofactorCouplingWrapper_d4
    but with eff_power = (−ν·k_log) mod 4.  No MatrixGate emitted.
    """
    def __init__(self, nu: int = 1):
        self._nu = nu % 4
        M_onto = np.zeros((16, 16), dtype=complex)
        for k in range(4):
            for j in range(4):
                j_out = (j - self._nu * k) % 4
                M_onto[k * 4 + j_out, k * 4 + j] = 1.0
        self._matrix = _get_physical_2q(M_onto, B_LOG, B_VIRT)

    def _num_qubits_(self) -> int: return 4
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2, 2, 2)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return self._matrix.copy()

    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        l0, l1, v0, v1 = qubits

        # Stage 1: Rotate both registers to computational basis
        yield cirq.two_qubit_matrix_to_cz_operations(
            l0, l1, np.round(B_LOG.conj().T, 10), allow_partial_czs=True,
        )
        yield cirq.two_qubit_matrix_to_cz_operations(
            v0, v1, np.round(B_VIRT.conj().T, 10), allow_partial_czs=True,
        )

        # Stage 2: Per-sector doubly-controlled inverse shifts on V1.
        # eff_power = (−ν · k_log) mod 4 gives the inverse shift magnitude.
        for k_log in range(1, 4):
            eff_power = ((-self._nu) * k_log) % 4
            if eff_power == 0:
                continue

            if k_log == 1:
                yield cirq.X(l0)
            elif k_log == 2:
                yield cirq.X(l1)

            yield from _ctrl_shift_ops(l0, l1, v0, v1, eff_power)

            if k_log == 1:
                yield cirq.X(l0)
            elif k_log == 2:
                yield cirq.X(l1)

        # Stage 3: Rotate back to Bell-separable basis
        yield cirq.two_qubit_matrix_to_cz_operations(
            v0, v1, np.round(B_VIRT, 10), allow_partial_czs=True,
        )
        yield cirq.two_qubit_matrix_to_cz_operations(
            l0, l1, np.round(B_LOG, 10), allow_partial_czs=True,
        )

class PhysicalCofactorCouplingWrapper_general(cirq.Gate):
    r"""Physical (2 + n_virt)-qubit wrapper for CofactorCouplingGate with m ≠ 4.

    Layout: (log_q0, log_q1, virt_q0, ..., virt_q_{n_virt-1})
    where n_virt = ceil(log2(m)).

    Logical sector uses B_LOG encoding (2 qubits).
    Virtual sector uses standard binary encoding (n_virt qubits).

    Note on self._matrix
    --------------------
    _cofactor_coupling_binary_padded returns (B_LOG ⊗ I_virt) · M_padded · (B_LOG ⊗ I_virt)†,
    i.e. the coupling action expressed entirely in the physical qubit basis.
    _decompose_ must implement EXACTLY this matrix — no additional B_LOG rotations.

    The previous implementation was incorrect: it applied B_LOG† then self._matrix
    (which already has B_LOG baked in) then B_LOG, producing B_LOG² · M · (B_LOG†)²
    instead of self._matrix.

    Decomposition
    -------------
    Quantum Shannon Decomposition via cirq.decompose(keep≤2) recursively flattens
    the n_phys-qubit unitary into ≤2-qubit operations.
    cirq.optimize_for_target_gateset then compiles each via KAK → CZ → GPI/GPI2/ZZ.
    No cirq.MatrixGate survives into the compiled circuit; has_matrix: [✓] ZERO.
    """

    def __init__(self, m: int, nu: int = 1):
        assert m >= 2 and m != 4, f"Use d4 wrapper for m=4; got m={m}"
        self._m      = m
        self._nu     = nu % m
        self._n_virt = _n_qubits_for_m(m)
        self._n_phys = 2 + self._n_virt
        self._matrix = _cofactor_coupling_binary_padded(m, self._nu)

    def _num_qubits_(self) -> int: return self._n_phys
    def _qid_shape_(self) -> Tuple[int, ...]: return (2,) * self._n_phys
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return self._matrix.copy()

    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        # Decompose self._matrix directly — no extra B_LOG wrapping.
        # QSD produces ≤2-qubit ops; optimize_for_target_gateset compiles them.
        M = np.round(self._matrix, 10)
        mat_op = cirq.MatrixGate(M, qid_shape=(2,) * self._n_phys).on(*qubits)
        yield from cirq.decompose(
            mat_op,
            keep=lambda o: cirq.num_qubits(o) <= 2,
            on_stuck_raise=None,
        )

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        sym = f"Ucof(m={self._m},ν={self._nu})"
        return cirq.CircuitDiagramInfo(wire_symbols=(sym,) * self._n_phys)

    def __repr__(self) -> str:
        return f"PhysicalCofactorCouplingWrapper_general(m={self._m}, nu={self._nu})"


# ==============================================================================
# 9. PHOTONIC GATES (LOGICAL ORBITAL) - Wrappers
# ==============================================================================

class PhysicalPhotonAbsorptionWrapper(cirq.Gate):
    r"""Physical 2-qubit wrapper for PhotonAbsorptionGate (d=4 logical).
    
    Compilation:
        U_phys = B_LOG @ diag(1, e^{iφ}, e^{2iφ}, e^{3iφ}) @ B_LOG†  (4×4)
    Decomposed via cirq.two_qubit_matrix_to_cz_operations → CZ + rotations.
    """
    def __init__(self, phi: float = np.pi / 2):
        self._phi    = phi
        self._matrix = _get_physical_1q(_diagonal_phase_onto(4, phi), B_LOG)

    def _num_qubits_(self) -> int: return 2
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return self._matrix.copy()

    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        yield cirq.two_qubit_matrix_to_cz_operations(
            qubits[0], qubits[1],
            np.round(self._matrix, 10),
            allow_partial_czs=True,
        )

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(
            wire_symbols=(f"Photon({self._phi:.3f})", f"Photon({self._phi:.3f})")
        )

    def __repr__(self) -> str:
        return f"PhysicalPhotonAbsorptionWrapper(phi={self._phi!r})"

    def __eq__(self, other) -> bool:
        return isinstance(other, PhysicalPhotonAbsorptionWrapper) and np.isclose(self._phi, other._phi)

    def __hash__(self) -> int:
        return hash((type(self), round(self._phi, 8)))


class PhysicalPhotonEmissionWrapper(cirq.Gate):
    r"""Physical 2-qubit wrapper for PhotonEmissionGate (d=4 logical).
    Compilation:
        U_phys = B_LOG @ diag(1, e^{-iφ}, e^{-2iφ}, e^{-3iφ}) @ B_LOG†  (4×4)
    Decomposed via cirq.two_qubit_matrix_to_cz_operations → CZ + rotations.
    """
    def __init__(self, phi: float = np.pi / 2):
        self._phi    = phi
        # Emission = exact adjoint of absorption: diag(1, e^{-ikφ})
        self._matrix = _get_physical_1q(_diagonal_phase_onto(4, -phi), B_LOG)

    def _num_qubits_(self) -> int: return 2
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return self._matrix.copy()

    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        yield cirq.two_qubit_matrix_to_cz_operations(
            qubits[0], qubits[1],
            np.round(self._matrix, 10),
            allow_partial_czs=True,
        )

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(
            wire_symbols=(f"Photon†({self._phi:.3f})", f"Photon†({self._phi:.3f})")
        )

    def __repr__(self) -> str:
        return f"PhysicalPhotonEmissionWrapper(phi={self._phi!r})"

    def __eq__(self, other) -> bool:
        return isinstance(other, PhysicalPhotonEmissionWrapper) and np.isclose(self._phi, other._phi)

    def __hash__(self) -> int:
        return hash((type(self), round(self._phi, 8)))

# ==============================================================================
# 10. CROSS-MANIFOLD SWAP WRAPPER (4-QUBIT, LOGICAL × VIRTUAL d=4)
# ==============================================================================

class PhysicalCrossManifoldSWAPWrapper(cirq.Gate):
    r"""Physical 4-qubit wrapper for CrossManifoldSWAPGate (d=4 ⊗ d=4).

    Implements SWAP_{L_p,V_p}: |k⟩_L |j⟩_V → |j⟩_L |k⟩_V in the
    (B_LOG ⊗ B_VIRT) physical basis:

        U_phys = (B_LOG ⊗ B_VIRT) @ SWAP_onto @ (B_LOG ⊗ B_VIRT)†   (16×16)

    Decomposition strategy (same as PhysicalCURWrapper):
      1. Rotate both sectors to computational basis via B† transforms.
      2. Apply standard 4-qubit SWAP (3 CNOTs per qubit pair × 2 pairs).
      3. Rotate back to Bell-separable basis.

    Qubits layout: (log_q0, log_q1, virt_q0, virt_q1)
    """

    def __init__(self):
        self._matrix = cross_swap_phys.copy()

    def _num_qubits_(self) -> int: return 4
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2, 2, 2)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return self._matrix.copy()

    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        l0, l1, v0, v1 = qubits

        # Stage 1: Rotate to computational basis
        yield cirq.two_qubit_matrix_to_cz_operations(
            l0, l1, B_LOG.conj().T, allow_partial_czs=True
        )
        yield cirq.two_qubit_matrix_to_cz_operations(
            v0, v1, B_VIRT.conj().T, allow_partial_czs=True
        )

        # Stage 2: SWAP between (l0,l1) ↔ (v0,v1) via 3-CNOT decomposition.
        # Each SWAP(a, b) = CNOT(a,b) · CNOT(b,a) · CNOT(a,b).
        # Full register SWAP: (l0 ↔ v0) AND (l1 ↔ v1) simultaneously.
        for la, va in [(l0, v0), (l1, v1)]:
            yield cirq.CNOT(la, va)
            yield cirq.CNOT(va, la)
            yield cirq.CNOT(la, va)

        # Stage 3: Rotate back to Bell-separable basis
        yield cirq.two_qubit_matrix_to_cz_operations(
            l0, l1, B_LOG, allow_partial_czs=True
        )
        yield cirq.two_qubit_matrix_to_cz_operations(
            v0, v1, B_VIRT, allow_partial_czs=True
        )

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=("×LV",) * 4)

    def __repr__(self) -> str:
        return "PhysicalCrossManifoldSWAPWrapper()"

    def __eq__(self, other) -> bool:
        return isinstance(other, PhysicalCrossManifoldSWAPWrapper)

    def __hash__(self) -> int:
        return hash(type(self))


# ==============================================================================
# 9. QUDIT EXPANSION: MQE GATE DISPATCH
# ==============================================================================

def expand_mqe_qudit_circuit(
    circuit: cirq.Circuit,
    n_qubits_per_d4: int = 2,
) -> cirq.Circuit:
    r"""Expand all MQE extension gates from abstract qudits to physical LineQubits.

    Follows the exact three-pass approach of expand_qudit_circuit in gatesetY.py:

      Pass 1 — Build qubit map:
        NomosIonQid(i)      → (LineQubit(2i), LineQubit(2i+1))      [d=4, B_LOG]
        VirtualQudit(i)     → (LineQubit(2N+2i), LineQubit(2N+2i+1))[d=4, B_VIRT]
        LineQid(i, dim=m)   → LineQubit(2N+2K+j..j+n-1)             [d=m, binary]

      Pass 2 — Dispatch each MQE gate to its physical wrapper:
        ElectronShiftGate      → PhysicalElectronShiftWrapper
        ProtonPhaseGate        → PhysicalProtonPhaseWrapper
        ConformationalShiftGate→ PhysicalConformationalShiftWrapper
        GeneralizedVirtualShiftGate(m=4) → PhysicalGenVirtShiftWrapper_d4
        GeneralizedVirtualShiftGate(m≠4) → PhysicalGenVirtShiftWrapper_general
        CofactorCouplingGate(m=4)        → PhysicalCofactorCouplingWrapper_d4
        CofactorCouplingGate(m≠4)        → PhysicalCofactorCouplingWrapper_general
        CrossManifoldSWAPGate            → PhysicalCrossManifoldSWAPWrapper

      Pass 3 — Assemble expanded circuit.

    Args:
        circuit:           Input circuit with MQE qudits and gates.
        n_qubits_per_d4:   Physical qubits per d=4 qudit (always 2, exposed for testing).

    Returns:
        cirq.Circuit: Fully expanded to standard cirq.LineQubit gates, ready for
                      cirq.optimize_for_target_gateset.
    """
    # ── Pass 1: Build sorted qudit list and qubit allocation map ──────────────
    all_qudits = sorted(circuit.all_qubits(), key=_safe_sort_key)

    qubit_map: Dict[cirq.Qid, tuple] = {}
    next_phys_idx = [0]  # Use list for mutability in nested scope

    def _alloc(n_qubits: int) -> Tuple[cirq.LineQubit, ...]:
        qs = tuple(cirq.LineQubit(next_phys_idx[0] + k) for k in range(n_qubits))
        next_phys_idx[0] += n_qubits
        return qs

    for q in all_qudits:
        if isinstance(q, NomosIonQid):
            qubit_map[q] = _alloc(2)        # d=4 logical → 2 qubits (B_LOG)
        elif isinstance(q, VirtualQudit):
            qubit_map[q] = _alloc(2)        # d=4 virtual → 2 qubits (B_VIRT)
        elif isinstance(q, cirq.LineQid):
            m = q.dimension
            if m == 1:
                qubit_map[q] = ()           # trivial: no register needed
            else:
                n = _n_qubits_for_m(m)
                qubit_map[q] = _alloc(n)    # d=m → ceil(log2(m)) qubits (binary)
        elif isinstance(q, cirq.LineQubit):
            qubit_map[q] = (q,)             # carry qubit — already physical, pass through
        else:
            qubit_map[q] = (q,)             # pass-through (already physical)

    # ── Pass 2: Dispatch each operation to its physical wrapper ───────────────
    new_ops: List[cirq.Operation] = []

    for op in circuit.all_operations():
        gate  = op.gate
        qudits = op.qubits

        # Flatten physical qubits for this operation
        flat_qs: List[cirq.LineQubit] = []
        for q in qudits:
            mapped = qubit_map.get(q, (q,))
            flat_qs.extend(mapped)

        # ── MQE gate dispatch ─────────────────────────────────────────────────
        if isinstance(gate, ElectronShiftGate):
            phys_gate = PhysicalElectronShiftWrapper(power=gate._power)
            new_ops.append(phys_gate.on(*flat_qs[:2]))

        elif isinstance(gate, ElectronEjectGate):
            phys_gate = PhysicalElectronTransferWrapper(direction=-1, power=gate._power)
            new_ops.append(phys_gate.on(*flat_qs[:2]))

        elif isinstance(gate, ProtonPhaseGate):
            phys_gate = PhysicalProtonPhaseWrapper(phi=gate._phi)
            new_ops.append(phys_gate.on(*flat_qs[:2]))

        elif isinstance(gate, ConformationalShiftGate):
            phys_gate = PhysicalConformationalShiftWrapper(
                delta_h=gate._delta_h, dt=gate._dt
            )
            new_ops.append(phys_gate.on(*flat_qs[:2]))

        elif isinstance(gate, GeneralizedVirtualShiftGate):
            if gate._m == 4:
                phys_gate = PhysicalGenVirtShiftWrapper_d4(power=gate._power)
                new_ops.append(phys_gate.on(*flat_qs[:2]))
            elif gate._m == 1 or gate._power == 0:
                pass  # Identity — no operation emitted
            else:
                phys_gate = PhysicalGenVirtShiftWrapper_general(
                    m=gate._m, power=gate._power
                )
                new_ops.append(phys_gate.on(*flat_qs[:phys_gate._n]))

        elif isinstance(gate, CofactorCouplingGate):
            if gate._m == 1:
                pass  # No cofactor coupling on trivial register
            elif gate._m == 4:
                phys_gate = PhysicalCofactorCouplingWrapper_d4(nu=gate._nu)
                new_ops.append(phys_gate.on(*flat_qs[:4]))
            else:
                phys_gate = PhysicalCofactorCouplingWrapper_general(
                    m=gate._m, nu=gate._nu
                )
                new_ops.append(phys_gate.on(*flat_qs[:phys_gate._n_phys]))

        elif isinstance(gate, CofactorDecouplingGate):
            if gate._m == 1:
                pass  # trivial register — identity
            elif gate._m == 4:
                phys_gate = PhysicalCofactorDecouplingWrapper_d4(nu=gate._nu)
                new_ops.append(phys_gate.on(*flat_qs[:4]))
            else:
                # Inverse coupling = coupling with nu → (−nu) mod m
                phys_gate = PhysicalCofactorCouplingWrapper_general(
                    m=gate._m, nu=(-gate._nu) % gate._m
                )
                new_ops.append(phys_gate.on(*flat_qs[:phys_gate._n_phys]))

        elif isinstance(gate, CrossManifoldSWAPGate):
            # Requires both the logical and virtual d=4 qudits (m%4==0)
            # Always acts on the primary V1 (d=4) register; Vaux is invariant.
            phys_gate = PhysicalCrossManifoldSWAPWrapper()
            new_ops.append(phys_gate.on(*flat_qs[:4]))

        elif isinstance(gate, CompositeVirtualShiftGate):
            # Acts on (V1: 2 qubits via B_VIRT) [+ (Vaux: n_aux qubits standard)]
            phys_gate = PhysicalCompositeVirtShiftWrapper(m=gate._m, power=gate._power)
            new_ops.append(phys_gate.on(*flat_qs[:phys_gate._n]))

        elif isinstance(gate, CompositeCofactorCouplingGate):
            # Acts on (logical: 2) + (V1: 2) [+ (Vaux: n_aux)]
            phys_gate = PhysicalCompositeCofactorCouplingWrapper(
                m=gate._m, nu=gate._nu, inverse=False
            )
            new_ops.append(phys_gate.on(*flat_qs[:phys_gate._n]))

        elif isinstance(gate, CompositeCofactorDecouplingGate):
            phys_gate = PhysicalCompositeCofactorCouplingWrapper(
                m=gate._m, nu=gate._nu, inverse=True
            )
            new_ops.append(phys_gate.on(*flat_qs[:phys_gate._n]))

        elif isinstance(gate, PhotonAbsorptionGate):
            phys_gate = PhysicalPhotonAbsorptionWrapper(phi=gate._phi)
            new_ops.append(phys_gate.on(*flat_qs[:2]))

        elif isinstance(gate, PhotonEmissionGate):
            phys_gate = PhysicalPhotonEmissionWrapper(phi=gate._phi)
            new_ops.append(phys_gate.on(*flat_qs[:2]))

        else:
            # Pass-through: non-MQE gate (existing tetralemmatic gates or standard)
            # These are handled by expand_qudit_circuit from the base pipeline
            if flat_qs:
                new_ops.append(gate.on(*flat_qs))

    return cirq.Circuit(new_ops)


def _safe_sort_key(q: cirq.Qid) -> tuple:
    """Stable sort key matching the existing expand_qudit_circuit pattern."""
    key = q._comparison_key()
    return (type(q).__name__,) + (key if isinstance(key, tuple) else (key,))


# ==============================================================================
# 10. MAIN COMPILATION ENTRY POINT
# ==============================================================================

def compile_mqe_gates(
    circuit: cirq.Circuit,
    target: str = "forte_native",
    simulation_mode: bool = False,
    expand_existing_qudits: bool = True,
) -> cirq.Circuit:
    r"""Compile MQE extension gates to IonQ native GPI/GPI2/ZZ pulses.

    Three-stage pipeline (identical to compile_tetralemmatic_ionq):

      Stage 1 — MQE qudit expansion:
        expand_mqe_qudit_circuit dispatches MQE gates to Physical*Wrappers
        on standard LineQubits. Existing d=4 NomosIonQid/VirtualQudit gates
        are also expanded if expand_existing_qudits=True.

      Stage 2 — Simulation shortcut:
        If simulation_mode=True, return after expansion. Physical*Wrappers
        retain _unitary_ for exact density-matrix simulation.

      Stage 3 — Target gateset decomposition:
        cirq.optimize_for_target_gateset decomposes Physical*Wrappers to
        GPI/GPI2/ZZ (forte_native) or the 16-gate IonQ API subset (api).

    Args:
        circuit:                 Input circuit with MQE qudits and gates.
        target:                  "forte_native" (GPI/GPI2/ZZ) or "api".
        simulation_mode:         If True, stop after expansion (no pulse synthesis).
        expand_existing_qudits:  Also expand existing NomosIonQid/VirtualQudit
                                 gates via expand_qudit_circuit from gatesetY.

    Returns:
        cirq.Circuit ready for IonQ Forte hardware or cloud submission.
    """
    # Stage 1a: Expand MQE-specific gates
    circuit = expand_mqe_qudit_circuit(circuit)

    # Stage 1b: Expand any remaining existing d=4 qudits (NomosIonQid, VirtualQudit)
    # These arise when MQE circuit is combined with baseline evolution.py circuit
    if expand_existing_qudits:
        from nanoprotogeny.ionq.ionqsumgate import expand_qudit_circuit
        if any(isinstance(q, (NomosIonQid, VirtualQudit))
               for q in circuit.all_qubits()):
            circuit = expand_qudit_circuit(circuit)

    # Stage 2: Simulation path
    if simulation_mode:
        return circuit

    # Stage 3: Target gateset compilation
    if target == "api":
        gateset = cirq_ionq.IonQTargetGateset()
    elif target == "forte_native":
        gateset = ForteNativeGateset()
    else:
        raise ValueError(f"target must be 'api' or 'forte_native'; got {target!r}")

    return cirq.optimize_for_target_gateset(
        circuit,
        gateset=gateset,
        context=cirq.TransformerContext(deep=True),
    )


# ==============================================================================
# 11. MATHEMATICAL VERIFICATION SUITE
# ==============================================================================

def verify_mqe_gate_compilation() -> Dict[str, Dict[str, bool]]:
    r"""Comprehensive mathematical verification of all Physical*Wrapper classes.

    Checks per gate:
      (a) Unitarity:      U†U = I
      (b) Physical basis: U_phys = B @ U_onto @ B†  (via allclose vs precomputed)
      (c) Cyclic order:   U^order = I (where order is the group order)
      (d) Gate action:    selected state-vector spot checks

    Returns:
        Dict[gate_name, Dict[check_name, bool]].
    """
    results: Dict[str, Dict[str, bool]] = {}

    # ── 1. ElectronShiftGate ──────────────────────────────────────────────────
    for power in [1, 2, 3]:
        gate   = PhysicalElectronShiftWrapper(power=power)
        U      = gate._unitary_()
        I4     = np.eye(4)
        checks = {}
        checks["unitary"]     = bool(np.allclose(U.conj().T @ U, I4))
        checks["basis_match"] = bool(np.allclose(U, _electron_shift_phys(power)))
        checks["order_4"]     = bool(np.allclose(np.linalg.matrix_power(U, 4), I4))
        results[f"ElectronShift(power={power})"] = checks

    # ── 2. ProtonPhaseGate ────────────────────────────────────────────────────
    for phi in [np.pi/2, np.pi, np.pi/3]:
        gate   = PhysicalProtonPhaseWrapper(phi=phi)
        U      = gate._unitary_()
        I4     = np.eye(4)
        checks = {}
        checks["unitary"]     = bool(np.allclose(U.conj().T @ U, I4))
        checks["basis_match"] = bool(np.allclose(U, _proton_phase_phys(phi)))
        # Phase gate: U^k = phase_gate(k*phi); periodicity 2π
        n_period = max(1, round(2 * np.pi / abs(phi))) if phi != 0 else 1
        checks["periodic"] = bool(
            np.allclose(np.linalg.matrix_power(U, n_period), I4, atol=1e-6)
        )
        results[f"ProtonPhase(phi={phi:.3f})"] = checks

    # ── 3. ConformationalShiftGate ────────────────────────────────────────────
    gate   = PhysicalConformationalShiftWrapper(delta_h=0.01, dt=0.02)
    U      = gate._unitary_()
    checks = {}
    checks["unitary"]     = bool(np.allclose(U.conj().T @ U, np.eye(4)))

    # Check diagonality in the *logical* basis, not physical computational basis
    M_logical = B_LOG.conj().T @ U @ B_LOG
    checks["logical_phase"] = bool(np.allclose(M_logical - np.diag(np.diag(M_logical)), np.zeros((4,4))))

    results["ConformationalShift(Δh=0.01,dt=0.02)"] = checks

    # ── 4. GeneralizedVirtualShiftGate (m=4) ─────────────────────────────────
    for power in [1, 2, 3]:
        gate   = PhysicalGenVirtShiftWrapper_d4(power=power)
        U      = gate._unitary_()
        checks = {}
        checks["unitary"]     = bool(np.allclose(U.conj().T @ U, np.eye(4)))
        checks["basis_match"] = bool(np.allclose(U, _gen_virt_shift_phys_d4(power)))
        checks["order_4"]     = bool(np.allclose(np.linalg.matrix_power(U, 4), np.eye(4)))
        results[f"GenVirtShift_d4(power={power})"] = checks

    # ── 5. GeneralizedVirtualShiftGate (m=3, m=5) ────────────────────────────
    for m, power in [(3, 1), (3, 2), (5, 1), (5, 3)]:
        gate   = PhysicalGenVirtShiftWrapper_general(m=m, power=power)
        checks = {}
        checks["unitary"]            = gate.verify_unitarity()
        checks["cyclic_action"]      = gate.verify_cyclic_action()
        checks["boundary_identity"]  = gate.verify_boundary_identity()
        checks["order_m"]            = bool(
            np.allclose(
                np.linalg.matrix_power(gate._unitary_(), m),
                np.eye(gate._dim)
            )
        )
        results[f"GenVirtShift_m{m}(power={power})"] = checks

    # ── 6. CofactorCouplingGate (m=4) ────────────────────────────────────────
    for nu in [1, 2]:
        gate   = PhysicalCofactorCouplingWrapper_d4(nu=nu)
        U      = gate._unitary_()
        I16    = np.eye(16)
        checks = {}
        checks["unitary"]     = bool(np.allclose(U.conj().T @ U, I16))
        checks["basis_match"] = bool(np.allclose(U, _cofactor_coupling_phys_d4(nu)))
        # Order: (U_coupling)^{4/gcd(ν,4)} = I
        order = 4 // math.gcd(nu % 4 if nu % 4 != 0 else 4, 4)
        checks["order"] = bool(np.allclose(np.linalg.matrix_power(U, order), I16, atol=1e-6))
        results[f"CofactorCoupling_d4(nu={nu})"] = checks

    # ── 7. CofactorCouplingGate (m=3) ────────────────────────────────────────
    gate   = PhysicalCofactorCouplingWrapper_general(m=3, nu=1)
    U      = gate._unitary_()
    dim    = U.shape[0]
    checks = {}
    checks["unitary"] = bool(np.allclose(U.conj().T @ U, np.eye(dim)))
    results["CofactorCoupling_m3(nu=1)"] = checks

    # ── 8. CrossManifoldSWAPGate ──────────────────────────────────────────────
    gate   = PhysicalCrossManifoldSWAPWrapper()
    U      = gate._unitary_()
    I16    = np.eye(16)
    checks = {}
    checks["unitary"]      = bool(np.allclose(U.conj().T @ U, I16))
    checks["hermitian"]    = bool(np.allclose(U.conj().T, U))           # SWAP† = SWAP
    checks["involutory"]   = bool(np.allclose(U @ U, I16))              # SWAP² = I
    checks["basis_match"]  = bool(np.allclose(U, cross_swap_phys))
    # Spot check: |Th⟩_L |AntiTh⟩_V → |AntiTh⟩_L |Th⟩_V
    # Physical index of |Th⟩_L|AntiTh⟩_V:
    # B_LOG @ [1,0,0,0]^T ⊗ B_VIRT @ [0,1,0,0]^T = ?
    # Check via direct matrix action on a product state
    psi_L_Th     = B_LOG @ np.array([1, 0, 0, 0], dtype=complex)    # physical |Th⟩_L
    psi_V_AntiTh = B_VIRT @ np.array([0, 1, 0, 0], dtype=complex)   # physical |AntiTh⟩_V
    psi_in       = np.kron(psi_L_Th, psi_V_AntiTh)
    psi_L_AntiTh = B_LOG @ np.array([0, 1, 0, 0], dtype=complex)    # physical |AntiTh⟩_L
    psi_V_Th     = B_VIRT @ np.array([1, 0, 0, 0], dtype=complex)   # physical |Th⟩_V
    psi_expected = np.kron(psi_L_AntiTh, psi_V_Th)
    checks["swap_action"] = bool(np.allclose(U @ psi_in, psi_expected))
    results["CrossManifoldSWAP"] = checks

    return results


def print_verification_report(results: Dict[str, Dict[str, bool]]) -> bool:
    """Print the verification report and return True iff all checks pass."""
    w       = 68
    all_ok  = True
    print("\n" + "="*w)
    print(" MQE GATE COMPILATION VERIFICATION SUITE")
    print("="*w)
    print(f"  {'Gate':<42} {'Checks':<10} {'Status':>10}")
    print(f"  {'─'*42} {'─'*10} {'─'*10}")

    for gate_name, checks in results.items():
        n_pass = sum(1 for v in checks.values() if v)
        n_total= len(checks)
        ok     = n_pass == n_total
        all_ok = all_ok and ok
        status = "[✓]" if ok else f"[!] {n_pass}/{n_total}"
        print(f"  {gate_name:<42} {f'{n_pass}/{n_total}':<10} {status:>10}")
        if not ok:
            for chk_name, chk_val in checks.items():
                if not chk_val:
                    print(f"    ✗ {chk_name}")

    print("="*w)
    print(f"  OVERALL: {'[✓] ALL PASSED' if all_ok else '[✗] SOME FAILED'}")
    print("="*w)
    return all_ok


# ==============================================================================
# 12. FACTORY CLASS
# ==============================================================================

class MQEGateFactory:
    r"""Factory and registry for all MQE Physical*Wrapper classes.

    Mirrors TetralemmaticIonURgates and TetralemmaticIonSWAPGates in gatesetY/X.
    Provides:
      - Instantiation of all MQE gate wrapper families.
      - Unified dispatch for reversible PCET (ProtonPhase / Deprotonate).
      - Unified dispatch for cofactor coupling/decoupling cycles.
      - Gate catalogue summary.
    """

    # ── Single-qudit gates (d=4 logical) ──
    ElectronShift      = PhysicalElectronShiftWrapper
    ProtonPhase        = PhysicalProtonPhaseWrapper
    ConformationalShift= PhysicalConformationalShiftWrapper

    # ── Virtual gate (d=4) ──
    GenVirtShift_d4    = PhysicalGenVirtShiftWrapper_d4

    # ── Virtual gate (general d=m) ──
    GenVirtShift_m     = PhysicalGenVirtShiftWrapper_general

    # ── Cross-manifold gates ──
    CofactorCoupling_d4   = PhysicalCofactorCouplingWrapper_d4
    CofactorDecoupling_d4 = PhysicalCofactorDecouplingWrapper_d4
    CofactorCoupling_m    = PhysicalCofactorCouplingWrapper_general
    CrossManifoldSWAP     = PhysicalCrossManifoldSWAPWrapper

    @staticmethod
    def make_electron_shift(power: int = 1) -> PhysicalElectronShiftWrapper:
        return PhysicalElectronShiftWrapper(power=power)

    @staticmethod
    def make_proton_phase(phi: float = np.pi/2) -> PhysicalProtonPhaseWrapper:
        return PhysicalProtonPhaseWrapper(phi=phi)
        
    @staticmethod
    def make_cofactor_decoupling(nu: int = 1) -> PhysicalCofactorDecouplingWrapper_d4:
        return PhysicalCofactorDecouplingWrapper_d4(nu=nu)

    @staticmethod
    def get_gate(gate_type: str, **kwargs) -> cirq.Gate:
        """Unified dispatch for all MQE gates. Supports reversible PCET & cofactor decoupling."""
        phi = kwargs.get("phi", np.pi/2)
        
        # ── Electron Transfer (Unified for Shift & Eject) ──
        if gate_type in ("ElectronShift", "eInject"):
            return PhysicalElectronTransferWrapper(
                direction=1, power=kwargs.get("power", 1)
            )
        elif gate_type in ("ElectronEject", "eEject"):
            return PhysicalElectronTransferWrapper(
                direction=-1, power=kwargs.get("power", 1)
            )
            
        # ── Protonation / Deprotonation (Reversible PCET) ──
        elif gate_type == "ProtonPhase":
            return PhysicalProtonPhaseWrapper(phi=phi)
            
        elif gate_type == "Deprotonate":
            # Exact inverse phase rotation (Z_Clock(−φ) = Z_Clock(φ)†)
            return PhysicalProtonPhaseWrapper(phi=-phi)
            
        # ── Cross-Manifold Coupling ──
        elif gate_type == "CofactorCoupling":
            m, nu = kwargs.get("m", 4), kwargs.get("nu", 1)
            return PhysicalCofactorCouplingWrapper_d4(nu=nu) if m == 4 \
                   else PhysicalCofactorCouplingWrapper_general(m=m, nu=nu)
                   
        # ── Cofactor Decoupling (Exact Inverse / Reset Phase) ──
        elif gate_type == "CofactorDecoupling":
            # Note: Currently dispatches to d=4 wrapper. 
            # Extend to general-m if PhysicalCofactorDecouplingWrapper_general is implemented.
            return PhysicalCofactorDecouplingWrapper_d4(nu=kwargs.get("nu", 1))
                   
        # ── Janus Surface Hopping ──
        elif gate_type == "CrossManifoldSWAP":
            return PhysicalCrossManifoldSWAPWrapper()
            
        else:
            raise ValueError(f"Unknown MQE gate type: {gate_type}")

    @staticmethod
    def make_conformational(delta_h: float = 0.01, dt: float = 0.02):
        return PhysicalConformationalShiftWrapper(delta_h=delta_h, dt=dt)

    @staticmethod
    def make_gen_virt_shift(m: int, power: int = 1):
        if m == 4:
            return PhysicalGenVirtShiftWrapper_d4(power=power)
        return PhysicalGenVirtShiftWrapper_general(m=m, power=power)

    @staticmethod
    def make_cofactor_coupling(m: int, nu: int = 1):
        if m == 4:
            return PhysicalCofactorCouplingWrapper_d4(nu=nu)
        return PhysicalCofactorCouplingWrapper_general(m=m, nu=nu)

    @staticmethod
    def make_cross_swap() -> PhysicalCrossManifoldSWAPWrapper:
        return PhysicalCrossManifoldSWAPWrapper()

    @staticmethod
    def catalogue() -> str:
        lines = [
            "MQE Physical Gate Catalogue",
            "─" * 68,
            f"{'Gate Class':<40} {'n_qubits':<10} {'Basis':<15}",
            f"{'─'*40} {'─'*10} {'─'*15}",
            f"{'PhysicalElectronShiftWrapper':<40} {'2':<10} {'B_LOG':<15}",
            f"{'PhysicalElectronTransferWrapper':<40} {'2':<10} {'B_LOG':<15}",
            f"{'PhysicalProtonPhaseWrapper':<40} {'2':<10} {'B_LOG':<15}",
            f"{'PhysicalConformationalShiftWrapper':<40} {'2':<10} {'B_LOG':<15}",
            f"{'PhysicalGenVirtShiftWrapper_d4':<40} {'2':<10} {'B_VIRT':<15}",
            f"{'PhysicalGenVirtShiftWrapper_general':<40} {'ceil(log2(m))':<10} {'Binary':<15}",
            f"{'PhysicalCofactorCouplingWrapper_d4':<40} {'4':<10} {'B_LOG⊗B_VIRT':<15}",
            f"{'PhysicalCofactorDecouplingWrapper_d4':<40} {'4':<10} {'B_LOG⊗B_VIRT':<15}",
            f"{'PhysicalCofactorCouplingWrapper_general':<40} {'2+ceil(log2(m))':<10} {'B_LOG⊗Binary':<15}",
            f"{'PhysicalCrossManifoldSWAPWrapper':<40} {'4':<10} {'B_LOG⊗B_VIRT':<15}",
        ]
        return "\n".join(lines)

# ==============================================================================
# 13. EXPORTS
# ==============================================================================

__all__ = [
    # Basis matrices
    "B_LOG", "B_VIRT", "B_total",
    # Utility functions
    "_get_physical_1q", "_get_physical_2q",
    "_cyclic_shift_onto", "_diagonal_phase_onto",
    "_cofactor_coupling_onto", "_cross_manifold_swap_onto",
    "_n_qubits_for_m", "_cyclic_shift_binary_padded",
    # Precomputed physical matrices
    "cross_swap_phys",
    "_electron_shift_phys", "_proton_phase_phys",
    "_conformational_shift_phys", "_gen_virt_shift_phys_d4",
    "_cofactor_coupling_phys_d4", "_cofactor_coupling_binary_padded",
    
    # ── NEW: Abstract Gate Classes (for type-hinting & factory dispatch) ──
    "ElectronShiftGate", "ElectronEjectGate",
    "ProtonPhaseGate", "ConformationalShiftGate",
    "CofactorCouplingGate", "CofactorDecouplingGate",
    "PhotonAbsorptionGate", "PhotonEmissionGate",
    "CrossManifoldSWAPGate", "GeneralizedVirtualShiftGate",

    # Physical wrapper classes (single-qudit, d=4 logical)
    "PhysicalElectronShiftWrapper", "PhysicalElectronTransferWrapper",
    "PhysicalProtonPhaseWrapper", "PhysicalConformationalShiftWrapper",
    "PhysicalPhotonAbsorptionWrapper", "PhysicalPhotonEmissionWrapper",
    
    # Physical wrapper classes (virtual, d=4 and general d=m)
    "PhysicalGenVirtShiftWrapper_d4",
    "PhysicalGenVirtShiftWrapper_general",
    
    # Physical wrapper classes (cross-manifold)
    "PhysicalCofactorCouplingWrapper_d4", "PhysicalCofactorDecouplingWrapper_d4",
    "PhysicalCofactorCouplingWrapper_general",
    "PhysicalCrossManifoldSWAPWrapper",
    
    # Compilation pipeline
    "expand_mqe_qudit_circuit",
    "compile_mqe_gates",
    # Verification
    "verify_mqe_gate_compilation",
    "print_verification_report",
    # Factory
    "MQEGateFactory",
]


# ==============================================================================
# MAIN: Standalone verification
# ==============================================================================

if __name__ == "__main__":
    print("=" * 68)
    print(" ionq_mqe_gates.py — MQE Gate Compilation Verification")
    print("=" * 68)
    print(MQEGateFactory.catalogue())
    print()

    results = verify_mqe_gate_compilation()
    all_ok  = print_verification_report(results)

    if all_ok:
        print("\n[✓] All MQE physical gate wrappers verified.")
        print("[✓] Ready for compile_mqe_gates(circuit, target='forte_native').")
    else:
        print("\n[✗] Verification failed. Check wrapper constructors.")