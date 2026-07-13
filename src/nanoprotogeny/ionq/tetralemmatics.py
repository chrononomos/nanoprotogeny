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
nanoprotogeny.ionq.tetralemmatics
==================================
Platform-independent mathematical kernel of the Tetralemmatic Fock
isomorphism.  Contains every object that is true regardless of the
target quantum hardware: basis transformation matrices, abstract
Heisenberg–Weyl operators, physical matrix forms, gate wrappers using
standard Cirq CZ decomposition, and abstract d=4 qudit gate classes.

**Nothing in this file imports cirq_ionq or any hardware-specific
library.**  Swapping from IonQ Forte to any other platform (Quantinuum,
IBM, photonic, neutral atoms) requires only a new `<platform>tetralemmatics.py`
that provides compiled basis-change sequences; this file remains untouched.

Contents
--------
Basis matrices
    B_LOG   — logical manifold (S₁/₂ hyperfine, Bell-separable encoding)
    B_VIRT  — virtual manifold (²D₃/₂ shelving, Bell-separable encoding)

Abstract onto-operators  (Heisenberg–Weyl algebra of ℤ₄)
    U_R_shift_onto  — quarter-turn / complex structure J
    Z_clock_onto    — clock phase operator
    DFT_onto        — discrete Fourier transform of U_R

Physical matrices  (B · U_onto · B†, computed once)
    UR_phys_{log,virt}, Z_phys_{log,virt},
    DFT_phys_{log,virt}, DFT_phys_{log,virt}_inv

Helpers
    get_physical_matrix(M_onto, B) → B @ M_onto @ B†
    apply_basis_ops(gate_seq, q0, q1) — platform-agnostic replay

Physical gate wrappers  (Cirq CZ path; compilable by any gateset)
    PhysicalURWrapper, PhysicalZClockWrapper,
    PhysicalDFTWrapper, InversePhysicalDFTWrapper

Abstract qudit gate classes  (d=4, defer to holographic expansion)
    TetralemmaticIonURShiftGate, TetralemmaticIonZClockGate,
    TetralemmaticIonDFTGate, TetralemmaticIonInverseDFTGate

Verification factory
    TetralemmaticIonURgates

Hardware map (both manifolds share this vertex labelling)
    Th=0, AntiTh=1, SynTh=2, HoloTh=3
    HoloTh_F=4, HoloTh_P=5, HoloTh_M=6, HoloTh_R=7
"""

from __future__ import annotations

import numpy as np
import cirq
from typing import Dict, Iterator, Tuple, Union
from cirq import OP_TREE

from nanoprotogeny.ionq.YB171PLUSHARDWARE import (
    NomosState, IonManifold, NomosIonQid, VirtualQudit,
)

AnyQudit = Union[NomosIonQid, VirtualQudit]

# ==============================================================================
# 1. BASIS TRANSFORMATION MATRICES
# ==============================================================================
# B_LOG columns map logical indices [Th, AntiTh, SynTh, HoloTh]
# to physical Bell states [|00⟩, |11⟩, |Ψ⁺⟩, |Ψ⁻⟩].
#
# Encoding principle: definite-occupation-number Fock states (vacuum, doubly
# occupied) → product states; superposition-occupation states (spin-up,
# spin-down) → maximally entangled Bell states.  The fermionic anticommutation
# phase is carried by the entanglement geometry, eliminating Jordan–Wigner
# strings site-locally.
B_LOG = np.array([
    [1.0, 0.0,          0.0,          0.0],
    [0.0, 0.0, 1/np.sqrt(2),  1/np.sqrt(2)],
    [0.0, 0.0, 1/np.sqrt(2), -1/np.sqrt(2)],
    [0.0, 1.0,          0.0,          0.0],
], dtype=complex)

# B_VIRT columns map virtual indices [F, P, M, R]
# to physical Bell states [|Ψ⁻⟩, |00⟩, |Ψ⁺⟩, |11⟩].
#
# The cyclic phase accumulation k ∈ ℤ₄ is encoded so that the baseline
# state |F⟩ (k=0) maps to |Ψ⁻⟩, matching the HoloTh boundary role
# across the cross-manifold SWAP.
B_VIRT = np.array([
    [0.0,           1.0,          0.0,          0.0],
    [1/np.sqrt(2),  0.0, 1/np.sqrt(2),          0.0],
    [-1/np.sqrt(2), 0.0, 1/np.sqrt(2),          0.0],
    [0.0,           0.0,          0.0,          1.0],
], dtype=complex)


# ==============================================================================
# 2. ABSTRACT ONTO-OPERATORS (Heisenberg–Weyl algebra of ℤ₄)
# ==============================================================================
# Quarter-turn cyclic shift: |k⟩ → |(k+1) mod 4⟩
# Acts on the tetralemmatic vertices as the complex structure J (order-4 symmetry,
# J⁴ = I, J² ≠ I).  This is the generator of the virtual phase register's cyclic
# orbit and the kinematic bridge between the logical and virtual manifolds.
U_R_shift_onto = np.zeros((4, 4), dtype=complex)
for _i in range(3):
    U_R_shift_onto[_i + 1, _i] = 1.0
U_R_shift_onto[0, 3] = 1.0

# Clock phase operator: Z|k⟩ = i^k |k⟩ = e^{2πik/4}|k⟩
Z_clock_onto = np.diag([1, 1j, -1, -1j]).astype(complex)

# Discrete Fourier transform (modal basis generator): F · U_R · F† = Z_clock
# Heisenberg–Weyl relation: Z_clock · U_R = i · U_R · Z_clock
_omega = 1j
omega  = _omega   # public alias (used by __init__.py and downstream modules)
DFT_onto = 0.5 * np.array([
    [1,       1,      1,       1     ],
    [1,  _omega,     -1, -_omega     ],
    [1,      -1,      1,      -1     ],
    [1, -_omega,     -1,  _omega     ],
], dtype=complex)


# ==============================================================================
# 3. PHYSICAL MATRICES  (B · U_onto · B†)
# ==============================================================================
def get_physical_matrix(M_onto: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Lift an ontological operator to the physical qubit-pair basis."""
    return B @ M_onto @ B.conj().T


UR_phys_log  = get_physical_matrix(U_R_shift_onto, B_LOG)
Z_phys_log   = get_physical_matrix(Z_clock_onto,   B_LOG)
DFT_phys_log = get_physical_matrix(DFT_onto,        B_LOG)

UR_phys_virt  = get_physical_matrix(U_R_shift_onto, B_VIRT)
Z_phys_virt   = get_physical_matrix(Z_clock_onto,   B_VIRT)
DFT_phys_virt = get_physical_matrix(DFT_onto,        B_VIRT)

DFT_phys_log_inv  = DFT_phys_log.conj().T
DFT_phys_virt_inv = DFT_phys_virt.conj().T


# ==============================================================================
# 4. PLATFORM-AGNOSTIC HELPER
# ==============================================================================
def apply_basis_ops(gate_seq, q0: cirq.Qid, q1: cirq.Qid) -> cirq.OP_TREE:
    """Replay a precomputed basis-change gate sequence on qubit pair (q0, q1).

    ``gate_seq`` is a list of ``(gate, qubit_index_tuple)`` entries where
    indices 0/1 select from ``(q0, q1)``.  The gate objects themselves are
    whatever the platform-specific module provides; this function is a
    platform-agnostic replay adapter.
    """
    qs = (q0, q1)
    for gate, idx in gate_seq:
        yield gate.on(*[qs[i] for i in idx])


# ==============================================================================
# 5. PHYSICAL GATE WRAPPERS  (Cirq CZ path — compilable by any gateset)
# ==============================================================================
# These wrappers decompose via cirq.two_qubit_matrix_to_cz_operations on the
# full physical matrix.  They are correct on any platform; platform-specific
# modules (e.g. ionqtetralemmatics.py) supply more efficient decompositions
# using native gate sequences via apply_basis_ops + sandwich pattern.

class PhysicalURWrapper(cirq.Gate):
    """Physical U_R gate in the Bell-separable basis."""
    def __init__(self, is_virtual: bool):
        self._is_virtual = is_virtual
        self._matrix = UR_phys_virt.copy() if is_virtual else UR_phys_log.copy()

    def _num_qubits_(self) -> int: return 2
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return self._matrix.copy()

    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        yield cirq.two_qubit_matrix_to_cz_operations(
            qubits[0], qubits[1], np.round(self._matrix, 10), allow_partial_czs=True
        )

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        sym = "VUR" if self._is_virtual else "TIUR"
        return cirq.CircuitDiagramInfo(wire_symbols=(sym, sym))

    def __repr__(self) -> str:
        return f"PhysicalURWrapper(is_virtual={self._is_virtual})"


class PhysicalZClockWrapper(cirq.Gate):
    """Physical Z_clock gate in the Bell-separable basis."""
    def __init__(self, is_virtual: bool):
        self._is_virtual = is_virtual
        self._matrix = Z_phys_virt.copy() if is_virtual else Z_phys_log.copy()

    def _num_qubits_(self) -> int: return 2
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return self._matrix.copy()

    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        yield cirq.two_qubit_matrix_to_cz_operations(
            qubits[0], qubits[1], np.round(self._matrix, 10), allow_partial_czs=True
        )

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        sym = "VZc" if self._is_virtual else "TIZc"
        return cirq.CircuitDiagramInfo(wire_symbols=(sym, sym))

    def __repr__(self) -> str:
        return f"PhysicalZClockWrapper(is_virtual={self._is_virtual})"


class PhysicalDFTWrapper(cirq.Gate):
    """Physical DFT gate in the Bell-separable basis."""
    def __init__(self, is_virtual: bool):
        self._is_virtual = is_virtual
        self._matrix = DFT_phys_virt.copy() if is_virtual else DFT_phys_log.copy()

    def _num_qubits_(self) -> int: return 2
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return self._matrix.copy()

    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        yield cirq.two_qubit_matrix_to_cz_operations(
            qubits[0], qubits[1], np.round(self._matrix, 10), allow_partial_czs=True
        )

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        sym = "VDFT" if self._is_virtual else "TIDFT"
        return cirq.CircuitDiagramInfo(wire_symbols=(sym, sym))

    def __repr__(self) -> str:
        return f"PhysicalDFTWrapper(is_virtual={self._is_virtual})"


class InversePhysicalDFTWrapper(cirq.Gate):
    """Physical DFT† (inverse Fourier transform) in the Bell-separable basis."""
    def __init__(self, is_virtual: bool):
        self._is_virtual = is_virtual
        self._matrix = DFT_phys_virt_inv.copy() if is_virtual else DFT_phys_log_inv.copy()

    def _num_qubits_(self) -> int: return 2
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return self._matrix.copy()

    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        yield cirq.two_qubit_matrix_to_cz_operations(
            qubits[0], qubits[1], np.round(self._matrix, 10), allow_partial_czs=True
        )

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        sym = "VF†" if self._is_virtual else "TIF†"
        return cirq.CircuitDiagramInfo(wire_symbols=(sym, sym))

    def __repr__(self) -> str:
        return f"InversePhysicalDFTWrapper(is_virtual={self._is_virtual})"


# ==============================================================================
# 6. ABSTRACT QUDIT GATE CLASSES  (d=4, defer to holographic expansion)
# ==============================================================================
# These gates operate on NomosIonQid / VirtualQudit (d=4) qudits.
# They do not decompose themselves; the unified_expand_qudit_circuit in
# holographic.py dispatches them to the appropriate physical wrappers.

class TetralemmaticIonURShiftGate(cirq.Gate):
    """Abstract d=4 U_R shift gate (quarter-turn automorphism)."""
    def _qid_shape_(self) -> Tuple[int, ...]: return (4,)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return UR_phys_log.copy()
    def _decompose_(self, qubits) -> Iterator[OP_TREE]: return NotImplemented
    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=("TURS4",))
    def __repr__(self) -> str: return "TetralemmaticIonURShiftGate()"
    def __eq__(self, other) -> bool: return isinstance(other, TetralemmaticIonURShiftGate)
    def __hash__(self) -> int: return hash(type(self))


class TetralemmaticIonZClockGate(cirq.Gate):
    """Abstract d=4 Z_clock gate (clock phase operator)."""
    def _qid_shape_(self) -> Tuple[int, ...]: return (4,)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return Z_phys_log.copy()
    def _decompose_(self, qubits) -> Iterator[OP_TREE]: return NotImplemented
    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=("TZC4",))
    def __repr__(self) -> str: return "TetralemmaticIonZClockGate()"
    def __eq__(self, other) -> bool: return isinstance(other, TetralemmaticIonZClockGate)
    def __hash__(self) -> int: return hash(type(self))


class TetralemmaticIonDFTGate(cirq.Gate):
    """Abstract d=4 DFT gate (modal basis generator)."""
    def _qid_shape_(self) -> Tuple[int, ...]: return (4,)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return DFT_phys_log.copy()
    def _decompose_(self, qubits) -> Iterator[OP_TREE]: return NotImplemented
    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=("TDFT4",))
    def __repr__(self) -> str: return "TetralemmaticIonDFTGate()"
    def __eq__(self, other) -> bool: return isinstance(other, TetralemmaticIonDFTGate)
    def __hash__(self) -> int: return hash(type(self))


class TetralemmaticIonInverseDFTGate(cirq.Gate):
    """Abstract d=4 DFT† gate (adjoint of the forward DFT)."""
    def _qid_shape_(self) -> Tuple[int, ...]: return (4,)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return DFT_onto.conj().T
    def _decompose_(self, qubits) -> Iterator[OP_TREE]: return NotImplemented
    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=("F†",))
    def __repr__(self) -> str: return "TetralemmaticIonInverseDFTGate()"
    def __eq__(self, other) -> bool: return isinstance(other, TetralemmaticIonInverseDFTGate)
    def __hash__(self) -> int: return hash(type(self))


# ==============================================================================
# 7. QUDIT CIRCUIT EXPANSION  (abstract → physical qubit pairs)
# ==============================================================================
def expand_qudit_circuit(circuit: cirq.Circuit) -> cirq.Circuit:
    """Expand U_R / Z_clock / DFT qudit gates to physical 2-qubit wrappers.

    Maps each NomosIonQid or VirtualQudit to a (hi, lo) LineQubit pair and
    dispatches the three fundamental qudit gate types to their physical
    wrappers using the CZ decomposition path.

    For full MQE gate support (Coulomb, Exchange, SUM, cross-manifold, etc.)
    use ``holographic.unified_expand_qudit_circuit`` instead.
    """
    qubit_map: dict = {}
    phys_qubits = cirq.LineQubit.range(len(circuit.all_qubits()) * 2)
    idx = 0

    def _sort_key(q):
        key = q._comparison_key()
        return (type(q).__name__,) + (key if isinstance(key, tuple) else (key,))

    for q in sorted(circuit.all_qubits(), key=_sort_key):
        if isinstance(q, (NomosIonQid, VirtualQudit)):
            qubit_map[q] = (phys_qubits[idx], phys_qubits[idx + 1])
            idx += 2
        else:
            qubit_map[q] = q

    new_ops = []
    for op in circuit.all_operations():
        flat_qs = []
        for q in op.qubits:
            mapped = qubit_map.get(q, q)
            if isinstance(mapped, tuple):
                flat_qs.extend(mapped)
            else:
                flat_qs.append(mapped)

        gate = op.gate
        is_virt = isinstance(op.qubits[0], VirtualQudit) if op.qubits else False
        if isinstance(gate, TetralemmaticIonURShiftGate):
            gate = PhysicalURWrapper(is_virt)
        elif isinstance(gate, TetralemmaticIonZClockGate):
            gate = PhysicalZClockWrapper(is_virt)
        elif isinstance(gate, TetralemmaticIonDFTGate):
            gate = PhysicalDFTWrapper(is_virt)
        elif isinstance(gate, TetralemmaticIonInverseDFTGate):
            gate = InversePhysicalDFTWrapper(is_virt)

        new_ops.append(gate.on(*flat_qs))
    return cirq.Circuit(new_ops)


# ==============================================================================
# 8. VERIFICATION FACTORY
# ==============================================================================
class TetralemmaticIonURgates:
    """Factory and verification suite for the three fundamental onto-operators."""

    def __init__(self):
        self.UR = TetralemmaticIonURShiftGate()
        self.Z  = TetralemmaticIonZClockGate()
        self.F  = TetralemmaticIonDFTGate()

    def _basis_states(self, B: np.ndarray):
        return tuple(B @ np.eye(4, dtype=complex)[:, k] for k in range(4))

    def verify_logical_properties(self) -> Dict[str, bool]:
        U, Z, F = UR_phys_log, Z_phys_log, DFT_phys_log
        I4 = np.eye(4, dtype=complex)
        th, anti, syn, holo = self._basis_states(B_LOG)
        checks = {
            "UR_unitary":        np.allclose(U.conj().T @ U, I4),
            "Z_unitary":         np.allclose(Z.conj().T @ Z, I4),
            "F_unitary":         np.allclose(F.conj().T @ F, I4),
            "UR_order4":         np.allclose(np.linalg.matrix_power(U, 4), I4),
            "Z_order4":          np.allclose(Z @ Z @ Z @ Z, I4),
            "F_order4":          np.allclose(np.linalg.matrix_power(F, 4), I4),
            "weyl_conjugation":  np.allclose(F @ U @ F.conj().T, Z),
            "UZ_commutation":    np.allclose(Z @ U, 1j * U @ Z),
            "UR_Th_to_Anti":     np.allclose(U @ th, anti),
            "UR_Anti_to_Syn":    np.allclose(U @ anti, syn),
            "UR_Syn_to_Holo":    np.allclose(U @ syn, holo),
            "UR_Holo_to_Th":     np.allclose(U @ holo, th),
        }
        return {k: bool(v) for k, v in checks.items()}

    def verify_virtual_properties(self) -> Dict[str, bool]:
        U, Z, F = UR_phys_virt, Z_phys_virt, DFT_phys_virt
        I4 = np.eye(4, dtype=complex)
        f, p, m, r = self._basis_states(B_VIRT)
        checks = {
            "VUR_unitary":          np.allclose(U.conj().T @ U, I4),
            "VZ_unitary":           np.allclose(Z.conj().T @ Z, I4),
            "VF_unitary":           np.allclose(F.conj().T @ F, I4),
            "VUR_order4":           np.allclose(np.linalg.matrix_power(U, 4), I4),
            "V_weyl_conjugation":   np.allclose(F @ U @ F.conj().T, Z),
            "V_UZ_commutation":     np.allclose(Z @ U, 1j * U @ Z),
            "VUR_F_to_P":           np.allclose(U @ f, p),
            "VUR_P_to_M":           np.allclose(U @ p, m),
            "VUR_M_to_R":           np.allclose(U @ m, r),
            "VUR_R_to_F":           np.allclose(U @ r, f),
        }
        return {k: bool(v) for k, v in checks.items()}


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================
__all__ = [
    # Basis matrices
    "B_LOG", "B_VIRT",
    # Abstract onto-operators
    "U_R_shift_onto", "Z_clock_onto", "DFT_onto",
    # Physical matrices
    "UR_phys_log", "Z_phys_log", "DFT_phys_log",
    "UR_phys_virt", "Z_phys_virt", "DFT_phys_virt",
    "DFT_phys_log_inv", "DFT_phys_virt_inv",
    # Helpers
    "get_physical_matrix", "apply_basis_ops",
    # Physical wrappers (CZ path)
    "PhysicalURWrapper", "PhysicalZClockWrapper",
    "PhysicalDFTWrapper", "InversePhysicalDFTWrapper",
    # Abstract gate classes
    "TetralemmaticIonURShiftGate", "TetralemmaticIonZClockGate",
    "TetralemmaticIonDFTGate", "TetralemmaticIonInverseDFTGate",
    # Circuit helpers
    "expand_qudit_circuit",
    # Factory
    "TetralemmaticIonURgates",
    # Re-export type alias
    "AnyQudit",
    # Public alias for DFT phase root (backward compat)
    "omega",
]
