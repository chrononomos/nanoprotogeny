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
nanoprotogeny.ionq.ionqvirtualgates
Refactored Virtual Phase Register Gates (now also compatible with Logical Manifold).
Operates on NomosIonQid (logical) and VirtualQudit (virtual), both d=4.
Maps abstract d=4 operators -> 2-qubit Bell-separable physical encoding -> IonQ native gates.

Gates:
- VURShiftGate: cyclic quarter-turn (F→P→M→R→F on virtual, Th→Syn→Anti→Holo→Th on logical)
- VZClockGate: phase clock (1, i, -1, -i)
- VDFTGate: discrete Fourier transform
- VProjectorGate: sharp/unsharp projector (Kraus channel)
- VPhaseCompensateGate: U_R^{-k} compensation
"""
import numpy as np
import cirq
import cirq_ionq
from typing import Dict, Iterator, Tuple, Union
from cirq import OP_TREE
from nanoprotogeny.ionq.YB171PLUSHARDWARE import NomosIonQid, VirtualQudit

AnyQudit = Union[NomosIonQid, VirtualQudit]

# ==============================================================================
# 1. BASIS TRANSFORMATIONS (Logical & Virtual)
# ==============================================================================
B_LOG = np.array([
    [1.0, 0.0,          0.0,          0.0],
    [0.0, 0.0, 1/np.sqrt(2),  1/np.sqrt(2)],
    [0.0, 0.0, 1/np.sqrt(2), -1/np.sqrt(2)],
    [0.0, 1.0,          0.0,          0.0]
], dtype=complex)

B_VIRT = np.array([
    [0.0, 1.0,          0.0,          0.0],
    [1/np.sqrt(2), 0.0, 1/np.sqrt(2),  0.0],
    [-1/np.sqrt(2),0.0, 1/np.sqrt(2), 0.0],
    [0.0, 0.0,          0.0,          1.0]
], dtype=complex)

# ==============================================================================
# 2. ABSTRACT LOGICAL MATRICES (d=4)
# ==============================================================================
VUR_onto  = np.array([[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,0,1,0]], dtype=complex)
VZ_onto   = np.diag([1, 1j, -1, -1j]).astype(complex)
VDFT_onto = 0.5 * np.array([
    [1,  1,   1,   1 ],
    [1,  1j, -1, -1j],
    [1, -1,   1,  -1 ],
    [1, -1j, -1,  1j]
], dtype=complex)

# ==============================================================================
# 3. PHYSICAL MATRICES FOR EACH MANIFOLD
# ==============================================================================
def to_physical(M_onto: np.ndarray, B: np.ndarray) -> np.ndarray:
    return B @ M_onto @ B.conj().T

VUR_phys_log  = to_physical(VUR_onto, B_LOG)
VZ_phys_log   = to_physical(VZ_onto, B_LOG)
VDFT_phys_log = to_physical(VDFT_onto, B_LOG)

VUR_phys_virt  = to_physical(VUR_onto, B_VIRT)
VZ_phys_virt   = to_physical(VZ_onto, B_VIRT)
VDFT_phys_virt = to_physical(VDFT_onto, B_VIRT)

# ==============================================================================
# 4. GATE WRAPPERS (Unitary + Decomposition)
# ==============================================================================
class PhysicalUnitaryWrapper(cirq.Gate):
    """Wraps a unitary virtual/logical gate and decomposes it to standard gates."""
    def __init__(self, matrix: np.ndarray, symbol: str):
        self._matrix = matrix
        self._symbol = symbol

    def _num_qubits_(self) -> int: return 2
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return self._matrix.copy()

    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        yield cirq.two_qubit_matrix_to_cz_operations(
            qubits[0], qubits[1], np.round(self._matrix, 10), allow_partial_czs=True
        )

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=(self._symbol, self._symbol))
    def __repr__(self) -> str:
        return f"PhysicalUnitaryWrapper({self._symbol})"

class PhysicalProjectorWrapper(cirq.Gate):
    """Wraps VProjectorGate with correct physical matrix for the manifold."""
    def __init__(self, virtual_idx: int, T: float, is_virtual: bool):
        self._idx = virtual_idx
        self._T = T
        self._is_virtual = is_virtual
        B = B_VIRT if is_virtual else B_LOG
        # Build the physical Kraus operators
        sqrt_E = np.zeros((4,4), dtype=complex)
        sqrt_E[virtual_idx, virtual_idx] = np.sqrt(T)
        self._K0 = B @ sqrt_E @ B.conj().T
        sqrt_I = np.eye(4, dtype=complex)
        sqrt_I[virtual_idx, virtual_idx] = np.sqrt(1.0 - T)
        self._K1 = B @ sqrt_I @ B.conj().T

    def _num_qubits_(self) -> int: return 2
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2)
    def _has_unitary_(self) -> bool: return False
    def _has_kraus_(self) -> bool: return True
    def _kraus_(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self._K0, self._K1)
    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        roles = {0: "F/Th", 1: "P/Anti", 2: "M/Syn", 3: "R/Holo"}
        return cirq.CircuitDiagramInfo(wire_symbols=(f"Proj({roles[self._idx]})",)*2)
    def __repr__(self) -> str:
        return f"PhysicalProjectorWrapper(idx={self._idx}, T={self._T}, is_virtual={self._is_virtual})"

class PhysicalPhaseCompensateWrapper(cirq.Gate):
    """Wraps VPhaseCompensateGate with correct physical matrix."""
    def __init__(self, k: int, is_virtual: bool):
        self._k = k % 4
        self._is_virtual = is_virtual
        B = B_VIRT if is_virtual else B_LOG
        comp_onto = np.linalg.matrix_power(VUR_onto.conj().T, self._k)
        self._matrix = B @ comp_onto @ B.conj().T

    def _num_qubits_(self) -> int: return 2
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return self._matrix.copy()
    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        yield cirq.two_qubit_matrix_to_cz_operations(
            qubits[0], qubits[1], np.round(self._matrix, 10), allow_partial_czs=True
        )
    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=(f"Comp({self._k})",)*2)
    def __repr__(self) -> str:
        return f"PhysicalPhaseCompensateWrapper(k={self._k}, is_virtual={self._is_virtual})"

# ==============================================================================
# 5. QUDIT-NATIVE GATE CLASSES
# ==============================================================================
class VURShiftGate(cirq.Gate):
    def _qid_shape_(self) -> Tuple[int, ...]: return (4,)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return VUR_phys_log.copy()
    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=("VUR",))
    def __repr__(self) -> str: return "VURShiftGate()"
    def __eq__(self, other) -> bool: return isinstance(other, VURShiftGate)
    def __hash__(self) -> int: return hash(type(self))

class VZClockGate(cirq.Gate):
    def _qid_shape_(self) -> Tuple[int, ...]: return (4,)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return VZ_phys_log.copy()
    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=("VZc",))
    def __repr__(self) -> str: return "VZClockGate()"
    def __eq__(self, other) -> bool: return isinstance(other, VZClockGate)
    def __hash__(self) -> int: return hash(type(self))

class VDFTGate(cirq.Gate):
    def _qid_shape_(self) -> Tuple[int, ...]: return (4,)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return VDFT_phys_log.copy()
    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=("VDFT",))
    def __repr__(self) -> str: return "VDFTGate()"
    def __eq__(self, other) -> bool: return isinstance(other, VDFTGate)
    def __hash__(self) -> int: return hash(type(self))

class VProjectorGate(cirq.Gate):
    def __init__(self, virtual_idx: int, T: float = 1.0):
        if not (0.0 <= T <= 1.0): raise ValueError("T must be in [0, 1]")
        self._idx = virtual_idx
        self._T = T

    def _qid_shape_(self) -> Tuple[int, ...]: return (4,)
    def _has_unitary_(self) -> bool: return False
    def _has_kraus_(self) -> bool: return True
    # The actual Kraus operators will be supplied by the wrapper.
    def _kraus_(self) -> Tuple[np.ndarray, np.ndarray]:
        # Default to logical (wrapper overrides)
        sqrt_E = np.zeros((4,4), dtype=complex); sqrt_E[self._idx, self._idx] = np.sqrt(self._T)
        K0 = B_LOG @ sqrt_E @ B_LOG.conj().T
        sqrt_I = np.eye(4, dtype=complex); sqrt_I[self._idx, self._idx] = np.sqrt(1.0 - self._T)
        K1 = B_LOG @ sqrt_I @ B_LOG.conj().T
        return (K0, K1)
    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        roles = {0: "0", 1: "1", 2: "2", 3: "3"}
        return cirq.CircuitDiagramInfo(wire_symbols=(f"VProj({roles[self._idx]})",))
    def __repr__(self) -> str: return f"VProjectorGate(idx={self._idx}, T={self._T})"
    def __eq__(self, other) -> bool: return isinstance(other, VProjectorGate) and self._idx == other._idx and np.isclose(self._T, other._T)
    def __hash__(self) -> int: return hash((type(self), self._idx, round(self._T, 6)))

class VPhaseCompensateGate(cirq.Gate):
    def __init__(self, k: int):
        self.k = k % 4

    def _qid_shape_(self) -> Tuple[int, ...]: return (4,)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray:
        comp_onto = np.linalg.matrix_power(VUR_onto.conj().T, self.k)
        return B_LOG @ comp_onto @ B_LOG.conj().T
    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=(f"V-Comp({self.k})",))
    def __repr__(self) -> str: return f"VPhaseCompensateGate(k={self.k})"
    def __eq__(self, other) -> bool: return isinstance(other, VPhaseCompensateGate) and self.k == other.k
    def __hash__(self) -> int: return hash((type(self), self.k))

# ==============================================================================
# 6. EXPANSION & COMPILATION (Dual-Manifold)
# ==============================================================================
def expand_qudit_circuit(circuit: cirq.Circuit) -> cirq.Circuit:
    qubit_map = {}
    phys_qubits = cirq.LineQubit.range(len(circuit.all_qubits()) * 2)
    idx = 0

    def _safe_sort_key(q):
        key = q._comparison_key()
        if isinstance(key, tuple):
            return (type(q).__name__,) + key
        else:
            return (type(q).__name__, key)

    for q in sorted(circuit.all_qubits(), key=_safe_sort_key):
        if isinstance(q, (NomosIonQid, VirtualQudit)):
            qubit_map[q] = (phys_qubits[idx], phys_qubits[idx+1])
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
        is_virtual = isinstance(op.qubits[0], VirtualQudit) if op.qubits else False

        if isinstance(gate, VURShiftGate):
            matrix = VUR_phys_virt if is_virtual else VUR_phys_log
            gate = PhysicalUnitaryWrapper(matrix, "VUR")
        elif isinstance(gate, VZClockGate):
            matrix = VZ_phys_virt if is_virtual else VZ_phys_log
            gate = PhysicalUnitaryWrapper(matrix, "VZc")
        elif isinstance(gate, VDFTGate):
            matrix = VDFT_phys_virt if is_virtual else VDFT_phys_log
            gate = PhysicalUnitaryWrapper(matrix, "VDFT")
        elif isinstance(gate, VProjectorGate):
            gate = PhysicalProjectorWrapper(gate._idx, gate._T, is_virtual)
        elif isinstance(gate, VPhaseCompensateGate):
            gate = PhysicalPhaseCompensateWrapper(gate.k, is_virtual)

        new_ops.append(gate.on(*flat_qs))
    return cirq.Circuit(new_ops)

def compile_to_ionq_native(circuit: cirq.Circuit) -> cirq.Circuit:
    if any(isinstance(q, (NomosIonQid, VirtualQudit)) for q in circuit.all_qubits()):
        circuit = expand_qudit_circuit(circuit)

    while True:
        decomposed_ops = cirq.decompose(circuit.all_operations())
        new_circuit = cirq.Circuit(decomposed_ops)
        if new_circuit == circuit:
            break
        circuit = new_circuit

    ionq_gateset = cirq_ionq.IonQTargetGateset()
    return cirq.optimize_for_target_gateset(circuit, gateset=ionq_gateset)

# ==============================================================================
# 7. VERIFICATION FACTORY (Extended for Logical)
# ==============================================================================
class VirtualGateFactory:
    def verify_logical_properties(self) -> Dict[str, bool]:
        checks = {}
        I4 = np.eye(4, dtype=complex)
        U = VUR_phys_log
        Z = VZ_phys_log
        F = VDFT_phys_log
        checks["VUR_unitary"] = bool(np.allclose(U.conj().T @ U, I4))
        checks["VUR_order4"]  = bool(np.allclose(np.linalg.matrix_power(U, 4), I4))
        checks["VZ_unitary"]  = bool(np.allclose(Z.conj().T @ Z, I4))
        checks["VZ_order4"]   = bool(np.allclose(np.linalg.matrix_power(Z, 4), I4))
        checks["VDFT_unitary"] = bool(np.allclose(F.conj().T @ F, I4))
        return checks

    def verify_virtual_properties(self) -> Dict[str, bool]:
        checks = {}
        I4 = np.eye(4, dtype=complex)
        U = VUR_phys_virt
        Z = VZ_phys_virt
        F = VDFT_phys_virt
        checks["VUR_unitary"] = bool(np.allclose(U.conj().T @ U, I4))
        checks["VUR_order4"]  = bool(np.allclose(np.linalg.matrix_power(U, 4), I4))
        checks["VZ_unitary"]  = bool(np.allclose(Z.conj().T @ Z, I4))
        checks["VZ_order4"]   = bool(np.allclose(np.linalg.matrix_power(Z, 4), I4))
        checks["VDFT_unitary"] = bool(np.allclose(F.conj().T @ F, I4))

        proj = VProjectorGate(1, T=0.7)
        K0, K1 = proj._kraus_()  # uses logical default; but wrapper will use correct basis
        # We test the wrapper directly for virtual:
        wrapper = PhysicalProjectorWrapper(1, 0.7, is_virtual=True)
        K0v, K1v = wrapper._kraus_()
        checks["VProj_Kraus_TP"] = bool(np.allclose(K0v.conj().T @ K0v + K1v.conj().T @ K1v, I4))

        comp_gate = VPhaseCompensateGate(k=2)
        comp_U = cirq.unitary(PhysicalPhaseCompensateWrapper(2, is_virtual=True))
        checks["VPhase_Closure"] = bool(np.allclose(comp_U @ (U @ U), I4))
        return checks

if __name__ == "__main__":
    print("=== Virtual/Logical Gate Verification (Dual-Manifold) ===")
    factory = VirtualGateFactory()

    print("\n--- Logical Manifold Checks ---")
    log_results = factory.verify_logical_properties()
    for k, v in log_results.items():
        print(f"{'✓' if v else '✗'} {k}")

    print("\n--- Virtual Manifold Checks ---")
    virt_results = factory.verify_virtual_properties()
    for k, v in virt_results.items():
        print(f"{'✓' if v else '✗'} {k}")

    def summarize_compiled(circuit: cirq.Circuit, label: str):
        print(f"\n--- {label} ---")
        print(f"Total moments: {len(circuit)}")
        gate_counts = {}
        for op in circuit.all_operations():
            name = op.gate.__class__.__name__
            gate_counts[name] = gate_counts.get(name, 0) + 1
        print("Gate counts:")
        for gate, count in sorted(gate_counts.items()):
            print(f"  {gate}: {count}")
        has_matrix = any(isinstance(op.gate, cirq.MatrixGate) for op in circuit.all_operations())
        print(f"Contains MatrixGate: {has_matrix}")

    print("\n=== Compilation Test (Logical) ===")
    q_log = NomosIonQid(0)
    circ_log = cirq.Circuit([
        VURShiftGate().on(q_log),
        VDFTGate().on(q_log),
        VZClockGate().on(q_log),
        VPhaseCompensateGate(k=2).on(q_log),
        cirq.measure(*cirq.LineQubit.range(2), key="m")
    ])
    print("Original Circuit:")
    print(circ_log)
    try:
        compiled_log = compile_to_ionq_native(circ_log)
        print("\nCompiled Circuit (GPI/GPI2/MS):")
        print(compiled_log)
        summarize_compiled(compiled_log, "Logical Circuit Summary")
    except Exception as e:
        print(f"\nCompilation Failed: {e}")

    print("\n=== Compilation Test (Virtual) ===")
    q_virt = VirtualQudit(0)
    circ_virt = cirq.Circuit([
        VURShiftGate().on(q_virt),
        VDFTGate().on(q_virt),
        VZClockGate().on(q_virt),
        VPhaseCompensateGate(k=2).on(q_virt),
        cirq.measure(*cirq.LineQubit.range(2), key="m")
    ])
    print("Original Circuit:")
    print(circ_virt)
    try:
        compiled_virt = compile_to_ionq_native(circ_virt)
        print("\nCompiled Circuit (GPI/GPI2/MS):")
        print(compiled_virt)
        summarize_compiled(compiled_virt, "Virtual Circuit Summary")
    except Exception as e:
        print(f"\nCompilation Failed: {e}")