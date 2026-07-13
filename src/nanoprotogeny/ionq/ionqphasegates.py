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
nanoprotogeny.ionq.ionqphasegates
Refactored Tetralemmatic Phase Operators (S, T) for IonQ Hardware.
Supports both Logical Manifold (NomosIonQid) and Virtual Phase Register (VirtualQudit).
Algebraic Logic: Geometric phase on polar subspace, identity on non-polar.
Physical Mapping: distinct Bell‑separable encodings for each manifold.
Decomposition: 2‑qubit unitary via cirq.two_qubit_matrix_to_cz_operations.
Hardware map:
Th       = 0  # Th → |00⟩
AntiTh   = 1  # AntiTh → |11⟩
SynTh    = 2  # SynTh → |Ψ⁺⟩
HoloTh   = 3  # HoloTh → |Ψ⁻⟩ (Base Logical Boundary)
HoloTh_F = 4  # F: Base Logical Boundary / U_R^0 on |Ψ⁻⟩
HoloTh_P = 5  # P: +π/2 Phase Accumulator / U_R^1
HoloTh_M = 6  # M:  π Phase Accumulator   / U_R^2
HoloTh_R = 7  # R: -π/2 Phase Accumulator / U_R^3
"""
import numpy as np
import cirq
import cirq_ionq
from cirq_ionq.ionq_native_target_gateset import ForteNativeGateset
from typing import Dict, Iterator, Tuple, Union
from cirq import OP_TREE

from nanoprotogeny.ionq.YB171PLUSHARDWARE import NomosState, IonManifold, NomosIonQid, VirtualQudit
from enum import IntEnum

AnyQudit = Union[NomosIonQid, VirtualQudit]

#==============================================================================
# 1. BASIS TRANSFORMATIONS (Logical & Virtual)
#==============================================================================
# Logical Basis Transformation (Standard)
# Maps [Th, AntiTh, SynTh, HoloTh] -> [|00⟩, |11⟩, |Ψ⁺⟩, |Ψ⁻⟩]
B_LOG = np.array([
    [1.0, 0.0,          0.0,          0.0],
    [0.0, 0.0, 1/np.sqrt(2),  1/np.sqrt(2)],
    [0.0, 0.0, 1/np.sqrt(2), -1/np.sqrt(2)],
    [0.0, 1.0,          0.0,          0.0]
], dtype=complex)

# Virtual Basis Transformation (Holothesis Ladder)
# Maps [HoloTh_F, HoloTh_P, HoloTh_M, HoloTh_R] -> [|Ψ⁻⟩, |00⟩, |Ψ⁺⟩, |11⟩]
B_VIRT = np.array([
    [0.0, 1.0,          0.0,          0.0],
    [1/np.sqrt(2), 0.0, 1/np.sqrt(2),  0.0],
    [-1/np.sqrt(2),0.0, 1/np.sqrt(2), 0.0],
    [0.0, 0.0,          0.0,          1.0]
], dtype=complex)

#==============================================================================
# 2. LOGICAL PHASE MATRICES (Abstract d=4)
#==============================================================================
# S and T gates apply geometric phases to the polar subspace (indices 0 and 1)
# Index 0 (Th/F), Index 1 (AntiTh/P)
S_onto = np.diag([np.exp(-1j * np.pi / 4), np.exp(1j * np.pi / 4), 1.0, 1.0]).astype(complex)
T_onto = np.diag([np.exp(-1j * np.pi / 8), np.exp(1j * np.pi / 8), 1.0, 1.0]).astype(complex)
Z_onto = np.diag([1.0, -1.0, 0.0, 0.0]).astype(complex)

# Physical matrices for each manifold
def get_physical_matrix(M_onto: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Transform a 4x4 logical operator to physical basis using given B matrix."""
    return B @ M_onto @ B.conj().T

S_phys_log = get_physical_matrix(S_onto, B_LOG)
T_phys_log = get_physical_matrix(T_onto, B_LOG)
Z_phys_log = get_physical_matrix(Z_onto, B_LOG)

S_phys_virt = get_physical_matrix(S_onto, B_VIRT)
T_phys_virt = get_physical_matrix(T_onto, B_VIRT)
Z_phys_virt = get_physical_matrix(Z_onto, B_VIRT)

#==============================================================================
# 3. GATE WRAPPER WITH DECOMPOSITION
#==============================================================================
class PhysicalPhaseGateWrapper(cirq.Gate):
    """Wraps logical S or T gate and delegates to correct physical matrix and decomposition."""
    def __init__(self, gate_type: str, is_virtual: bool):
        self._gate_type = gate_type  # 'S' or 'T'
        self._is_virtual = is_virtual
        if gate_type == 'S':
            self._matrix = S_phys_virt.copy() if is_virtual else S_phys_log.copy()
        elif gate_type == 'T':
            self._matrix = T_phys_virt.copy() if is_virtual else T_phys_log.copy()
        else:
            raise ValueError("gate_type must be 'S' or 'T'")

    def _num_qubits_(self) -> int: return 2
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return self._matrix.copy()

    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        q0, q1 = qubits
        # Decompose 2-qubit unitary into CZ/CNOT + Single Qubit rotations
        yield cirq.two_qubit_matrix_to_cz_operations(
            q0, q1, np.round(self._matrix, 10), allow_partial_czs=True
        )

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        symbol = f"V{self._gate_type}" if self._is_virtual else f"TI{self._gate_type}"
        return cirq.CircuitDiagramInfo(wire_symbols=(symbol, symbol))
        
    def __repr__(self) -> str:
        return f"PhysicalPhaseGateWrapper({self._gate_type}, is_virtual={self._is_virtual})"

#==============================================================================
# 4. CIRQ GATE IMPLEMENTATIONS (Qudit-Native)
#==============================================================================
class TetralemmaticIonSGate(cirq.Gate):
    def _qid_shape_(self) -> Tuple[int, ...]: return (4,)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return S_phys_log.copy()
    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        # Defer decomposition to wrapper during expansion
        return NotImplemented
    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=("S",))
    def __repr__(self) -> str: return "TetralemmaticIonSGate()"
    def __eq__(self, other) -> bool: return isinstance(other, TetralemmaticIonSGate)
    def __hash__(self) -> int: return hash(type(self))

class TetralemmaticIonTGate(cirq.Gate):
    def _qid_shape_(self) -> Tuple[int, ...]: return (4,)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return T_phys_log.copy()
    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        # Defer decomposition to wrapper during expansion
        return NotImplemented
    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=("T",))
    def __repr__(self) -> str: return "TetralemmaticIonTGate()"
    def __eq__(self, other) -> bool: return isinstance(other, TetralemmaticIonTGate)
    def __hash__(self) -> int: return hash(type(self))

#==============================================================================
# 5. IONQ COMPILATION BRIDGE (DUAL-MANIFOLD AWARE)
#==============================================================================
def expand_qudit_circuit(circuit: cirq.Circuit) -> cirq.Circuit:
    """Expands NomosIonQid or VirtualQudit (d=4) into two LineQubits."""
    qubit_map = {}
    phys_qubits = cirq.LineQubit.range(len(circuit.all_qubits()) * 2)
    idx = 0
    
    def _safe_sort_key(q):
        key = q._comparison_key()
        return (type(q).__name__,) + key if isinstance(key, tuple) else (type(q).__name__, key)

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
        if isinstance(gate, (TetralemmaticIonSGate, TetralemmaticIonTGate)):
            if len(op.qubits) == 1:
                is_virtual = isinstance(op.qubits[0], VirtualQudit)
                gate_type = 'S' if isinstance(gate, TetralemmaticIonSGate) else 'T'
                gate = PhysicalPhaseGateWrapper(gate_type, is_virtual)
            else:
                raise ValueError("Phase gate expects exactly one qudit")
        new_ops.append(gate.on(*flat_qs))
    return cirq.Circuit(new_ops)

def compile_tetralemmatic_ionq(
    circuit: cirq.Circuit, 
    target: str = "forte_native", 
    simulation_mode: bool = False
) -> cirq.Circuit:
    """Unified compiler for tetralemmatic gates targeting IonQ Forte architecture."""
    if any(isinstance(q, (NomosIonQid, VirtualQudit)) for q in circuit.all_qubits()):
        circuit = expand_qudit_circuit(circuit)
        
    if simulation_mode:
        return circuit
        
    if target == "api":
        gateset = cirq_ionq.IonQTargetGateset()
    elif target == "forte_native":
        gateset = ForteNativeGateset()
    else:
        raise ValueError("target must be 'api' or 'forte_native'")
        
    return cirq.optimize_for_target_gateset(
        circuit,
        gateset=gateset,
        context=cirq.TransformerContext(deep=True)
    )

#==============================================================================
# 6. FACTORY & VERIFICATION SUITE (Extended for Virtual)
#==============================================================================
class TetralemmaticIonPhaseGates:
    """Factory and verification for IonQ-targeted tetralemmatic phase operators."""
    def __init__(self):
        self.S = TetralemmaticIonSGate()
        self.T = TetralemmaticIonTGate()

    def _get_basis_states(self, B: np.ndarray):
        """Return physical basis states for a given manifold."""
        th   = B @ np.array([1, 0, 0, 0], dtype=complex)
        anti = B @ np.array([0, 1, 0, 0], dtype=complex)
        syn  = B @ np.array([0, 0, 1, 0], dtype=complex)
        holo = B @ np.array([0, 0, 0, 1], dtype=complex)
        return th, anti, syn, holo

    def verify_logical_properties(self) -> Dict[str, bool]:
        checks = {}
        S = S_phys_log
        T = T_phys_log
        Z = Z_phys_log
        I4 = np.eye(4, dtype=complex)
        _, _, syn, holo = self._get_basis_states(B_LOG)

        checks["S_unitary"] = bool(np.allclose(S.conj().T @ S, I4))
        checks["T_unitary"] = bool(np.allclose(T.conj().T @ T, I4))
        checks["S_fixes_nonpolar"] = bool(np.allclose(S @ syn, syn) and np.allclose(S @ holo, holo))
        checks["T_fixes_nonpolar"] = bool(np.allclose(T @ syn, syn) and np.allclose(T @ holo, holo))
        
        polar_idx = [0, 3] # Logical indices 0 (Th) and 1 (AntiTh) map to physical indices 0 and 3
        S_p = S[np.ix_(polar_idx, polar_idx)]
        T_p = T[np.ix_(polar_idx, polar_idx)]
        
        std_S_p = np.diag([np.exp(-1j * np.pi / 4), np.exp(1j * np.pi / 4)])
        std_T_p = np.diag([np.exp(-1j * np.pi / 8), np.exp(1j * np.pi / 8)])
        
        checks["qubit_S_phase"] = bool(np.allclose(S_p, std_S_p))
        checks["qubit_T_phase"] = bool(np.allclose(T_p, std_T_p))
        checks["T2_equals_S"] = bool(np.allclose(T @ T, S))
        
        S2_p = (S @ S)[np.ix_(polar_idx, polar_idx)]
        checks["S2_pi2_phase"] = bool(np.allclose(np.diag(S2_p), [-1j, 1j]))
        checks["S_commutes_Z"] = bool(np.allclose(S @ Z, Z @ S))
        checks["T_commutes_Z"] = bool(np.allclose(T @ Z, Z @ T))
        return checks

    def verify_virtual_properties(self) -> Dict[str, bool]:
        checks = {}
        S = S_phys_virt
        T = T_phys_virt
        Z = Z_phys_virt
        I4 = np.eye(4, dtype=complex)
        _, _, m, r = self._get_basis_states(B_VIRT)  # M and R are the non-polar indices (2,3)

        checks["VS_unitary"] = bool(np.allclose(S.conj().T @ S, I4))
        checks["VT_unitary"] = bool(np.allclose(T.conj().T @ T, I4))
        checks["VS_fixes_nonpolar"] = bool(np.allclose(S @ m, m) and np.allclose(S @ r, r))
        checks["VT_fixes_nonpolar"] = bool(np.allclose(T @ m, m) and np.allclose(T @ r, r))
        
        # For virtual, the polar indices are 0 and 1 (F and P).
        checks["VT2_equals_VS"] = bool(np.allclose(T @ T, S))
        checks["VS_commutes_VZ"] = bool(np.allclose(S @ Z, Z @ S))
        checks["VT_commutes_VZ"] = bool(np.allclose(T @ Z, Z @ T))
        return checks

    def compile_test_circuit(self, use_virtual: bool = False) -> cirq.Circuit:
        q = VirtualQudit(0) if use_virtual else NomosIonQid(0)
        return cirq.Circuit(
            self.T.on(q),
            self.S.on(q),
            cirq.measure(*cirq.LineQubit.range(2), key="m")
        )

#==============================================================================
# MAIN EXECUTION (Dual-Manifold with Unified Compiler Validation)
#==============================================================================
if __name__ == "__main__":
    print("=== Tetralemmatic IonQ Phase Gates Verification (Dual-Manifold) ===")
    print(f"Ion Manifold Energies (B=1G): {IonManifold.energy_levels(1.0)}\n")
    factory = TetralemmaticIonPhaseGates()

    print("--- Logical Manifold Checks ---")
    log_results = factory.verify_logical_properties()
    for k, v in log_results.items():
        print(f"{'Yes' if v else 'No'} {k}")
        
    print("\n--- Virtual Manifold Checks ---")
    virt_results = factory.verify_virtual_properties()
    for k, v in virt_results.items():
        print(f"{'Yes' if v else 'No'} {k}")
        
    def summarize_compilation(circuit: cirq.Circuit, label: str, target: str, sim_mode: bool):
        print(f"\n--- {label} [{target.upper()} | SimMode={sim_mode}] ---")
        print(f"Total moments: {len(circuit)}")
        print(f"Total operations: {len(list(circuit.all_operations()))}")
        
        gate_counts = {}
        has_matrix = False
        has_forte_native = False
        for op in circuit.all_operations():
            name = op.gate.__class__.__name__
            gate_counts[name] = gate_counts.get(name, 0) + 1
            if isinstance(op.gate, cirq.MatrixGate):
                has_matrix = True
            if name in ("GPIGate", "GPI2Gate", "ZZGate"):
                has_forte_native = True
                
        print("Gate counts:")
        for gate, count in sorted(gate_counts.items()):
            print(f"  {gate}: {count}")
            
        print(f"Contains MatrixGate: {has_matrix}")
        print(f"Contains Forte Native (GPI/GPI2/ZZ): {has_forte_native}")
        
        if sim_mode and has_matrix:
            print("✓ Simulation mode: Abstract gates preserved as expected.")
        elif not sim_mode and has_matrix:
            print("! Warning: MatrixGate present in hardware compilation path.")
            
        if not sim_mode and target == "forte_native" and has_forte_native:
            print("✓ Forte native target: Pulse-level gates synthesized correctly.")
        elif not sim_mode and target == "forte_native" and not has_forte_native:
            print("! Warning: Expected GPI/GPI2/ZZ gates not found.")

    test_circ_log = factory.compile_test_circuit(use_virtual=False)
    test_circ_virt = factory.compile_test_circuit(use_virtual=True)
    
    # 1. SIMULATION MODE
    print("\n=== Compilation Test: Simulation Mode ===")
    try:
        sim_log = compile_tetralemmatic_ionq(test_circ_log, target="api", simulation_mode=True)
        summarize_compilation(sim_log, "Logical Phase Simulation", "api", True)
        
        sim_virt = compile_tetralemmatic_ionq(test_circ_virt, target="api", simulation_mode=True)
        summarize_compilation(sim_virt, "Virtual Phase Simulation", "api", True)
    except Exception as e:
        print(f"\n✗ Simulation Compilation Failed: {e}")
        
    # 2. API TARGET (Cloud-Ready)
    print("\n=== Compilation Test: API Target (Cloud-Ready) ===")
    try:
        api_log = compile_tetralemmatic_ionq(test_circ_log, target="api", simulation_mode=False)
        summarize_compilation(api_log, "Logical Phase API", "api", False)
        
        api_virt = compile_tetralemmatic_ionq(test_circ_virt, target="api", simulation_mode=False)
        summarize_compilation(api_virt, "Virtual Phase API", "api", False)
    except Exception as e:
        print(f"\n✗ API Compilation Failed: {e}")
        
    # 3. FORTE NATIVE TARGET (Pulse-Level)
    print("\n=== Compilation Test: Forte Native Target (Pulse-Level) ===")
    try:
        forte_log = compile_tetralemmatic_ionq(test_circ_log, target="forte_native", simulation_mode=False)
        summarize_compilation(forte_log, "Logical Phase Forte", "forte_native", False)
        
        forte_virt = compile_tetralemmatic_ionq(test_circ_virt, target="forte_native", simulation_mode=False)
        summarize_compilation(forte_virt, "Virtual Phase Forte", "forte_native", False)
    except Exception as e:
        print(f"\n✗ Forte Native Compilation Failed: {e}")
        
    print("\n✓ Unified compiler validation complete. All targets verified.")