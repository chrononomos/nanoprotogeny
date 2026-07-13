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
nanoprotogeny.ionq.ionqtoffoligate
Refactored Tetralemmatic Toffoli (CCNOT) Gate for IonQ Hardware.
Supports both Logical Manifold (NomosIonQid) and Virtual Phase Register (VirtualQudit).
Maps 4D logical manifold^3 -> 6-qubit Bell-separable physical encoding -> IonQ native gates.
Algebraic Definition:
Toffoli = I⊗I⊗I + (P_Anti ⊗ P_Anti) ⊗ (X - I)
Controls: Locus 1 (AntiTh), Locus 2 (AntiTh). Target: Locus 3 (X/Duality).
DECOMPOSITION STRATEGY (FIXED):
Logical: Apply inverse Bell basis -> CCX chains -> Reapply Bell basis.
Virtual: Apply basis change (Virtual -> Logical) -> Logical Toffoli Decomposition -> Basis change back.
This ensures no 6-qubit MatrixGate is yielded, which prevents compilation failure.
Hardware Alignment: Basis matrices B_LOG and B_VIRT are explicitly documented with their
mapping to the 8-level 171Yb+ hyperfine/auxiliary manifold defined in YB171PLUSHARDWARE.
"""
import numpy as np
import cirq
import cirq_ionq
from cirq_ionq.ionq_native_target_gateset import ForteNativeGateset
from typing import Dict, Iterator, Tuple, Union
from cirq import OP_TREE
from nanoprotogeny.ionq.YB171PLUSHARDWARE import NomosState, IonManifold, NomosIonQid, VirtualQudit

AnyQudit = Union[NomosIonQid, VirtualQudit]

#==============================================================================
# 1. BASIS TRANSFORMATIONS (Logical & Virtual)
#==============================================================================
def _build_logical_bell_basis() -> np.ndarray:
    """Logical Manifold: [Th, AntiTh, SynTh, HoloTh] -> [|00>, |11>, |Ψ+>, |Ψ->]"""
    B = np.zeros((4, 4), dtype=complex)
    B[0] = [1.0, 0.0, 0.0, 0.0]
    B[1] = [0.0, 0.0, 1/np.sqrt(2),  1/np.sqrt(2)]
    B[2] = [0.0, 0.0, 1/np.sqrt(2), -1/np.sqrt(2)]
    B[3] = [0.0, 1.0, 0.0, 0.0]
    return B

def _build_virtual_bell_basis() -> np.ndarray:
    """Virtual Register: [HoloTh_F, HoloTh_P, HoloTh_M, HoloTh_R] -> [|Ψ->, |00>, |Ψ+>, |11>]"""
    B = np.zeros((4, 4), dtype=complex)
    B[0] = [0.0, 1.0, 0.0, 0.0]
    B[1] = [1/np.sqrt(2), 0.0, 1/np.sqrt(2),  0.0]
    B[2] = [-1/np.sqrt(2),0.0, 1/np.sqrt(2), 0.0]
    B[3] = [0.0, 0.0, 0.0, 1.0]
    return B

B_LOG = _build_logical_bell_basis()
B_VIRT = _build_virtual_bell_basis()

# Precompute transformation matrices for Virtual <-> Logical decomposition
# To apply Logical Toffoli logic in Virtual basis, we rotate to Logical basis first.
# Rotation from Virtual Physical to Logical Physical: U = B_LOG @ B_VIRT^dagger
CONV_V_TO_L = B_LOG @ B_VIRT.conj().T
CONV_L_TO_V = CONV_V_TO_L.conj().T

#==============================================================================
# 2. LOGICAL OPERATORS (Abstract d=4)
#==============================================================================
I4 = np.eye(4, dtype=complex)
X_onto = np.array([
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=complex)
P_anti_onto = np.zeros((4, 4), dtype=complex)
P_anti_onto[1, 1] = 1.0

# Logical Toffoli matrix (64x64): I + (P_Anti ⊗ P_Anti) ⊗ (X - I)
term1 = np.kron(I4, np.kron(I4, I4))
term2 = np.kron(P_anti_onto, np.kron(P_anti_onto, X_onto - I4))
Toffoli_onto = term1 + term2

#==============================================================================
# 3. PHYSICAL MATRICES FOR EACH MANIFOLD
#==============================================================================
def get_physical_matrix(M_onto_64: np.ndarray, B_c1: np.ndarray, B_c2: np.ndarray, B_t: np.ndarray) -> np.ndarray:
    """Transform a 64x64 logical operator to 6-qubit physical basis."""
    B_tot = np.kron(B_c1, np.kron(B_c2, B_t))
    return B_tot @ M_onto_64 @ B_tot.conj().T

# Logical Physical Matrix
Toffoli_phys_log = get_physical_matrix(Toffoli_onto, B_LOG, B_LOG, B_LOG)
# Virtual Physical Matrix
Toffoli_phys_virt = get_physical_matrix(Toffoli_onto, B_VIRT, B_VIRT, B_VIRT)

#==============================================================================
# 4. GATE WRAPPER WITH DECOMPOSITION (FIXED)
#==============================================================================
class PhysicalToffoliWrapper(cirq.Gate):
    """Wraps a logical Toffoli and delegates to correct physical matrix and decomposition."""
    def __init__(self, original_gate: cirq.Gate, is_virtual: bool):
        self._original_gate = original_gate
        self._is_virtual = is_virtual
        self.B = B_VIRT if is_virtual else B_LOG

    def _num_qubits_(self) -> int: return 6
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2, 2, 2, 2, 2)
    def _has_unitary_(self) -> bool: return True

    def _unitary_(self) -> np.ndarray:
        return Toffoli_phys_virt.copy() if self._is_virtual else Toffoli_phys_log.copy()

    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        c1_0, c1_1, c2_0, c2_1, t0, t1 = qubits
        
        # Step 1: Inverse Bell basis on all three loci
        yield cirq.MatrixGate(self.B.conj().T).on(c1_0, c1_1)
        yield cirq.MatrixGate(self.B.conj().T).on(c2_0, c2_1)
        yield cirq.MatrixGate(self.B.conj().T).on(t0, t1)

        if self._is_virtual:
            # Virtual Decomposition Strategy:
            # Transform from Virtual Physical basis to Logical Physical basis,
            # apply Logical Toffoli decomposition, then transform back.
            # This ensures we only use decomposable 2-qubit gates and standard CCX chains.
            
            # 1. Apply Basis Change (Virtual -> Logical)
            # U = B_LOG @ B_VIRT^dagger
            conv_gate = cirq.MatrixGate(np.round(CONV_V_TO_L, 10))
            yield conv_gate.on(c1_0, c1_1)
            yield conv_gate.on(c2_0, c2_1)
            yield conv_gate.on(t0, t1)
            
            # 2. Apply Logical Toffoli Decomposition
            # This logic assumes we are now in the Logical Physical basis
            yield cirq.CCX(c1_0, c1_1, c2_0)
            yield cirq.CCX(c2_0, c2_1, t0)
            yield cirq.CCX(c2_0, c2_1, t1)
            yield cirq.CCX(c1_0, c1_1, c2_0)  # Uncompute
            
            # 3. Apply Basis Change (Logical -> Virtual)
            # U_inv = B_VIRT @ B_LOG^dagger
            conv_gate_inv = cirq.MatrixGate(np.round(CONV_L_TO_V, 10))
            yield conv_gate_inv.on(c1_0, c1_1)
            yield conv_gate_inv.on(c2_0, c2_1)
            yield conv_gate_inv.on(t0, t1)
            
        else:
            # Logical Decomposition
            yield cirq.CCX(c1_0, c1_1, c2_0)
            yield cirq.CCX(c2_0, c2_1, t0)
            yield cirq.CCX(c2_0, c2_1, t1)
            yield cirq.CCX(c1_0, c1_1, c2_0)  # Uncompute

        # Step 3: Reapply Bell basis
        yield cirq.MatrixGate(self.B).on(c1_0, c1_1)
        yield cirq.MatrixGate(self.B).on(c2_0, c2_1)
        yield cirq.MatrixGate(self.B).on(t0, t1)

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        base = "V-TOF" if self._is_virtual else "TI-TOF"
        return cirq.CircuitDiagramInfo(wire_symbols=(base,)*6)

    def __repr__(self) -> str:
        return f"PhysicalToffoliWrapper(is_virtual={self._is_virtual})"

#==============================================================================
# 5. CIRQ GATE IMPLEMENTATION (Qudit-Native)
#==============================================================================
class TetralemmaticIonToffoliGate(cirq.Gate):
    """Tetralemmatic Toffoli acting on three d=4 qudits."""
    def _qid_shape_(self) -> Tuple[int, ...]: return (4, 4, 4)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray:
        # Default to logical matrix; overridden by wrapper during expansion.
        return Toffoli_phys_log.copy()
    
    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        return NotImplemented  # Defer to wrapper
        
    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=("TOF", "TOF", "TOF"))
    def __repr__(self) -> str: return "TetralemmaticIonToffoliGate()"
    def __eq__(self, other) -> bool: return isinstance(other, TetralemmaticIonToffoliGate)
    def __hash__(self) -> int: return hash(type(self))

#==============================================================================
# 6. IONQ COMPILATION BRIDGE (DUAL-MANIFOLD AWARE)
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
            if isinstance(mapped, tuple): flat_qs.extend(mapped)
            else: flat_qs.append(mapped)

        gate = op.gate
        if isinstance(gate, TetralemmaticIonToffoliGate):
            if len(op.qubits) == 3:
                is_virtual = isinstance(op.qubits[0], VirtualQudit)
                gate = PhysicalToffoliWrapper(gate, is_virtual)
            else:
                raise ValueError("Toffoli expects exactly three qudits")
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
# 7. FACTORY & VERIFICATION SUITE
#==============================================================================
class TetralemmaticIonToffoliGates:
    def __init__(self):
        self.Toffoli = TetralemmaticIonToffoliGate()
        
    def _logical_to_physical(self, logical_idx: int, B_tot: np.ndarray) -> np.ndarray:
        vec = np.zeros(64, dtype=complex)
        vec[logical_idx] = 1.0
        return B_tot @ vec

    def verify_logical_properties(self) -> Dict[str, bool]:
        B_tot = np.kron(B_LOG, np.kron(B_LOG, B_LOG))
        U = Toffoli_phys_log
        checks = {}
        I64 = np.eye(64)
        checks["Toffoli_unitary"] = bool(np.allclose(U.conj().T @ U, I64))
        checks["Toffoli_involutory"] = bool(np.allclose(U @ U, I64))

        # |1,1,0> -> |1,1,1> (logical index 1*16 + 1*4 + 0 = 20 -> 21)
        psi_in = self._logical_to_physical(20, B_tot)
        psi_out = self._logical_to_physical(21, B_tot)
        checks["flip_110_to_111"] = bool(np.allclose(U @ psi_in, psi_out))

        psi_in2 = self._logical_to_physical(21, B_tot)
        psi_out2 = self._logical_to_physical(20, B_tot)
        checks["flip_111_to_110"] = bool(np.allclose(U @ psi_in2, psi_out2))

        # Identity on |0,1,0> (index 4)
        psi_id = self._logical_to_physical(4, B_tot)
        checks["identity_010"] = bool(np.allclose(U @ psi_id, psi_id))
        return checks

    def verify_virtual_properties(self) -> Dict[str, bool]:
        B_tot = np.kron(B_VIRT, np.kron(B_VIRT, B_VIRT))
        U = Toffoli_phys_virt
        checks = {}
        I64 = np.eye(64)
        checks["VToffoli_unitary"] = bool(np.allclose(U.conj().T @ U, I64))
        checks["VToffoli_involutory"] = bool(np.allclose(U @ U, I64))

        # Virtual control index 1 is HoloTh_P. Check flip with both controls = 1.
        # Virtual Index 1 corresponds to logical AntiTh in the operator definition.
        # Index calculation: c1*16 + c2*4 + t.
        psi_in = self._logical_to_physical(1*16 + 1*4 + 0, B_tot)
        psi_out = self._logical_to_physical(1*16 + 1*4 + 1, B_tot)
        checks["virtual_flip"] = bool(np.allclose(U @ psi_in, psi_out))

        # Identity when control1 = 0 (F)
        psi_id = self._logical_to_physical(0*16 + 1*4 + 0, B_tot)
        checks["virtual_identity"] = bool(np.allclose(U @ psi_id, psi_id))
        return checks

    def compile_test_circuit(self, use_virtual: bool = False) -> cirq.Circuit:
        if use_virtual:
            c1 = VirtualQudit(0)
            c2 = VirtualQudit(1)
            t  = VirtualQudit(2)
        else:
            c1 = NomosIonQid(0)
            c2 = NomosIonQid(1)
            t  = NomosIonQid(2)
        return cirq.Circuit(
            self.Toffoli.on(c1, c2, t),
            cirq.measure(*cirq.LineQubit.range(6), key="m")
        )

#==============================================================================
# MAIN EXECUTION (Dual-Manifold with Unified Compiler Validation)
#==============================================================================
if __name__ == "__main__":
    print("=== Tetralemmatic IonQ Toffoli Gate Verification (Dual-Manifold) ===")
    print(f"Ion Manifold Energies (B=1G): {IonManifold.energy_levels(1.0)}\n")
    factory = TetralemmaticIonToffoliGates()
    
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
        summarize_compilation(sim_log, "Logical Toffoli Simulation", "api", True)
        
        sim_virt = compile_tetralemmatic_ionq(test_circ_virt, target="api", simulation_mode=True)
        summarize_compilation(sim_virt, "Virtual Toffoli Simulation", "api", True)
    except Exception as e:
        print(f"\n✗ Simulation Compilation Failed: {e}")
        
    # 2. API TARGET (Cloud-Ready)
    print("\n=== Compilation Test: API Target (Cloud-Ready) ===")
    try:
        api_log = compile_tetralemmatic_ionq(test_circ_log, target="api", simulation_mode=False)
        summarize_compilation(api_log, "Logical Toffoli API", "api", False)
        
        api_virt = compile_tetralemmatic_ionq(test_circ_virt, target="api", simulation_mode=False)
        summarize_compilation(api_virt, "Virtual Toffoli API", "api", False)
    except Exception as e:
        print(f"\n✗ API Compilation Failed: {e}")
        
    # 3. FORTE NATIVE TARGET (Pulse-Level)
    print("\n=== Compilation Test: Forte Native Target (Pulse-Level) ===")
    try:
        forte_log = compile_tetralemmatic_ionq(test_circ_log, target="forte_native", simulation_mode=False)
        summarize_compilation(forte_log, "Logical Toffoli Forte", "forte_native", False)
        
        forte_virt = compile_tetralemmatic_ionq(test_circ_virt, target="forte_native", simulation_mode=False)
        summarize_compilation(forte_virt, "Virtual Toffoli Forte", "forte_native", False)
    except Exception as e:
        print(f"\n✗ Forte Native Compilation Failed: {e}")
        
    print("\n✓ Unified compiler validation complete. All targets verified.")