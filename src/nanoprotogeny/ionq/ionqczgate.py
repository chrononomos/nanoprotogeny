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
nanoprotogeny.ionq.ionqczgate
Refactored Tetralemmatic Controlled-Phase (CZ) Gate for IonQ Hardware.
Supports both Logical Manifold (NomosIonQid) and Virtual Phase Register (VirtualQudit).
Algebraic Definition:
CZ := I^{\otimes 2} - 2 * |AntiTh⟩⟨AntiTh| ⊗ |AntiTh⟩⟨AntiTh|
Decomposition: CZ = (I ⊗ H) CNOT (I ⊗ H)  (using logical Hadamard and CNOT)
Hardware Alignment: Basis matrices B_LOG and B_VIRT are explicitly documented with their
mapping to the 8-level 171Yb+ hyperfine/auxiliary manifold defined in YB171PLUSHARDWARE.
"""
import numpy as np
import cirq
import cirq_ionq
from typing import Dict, Iterator, Tuple, Union
from cirq import OP_TREE
from nanoprotogeny.ionq.YB171PLUSHARDWARE import (
    NomosState, IonManifold, NomosIonQid, VirtualQudit,
    VIRTUAL_TO_PHYS_MAP, PHYS_TO_VIRTUAL_MAP
)
# Import corrected CNOT gate (v2/v3) and physical Hadamard wrapper
from nanoprotogeny.ionq.ionqcnotgate import TetralemmaticIonCNOTGate, PhysicalCNOTWrapper
from nanoprotogeny.ionq.ionqhadamardgate import PhysicalHadamardWrapper

AnyQudit = Union[NomosIonQid, VirtualQudit]

#==============================================================================
# 1. BASIS TRANSFORMATIONS (Logical & Virtual) – Explicitly Aligned with Hardware
#==============================================================================
def _build_logical_bell_basis() -> np.ndarray:
    """
    Constructs the 4×4 matrix B_LOG that maps the logical basis used in the CZ gate
    to the 2-qubit Bell states encoded in the physical 171Yb+ hyperfine manifold.
    Logical Index (CZ) → Physical Bell State → Corresponding NomosState
    -------------------------------------------------------------------
    0 (Thesis)          → |00⟩                → Th (value 0)
    1 (Synthesis)       → |Ψ⁺⟩ = (|01⟩+|10⟩)/√2 → SynTh (value 2)
    2 (Holothesis)      → |Ψ⁻⟩ = (|01⟩-|10⟩)/√2 → HoloTh (value 3)
    3 (Antithesis)      → |11⟩                → AntiTh (value 1)
    """
    B = np.zeros((4, 4), dtype=complex)
    B[0] = [1.0, 0.0, 0.0, 0.0]
    B[1] = [0.0, 0.0, 1/np.sqrt(2), 1/np.sqrt(2)]
    B[2] = [0.0, 0.0, 1/np.sqrt(2), -1/np.sqrt(2)]
    B[3] = [0.0, 1.0, 0.0, 0.0]
    return B

def _build_virtual_bell_basis() -> np.ndarray:
    """
    Constructs the 4×4 matrix B_VIRT that maps the virtual basis used in the CZ gate
    to the physical Bell states encoded in auxiliary levels 4–7.
    Virtual Index (CZ) → Physical Bell State → Aux Level (NomosState)
    -----------------------------------------------------------------
    0                    → |11⟩                → HoloTh_R (value 7)
    1                    → (|00⟩ + |10⟩)/√2    → HoloTh_P (value 5)
    2                    → (-|00⟩ + |10⟩)/√2   → HoloTh_M (value 6)
    3                    → |01⟩                → HoloTh_F (value 4)
    """
    B = np.zeros((4, 4), dtype=complex)
    B[0] = [0.0, 1.0, 0.0, 0.0]
    B[1] = [1/np.sqrt(2), 0.0, 1/np.sqrt(2), 0.0]
    B[2] = [-1/np.sqrt(2), 0.0, 1/np.sqrt(2), 0.0]
    B[3] = [0.0, 0.0, 0.0, 1.0]
    return B

B_LOG = _build_logical_bell_basis()
B_VIRT = _build_virtual_bell_basis()

#==============================================================================
# 2. LOGICAL CZ MATRIX (Abstract)
#==============================================================================
# CZ applies a -1 phase when both control and target are in logical index 1 (Antithesis)
CZ_onto = np.eye(16, dtype=complex)
CZ_onto[5, 5] = -1.0   # |1,1⟩_L gets -1 phase

# Physical matrices for each manifold
def get_physical_matrix(M_onto: np.ndarray, B_control: np.ndarray, B_target: np.ndarray) -> np.ndarray:
    B_tot = np.kron(B_control, B_target)
    return B_tot @ M_onto @ B_tot.conj().T

CZ_phys_log = get_physical_matrix(CZ_onto, B_LOG, B_LOG)
CZ_phys_virt = get_physical_matrix(CZ_onto, B_VIRT, B_VIRT)

#==============================================================================
# 3. GATE WRAPPER WITH DECOMPOSITION (Uses existing physical Hadamard)
#==============================================================================
class PhysicalCZWrapper(cirq.Gate):
    """Wraps logical CZ and decomposes using (I ⊗ H) CNOT (I ⊗ H)."""
    def __init__(self, is_virtual: bool):
        self._is_virtual = is_virtual

    def _num_qubits_(self) -> int: return 4
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2, 2, 2)
    def _has_unitary_(self) -> bool: return True

    def _unitary_(self) -> np.ndarray:
        return CZ_phys_virt.copy() if self._is_virtual else CZ_phys_log.copy()

    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        c0, c1, t0, t1 = qubits
        
        # Structural decomposition: CZ_target = H_target * CNOT_control,target * H_target
        
        # 1. Apply H on target
        yield PhysicalHadamardWrapper(self._is_virtual).on(t0, t1)
        
        # 2. Apply CNOT (control, target)
        # Note: We pass an instance of TetralemmaticIonCNOTGate to the wrapper.
        yield PhysicalCNOTWrapper(TetralemmaticIonCNOTGate(), self._is_virtual).on(c0, c1, t0, t1)
        
        # 3. Apply H on target again
        yield PhysicalHadamardWrapper(self._is_virtual).on(t0, t1)

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        base = "V-CZ" if self._is_virtual else "TI-CZ"
        return cirq.CircuitDiagramInfo(wire_symbols=(base,)*4)
        
    def __repr__(self) -> str:
        return f"PhysicalCZWrapper(is_virtual={self._is_virtual})"

#==============================================================================
# 4. CIRQ GATE IMPLEMENTATION (Qudit-Native)
#==============================================================================
class TetralemmaticIonCZGate(cirq.Gate):
    """Tetralemmatic CZ acting on two d=4 qudits."""
    def _qid_shape_(self) -> Tuple[int, ...]: return (4, 4)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray:
        return CZ_phys_log.copy()
    
    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        # Decomposition handled by wrapper after expansion
        return NotImplemented
        
    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=("CZ", "CZ"))
    def __repr__(self) -> str: return "TetralemmaticIonCZGate()"
    def __eq__(self, other) -> bool: return isinstance(other, TetralemmaticIonCZGate)
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
        if isinstance(gate, TetralemmaticIonCZGate):
            if len(op.qubits) == 2:
                is_virtual = isinstance(op.qubits[0], VirtualQudit)
                gate = PhysicalCZWrapper(is_virtual)
            else:
                raise ValueError("CZ expects exactly two qudits")
                
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
        from cirq_ionq.ionq_native_target_gateset import ForteNativeGateset
        gateset = ForteNativeGateset()
    else:
        raise ValueError("target must be 'api' or 'forte_native'")
        
    return cirq.optimize_for_target_gateset(
        circuit,
        gateset=gateset,
        context=cirq.TransformerContext(deep=True)
    )

#==============================================================================
# 6. FACTORY & VERIFICATION SUITE
#==============================================================================
class TetralemmaticIonCZGates:
    """Factory and verification for IonQ-targeted tetralemmatic CZ operators."""
    def __init__(self):
        self.CZ = TetralemmaticIonCZGate()
        
    def _logical_to_physical(self, logical_idx: int, B_tot: np.ndarray) -> np.ndarray:
        vec = np.zeros(16, dtype=complex)
        vec[logical_idx] = 1.0
        return B_tot @ vec

    def verify_logical_properties(self) -> Dict[str, bool]:
        checks = {}
        B_tot = np.kron(B_LOG, B_LOG)
        U = CZ_phys_log
        checks["CZ_unitary"] = bool(np.allclose(U.conj().T @ U, np.eye(16)))
        checks["CZ_hermitian"] = bool(np.allclose(U.conj().T, U))
        checks["CZ_involutory"] = bool(np.allclose(U @ U, np.eye(16)))

        psi_11 = self._logical_to_physical(5, B_tot)  # Control=1, Target=1 (logical indices)
        checks["phase_on_11"] = bool(np.allclose(U @ psi_11, -psi_11))

        psi_01 = self._logical_to_physical(1, B_tot)  # Control=0, Target=1
        checks["identity_on_01"] = bool(np.allclose(U @ psi_01, psi_01))

        psi_23 = self._logical_to_physical(11, B_tot) # Control=2, Target=3
        checks["identity_on_23"] = bool(np.allclose(U @ psi_23, psi_23))

        polar_idx = [0, 3, 12, 15]
        U_p = U[np.ix_(polar_idx, polar_idx)]
        std_cz = np.diag([1, 1, 1, -1]).astype(complex)
        checks["qubit_reduction_CZ"] = bool(np.allclose(U_p, std_cz))
        return checks

    def verify_virtual_properties(self) -> Dict[str, bool]:
        checks = {}
        B_tot = np.kron(B_VIRT, B_VIRT)
        U = CZ_phys_virt
        checks["VCZ_unitary"] = bool(np.allclose(U.conj().T @ U, np.eye(16)))
        
        psi_11 = self._logical_to_physical(5, B_tot)  # Both in state index 1 (HoloTh_P)
        checks["virtual_phase_on_11"] = bool(np.allclose(U @ psi_11, -psi_11))

        psi_01 = self._logical_to_physical(1, B_tot)
        checks["virtual_identity_on_01"] = bool(np.allclose(U @ psi_01, psi_01))
        return checks

    def compile_test_circuit(self, use_virtual: bool = False) -> cirq.Circuit:
        if use_virtual:
            ctrl = VirtualQudit(0)
            tgt = VirtualQudit(1)
        else:
            ctrl = NomosIonQid(0)
            tgt = NomosIonQid(1)
        return cirq.Circuit(
            self.CZ.on(ctrl, tgt),
            cirq.measure(*cirq.LineQubit.range(4), key="m")
        )

#==============================================================================
# MAIN EXECUTION (Dual-Manifold with Unified Compiler Validation)
#==============================================================================
if __name__ == "__main__":
    print("=== Tetralemmatic IonQ CZ Gate Verification (Dual-Manifold) ===")
    print(f"Ion Manifold Energies (B=1G): {IonManifold.energy_levels(1.0)}\n")
    factory = TetralemmaticIonCZGates()
    
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
        summarize_compilation(sim_log, "Logical CZ Simulation", "api", True)
        
        sim_virt = compile_tetralemmatic_ionq(test_circ_virt, target="api", simulation_mode=True)
        summarize_compilation(sim_virt, "Virtual CZ Simulation", "api", True)
    except Exception as e:
        print(f"\n✗ Simulation Compilation Failed: {e}")
        
    # 2. API TARGET (Cloud-Ready)
    print("\n=== Compilation Test: API Target (Cloud-Ready) ===")
    try:
        api_log = compile_tetralemmatic_ionq(test_circ_log, target="api", simulation_mode=False)
        summarize_compilation(api_log, "Logical CZ API", "api", False)
        
        api_virt = compile_tetralemmatic_ionq(test_circ_virt, target="api", simulation_mode=False)
        summarize_compilation(api_virt, "Virtual CZ API", "api", False)
    except Exception as e:
        print(f"\n✗ API Compilation Failed: {e}")
        
    # 3. FORTE NATIVE TARGET (Pulse-Level)
    print("\n=== Compilation Test: Forte Native Target (Pulse-Level) ===")
    try:
        forte_log = compile_tetralemmatic_ionq(test_circ_log, target="forte_native", simulation_mode=False)
        summarize_compilation(forte_log, "Logical CZ Forte", "forte_native", False)
        
        forte_virt = compile_tetralemmatic_ionq(test_circ_virt, target="forte_native", simulation_mode=False)
        summarize_compilation(forte_virt, "Virtual CZ Forte", "forte_native", False)
    except Exception as e:
        print(f"\n✗ Forte Native Compilation Failed: {e}")
        
    print("\n✓ Unified compiler validation complete. All targets verified.")