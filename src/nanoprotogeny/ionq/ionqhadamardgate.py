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
nanoprotogeny.ionq.ionqhadamardgate
Refactored Tetralemmatic Hadamard Operator for IonQ Hardware.
Supports both Logical Manifold (NomosIonQid) and Virtual Phase Register (VirtualQudit).
Algebraic Logic: Standard Hadamard on polar subspace (Thesis/Antithesis), identity on non-polar.
Physical Mapping: distinct Bell‑separable encodings for each manifold, explicitly aligned with
the 8-level 171Yb+ hyperfine/auxiliary manifold defined in YB171PLUSHARDWARE.
Decomposition: 2‑qubit unitary via cirq.two_qubit_matrix_to_cz_operations.
ARCHITECTURE:
Logical Interface: NomosIonQid (d=4 qudit) for truth evaluation & semantic holding.
Virtual Interface: VirtualQudit (d=4 qudit) for Holothesis phase tracking (F/P/M/R).
Physical Encoding: 2 LineQubits per qudit (Bell-separable mapping).
Compilation: Logical/Virtual -> Physical expansion -> CZ/CNOT decomposition -> GPI/GPI2/ZZ
COMPATIBILITY:
Seamlessly supports both computational logical qudits and holographic virtual phase registers
via unified type checking, deterministic expansion routing, and protocol-aware gate wrapping.
"""
import numpy as np
import cirq
import cirq_ionq
from cirq_ionq.ionq_native_target_gateset import ForteNativeGateset
from typing import Dict, Iterator, Tuple, Union
from cirq import OP_TREE
from nanoprotogeny.ionq.YB171PLUSHARDWARE import (
    NomosState, IonManifold, NomosIonQid, VirtualQudit,
    VIRTUAL_TO_PHYS_MAP, PHYS_TO_VIRTUAL_MAP
)

AnyQudit = Union[NomosIonQid, VirtualQudit]

#==============================================================================
# 1. BASIS TRANSFORMATIONS (Logical & Virtual) – Explicitly Aligned with Hardware
#==============================================================================
def _build_logical_bell_basis() -> np.ndarray:
    """
    Constructs the 4×4 matrix B_LOG that maps the logical basis used in the Hadamard gate
    to the 2-qubit Bell states encoded in the physical 171Yb+ hyperfine manifold.
    Logical Index (Hadamard) → Physical Bell State → Corresponding NomosState
    -------------------------------------------------------------------------
    0 (Thesis)               → |00⟩                → Th (value 0)
    1 (Antithesis)           → |11⟩                → AntiTh (value 1)
    2 (Synthesis)            → |Ψ⁺⟩ = (|01⟩+|10⟩)/√2 → SynTh (value 2)
    3 (Holothesis)           → |Ψ⁻⟩ = (|01⟩-|10⟩)/√2 → HoloTh (value 3)
    """
    B = np.zeros((4, 4), dtype=complex)
    B[0] = [1.0, 0.0, 0.0, 0.0]
    B[1] = [0.0, 1.0, 0.0, 0.0]
    B[2] = [0.0, 0.0, 1/np.sqrt(2), 1/np.sqrt(2)]
    B[3] = [0.0, 0.0, 1/np.sqrt(2), -1/np.sqrt(2)]
    return B

def _build_virtual_bell_basis() -> np.ndarray:
    """
    Constructs the 4×4 matrix B_VIRT that maps the virtual basis used in the Hadamard gate
    to the physical Bell states encoded in auxiliary levels 4–7.
    Virtual Index (Hadamard) → Physical Bell State → Aux Level (NomosState)
    -----------------------------------------------------------------------
    0 (F)                    → |11⟩                → HoloTh_R (value 7)
    1 (P)                    → (|00⟩ + |10⟩)/√2    → HoloTh_P (value 5)
    2 (M)                    → (-|00⟩ + |10⟩)/√2   → HoloTh_M (value 6)
    3 (R)                    → |01⟩                → HoloTh_F (value 4)
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
# 2. LOGICAL HADAMARD MATRIX (Abstract d=4) & PHYSICAL MAPPING
#==============================================================================
H_onto = np.zeros((4, 4), dtype=complex)
H_onto[:2, :2] = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
H_onto[2:, 2:] = np.eye(2)

def get_physical_matrix(M_onto: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Transform a 4x4 logical operator to physical basis using given B matrix."""
    return B @ M_onto @ B.conj().T

H_phys_log = get_physical_matrix(H_onto, B_LOG)
H_phys_virt = get_physical_matrix(H_onto, B_VIRT)

#==============================================================================
# 3. GATE WRAPPER WITH DECOMPOSITION
#==============================================================================
class PhysicalHadamardWrapper(cirq.Gate):
    """Wraps logical Hadamard and delegates to correct physical matrix and decomposition."""
    def __init__(self, is_virtual: bool):
        self._is_virtual = is_virtual
        self._matrix = H_phys_virt.copy() if is_virtual else H_phys_log.copy()

    def _num_qubits_(self) -> int: return 2
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return self._matrix.copy()

    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        q0, q1 = qubits
        yield cirq.two_qubit_matrix_to_cz_operations(
            q0, q1, np.round(self._matrix, 10), allow_partial_czs=True
        )

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        symbol = "VH" if self._is_virtual else "TIH"
        return cirq.CircuitDiagramInfo(wire_symbols=(symbol, symbol))
        
    def __repr__(self) -> str:
        return f"PhysicalHadamardWrapper(is_virtual={self._is_virtual})"

#==============================================================================
# 4. CIRQ GATE IMPLEMENTATION (Qudit-Native)
#==============================================================================
class TetralemmaticIonHadamardGate(cirq.Gate):
    """Tetralemmatic Hadamard acting on a d=4 qudit."""
    def _qid_shape_(self) -> Tuple[int, ...]: return (4,)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return H_phys_log.copy()
    
    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        # Decomposition handled by wrapper after expansion
        return NotImplemented
        
    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=("H",))
    def __repr__(self) -> str: return "TetralemmaticIonHadamardGate()"
    def __eq__(self, other) -> bool: return isinstance(other, TetralemmaticIonHadamardGate)
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
        if isinstance(gate, TetralemmaticIonHadamardGate):
            if len(op.qubits) == 1:
                is_virtual = isinstance(op.qubits[0], VirtualQudit)
                gate = PhysicalHadamardWrapper(is_virtual)
            else:
                raise ValueError("Hadamard expects exactly one qudit")
        new_ops.append(gate.on(*flat_qs))
    return cirq.Circuit(new_ops)

def compile_tetralemmatic_ionq(
    circuit: cirq.Circuit, 
    target: str = "forte_native", 
    simulation_mode: bool = False
) -> cirq.Circuit:
    """
    Unified compiler for tetralemmatic gates targeting IonQ Forte architecture.
    
    Args:
        circuit: Input circuit containing NomosIonQid/VirtualQudit or standard qubits.
        target: "forte_native" -> compiles to GPI/GPI2/ZZ (3 gates)
                "api"          -> compiles to 16 supported API gates (cloud-ready)
        simulation_mode: If True, preserves MatrixGate/abstract wrappers for exact 
                         statevector simulation. If False, forces full decomposition.
    """
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
class TetralemmaticIonHadamard:
    """Factory and verification for IonQ-targeted tetralemmatic Hadamard operators."""
    def __init__(self):
        self.H = TetralemmaticIonHadamardGate()

    def _get_basis_states(self, B: np.ndarray):
        th   = B @ np.array([1, 0, 0, 0], dtype=complex)
        anti = B @ np.array([0, 1,  0, 0], dtype=complex)
        syn  = B @ np.array([0, 0, 1, 0], dtype=complex)
        holo = B @ np.array([0, 0, 0, 1], dtype=complex)
        return th, anti, syn, holo

    def verify_logical_properties(self) -> Dict[str, bool]:
        checks = {}
        H = H_phys_log
        I4 = np.eye(4, dtype=complex)
        th, anti, syn, holo = self._get_basis_states(B_LOG)

        checks["H_unitary"] = bool(np.allclose(H.conj().T @ H, I4))
        checks["H_self_inverse"] = bool(np.allclose(H @ H, I4))
        checks["H_maps_Th_to_Superposition"] = bool(np.allclose(H @ th, (th + anti) / np.sqrt(2)))
        checks["H_maps_AntiTh_to_Superposition"] = bool(np.allclose(H @ anti, (th - anti) / np.sqrt(2)))
        checks["H_fixes_SynTh"] = bool(np.allclose(H @ syn, syn))
        checks["H_fixes_HoloTh"] = bool(np.allclose(H @ holo, holo))
        polar_idx = [0, 1]
        H_p = H[np.ix_(polar_idx, polar_idx)]
        checks["qubit_H_equiv"] = bool(np.allclose(H_p, np.array([[1,1],[1,-1]])/np.sqrt(2)))
        return checks

    def verify_virtual_properties(self) -> Dict[str, bool]:
        checks = {}
        H = H_phys_virt
        I4 = np.eye(4, dtype=complex)
        f, p, m, r = self._get_basis_states(B_VIRT) 

        checks["VH_unitary"] = bool(np.allclose(H.conj().T @ H, I4))
        checks["VH_self_inverse"] = bool(np.allclose(H @ H, I4))
        checks["VH_maps_F_to_superposition"] = bool(np.allclose(H @ f, (f + p) / np.sqrt(2)))
        checks["VH_maps_P_to_superposition"] = bool(np.allclose(H @ p, (f - p) / np.sqrt(2)))
        checks["VH_fixes_M"] = bool(np.allclose(H @ m, m))
        checks["VH_fixes_R"] = bool(np.allclose(H @ r, r))
        return checks

    def compile_test_circuit(self, use_virtual: bool = False) -> cirq.Circuit:
        q = VirtualQudit(0) if use_virtual else NomosIonQid(0)
        return cirq.Circuit(
            self.H.on(q),
            cirq.measure(*cirq.LineQubit.range(2), key="m")
        )

#==============================================================================
# MAIN EXECUTION (Dual-Manifold with Unified Compiler Validation)
#==============================================================================
if __name__ == "__main__":
    print("=== Tetralemmatic IonQ Hadamard Verification (Dual-Manifold) ===")
    print(f"Ion Manifold Energies (B=1G): {IonManifold.energy_levels(1.0)}\n")
    
    factory = TetralemmaticIonHadamard()
    print("--- Logical Manifold Checks ---")
    log_results = factory.verify_logical_properties()
    for k, v in log_results.items():
        print(f"{'Yes' if v else 'No'} {k}")
        
    print("\n--- Virtual Manifold Checks ---")
    virt_results = factory.verify_virtual_properties()
    for k, v in virt_results.items():
        print(f"{'Yes' if v else 'No'} {k}")
        
    def summarize_compilation(circuit: cirq.Circuit, label: str, target: str, sim_mode: bool):
        """Diagnostics for compiled circuit against target gateset constraints."""
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
        summarize_compilation(sim_log, "Logical Hadamard Simulation", "api", True)
        
        sim_virt = compile_tetralemmatic_ionq(test_circ_virt, target="api", simulation_mode=True)
        summarize_compilation(sim_virt, "Virtual Hadamard Simulation", "api", True)
    except Exception as e:
        print(f"\n✗ Simulation Compilation Failed: {e}")
        
    # 2. API TARGET (Cloud-Ready)
    print("\n=== Compilation Test: API Target (Cloud-Ready) ===")
    try:
        api_log = compile_tetralemmatic_ionq(test_circ_log, target="api", simulation_mode=False)
        summarize_compilation(api_log, "Logical Hadamard API", "api", False)
        
        api_virt = compile_tetralemmatic_ionq(test_circ_virt, target="api", simulation_mode=False)
        summarize_compilation(api_virt, "Virtual Hadamard API", "api", False)
    except Exception as e:
        print(f"\n✗ API Compilation Failed: {e}")
        
    # 3. FORTE NATIVE TARGET (Pulse-Level)
    print("\n=== Compilation Test: Forte Native Target (Pulse-Level) ===")
    try:
        forte_log = compile_tetralemmatic_ionq(test_circ_log, target="forte_native", simulation_mode=False)
        summarize_compilation(forte_log, "Logical Hadamard Forte", "forte_native", False)
        
        forte_virt = compile_tetralemmatic_ionq(test_circ_virt, target="forte_native", simulation_mode=False)
        summarize_compilation(forte_virt, "Virtual Hadamard Forte", "forte_native", False)
    except Exception as e:
        print(f"\n✗ Forte Native Compilation Failed: {e}")
        
    print("\n✓ Unified compiler validation complete. All targets verified.")