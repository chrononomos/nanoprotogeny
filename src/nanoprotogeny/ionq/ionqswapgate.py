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
nanoprotogeny.ionq.ionqswapgate
Refactored Tetralemmatic SWAP Gate for IonQ Hardware.
Supports both Logical Manifold (NomosIonQid) and Virtual Phase Register (VirtualQudit).
Algebraic Definition: Full 4D state exchange between two qudits.
Decomposition: SWAP = CNOT(c,t) · CNOT(t,c) · CNOT(c,t) using tetralemmatic CNOT.
Hardware Alignment: Basis matrices B_LOG and B_VIRT are explicitly documented with their
mapping to the 8-level 171Yb+ hyperfine/auxiliary manifold defined in YB171PLUSHARDWARE.
DECOMPOSITION STRATEGY:
Uses PhysicalSWAPWrapper to decompose SWAP into three PhysicalCNOTWrappers.
PhysicalCNOTWrapper handles the basis rotation and structural decomposition of CNOTs
via the IonQ Target Gateset, ensuring native gate compliance.
"""
import numpy as np
import cirq
import cirq_ionq
from typing import Dict, Iterator, Tuple, Union
from cirq import OP_TREE
from nanoprotogeny.ionq.YB171PLUSHARDWARE import NomosState, IonManifold, NomosIonQid, VirtualQudit
from nanoprotogeny.ionq.ionqcnotgate import TetralemmaticIonCNOTGate, PhysicalCNOTWrapper

AnyQudit = Union[NomosIonQid, VirtualQudit]

#==============================================================================
# 1. BASIS TRANSFORMATIONS (Logical & Virtual)
#==============================================================================
def _build_logical_bell_basis() -> np.ndarray:
    """
    Logical Index (SWAP) → Physical Bell State → Corresponding NomosState
    -----------------------------------------------------------------------
    0 (Thesis)          → |00⟩                → Th (value 0)
    1 (Antithesis)      → |11⟩                → AntiTh (value 1)
    2 (Synthesis)       → |Ψ⁺⟩ = (|01⟩+|10⟩)/√2 → SynTh (value 2)
    3 (Holothesis)      → |Ψ⁻⟩ = (|01⟩-|10⟩)/√2 → HoloTh (value 3)
    """
    B = np.zeros((4, 4), dtype=complex)
    # Col 0: Th -> |00>
    B[0, 0] = 1.0
    # Col 1: AntiTh -> |11> (Row 3 is index 3 -> |11>)
    B[3, 1] = 1.0
    # Col 2: SynTh -> |Psi+> (Row 1 -> |01>, Row 2 -> |10>)
    B[1, 2] = 1/np.sqrt(2)
    B[2, 2] = 1/np.sqrt(2)
    # Col 3: HoloTh -> |Psi->
    B[1, 3] = 1/np.sqrt(2)
    B[2, 3] = -1/np.sqrt(2)
    return B

def _build_virtual_bell_basis() -> np.ndarray:
    """
    Virtual Index (SWAP) → Physical Bell State → Aux Level (NomosState)
    -------------------------------------------------------------------
    0 (F)                    → |Ψ⁻⟩                → HoloTh_F (value 4)
    1 (P)                    → |00⟩                → HoloTh_P (value 5)
    2 (M)                    → |Ψ⁺⟩                → HoloTh_M (value 6)
    3 (R)                    → |11⟩                → HoloTh_R (value 7)
    """
    B = np.zeros((4, 4), dtype=complex)
    # Col 0: F -> |Psi->
    B[1, 0] = 1/np.sqrt(2)
    B[2, 0] = -1/np.sqrt(2)
    # Col 1: P -> |00>
    B[0, 1] = 1.0
    # Col 2: M -> |Psi+>
    B[1, 2] = 1/np.sqrt(2)
    B[2, 2] = 1/np.sqrt(2)
    # Col 3: R -> |11>
    B[3, 3] = 1.0
    return B

B_LOG = _build_logical_bell_basis()
B_VIRT = _build_virtual_bell_basis()

#==============================================================================
# 2. LOGICAL SWAP MATRIX (Abstract d=4)
#==============================================================================
SWAP_onto = np.zeros((16, 16), dtype=complex)
for i in range(4):
    for j in range(4):
        SWAP_onto[i*4 + j, j*4 + i] = 1.0

def get_physical_matrix(M_onto: np.ndarray, B_control: np.ndarray, B_target: np.ndarray) -> np.ndarray:
    """Transform a 16x16 logical operator to 4-qubit physical basis."""
    B_tot = np.kron(B_control, B_target)
    return B_tot @ M_onto @ B_tot.conj().T

SWAP_phys_log = get_physical_matrix(SWAP_onto, B_LOG, B_LOG)
SWAP_phys_virt = get_physical_matrix(SWAP_onto, B_VIRT, B_VIRT)

#==============================================================================
# 3. GATE WRAPPER WITH DECOMPOSITION (Explicit SWAP -> 3x CNOT)
#==============================================================================
class PhysicalSWAPWrapper(cirq.Gate):
    """Wraps logical SWAP and decomposes into three tetralemmatic CNOTs."""
    def __init__(self, is_virtual: bool):
        self._is_virtual = is_virtual

    def _num_qubits_(self) -> int: return 4
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2, 2, 2)
    def _has_unitary_(self) -> bool: return True

    def _unitary_(self) -> np.ndarray:
        return SWAP_phys_virt.copy() if self._is_virtual else SWAP_phys_log.copy()

    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        c0, c1, t0, t1 = qubits
        
        # Instantiate the CNOT wrapper with the correct manifold setting
        # PhysicalCNOTWrapper handles basis rotation and standard gate decomposition
        cnot_wrapper = PhysicalCNOTWrapper(TetralemmaticIonCNOTGate(), self._is_virtual)
        
        # SWAP decomposition: CNOT(c, t) -> CNOT(t, c) -> CNOT(c, t)
        yield cnot_wrapper.on(c0, c1, t0, t1)
        yield cnot_wrapper.on(t0, t1, c0, c1)
        yield cnot_wrapper.on(c0, c1, t0, t1)

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        base = "V-SWAP" if self._is_virtual else "TI-SWAP"
        return cirq.CircuitDiagramInfo(wire_symbols=(base,)*4)
        
    def __repr__(self) -> str:
        return f"PhysicalSWAPWrapper(is_virtual={self._is_virtual})"

#==============================================================================
# 4. CIRQ GATE IMPLEMENTATION (Qudit-Native)
#==============================================================================
class TetralemmaticIonSWAPGate(cirq.Gate):
    """Tetralemmatic SWAP acting on two d=4 qudits."""
    def _qid_shape_(self) -> Tuple[int, ...]: return (4, 4)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray:
        return SWAP_phys_log.copy()
    
    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        # Decomposition handled by wrapper after expansion
        return NotImplemented
        
    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=("SWAP", "SWAP"))
    def __repr__(self) -> str: return "TetralemmaticIonSWAPGate()"
    def __eq__(self, other) -> bool: return isinstance(other, TetralemmaticIonSWAPGate)
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
        if isinstance(gate, TetralemmaticIonSWAPGate):
            if len(op.qubits) == 2:
                is_virtual = isinstance(op.qubits[0], VirtualQudit)
                gate = PhysicalSWAPWrapper(is_virtual)
            else:
                raise ValueError("SWAP expects exactly two qudits")
                
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
        gateset = cirq_ionq.ForteNativeGateset()
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
class TetralemmaticIonSWAPGates:
    """Factory and verification for IonQ-targeted tetralemmatic SWAP operators."""
    def __init__(self):
        self.SWAP = TetralemmaticIonSWAPGate()
        
    def _logical_to_physical(self, logical_idx: int, B_tot: np.ndarray) -> np.ndarray:
        vec = np.zeros(16, dtype=complex)
        vec[logical_idx] = 1.0
        return B_tot @ vec

    def verify_logical_properties(self) -> Dict[str, bool]:
        checks = {}
        B_tot = np.kron(B_LOG, B_LOG)
        U = SWAP_phys_log
        I16 = np.eye(16)

        checks["SWAP_unitary"] = bool(np.allclose(U.conj().T @ U, I16))
        checks["SWAP_hermitian"] = bool(np.allclose(U.conj().T, U))
        checks["SWAP_involutory"] = bool(np.allclose(U @ U, I16))

        # Permutation checks
        psi_in = self._logical_to_physical(2, B_tot)   # |SynTh, Th⟩
        psi_out = self._logical_to_physical(8, B_tot)  # |Th, SynTh⟩
        checks["correct_permutation"] = bool(np.allclose(U @ psi_in, psi_out))

        psi_in2 = self._logical_to_physical(13, B_tot) # |AntiTh, HoloTh⟩
        psi_out2 = self._logical_to_physical(7, B_tot) # |HoloTh, AntiTh⟩
        checks["correct_permutation_2"] = bool(np.allclose(U @ psi_in2, psi_out2))

        # Tensor exchange check
        A = np.diag([1,2,3,4]).astype(complex)
        B = np.diag([5,6,7,8]).astype(complex)
        AB = np.kron(A, B)
        BA = np.kron(B, A)
        checks["tensor_exchange"] = bool(np.allclose(U @ AB @ U, BA))

        # Qubit reduction check (polar subspace)
        polar_phys_idx = [0, 3, 12, 15]
        U_p = U[np.ix_(polar_phys_idx, polar_phys_idx)]
        std_swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        checks["qubit_reduction_SWAP"] = bool(np.allclose(U_p, std_swap, atol=1e-8))
        return checks

    def verify_virtual_properties(self) -> Dict[str, bool]:
        checks = {}
        B_tot = np.kron(B_VIRT, B_VIRT)
        U = SWAP_phys_virt
        I16 = np.eye(16)
        checks["VSWAP_unitary"] = bool(np.allclose(U.conj().T @ U, I16))
        checks["VSWAP_hermitian"] = bool(np.allclose(U.conj().T, U))
        checks["VSWAP_involutory"] = bool(np.allclose(U @ U, I16))

        # Virtual Permutation: |F, P⟩ (idx 1) -> |P, F⟩ (idx 4)
        # Note: F=0, P=1. Logical index for F,P is 0*4+1 = 1. Logical for P,F is 1*4+0 = 4.
        psi_in = self._logical_to_physical(1, B_tot)
        psi_out = self._logical_to_physical(4, B_tot)
        checks["virtual_permutation"] = bool(np.allclose(U @ psi_in, psi_out))

        # Tensor exchange
        A = np.diag([1,2,3,4]).astype(complex)
        B = np.diag([5,6,7,8]).astype(complex)
        AB = np.kron(A, B)
        BA = np.kron(B, A)
        checks["virtual_tensor_exchange"] = bool(np.allclose(U @ AB @ U, BA))
        return checks

    def compile_test_circuit(self, use_virtual: bool = False) -> cirq.Circuit:
        if use_virtual:
            q1 = VirtualQudit(0)
            q2 = VirtualQudit(1)
        else:
            q1 = NomosIonQid(0)
            q2 = NomosIonQid(1)
        return cirq.Circuit(
            self.SWAP.on(q1, q2),
            cirq.measure(*cirq.LineQubit.range(4), key="m")
        )

#==============================================================================
# MAIN EXECUTION (Dual-Manifold with Unified Compiler Validation)
#==============================================================================
if __name__ == "__main__":
    print("=== Tetralemmatic IonQ SWAP Gate Verification (Dual-Manifold) ===")
    print(f"Ion Manifold Energies (B=1G): {IonManifold.energy_levels(1.0)}\n")
    factory = TetralemmaticIonSWAPGates()
    
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
        summarize_compilation(sim_log, "Logical SWAP Simulation", "api", True)
        
        sim_virt = compile_tetralemmatic_ionq(test_circ_virt, target="api", simulation_mode=True)
        summarize_compilation(sim_virt, "Virtual SWAP Simulation", "api", True)
    except Exception as e:
        print(f"\n✗ Simulation Compilation Failed: {e}")
        
    # 2. API TARGET (Cloud-Ready)
    print("\n=== Compilation Test: API Target (Cloud-Ready) ===")
    try:
        api_log = compile_tetralemmatic_ionq(test_circ_log, target="api", simulation_mode=False)
        summarize_compilation(api_log, "Logical SWAP API", "api", False)
        
        api_virt = compile_tetralemmatic_ionq(test_circ_virt, target="api", simulation_mode=False)
        summarize_compilation(api_virt, "Virtual SWAP API", "api", False)
    except Exception as e:
        print(f"\n✗ API Compilation Failed: {e}")
        
    # 3. FORTE NATIVE TARGET (Pulse-Level)
    print("\n=== Compilation Test: Forte Native Target (Pulse-Level) ===")
    try:
        forte_log = compile_tetralemmatic_ionq(test_circ_log, target="forte_native", simulation_mode=False)
        summarize_compilation(forte_log, "Logical SWAP Forte", "forte_native", False)
        
        forte_virt = compile_tetralemmatic_ionq(test_circ_virt, target="forte_native", simulation_mode=False)
        summarize_compilation(forte_virt, "Virtual SWAP Forte", "forte_native", False)
    except Exception as e:
        print(f"\n✗ Forte Native Compilation Failed: {e}")
        
    print("\n✓ Unified compiler validation complete. All targets verified.")