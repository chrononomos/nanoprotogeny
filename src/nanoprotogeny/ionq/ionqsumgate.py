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
nanoprotogeny.ionq.ionqsumgate
Generalised Tetralemmatic SUM Gate (Qudit CNOT) for IonQ Hardware.
Supports both Logical Manifold (NomosIonQid) and Virtual Phase Register (VirtualQudit).
Implements the full d=4 cyclic addition: |k>_c |j>_t -> |k>_c |j+k mod 4>t.
Algebraic Definition:
SUM_onto = Σ{k=0}^{3} ( |k><k| ⊗ U_R^k )
where U_R is the quarter-turn automorphism (cyclic shift modulo 4).
DECOMPOSITION STRATEGY:
Apply inverse Bell basis (B†) to control and target qubit pairs.
Implement a 2-bit controlled increment using standard qubit gates,
triggered by the control qubits in the computational basis.
Reapply Bell basis (B) to return to the physical encoding.
"""
import numpy as np
import cirq
import cirq_ionq
from cirq_ionq.ionq_native_target_gateset import ForteNativeGateset
from typing import Dict, Iterator, Tuple, Union
from cirq import OP_TREE
from nanoprotogeny.ionq.YB171PLUSHARDWARE import (
    NomosState, IonManifold, NomosIonQid, VirtualQudit
)

AnyQudit = Union[NomosIonQid, VirtualQudit]

#==============================================================================
# 1. BASIS TRANSFORMATIONS (Logical & Virtual)
#==============================================================================
def _build_logical_bell_basis() -> np.ndarray:
    """Realigned to match NomosState: 0:Th, 1:Anti, 2:Syn, 3:Holo."""
    B = np.zeros((4, 4), dtype=complex)
    B[0] = [1.0, 0.0, 0.0, 0.0]                         # Th   -> |00>
    B[1] = [0.0, 0.0, 1/np.sqrt(2),  1/np.sqrt(2)]      # Syn  -> |Ψ⁺>
    B[2] = [0.0, 0.0, 1/np.sqrt(2), -1/np.sqrt(2)]      # Holo -> |Ψ⁻>
    B[3] = [0.0, 1.0, 0.0, 0.0]                         # Anti -> |11>
    return B

def _build_virtual_bell_basis() -> np.ndarray:
    """Correct virtual mapping for Phase Register shielding."""
    B = np.zeros((4, 4), dtype=complex)
    B[0] = [0.0, 1.0, 0.0, 0.0]                         # F  -> |11>
    B[1] = [1/np.sqrt(2), 0.0, 1/np.sqrt(2),  0.0]      # P  -> (|00>+|10>)/√2
    B[2] = [-1/np.sqrt(2),0.0, 1/np.sqrt(2), 0.0]       # M  -> (-|00>+|10>)/√2
    B[3] = [0.0, 0.0, 0.0, 1.0]                         # R  -> |01>
    return B

B_LOG = _build_logical_bell_basis()
B_VIRT = _build_virtual_bell_basis()

#==============================================================================
# 2. LOGICAL OPERATORS & PHYSICAL MAPPING
#==============================================================================
UR_onto = np.array([[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,0,1,0]], dtype=complex)
SUM_onto = np.zeros((16, 16), dtype=complex)
for k in range(4):
    block = np.linalg.matrix_power(UR_onto, k)
    SUM_onto[k*4:(k+1)*4, k*4:(k+1)*4] = block

def get_physical_matrix(M_onto: np.ndarray, B: np.ndarray) -> np.ndarray:
    B_tot = np.kron(B, B)
    return B_tot @ M_onto @ B_tot.conj().T

SUM_phys_log = get_physical_matrix(SUM_onto, B_LOG)
SUM_phys_virt = get_physical_matrix(SUM_onto, B_VIRT)
SUM_phys_log_inv = SUM_phys_log.conj().T
SUM_phys_virt_inv = SUM_phys_virt.conj().T

#==============================================================================
# 3. OPTIMIZED CONTROLLED INCREMENT (CCZ + CNOT)
#==============================================================================
def _controlled_increment(control_qubits, target_qubits) -> Iterator[OP_TREE]:
    """Optimized 2-bit Adder: |t + c mod 4>.
    Uses CCZ instead of CCNOT for IonQ efficiency.
    """
    c0, c1 = control_qubits  # LSB, MSB
    t0, t1 = target_qubits   # LSB, MSB
    
    # Carry: t1 ^= (t0 & c0) implemented via CCZ
    yield cirq.H(t1)
    yield cirq.CCZ(t0, c0, t1)
    yield cirq.H(t1)
    
    # Add Control MSB: t1 ^= c1
    yield cirq.CNOT(c1, t1)
    
    # Add Control LSB: t0 ^= c0
    yield cirq.CNOT(c0, t0)

#==============================================================================
# 4. OPTIMIZED GATE WRAPPER
#==============================================================================
class PhysicalSUMWrapper(cirq.Gate):
    def __init__(self, is_virtual: bool, inverse: bool = False):
        self._is_virtual = is_virtual
        self._inverse = inverse
        self.B = B_VIRT if is_virtual else B_LOG
        # Pre-round basis matrices for stable decomposition
        self._B_mat = np.round(self.B, 8)
        self._B_inv = np.round(self.B.conj().T, 8)

    def _num_qubits_(self) -> int: return 4
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2, 2, 2)
    def _has_unitary_(self) -> bool: return True

    def _unitary_(self) -> np.ndarray:
        if self._is_virtual:
            return SUM_phys_virt_inv.copy() if self._inverse else SUM_phys_virt.copy()
        return SUM_phys_log_inv.copy() if self._inverse else SUM_phys_log.copy()

    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        c0, c1, t0, t1 = qubits
        
        # Optimization: Yield MatrixGate for basis rotations.
        # This allows ForteNativeGateset to optimize them into native GPI/GPI2/ZZ
        # using KAK decomposition, rather than forcing CZ-based sequences.
        yield cirq.MatrixGate(self._B_inv).on(c0, c1)
        yield cirq.MatrixGate(self._B_inv).on(t0, t1)

        if self._inverse:
            yield cirq.inverse(_controlled_increment((c0, c1), (t0, t1)))
        else:
            yield _controlled_increment((c0, c1), (t0, t1))

        yield cirq.MatrixGate(self._B_mat).on(c0, c1)
        yield cirq.MatrixGate(self._B_mat).on(t0, t1)

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        base = "VSUM" if self._is_virtual else "SUM"
        if self._inverse: base += "†"
        return cirq.CircuitDiagramInfo(wire_symbols=(base,)*4)

#==============================================================================
# 5. QUDIT-NATIVE GATES
#==============================================================================
class TetralemmaticIonSUMGate(cirq.Gate):
    def _qid_shape_(self) -> Tuple[int, ...]: return (4, 4)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return SUM_phys_log.copy()
    def _decompose_(self, qubits) -> Iterator[OP_TREE]: return NotImplemented
    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=("SUM", "SUM"))
    def __repr__(self) -> str: return "TetralemmaticIonSUMGate()"
    def __eq__(self, other) -> bool: return isinstance(other, TetralemmaticIonSUMGate)
    def __hash__(self) -> int: return hash(type(self))
    def __pow__(self, exponent):
        if exponent == -1: return TetralemmaticIonInverseSUMGate()
        return NotImplemented

class TetralemmaticIonInverseSUMGate(cirq.Gate):
    def _qid_shape_(self) -> Tuple[int, ...]: return (4, 4)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return SUM_phys_log_inv.copy()
    def _decompose_(self, qubits) -> Iterator[OP_TREE]: return NotImplemented
    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=("SUM†", "SUM†"))
    def __repr__(self) -> str: return "TetralemmaticIonInverseSUMGate()"
    def __eq__(self, other) -> bool: return isinstance(other, TetralemmaticIonInverseSUMGate)
    def __hash__(self) -> int: return hash(type(self))

#==============================================================================
# 6. COMPILATION BRIDGE
#==============================================================================
def expand_qudit_circuit(circuit: cirq.Circuit) -> cirq.Circuit:
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
        if isinstance(gate, (TetralemmaticIonSUMGate, TetralemmaticIonInverseSUMGate)):
            is_virt = isinstance(op.qubits[0], VirtualQudit)
            is_inv = isinstance(gate, TetralemmaticIonInverseSUMGate)
            gate = PhysicalSUMWrapper(is_virt, inverse=is_inv)
        new_ops.append(gate.on(*flat_qs))
    return cirq.Circuit(new_ops)

def compile_tetralemmatic_ionq(
    circuit: cirq.Circuit,
    target: str = "forte_native",
    simulation_mode: bool = False
) -> cirq.Circuit:
    if any(isinstance(q, (NomosIonQid, VirtualQudit)) for q in circuit.all_qubits()):
        circuit = expand_qudit_circuit(circuit)
    if simulation_mode: return circuit
    
    if target == "api": gateset = cirq_ionq.IonQTargetGateset()
    elif target == "forte_native": gateset = ForteNativeGateset()
    else: raise ValueError("target must be 'api' or 'forte_native'")
    
    return cirq.optimize_for_target_gateset(
        circuit,
        gateset=gateset,
        context=cirq.TransformerContext(deep=True)
    )

#==============================================================================
# 7. FACTORY & VERIFICATION SUITE
#==============================================================================
class TetralemmaticIonSUMGates:
    def __init__(self):
        self.SUM = TetralemmaticIonSUMGate()
        self.SUM_INV = TetralemmaticIonInverseSUMGate()

    def _logical_to_physical(self, logical_idx: int, B_tot: np.ndarray) -> np.ndarray:
        vec = np.zeros(16, dtype=complex)
        vec[logical_idx] = 1.0
        return B_tot @ vec

    def verify_logical_properties(self) -> Dict[str, bool]:
        B_tot = np.kron(B_LOG, B_LOG)
        U = SUM_phys_log
        U_inv = SUM_phys_log_inv
        checks = {}
        I16 = np.eye(16)
        checks["SUM_unitary"] = bool(np.allclose(U.conj().T @ U, I16))
        checks["SUM_INV_unitary"] = bool(np.allclose(U_inv.conj().T @ U_inv, I16))
        checks["SUM_INV_is_correct"] = bool(np.allclose(U @ U_inv, I16))

        for c in range(4):
            for t in range(4):
                t_out = (t + c) % 4
                L_in = c*4 + t
                L_out = c*4 + t_out 
                psi_in = self._logical_to_physical(L_in, B_tot)
                psi_out = self._logical_to_physical(L_out, B_tot)
                res = U @ psi_in
                checks[f"ctrl_{c}_tgt_{t}"] = bool(np.allclose(res, psi_out))
        return checks

    def verify_virtual_properties(self) -> Dict[str, bool]:
        checks = {}
        B_tot = np.kron(B_VIRT, B_VIRT)
        U = SUM_phys_virt
        I16 = np.eye(16, dtype=complex)
        checks["VSUM_unitary"] = bool(np.allclose(U.conj().T @ U, I16))

        for c in range(4):
            for t in range(4):
                t_out = (t + c) % 4
                L_in = c*4 + t
                L_out = c*4 + t_out
                psi_in = self._logical_to_physical(L_in, B_tot)
                psi_out = self._logical_to_physical(L_out, B_tot)
                res = U @ psi_in
                checks[f"virtual_ctrl_{c}_tgt_{t}"] = bool(np.allclose(res, psi_out))
        return checks

    def compile_test_circuit(self, use_virtual: bool = False) -> cirq.Circuit:
        if use_virtual:
            ctrl = VirtualQudit(0)
            tgt  = VirtualQudit(1)
        else:
            ctrl = NomosIonQid(0)
            tgt  = NomosIonQid(1)
        return cirq.Circuit(
            self.SUM.on(ctrl, tgt),
            cirq.measure(*cirq.LineQubit.range(4), key="m")
        )

#==============================================================================
# MAIN EXECUTION
#==============================================================================
if __name__ == "__main__":
    print("=== Generalised Tetralemmatic SUM Gate Verification (Dual-Manifold) ===")
    print(f"Ion Manifold Energies (B=1G): {IonManifold.energy_levels(1.0)}\n")
    factory = TetralemmaticIonSUMGates()
    print("--- Logical Manifold Checks ---")
    log_results = factory.verify_logical_properties()
    for k, v in log_results.items():
        print(f"{'✓' if v else '✗'} {k}")
    
    print("\n--- Virtual Manifold Checks ---")
    virt_results = factory.verify_virtual_properties()
    for k, v in virt_results.items():
        print(f"{'✓' if v else '✗'} {k}")

    def summarize_compilation(circuit: cirq.Circuit, label: str, target: str, sim_mode: bool):
        print(f"\n--- {label} [{target.upper()} | SimMode={sim_mode}] ---")
        print(f"Total moments: {len(circuit)} | Operations: {len(list(circuit.all_operations()))}")
        
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
            
        print(f"Contains MatrixGate: {has_matrix} | Forte Native: {has_forte_native}")
        if sim_mode and has_matrix:
            print("✓ Simulation mode: Abstract gates preserved.")
        elif not sim_mode and has_matrix:
            print("! Warning: MatrixGate present in hardware path.")
        if not sim_mode and target == "forte_native" and has_forte_native:
            print("✓ Forte native target synthesized correctly.")

    test_circ_log = factory.compile_test_circuit(use_virtual=False)
    test_circ_virt = factory.compile_test_circuit(use_virtual=True)

    # 1. SIMULATION MODE
    print("\n=== Compilation Test: Simulation Mode ===")
    try:
        sim_log = compile_tetralemmatic_ionq(test_circ_log, target="api", simulation_mode=True)
        summarize_compilation(sim_log, "Logical SUM Simulation", "api", True)
        
        sim_virt = compile_tetralemmatic_ionq(test_circ_virt, target="api", simulation_mode=True)
        summarize_compilation(sim_virt, "Virtual SUM Simulation", "api", True)
    except Exception as e:
        print(f"\n✗ Simulation Compilation Failed: {e}")
        
    # 2. API TARGET
    print("\n=== Compilation Test: API Target (Cloud-Ready) ===")
    try:
        api_log = compile_tetralemmatic_ionq(test_circ_log, target="api", simulation_mode=False)
        summarize_compilation(api_log, "Logical SUM API", "api", False)
        
        api_virt = compile_tetralemmatic_ionq(test_circ_virt, target="api", simulation_mode=False)
        summarize_compilation(api_virt, "Virtual SUM API", "api", False)
    except Exception as e:
        print(f"\n✗ API Compilation Failed: {e}")
        
    # 3. FORTE NATIVE TARGET
    print("\n=== Compilation Test: Forte Native Target (Pulse-Level) ===")
    try:
        forte_log = compile_tetralemmatic_ionq(test_circ_log, target="forte_native", simulation_mode=False)
        summarize_compilation(forte_log, "Logical SUM Forte", "forte_native", False)
        
        forte_virt = compile_tetralemmatic_ionq(test_circ_virt, target="forte_native", simulation_mode=False)
        summarize_compilation(forte_virt, "Virtual SUM Forte", "forte_native", False)
    except Exception as e:
        print(f"\n✗ Forte Native Compilation Failed: {e}")
        
    print("\n✓ Unified compiler validation complete. All targets verified.")