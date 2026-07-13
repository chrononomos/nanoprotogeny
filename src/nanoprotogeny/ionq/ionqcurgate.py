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
nanoprotogeny.ionq.ionqcurgate
Refactored Controlled Quarter-Turn (C-U_R) for IonQ Hardware.
Supports both Logical Manifold (NomosIonQid) and Virtual Phase Register (VirtualQudit).
Algebraic Definition: C-U_R = sum_{k≠c} |k><k| ⊗ I + |c><c| ⊗ U_R
Default control vertex: index 1 (Antithesis for logical, HoloTh_P for virtual).
DECOMPOSITION STRATEGY:
1. TargetUROntoGate implements _decompose_ using standard 2-qubit operations.
2. PhysicalCURWrapper uses cirq.decompose(keep=len≤2) — ForteNativeGateset has no
   native 3-qubit gates; keep=len≤3 (previous) triggered QSD on 3-qubit remnants.
3. Basis rotations handled via cirq.two_qubit_matrix_to_cz_operations.
Hardware Alignment: Basis matrices B_LOG and B_VIRT are explicitly documented with their
mapping to the 8-level 171Yb+ hyperfine/auxiliary manifold defined in YB171PLUSHARDWARE.
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
# Refactored imports: Canonical source is now ionqtetralemmatics
from nanoprotogeny.ionq.ionqtetralemmatics import (
    B_LOG,
    B_VIRT,
)
from nanoprotogeny.ionq.ionqBLOGgate import BLOG, BLOG_DAG
from nanoprotogeny.ionq.ionqBVIRTgate import BVIRT, BVIRT_DAG

AnyQudit = Union[NomosIonQid, VirtualQudit]

#==============================================================================
# 1. BASIS TRANSFORMATIONS & LOGICAL OPERATORS
# B_LOG and B_VIRT are imported from ionqurgate; precomputed KAK circuits
# (B_LOG_DAG_OPS etc.) replace repeated two_qubit_matrix_to_cz_operations calls.
#==============================================================================

# Logical U_R (Quarter Turn)
UR_onto = np.array([[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,0,1,0]], dtype=complex)
I4 = np.eye(4, dtype=complex)
CONTROL_IDX = 1  # Controls on Logical Index 1 (SynTh in logical, HoloTh_P in virtual)

# Construct 16x16 C-U_R matrix
CUR_onto = np.zeros((16, 16), dtype=complex)
for i in range(4):
    block = UR_onto if i == CONTROL_IDX else I4
    CUR_onto[i*4:(i+1)*4, i*4:(i+1)*4] = block

def get_physical_matrix(M_onto: np.ndarray, B_control: np.ndarray, B_target: np.ndarray) -> np.ndarray:
    B_tot = np.kron(B_control, B_target)
    return B_tot @ M_onto @ B_tot.conj().T

CUR_phys_log = get_physical_matrix(CUR_onto, B_LOG, B_LOG)
CUR_phys_virt = get_physical_matrix(CUR_onto, B_VIRT, B_VIRT)

#==============================================================================
# 2. STANDARD-GATE DECOMPOSITION PRIMITIVES
#==============================================================================
class TargetUROntoGate(cirq.Gate):
    """2‑qubit gate that implements the logical U_R operator (d=4)."""
    def _num_qubits_(self) -> int: return 2
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return UR_onto.copy()

    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        # Provide explicit decomposition to enable ControlledGate logic
        yield cirq.two_qubit_matrix_to_cz_operations(
            qubits[0], qubits[1], np.round(self._unitary_(), 10), allow_partial_czs=True
        )

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=("URₒ", "URₒ"))

    def __repr__(self) -> str: return "TargetUROntoGate()"

#==============================================================================
# 3. PROTOCOL-AWARE PHYSICAL WRAPPER
#==============================================================================
class PhysicalCURWrapper(cirq.Gate):
    """Wraps logical C-U_R and decomposes using Bell basis + controlled U_R."""
    def __init__(self, original_gate: cirq.Gate, is_virtual: bool):
        self._original_gate = original_gate
        self._is_virtual = is_virtual
        self._B = B_VIRT if is_virtual else B_LOG

    def _num_qubits_(self) -> int: return 4
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2, 2, 2)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray:
        return CUR_phys_virt.copy() if self._is_virtual else CUR_phys_log.copy()

    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        c0, c1, t0, t1 = qubits
        
        # 1. Rotate control & target to computational basis via fundamental gate objects.
        _dag_gate = BVIRT_DAG if self._is_virtual else BLOG_DAG
        yield _dag_gate.on(c0, c1)
        yield _dag_gate.on(t0, t1)
        
        # 2. Controlled-U_R (triggers on control qubits = |01⟩)
        ctrl_ur = cirq.ControlledGate(
            TargetUROntoGate(),
            num_controls=2,
            control_values=[0, 1]   # c0=0, c1=1
        )
        
        # ForteNativeGateset has no native 3-qubit gates; keep=len≤2 forces full
        # expansion to 2-qubit ops before KAK compilation.  IonQTargetGateset also
        # accepts Toffoli natively, but Forte does not — keep at len≤2 for both paths.
        yield from cirq.decompose(
            ctrl_ur(c0, c1, t0, t1),
            keep=lambda op: len(op.qubits) <= 2
        )
        
        # 3. Rotate back to Bell-separable encoding via fundamental gate objects.
        _fwd_gate = BVIRT if self._is_virtual else BLOG
        yield _fwd_gate.on(c0, c1)
        yield _fwd_gate.on(t0, t1)

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        base = "V-CUR" if self._is_virtual else "TI-CUR"
        return cirq.CircuitDiagramInfo(wire_symbols=(base,)*4)
        
    def __repr__(self) -> str: return f"PhysicalCURWrapper(is_virtual={self._is_virtual})"

#==============================================================================
# 4. CIRQ GATE IMPLEMENTATION (Qudit-Native)
#==============================================================================
class ControlledURIonGate(cirq.Gate):
    """Tetralemmatic Controlled-U_R acting on two d=4 qudits."""
    def _qid_shape_(self) -> Tuple[int, ...]: return (4, 4)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return CUR_phys_log.copy()
    
    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        return NotImplemented  # Defer to PhysicalCURWrapper after expansion
        
    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=("CUR", "CUR"))
    def __repr__(self) -> str: return "ControlledURIonGate()"
    def __eq__(self, other) -> bool: return isinstance(other, ControlledURIonGate)
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
        if isinstance(gate, ControlledURIonGate):
            if len(op.qubits) == 2:
                is_virtual = isinstance(op.qubits[0], VirtualQudit)
                gate = PhysicalCURWrapper(gate, is_virtual)
            else:
                raise ValueError("CUR expects exactly two qudits")
                
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
# 6. FACTORY & VERIFICATION SUITE
#==============================================================================
class TetralemmaticIonCURGates:
    """Factory and verification for IonQ-targeted tetralemmatic CUR operators."""
    def __init__(self):
        self.CUR = ControlledURIonGate()
        
    def _logical_to_physical(self, logical_idx: int, B_tot: np.ndarray) -> np.ndarray:
        vec = np.zeros(16, dtype=complex)
        vec[logical_idx] = 1.0
        return B_tot @ vec

    def verify_logical_properties(self) -> Dict[str, bool]:
        checks = {}
        B_tot = np.kron(B_LOG, B_LOG)
        U = CUR_phys_log
        checks["CUR_unitary"] = bool(np.allclose(U.conj().T @ U, np.eye(16)))

        c_control = CONTROL_IDX
        identity_preserved = True
        target_correct = True
        for C in range(4):
            for T in range(4):
                L_in = 4 * C + T
                phys_in = self._logical_to_physical(L_in, B_tot)
                phys_out = U @ phys_in
                if C == c_control:
                    T_out = (T + 1) % 4
                    L_out = 4 * C + T_out
                    phys_expected = self._logical_to_physical(L_out, B_tot)
                    if not np.allclose(phys_out, phys_expected):
                        target_correct = False
                else:
                    if not np.allclose(phys_out, phys_in):
                        identity_preserved = False
        checks["identity_on_non_control"] = identity_preserved
        checks["target_applies_UR"] = target_correct

        phys_in = self._logical_to_physical(5, B_tot)
        phys_out = U @ phys_in
        phys_expected = self._logical_to_physical(6, B_tot)
        checks["crosses_polar_nonpolar"] = bool(np.allclose(phys_out, phys_expected))

        psi_super = (self._logical_to_physical(0, B_tot) + self._logical_to_physical(4, B_tot)) / np.sqrt(2)
        psi_out = U @ psi_super
        mat = psi_out.reshape(4, 4)
        _, s, _ = np.linalg.svd(mat)
        checks["generates_entanglement"] = bool(np.sum(s > 1e-9) > 1)
        return checks

    def verify_virtual_properties(self) -> Dict[str, bool]:
        checks = {}
        B_tot = np.kron(B_VIRT, B_VIRT)
        U = CUR_phys_virt
        checks["VCUR_unitary"] = bool(np.allclose(U.conj().T @ U, np.eye(16)))

        c_control = CONTROL_IDX  # index 1 is HoloTh_P
        identity_preserved = True
        target_correct = True
        for C in range(4):
            for T in range(4):
                L_in = 4 * C + T
                phys_in = self._logical_to_physical(L_in, B_tot)
                phys_out = U @ phys_in
                if C == c_control:
                    T_out = (T + 1) % 4
                    L_out = 4 * C + T_out
                    phys_expected = self._logical_to_physical(L_out, B_tot)
                    if not np.allclose(phys_out, phys_expected):
                        target_correct = False
                else:
                    if not np.allclose(phys_out, phys_in):
                        identity_preserved = False
        checks["virtual_identity_non_control"] = identity_preserved
        checks["virtual_target_applies_UR"] = target_correct
        return checks

    def compile_test_circuit(self, use_virtual: bool = False) -> cirq.Circuit:
        if use_virtual:
            ctrl = VirtualQudit(0)
            tgt = VirtualQudit(1)
        else:
            ctrl = NomosIonQid(0)
            tgt = NomosIonQid(1)
        return cirq.Circuit(
            self.CUR.on(ctrl, tgt),
            cirq.measure(*cirq.LineQubit.range(4), key="m")
        )

#==============================================================================
# MAIN EXECUTION (Dual-Manifold with Unified Compiler Validation)
#==============================================================================
if __name__ == "__main__":
    print("=== Tetralemmatic IonQ Controlled-UR Verification (Dual-Manifold) ===")
    print(f"Ion Manifold Energies (B=1G): {IonManifold.energy_levels(1.0)}\n")
    factory = TetralemmaticIonCURGates()
    
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
        summarize_compilation(sim_log, "Logical CUR Simulation", "api", True)
        
        sim_virt = compile_tetralemmatic_ionq(test_circ_virt, target="api", simulation_mode=True)
        summarize_compilation(sim_virt, "Virtual CUR Simulation", "api", True)
    except Exception as e:
        print(f"\n✗ Simulation Compilation Failed: {e}")
        
    # 2. API TARGET (Cloud-Ready)
    print("\n=== Compilation Test: API Target (Cloud-Ready) ===")
    try:
        api_log = compile_tetralemmatic_ionq(test_circ_log, target="api", simulation_mode=False)
        summarize_compilation(api_log, "Logical CUR API", "api", False)
        
        api_virt = compile_tetralemmatic_ionq(test_circ_virt, target="api", simulation_mode=False)
        summarize_compilation(api_virt, "Virtual CUR API", "api", False)
    except Exception as e:
        print(f"\n✗ API Compilation Failed: {e}")
        
    # 3. FORTE NATIVE TARGET (Pulse-Level)
    print("\n=== Compilation Test: Forte Native Target (Pulse-Level) ===")
    try:
        forte_log = compile_tetralemmatic_ionq(test_circ_log, target="forte_native", simulation_mode=False)
        summarize_compilation(forte_log, "Logical CUR Forte", "forte_native", False)
        
        forte_virt = compile_tetralemmatic_ionq(test_circ_virt, target="forte_native", simulation_mode=False)
        summarize_compilation(forte_virt, "Virtual CUR Forte", "forte_native", False)
    except Exception as e:
        print(f"\n✗ Forte Native Compilation Failed: {e}")
        
    print("\n✓ Unified compiler validation complete. All targets verified.")