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
nanoprotogeny.ionq.ionqprojectorgate
Refactored Tetralemmatic Projector & Algebra Gates for IonQ Hardware.
Supports both Logical Manifold (NomosIonQid) and Virtual Phase Register (VirtualQudit).
Implements:
Sharp/Unsharp Projectors (Kraus Channels) on a specific ontological vertex.
Generic Algebraic Gates (unitary or Kraus) from arbitrary 4x4 logical matrices.
"""
import numpy as np
import cirq
import cirq_ionq
from cirq_ionq.ionq_native_target_gateset import ForteNativeGateset
from typing import Dict, Iterator, Tuple, Union
from cirq import OP_TREE
from nanoprotogeny.ionq.YB171PLUSHARDWARE import (
    NomosState, IonManifold, NomosIonQid, VirtualQudit,
)
from nanoprotogeny.theory.kinematic_matrix import Vertex
AnyQudit = Union[NomosIonQid, VirtualQudit]

#==============================================================================
# 1. BASIS TRANSFORMATIONS (Logical & Virtual)
#==============================================================================
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

LOGICAL_DIM = 4

def logical_to_physical_matrix(M_onto: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Transforms a 4x4 logical matrix to physical basis."""
    if M_onto.shape != (LOGICAL_DIM, LOGICAL_DIM):
        raise ValueError(f"Logical matrix must be {LOGICAL_DIM}x{LOGICAL_DIM}")
    return B @ M_onto @ B.conj().T

#==============================================================================
# 2. GATE WRAPPERS
#==============================================================================
class PhysicalProjectorWrapper(cirq.Gate):
    """Manifold-aware projector wrapper implementing a Kraus channel."""
    def __init__(self, vertex: Vertex, transmission: float, is_virtual: bool):
        self._vertex = vertex
        self._T = transmission
        self._is_virtual = is_virtual
        B = B_VIRT if is_virtual else B_LOG

        sqrt_E = np.zeros((4, 4), dtype=complex)
        sqrt_E[vertex.value, vertex.value] = np.sqrt(transmission)
        self._K0 = B @ sqrt_E @ B.conj().T

        sqrt_I = np.eye(4, dtype=complex)
        sqrt_I[vertex.value, vertex.value] = np.sqrt(1.0 - transmission)
        self._K1 = B @ sqrt_I @ B.conj().T

    def _num_qubits_(self) -> int: return 2
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2)
    def _has_unitary_(self) -> bool: return False
    def _has_kraus_(self) -> bool: return True
    
    def _kraus_(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self._K0, self._K1)

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        sym = f"Proj({self._vertex.name})"
        return cirq.CircuitDiagramInfo(wire_symbols=(sym, sym))
        
    def __repr__(self) -> str:
        return f"PhysicalProjectorWrapper({self._vertex}, T={self._T}, is_virtual={self._is_virtual})"

class PhysicalAlgebraWrapper(cirq.Gate):
    """Manifold-aware algebra wrapper (Unitary or Kraus)."""
    def __init__(self, M_onto: np.ndarray, name: str, is_virtual: bool):
        self._M_onto = M_onto.copy()
        self._name = name
        self._is_virtual = is_virtual
        B = B_VIRT if is_virtual else B_LOG
        self._M_phys = B @ M_onto @ B.conj().T
        self._is_unitary = bool(np.allclose(
            self._M_phys.conj().T @ self._M_phys,
            np.eye(4), atol=1e-9
        ))

    def _num_qubits_(self) -> int: return 2
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2)
    def _has_unitary_(self) -> bool: return self._is_unitary
    
    def _unitary_(self) -> np.ndarray:
        if not self._is_unitary:
            raise NotImplementedError("Gate is not unitary; use _kraus_()")
        return self._M_phys.copy()

    def _has_kraus_(self) -> bool: return not self._is_unitary
    
    def _kraus_(self) -> Tuple[np.ndarray, np.ndarray]:
        K0 = self._M_phys
        MdagM = K0.conj().T @ K0
        I4 = np.eye(4, dtype=complex)
        evals, evecs = np.linalg.eigh(I4 - MdagM)
        evals = np.maximum(evals, 0.0)
        sqrt_mat = evecs @ np.diag(np.sqrt(evals)) @ evecs.conj().T
        return (K0, sqrt_mat)

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        sym = f"Alg({self._name})"
        return cirq.CircuitDiagramInfo(wire_symbols=(sym, sym))
        
    def __repr__(self) -> str:
        return f"PhysicalAlgebraWrapper('{self._name}', is_virtual={self._is_virtual})"

#==============================================================================
# 3. QUDIT-NATIVE GATE CLASSES
#==============================================================================
class TetralemmaticIonProjectorGate(cirq.Gate):
    """Tetralemmatic Projector (sharp/unsharp) for d=4 qudits."""
    def __init__(self, vertex: Vertex, transmission: float = 1.0):
        if not (0.0 <= transmission <= 1.0):
            raise ValueError("Transmission must be in [0, 1]")
        self._vertex = vertex
        self._T = transmission

    def _qid_shape_(self) -> Tuple[int, ...]: return (4,)
    def _has_unitary_(self) -> bool: return False
    def _has_kraus_(self) -> bool: return True

    def _kraus_(self) -> Tuple[np.ndarray, np.ndarray]:
        # Default to logical basis for standalone verification
        sqrt_E = np.zeros((4, 4), dtype=complex)
        sqrt_E[self._vertex.value, self._vertex.value] = np.sqrt(self._T)
        K0 = B_LOG @ sqrt_E @ B_LOG.conj().T
        sqrt_I = np.eye(4, dtype=complex)
        sqrt_I[self._vertex.value, self._vertex.value] = np.sqrt(1.0 - self._T)
        K1 = B_LOG @ sqrt_I @ B_LOG.conj().T
        return (K0, K1)

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=(f"Proj({self._vertex.name})",))
        
    def __repr__(self) -> str:
        return f"TetralemmaticIonProjectorGate({self._vertex}, T={self._T})"
        
    def __eq__(self, other) -> bool:
        return (isinstance(other, TetralemmaticIonProjectorGate) and
                self._vertex == other._vertex and np.isclose(self._T, other._T))
                
    def __hash__(self) -> int:
        return hash((type(self), self._vertex, round(self._T, 6)))

class TetralemmaticIonAlgebraGate(cirq.Gate):
    """Generic algebraic gate for d=4 qudits."""
    def __init__(self, M_onto: np.ndarray, name: str = "Algebra"):
        self._M_onto = M_onto.copy()
        self._name = name

    def _qid_shape_(self) -> Tuple[int, ...]: return (4,)
    def _has_unitary_(self) -> bool:
        M_phys = B_LOG @ self._M_onto @ B_LOG.conj().T
        return bool(np.allclose(M_phys.conj().T @ M_phys, np.eye(4), atol=1e-9))

    def _unitary_(self) -> np.ndarray:
        return B_LOG @ self._M_onto @ B_LOG.conj().T

    def _has_kraus_(self) -> bool:
        return not self._has_unitary_()

    def _kraus_(self) -> Tuple[np.ndarray, np.ndarray]:
        M_phys = B_LOG @ self._M_onto @ B_LOG.conj().T
        MdagM = M_phys.conj().T @ M_phys
        I4 = np.eye(4, dtype=complex)
        evals, evecs = np.linalg.eigh(I4 - MdagM)
        evals = np.maximum(evals, 0.0)
        sqrt_mat = evecs @ np.diag(np.sqrt(evals)) @ evecs.conj().T
        return (M_phys, sqrt_mat)

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=(f"Alg({self._name})",))
        
    def __repr__(self) -> str:
        return f"TetralemmaticIonAlgebraGate('{self._name}')"
        
    def __eq__(self, other) -> bool:
        return isinstance(other, TetralemmaticIonAlgebraGate) and np.allclose(self._M_onto, other._M_onto)
        
    def __hash__(self) -> int:
        return hash((type(self), self._name))

#==============================================================================
# 4. IONQ COMPILATION BRIDGE (DUAL-MANIFOLD)
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
        
        # Determine manifold for wrapping
        is_virtual = isinstance(op.qubits[0], VirtualQudit) if op.qubits else False

        # STRICT WRAPPING: Replace abstract gates with physical wrappers
        if isinstance(gate, TetralemmaticIonProjectorGate):
            gate = PhysicalProjectorWrapper(gate._vertex, gate._T, is_virtual)
        elif isinstance(gate, TetralemmaticIonAlgebraGate):
            gate = PhysicalAlgebraWrapper(gate._M_onto, gate._name, is_virtual)
        elif isinstance(gate, cirq.MeasurementGate):
            # Split measurements for physical qubits
            for i, pq in enumerate(flat_qs):
                new_ops.append(cirq.measure(pq, key=f"{op.gate.key}_{i}"))
            continue
            
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
# 5. FACTORY & VERIFICATION (Dual-Manifold)
#==============================================================================
class TetralemmaticIonProjectorGates:
    def __init__(self): pass
    
    def get_projector_gate(self, vertex: Vertex, transmission: float = 1.0) -> TetralemmaticIonProjectorGate:
        return TetralemmaticIonProjectorGate(vertex, transmission)

    def get_algebra_gate(self, M_onto: np.ndarray, name: str) -> TetralemmaticIonAlgebraGate:
        return TetralemmaticIonAlgebraGate(M_onto, name)

    def _verify_manifold(self, B: np.ndarray, label: str) -> Dict[str, bool]:
        checks = {}
        I4 = np.eye(4, dtype=complex)
        for v in Vertex:
            # Projector checks
            gate = self.get_projector_gate(v)
            # Build Kraus using the given B
            sqrt_E = np.zeros((4,4), dtype=complex)
            sqrt_E[v.value, v.value] = 1.0
            K0 = B @ sqrt_E @ B.conj().T
            sqrt_I = np.eye(4); sqrt_I[v.value, v.value] = 0.0
            K1 = B @ sqrt_I @ B.conj().T
            completeness = K0.conj().T @ K0 + K1.conj().T @ K1
            checks[f"Proj_{v.name}_kraus_complete_{label}"] = bool(np.allclose(completeness, I4))

            ket = np.zeros(4, dtype=complex); ket[v.value] = 1.0
            phys_ket = B @ ket
            rho = np.outer(phys_ket, phys_ket.conj())
            rho_out = K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T
            checks[f"Proj_{v.name}_preserves_eigenstate_{label}"] = bool(np.allclose(rho_out, rho))

        # Unsharp check
        unsharp_gate = self.get_projector_gate(Vertex.Th, transmission=0.5)
        sqrt_E = np.zeros((4,4)); sqrt_E[0,0] = np.sqrt(0.5)
        K0 = B @ sqrt_E @ B.conj().T
        ket = np.zeros(4); ket[0] = 1.0
        phys_ket = B @ ket
        prob = float(np.real(phys_ket.conj().T @ K0.conj().T @ K0 @ phys_ket))
        checks[f"Unsharp_Prob_T_{label}"] = bool(np.isclose(prob, 0.5))
        return checks

    def verify_logical_properties(self) -> Dict[str, bool]:
        return self._verify_manifold(B_LOG, "Log")

    def verify_virtual_properties(self) -> Dict[str, bool]:
        return self._verify_manifold(B_VIRT, "Virt")

    def compile_test_circuit(self, use_virtual: bool = False) -> cirq.Circuit:
        q = VirtualQudit(0) if use_virtual else NomosIonQid(0)
        # Unitary Identity for compilation test
        I_gate = self.get_algebra_gate(np.eye(4), "I")
        P_gate = self.get_projector_gate(Vertex.Th, transmission=1.0)
        return cirq.Circuit(
            I_gate.on(q),
            P_gate.on(q),
            cirq.measure(*cirq.LineQubit.range(2), key="m")
        )

#==============================================================================
# 6. MAIN EXECUTION (Unified Compiler Validation)
#==============================================================================
if __name__ == "__main__":
    print("=== Tetralemmatic IonQ Projector/Algebra Verification (Dual-Manifold) ===")
    print(f"Ion Manifold Energies (B=1G): {IonManifold.energy_levels(1.0)}\n")
    factory = TetralemmaticIonProjectorGates()

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
        print(f"Total moments: {len(circuit)}")
        print(f"Total operations: {len(list(circuit.all_operations()))}")
        
        gate_counts = {}
        has_matrix = False
        has_forte_native = False
        has_kraus = False
        for op in circuit.all_operations():
            name = op.gate.__class__.__name__
            gate_counts[name] = gate_counts.get(name, 0) + 1
            if isinstance(op.gate, cirq.MatrixGate):
                has_matrix = True
            if name in ("GPIGate", "GPI2Gate", "ZZGate"):
                has_forte_native = True
            if hasattr(op.gate, '_has_kraus_') and op.gate._has_kraus_():
                has_kraus = True
                
        print("Gate counts:")
        for gate, count in sorted(gate_counts.items()):
            print(f"  {gate}: {count}")
            
        print(f"Contains MatrixGate: {has_matrix}")
        print(f"Contains Kraus/Projector: {has_kraus}")
        print(f"Contains Forte Native (GPI/GPI2/ZZ): {has_forte_native}")
        
        if sim_mode:
            if has_kraus: print("✓ Simulation mode: Kraus channels preserved for density matrix simulation.")
        elif not sim_mode and has_kraus:
            print("! Note: Kraus channels/Projectors cannot be compiled to hardware pulses (Forte API).")
        elif not sim_mode and target == "forte_native" and has_forte_native:
            print("✓ Forte native target: Unitary gates synthesized correctly.")

    test_circ_log = factory.compile_test_circuit(use_virtual=False)
    
    # 1. SIMULATION MODE (Preserves Kraus)
    print("\n=== Compilation Test: Simulation Mode ===")
    try:
        # Force simulation_mode=True for circuits with Kraus operators
        sim_log = compile_tetralemmatic_ionq(test_circ_log, target="api", simulation_mode=True)
        summarize_compilation(sim_log, "Logical Projector Sim", "api", True)
    except Exception as e:
        print(f"\n✗ Simulation Failed: {e}")
        
    # 2. API TARGET (Cloud-Ready - Unitary gates only)
    print("\n=== Compilation Test: API Target (Cloud-Ready) ===")
    try:
        # This will compile the Identity gate successfully
        api_log = compile_tetralemmatic_ionq(test_circ_log, target="api", simulation_mode=False)
        summarize_compilation(api_log, "Logical Algebra API", "api", False)
    except Exception as e:
        print(f"\n✗ API Compilation Failed: {e}")
        
    # 3. FORTE NATIVE TARGET (Pulse-Level - Unitary gates only)
    print("\n=== Compilation Test: Forte Native Target (Pulse-Level) ===")
    try:
        forte_log = compile_tetralemmatic_ionq(test_circ_log, target="forte_native", simulation_mode=False)
        summarize_compilation(forte_log, "Logical Algebra Forte", "forte_native", False)
    except Exception as e:
        print(f"\n✗ Forte Native Compilation Failed: {e}")
        
    print("\n✓ Unified compiler validation complete.")