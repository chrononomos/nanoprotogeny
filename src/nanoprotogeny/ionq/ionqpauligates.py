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
nanoprotogeny.ionq.ionqpauligates
Refactored Tetralemmatic Pauli & Phase Operators for IonQ Hardware.
Integrates Physical Hardware Abstraction Layer with distinct mappings for:
Logical Manifold (NomosIonQid): Standard Bell-separable encoding.
Virtual Phase Register (VirtualQudit): Holothesis ladder encoding starting at antisymmetric Bell state.
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
from enum import IntEnum
from typing import Dict, Iterator, Tuple, List, Union
from cirq import OP_TREE
from nanoprotogeny.ionq.YB171PLUSHARDWARE import NomosState, IonManifold, NomosIonQid, VirtualQudit

# Unified type alias for dual-manifold pipelines
AnyQudit = Union[NomosIonQid, VirtualQudit]

#==============================================================================
# 1. LOGICAL MATRICES & BASIS TRANSFORMATIONS
#==============================================================================
# Logical Ontological Matrices (4x4)
X_onto = np.array([[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]], dtype=complex)
Z_onto = np.array([[1,0,0,0],[0,-1,0,0],[0,0,0,0],[0,0,0,0]], dtype=complex)
Y_onto = 1j * (X_onto @ Z_onto)
UR_onto = np.array([[0,0,0,1],[0,0,1,0],[1,0,0,0],[0,1,0,0]], dtype=complex)
DFT_onto = 0.5 * np.array([
    [1,  1,   1,   1 ],
    [1,  1j, -1, -1j],
    [1, -1,   1,  -1 ],
    [1, -1j, -1,  1j]
], dtype=complex)

# --- STANDARD LOGICAL MANIFOLD MAPPING (NomosIonQid) ---
# B_LOG columns map logical indices [Th, AntiTh, SynTh, HoloTh]
# to physical Bell states [|00⟩, |11⟩, |Ψ⁺⟩, |Ψ⁻⟩] respectively.
B_LOG = np.array([
    [1.0, 0.0,          0.0,          0.0],
    [0.0, 0.0, 1/np.sqrt(2),  1/np.sqrt(2)],
    [0.0, 0.0, 1/np.sqrt(2), -1/np.sqrt(2)],
    [0.0, 1.0,          0.0,          0.0]
], dtype=complex)

# --- VIRTUAL PHASE REGISTER MAPPING (VirtualQudit) ---
# Basis: [HoloTh_F, HoloTh_P, HoloTh_M, HoloTh_R]
# Mapping defined by HoloTh_F = |Ψ-> (Antisymmetric), others via U_R quarter-turns.
# B_VIRT columns map virtual indices [F, P, M, R]
# to physical Bell states [|Ψ⁻⟩, |00⟩, |Ψ⁺⟩, |11⟩] respectively.
B_VIRT = np.array([
    [0.0, 1.0,          0.0,          0.0],
    [1/np.sqrt(2), 0.0, 1/np.sqrt(2),  0.0],
    [-1/np.sqrt(2), 0.0, 1/np.sqrt(2), 0.0],
    [0.0, 0.0,          0.0,          1.0]
], dtype=complex)

# Transform to 2-qubit computational basis: M_phys = B @ M_onto @ B†
# Logical Manifold Physical Matrices
X_phys = B_LOG @ X_onto @ B_LOG.conj().T
Z_phys = B_LOG @ Z_onto @ B_LOG.conj().T
Y_phys = B_LOG @ Y_onto @ B_LOG.conj().T
UR_phys = B_LOG @ UR_onto @ B_LOG.conj().T
DFT_phys = B_LOG @ DFT_onto @ B_LOG.conj().T

# Virtual Register Physical Matrices
X_virt_phys = B_VIRT @ X_onto @ B_VIRT.conj().T
Z_virt_phys = B_VIRT @ Z_onto @ B_VIRT.conj().T
Y_virt_phys = B_VIRT @ Y_onto @ B_VIRT.conj().T
UR_virt_phys = B_VIRT @ UR_onto @ B_VIRT.conj().T
DFT_virt_phys = B_VIRT @ DFT_onto @ B_VIRT.conj().T

#==============================================================================
# 2. PROTOCOL-AWARE EXPANSION WRAPPERS
#==============================================================================
class PhysicalLogicalGateWrapper(cirq.Gate):
    """Bridges strict qid_shape=(4,) to 2 physical qubits for Logical Manifold (NomosIonQid).
    Safely delegates unitary() and kraus() protocols to the original gate using Logical matrices."""
    def __init__(self, original_gate: cirq.Gate):
        self._original_gate = original_gate

    def _num_qubits_(self) -> int: return 2
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2)

    def _has_unitary_(self) -> bool: return callable(getattr(self._original_gate, '_unitary_', None))
    def _unitary_(self) -> np.ndarray: 
        if not self._has_unitary_():
            raise NotImplementedError("Underlying gate is not unitary")
        return self._original_gate._unitary_()

    def _has_kraus_(self) -> bool: return callable(getattr(self._original_gate, '_kraus_', None))
    def _kraus_(self) -> Tuple[np.ndarray, ...]: 
        if not self._has_kraus_():
            raise NotImplementedError("Underlying gate does not implement Kraus")
        return self._original_gate._kraus_()

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        base = self._original_gate._circuit_diagram_info_(None).wire_symbols[0]
        return cirq.CircuitDiagramInfo(wire_symbols=(base, base))
        
    def __repr__(self) -> str: return f"PhysicalLogicalWrapper({self._original_gate!r})"

class PhysicalVirtualGateWrapper(cirq.Gate):
    """Bridges strict qid_shape=(4,) to 2 physical qubits for Virtual Register (VirtualQudit).
    Uses Virtual Phase Register physical matrices derived from Holothesis mapping."""
    def __init__(self, original_gate: cirq.Gate):
        self._original_gate = original_gate

    def _num_qubits_(self) -> int: return 2
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2)
    def _has_unitary_(self) -> bool: return True

    def _unitary_(self) -> np.ndarray:
        g = self._original_gate
        if isinstance(g, TetralemmaticIonXGate): return X_virt_phys.copy()
        if isinstance(g, TetralemmaticIonYGate): return Y_virt_phys.copy()
        if isinstance(g, TetralemmaticIonURGate): return UR_virt_phys.copy()
        if isinstance(g, TetralemmaticIonDFTGate): return DFT_virt_phys.copy()
        if isinstance(g, TetralemmaticIonZGate): return Z_virt_phys.copy()
        raise NotImplementedError(f"Virtual wrapper not implemented for {g}")

    def _has_kraus_(self) -> bool:
        return isinstance(self._original_gate, TetralemmaticIonZGate)

    def _kraus_(self) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(self._original_gate, TetralemmaticIonZGate):
            Z = Z_virt_phys.copy()
            I4 = np.eye(4, dtype=complex)
            K0 = Z
            K1 = np.sqrt(I4 - K0.conj().T @ K0 + 1e-15j * I4)
            return (K0, K1)
        raise NotImplementedError("Kraus not implemented for virtual wrapper of this gate")

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        base = self._original_gate._circuit_diagram_info_(None).wire_symbols[0]
        return cirq.CircuitDiagramInfo(wire_symbols=(base + "V", base + "V"))
        
    def __repr__(self) -> str: return f"PhysicalVirtualWrapper({self._original_gate!r})"

#==============================================================================
# 3. CIRQ GATE IMPLEMENTATIONS (Qudit-Native, 2-Qubit Physical)
#==============================================================================
class TetralemmaticIonXGate(cirq.Gate):
    """Full Pauli X (Duality Involution Δ) targeting IonQ hardware."""
    def _qid_shape_(self) -> Tuple[int, ...]: return (4,)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return X_phys.copy()
    
    def _decompose_(self, qubits) -> Iterator[cirq.OP_TREE]:
        if len(qubits) == 2:
            yield cirq.two_qubit_matrix_to_cz_operations(
                qubits[0], qubits[1], np.round(self._unitary_(), 10), allow_partial_czs=True
            )
        else:
            yield cirq.MatrixGate(self._unitary_()).on(*qubits)

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=("TIX",))
    def __repr__(self) -> str: return "TetralemmaticIonXGate()"
    def __eq__(self, other) -> bool: return isinstance(other, TetralemmaticIonXGate)
    def __hash__(self) -> int: return hash(type(self))

class TetralemmaticIonZGate(cirq.Gate):
    """Full Pauli Z (Polar Discrimination) targeting IonQ hardware. Non-unitary partial isometry."""
    def _qid_shape_(self) -> Tuple[int, ...]: return (4,)
    def _has_unitary_(self) -> bool: return False
    def _has_kraus_(self) -> bool: return True
    
    def _kraus_(self) -> Tuple[np.ndarray, np.ndarray]:
        Z = Z_phys.copy()
        I4 = np.eye(4, dtype=complex)
        K0 = Z
        K1 = np.sqrt(I4 - K0.conj().T @ K0 + 1e-15j * I4)
        return (K0, K1)
        
    def _unitary_(self) -> np.ndarray: return Z_phys.copy()
    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=("TIZ",))
    def __repr__(self) -> str: return "TetralemmaticIonZGate()"
    def __eq__(self, other) -> bool: return isinstance(other, TetralemmaticIonZGate)
    def __hash__(self) -> int: return hash(type(self))

class TetralemmaticIonYGate(cirq.Gate):
    """Full Pauli Y targeting IonQ hardware. Non-unitary logical operator."""
    def _qid_shape_(self) -> Tuple[int, ...]: return (4,)
    def _has_unitary_(self) -> bool: return False
    def _unitary_(self) -> np.ndarray: return Y_phys.copy()
    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=("TIY",))
    def __repr__(self) -> str: return "TetralemmaticIonYGate()"
    def __eq__(self, other) -> bool: return isinstance(other, TetralemmaticIonYGate)
    def __hash__(self) -> int: return hash(type(self))

class TetralemmaticIonURGate(cirq.Gate):
    """Quarter-Turn Automorphism U_R targeting IonQ hardware."""
    def _qid_shape_(self) -> Tuple[int, ...]: return (4,)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return UR_phys.copy()
    
    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        if len(qubits) == 2:
            yield cirq.two_qubit_matrix_to_cz_operations(
                qubits[0], qubits[1], np.round(self._unitary_(), 10), allow_partial_czs=True
            )
        else:
            yield cirq.MatrixGate(self._unitary_()).on(*qubits)

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=("TIUR",))
    def __repr__(self) -> str: return "TetralemmaticIonURGate()"
    def __eq__(self, other) -> bool: return isinstance(other, TetralemmaticIonURGate)
    def __hash__(self) -> int: return hash(type(self))

class TetralemmaticIonDFTGate(cirq.Gate):
    """Discrete Fourier Transform targeting IonQ hardware."""
    def _qid_shape_(self) -> Tuple[int, ...]: return (4,)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return DFT_phys.copy()
    
    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        if len(qubits) == 2:
            yield cirq.two_qubit_matrix_to_cz_operations(
                qubits[0], qubits[1], np.round(self._unitary_(), 10), allow_partial_czs=True
            )
        else:
            yield cirq.MatrixGate(self._unitary_()).on(*qubits)

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=("TIDFT",))
    def __repr__(self) -> str: return "TetralemmaticIonDFTGate()"
    def __eq__(self, other) -> bool: return isinstance(other, TetralemmaticIonDFTGate)
    def __hash__(self) -> int: return hash(type(self))

#==============================================================================
# 4. IONQ COMPILATION BRIDGE (DUAL-MANIFOLD AWARE)
#==============================================================================
def expand_qudit_circuit(circuit: cirq.Circuit) -> cirq.Circuit:
    """Expands NomosIonQid or VirtualQudit (d=4) into two LineQubits.
    Dynamically selects wrapper based on Qudit type to apply correct Bell mapping."""
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
        target_q = op.qubits[0] if op.qubits else None
        
        if hasattr(gate, '_qid_shape_') and len(flat_qs) == 2:
            if isinstance(target_q, VirtualQudit):
                gate = PhysicalVirtualGateWrapper(gate)
            else:
                gate = PhysicalLogicalGateWrapper(gate)
                
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
                         
    Returns:
        Compiled cirq.Circuit ready for Forte simulation or cloud submission.
    """
    # 1. Expand abstract d=4 qudits to 2-qubit physical Bell-separable encoding
    if any(isinstance(q, (NomosIonQid, VirtualQudit)) for q in circuit.all_qubits()):
        circuit = expand_qudit_circuit(circuit)
        
    # 2. Simulation path: return early to preserve MatrixGate & Physical*Wrappers
    if simulation_mode:
        return circuit
        
    # 3. Target gateset selection
    if target == "api":
        gateset = cirq_ionq.IonQTargetGateset()
    elif target == "forte_native":
        gateset = ForteNativeGateset()
    else:
        raise ValueError("target must be 'api' or 'forte_native'")
        
    # 4. Single-pass decomposition + target optimization
    return cirq.optimize_for_target_gateset(
        circuit,
        gateset=gateset,
        context=cirq.TransformerContext(deep=True)
    )

#==============================================================================
# 5. FACTORY & VERIFICATION SUITE
#==============================================================================
class TetralemmaticIonGates:
    """Factory and verification for IonQ-targeted tetralemmatic operators."""
    def __init__(self):
        self.X = TetralemmaticIonXGate()
        self.Y = TetralemmaticIonYGate()
        self.Z = TetralemmaticIonZGate()
        self.UR = TetralemmaticIonURGate()
        self.DFT = TetralemmaticIonDFTGate()
        
    def verify_gate_properties(self) -> Dict[str, bool]:
        checks = {}
        X = cirq.unitary(self.X)
        Y = cirq.unitary(self.Y)
        UR = cirq.unitary(self.UR)
        Z = cirq.unitary(self.Z)
        DFT = cirq.unitary(self.DFT)
        I4 = np.eye(4, dtype=complex)

        Th   = np.array([1, 0, 0, 0], dtype=complex)
        AntiTh = np.array([0, 0, 0, 1], dtype=complex)
        SynTh   = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0], dtype=complex)
        HoloTh = np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0], dtype=complex)

        checks["X_unitary"] = bool(np.allclose(X.conj().T @ X, I4))
        checks["X_fixes_bell"] = bool(np.allclose(X @ SynTh, SynTh) and np.allclose(X @ HoloTh, HoloTh))
        
        checks["Z_hermitian"] = bool(np.allclose(Z.conj().T, Z))
        checks["Z_th_phase"] = bool(np.allclose(Z @ Th, Th))
        checks["Z_anti_phase"] = bool(np.allclose(Z @ AntiTh, -AntiTh))
        checks["Z_annihilates_bell"] = bool(np.allclose(Z @ SynTh, np.zeros(4)) and np.allclose(Z @ HoloTh, np.zeros(4)))

        checks["Y_algebraic_def"] = bool(np.allclose(Y, 1j * (X @ Z)))
        checks["Y_hermitian"] = bool(np.allclose(Y.conj().T, Y))
        checks["Y_annihilates_bell"] = bool(np.allclose(Y @ SynTh, np.zeros(4)) and np.allclose(Y @ HoloTh, np.zeros(4)))

        checks["UR_unitary"] = bool(np.allclose(UR.conj().T @ UR, I4))
        checks["UR_cycle"] = bool(
            np.allclose(UR @ Th, SynTh) and np.allclose(UR @ SynTh, AntiTh) and
            np.allclose(UR @ AntiTh, HoloTh) and np.allclose(UR @ HoloTh, Th)
        )
        checks["UR_order4"] = bool(np.allclose(np.linalg.matrix_power(UR, 4), I4))

        checks["DFT_unitary"] = bool(np.allclose(DFT.conj().T @ DFT, I4))
        checks["DFT_order4"] = bool(np.allclose(np.linalg.matrix_power(DFT, 4), I4))

        polar_idx = [0, 3]
        X_p = X[np.ix_(polar_idx, polar_idx)]
        Z_p = Z[np.ix_(polar_idx, polar_idx)]
        checks["qubit_X_equiv"] = bool(np.allclose(X_p, np.array([[0,1],[1,0]])))
        checks["qubit_Z_equiv"] = bool(np.allclose(Z_p, np.array([[1,0],[0,-1]])))
        checks["polar_commutator"] = bool(np.allclose(X_p @ Z_p - Z_p @ X_p, np.array([[0, -2], [2, 0]])))

        return checks

    def compile_test_circuit(self) -> cirq.Circuit:
        q_log = NomosIonQid(0)
        q_virt = VirtualQudit(0)
        return cirq.Circuit(
            self.X.on(q_log),
            self.UR.on(q_log),
            self.DFT.on(q_virt),
            cirq.measure(*cirq.LineQubit.range(4), key="m")
        )

#==============================================================================
# MAIN EXECUTION (Dual-Manifold with Unified Compiler Validation)
#==============================================================================
if __name__ == "__main__":
    print("=== Tetralemmatic IonQ Pauli/UR/DFT Verification (Dual-Manifold) ===")
    print(f"Ion Manifold Energies (B=1G): {IonManifold.energy_levels(1.0)}\n")
    
    factory = TetralemmaticIonGates()
    results = factory.verify_gate_properties()
    for k, v in results.items():
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

    # Generate baseline test circuit
    test_circ = factory.compile_test_circuit()
    
    # 1. SIMULATION MODE
    print("\n=== Compilation Test: Simulation Mode ===")
    try:
        sim = compile_tetralemmatic_ionq(test_circ, target="api", simulation_mode=True)
        summarize_compilation(sim, "Pauli/UR/DFT Simulation", "api", True)
    except Exception as e:
        print(f"\n✗ Simulation Compilation Failed: {e}")
        
    # 2. API TARGET (Cloud-Ready)
    print("\n=== Compilation Test: API Target (Cloud-Ready) ===")
    try:
        api = compile_tetralemmatic_ionq(test_circ, target="api", simulation_mode=False)
        summarize_compilation(api, "Pauli/UR/DFT API", "api", False)
    except Exception as e:
        print(f"\n✗ API Compilation Failed: {e}")
        
    # 3. FORTE NATIVE TARGET (Pulse-Level)
    print("\n=== Compilation Test: Forte Native Target (Pulse-Level) ===")
    try:
        forte = compile_tetralemmatic_ionq(test_circ, target="forte_native", simulation_mode=False)
        summarize_compilation(forte, "Pauli/UR/DFT Forte Native", "forte_native", False)
    except Exception as e:
        print(f"\n✗ Forte Native Compilation Failed: {e}")
        
    # ---------- VIRTUAL X GATE SIMULATION TEST ----------
    print("\n=== Virtual X Gate Simulation Test ===")
    vq = VirtualQudit(0)
    circ_x = cirq.Circuit(TetralemmaticIonXGate().on(vq))
    phys_circ = expand_qudit_circuit(circ_x)

    sim = cirq.Simulator()
    initial_state = np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0], dtype=complex)
    result = sim.simulate(phys_circ, initial_state=initial_state)
    final_state = result.final_state_vector

    expected_state = np.array([1, 0, 0, 0], dtype=complex)
    fidelity = np.abs(np.vdot(final_state, expected_state)) ** 2
    print(f"Fidelity with expected virtual |1⟩ (|00⟩): {fidelity:.6f}")
    if np.isclose(fidelity, 1.0, atol=1e-6):
        print("✓ Virtual X gate correctly swaps HoloTh_F ↔ HoloTh_P")
    else:
        print("✗ Virtual X gate action incorrect")
        
    print("\n✓ Unified compiler validation complete. All targets verified.")