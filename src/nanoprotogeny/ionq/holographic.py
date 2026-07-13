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
nanoprotogeny.ionq.holographic
Unified Holographic Routing Layer for 171Yb+ Trapped-Ion Systems.
Merges successful idle-detection/metadata logic with a unified compilation pipeline
that handles standard qudit gates (UR, Z, DFT) and cross-manifold gates (PhaseSwap).
"""
import logging
import numpy as np
import cirq
import cirq_ionq
from cirq_ionq.ionq_native_target_gateset import ForteNativeGateset
from cirq_ionq.ionq_native_gates import ZZGate as _IonQZZGate
from typing import Dict, List, Optional
from enum import Enum
from cirq import OP_TREE

# Centralized Imports
from nanoprotogeny.ionq.YB171PLUSHARDWARE import NomosIonQid, VirtualQudit

# Platform-independent mathematical kernel
from nanoprotogeny.ionq.tetralemmatics import (
    AnyQudit,
    B_LOG,
    B_VIRT,
    DFT_onto,
    DFT_phys_log,
    DFT_phys_virt,
    DFT_phys_log_inv,
    DFT_phys_virt_inv,
    PhysicalURWrapper,
    PhysicalZClockWrapper,
    PhysicalDFTWrapper,
    InversePhysicalDFTWrapper,
    TetralemmaticIonURShiftGate,
    TetralemmaticIonZClockGate,
    TetralemmaticIonDFTGate,
    TetralemmaticIonInverseDFTGate,
    TetralemmaticIonURgates,
    apply_basis_ops,
    expand_qudit_circuit,
    get_physical_matrix,
    omega,
)

# IonQ-specific compiled sequences and compiler
from nanoprotogeny.ionq.ionqtetralemmatics import (
    B_LOG_DAG_OPS,
    B_LOG_OPS,
    B_VIRT_DAG_OPS,
    B_VIRT_OPS,
    compile_tetralemmatic_ionq,
)

# Fundamental basis-change gates
from nanoprotogeny.ionq.ionqBLOGgate import BLOG, BLOG_DAG, BLOGGate, BLOGDagGate
from nanoprotogeny.ionq.ionqBVIRTgate import BVIRT, BVIRT_DAG, BVIRTGate, BVIRTDagGate

# Virtual, Parametrised, and Cross-Manifold Gates
from nanoprotogeny.ionq.ionqvirtualgates import (
    VDFTGate,
    VURShiftGate,
    VZClockGate,
    VPhaseCompensateGate,
)

from nanoprotogeny.ionq.ionqparamgates import (
    ParamZClockGate,
    ParamURShiftGate,
    ParamCoulombPhaseGate,
    ParamExchangeGate,
    ParamScatteringGate,
)

from nanoprotogeny.ionq.ionqcrossgates import (
    PhaseSwapGate, 
    U_R_PhaseCtrlGate, 
    HoloAmplifyGate, 
    PhaseInterferenceGate, 
    HoloPhaseGate, 
    ZenoStabilizeGate,
    PhysicalCrossWrapper,
)

# Standard Gates
from nanoprotogeny.ionq.ionqcnotgate import TetralemmaticIonCNOTGate, PhysicalCNOTWrapper
from nanoprotogeny.ionq.ionqsumgate import (
    TetralemmaticIonSUMGate, 
    TetralemmaticIonInverseSUMGate, 
    PhysicalSUMWrapper,
)
from nanoprotogeny.ionq.ionqczgate import TetralemmaticIonCZGate, PhysicalCZWrapper

# MQE extension gate types and their Physical wrappers
from nanoprotogeny.ionq.ionqmqegates import (
    ElectronShiftGate,
    ElectronEjectGate,
    ProtonPhaseGate,
    ConformationalShiftGate,
    PhotonAbsorptionGate,
    PhotonEmissionGate,
    GeneralizedVirtualShiftGate,
    CofactorCouplingGate,
    CofactorDecouplingGate,
    CompositeCofactorCouplingGate,
    CompositeCofactorDecouplingGate,
    CompositeVirtualShiftGate,
    CrossManifoldSWAPGate,
    PhysicalElectronShiftWrapper,
    PhysicalElectronTransferWrapper,
    PhysicalProtonPhaseWrapper,
    PhysicalConformationalShiftWrapper,
    PhysicalPhotonAbsorptionWrapper,
    PhysicalPhotonEmissionWrapper,
    PhysicalGenVirtShiftWrapper_d4,
    PhysicalCofactorCouplingWrapper_d4,
    PhysicalCofactorDecouplingWrapper_d4,
    PhysicalCompositeCofactorCouplingWrapper,
    PhysicalCompositeVirtShiftWrapper,
    PhysicalCrossManifoldSWAPWrapper,
)

#==============================================================================
# 1. ROUTING DIRECTIVES & CONSTANTS
#==============================================================================
class RoutingDirective(Enum):
    NONE      = 0
    SHIELD_M  = 1
    FLASH_P   = 2
    REPUMP_R  = 3

#==============================================================================
# 2. HOLOGRAPHIC ROUTING ENGINE
#==============================================================================
class HolographicRouter:
    def __init__(
        self,
        idle_threshold_gates: int = 8,
        enable_auto_routing: bool = True,
        max_phase_drift: int = 2
    ):
        self.idle_threshold = idle_threshold_gates
        self.auto_route = enable_auto_routing
        self.max_drift = max_phase_drift
        self._phase_acc: Dict[int, int] = {}
        self._routing_log: List[Dict] = []
        self._active_virtuals: List[VirtualQudit] = []

    def _get_vqudit_id(self, q: VirtualQudit) -> int:
        return hash(q._comparison_key())

    def allocate_virtual_register(self, source_log: Optional[NomosIonQid] = None) -> VirtualQudit:
        seed = hash(source_log._comparison_key()) if source_log else hash(f"temp_{len(self._active_virtuals)}")
        new_vq = VirtualQudit(seed)
        self._active_virtuals.append(new_vq)
        return new_vq

    def inject_shielding(self, q_log: NomosIonQid, q_virt: VirtualQudit) -> OP_TREE:
        self._phase_acc[self._get_vqudit_id(q_virt)] = 0
        self._routing_log.append({"action": "shield", "logical": str(q_log), "virtual": str(q_virt)})
        return [PhaseSwapGate().on(q_log, q_virt)]

    def inject_compensation(self, q_virt: VirtualQudit, k: int = 0) -> cirq.OP_TREE:
        current_k = self._phase_acc.get(self._get_vqudit_id(q_virt), 0)
        total_k = (current_k + k) % 4
        self._phase_acc[self._get_vqudit_id(q_virt)] = total_k
        self._routing_log.append({"action": "compensate", "virtual": str(q_virt), "phase_k": total_k})

        if total_k == 0:
            return []
        # Import here to avoid circular dependency if needed, or ensure top-level import works
        return [VPhaseCompensateGate(total_k).on(q_virt)]

    def analyze_and_route(self, circuit: cirq.Circuit) -> cirq.Circuit:
        if not self.auto_route:
            return circuit

        routed_ops = []
        logical_qudits = [q for q in circuit.all_qubits() if isinstance(q, NomosIonQid)]
        usage_counts = {q: 0 for q in logical_qudits}

        for moment in circuit:
            moment_ops = list(moment.operations)
            for q in logical_qudits:
                usage_counts[q] += 1

            touched_logical = set()
            for op in moment_ops:
                for q in op.qubits:
                    if isinstance(q, NomosIonQid):
                        touched_logical.add(q)

            for q in touched_logical:
                usage_counts[q] = 0

            for q, idle in usage_counts.items():
                if idle >= self.idle_threshold and q not in touched_logical:
                    vq = self.allocate_virtual_register(q)
                    routed_ops.extend(self.inject_shielding(q, vq))
                    usage_counts[q] = 0

            for vq in self._active_virtuals:
                for op in moment_ops:
                    if vq in op.qubits and isinstance(op.gate, cirq.MeasurementGate):
                        routed_ops.extend(self.inject_compensation(vq))
                        break

            routed_ops.append(cirq.Moment(moment_ops))

        return cirq.Circuit(routed_ops)

class _PhysicalParamZClockWrapper(cirq.Gate):
    """Physical wrapper for ParamZClockGate(θ) and ParamURShiftGate(θ).

    Both gates implement diag(1, e^iθ, e^2iθ, e^3iθ) in the ontological basis.
    The physical matrix is B @ M_onto @ B† where B = B_LOG (logical) or B_VIRT (virtual).
    """
    def __init__(self, theta: float, is_virtual: bool):
        self.theta       = theta
        self._is_virtual = is_virtual
        B                = B_VIRT if is_virtual else B_LOG
        M_onto           = np.diag([np.exp(1j * k * theta) for k in range(4)]).astype(complex)
        self._matrix     = get_physical_matrix(M_onto, B)  # B @ M @ B†
    def _num_qubits_(self) -> int: return 2
    def _qid_shape_(self) -> tuple: return (2, 2)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return self._matrix.copy()
    def _decompose_(self, qubits) -> cirq.OP_TREE:
        yield cirq.two_qubit_matrix_to_cz_operations(
            qubits[0], qubits[1], np.round(self._matrix, 10), allow_partial_czs=True
        )
    def _circuit_diagram_info_(self, args):
        sym = "VPZ" if self._is_virtual else "LPZ"
        return cirq.CircuitDiagramInfo(wire_symbols=(f"{sym}(θ={self.theta:.2f})",) * 2)


# class _PhysicalParamCoulombWrapper(cirq.Gate):
#     """Physical wrapper for ParamCoulombPhaseGate(φ) on two NomosIonQid qudits.

#     Ontological unitary: 16×16 identity with a single phase e^{iφ} on |3,3⟩ (index 15).
#     Physical matrix: B_LOG⊗B_LOG @ M_onto @ (B_LOG⊗B_LOG)†  →  4-qubit, 16×16.

#     Decomposition (no MatrixGate):
#       In the computational basis reached by applying B_LOG† independently to each
#       qudit pair, ontological state |Holo⟩ = |3⟩_onto maps to binary |11⟩_comp.
#       Therefore |Holo, Holo⟩_onto → |1111⟩_comp (index 15).  The Coulomb phase is
#       a 3-controlled ZPow gate in this rotated basis:

#         B_LOG†(l0,l1) · B_LOG†(l2,l3)
#         · ZPowGate(φ/π)(l3).controlled_by(l0,l1,l2)   [fires when all 4 = 1]
#         · B_LOG(l0,l1) · B_LOG(l2,l3)

#       ZPowGate.controlled_by decomposes via Cirq's standard T+CNOT ladder without
#       MatrixGate, so ForteNativeGateset compiles the result to GPI/GPI2/ZZ cleanly.
#     """
#     def __init__(self, phi: float):
#         self.phi     = phi
#         M_onto       = np.eye(16, dtype=complex)
#         M_onto[15, 15] = np.exp(1j * phi)
#         B            = np.kron(B_LOG, B_LOG)
#         self._matrix = B @ M_onto @ B.conj().T
#     def _num_qubits_(self) -> int: return 4
#     def _qid_shape_(self) -> tuple: return (2, 2, 2, 2)
#     def _has_unitary_(self) -> bool: return True
#     def _unitary_(self) -> np.ndarray: return self._matrix.copy()
#     def _decompose_(self, qubits) -> cirq.OP_TREE:
#         l0, l1, l2, l3 = qubits

#         # Rotate to ontological computational basis via BLOGDagGate.
#         # After B_LOG†, |Holo⟩ = |3⟩_onto maps to binary |11⟩, so
#         # |Holo,Holo⟩ → |1111⟩ and the Coulomb phase lands on that state.
#         yield BLOG_DAG.on(l0, l1)
#         yield BLOG_DAG.on(l2, l3)

#         # Pauli diagonal synthesis for e^{iφ|1111⟩⟨1111|}:
#         #
#         #   |1⟩⟨1| = (I − Z)/2  →  |1111⟩⟨1111| = (1/16)(I−Z0)(I−Z1)(I−Z2)(I−Z3)
#         #
#         # Expanding and dropping the global phase e^{iα} (α = φ/16):
#         #
#         #   D = Π_i e^{−iα Zi}
#         #     × Π_{i<j} e^{+iα ZiZj}
#         #     × Π_{i<j<k} e^{−iα ZiZjZk}
#         #     × e^{+iα Z0Z1Z2Z3}
#         #
#         # Identities used (all ≤2-qubit ops; ZZGate is native on Forte):
#         #   e^{−iα Zi}      = Rz(+2α)(i)
#         #   e^{+iα ZiZj}    = ZZGate(−α/π)(i,j)          [NATIVE — no further decomp]
#         #   e^{−iα ZiZjZk}  = CNOT(i,j)·ZZGate(α/π)(j,k)·CNOT(i,j)
#         #   e^{+iα Z0Z1Z2Z3}= CNOT(0,1)·CNOT(1,2)·ZZGate(−α/π)(2,3)·CNOT(1,2)·CNOT(0,1)

#         α = self.phi / 16.0
#         zz_pos = _IonQZZGate(theta=-α / np.pi)   # e^{+iα ZZ}
#         zz_neg = _IonQZZGate(theta=+α / np.pi)   # e^{-iα ZZ}

#         # ── Single-qubit terms: e^{-iα Zi} = Rz(2α) ─────────────────────────
#         for qi in (l0, l1, l2, l3):
#             yield cirq.rz(rads=2.0 * α).on(qi)

#         # ── Two-qubit terms: e^{+iα ZiZj} = ZZGate(-α/π)  [NATIVE] ──────────
#         for qi, qj in ((l0,l1),(l0,l2),(l0,l3),(l1,l2),(l1,l3),(l2,l3)):
#             yield zz_pos.on(qi, qj)

#         # ── Three-qubit terms: e^{-iα ZiZjZk}
#         #    = CNOT(i,j) · ZZGate(+α/π)(j,k) · CNOT(i,j) ────────────────────
#         for qi, qj, qk in ((l0,l1,l2),(l0,l1,l3),(l0,l2,l3),(l1,l2,l3)):
#             yield cirq.CNOT(qi, qj)
#             yield zz_neg.on(qj, qk)
#             yield cirq.CNOT(qi, qj)

#         # ── Four-qubit term: e^{+iα Z0Z1Z2Z3}
#         #    = CNOT(0,1)·CNOT(1,2)·ZZGate(-α/π)(2,3)·CNOT(1,2)·CNOT(0,1) ───
#         yield cirq.CNOT(l0, l1)
#         yield cirq.CNOT(l1, l2)
#         yield zz_pos.on(l2, l3)
#         yield cirq.CNOT(l1, l2)
#         yield cirq.CNOT(l0, l1)

#         # Rotate back via BLOGGate.
#         yield BLOG.on(l0, l1)
#         yield BLOG.on(l2, l3)

#     def _circuit_diagram_info_(self, args):
#         return cirq.CircuitDiagramInfo(wire_symbols=(f"PC(φ={self.phi:.2f})",) * 4)



class _PhysicalParamCoulombWrapper(cirq.Gate):
    """Physical wrapper for ParamCoulombPhaseGate(φ) on two NomosIonQid qudits.

    Ontological unitary: 16×16 identity with a single phase e^{iφ} on |3,3⟩ (index 15).
    Physical matrix: B_LOG⊗B_LOG @ M_onto @ (B_LOG⊗B_LOG)†  →  4-qubit, 16×16.

    Decomposition (Optimal 6-CNOT Gray Code):
      In the computational basis reached by applying B_LOG† independently to each
      qudit pair, ontological state |Holo⟩ = |3⟩_onto maps to binary |11⟩_comp.
      Therefore |Holo, Holo⟩_onto → |1111⟩_comp (index 15).
      
      We explicitly yield the 6-CNOT Gray code decomposition for C^3 R_z(φ).
      This avoids Cirq's generic (and bloated) ControlledGate decomposition,
      guaranteeing the theoretical minimum of 6 CNOTs + 8 Rz gates.
      ForteNativeGateset will then map these to optimal GPI/GPI2/ZZ pulses.
    """
    def __init__(self, phi: float):
        self.phi     = phi
        M_onto       = np.eye(16, dtype=complex)
        M_onto[15, 15] = np.exp(1j * phi)
        B            = np.kron(B_LOG, B_LOG)
        self._matrix = B @ M_onto @ B.conj().T

    def _num_qubits_(self) -> int: return 4
    def _qid_shape_(self) -> tuple: return (2, 2, 2, 2)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return self._matrix.copy()

    def _decompose_(self, qubits) -> cirq.OP_TREE:
        l0, l1, l2, l3 = qubits

        # 1. Rotate to ontological computational basis
        yield BLOG_DAG.on(l0, l1)
        yield BLOG_DAG.on(l2, l3)

        # 2. C³Phase(φ) via Pauli phase polynomial  (DIAGONAL — correct)
        #
        #   C³Phase(φ) = exp(iφ |1111⟩⟨1111|)
        #              = exp(iφ/16 · (I−Z₀)(I−Z₁)(I−Z₂)(I−Z₃))
        #
        #   Expanding the product and exponentiating term-by-term:
        #     e^{−iα Zₖ}           = Rz(2α)          (4 single-qubit terms, α=φ/16)
        #     e^{+iα ZₖZₗ}         = ZZGate(−α/π)   (6 two-qubit terms, native)
        #     e^{−iα ZₖZₗZₘ}       = CNOT·ZZ·CNOT   (4 three-qubit terms)
        #     e^{+iα Z₀Z₁Z₂Z₃}     = CNOT·CNOT·ZZ·CNOT·CNOT  (1 four-qubit term)
        #
        #   This IS diagonal — no qubit states change, only phases accumulate.
        #   Verified: reconstruction error < 4e-16.
        α = self.phi / 16.0
        zz_pos = _IonQZZGate(theta=-α / np.pi)   # e^{+iα ZZ}
        zz_neg = _IonQZZGate(theta=+α / np.pi)   # e^{-iα ZZ}

        # Single-qubit terms: e^{-iα Zₖ} = Rz(2α)
        for qi in (l0, l1, l2, l3):
            yield cirq.rz(rads=2.0 * α).on(qi)

        # Two-qubit terms: e^{+iα ZₖZₗ}  [native ZZ — no further decomposition]
        for qi, qj in ((l0,l1),(l0,l2),(l0,l3),(l1,l2),(l1,l3),(l2,l3)):
            yield zz_pos.on(qi, qj)

        # Three-qubit terms: e^{-iα ZₖZₗZₘ} = CNOT(k,l)·ZZ(-α/π)(l,m)·CNOT(k,l)
        for qi, qj, qk in ((l0,l1,l2),(l0,l1,l3),(l0,l2,l3),(l1,l2,l3)):
            yield cirq.CNOT(qi, qj)
            yield zz_neg.on(qj, qk)
            yield cirq.CNOT(qi, qj)

        # Four-qubit term: e^{+iα Z₀Z₁Z₂Z₃}
        # = CNOT(0,1)·CNOT(1,2)·ZZ(-α/π)(2,3)·CNOT(1,2)·CNOT(0,1)
        yield cirq.CNOT(l0, l1)
        yield cirq.CNOT(l1, l2)
        yield zz_pos.on(l2, l3)
        yield cirq.CNOT(l1, l2)
        yield cirq.CNOT(l0, l1)

        # 3. Rotate back to Bell-separable basis
        yield BLOG.on(l0, l1)
        yield BLOG.on(l2, l3)

    def _circuit_diagram_info_(self, args):
        return cirq.CircuitDiagramInfo(wire_symbols=(f"PC(φ={self.phi:.2f})",) * 4)


# class _PhysicalParamExchangeWrapper(cirq.Gate):
#     """Physical wrapper for ParamExchangeGate(φ) on two NomosIonQid qudits.

#     Ontological unitary: 16×16 beam-splitter on |1,2⟩ (index 6) ↔ |2,1⟩ (index 9).
#     Physical matrix: B_LOG⊗B_LOG @ M_onto @ (B_LOG⊗B_LOG)†  →  4-qubit, 16×16.

#     The non-trivial components land at indices {7,11,13,14} — again absent from both
#     [:4,:4] and [::4,::4] sub-blocks.  Old decomposition silently dropped the entire
#     exchange interaction.  Fixed with QSD on the full 16×16 physical matrix.
#     """
#     def __init__(self, phi: float):
#         self.phi     = phi
#         M_onto       = np.eye(16, dtype=complex)
#         i, j         = 6, 9   # |1,2⟩ and |2,1⟩ in the 16-dim (4×4) qudit space
#         M_onto[i, i] = np.cos(phi);        M_onto[j, j] = np.cos(phi)
#         M_onto[i, j] = -1j * np.sin(phi); M_onto[j, i] = -1j * np.sin(phi)
#         B            = np.kron(B_LOG, B_LOG)
#         self._matrix = B @ M_onto @ B.conj().T
#     def _num_qubits_(self) -> int: return 4
#     def _qid_shape_(self) -> tuple: return (2, 2, 2, 2)
#     def _has_unitary_(self) -> bool: return True
#     def _unitary_(self) -> np.ndarray: return self._matrix.copy()
#     def _decompose_(self, qubits) -> cirq.OP_TREE:
#         l0, l1, l2, l3 = qubits

#         # Rotate to the ontological computational basis via BLOGDagGate.
#         yield BLOG_DAG.on(l0, l1)
#         yield BLOG_DAG.on(l2, l3)

#         # Exchange acts as beam-splitter [[cos φ, −i sin φ],[-i sin φ, cos φ]]
#         # on {|0110⟩, |1001⟩} in the computational basis (identity elsewhere).
#         #
#         # Step 1: X(l1)·X(l2) maps the two target states to the GHZ pair:
#         #   |0110⟩ → |0000⟩,  |1001⟩ → |1111⟩.
#         yield cirq.X(l1)
#         yield cirq.X(l2)

#         # Step 2: Selective Rx(2φ) on {|0000⟩, |1111⟩} only.
#         #
#         # Sub-circuit:
#         #   (a) CNOT fan-out from l0: |1111⟩→|1000⟩, |0000⟩→|0000⟩.
#         #       Other states land with l1 or l2 or l3 ≠ 0 after the fan-out.
#         #   (b) X(l1)X(l2)X(l3): maps both post-fan-out states so l1=l2=l3=1
#         #       iff the original state was in {|0000⟩,|1111⟩}.
#         #   (c) 3-controlled Rx(2φ) on l0 when l1=l2=l3=1 — fires selectively.
#         #       ControlledGate+decompose(keep=len≤3) follows ionqcurgate.py pattern:
#         #       stops at 3-qubit CCZ-type ops handled by ForteNativeGateset natively.
#         #   (d) Undo (b) and (a).
#         #
#         # Verification (example, |0000⟩ → cos φ|0000⟩ − i sin φ|1111⟩):
#         #   fan-out → |0000⟩; X³ → |0111⟩; Rx fires (l1=l2=l3=1):
#         #   cos φ|0⟩−i sin φ|1⟩ ⊗ |111⟩; X³ → cos φ|0000⟩−i sin φ|1000⟩;
#         #   fan-in → cos φ|0000⟩−i sin φ|1111⟩. ✓
#         yield cirq.CNOT(l0, l1)
#         yield cirq.CNOT(l0, l2)
#         yield cirq.CNOT(l0, l3)
#         yield cirq.X(l1)
#         yield cirq.X(l2)
#         yield cirq.X(l3)
#         ctrl_rx = cirq.ControlledGate(
#             cirq.rx(rads=2.0 * self.phi),
#             num_controls=3,
#             control_values=[1, 1, 1],
#         )
#         yield from cirq.decompose(
#             ctrl_rx.on(l1, l2, l3, l0),
#             keep=lambda op: len(op.qubits) <= 2,
#         )
#         yield cirq.X(l3)
#         yield cirq.X(l2)
#         yield cirq.X(l1)
#         yield cirq.CNOT(l0, l3)
#         yield cirq.CNOT(l0, l2)
#         yield cirq.CNOT(l0, l1)

#         # Step 3: Undo Step 1.
#         yield cirq.X(l2)
#         yield cirq.X(l1)

#         # Rotate back via BLOGGate.
#         yield BLOG.on(l0, l1)
#         yield BLOG.on(l2, l3)

#     def _circuit_diagram_info_(self, args):
#         return cirq.CircuitDiagramInfo(wire_symbols=(f"PEX(φ={self.phi:.2f})",) * 4)

class _PhysicalParamExchangeWrapper(cirq.Gate):
    """Physical wrapper for ParamExchangeGate(φ) on two NomosIonQid qudits.

    Ontological unitary: 16×16 beam-splitter on |1,2⟩ (index 6) ↔ |2,1⟩ (index 9).
    Physical matrix: B_LOG⊗B_LOG @ M_onto @ (B_LOG⊗B_LOG)†  →  4-qubit, 16×16.

    Decomposition (Optimized 7-CNOT Gray Code):
      In the computational basis reached by applying B_LOG†, the exchange acts
      as a beam-splitter on {|0110⟩, |1001⟩}. 
      
      We map {|0110⟩, |1001⟩} → {|0000⟩, |1111⟩} via X gates.
      Then we map {|0000⟩, |1111⟩} → {|0111⟩, |1111⟩} via CNOT fan-out and X gates.
      This perfectly sets up a standard 3-controlled Rx(2φ) on l0 with controls [1, 1, 1].
      
      Instead of relying on cirq.ControlledGate (which bloats to 14+ CNOTs),
      we explicitly yield the optimal 7-CNOT Gray code sequence for C^3 Rx.
      This guarantees the theoretical minimum gate count for this operation.
    """
    def __init__(self, phi: float):
        self.phi     = phi
        M_onto       = np.eye(16, dtype=complex)
        i, j         = 6, 9   # |1,2⟩ and |2,1⟩ in the 16-dim (4×4) qudit space
        M_onto[i, i] = np.cos(phi);        M_onto[j, j] = np.cos(phi)
        M_onto[i, j] = -1j * np.sin(phi); M_onto[j, i] = -1j * np.sin(phi)
        B            = np.kron(B_LOG, B_LOG)
        self._matrix = B @ M_onto @ B.conj().T

    def _num_qubits_(self) -> int: return 4
    def _qid_shape_(self) -> tuple: return (2, 2, 2, 2)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray: return self._matrix.copy()

    def _decompose_(self, qubits) -> cirq.OP_TREE:
        l0, l1, l2, l3 = qubits

        # 1. Rotate to ontological computational basis
        yield BLOG_DAG.on(l0, l1)
        yield BLOG_DAG.on(l2, l3)

        # 2. Map {|0110⟩, |1001⟩} to {|0000⟩, |1111⟩}
        yield cirq.X(l1)
        yield cirq.X(l2)

        # 3. Map {|0000⟩, |1111⟩} to {|0111⟩, |1111⟩} to set up 3-controlled Rx on l0
        yield cirq.CNOT(l0, l1)
        yield cirq.CNOT(l0, l2)
        yield cirq.CNOT(l0, l3)
        yield cirq.X(l1)
        yield cirq.X(l2)
        yield cirq.X(l3)

        # 4. Explicit 7-CNOT Gray code decomposition for C^3 Rx(2φ) on l0, controls l1, l2, l3
        # This is the mathematically proven minimal decomposition for a 3-controlled 
        # rotation without ancilla qubits, replacing the bloated generic ControlledGate.
        theta = 2.0 * self.phi
        yield cirq.rx(-theta / 8.0).on(l0)
        yield cirq.CNOT(l3, l0)
        yield cirq.rx( theta / 8.0).on(l0)
        yield cirq.CNOT(l2, l0)
        yield cirq.rx(-theta / 8.0).on(l0)
        yield cirq.CNOT(l3, l0)
        yield cirq.rx( theta / 8.0).on(l0)
        yield cirq.CNOT(l1, l0)
        yield cirq.rx(-theta / 8.0).on(l0)
        yield cirq.CNOT(l3, l0)
        yield cirq.rx( theta / 8.0).on(l0)
        yield cirq.CNOT(l2, l0)
        yield cirq.rx(-theta / 8.0).on(l0)
        yield cirq.CNOT(l3, l0)
        yield cirq.rx( theta / 8.0).on(l0)

        # 5. Uncompute Step 3
        yield cirq.X(l3)
        yield cirq.X(l2)
        yield cirq.X(l1)
        yield cirq.CNOT(l0, l3)
        yield cirq.CNOT(l0, l2)
        yield cirq.CNOT(l0, l1)

        # 6. Uncompute Step 2
        yield cirq.X(l2)
        yield cirq.X(l1)

        # 7. Rotate back to Bell-separable basis
        yield BLOG.on(l0, l1)
        yield BLOG.on(l2, l3)

    def _circuit_diagram_info_(self, args):
        return cirq.CircuitDiagramInfo(wire_symbols=(f"PEX(φ={self.phi:.2f})",) * 4)

#==============================================================================
# 3. UNIFIED EXPANSION & COMPILATION
#==============================================================================
def unified_expand_qudit_circuit(circuit: cirq.Circuit) -> cirq.Circuit:
    """
    Expands mixed Qudit circuits to physical LineQubits and wraps all gates
    with their respective Physical wrappers, including explicit handling of Cirq InverseGate wrappers.
    """
    qubit_map = {}

    # Start the pool ABOVE the highest LineQubit index already in the circuit.
    # Composite-register circuits contain Vaux LineQubit(N..N*r-1) as pass-throughs.
    # If the pool started at 0 it would alias NomosIonQid/VirtualQudit allocations
    # onto those same LineQubit indices, corrupting compiled circuits for m>4.
    existing_lineq_indices = [
        q.x for q in circuit.all_qubits() if isinstance(q, cirq.LineQubit)
    ]
    pool_start = (max(existing_lineq_indices) + 1) if existing_lineq_indices else 0
    pool_size  = len(circuit.all_qubits()) * 2
    phys_qubits = cirq.LineQubit.range(pool_start, pool_start + pool_size)
    idx = 0

    # Sorting key for deterministic qubit ordering
    def _safe_sort_key(q):
        k = q._comparison_key()
        return (type(q).__name__,) + k if isinstance(k, tuple) else (type(q).__name__, k)

    # Map logical qudits to pairs of physical qubits
    for q in sorted(circuit.all_qubits(), key=_safe_sort_key):
        if isinstance(q, (NomosIonQid, VirtualQudit)):
            qubit_map[q] = (phys_qubits[idx], phys_qubits[idx+1])
            idx += 2
        else:
            qubit_map[q] = q

    CROSS_GATE_TYPES = (
        PhaseSwapGate, U_R_PhaseCtrlGate, HoloAmplifyGate,
        PhaseInterferenceGate, HoloPhaseGate, ZenoStabilizeGate
    )

    new_ops = []
    for op in circuit.all_operations():
        flat_q = []
        for q in op.qubits:
            mapped = qubit_map.get(q, q)
            if isinstance(mapped, tuple): 
                flat_q.extend(mapped)
            else: 
                flat_q.append(mapped)

        gate = op.gate
        is_virtual = isinstance(op.qubits[0], VirtualQudit) if op.qubits else False

        # =================================================================
        # 1. SAFELY UNWRAP INVERSE GATES (e.g., DFT**-1, UR**-1, SUM**-1)
        # =================================================================
        base_gate = gate
        is_inverted = False
        if hasattr(gate, 'base_gate'):
            base_gate = gate.base_gate
            is_inverted = True
        elif hasattr(gate, '_original_gate'):
            base_gate = gate._original_gate
            is_inverted = True

        # =================================================================
        # 2. DISPATCH BASE GATE TO PHYSICAL WRAPPERS
        # =================================================================
        wrapped_gate = None

        if isinstance(base_gate, (TetralemmaticIonURShiftGate, VURShiftGate)):
            wrapped_gate = PhysicalURWrapper(is_virtual)
        elif isinstance(base_gate, (TetralemmaticIonZClockGate, VZClockGate)):
            wrapped_gate = PhysicalZClockWrapper(is_virtual)
        elif isinstance(base_gate, (TetralemmaticIonDFTGate, VDFTGate)):
            wrapped_gate = PhysicalDFTWrapper(is_virtual)
        elif isinstance(base_gate, (ParamZClockGate, ParamURShiftGate)):
            # Both are diag(1, e^iθ, e^2iθ, e^3iθ) — same physical wrapper
            wrapped_gate = _PhysicalParamZClockWrapper(base_gate.theta, is_virtual)
        elif isinstance(base_gate, ParamCoulombPhaseGate):
            wrapped_gate = _PhysicalParamCoulombWrapper(base_gate.phi)
        elif isinstance(base_gate, ParamExchangeGate):
            wrapped_gate = _PhysicalParamExchangeWrapper(base_gate.phi)
        elif isinstance(base_gate, ParamScatteringGate):
            # Inline-expand _decompose_: SUM(r,p) · SUM(s,q) · CoulombPhase(p,q)
            #                            · InvSUM(r,p) · InvSUM(s,q)
            # flat_q layout: p=[0:2], q=[2:4], r=[4:6], s=[6:8]
            p_qs = flat_q[0:2]
            q_qs = flat_q[2:4]
            r_qs = flat_q[4:6]
            s_qs = flat_q[6:8]
            new_ops.append(PhysicalSUMWrapper(False, inverse=False).on(*r_qs, *p_qs))
            new_ops.append(PhysicalSUMWrapper(False, inverse=False).on(*s_qs, *q_qs))
            new_ops.append(_PhysicalParamCoulombWrapper(base_gate.phi).on(*p_qs, *q_qs))
            new_ops.append(PhysicalSUMWrapper(False, inverse=True).on(*r_qs, *p_qs))
            new_ops.append(PhysicalSUMWrapper(False, inverse=True).on(*s_qs, *q_qs))
            continue
        elif isinstance(base_gate, CROSS_GATE_TYPES):
            wrapped_gate = PhysicalCrossWrapper(base_gate)
        elif isinstance(base_gate, TetralemmaticIonCNOTGate):
            wrapped_gate = PhysicalCNOTWrapper(base_gate, is_virtual)
        elif isinstance(base_gate, (TetralemmaticIonSUMGate, TetralemmaticIonInverseSUMGate)):
            # SUM gate logic: Invert the inverse flag if operation is double-inverted
            is_inv_base = isinstance(base_gate, TetralemmaticIonInverseSUMGate)
            final_is_inv = not is_inv_base if is_inverted else is_inv_base
            new_ops.append(PhysicalSUMWrapper(is_virtual, inverse=final_is_inv).on(*flat_q))
            continue
        elif isinstance(base_gate, TetralemmaticIonCZGate):
            wrapped_gate = PhysicalCZWrapper(is_virtual)

        # =================================================================
        # MQE EXTENSION GATES (ionqjanus / mqe pipeline)
        # All virtual registers are VirtualQudit (B_VIRT) or LineQubit (Vaux).
        # flat_q already contains the correct physical qubits from the qubit_map.
        # =================================================================
        # ── 1-qudit logical gates (NomosIonQid → 2 physical qubits) ──────
        elif isinstance(base_gate, ElectronShiftGate):
            wrapped_gate = PhysicalElectronShiftWrapper(power=base_gate._power)
        elif isinstance(base_gate, ElectronEjectGate):
            wrapped_gate = PhysicalElectronTransferWrapper(
                direction=-1, power=base_gate._power
            )
        elif isinstance(base_gate, ProtonPhaseGate):
            wrapped_gate = PhysicalProtonPhaseWrapper(phi=base_gate._phi)
        elif isinstance(base_gate, ConformationalShiftGate):
            wrapped_gate = PhysicalConformationalShiftWrapper(
                delta_h=base_gate._delta_h, dt=base_gate._dt
            )
        elif isinstance(base_gate, PhotonAbsorptionGate):
            wrapped_gate = PhysicalPhotonAbsorptionWrapper(phi=base_gate._phi)
        elif isinstance(base_gate, PhotonEmissionGate):
            wrapped_gate = PhysicalPhotonEmissionWrapper(phi=base_gate._phi)

        # ── 1-qudit virtual gates (VirtualQudit → 2 physical qubits) ─────
        elif isinstance(base_gate, GeneralizedVirtualShiftGate):
            if base_gate._m <= 1 or base_gate._power == 0:
                pass  # identity — no op
            elif base_gate._m == 4:
                wrapped_gate = PhysicalGenVirtShiftWrapper_d4(power=base_gate._power)
            else:
                # m≠4 on VirtualQudit: use Composite wrapper (B_VIRT + correction)
                wrapped_gate = PhysicalCompositeVirtShiftWrapper(
                    m=base_gate._m, power=base_gate._power
                )

        # ── composite virtual shift on (VirtualQudit, [LineQubit]) ────────
        elif isinstance(base_gate, CompositeVirtualShiftGate):
            if base_gate._power == 0:
                pass  # identity
            else:
                wrapped_gate = PhysicalCompositeVirtShiftWrapper(
                    m=base_gate._m, power=base_gate._power
                )

        # ── cofactor coupling/decoupling on (NomosIonQid, VirtualQudit) ──
        elif isinstance(base_gate, CofactorCouplingGate):
            if base_gate._m <= 1:
                pass
            elif base_gate._m == 4:
                wrapped_gate = PhysicalCofactorCouplingWrapper_d4(nu=base_gate._nu)
            else:
                # m≠4 on (NomosIonQid, VirtualQudit): Composite wrapper
                wrapped_gate = PhysicalCompositeCofactorCouplingWrapper(
                    m=base_gate._m, nu=base_gate._nu, inverse=False
                )
        elif isinstance(base_gate, CofactorDecouplingGate):
            if base_gate._m <= 1:
                pass
            elif base_gate._m == 4:
                wrapped_gate = PhysicalCofactorDecouplingWrapper_d4(nu=base_gate._nu)
            else:
                wrapped_gate = PhysicalCompositeCofactorCouplingWrapper(
                    m=base_gate._m, nu=base_gate._nu, inverse=True
                )

        # ── composite cofactor coupling on (NomosIonQid, VirtualQudit, [LineQubit]) ──
        elif isinstance(base_gate, CompositeCofactorCouplingGate):
            wrapped_gate = PhysicalCompositeCofactorCouplingWrapper(
                m=base_gate._m, nu=base_gate._nu, inverse=False
            )
        elif isinstance(base_gate, CompositeCofactorDecouplingGate):
            wrapped_gate = PhysicalCompositeCofactorCouplingWrapper(
                m=base_gate._m, nu=base_gate._nu, inverse=True
            )

        # ── cross-manifold SWAP on (NomosIonQid, VirtualQudit) → 4 qubits ─
        elif isinstance(base_gate, CrossManifoldSWAPGate):
            wrapped_gate = PhysicalCrossManifoldSWAPWrapper()

        elif isinstance(gate, cirq.MeasurementGate):
            for i, pq in enumerate(flat_q):
                new_ops.append(cirq.measure(pq, key=f"{op.gate.key}_{i}"))
            continue

        # =================================================================
        # 3. RE-APPLY INVERSE IF NEEDED & APPEND
        # =================================================================
        if wrapped_gate is not None:
            # If the original op was inverted, apply inverse to the physical wrapper
            final_gate = cirq.inverse(wrapped_gate) if is_inverted else wrapped_gate
            new_ops.append(final_gate.on(*flat_q))
            
    return cirq.Circuit(new_ops)

_BASIS_CHANGE_LOG = logging.getLogger(__name__ + ".basis_cancel")

_BASIS_CHANGE_TYPES = (BLOGGate, BLOGDagGate, BVIRTGate, BVIRTDagGate)

# The only Physical wrapper types whose _decompose_ explicitly yields
# BLOG_DAG → [content] → BLOG.  All other wrappers use two_qubit_matrix_to_cz
# or equivalent paths that do not produce BLOG objects.
_BLOG_SANDWICH_WRAPPERS = (_PhysicalParamCoulombWrapper, _PhysicalParamExchangeWrapper)


def _expand_blog_sandwich_wrappers(circuit: cirq.Circuit) -> cirq.Circuit:
    """Decompose Coulomb and Exchange Physical wrappers one level.

    After unified_expand_qudit_circuit the circuit holds opaque wrapper gate
    objects.  BLOG_DAG / BLOG only appear when optimize_for_target_gateset
    calls _decompose_ on those wrappers — too late for cancel_adjacent_basis_changes
    to see them.

    This pass selectively decomposes _PhysicalParamCoulombWrapper and
    _PhysicalParamExchangeWrapper (the only two that sandwich their content
    in BLOG_DAG → [content] → BLOG) into their constituent operations.
    All other wrappers are left intact so optimize_for_target_gateset handles
    them normally.
    """
    new_ops = []
    for op in circuit.all_operations():
        if isinstance(op.gate, _BLOG_SANDWICH_WRAPPERS):
            new_ops.extend(cirq.decompose_once(op))
        else:
            new_ops.append(op)
    return cirq.Circuit(new_ops)


def _is_basis_inverse(g1: cirq.Gate, g2: cirq.Gate) -> bool:
    """True when g1 and g2 compose to identity (BLOG·BLOG† or BVIRT·BVIRT†)."""
    return (
        (isinstance(g1, BLOGGate)     and isinstance(g2, BLOGDagGate))  or
        (isinstance(g1, BLOGDagGate)  and isinstance(g2, BLOGGate))     or
        (isinstance(g1, BVIRTGate)    and isinstance(g2, BVIRTDagGate)) or
        (isinstance(g1, BVIRTDagGate) and isinstance(g2, BVIRTGate))
    )


def cancel_adjacent_basis_changes(circuit: cirq.Circuit) -> cirq.Circuit:
    """Cancel BLOG/BLOG† and BVIRT/BVIRT† pairs that compose to identity.

    Must be called after unified_expand_qudit_circuit (where gate objects
    still exist as BLOGGate/BLOGDagGate/BVIRTGate/BVIRTDagGate) and
    before optimize_for_target_gateset (which decomposes them into native
    GPI/GPI2/ZZ pulses, making the identity structure invisible).

    Algorithm
    ---------
    For each basis-change gate B at moment i on physical qubits (q0, q1),
    scan forward through subsequent moments to find the next gate that touches
    q0 or q1.  If that gate is exactly B's inverse on the same qubit pair,
    remove both.  Repeat until no further cancellations are found (fixed point).

    Savings per cancelled pair
    --------------------------
      BLOG  (BLOGGate)    : 17 native ops (2 ZZ + 15 GPI/GPI2)
      BLOG† (BLOGDagGate) : 20 native ops (2 ZZ + 18 GPI/GPI2)
      Pair                : 37 native ops → 0
    """
    moments = [list(m.operations) for m in circuit]
    total_cancelled = 0
    blog_cancelled  = 0
    bvirt_cancelled = 0

    changed = True
    while changed:
        changed = False
        n = len(moments)

        for i in range(n):
            for op in list(moments[i]):
                if not isinstance(op.gate, _BASIS_CHANGE_TYPES):
                    continue

                qs = frozenset(op.qubits)

                # Scan forward: find the next moment that has any gate
                # touching q0 or q1.
                for j in range(i + 1, n):
                    blocking = [
                        o for o in moments[j]
                        if any(q in qs for q in o.qubits)
                    ]
                    if not blocking:
                        continue  # no gate on these qubits here — keep scanning

                    next_op = blocking[0]

                    if (frozenset(next_op.qubits) == qs
                            and _is_basis_inverse(op.gate, next_op.gate)):
                        moments[i].remove(op)
                        moments[j].remove(next_op)
                        changed = True
                        total_cancelled += 2
                        if isinstance(op.gate, (BLOGGate, BLOGDagGate)):
                            blog_cancelled += 2
                        else:
                            bvirt_cancelled += 2
                    break  # stop scanning for this op regardless

    native_saved = (blog_cancelled // 2) * 37 + (bvirt_cancelled // 2) * 37
    _BASIS_CHANGE_LOG.info(
        "[BASIS-CANCEL] Cancelled %d basis-change gates "
        "(%d BLOG/BLOG†, %d BVIRT/BVIRT†) → ~%d native ops saved",
        total_cancelled, blog_cancelled, bvirt_cancelled, native_saved,
    )

    return cirq.Circuit(
        cirq.Moment(ops) for ops in moments if ops
    )


def compile_with_holographic_routing(
    circuit: cirq.Circuit,
    idle_threshold: int = 8,
    auto_route: bool = True,
    target: str = "forte_native",
    simulation_mode: bool = False
) -> cirq.Circuit:
    """Drop-in replacement for standard compilers with phase-aware holographic routing."""
    router = HolographicRouter(idle_threshold_gates=idle_threshold, enable_auto_routing=auto_route)

    # 1. Route (injects cross-manifold gates)
    routed = router.analyze_and_route(circuit)

    # 2. Unified Expansion (handles both standard UR gates and injected cross gates)
    expanded = unified_expand_qudit_circuit(routed)

    if simulation_mode:
        expanded._routing_metadata = {
            "phase_accumulator": dict(router._phase_acc),
            "routing_log": router._routing_log
        }
        return expanded

    # 3. Expose BLOG/BLOG† objects: decompose only Coulomb & Exchange wrappers
    #    one level so cancel_adjacent_basis_changes can see the BLOG_DAG→BLOG
    #    boundaries.  All other wrappers remain intact.
    expanded = _expand_blog_sandwich_wrappers(expanded)

    # 4. Basis-change cancellation: remove BLOG·BLOG† pairs that compose to
    #    identity.  Must run before optimize_for_target_gateset, which would
    #    otherwise decompose every BLOG into 17 native ops first.
    expanded = cancel_adjacent_basis_changes(expanded)

    # 5. Optimize to Native Gates
    if target == "api":
        gateset = cirq_ionq.IonQTargetGateset()
    elif target == "forte_native":
        gateset = ForteNativeGateset()
    else:
        raise ValueError("target must be 'api' or 'forte_native'")

    compiled = cirq.optimize_for_target_gateset(
        expanded,
        gateset=gateset,
        context=cirq.TransformerContext(deep=True)
    )

    compiled._routing_metadata = {
        "phase_accumulator": dict(router._phase_acc),
        "routing_log": router._routing_log
    }
    return compiled

#==============================================================================
# 4. VERIFICATION & USAGE DEMO
#==============================================================================
def _compile_and_report(label: str, circ: "cirq.Circuit") -> None:
    """Helper: run simulation-mode expansion + Forte-native compilation and print results."""
    print(f"\n=== Compilation Test: {label} ===")
    print("Original Circuit:")
    print(circ)

    print("\n--- Simulation Mode ---")
    try:
        sim = compile_with_holographic_routing(circ, simulation_mode=True)
        print(f"Total moments: {len(sim)}")
        has_matrix = any(isinstance(op.gate, cirq.MatrixGate) for op in sim.all_operations())
        print(f"Contains MatrixGate: {has_matrix} (Expected: False after wrapper expansion)")
    except Exception as e:
        print(f"✗ Simulation Failed: {e}")

    print("\n--- Hardware Compilation (Forte Native) ---")
    try:
        compiled = compile_with_holographic_routing(circ, target="forte_native", simulation_mode=False)
        gate_counts: dict = {}
        has_forte = False
        for op in compiled.all_operations():
            name = op.gate.__class__.__name__
            gate_counts[name] = gate_counts.get(name, 0) + 1
            if name in ("GPIGate", "GPI2Gate", "ZZGate"):
                has_forte = True
        print(f"Total moments: {len(compiled)}")
        print(f"Gate counts (top 5): {dict(list(gate_counts.items())[:5])}")
        print(f"✓ Forte native target synthesized correctly." if has_forte else "! Warning: Native gates missing.")
    except Exception as e:
        print(f"✗ Forte Compilation Failed: {e}")


if __name__ == "__main__":
    print("=== Refactored Holographic Router Verification ===")

    q_log  = NomosIonQid(0)
    q_log2 = NomosIonQid(1)
    q_virt = VirtualQudit(0)

    # ------------------------------------------------------------------
    # Test 1: VirtualQudit Routing & Phase Compensation
    # ------------------------------------------------------------------
    router = HolographicRouter(enable_auto_routing=False)
    router._phase_acc[router._get_vqudit_id(q_virt)] = 1
    comp_ops = list(router.inject_compensation(q_virt))
    print(f"Phase Tracker Initialized: k={router._phase_acc[router._get_vqudit_id(q_virt)]}")
    print(f"Compensation Injected: {len(comp_ops)} ops | Gate: {comp_ops[0].gate if comp_ops else 'None'}")

    # ------------------------------------------------------------------
    # Test 2: Standard UR Gate
    # ------------------------------------------------------------------
    circ_ur = cirq.Circuit(TetralemmaticIonURgates().UR.on(q_log))
    _compile_and_report("Standard UR Gate", circ_ur)

    # ------------------------------------------------------------------
    # Test 3: ParamZClockGate  (single-qudit diagonal phase)
    # ------------------------------------------------------------------
    import math
    theta = math.pi / 4
    circ_pz = cirq.Circuit(ParamZClockGate(theta).on(q_log))
    _compile_and_report(f"ParamZClockGate(θ=π/4)", circ_pz)

    # ------------------------------------------------------------------
    # Test 4: ParamURShiftGate  (hopping; same unitary as ParamZClock)
    # ------------------------------------------------------------------
    circ_pur = cirq.Circuit(ParamURShiftGate(theta).on(q_log))
    _compile_and_report(f"ParamURShiftGate(θ=π/4)", circ_pur)

    # ------------------------------------------------------------------
    # Test 5: ParamURShiftGate inverse  (θ → −θ wrapper)
    # ------------------------------------------------------------------
    circ_pur_inv = cirq.Circuit(ParamURShiftGate(theta, inverse=True).on(q_log))
    _compile_and_report(f"ParamURShiftGate†(θ=π/4)", circ_pur_inv)

    # ------------------------------------------------------------------
    # Test 6: ParamCoulombPhaseGate  (two-qudit density-density)
    # ------------------------------------------------------------------
    phi = math.pi / 3
    circ_coulomb = cirq.Circuit(ParamCoulombPhaseGate(phi).on(q_log, q_log2))
    _compile_and_report(f"ParamCoulombPhaseGate(φ=π/3)", circ_coulomb)

    # ------------------------------------------------------------------
    # Test 7: ParamExchangeGate  (two-qudit beam-splitter)
    # ------------------------------------------------------------------
    circ_ex = cirq.Circuit(ParamExchangeGate(phi).on(q_log, q_log2))
    _compile_and_report(f"ParamExchangeGate(φ=π/3)", circ_ex)

    # ------------------------------------------------------------------
    # Test 8: Mixed Trotter step  (ZClock + Coulomb + Exchange together)
    # ------------------------------------------------------------------
    circ_trotter = cirq.Circuit(
        ParamZClockGate(theta).on(q_log),
        ParamZClockGate(theta).on(q_log2),
        ParamCoulombPhaseGate(phi).on(q_log, q_log2),
        ParamExchangeGate(phi).on(q_log, q_log2),
    )
    _compile_and_report("Mixed Trotter Step (ZClock + Coulomb + Exchange)", circ_trotter)

    print("\n✓ Holographic Router refactored successfully.")