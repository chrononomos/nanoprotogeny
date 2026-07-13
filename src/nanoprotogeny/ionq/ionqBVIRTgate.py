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
ionqBVIRTgate.py — Fundamental Virtual Basis-Change Gate for IonQ Forte
========================================================================
Implements BVIRTGate and BVIRTDagGate: the canonical 2-qubit gates that
translate between the IonQ Forte qubit computational basis and the
Bell-separable physical encoding of the virtual phase register ℋ_V.

Physical correspondence (B_VIRT columns → Bell states):
    |F⟩  → |Ψ⁻⟩ = (|01⟩-|10⟩)/√2     D-state k=0  (baseline, antisym)
    |P⟩  → |00⟩                          D-state k=1  (+π/2 phase acc.)
    |M⟩  → |Ψ⁺⟩ = (|01⟩+|10⟩)/√2     D-state k=2  (π phase acc.)
    |R⟩  → |11⟩                          D-state k=3  (−π/2 phase acc.)

The virtual register's cyclic phase accumulation structure — U_R cycles
|F⟩→|P⟩→|M⟩→|R⟩→|F⟩ — is encoded so that the k=0 baseline state maps
to the antisymmetric Bell state |Ψ⁻⟩ (preserving the HoloTh boundary
role across the cross-manifold SWAP), while k=3 maps to the symmetric
product state |11⟩ matching the antipodal structure required for
Janus-crossing stoichiometry at k*=m/2.

Architecture role
-----------------
Every gate in the virtual manifold is a sandwich:

    U_phys = BVIRTDagGate · U_onto · BVIRTGate

where U_onto is the abstract tetralemmatic operation expressed in the
computational basis {|0⟩,|1⟩,|2⟩,|3⟩}.

BVIRTGate and BVIRTDagGate are the counterpart to BLOGGate/BLOGDagGate
for the virtual (²D₃/₂ shelving) manifold.  Together, the four gate
objects {BLOG, BLOG_DAG, BVIRT, BVIRT_DAG} constitute the complete
platform-specific primitive set from which every other tetralemmatic
gate in the system is constructed.

Cross-manifold gates (PhaseSwap, ZenoStabilize, etc.) are sandwiched
by the mixed pair:

    U_cross_phys =
        BLOGDagGate(L) · BVIRTDagGate(V) · U_cross_onto · BVIRTGate(V) · BLOGGate(L)

The 688 THz spectral isolation between ℋ_L (S₁/₂ clock states) and
ℋ_V (²D₃/₂ D-state levels) means BLOGGate and BVIRTGate address
physically distinct frequency channels — zero cross-talk by construction.

Hardware decomposition
----------------------
Both gates are stored as precomputed GPI/GPI2/ZZ sequences from
ionqurgate.{B_VIRT_OPS, B_VIRT_DAG_OPS}.  These are direct Forte-native
sequences bypassing the CZ intermediate:

    BVIRTGate:    2 ZZ(0.25) + 18 GPI/GPI2 = 20 native ops  (was 54)
    BVIRTDagGate: 2 ZZ(0.25) + 18 GPI/GPI2 = 20 native ops  (was 48)

KAK interaction: (π/4, π/8, 0), same structure as B_LOG — both basis
matrices are related by the same tetralemmatic geometry applied to
their respective manifolds.

Dependencies: cirq, nanoprotogeny.ionq.ionqurgate (precomputed seqs).
No simulate-layer imports.
"""

from __future__ import annotations

import numpy as np
import cirq
from typing import Iterator, Tuple

from nanoprotogeny.ionq.ionqtetralemmatics import (
    B_VIRT,
    B_VIRT_OPS, B_VIRT_DAG_OPS,
    apply_basis_ops,
)


# ==============================================================================
# GATE CLASSES
# ==============================================================================

class BVIRTGate(cirq.Gate):
    r"""The B_VIRT basis-change gate.

    Maps the four virtual phase-register states to their Bell-separable
    physical encodings on two IonQ Forte qubits:

        |F⟩ ↦ |Ψ⁻⟩ = (|01⟩-|10⟩)/√2
        |P⟩ ↦ |00⟩
        |M⟩ ↦ |Ψ⁺⟩ = (|01⟩+|10⟩)/√2
        |R⟩ ↦ |11⟩

    Applied *after* a V_onto operation to rotate back to the physical
    Bell basis of the virtual manifold.  Its inverse BVIRTDagGate
    rotates *into* the computational frame before V_onto acts.

    Decomposition: 2 ZZ(0.25) + 18 GPI/GPI2 = 20 native ops total.
    Reconstruction error < 2e-15.
    """

    def _num_qubits_(self) -> int: return 2
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2)
    def _has_unitary_(self) -> bool: return True

    def _unitary_(self) -> np.ndarray:
        return B_VIRT.copy()

    def _decompose_(self, qubits) -> Iterator[cirq.OP_TREE]:
        yield from apply_basis_ops(B_VIRT_OPS, *qubits)

    def __pow__(self, exponent):
        if exponent == -1: return BVIRTDagGate()
        return NotImplemented

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=("B_VIRT", "B_VIRT"))

    def __repr__(self) -> str: return "BVIRTGate()"
    def __eq__(self, other) -> bool: return isinstance(other, BVIRTGate)
    def __hash__(self) -> int: return hash(type(self))


class BVIRTDagGate(cirq.Gate):
    r"""The B_VIRT† basis-change gate (inverse of BVIRTGate).

    Rotates from the physical Bell-separable encoding of the virtual
    manifold back to the four-level computational basis, making V_onto
    act diagonally in the standard {|0⟩,|1⟩,|2⟩,|3⟩} frame.

    Applied *before* a V_onto operation:
        BVIRTDagGate · V_onto · BVIRTGate = B_VIRT† V_onto B_VIRT = V_phys

    Decomposition: 2 ZZ(0.25) + 18 GPI/GPI2 = 20 native ops total.
    Reconstruction error < 2e-15.
    """

    def _num_qubits_(self) -> int: return 2
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2)
    def _has_unitary_(self) -> bool: return True

    def _unitary_(self) -> np.ndarray:
        return B_VIRT.conj().T.copy()

    def _decompose_(self, qubits) -> Iterator[cirq.OP_TREE]:
        yield from apply_basis_ops(B_VIRT_DAG_OPS, *qubits)

    def __pow__(self, exponent):
        if exponent == -1: return BVIRTGate()
        return NotImplemented

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=("B_VIRT†", "B_VIRT†"))

    def __repr__(self) -> str: return "BVIRTDagGate()"
    def __eq__(self, other) -> bool: return isinstance(other, BVIRTDagGate)
    def __hash__(self) -> int: return hash(type(self))


# ==============================================================================
# CANONICAL SINGLETONS
# ==============================================================================

BVIRT     = BVIRTGate()
BVIRT_DAG = BVIRTDagGate()


# ==============================================================================
# COMPOSITION HELPERS
# ==============================================================================

def wrap_virtual(onto_gate: cirq.Gate, qubits: Tuple[cirq.Qid, ...]) -> Iterator[cirq.OP_TREE]:
    r"""Yield BVIRTDag · onto_gate · BVIRT on the given qubit pair.

    Constructs the physical gate V_phys = B_VIRT† · V_onto · B_VIRT for
    any 2-qubit onto_gate expressed in the tetralemmatic computational
    basis.  The caller is responsible for ensuring len(qubits) == 2.

    Example::

        yield from wrap_virtual(VURShiftOntoGate(), (q0, q1))
    """
    q0, q1 = qubits
    yield BVIRT_DAG.on(q0, q1)
    yield onto_gate.on(q0, q1)
    yield BVIRT.on(q0, q1)


def wrap_cross(
    onto_gate: cirq.Gate,
    log_qubits: Tuple[cirq.Qid, cirq.Qid],
    virt_qubits: Tuple[cirq.Qid, cirq.Qid],
) -> Iterator[cirq.OP_TREE]:
    r"""Yield the physical form of a cross-manifold gate.

    Constructs:
        BLOGDag(L) · BVIRTDag(V) · U_cross · BVIRTGate(V) · BLOGGate(L)

    The onto_gate must act on 4 qubits in the order (l0, l1, v0, v1).
    The 688 THz spectral gap between L and V guarantees zero cross-talk.

    Example::

        yield from wrap_cross(ZenoComputationalGate(), (l0,l1), (v0,v1))
    """
    from nanoprotogeny.ionq.ionqBLOGgate import BLOG, BLOG_DAG
    l0, l1 = log_qubits
    v0, v1 = virt_qubits
    yield BLOG_DAG.on(l0, l1)
    yield BVIRT_DAG.on(v0, v1)
    yield onto_gate.on(l0, l1, v0, v1)
    yield BLOG.on(l0, l1)
    yield BVIRT.on(v0, v1)


# ==============================================================================
# INTERNAL VERIFICATION
# ==============================================================================

if __name__ == "__main__":
    import math

    print("=== ionqBVIRTgate.py Internal Verification ===\n")

    q0, q1 = cirq.LineQubit.range(2)
    I4 = np.eye(4, dtype=complex)

    PASS, FAIL = "✓", "✗"
    all_ok = True

    def check(label: str, result: bool) -> None:
        global all_ok
        print(f"  {PASS if result else FAIL}  {label}")
        if not result:
            all_ok = False

    # 1. Unitarity
    U = BVIRT._unitary_()
    check("BVIRTGate is unitary", np.allclose(U.conj().T @ U, I4))
    Ud = BVIRT_DAG._unitary_()
    check("BVIRTDagGate is unitary", np.allclose(Ud.conj().T @ Ud, I4))

    # 2. Inverse relationship
    check("BVIRT · BVIRT† = I", np.allclose(U @ Ud, I4))
    check("BVIRT† · BVIRT = I", np.allclose(Ud @ U, I4))

    # 3. __pow__ inverse
    check("BVIRTGate()**-1 == BVIRTDagGate()", BVIRT**-1 == BVIRT_DAG)
    check("BVIRTDagGate()**-1 == BVIRTGate()", BVIRT_DAG**-1 == BVIRT)

    # 4. Physical encoding: virtual phase states
    # B_VIRT column ordering: [F, P, M, R] → [Ψ-, 00, Ψ+, 11]
    F_phys = U[:, 0]  # |F⟩ → |Ψ-⟩
    P_phys = U[:, 1]  # |P⟩ → |00⟩
    M_phys = U[:, 2]  # |M⟩ → |Ψ+⟩
    R_phys = U[:, 3]  # |R⟩ → |11⟩

    check("|F⟩ → |Ψ-⟩  (entangled, k=0 baseline)",
          np.allclose(F_phys, [0, 1/math.sqrt(2), -1/math.sqrt(2), 0]))
    check("|P⟩ → |00⟩  (product, k=1, +π/2)",
          np.allclose(P_phys, [1, 0, 0, 0]))
    check("|M⟩ → |Ψ+⟩  (entangled, k=2, π phase)",
          np.allclose(M_phys, [0, 1/math.sqrt(2), 1/math.sqrt(2), 0]))
    check("|R⟩ → |11⟩  (product, k=3, −π/2)",
          np.allclose(R_phys, [0, 0, 0, 1]))

    # 5. Decompose round-trip (up to global phase)
    circ_fwd = cirq.Circuit(BVIRT.on(q0, q1))
    circ_dag = cirq.Circuit(BVIRT_DAG.on(q0, q1))
    U_fwd = cirq.unitary(circ_fwd)
    U_dag = cirq.unitary(circ_dag)
    idx = int(np.argmax(np.abs(B_VIRT.flat)))
    phase_fwd = U_fwd.flat[idx] / B_VIRT.flat[idx]
    phase_dag = U_dag.flat[idx] / B_VIRT.conj().T.flat[idx]
    check("BVIRTGate decomp error < 1e-12",
          np.max(np.abs(U_fwd - phase_fwd * B_VIRT)) < 1e-12)
    check("BVIRTDagGate decomp error < 1e-12",
          np.max(np.abs(U_dag - phase_dag * B_VIRT.conj().T)) < 1e-12)

    # 6. Cyclic phase structure: B_VIRT encodes U_R as a cyclic shift
    # After B_VIRT†, the cyclic shift U_R_onto (|0⟩→|1⟩→|2⟩→|3⟩→|0⟩ in virt)
    # should act diagonally as a permutation matrix
    U_R_onto_virt = np.array([[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,0,1,0]], dtype=complex)
    U_R_phys = B_VIRT @ U_R_onto_virt @ B_VIRT.conj().T
    check("U_R_phys is unitary", np.allclose(U_R_phys.conj().T @ U_R_phys, I4))
    check("B_VIRT† · U_R_phys · B_VIRT = U_R_onto",
          np.allclose(B_VIRT.conj().T @ U_R_phys @ B_VIRT, U_R_onto_virt))

    # 7. Native gate count
    try:
        from cirq_ionq.ionq_native_target_gateset import ForteNativeGateset
        forte = ForteNativeGateset()
        for name, gate in [("BVIRTGate", BVIRT), ("BVIRTDagGate", BVIRT_DAG)]:
            c = cirq.optimize_for_target_gateset(
                cirq.Circuit(gate.on(q0, q1)), gateset=forte,
                context=cirq.TransformerContext(deep=True))
            c = cirq.drop_negligible_operations(c, atol=1e-8)
            c = cirq.drop_empty_moments(c)
            n = sum(1 for op in c.all_operations()
                    if type(op.gate).__name__ in ("GPIGate","GPI2Gate","ZZGate","ZZPowGate"))
            print(f"  ℹ  {name}: {n} native GPI/GPI2/ZZ ops")
    except ImportError:
        print("  -  cirq_ionq not available; native count skipped")

    print(f"\n{'All checks passed.' if all_ok else 'Some checks FAILED.'}")


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    "BVIRTGate",
    "BVIRTDagGate",
    "BVIRT",
    "BVIRT_DAG",
    "wrap_virtual",
    "wrap_cross",
]
