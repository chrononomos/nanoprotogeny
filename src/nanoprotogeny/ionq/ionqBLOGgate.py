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
ionqBLOGgate.py — Fundamental Logical Basis-Change Gate for IonQ Forte
=======================================================================
Implements BLOGGate and BLOGDagGate: the canonical 2-qubit gates that
translate between the IonQ Forte qubit computational basis and the
Bell-separable physical encoding of the logical manifold ℋ_L.

Physical correspondence (B_LOG columns → Bell states):
    |Th⟩     → |00⟩          vacuum; product state
    |AntiTh⟩ → |11⟩          spin-up; product state
    |SynTh⟩  → |Ψ⁺⟩          spin-down; maximally entangled
    |HoloTh⟩ → |Ψ⁻⟩          doubly occupied; maximally entangled

This encoding is the algebraic heart of the Tetralemmatic Fock
isomorphism: definite-occupation-number states (vacuum, doubly
occupied) map to separable product states; superposition-occupation
states (spin-up, spin-down) map to Bell-entangled states. The
fermionic antisymmetry is carried by the entanglement geometry rather
than by explicit Jordan–Wigner parity strings.

Architecture role
-----------------
Every gate in the logical manifold is a sandwich:

    U_phys = BLOGDagGate · U_onto · BLOGGate

where U_onto is the abstract tetralemmatic operation expressed in the
four-level computational basis {|0⟩,|1⟩,|2⟩,|3⟩} = {|00⟩,|01⟩,|10⟩,|11⟩}.

BLOGGate and BLOGDagGate are therefore the ONLY hardware-specific
primitives for the logical manifold. Swapping them to a different
platform (superconducting, photonic, neutral atoms) automatically
re-targets the entire gate library without changing any U_onto logic.

Hardware decomposition
----------------------
Both gates are stored as precomputed GPI/GPI2/ZZ sequences from
ionqurgate.{B_LOG_OPS, B_LOG_DAG_OPS}.  These are direct Forte-native
sequences bypassing the CZ intermediate, reducing per-call cost from
54 → 17 (BLOGGate) and 48 → 20 (BLOGDagGate) native ops.

KAK structure: interaction coefficients (π/4, π/8, 0) → exactly 2
native ZZ(0.25) gates each, which is the theoretical minimum for this
unitary.

Dependencies: cirq, nanoprotogeny.ionq.ionqurgate (precomputed seqs).
No simulate-layer imports.
"""

from __future__ import annotations

import numpy as np
import cirq
from typing import Iterator, Tuple

from nanoprotogeny.ionq.ionqtetralemmatics import (
    B_LOG,
    B_LOG_OPS, B_LOG_DAG_OPS,
    apply_basis_ops,
)


# ==============================================================================
# GATE CLASSES
# ==============================================================================

class BLOGGate(cirq.Gate):
    r"""The B_LOG basis-change gate.

    Maps the four tetralemmatic computational-basis states to their
    Bell-separable physical encodings on two IonQ Forte qubits:

        |Th⟩     ↦ |00⟩
        |AntiTh⟩ ↦ |11⟩
        |SynTh⟩  ↦ |Ψ⁺⟩ = (|01⟩+|10⟩)/√2
        |HoloTh⟩ ↦ |Ψ⁻⟩ = (|01⟩-|10⟩)/√2

    Applied *after* a U_onto operation to rotate back to the physical
    Bell basis.  Its inverse BLOGDagGate rotates *into* the
    computational frame before U_onto acts.

    Decomposition: 2 ZZ(0.25) + 15 GPI/GPI2 = 17 native ops total.
    Reconstruction error < 1e-15.
    """

    def _num_qubits_(self) -> int: return 2
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2)
    def _has_unitary_(self) -> bool: return True

    def _unitary_(self) -> np.ndarray:
        return B_LOG.copy()

    def _decompose_(self, qubits) -> Iterator[cirq.OP_TREE]:
        yield from apply_basis_ops(B_LOG_OPS, *qubits)

    def __pow__(self, exponent):
        if exponent == -1: return BLOGDagGate()
        return NotImplemented

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=("B_LOG", "B_LOG"))

    def __repr__(self) -> str: return "BLOGGate()"
    def __eq__(self, other) -> bool: return isinstance(other, BLOGGate)
    def __hash__(self) -> int: return hash(type(self))


class BLOGDagGate(cirq.Gate):
    r"""The B_LOG† basis-change gate (inverse of BLOGGate).

    Rotates from the physical Bell-separable encoding back to the
    four-level ontological computational basis, making U_onto act
    diagonally in the standard {|0⟩,|1⟩,|2⟩,|3⟩} frame.

    Applied *before* a U_onto operation:
        BLOGDagGate · U_onto · BLOGGate = B_LOG† U_onto B_LOG = U_phys

    Decomposition: 2 ZZ(0.25) + 18 GPI/GPI2 = 20 native ops total.
    Reconstruction error < 1e-15.
    """

    def _num_qubits_(self) -> int: return 2
    def _qid_shape_(self) -> Tuple[int, ...]: return (2, 2)
    def _has_unitary_(self) -> bool: return True

    def _unitary_(self) -> np.ndarray:
        return B_LOG.conj().T.copy()

    def _decompose_(self, qubits) -> Iterator[cirq.OP_TREE]:
        yield from apply_basis_ops(B_LOG_DAG_OPS, *qubits)

    def __pow__(self, exponent):
        if exponent == -1: return BLOGGate()
        return NotImplemented

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=("B_LOG†", "B_LOG†"))

    def __repr__(self) -> str: return "BLOGDagGate()"
    def __eq__(self, other) -> bool: return isinstance(other, BLOGDagGate)
    def __hash__(self) -> int: return hash(type(self))


# ==============================================================================
# CANONICAL SINGLETONS
# ==============================================================================

BLOG     = BLOGGate()
BLOG_DAG = BLOGDagGate()


# ==============================================================================
# COMPOSITION HELPER
# ==============================================================================

def wrap_logical(onto_gate: cirq.Gate, qubits: Tuple[cirq.Qid, ...]) -> Iterator[cirq.OP_TREE]:
    r"""Yield BLOGDag · onto_gate · BLOG on the given qubit pair.

    Constructs the physical gate U_phys = B_LOG† · U_onto · B_LOG for
    any 2-qubit onto_gate expressed in the tetralemmatic computational
    basis.  The caller is responsible for ensuring len(qubits) == 2.

    Example::

        yield from wrap_logical(ParamZClockOntoGate(theta), (q0, q1))
    """
    q0, q1 = qubits
    yield BLOG_DAG.on(q0, q1)
    yield onto_gate.on(q0, q1)
    yield BLOG.on(q0, q1)


# ==============================================================================
# INTERNAL VERIFICATION
# ==============================================================================

if __name__ == "__main__":
    import math

    print("=== ionqBLOGgate.py Internal Verification ===\n")

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
    U = BLOG._unitary_()
    check("BLOGGate is unitary", np.allclose(U.conj().T @ U, I4))
    Ud = BLOG_DAG._unitary_()
    check("BLOGDagGate is unitary", np.allclose(Ud.conj().T @ Ud, I4))

    # 2. Inverse relationship
    check("BLOG · BLOG† = I", np.allclose(U @ Ud, I4))
    check("BLOG† · BLOG = I", np.allclose(Ud @ U, I4))

    # 3. __pow__ inverse
    check("BLOGGate()**-1 == BLOGDagGate()", BLOG**-1 == BLOG_DAG)
    check("BLOGDagGate()**-1 == BLOGGate()", BLOG_DAG**-1 == BLOG)

    # 4. Physical encoding: pole states → product states
    Th_phys     = U[:, 0]  # B_LOG|Th⟩ = col 0
    AntiTh_phys = U[:, 1]  # B_LOG|AntiTh⟩ = col 1
    SynTh_phys  = U[:, 2]
    HoloTh_phys = U[:, 3]

    check("|Th⟩ → |00⟩  (product)",
          np.allclose(Th_phys, [1, 0, 0, 0]))
    check("|AntiTh⟩ → |11⟩  (product)",
          np.allclose(AntiTh_phys, [0, 0, 0, 1]))
    check("|SynTh⟩ → |Ψ+⟩  (entangled)",
          np.allclose(SynTh_phys, [0, 1/math.sqrt(2), 1/math.sqrt(2), 0]))
    check("|HoloTh⟩ → |Ψ-⟩  (entangled)",
          np.allclose(HoloTh_phys, [0, 1/math.sqrt(2), -1/math.sqrt(2), 0]))

    # 5. Decompose round-trip
    circ_fwd = cirq.Circuit(BLOG.on(q0, q1))
    circ_dag = cirq.Circuit(BLOG_DAG.on(q0, q1))
    U_fwd = cirq.unitary(circ_fwd)
    U_dag = cirq.unitary(circ_dag)
    idx = int(np.argmax(np.abs(B_LOG.flat)))
    phase_fwd = U_fwd.flat[idx] / B_LOG.flat[idx]
    phase_dag = U_dag.flat[idx] / B_LOG.conj().T.flat[idx]
    check("BLOGGate decomp error < 1e-12",
          np.max(np.abs(U_fwd - phase_fwd * B_LOG)) < 1e-12)
    check("BLOGDagGate decomp error < 1e-12",
          np.max(np.abs(U_dag - phase_dag * B_LOG.conj().T)) < 1e-12)

    # 6. Native gate count
    try:
        from cirq_ionq.ionq_native_target_gateset import ForteNativeGateset
        forte = ForteNativeGateset()
        for name, gate in [("BLOGGate", BLOG), ("BLOGDagGate", BLOG_DAG)]:
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
    "BLOGGate",
    "BLOGDagGate",
    "BLOG",
    "BLOG_DAG",
    "wrap_logical",
]
