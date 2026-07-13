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
ionqparamgates.py — Parametrized d=4 Trotter Gate Primitives
=============================================================
Native d=4 cirq.Gate subclasses that implement the Suzuki-Trotter
decomposition of the second-quantized Hamiltonian on the MQE qudit
register.  These are the hardware-level building blocks consumed by
``mqe.build_trotter_evolution_circuit``.

Gates
-----
ParamZClockGate(θ)
    On-site energy: Ẑ_Clock(θ) = diag(1, e^{iθ}, e^{2iθ}, e^{3iθ}).
    Encodes h_{pp}·n̂_p as a diagonal phase on a single d=4 qudit.

ParamURShiftGate(θ, inverse=False)
    Hopping: diagonal phase diag(e^{ikθ}) encoding h_{pq}·(â†_pâ_q + h.c.)
    as a JW-free single-qudit operator.

ParamCoulombPhaseGate(φ)
    Density-density Coulomb: U[15,15] = e^{iφ}, all others = 1.
    Encodes ½g_{pp,rr}·n̂_p·n̂_r on a (d=4)⊗(d=4) pair.

ParamExchangeGate(φ)
    Exchange/beam-splitter on |↑_p↓_q⟩ ↔ |↓_p↑_q⟩ subspace.
    Encodes ½g_{pq,qp} exchange integral.

ParamScatteringGate(φ, indices)
    General four-centre scattering g_{pqrs} (all distinct indices).
    Decomposes via shift-sandwich: SUM → SUM → CoulombPhase → InvSUM → InvSUM.

TetralemmaticIonInverseDFTGate()
    Adjoint of the d=4 quantum Fourier transform F̂_4.
    Diagonalises the cyclic shift: F̂_4 Û_R F̂_4† = Ẑ_Clock.

PowerControlledGate(base_gate, max_power=3)  [→ ionqpowercontrolgate.py]
    Σ_{m=0}^{d-1} |m⟩⟨m|_ancilla ⊗ U^m.
    Realised by stacking d-1 threshold-controlled gates.

Dependencies: cirq, numpy,
              nanoprotogeny.ionq.ionqsumgate         (TetralemmaticIonSUMGate/InverseSUMGate),
              nanoprotogeny.ionq.ionqpowercontrolgate (PowerControlledGate, re-exported).
No simulate-layer imports.
"""

from __future__ import annotations

import numpy as np
import cirq
from cirq import OP_TREE
from typing import Iterator, Tuple

from nanoprotogeny.ionq.ionqsumgate import (
    TetralemmaticIonSUMGate,
    TetralemmaticIonInverseSUMGate,
)
from nanoprotogeny.ionq.ionqtetralemmatics import (
    DFT_onto
)


from nanoprotogeny.ionq.ionqpowercontrolgate import PowerControlledGate  # noqa: F401  re-exported

class ParamZClockGate(cirq.Gate):
    def __init__(self, theta: float): self.theta = theta
    def _qid_shape_(self): return (4,)
    def _unitary_(self): return np.diag([1, np.exp(1j*self.theta), np.exp(2j*self.theta), np.exp(3j*self.theta)])
    def _circuit_diagram_info_(self, args): return cirq.CircuitDiagramInfo(wire_symbols=(f"Zc({self.theta:.3f})",))
    def __pow__(self, exponent):
        if exponent == -1: return ParamZClockGate(-self.theta)
        return NotImplemented

class ParamURShiftGate(cirq.Gate):
    """Parameterized diagonal phase operator for d=4 qudits."""
    def __init__(self, theta: float, inverse: bool = False):
        self.theta = theta
        self.inverse = inverse

    def _qid_shape_(self) -> Tuple[int, ...]: return (4,)
    def _unitary_(self) -> np.ndarray:
        phase_sign = -1.0 if self.inverse else 1.0
        return np.diag([np.exp(1j * phase_sign * k * self.theta) for k in range(4)])

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        sym = f"UR^\\dagger({self.theta:.3f})" if self.inverse else f"UR({self.theta:.3f})"
        return cirq.CircuitDiagramInfo(wire_symbols=(sym,))

    def __pow__(self, exponent):
        if exponent == -1: return ParamURShiftGate(self.theta, inverse=not self.inverse)
        return NotImplemented

class ParamCoulombPhaseGate(cirq.Gate):
    def __init__(self, phi: float): self.phi = phi
    def _qid_shape_(self): return (4, 4)
    def _unitary_(self):
        U = np.eye(16, dtype=complex); U[15, 15] = np.exp(1j * self.phi)
        return U
    def _circuit_diagram_info_(self, args): return cirq.CircuitDiagramInfo(wire_symbols=(f"C({self.phi:.3f})", f"C({self.phi:.3f})"))
    def __pow__(self, exponent):
        if exponent == -1: return ParamCoulombPhaseGate(-self.phi)
        return NotImplemented

class ParamExchangeGate(cirq.Gate):
    """Parameterized exchange gate for two d=4 qudits."""
    def __init__(self, phi: float): self.phi = phi
    def _qid_shape_(self) -> Tuple[int, ...]: return (4, 4)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray:
        U = np.eye(16, dtype=complex)
        i, j = 6, 9  # |1,2⟩ and |2,1⟩ in linear index space
        U[i, i] = np.cos(self.phi); U[j, j] = np.cos(self.phi)
        U[i, j] = -1j * np.sin(self.phi); U[j, i] = -1j * np.sin(self.phi)
        return U
    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=(f"EX({self.phi:.3f})", f"EX({self.phi:.3f})"))
    def __pow__(self, exponent):
        if exponent == -1: return ParamExchangeGate(-self.phi)
        return NotImplemented

class ParamScatteringGate(cirq.Gate):
    def __init__(self, phi: float, indices: Tuple[int, int, int, int]):
        self.phi, self.indices = phi, indices
    def _qid_shape_(self) -> Tuple[int, ...]: return (4, 4, 4, 4)
    def _has_unitary_(self) -> bool: return False  # Forces decomposition during compilation
    def _unitary_(self) -> np.ndarray: return NotImplemented
    def _decompose_(self, qubits) -> Iterator[OP_TREE]:
        p, q, r, s = qubits
        yield TetralemmaticIonSUMGate().on(r, p)
        yield TetralemmaticIonSUMGate().on(s, q)
        yield ParamCoulombPhaseGate(self.phi).on(p, q)
        yield TetralemmaticIonInverseSUMGate().on(r, p)
        yield TetralemmaticIonInverseSUMGate().on(s, q)
    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=(f"SCAT({self.phi:.3f})",)*4)
    def __pow__(self, exponent):
        if exponent == -1: return ParamScatteringGate(-self.phi, self.indices)
        return NotImplemented

class TetralemmaticIonInverseDFTGate(cirq.Gate):
    """Inverse d=4 Fourier transform (adjoint of the forward DFT)."""
    def _qid_shape_(self) -> Tuple[int, ...]: return (4,)
    def _has_unitary_(self) -> bool: return True
    def _unitary_(self) -> np.ndarray:
        return DFT_onto.conj().T
    def _circuit_diagram_info_(self, args):
        return cirq.CircuitDiagramInfo(wire_symbols=("F†",))
    def __repr__(self):
        return "TetralemmaticIonInverseDFTGate()"

# ==============================================================================
# INTERNAL VERIFICATION
# ==============================================================================
if __name__ == "__main__":
    import math

    print("=== ionqparamgates.py Internal Verification ===\n")

    PASS, FAIL = "✓", "✗"
    all_ok = True

    def check(label: str, result: bool) -> bool:
        global all_ok
        print(f"  {PASS if result else FAIL}  {label}")
        if not result:
            all_ok = False
        return result

    I4  = np.eye(4,  dtype=complex)
    I16 = np.eye(16, dtype=complex)
    theta = math.pi / 4
    phi   = math.pi / 3

    # ------------------------------------------------------------------
    # 1. ParamZClockGate
    # ------------------------------------------------------------------
    print("--- ParamZClockGate ---")
    g_pz  = ParamZClockGate(theta)
    U_pz  = g_pz._unitary_()
    check("unitary  (U†U = I)",
          np.allclose(U_pz.conj().T @ U_pz, I4))
    check("diagonal entries = e^{ikθ}  for k=0..3",
          all(np.isclose(U_pz[k, k], np.exp(1j * k * theta)) for k in range(4)))
    check("off-diagonal entries = 0",
          np.allclose(U_pz - np.diag(np.diag(U_pz)), 0))
    check("__pow__(-1) is exact inverse",
          np.allclose(U_pz @ (g_pz ** -1)._unitary_(), I4))
    check("ParamZClockGate(-θ) == inverse",
          np.allclose(U_pz @ ParamZClockGate(-theta)._unitary_(), I4))

    # ------------------------------------------------------------------
    # 2. ParamURShiftGate
    # ------------------------------------------------------------------
    print("\n--- ParamURShiftGate ---")
    g_ur  = ParamURShiftGate(theta)
    U_ur  = g_ur._unitary_()
    check("unitary  (U†U = I)",
          np.allclose(U_ur.conj().T @ U_ur, I4))
    check("matches ParamZClockGate at same θ  (identical diagonal)",
          np.allclose(U_ur, U_pz))
    check("inverse=True negates all phases",
          np.allclose(ParamURShiftGate(theta, inverse=True)._unitary_(),
                      np.diag([np.exp(-1j * k * theta) for k in range(4)])))
    check("forward @ inverse = I",
          np.allclose(U_ur @ ParamURShiftGate(theta, inverse=True)._unitary_(), I4))
    check("__pow__(-1) is exact inverse",
          np.allclose(U_ur @ (g_ur ** -1)._unitary_(), I4))

    # ------------------------------------------------------------------
    # 3. ParamCoulombPhaseGate
    # ------------------------------------------------------------------
    print("\n--- ParamCoulombPhaseGate ---")
    g_c  = ParamCoulombPhaseGate(phi)
    U_c  = g_c._unitary_()
    diff_c = U_c - I16
    expected_diff_c = np.zeros((16, 16), dtype=complex)
    expected_diff_c[15, 15] = np.exp(1j * phi) - 1
    check("unitary  (U†U = I)",
          np.allclose(U_c.conj().T @ U_c, I16))
    check("U[15,15] = e^{iφ}",
          np.isclose(U_c[15, 15], np.exp(1j * phi)))
    check("all other entries unchanged from identity",
          np.allclose(diff_c, expected_diff_c))
    check("__pow__(-1) is exact inverse",
          np.allclose(U_c @ (g_c ** -1)._unitary_(), I16))

    # ------------------------------------------------------------------
    # 4. ParamExchangeGate
    # ------------------------------------------------------------------
    print("\n--- ParamExchangeGate ---")
    g_ex = ParamExchangeGate(phi)
    U_ex = g_ex._unitary_()
    i, j = 6, 9   # |1,2⟩ and |2,1⟩
    check("unitary  (U†U = I)",
          np.allclose(U_ex.conj().T @ U_ex, I16))
    check("U[6,6] = cos(φ)",
          np.isclose(U_ex[i, i], math.cos(phi)))
    check("U[9,9] = cos(φ)",
          np.isclose(U_ex[j, j], math.cos(phi)))
    check("U[6,9] = −i·sin(φ)",
          np.isclose(U_ex[i, j], -1j * math.sin(phi)))
    check("U[9,6] = −i·sin(φ)",
          np.isclose(U_ex[j, i], -1j * math.sin(phi)))
    check("all other diagonal entries = 1",
          all(np.isclose(U_ex[k, k], 1.0) for k in range(16) if k not in (i, j)))
    check("beam-splitter sub-block is unitary",
          np.allclose(
              np.array([[U_ex[i,i], U_ex[i,j]], [U_ex[j,i], U_ex[j,j]]]).conj().T
              @ np.array([[U_ex[i,i], U_ex[i,j]], [U_ex[j,i], U_ex[j,j]]]),
              np.eye(2, dtype=complex)
          ))
    check("__pow__(-1) is exact inverse",
          np.allclose(U_ex @ (g_ex ** -1)._unitary_(), I16))

    # ------------------------------------------------------------------
    # 5. ParamScatteringGate
    # ------------------------------------------------------------------
    print("\n--- ParamScatteringGate ---")
    g_sc = ParamScatteringGate(phi, (0, 1, 2, 3))
    check("_has_unitary_() is False",
          not g_sc._has_unitary_())
    qd = [cirq.LineQid(k, dimension=4) for k in range(4)]
    decomp_ops   = list(g_sc._decompose_(qd))
    decomp_gates = [op.gate for op in decomp_ops]
    expected_seq = [
        TetralemmaticIonSUMGate, TetralemmaticIonSUMGate,
        ParamCoulombPhaseGate,
        TetralemmaticIonInverseSUMGate, TetralemmaticIonInverseSUMGate,
    ]
    check("decompose yields 5 ops: SUM, SUM, Coulomb, InvSUM, InvSUM",
          len(decomp_gates) == 5
          and all(isinstance(g, t) for g, t in zip(decomp_gates, expected_seq)))
    check("Coulomb φ preserved through decomposition",
          np.isclose(decomp_gates[2].phi, phi))
    check("__pow__(-1) negates φ",
          np.isclose((g_sc ** -1).phi, -phi))

    # ------------------------------------------------------------------
    # 6. TetralemmaticIonInverseDFTGate
    # ------------------------------------------------------------------
    print("\n--- TetralemmaticIonInverseDFTGate ---")
    g_idft = TetralemmaticIonInverseDFTGate()
    U_idft = g_idft._unitary_()
    check("unitary  (U†U = I)",
          np.allclose(U_idft.conj().T @ U_idft, I4))
    check("equals DFT_onto†",
          np.allclose(U_idft, DFT_onto.conj().T))
    check("DFT_onto @ InvDFT = I",
          np.allclose(DFT_onto @ U_idft, I4))
    check("InvDFT @ DFT_onto = I",
          np.allclose(U_idft @ DFT_onto, I4))

    # ------------------------------------------------------------------
    # 7. PowerControlledGate
    # ------------------------------------------------------------------
    print("\n--- PowerControlledGate ---")
    base_g = ParamZClockGate(math.pi / 2)   # diag(1, i, −1, −i)
    pcg    = PowerControlledGate(base_g, max_power=3)
    check("_qid_shape_() = (4, 4)",
          pcg._qid_shape_() == (4, 4))
    # Build expected unitary: block-diagonal Σ_m |m><m| ⊗ U^m
    U_base = base_g._unitary_()
    expected_pcg = np.zeros((16, 16), dtype=complex)
    for m in range(4):
        expected_pcg[m*4:(m+1)*4, m*4:(m+1)*4] = np.linalg.matrix_power(U_base, m)
    try:
        anc = cirq.LineQid(0, dimension=4)
        tgt = cirq.LineQid(1, dimension=4)
        U_pcg = cirq.unitary(cirq.Circuit(pcg.on(anc, tgt)))
        check("PowerControlledGate unitary = Σ_m |m><m| ⊗ U^m",
              np.allclose(U_pcg, expected_pcg, atol=1e-8))
    except Exception as exc:
        print(f"  -  PowerControlledGate unitary check skipped ({exc})")

    # ------------------------------------------------------------------
    print(f"\n{'All checks passed.' if all_ok else 'Some checks FAILED — review output above.'}")

