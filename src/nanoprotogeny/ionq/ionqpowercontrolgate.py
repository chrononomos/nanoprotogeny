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
ionqpowercontrolgate.py — Power-conditioned qudit control gate
==============================================================
Canonical home for ``PowerControlledGate``.  Lifted from
``ionqparamgates.py`` so that QPE and scheduler modules can import it
without pulling in the full Trotter-gate suite.

Gate
----
PowerControlledGate(base_gate, max_power=3)
    Σ_{m=0}^{d-1} |m⟩⟨m|_ancilla ⊗ U^m.
    Realised by stacking d-1 threshold-controlled gates:
        Π_{t=1}^{max_power} C_{ancilla ≥ t}(U)

Dependencies: cirq, typing.
No simulate-layer imports.
"""

from __future__ import annotations

import cirq
from cirq import OP_TREE
from typing import Iterator, Tuple


class PowerControlledGate(cirq.Gate):
    r"""
    Power-conditioned qudit control:  Σ_{m=0}^{d-1} |m⟩⟨m|_a ⊗ U^m.

    Realised by stacking d-1 threshold-controlled gates:
        Π_{t=1}^{max_power} C_{ancilla ≥ t}(U)

    For ancilla value m, C_{≥t}(U) fires iff m ≥ t, so U is applied
    exactly m times in total. Verified correct for d=4:
        |0⟩ → 0 firings → U^0 = I   ✓
        |1⟩ → 1 firing  → U^1       ✓
        |2⟩ → 2 firings → U^2       ✓
        |3⟩ → 3 firings → U^3       ✓
    """
    def __init__(self, base_gate: cirq.Gate, max_power: int = 3):
        self.base_gate = base_gate
        self.max_power = max_power

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (4,) + self.base_gate._qid_shape_()

    def _decompose_(self, qubits: Tuple[cirq.Qid, ...]) -> Iterator[OP_TREE]:
        ancilla, *targets = qubits
        for threshold in range(1, self.max_power + 1):
            ctrl_vals = list(range(threshold, 4))
            control_values = cirq.SumOfProducts([[v] for v in ctrl_vals])
            c_gate = cirq.ControlledGate(
                self.base_gate,
                num_controls=1,
                control_values=control_values,
                control_qid_shape=(4,),
            )
            yield c_gate.on(ancilla, *targets)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        base_sym = "G"
        if hasattr(self.base_gate, "_circuit_diagram_info_"):
            base_sym = self.base_gate._circuit_diagram_info_(args).wire_symbols[0]
        n_targets = len(self._qid_shape_()) - 1
        return cirq.CircuitDiagramInfo(
            wire_symbols=(f"C^≤{self.max_power}({base_sym})",) + ("@",) * n_targets
        )

    def __repr__(self) -> str:
        return f"PowerControlledGate(base_gate={self.base_gate!r}, max_power={self.max_power})"

    def __eq__(self, other) -> bool:
        return (isinstance(other, PowerControlledGate)
                and self.base_gate == other.base_gate
                and self.max_power == other.max_power)

    def __hash__(self) -> int:
        return hash((type(self), self.base_gate, self.max_power))
