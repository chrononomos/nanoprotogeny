r"""
Forte Enterprise 1 Noisy Quantum Simulator
Distributed cloud • simulator

Statevector simulator with hardware-derived depolarizing noise model for testing and optimization in a production-like environment.

Qubits
36

Simulated Hardware Noise Model
Forte Enterprise 1

Predicted Queue Time
0min

Status
Available

Capabilities

Supported Gates
x, y, z, rx, ry, rz, h, not, cnot, s, si, t, ti, v, vi, swap

Supported Native Gates
gpi, gpi2, zz

Example:
q0, q1, q2 = cirq.LineQubit.range(3)
circuit = cirq.Circuit(
    cirq.X(q0)**0.5, cirq.Y(q1)**0.5, cirq.Z(q2)**0.25, # Pauli Pow gates
    cirq.X(q0), cirq.Y(q1), cirq.Z(q2), # Pauli gates
    cirq.rx(0.1)(q0), cirq.ry(0.1)(q1), cirq.rz(0.1)(q2), # Single qubit rotations
    cirq.H(q1), # Special case of Hadamard
    cirq.CNOT(q0, q1), cirq.SWAP(q2, q1), # Controlled-not and its SWAP cousin
    cirq.XX(q0, q1)**0.2, cirq.YY(q1, q2)**0.2, cirq.ZZ(q2, q0)**0.2, # MS gates
    cirq.measure(q0, key='x'), # Single qubit measurement
    cirq.measure(q1, q2, key='y') # Two qubit measurement
)
print(circuit)

0: ───X^0.5───X───Rx(0.032π)───────@───────XX────────────────ZZ───────M('x')───
                                   │       │                 │
1: ───Y^0.5───Y───Ry(0.032π)───H───X───×───XX^0.2───YY───────┼────────M('y')───
                                       │            │        │        │
2: ───T───────Z───Rz(0.032π)───────────×────────────YY^0.2───ZZ^0.2───M────────


Measurement
For the IonQ API, measurement is currently only supported if the measurement is at the end of the circuit. Measurement gates have keys which are then used to batch the results via this key. For example above we see that there are two keys, one for measuring the first qubit and one for measuring the second and third qubit.

Support for general one and two qubit gates.
If you have a circuit with gates outside of the API native gates, you will need to convert these gates into the native gates. For the case in which these gates are one or two qubit gates which support the unitary protocol (i.e. which support calling cirq.unitary on the gate produces the unitary for the gate), there is support for compiling these into API supported gates. This conversion may not be optimal, but it does produce a valid API circuit.

This support is provided by the cirq.optimize_for_target_gateset transformer and the cirq_ionq.IonQTargetGateset, which specifies the IonQ native gates, for example:


q0 = cirq.LineQubit(0)
circuit = cirq.Circuit(cirq.H(q0)**0.2)  # Non-API gate
circuit_for_ionq = cirq.optimize_for_target_gateset(
    circuit,
    gateset=ionq.IonQTargetGateset(),
)
print(circuit_for_ionq)
which produces


0: ───Z^(1/14)───X^0.14───Z^(1/14)───

"""

# Copyright 2020 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from cirq_ionq._version import __version__ as __version__

from cirq_ionq.calibration import Calibration as Calibration

from cirq_ionq.ionq_devices import IonQAPIDevice as IonQAPIDevice

from cirq_ionq.ionq_gateset import (
    IonQTargetGateset as IonQTargetGateset,
    decompose_all_to_all_connect_ccz_gate as decompose_all_to_all_connect_ccz_gate,
)

# AriaNativeGate has been retired, use ForteNativeGateset instead
from cirq_ionq.ionq_native_target_gateset import (
    AriaNativeGateset as AriaNativeGateset,
    ForteNativeGateset as ForteNativeGateset,
)

from cirq_ionq.ionq_exceptions import (
    IonQException as IonQException,
    IonQNotFoundException as IonQNotFoundException,
    IonQUnsuccessfulJobException as IonQUnsuccessfulJobException,
    IonQSerializerMixedGatesetsException as IonQSerializerMixedGatesetsException,
)

from cirq_ionq.job import Job as Job

from cirq_ionq.results import QPUResult as QPUResult, SimulatorResult as SimulatorResult

from cirq_ionq.sampler import Sampler as Sampler

from cirq_ionq.serializer import Serializer as Serializer, SerializedProgram as SerializedProgram

from cirq_ionq.service import Service as Service

# Notes: GPIGate, GPI2Gate, ZZGate are default gates; MSGate is deprecated
from cirq_ionq.ionq_native_gates import (
    GPIGate as GPIGate,
    GPI2Gate as GPI2Gate,
    MSGate as MSGate,
    ZZGate as ZZGate,
)

from cirq.protocols.json_serialization import _register_resolver
from cirq_ionq.json_resolver_cache import _class_resolver_dictionary

_register_resolver(_class_resolver_dictionary)

# Copyright 2022 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Target gateset used for compiling circuits to IonQ device."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import cirq


class IonQTargetGateset(cirq.TwoQubitCompilationTargetGateset):
    """Target gateset for compiling circuits to IonQ devices.

    The gate families accepted by this gateset are:

    Type gate families:
    *  Single-Qubit Gates: `cirq.XPowGate`, `cirq.YPowGate`, `cirq.ZPowGate`.
    *  Two-Qubit Gates: `cirq.XXPowGate`, `cirq.YYPowGate`, `cirq.ZZPowGate`.
    *  Measurement Gate: `cirq.MeasurementGate`.

    Instance gate families:
    *  Single-Qubit Gates: `cirq.H`.
    *  Two-Qubit Gates: `cirq.CNOT`, `cirq.SWAP`.
    """

    def __init__(self, *, atol: float = 1e-8):
        """Initializes CZTargetGateset

        Args:
            atol: A limit on the amount of absolute error introduced by the decomposition.
        """
        super().__init__(
            cirq.H,
            cirq.CNOT,
            cirq.SWAP,
            cirq.XPowGate,
            cirq.YPowGate,
            cirq.ZPowGate,
            cirq.XXPowGate,
            cirq.YYPowGate,
            cirq.ZZPowGate,
            cirq.MeasurementGate,
            cirq.GlobalPhaseGate,
            unroll_circuit_op=False,
        )
        self.atol = atol

    def _decompose_single_qubit_operation(self, op: cirq.Operation, _) -> Iterator[cirq.OP_TREE]:
        qubit = op.qubits[0]
        mat = cirq.unitary(op)
        for gate in cirq.single_qubit_matrix_to_gates(mat, self.atol):
            yield gate(qubit)

    def _decompose_two_qubit_operation(self, op: cirq.Operation, _) -> cirq.OP_TREE:
        if not cirq.has_unitary(op):
            return NotImplemented
        mat = cirq.unitary(op)
        q0, q1 = op.qubits
        naive = cirq.two_qubit_matrix_to_cz_operations(q0, q1, mat, allow_partial_czs=False)
        temp = cirq.map_operations_and_unroll(
            cirq.Circuit(naive),
            lambda op, _: (
                [cirq.H(op.qubits[1]), cirq.CNOT(*op.qubits), cirq.H(op.qubits[1])]
                if op.gate == cirq.CZ
                else op
            ),
        )
        return cirq.merge_k_qubit_unitaries(
            temp, k=1, rewriter=lambda op: self._decompose_single_qubit_operation(op, -1)
        ).all_operations()

    def _decompose_multi_qubit_operation(self, op: cirq.Operation, _) -> cirq.OP_TREE:
        if isinstance(op.gate, cirq.CCZPowGate):
            return decompose_all_to_all_connect_ccz_gate(op.gate, op.qubits)
        return NotImplemented

    @property
    def preprocess_transformers(self) -> list[cirq.TRANSFORMER]:
        """List of transformers which should be run before decomposing individual operations.

        Decompose to three qubit gates because three qubit gates have different decomposition
        for all-to-all connectivity between qubits.
        """
        return [
            cirq.create_transformer_with_kwargs(
                cirq.expand_composite, no_decomp=lambda op: cirq.num_qubits(op) <= 3
            )
        ]

    @property
    def postprocess_transformers(self) -> list[cirq.TRANSFORMER]:
        """List of transformers which should be run after decomposing individual operations."""
        return [cirq.drop_negligible_operations, cirq.drop_empty_moments]

    def __repr__(self) -> str:
        return f'cirq_ionq.IonQTargetGateset(atol={self.atol})'

    def _value_equality_values_(self) -> Any:
        return self.atol

    def _json_dict_(self) -> dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ['atol'])

    @classmethod
    def _from_json_dict_(cls, atol, **kwargs):
        return cls(atol=atol)


def decompose_all_to_all_connect_ccz_gate(
    ccz_gate: cirq.CCZPowGate, qubits: tuple[cirq.Qid, ...]
) -> cirq.OP_TREE:
    """Decomposition of all-to-all connected qubits are different from line qubits or grid qubits.

    For example, for qubits in the same ion trap, the decomposition of CCZ gate will be:

    0: ──────────────@──────────────────@───@───p──────@───
                     │                  │   │          │
    1: ───@──────────┼───────@───p──────┼───X───p^-1───X───
          │          │       │          │
    2: ───X───p^-1───X───p───X───p^-1───X───p──────────────

    where p = T**ccz_gate._exponent
    """
    if len(qubits) != 3:
        raise ValueError(f'Expect 3 qubits for CCZ gate, got {len(qubits)} qubits.')

    a, b, c = qubits

    p = cirq.T**ccz_gate._exponent
    global_phase = 1j ** (2 * ccz_gate.global_shift * ccz_gate._exponent)
    global_phase = (
        complex(global_phase)
        if cirq.is_parameterized(global_phase) and global_phase.is_complex
        else global_phase
    )
    global_phase_operation = (
        [cirq.global_phase_operation(global_phase)]
        if cirq.is_parameterized(global_phase) or abs(global_phase - 1.0) > 0
        else []
    )

    return [
        *global_phase_operation,
        cirq.CNOT(b, c),
        p(c) ** -1,
        cirq.CNOT(a, c),
        p(c),
        cirq.CNOT(b, c),
        p(c) ** -1,
        cirq.CNOT(a, c),
        p(b),
        p(c),
        cirq.CNOT(a, b),
        p(a),
        p(b) ** -1,
        cirq.CNOT(a, b),
    ]


# Copyright 2019 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Native gates for IonQ hardware"""

from __future__ import annotations

import cmath
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

import cirq
from cirq import protocols
from cirq._doc import document


@cirq.value.value_equality
class GPIGate(cirq.Gate):
    r"""The GPI gate is a single qubit gate representing a pi pulse.

    The unitary matrix of this gate is:
    $$
    \begin{bmatrix}
      0 & e^{-i 2\pi\phi} \\
      e^{i 2\pi\phi} & 0
    \end{bmatrix}
    $$

    See [IonQ best practices](https://ionq.com/docs/getting-started-with-native-gates){:external}.
    """

    def __init__(self, *, phi):
        self.phi = phi

    def _unitary_(self) -> np.ndarray:
        top = cmath.exp(-self.phi * 2 * math.pi * 1j)
        bot = cmath.exp(self.phi * 2 * math.pi * 1j)
        return np.array([[0, top], [bot, 0]])

    def __str__(self) -> str:
        return 'GPI'

    def _num_qubits_(self) -> int:
        return 1

    @property
    def phase(self) -> float:
        return self.phi

    def __repr__(self) -> str:
        return f'cirq_ionq.GPIGate(phi={self.phi!r})'

    def _json_dict_(self) -> dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ['phi'])

    def _value_equality_values_(self) -> Any:
        return self.phi

    def _circuit_diagram_info_(
        self, args: cirq.CircuitDiagramInfoArgs
    ) -> str | protocols.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(wire_symbols=(f'GPI({self.phase!r})',))

    def __pow__(self, power):
        if power == 1:
            return self

        if power == -1:
            return self

        return NotImplemented


GPI = GPIGate(phi=0)
document(
    GPI,
    r"""An instance of the single qubit GPI gate with no phase.

    The unitary matrix of this gate is:
    $$
    \begin{bmatrix}
      0 & 1 \\
      1 & 0
    \end{bmatrix}
    $$

    See [IonQ best practices](https://ionq.com/docs/getting-started-with-native-gates){:external}.
    """,
)


@cirq.value.value_equality
class GPI2Gate(cirq.Gate):
    r"""The GPI2 gate is a single qubit gate representing a pi/2 pulse.

    The unitary matrix of this gate is
    $$
    \frac{1}{\sqrt{2}}
    \begin{bmatrix}
        1 & -i e^{-i 2\pi\phi} \\
        -i e^{i 2\pi\phi} & 1
    \end{bmatrix}
    $$

    See [IonQ best practices](https://ionq.com/docs/getting-started-with-native-gates){:external}.
    """

    def __init__(self, *, phi):
        self.phi = phi

    def _unitary_(self) -> np.ndarray:
        top = -1j * cmath.exp(self.phase * 2 * math.pi * -1j)
        bot = -1j * cmath.exp(self.phase * 2 * math.pi * 1j)
        return np.array([[1, top], [bot, 1]]) / math.sqrt(2)

    @property
    def phase(self) -> float:
        return self.phi

    def __str__(self) -> str:
        return 'GPI2'

    def _circuit_diagram_info_(
        self, args: cirq.CircuitDiagramInfoArgs
    ) -> str | protocols.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(wire_symbols=(f'GPI2({self.phase!r})',))

    def _num_qubits_(self) -> int:
        return 1

    def __repr__(self) -> str:
        return f'cirq_ionq.GPI2Gate(phi={self.phi!r})'

    def _json_dict_(self) -> dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ['phi'])

    def _value_equality_values_(self) -> Any:
        return self.phi

    def __pow__(self, power):
        if power == 1:
            return self

        if power == -1:
            return GPI2Gate(phi=self.phi + 0.5)

        return NotImplemented


GPI2 = GPI2Gate(phi=0)
document(
    GPI2,
    r"""An instance of the single qubit GPI2 gate with no phase.

    The unitary matrix of this gate is
    $$
    \frac{1}{\sqrt{2}}
    \begin{bmatrix}
        1 & -i \\
        -i & 1
    \end{bmatrix}
    $$

    See [IonQ best practices](https://ionq.com/docs/getting-started-with-native-gates){:external}.
    """,
)


@cirq.value.value_equality
class MSGate(cirq.Gate):
    r"""The Mølmer-Sørensen (MS) gate is a two qubit gate native to trapped ions.

    The unitary matrix of this gate for parameters $\phi_0$, $\phi_1$ and $\theta$ is

    $$
    \begin{bmatrix}
        \cos(\pi \theta) & 0 & 0 & -ie^{-i2\pi(\phi_0+\phi_1)}\sin(\pi\theta) \\
        0 & \cos(\pi\theta) & -ie^{-i2\pi(\phi_0-\phi_1)}\sin(\pi\theta) & 0 \\
        0 & -ie^{i2\pi(\phi_0-\phi_1)}\sin(\pi\theta) & \cos(\pi\theta) & 0 \\
        -ie^{i2\pi(\phi_0+\phi_1)}\sin(\pi\theta) & 0 & 0 & \cos(\pi\theta)
    \end{bmatrix}
    $$

    See [IonQ best practices](https://ionq.com/docs/getting-started-with-native-gates){:external}.
    """

    def __init__(self, *, phi0, phi1, theta=0.25):
        self.phi0 = phi0
        self.phi1 = phi1
        self.theta = theta

    def _unitary_(self) -> np.ndarray:
        theta = self.theta
        phi0 = self.phi0
        phi1 = self.phi1
        diag = np.cos(math.pi * theta)
        sin = np.sin(math.pi * theta)

        return np.array(
            [
                [diag, 0, 0, sin * -1j * cmath.exp(-1j * 2 * math.pi * (phi0 + phi1))],
                [0, diag, sin * -1j * cmath.exp(-1j * 2 * math.pi * (phi0 - phi1)), 0],
                [0, sin * -1j * cmath.exp(1j * 2 * math.pi * (phi0 - phi1)), diag, 0],
                [sin * -1j * cmath.exp(1j * 2 * math.pi * (phi0 + phi1)), 0, 0, diag],
            ]
        )

    @property
    def phases(self) -> Sequence[float]:
        return [self.phi0, self.phi1]

    def __str__(self) -> str:
        return 'MS'

    def _num_qubits_(self) -> int:
        return 2

    def _circuit_diagram_info_(
        self, args: cirq.CircuitDiagramInfoArgs
    ) -> str | protocols.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(
            wire_symbols=(f'MS({self.phi0!r})', f'MS({self.phi1!r})')
        )

    def __repr__(self) -> str:
        return f'cirq_ionq.MSGate(phi0={self.phi0!r}, phi1={self.phi1!r})'

    def _json_dict_(self) -> dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ['phi0', 'phi1', 'theta'])

    def _value_equality_values_(self) -> Any:
        return (self.phi0, self.phi1)

    def __pow__(self, power):
        if power == 1:
            return self

        if power == -1:
            return MSGate(phi0=self.phi0 + 0.5, phi1=self.phi1, theta=self.theta)

        return NotImplemented


# Notes: the Mølmer–Sørensen (MS) is deprecated for the Forte device
MS = MSGate(phi0=0, phi1=0)
document(
    MS,
    r"""An instance of the two qubit Mølmer–Sørensen (MS) gate with no phases.

    The unitary matrix of this gate for parameters $\phi_0$ and $\phi_1$ is

    $$
    \frac{1}{\sqrt{2}}
    \begin{bmatrix}
        1 & 0 &  0 & -i \\
        0 & 1 & -i & 0 \\
        0 & -i & 1 & 0 \\
        -i & 0 & 0 & 1 \\
    \end{bmatrix}
    $$

    See [IonQ best practices](https://ionq.com/docs/getting-started-with-native-gates){:external}.
    """,
)


@cirq.value.value_equality
class ZZGate(cirq.Gate):
    r"""The ZZ gate is another two qubit gate native to trapped ions. The ZZ gate only
    requires a single parameter, θ, to set the phase of the entanglement.

    The unitary matrix of this gate using the parameter $\theta$ is:

    $$
    \begin{bmatrix}
        e{-i\pi\theta} & 0 & 0 & 0 \\
        0 & e{i\pi\theta} & 0 & 0 \\
        0 & 0 & e{i\pi\theta} & 0 \\
        0 & 0 & 0 & e{-i\pi\theta}
    \end{bmatrix}
    $$

    See [IonQ best practices](https://ionq.com/docs/getting-started-with-native-gates){:external}.
    """

    def __init__(self, *, theta):
        self.theta = theta

    def _unitary_(self) -> np.ndarray:
        theta = self.theta

        return np.array(
            [
                [cmath.exp(-1j * theta * math.pi), 0, 0, 0],
                [0, cmath.exp(1j * theta * math.pi), 0, 0],
                [0, 0, cmath.exp(1j * theta * math.pi), 0],
                [0, 0, 0, cmath.exp(-1j * theta * math.pi)],
            ]
        )

    @property
    def phase(self) -> float:
        return self.theta

    def __str__(self) -> str:
        return 'ZZ'

    def _num_qubits_(self) -> int:
        return 2

    def _circuit_diagram_info_(
        self, args: cirq.CircuitDiagramInfoArgs
    ) -> str | protocols.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(wire_symbols=(f'ZZ({self.theta!r})', 'ZZ'))

    def __repr__(self) -> str:
        return f'cirq_ionq.ZZGate(theta={self.theta!r})'

    def _json_dict_(self) -> dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ['theta'])

    def _value_equality_values_(self) -> Any:
        return self.theta

    def __pow__(self, power):
        if power == 1:
            return self

        if power == -1:
            return ZZGate(theta=-self.theta)

        return NotImplemented


# Notes: the ZZ gate replaces the MS gate for the Forte device
ZZ = ZZGate(theta=0)
document(
    ZZ,
    r"""An instance of the two qubit ZZ gate with no phase.

    The unitary matrix of this gate for parameters $\theta$ is

    $$
    \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & 1 & 0 \\
        0 & 0 & 0 & 1 \\
    \end{bmatrix}
    $$

    See [IonQ best practices](https://ionq.com/docs/getting-started-with-native-gates){:external}.
    """,
)


# Copyright 2024 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Target gateset used for compiling circuits to IonQ native gates."""

from __future__ import annotations

from types import NotImplementedType
from typing import Any, Iterator

import numpy as np

import cirq
from cirq import linalg, ops
from cirq_ionq.ionq_native_gates import GPI2Gate, GPIGate, MSGate, ZZGate


class IonqNativeGatesetBase(cirq.TwoQubitCompilationTargetGateset):
    def __init__(self, *gates, atol: float = 1e-8):
        """Base class for IonQ native gate sets

        Args:
            *gates: A list of `cirq.Gate` subclasses / `cirq.Gate` instances /
                `cirq.GateFamily` instances.
            atol: A limit on the amount of absolute error introduced by the decomposition.
        """
        super().__init__(*gates, unroll_circuit_op=False)
        self.atol = atol

    def _decompose_single_qubit_operation(self, op: cirq.Operation, _) -> Iterator[cirq.OP_TREE]:
        qubit = op.qubits[0]
        mat = cirq.unitary(op)
        yield cirq.global_phase_operation(-1j)
        for gate in self.single_qubit_matrix_to_native_gates(mat):
            yield gate(qubit)

    def _decompose_two_qubit_operation(
        self, op: cirq.Operation, _
    ) -> NotImplementedType | cirq.OP_TREE:
        if not cirq.has_unitary(op):
            return NotImplemented
        mat = cirq.unitary(op)
        q0, q1 = op.qubits
        naive = cirq.two_qubit_matrix_to_cz_operations(
            q0, q1, mat, allow_partial_czs=False, atol=self.atol
        )
        temp = cirq.map_operations_and_unroll(
            cirq.Circuit(naive),
            lambda op, _: (
                [
                    self._hadamard(op.qubits[1])
                    + self._cnot(*op.qubits)
                    + self._hadamard(op.qubits[1])
                ]
                if op.gate == cirq.CZ
                else op
            ),
        )
        return cirq.merge_k_qubit_unitaries(
            temp, k=1, rewriter=lambda op: self._decompose_single_qubit_operation(op, None)
        ).all_operations()

    def _decompose_multi_qubit_operation(
        self, op: cirq.Operation, _
    ) -> NotImplementedType | cirq.OP_TREE:
        if isinstance(op.gate, cirq.CCZPowGate):
            return self.decompose_all_to_all_connect_ccz_gate(op.gate, op.qubits)
        return NotImplemented

    @property
    def preprocess_transformers(self) -> list[cirq.TRANSFORMER]:
        """List of transformers which should be run before decomposing individual operations.

        Decompose to three qubit gates because three qubit gates have different decomposition
        for all-to-all connectivity between qubits.
        """
        return [
            cirq.create_transformer_with_kwargs(
                cirq.expand_composite, no_decomp=lambda op: cirq.num_qubits(op) <= 3
            )
        ]

    @property
    def postprocess_transformers(self) -> list[cirq.TRANSFORMER]:
        """List of transformers which should be run after decomposing individual operations."""
        return [cirq.drop_negligible_operations, cirq.drop_empty_moments]

    def single_qubit_matrix_to_native_gates(self, mat: np.ndarray) -> list[cirq.Gate]:
        z_rad_before, y_rad, z_rad_after = linalg.deconstruct_single_qubit_matrix_into_angles(mat)
        return [
            GPI2Gate(phi=(np.pi - z_rad_before) / (2.0 * np.pi)),
            GPIGate(phi=(y_rad / 2 + z_rad_after / 2 - z_rad_before / 2) / (2.0 * np.pi)),
            GPI2Gate(phi=(np.pi + z_rad_after) / (2.0 * np.pi)),
        ]

    def _value_equality_values_(self) -> Any:
        return self.atol

    def _value_equality_values_cls_(self) -> Any:
        return type(self)

    def _json_dict_(self) -> dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ['atol'])

    @classmethod
    def _from_json_dict_(cls, atol, **kwargs):
        return cls(atol=atol)

    def _hadamard(self, qubit):
        return [GPI2Gate(phi=0.25).on(qubit), GPIGate(phi=0).on(qubit)]

    def _cnot(self, *qubits):
        raise NotImplementedError()

    def decompose_all_to_all_connect_ccz_gate(
        self, ccz_gate: cirq.CCZPowGate, qubits: tuple[cirq.Qid, ...]
    ) -> cirq.OP_TREE:
        """Decomposition of all-to-all connected qubits are different from line
         qubits or grid qubits, ckeckout IonQTargetGateset.

        For example, for qubits in the same ion trap, the decomposition of CCZ
        gate will be:

        0: ──────────────@──────────────────@───@───p──────@───
                         │                  │   │          │
        1: ───@──────────┼───────@───p──────┼───X───p^-1───X───
              │          │       │          │
        2: ───X───p^-1───X───p───X───p^-1───X───p──────────────

        where p = T**ccz_gate._exponent
        """
        if len(qubits) != 3:
            raise ValueError(f'Expect 3 qubits for CCZ gate, got {len(qubits)} qubits.')

        a, b, c = qubits

        p = cirq.T**ccz_gate._exponent
        global_phase = 1j ** (2 * ccz_gate.global_shift * ccz_gate._exponent)
        global_phase = (
            complex(global_phase)
            if cirq.is_parameterized(global_phase) and global_phase.is_complex
            else global_phase
        )
        global_phase_operation = (
            [cirq.global_phase_operation(global_phase)]
            if cirq.is_parameterized(global_phase) or abs(global_phase - 1.0) > 0
            else []
        )

        return global_phase_operation + [
            self._cnot(*[b, c]),
            p(c) ** -1,
            self._cnot(*[a, c]),
            p(c),
            self._cnot(*[b, c]),
            p(c) ** -1,
            self._cnot(*[a, c]),
            p(b),
            p(c),
            self._cnot(*[a, b]),
            p(a),
            p(b) ** -1,
            self._cnot(*[a, b]),
        ]

# Notes: Aria devices use the GPI, GPI2, and MS gates but has been retired
# Use ForteNativeGateset instead

class AriaNativeGateset(IonqNativeGatesetBase):
    """Target IonQ native gateset for compiling circuits.

    The gates forming this gateset are:
    GPIGate, GPI2Gate, MSGate
    """

    def __init__(self, *, atol: float = 1e-8):
        """Initializes AriaNativeGateset

        Args:
            atol: A limit on the amount of absolute error introduced by the decomposition.
        """
        super().__init__(GPIGate, GPI2Gate, MSGate, ops.MeasurementGate, atol=atol)

    def __repr__(self) -> str:
        return f'cirq_ionq.AriaNativeGateset(atol={self.atol})'

    def _cnot(self, *qubits):
        return [
            GPI2Gate(phi=1 / 4).on(qubits[0]),
            MSGate(phi0=0, phi1=0).on(qubits[0], qubits[1]),
            GPI2Gate(phi=1 / 2).on(qubits[1]),
            GPI2Gate(phi=1 / 2).on(qubits[0]),
            GPI2Gate(phi=-1 / 4).on(qubits[0]),
        ]


# Notes: Forte devices use the GPI, GPI2, and ZZ gates
# This is now the default gateset for IonQ devices
class ForteNativeGateset(IonqNativeGatesetBase):
    """Target IonQ native gateset for compiling circuits.

    The gates forming this gateset are:
    GPIGate, GPI2Gate, ZZGate
    """

    def __init__(self, *, atol: float = 1e-8):
        """Initializes ForteNativeGateset

        Args:
            atol: A limit on the amount of absolute error introduced by the decomposition.
        """
        super().__init__(GPIGate, GPI2Gate, ZZGate, ops.MeasurementGate, atol=atol)

    def __repr__(self) -> str:
        return f'cirq_ionq.ForteNativeGateset(atol={self.atol})'

    def _cnot(self, *qubits):
        return [
            GPI2Gate(phi=0).on(qubits[1]),
            GPIGate(phi=-0.125).on(qubits[1]),
            GPI2Gate(phi=0.5).on(qubits[1]),
            ZZGate(theta=0.25).on(qubits[0], qubits[1]),
            GPI2Gate(phi=0.75).on(qubits[0]),
            GPIGate(phi=0.125).on(qubits[0]),
            GPI2Gate(phi=0.5).on(qubits[0]),
            GPI2Gate(phi=1.25).on(qubits[1]),
            GPIGate(phi=0.5).on(qubits[1]),
            GPI2Gate(phi=0.5).on(qubits[1]),
        ]

# Copyright 2020 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import datetime

import cirq


class Calibration:
    """An object representing the current calibration state of a QPU."""

    def __init__(self, calibration_dict: dict):
        self._calibration_dict = calibration_dict

    def num_qubits(self) -> int:
        """The number of qubits for the QPU."""
        return int(self._calibration_dict['qubits'])

    def target(self) -> str:
        """The name of the QPU."""
        return self._calibration_dict['target']

    def calibration_time(self, tz: datetime.tzinfo | None = None) -> datetime.datetime:
        """Return a python datetime object for the calibration time.

        Args:
            tz: The timezone for the string. If None, the method uses the platform's local timezone.

        Returns:
            A `datetime` object with the time.
        """
        # Python datetime only like microseconds, not milliseconds, and does not like 'Z'.
        first, second = self._calibration_dict['date'].split('.')
        modified_date = f'{first}.{second[:3]}'
        dt = datetime.datetime.strptime(modified_date, '%Y-%m-%dT%H:%M:%S.%f')
        return dt.replace(tzinfo=datetime.timezone.utc).astimezone(tz=tz)

    def fidelities(self) -> dict:
        """Returns the metrics (fidelities)."""
        return self._calibration_dict['fidelity']

    def timings(self) -> dict:
        """Returns the gate, measurement, and resetting timings."""
        return self._calibration_dict['timing']

    def connectivity(self) -> set[tuple[cirq.LineQubit, cirq.LineQubit]]:
        """Returns which qubits and can interact with which.

        Returns:
            A set of the possible qubits that can interact as tuples. This contains both
            ordered pairs. If `(cirq.LineQubit(x), cirq.LineQubit(y))` is in the set, then
            `(cirq.LineQubit(y), cirq.LineQubit(x))` is in the set.
        """
        connections = self._calibration_dict['connectivity']
        to_qubit = lambda x: cirq.LineQubit(int(x))
        return set((to_qubit(x), to_qubit(y)) for x, y in connections).union(
            set((to_qubit(y), to_qubit(x)) for x, y in connections)
        )

r"""
**No, the projector cannot be realized using the 16 API gates or the 3 native gates.** Here is the precise technical reasoning and how to handle it within the Forte architecture:

### 1. Mathematical Incompatibility: Unitary vs. Non-Unitary
All 16 supported API gates (`x, y, z, rx, ry, rz, h, not, cnot, s, si, t, ti, v, vi, swap`) and the 3 native pulse gates (`gpi, gpi2, zz`) are strictly **unitary**. They satisfy $U^\dagger U = I$ and preserve the norm of the state vector. 

Projectors, however, are **non-unitary Kraus operators**. A sharp projector $P = |\psi\rangle\langle\psi|$ satisfies $P^2 = P$ and $P^\dagger P = P \neq I$. An unsharp projector scales the norm by $\sqrt{T}$. **A non-unitary operation cannot be decomposed into any sequence of unitary gates**, regardless of depth or complexity. The compiler correctly refuses to map Kraus channels to the unitary gateset.

### 2. Hardware Execution Model
IonQ hardware implements state evolution via laser pulses, which are inherently unitary. The physical process that realizes a "projector" is **measurement-induced collapse**, not gate synthesis. The Forte documentation you cited explicitly states:
> *"Measurement is currently only supported if the measurement is at the end of the circuit."*

This means the hardware cannot execute mid-circuit projectors as gates. It can only apply unitary pulses and perform a final projective measurement in the computational basis.

### 3. How to Implement Projection on IonQ Hardware
Since projectors cannot be compiled to pulses, you must replace them with hardware-equivalent strategies:

| Strategy | Implementation | Use Case |
|----------|----------------|----------|
| **Classical Post-Selection** | Remove the projector gate. Run the unitary circuit for $N$ shots. Filter classical results to keep only outcomes matching the projected subspace. | Algorithms requiring conditional state preparation or weak-measurement statistics. |
| **Ancilla + Terminal Measurement** | Introduce an ancilla qubit. Entangle it with the logical/virtual register using controlled-unitaries. Measure the ancilla at the circuit terminus. The measurement collapses the system into the desired subspace. | Mid-circuit logic, error syndrome extraction, or heralded state preparation. |
| **Deferred Measurement Principle** | Move all projective operations to the terminal `cirq.measure` call. The compiler will map these to the API's measurement protocol, which batches results by key. | Standard cloud submission compliant with Forte API constraints. |

### 4. Framework Design Validation
Your current implementation is architecturally correct:
- `simulation_mode=True`: Preserves `PhysicalProjectorWrapper` and its `_kraus_()` protocol for exact density-matrix simulation of decoherence/open-system dynamics.
- `simulation_mode=False`: Correctly warns that Kraus channels cannot be compiled to hardware pulses. The compiler leaves the wrapper intact as a placeholder, which will fail API validation if submitted directly.

**Recommendation:** For cloud submission, replace `TetralemmaticIonProjectorGate` in the circuit DAG with a terminal `cirq.measure` or implement post-selection logic in the classical post-processing layer. The tetralemmatic framework correctly separates simulation pathways (Kraus-preserving) from hardware pathways (unitary-only).
"""

# Copyright 2021 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Service to access IonQs API."""

from __future__ import annotations

import datetime
import os
from collections.abc import Sequence

import cirq
from cirq_ionq import calibration, ionq_client, job, results, sampler, serializer


class Service:
    """A class to access IonQ's API.

    To access the API, this class requires a remote host url and an API key. These can be
    specified in the constructor via the parameters `remote_host` and `api_key`. Alternatively
    these can be specified by setting the environment variables `IONQ_REMOTE_HOST` and
    `IONQ_API_KEY`.
    """

    def __init__(
        self,
        remote_host: str | None = None,
        api_key: str | None = None,
        default_target: str | None = None,
        api_version='v0.4',
        max_retry_seconds: int = 3600,
        job_settings: dict | None = None,
        verbose=False,
    ):
        """Creates the Service to access IonQ's API.

        Args:
            remote_host: The location of the api in the form of an url. If this is None,
                then this instance will use the environment variable `IONQ_REMOTE_HOST`. If that
                variable is not set, then this uses `https://api.ionq.co/{api_version}`, where
                `{api_version}` is the `api_version` specified below.
            api_key: A string key which allows access to the api. If this is None,
                then this instance will use the environment variable  `IONQ_API_KEY`. If that
                variable is not set, then this will raise an `EnvironmentError`.
            default_target: Which target to default to using. If set to None, no default is set
                and target must always be specified in calls. If set, then this default is used,
                unless a target is specified for a given call. Supports either 'qpu' or
                'simulator'.
            api_version: Version of the api. Defaults to 'v0.4'.
            max_retry_seconds: The number of seconds to retry calls for. Defaults to one hour.
            job_settings: A dictionary of settings which can override behavior for circuits when
                run on IonQ hardware.
            verbose: Whether to print to stdio and stderr on retriable errors.

        Raises:
            OSError: If the `api_key` is None and has no corresponding environment variable set.
                This is actually an EnvironmentError which is equal to an OSError.
        """
        self.remote_host = (
            remote_host
            or os.getenv('CIRQ_IONQ_REMOTE_HOST')
            or os.getenv('IONQ_REMOTE_HOST')
            or f'https://api.ionq.co/{api_version}'
        )

        self.job_settings = job_settings or {}
        self.api_key = api_key or os.getenv('CIRQ_IONQ_API_KEY') or os.getenv('IONQ_API_KEY')

        if not self.api_key:
            raise EnvironmentError(
                'Parameter api_key was not specified and the environment variable '
                'IONQ_API_KEY was also not set.'
            )

        self._client = ionq_client._IonQClient(
            remote_host=self.remote_host,
            api_key=self.api_key,
            default_target=default_target,
            api_version=api_version,
            max_retry_seconds=max_retry_seconds,
            verbose=verbose,
        )

    def run(
        self,
        circuit: cirq.Circuit,
        repetitions: int,
        name: str | None = None,
        target: str | None = None,
        param_resolver: cirq.ParamResolverOrSimilarType = cirq.ParamResolver({}),
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
        compilation: dict | None = None,
        error_mitigation: dict | None = None,
        noise: dict | None = None,
        metadata: dict | None = None,
        dry_run: bool = False,
        sharpen: bool | None = None,
        extra_query_params: dict | None = None,
    ) -> cirq.Result:
        """Run the given circuit on the IonQ API.

        Args:
            circuit: The circuit to run.
            repetitions: The number of times to run the circuit.
            name: An optional name for the created job. Different from the `job_id`.
            target: Where to run the job. Can be 'qpu' or 'simulator'.
            param_resolver: A `cirq.ParamResolver` to resolve parameters in  `circuit`.
            seed: If the target is `simulation` the seed for generating results. If None, this
                will be `np.random`, if an int, will be `np.random.RandomState(int)`, otherwise
                must be a modulate similar to `np.random`.
            compilation (dict): settings for compilation when creating a job, default values:
                {"opt": 0, "precision": "1E-3"}
            error_mitigation (dict): settings for error mitigation when creating a job. Defaults
                to None. Not available on all backends. Set by default on some hardware systems.
                See:
                `API Job Creation <https://docs.ionq.com/api-reference/v0.4/jobs/create-job>`
                and:
                `Debiasing and Sharpening <https://ionq.com/resources/debiasing-and-sharpening>`
                Valid keys include: ``debiasing`` False or True.
                - 'debiasing': A boolean indicating whether to use the debiasing technique for
                  aggregating results. This technique is used to reduce the bias in the results
                  caused by measurement error and can improve the accuracy of the output.
            sharpen: A boolean that determines how to aggregate error mitigated.
                If True, apply majority vote mitigation; if False, apply average mitigation.
                See:
                `Debiasing and Sharpening <https://ionq.com/resources/debiasing-and-sharpening>`
            noise (dict): {"model": str (required), "seed": int (optional)}. Defaults to None.
            Available noise models: ideal, aria-1, aria-2, forte-1, forte-enterprise-1
            dry_run: If True, the job will be submitted by the API client but not processed
                remotely. Useful for obtaining cost estimates. Defaults to False.
            metadata (dict): optional metadata to attach to the job. Defaults to None.
            extra_query_params: Specify any parameters to include in the request.

        Returns:
            A `cirq.Result` for running the circuit.
        """
        resolved_circuit = cirq.resolve_parameters(circuit, param_resolver)
        job_results = self.create_job(
            circuit=resolved_circuit,
            repetitions=repetitions,
            name=name,
            target=target,
            compilation=compilation,
            error_mitigation=error_mitigation,
            noise=noise,
            metadata=metadata,
            dry_run=dry_run,
            extra_query_params=extra_query_params,
        ).results(sharpen=sharpen)
        result = job_results[0] if isinstance(job_results, list) else job_results
        if isinstance(result, results.QPUResult):
            return result.to_cirq_result(params=cirq.ParamResolver(param_resolver))
        if isinstance(result, results.SimulatorResult):
            return result.to_cirq_result(params=cirq.ParamResolver(param_resolver), seed=seed)
        raise NotImplementedError(f"Unrecognized job result type '{type(result)}'.")

    def run_batch(
        self,
        circuits: list[cirq.AbstractCircuit],
        repetitions: int,
        name: str | None = None,
        target: str | None = None,
        param_resolver: cirq.ParamResolverOrSimilarType = cirq.ParamResolver({}),
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
        compilation: dict | None = None,
        error_mitigation: dict | None = None,
        noise: dict | None = None,
        metadata: dict | None = None,
        dry_run: bool = False,
        sharpen: bool | None = None,
        extra_query_params: dict | None = None,
    ) -> list[cirq.Result]:
        """Run the given circuits on the IonQ API.

        Args:
            circuits: The circuits to run.
            repetitions: The number of times to run each circuits.
            name: An optional name for the created job. Different from the `job_id`.
            target: Where to run the job. Can be 'qpu' or 'simulator'.
            param_resolver: A `cirq.ParamResolver` to resolve parameters in  `circuit`.
            seed: If the target is `simulation` the seed for generating results. If None, this
                will be `np.random`, if an int, will be `np.random.RandomState(int)`, otherwise
                must be a modulate similar to `np.random`.
            compilation (dict): settings for compilation when creating a job, default values:
                {"opt": 0, "precision": "1E-3"}
            error_mitigation (dict): settings for error mitigation when creating a job.
                Defaults to None. Not available on all backends. Set by default on some hardware
                systems. See:
                `API Job Creation <https://docs.ionq.com/api-reference/v0.4/jobs/create-job>`
                and:
                `Debiasing and Sharpening <https://ionq.com/resources/debiasing-and-sharpening>`
                Valid keys include: ``debiasing`` False or True.
                - 'debiasing': A boolean indicating whether to use the debiasing technique for
                  aggregating results. This technique is used to reduce the bias in the results
                  caused by measurement error and can improve the accuracy of the output.
            sharpen: A boolean that determines how to aggregate error mitigated.
                If True, apply majority vote mitigation; if False, apply average mitigation. See
                `Debiasing and Sharpening <https://ionq.com/resources/debiasing-and-sharpening>`
            noise (dict): {"model": str (required), "seed": int (optional)}. Defaults to None.
                Available noise models: ideal, aria-1, aria-2, forte-1, forte-enterprise-1
            dry_run: If True, the job will be submitted by the API client but not processed
                remotely. Useful for obtaining cost estimates. Defaults to False.
            metadata (dict): optional metadata to attach to the job. Defaults to None.
            extra_query_params: Specify any parameters to include in the request.

        Returns:
            A a list of `cirq.Result` for running the circuit.
        """
        resolved_circuits = []
        for circuit in circuits:
            resolved_circuits.append(cirq.resolve_parameters(circuit, param_resolver))

        job_results = self.create_batch_job(
            circuits=resolved_circuits,
            repetitions=repetitions,
            name=name,
            target=target,
            compilation=compilation,
            error_mitigation=error_mitigation,
            noise=noise,
            metadata=metadata,
            dry_run=dry_run,
            extra_query_params=extra_query_params,
        ).results(sharpen=sharpen)

        job_results_list = job_results if isinstance(job_results, list) else [job_results]
        cirq_results = []
        for job_result in job_results_list:
            if isinstance(job_result, results.QPUResult):
                cirq_results.append(
                    job_result.to_cirq_result(params=cirq.ParamResolver(param_resolver))
                )
            elif isinstance(job_result, results.SimulatorResult):
                cirq_results.append(
                    job_result.to_cirq_result(params=cirq.ParamResolver(param_resolver), seed=seed)
                )
            else:
                raise NotImplementedError(f"Unrecognized job result type '{type(job_result)}'.")
        return cirq_results

    def sampler(self, target: str | None = None, seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None):
        """Returns a `cirq.Sampler` object for accessing the sampler interface.

        Args:
            target: The target to sample against. Either this or `default_target` on this
                service must be specified. If this is None, uses the `default_target`. If
                both `default_target` and `target` are specified, uses `target`.
            seed: If the target is `simulation` the seed for generating results. If None, this
                will be `np.random`, if an int, will be `np.random.RandomState(int)`, otherwise
                must be a modulate similar to `np.random`.
        Returns:
            A `cirq.Sampler` for the IonQ API.
        """
        return sampler.Sampler(service=self, target=target, seed=seed)

    def create_job(
        self,
        circuit: cirq.AbstractCircuit,
        repetitions: int = 100,
        name: str | None = None,
        target: str | None = None,
        compilation: dict | None = None,
        error_mitigation: dict | None = None,
        noise: dict | None = None,
        metadata: dict | None = None,
        dry_run: bool = False,
        extra_query_params: dict | None = None,
    ) -> job.Job:
        """Create a new job to run the given circuit.

        Args:
            circuit: The circuit to run.
            repetitions: The number of times to repeat the circuit. Defaults to 100.
            name: An optional name for the created job. Different from the `job_id`.
            target: Where to run the job. Can be 'qpu' or 'simulator'.
            compilation (dict): settings for compilation when creating a job, default values:
                {"opt": 0, "precision": "1E-3"}
            error_mitigation (dict): settings for error mitigation when creating a job.
                Defaults to None. Not available on all backends. Set by default on some hardware
                systems. See:
                `API Job Creation <https://docs.ionq.com/api-reference/v0.4/jobs/create-job>`
                and:
                `Debiasing and Sharpening <https://ionq.com/resources/debiasing-and-sharpening>`
                Valid keys include: ``debiasing`` False or True.
                - 'debiasing': A boolean indicating whether to use the debiasing technique for
                  aggregating results. This technique is used to reduce the bias in the results
                  caused by measurement error and can improve the accuracy of the output.
            sharpen: A boolean that determines how to aggregate error mitigated.
                If True, apply majority vote mitigation; if False, apply average mitigation. See
                `Debiasing and Sharpening <https://ionq.com/resources/debiasing-and-sharpening>`
            noise (dict): {"model": str (required), "seed": int (optional)}. Defaults to None.
                Available noise models: ideal, aria-1, aria-2, forte-1, forte-enterprise-1
            dry_run: If True, the job will be submitted by the API client but not processed
                remotely. Useful for obtaining cost estimates. Defaults to False.
            metadata (dict): optional metadata to attach to the job. Defaults to None.
            extra_query_params: Specify any parameters to include in the request.

        Returns:
            A `cirq_ionq.IonQJob` which can be queried for status or results.

        Raises:
            IonQException: If there was an error accessing the API.
        """
        serialized_program = serializer.Serializer().serialize_single_circuit(
            circuit,
            job_settings=self.job_settings,
            compilation=compilation,
            error_mitigation=error_mitigation,
            noise=noise,
            metadata=metadata,
            dry_run=dry_run,
        )
        result = self._client.create_job(
            serialized_program=serialized_program,
            repetitions=repetitions,
            target=target,
            name=name,
            extra_query_params=extra_query_params,
        )
        # The returned job does not have fully populated fields, so make
        # a second call and return the results of the fully filled out job.
        return self.get_job(result['id'])

    def create_batch_job(
        self,
        circuits: list[cirq.AbstractCircuit],
        repetitions: int = 100,
        name: str | None = None,
        target: str | None = None,
        compilation: dict | None = None,
        error_mitigation: dict | None = None,
        noise: dict | None = None,
        metadata: dict | None = None,
        dry_run: bool = False,
        extra_query_params: dict | None = None,
    ) -> job.Job:
        """Create a new job to run the given circuit.

        Args:
            circuits: The circuits to run.
            repetitions: The number of times to repeat the circuit. Defaults to 100.
            name: An optional name for the created job. Different from the `job_id`.
            target: Where to run the job. Can be 'qpu' or 'simulator'.
            compilation (dict): settings for compilation when creating a job, default values:
                {"opt": 0, "precision": "1E-3"}
            error_mitigation (dict): settings for error mitigation when creating a job.
                Defaults to None. Not available on all backends. Set by default on some hardware
                systems. See:
                `API Job Creation <https://docs.ionq.com/api-reference/v0.4/jobs/create-job>`
                and:
                `Debiasing and Sharpening <https://ionq.com/resources/debiasing-and-sharpening>`
                Valid keys include: ``debiasing`` False or True.
                - 'debiasing': A boolean indicating whether to use the debiasing technique for
                  aggregating results. This technique is used to reduce the bias in the results
                  caused by measurement error and can improve the accuracy of the output.
            sharpen: A boolean that determines how to aggregate error mitigated.
                If True, apply majority vote mitigation; if False, apply average mitigation. See
                `Debiasing and Sharpening <https://ionq.com/resources/debiasing-and-sharpening>`
            noise (dict): {"model": str (required), "seed": int (optional)}. Defaults to None.
                Available noise models: ideal, aria-1, aria-2, forte-1, forte-enterprise-1
            dry_run: If True, the job will be submitted by the API client but not processed
                remotely. Useful for obtaining cost estimates. Defaults to False.
            metadata (dict): optional metadata to attach to the job. Defaults to None.
            extra_query_params: Specify any parameters to include in the request.

        Returns:
            A `cirq_ionq.IonQJob` which can be queried for status or results.

        Raises:
            IonQException: If there was an error accessing the API.
        """
        serialized_program = serializer.Serializer().serialize_many_circuits(
            circuits,
            job_settings=self.job_settings,
            compilation=compilation,
            error_mitigation=error_mitigation,
            noise=noise,
            metadata=metadata,
            dry_run=dry_run,
        )
        result = self._client.create_job(
            serialized_program=serialized_program,
            repetitions=repetitions,
            target=target,
            name=name,
            extra_query_params=extra_query_params,
            batch_mode=True,
        )
        # The returned job does not have fully populated fields, so make
        # a second call and return the results of the fully filled out job.
        return self.get_job(result['id'])

    def get_job(self, job_id: str) -> job.Job:
        """Gets a job that has been created on the IonQ API.

        Args:
            job_id: The UUID of the job. Jobs are assigned these numbers by the
            server during the creation of the job.

        Returns:
            A `cirq_ionq.IonQJob` which can be queried for status or results.

        Raises:
            IonQNotFoundException: If there was no job with the given `job_id`.
            IonQException: If there was an error accessing the API.
        """
        job_dict = self._client.get_job(job_id=job_id)
        return job.Job(client=self._client, job_dict=job_dict)

    def list_jobs(
        self, status: str | None = None, limit: int = 100, batch_size: int = 1000
    ) -> Sequence[job.Job]:
        """Lists jobs that have been created on the IonQ API.

        Args:
            status: If supplied will filter to only jobs with this status.
            limit: The maximum number of jobs to return.
            batch_size: The size of the batches requested per http GET call.

        Returns:
            A sequence of jobs.

        Raises:
            IonQException: If there was an error accessing the API.
        """
        job_dicts = self._client.list_jobs(status=status, limit=limit, batch_size=batch_size)
        return tuple(job.Job(client=self._client, job_dict=job_dict) for job_dict in job_dicts)

    def get_current_calibration(self) -> calibration.Calibration:
        """Gets the most recent calbration via the API.

        Note that currently there is only one target, so this returns the calibration of that
        target.

        The calibration include device specification (number of qubits, connectivity), as well
        as fidelities and timings of gates.

        Returns:
            A `cirq_ionq.Calibration` containing the device specification and calibrations.

        Raises:
            IonQException: If there was an error accessing the API.
        """
        calibration_dict = self._client.get_current_calibration()
        return calibration.Calibration(calibration_dict=calibration_dict)

    def list_calibrations(
        self,
        start: datetime.datetime | None = None,
        end: datetime.datetime | None = None,
        limit: int = 100,
        batch_size: int = 1000,
    ) -> Sequence[calibration.Calibration]:
        """List calibrations via the API.

        Args:
            start: If supplied, only calibrations after this date and time. Accurate to seconds.
            end: If supplied, only calibrations before this date and time. Accurate to seconds.
            limit: The maximum number of calibrations to return.
            batch_size: The size of the batches requested per http GET call.

        Returns:
            A sequence of calibrations.

        Raises:
            IonQException: If there was an error accessing the API.
        """
        calibration_dicts = self._client.list_calibrations(
            start=start, end=end, limit=limit, batch_size=batch_size
        )
        return [calibration.Calibration(calibration_dict=c) for c in calibration_dicts]
