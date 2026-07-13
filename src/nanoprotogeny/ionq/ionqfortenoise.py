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
nanoprotogeny.ionq.ionqfortenoise
IonQ Forte 1 Hardware Noise Model for Qudit Simulations.

Implements a gate-type-aware, fully-symmetric depolarizing channel
calibrated to IonQ Forte 1 benchmark data. Supports both qudit (d≥2)
and qubit (d=2) registers natively via the Weyl (clock-shift) operator basis.
"""
from __future__ import annotations
import cirq
import numpy as np
from typing import List, Tuple, Dict, Optional, Union

# ── Calibration Parameters (IonQ Forte 1 Benchmarks) ──────────────────────
FORTE_NOISE_PARAMS: Dict[str, float] = {
    "p1q_error":    0.0026,
    "p2q_error":    0.0068,
    "p_meas_error": 0.0050,
    "p_idle_error": 0.00005,
}

USE_FORTE_NOISE_MODEL: bool = True
FALLBACK_DEPOL_P: float = 0.0068


class QuditDepolarizingChannel(cirq.Gate):
    r"""Generalized fully-symmetric depolarizing channel for a d-dimensional qudit.

    E(ρ) = (1 - p)ρ + p·(I/d)

    Implemented via the d² Weyl (clock-shift) operator basis {W_{a,b}}:
        K_0  = √(1 - p + p/d²) · I
        K_i  = √(p/d²)         · W_i   for i = 1, …, d²-1

    Σ K_i†K_i = I  ✓
    """

    def __init__(self, p: float, d: int) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p={p} must be in [0, 1].")
        self._p = p
        self._d = d
        self._weyl_ops = self._build_weyl_operators(d)

    @staticmethod
    def _build_weyl_operators(d: int) -> List[np.ndarray]:
        omega = np.exp(2j * np.pi / d)
        X = np.zeros((d, d), dtype=complex)
        for j in range(d):
            X[(j + 1) % d, j] = 1.0
        Z = np.diag([omega ** j for j in range(d)])
        ops = []
        for a in range(d):
            for b in range(d):
                ops.append(np.linalg.matrix_power(X, a) @ np.linalg.matrix_power(Z, b))
        return ops

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self._d,)

    def _kraus_(self) -> List[np.ndarray]:
        p, d = self._p, self._d
        k0 = np.sqrt(max(0.0, 1.0 - p + p / (d * d)))
        ki = np.sqrt(p / (d * d))
        return [k0 * self._weyl_ops[0]] + [ki * W for W in self._weyl_ops[1:]]

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=(f"ForteDep(p={self._p:.4f},d={self._d})",))

    def __repr__(self):
        return f"QuditDepolarizingChannel(p={self._p!r}, d={self._d!r})"


class ForteHardwareNoiseModel(cirq.NoiseModel):
    r"""Gate-type-aware noise model calibrated to IonQ Forte 1.

    Per operation type, injected after the gate moment:
        1Q gates      → QuditDepolarizingChannel(p=p1q, d) on the acting qudit
        2Q gates      → QuditDepolarizingChannel(p=p2q, d) on each qudit
        Measurement   → Depolarizing channel before measurement (SPAM model)
        Idle spectators → QuditDepolarizingChannel(p=p_idle, d) when others are active

    Zeno / VirtualQudit / Holographic gate classes are excluded (no physical pulse).
    """

    _VIRTUAL_GATE_SUBSTRINGS = frozenset({"Zeno", "VirtualQudit", "Holographic"})

    def __init__(
        self,
        p1q:    float = FORTE_NOISE_PARAMS["p1q_error"],
        p2q:    float = FORTE_NOISE_PARAMS["p2q_error"],
        p_meas: float = FORTE_NOISE_PARAMS["p_meas_error"],
        p_idle: float = FORTE_NOISE_PARAMS["p_idle_error"],
    ) -> None:
        self.p1q    = p1q
        self.p2q    = p2q
        self.p_meas = p_meas
        self.p_idle = p_idle

    def _is_virtual_op(self, op: cirq.Operation) -> bool:
        gate_name = type(op.gate).__name__ if op.gate is not None else ""
        return any(s in gate_name for s in self._VIRTUAL_GATE_SUBSTRINGS)

    def _depol_op(self, qubit: cirq.Qid, p: float) -> cirq.Operation:
        d = qubit.dimension
        if d == 2:
            return cirq.depolarize(min(1.0, 4.0 * p / 3.0)).on(qubit)
        return QuditDepolarizingChannel(p=p, d=d).on(qubit)

    def noisy_moment(
        self, moment: cirq.Moment, system_qubits: List[cirq.Qid]
    ) -> List[cirq.OP_TREE]:
        active_qubits: set = set()
        pre_meas_noise:  List[cirq.Operation] = []
        post_gate_noise: List[cirq.Operation] = []

        for op in moment.operations: 
            if op.gate is None or self._is_virtual_op(op):
                continue
            n = len(op.qubits)
            
            if isinstance(op.gate, cirq.MeasurementGate):
                for q in op.qubits:
                    pre_meas_noise.append(self._depol_op(q, self.p_meas))
                active_qubits.update(op.qubits)
            elif n == 1:
                post_gate_noise.append(self._depol_op(op.qubits[0], self.p1q))
                active_qubits.update(op.qubits)
            elif n == 2:
                for q in op.qubits:
                    post_gate_noise.append(self._depol_op(q, self.p2q))
                active_qubits.update(op.qubits)

        # Idle noise for qudits not involved in this moment
        if active_qubits and self.p_idle > 0.0:
            for q in set(system_qubits) - active_qubits:
                post_gate_noise.append(self._depol_op(q, self.p_idle))

        output: List[cirq.OP_TREE] = []
        if pre_meas_noise:
            output.append(cirq.Moment(pre_meas_noise))
        output.append(moment)
        if post_gate_noise:
            output.append(cirq.Moment(post_gate_noise))
        return output


def build_forte_noise_model(
    use_forte: bool = USE_FORTE_NOISE_MODEL,
    params: Optional[Dict[str, float]] = None,
) -> cirq.NoiseModel:
    """Return a Forte hardware noise model (or flat depolarising fallback)."""
    if not use_forte:
        print(f"[NOISE] Fallback: flat depolarising p={FALLBACK_DEPOL_P}")
        return cirq.ConstantQubitNoiseModel(cirq.depolarize(FALLBACK_DEPOL_P))
    
    effective_params = {**FORTE_NOISE_PARAMS, **(params or {})}
    model = ForteHardwareNoiseModel(
        p1q=effective_params["p1q_error"],
        p2q=effective_params["p2q_error"],
        p_meas=effective_params["p_meas_error"],
        p_idle=effective_params["p_idle_error"],
    )
    print(
        f"[NOISE] IonQ Forte 1 hardware noise model: "
        f"p1Q={model.p1q:.4f}, p2Q={model.p2q:.4f}, "
        f"pSPAM={model.p_meas:.4f}, pIdle={model.p_idle:.5f}"
    )
    return model

__all__ = [
    "FORTE_NOISE_PARAMS",
    "USE_FORTE_NOISE_MODEL",
    "FALLBACK_DEPOL_P",
    "QuditDepolarizingChannel",
    "ForteHardwareNoiseModel",
    "build_forte_noise_model",
]