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
nanoprotogeny.ionq.ionqtetralemmatics
=======================================
IonQ Forte-specific compiled form of the tetralemmatic kernel.

Imports and re-exports everything from ``tetralemmatics.py``, then adds
the IonQ-native GPI/GPI2/ZZ sequences for the four basis-change gates
(B_LOG†, B_LOG, B_VIRT†, B_VIRT) and the Forte compiler function.

Architecture role
-----------------
This file is the **single IonQ-specific dependency** for the entire
tetralemmatic gate library.  All higher-level files (ionqBLOGgate.py,
ionqBVIRTgate.py, holographic.py, ionqcrossgates.py, ionqcurgate.py)
import their compiled basis sequences from here.

To target a different hardware platform, create an analogous file
(e.g. ``quantinuumtetralemmatics.py``) containing sequences compiled
for that platform's native gate set, and update the gate files to import
from it.  ``tetralemmatics.py`` requires no changes.

Compiled basis-change sequences
---------------------------------
All four matrices (B_LOG, B_LOG†, B_VIRT, B_VIRT†) decompose to exactly
**2 ZZ(0.25) gates** each — the KAK minimum given their interaction
coefficients (π/4, π/8, 0).  This was achieved by compiling directly via

    cirq.optimize_for_target_gateset(MatrixGate(B), ForteNativeGateset())

bypassing the CZ intermediate used by ``two_qubit_matrix_to_cz_operations``.
All phase values are exact multiples of 1/16.  Reconstruction error < 2e-15.

Native op counts vs. previous CZ-intermediate path:
    B_LOG    17  (was 54)     B_LOG†   20  (was 48)
    B_VIRT   20  (was 54)     B_VIRT†  20  (was 48)

Impact on higher-level gate costs (isolated compilation):
    ParamCoulombPhaseGate   217  (was 347)
    ZenoStabilizeGate       240  (was 578)

Forte compiler
--------------
``compile_tetralemmatic_ionq(circuit, target)`` targets either
``ForteNativeGateset`` (GPI/GPI2/ZZ) or ``IonQTargetGateset`` (API subset).

Dependencies
------------
nanoprotogeny.ionq.tetralemmatics  (all mathematical content)
cirq, cirq_ionq (IonQ Forte-specific)
"""

from __future__ import annotations

import numpy as np
import cirq
import cirq_ionq
from cirq_ionq.ionq_native_gates import GPIGate, GPI2Gate, ZZGate as _ZZGate
from cirq_ionq.ionq_native_target_gateset import ForteNativeGateset

# Re-export the entire platform-agnostic kernel so callers only need one import
from nanoprotogeny.ionq.tetralemmatics import *          # noqa: F401, F403
from nanoprotogeny.ionq.tetralemmatics import (          # explicit for type checkers
    B_LOG, B_VIRT,
    U_R_shift_onto, Z_clock_onto, DFT_onto,
    UR_phys_log, Z_phys_log, DFT_phys_log,
    UR_phys_virt, Z_phys_virt, DFT_phys_virt,
    DFT_phys_log_inv, DFT_phys_virt_inv,
    get_physical_matrix, apply_basis_ops,
    PhysicalURWrapper, PhysicalZClockWrapper,
    PhysicalDFTWrapper, InversePhysicalDFTWrapper,
    TetralemmaticIonURShiftGate, TetralemmaticIonZClockGate,
    TetralemmaticIonDFTGate, TetralemmaticIonInverseDFTGate,
    expand_qudit_circuit, TetralemmaticIonURgates, AnyQudit,
)

from nanoprotogeny.ionq.YB171PLUSHARDWARE import NomosIonQid, VirtualQudit

# ==============================================================================
# COMPILED BASIS-CHANGE SEQUENCES  (IonQ Forte GPI/GPI2/ZZ)
# ==============================================================================
# Each list holds (native_gate, qubit_index_tuple) entries where
# indices 0/1 select from a (hi, lo) qubit pair.
# Phases are exact multiples of 1/16 (turns).
# Usage:  yield from apply_basis_ops(B_LOG_DAG_OPS, q_hi, q_lo)

B_LOG_DAG_OPS = [
    (GPI2Gate(phi=-0.25),    (0,)),
    (GPI2Gate(phi=-0.5),     (1,)),
    (GPIGate (phi=-0.125),   (0,)),
    (GPIGate (phi=-0.5),     (1,)),
    (GPI2Gate(phi=0.75),     (0,)),
    (GPI2Gate(phi=0.5),      (1,)),
    (_ZZGate (theta=0.25),   (0, 1)),
    (GPI2Gate(phi=-0.5),     (0,)),
    (GPI2Gate(phi=0.0),      (1,)),
    (GPIGate (phi=-0.5),     (0,)),
    (GPIGate (phi=-0.0625),  (1,)),
    (GPI2Gate(phi=0.25),     (0,)),
    (GPI2Gate(phi=0.75),     (1,)),
    (_ZZGate (theta=0.25),   (0, 1)),
    (GPI2Gate(phi=-0.25),    (0,)),
    (GPI2Gate(phi=0.5),      (1,)),
    (GPIGate (phi=-0.375),   (0,)),
    (GPIGate (phi=0.3125),   (1,)),
    (GPI2Gate(phi=0.5),      (0,)),
    (GPI2Gate(phi=1.0),      (1,)),
]   # 20 native ops  (2 ZZ + 18 GPI/GPI2)

B_LOG_OPS = [
    (GPI2Gate(phi=0.0),      (1,)),
    (GPIGate (phi=0.1875),   (1,)),
    (GPI2Gate(phi=1.0),      (1,)),
    (_ZZGate (theta=0.25),   (0, 1)),
    (GPI2Gate(phi=0.0),      (0,)),
    (GPI2Gate(phi=0.25),     (1,)),
    (GPIGate (phi=0.0),      (0,)),
    (GPIGate (phi=0.1875),   (1,)),
    (GPI2Gate(phi=0.75),     (0,)),
    (GPI2Gate(phi=1.0),      (1,)),
    (_ZZGate (theta=0.25),   (0, 1)),
    (GPI2Gate(phi=0.0),      (0,)),
    (GPI2Gate(phi=-0.25),    (1,)),
    (GPIGate (phi=0.0),      (0,)),
    (GPIGate (phi=-0.25),    (1,)),
    (GPI2Gate(phi=0.75),     (0,)),
    (GPI2Gate(phi=0.25),     (1,)),
]   # 17 native ops  (2 ZZ + 15 GPI/GPI2)

B_VIRT_DAG_OPS = [
    (GPI2Gate(phi=0.5),      (0,)),
    (GPI2Gate(phi=0.5),      (1,)),
    (GPIGate (phi=0.125),    (0,)),
    (GPIGate (phi=0.125),    (1,)),
    (GPI2Gate(phi=0.5),      (0,)),
    (GPI2Gate(phi=0.5),      (1,)),
    (_ZZGate (theta=0.25),   (0, 1)),
    (GPI2Gate(phi=-0.5),     (0,)),
    (GPI2Gate(phi=-0.25),    (1,)),
    (GPIGate (phi=-0.5),     (0,)),
    (GPIGate (phi=-0.3125),  (1,)),
    (GPI2Gate(phi=0.25),     (0,)),
    (GPI2Gate(phi=0.5),      (1,)),
    (_ZZGate (theta=0.25),   (0, 1)),
    (GPI2Gate(phi=0.375),    (0,)),
    (GPI2Gate(phi=0.0),      (1,)),
    (GPIGate (phi=-0.0625),  (0,)),
    (GPIGate (phi=-0.375),   (1,)),
    (GPI2Gate(phi=0.25),     (0,)),
    (GPI2Gate(phi=0.0),      (1,)),
]   # 20 native ops  (2 ZZ + 18 GPI/GPI2)

B_VIRT_OPS = [
    (GPI2Gate(phi=-0.25),    (0,)),
    (GPI2Gate(phi=0.0),      (1,)),
    (GPIGate (phi=-0.125),   (0,)),
    (GPIGate (phi=0.125),    (1,)),
    (GPI2Gate(phi=0.75),     (0,)),
    (GPI2Gate(phi=1.0),      (1,)),
    (_ZZGate (theta=0.25),   (0, 1)),
    (GPI2Gate(phi=0.125),    (0,)),
    (GPI2Gate(phi=0.0),      (1,)),
    (GPIGate (phi=0.125),    (0,)),
    (GPIGate (phi=-0.0625),  (1,)),
    (GPI2Gate(phi=0.875),    (0,)),
    (GPI2Gate(phi=0.75),     (1,)),
    (_ZZGate (theta=0.25),   (0, 1)),
    (GPI2Gate(phi=0.375),    (0,)),
    (GPI2Gate(phi=0.0),      (1,)),
    (GPIGate (phi=0.0625),   (0,)),
    (GPIGate (phi=-0.125),   (1,)),
    (GPI2Gate(phi=0.5),      (0,)),
    (GPI2Gate(phi=0.5),      (1,)),
]   # 20 native ops  (2 ZZ + 18 GPI/GPI2)


# ==============================================================================
# PRE-COMPILED COULOMB PHASE SKELETON (AOT Template)
# ==============================================================================
# def coulomb_phase_skeleton_ops(phi: float, q0, q1, q2, q3) -> cirq.OP_TREE:
#     """
#     Pre-compiled, optimal skeleton for C^3Z(phi).
    
#     This implements the theoretical minimum 6-CNOT decomposition for a 
#     3-controlled Z-rotation. By yielding this fixed entanglement topology, 
#     we bypass generic multi-qubit synthesis heuristics. 
    
#     When processed by ForteNativeGateset with deep=True, the post-processors 
#     will perfectly merge adjacent single-qubit GPI/GPI2 rotations, achieving 
#     the absolute hardware limit (~60-70 native gates, down from ~217).
#     """
#     # Abstract 3-controlled phase gate
#     ctrl_phase = cirq.ControlledGate(
#         cirq.ZPowGate(exponent=phi / np.pi),
#         num_controls=3,
#         control_values=[1, 1, 1]
#     )
    
#     # Force decomposition into ≤2-qubit operations. 
#     # This yields the fixed, optimal 6-CNOT ladder.
#     yield from cirq.decompose(
#         ctrl_phase.on(q0, q1, q2, q3),
#         keep=lambda op: len(op.qubits) <= 2,
#         on_stuck_raise=False
#     )


# ==============================================================================
# FORTE COMPILER
# ==============================================================================
def compile_tetralemmatic_ionq(
    circuit: cirq.Circuit,
    target: str = "forte_native",
    simulation_mode: bool = False,
) -> cirq.Circuit:
    """Compile a tetralemmatic qudit circuit to IonQ Forte native gates.

    Parameters
    ----------
    circuit:
        Input circuit on NomosIonQid / VirtualQudit (or standard LineQubits).
    target:
        ``"forte_native"`` → GPI/GPI2/ZZ (pulse-level, 3-gate set).
        ``"api"``          → IonQ API 16-gate subset (cloud submission).
    simulation_mode:
        If True, return the circuit after qudit expansion without native
        compilation; preserves Physical*Wrapper gates for exact statevector
        simulation.

    Returns
    -------
    cirq.Circuit compiled to the selected target gateset.
    """
    if any(isinstance(q, (NomosIonQid, VirtualQudit))
           for q in circuit.all_qubits()):
        circuit = expand_qudit_circuit(circuit)

    if simulation_mode:
        return circuit

    if target == "forte_native":
        gateset = ForteNativeGateset()
    elif target == "api":
        gateset = cirq_ionq.IonQTargetGateset()
    else:
        raise ValueError("target must be 'forte_native' or 'api'")

    return cirq.optimize_for_target_gateset(
        circuit,
        gateset=gateset,
        context=cirq.TransformerContext(deep=True),
    )


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================
__all__ = [
    # IonQ-specific compiled sequences
    "B_LOG_DAG_OPS", "B_LOG_OPS",
    "B_VIRT_DAG_OPS", "B_VIRT_OPS",
    # IonQ-specific compiler
    "compile_tetralemmatic_ionq",
    # Re-exported from tetralemmatics (platform-agnostic)
    "B_LOG", "B_VIRT",
    "U_R_shift_onto", "Z_clock_onto", "DFT_onto",
    "UR_phys_log", "Z_phys_log", "DFT_phys_log",
    "UR_phys_virt", "Z_phys_virt", "DFT_phys_virt",
    "DFT_phys_log_inv", "DFT_phys_virt_inv",
    "get_physical_matrix", "apply_basis_ops",
    "PhysicalURWrapper", "PhysicalZClockWrapper",
    "PhysicalDFTWrapper", "InversePhysicalDFTWrapper",
    "TetralemmaticIonURShiftGate", "TetralemmaticIonZClockGate",
    "TetralemmaticIonDFTGate", "TetralemmaticIonInverseDFTGate",
    "expand_qudit_circuit", "TetralemmaticIonURgates", "AnyQudit",
]
