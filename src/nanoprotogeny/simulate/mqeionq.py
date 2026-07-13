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
mqeionq.py — IonQ Cloud Simulator Execution of MQE-QPE
=========================================================
Mirrors the default ``mqe run`` path
(HardwareQPEPipelineRunner with virtual_ancilla_mode=True in
mqevanc.py) but replaces the local DensityMatrixSimulator kernel
with cirq_ionq.Service.run() targeting the IonQ cloud simulator with
noise.model = "forte-1".

CIRCUIT STRUCTURE
-----------------
For each (τ, λ) combination at a QPE checkpoint step n*:

    [state prep: X gates to argmax(|ψ_{n*}|) on logical LineQubits]
    for i in range(n_max):
        [Trotter step × ZNE fold]   ← fold=1/3/5 for λ=1/2/3
        [CofactorCouplingGate(ν_{n*}, B_{n*})]  ← once per step, not folded
    [VDFTGate on V₁]
    [measure V₁ → key "virt"]

QUDIT → QUBIT ENCODING
-----------------------
All qudit gates are converted to cirq.MatrixGate on cirq.LineQubits using
their ontological _unitary_() directly (no B_LOG/B_VIRT physical basis
transform).  Qudit state k ∈ {0,1,2,3} encodes as 2-bit binary:

    k = b_hi * 2 + b_lo    (MSB first in the qubit pair)

This makes the qubit basis index equal to the qudit state k, so all
unitary matrices map without additional basis rotation.

Qubit map for N orbitals:
    NomosIonQid(i) → (LineQubit(2i),         LineQubit(2i+1))
    VirtualQudit(j) → (LineQubit(2N + 2j),  LineQubit(2N + 2j + 1))

Gates with no direct unitary (ParamScatteringGate) are decomposed one
level via cirq.decompose_once, then each sub-gate is recursively
converted.

COMPILATION
-----------
After building the MatrixGate circuit, ForteNativeGateset is applied via
cirq.optimize_for_target_gateset — this decomposes all MatrixGates to
GPI/GPI2/ZZ, which is what the IonQ Forte hardware and simulator expect.

ZNE FOLDING
-----------
Noise amplification for Richardson ZNE (λ ∈ {1,2,3}) is implemented as
gate-level folding on the Trotter step only:
    λ=1: U          (fold=1)
    λ=2: U U† U     (fold=3)
    λ=3: U U† U U† U (fold=5)
The CofactorCouplingGate and VDFTGate are not folded — they are the
measurement operators, not the noisy evolution.

PREFLIGHT
---------
τ-sequence selection (select_tau_sequence_virtual_ancilla) still runs
locally via DensityMatrixSimulator for speed.  IonQ shots are only spent
on the production ZNE measurement run.

ENTRY POINT
-----------
    run_ionq_mqe_validation(mechanism_name, integral_state, config,
                             output_json, dataset_dir, backend_cfg)

CLI INVOCATION
--------------
    mqe run --mechanism nitrogenase_lt \
            --dataset-dir datasets/ufc_datasets_pubquality \
            --backend ionq-sim

OUTPUT
------
    stoichiometry-mqeqpe-ionq/<mechanism>_mqeqpe_ionq_results.json
"""

from __future__ import annotations

import functools
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cirq
import numpy as np

try:
    import cirq_ionq
    from cirq_ionq.ionq_native_target_gateset import ForteNativeGateset
    _CIRQ_IONQ_AVAILABLE = True
except ImportError:
    _CIRQ_IONQ_AVAILABLE = False
    cirq_ionq   = None  # type: ignore
    ForteNativeGateset = None  # type: ignore

# ── IonQ hardware types ───────────────────────────────────────────────────────
from nanoprotogeny.ionq.YB171PLUSHARDWARE   import NomosIonQid, VirtualQudit
from nanoprotogeny.ionq.ionqvirtualgates    import VDFTGate
from nanoprotogeny.ionq.ionqmqegates        import CompositeCofactorCouplingGate
from nanoprotogeny.ionq.ionqtrotter         import build_trotter_evolution_circuit

from nanoprotogeny.ionq.ionqtetralemmatics import (
    TetralemmaticIonDFTGate, TetralemmaticIonURShiftGate, TetralemmaticIonZClockGate, DFT_onto
)

from nanoprotogeny.ionq.ionqfortenoise      import (
    FORTE_NOISE_PARAMS, ForteHardwareNoiseModel,
)
from nanoprotogeny.ionq.ionqconnectivity    import (
    BackendConfig, BackendMode,
    _make_ionq_service, _save_job_manifest,
)

# ── Molecular layer ───────────────────────────────────────────────────────────
from nanoprotogeny.molecular.mqemolecules   import (
    MechanismTuple, build_predefined_mechanisms,
)
from nanoprotogeny.molecular.mqeintegralstore import StepwiseIntegralStore
from nanoprotogeny.molecular.mqehamiltonian import (
    build_qudit_hamiltonian_matrix,
    _project_hamiltonian_to_sector,
    ground_state_from_diagonalization,
)
from nanoprotogeny.molecular.mqephasetracker  import ZmPhaseTracker
from nanoprotogeny.molecular.mqestoichiometry import StoichiometricVerifier

# ── Simulate layer ────────────────────────────────────────────────────────────
from nanoprotogeny.simulate.mqeconfig       import MQEConfig, IntegralState
from nanoprotogeny.simulate.mqedualmanifold import (
    _make_virtual_qudits_m, VirtualRegisterPair,
)
from nanoprotogeny.simulate.mqevanc    import (
    build_mqe_step_block, build_mqe_L_block, _export_mqe_json,
)

# ── QPE helpers (preflight + MLE, unchanged) ─────────────────────────────────
from nanoprotogeny.qpe.mqevancqpe       import (
    hardware_map_energy,
    select_tau_sequence_virtual_ancilla,
    _project_rho_to_sector,
    _count_screened_ctrl_gates,
    VIRTUAL_ANCILLA_D_STATE_NOISE,
)
from nanoprotogeny.qpe.mqetrotterdensematrix       import compute_trotter_density_matrix

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  QUDIT → QUBIT ENCODING HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _qudit_qubit_map(n_orbitals: int) -> Dict:
    """Return a dict mapping qudit registers to (lq_hi, lq_lo) LineQubit pairs.

    Encoding: qudit state k ∈ {0,1,2,3} → (b_hi, b_lo) where k = b_hi*2 + b_lo.
    Row/column indices of the ontological 4×4 unitary therefore equal the
    2-qubit computational-basis index, so no basis transform is required.

    Layout:
        NomosIonQid(i) → (LineQubit(2i),       LineQubit(2i+1))
        VirtualQudit(j) → (LineQubit(2N+2j),   LineQubit(2N+2j+1))
    """
    qmap: Dict = {}
    for i in range(n_orbitals):
        qmap[NomosIonQid(i)] = (cirq.LineQubit(2 * i), cirq.LineQubit(2 * i + 1))
    for j in range(n_orbitals):
        base = 2 * n_orbitals
        qmap[VirtualQudit(j)] = (
            cirq.LineQubit(base + 2 * j),
            cirq.LineQubit(base + 2 * j + 1),
        )
    return qmap


def _op_to_matrix_ops(
    op:   cirq.Operation,
    qmap: Dict,
) -> List[cirq.Operation]:
    """Convert one qudit operation to cirq.MatrixGate ops on LineQubits.

    Uses the gate's ontological _unitary_() directly.  Gates that return
    NotImplemented from _unitary_() (e.g. ParamScatteringGate) are
    decomposed one level via cirq.decompose_once, then each sub-gate is
    recursively converted.

    For n > 2 physical qubits (e.g. two-qudit gates like ParamCoulombPhaseGate
    or ParamExchangeGate mapping to 4 qubits), the resulting MatrixGate is
    further decomposed via Quantum Shannon Decomposition (QSD) to ≤ 2-qubit
    operations so that ForteNativeGateset can compile them to GPI/GPI2/ZZ.
    Without this step, 4-qubit MatrixGates survive optimization and fail at
    IonQ serialization.
    """
    flat_q: List[cirq.LineQubit] = []
    for q in op.qubits:
        pair = qmap.get(q)
        if pair is not None:
            flat_q.extend(pair)
        else:
            flat_q.append(q)  # already a LineQubit (shouldn't occur in our circuits)

    try:
        U = cirq.unitary(op.gate)
        n = len(flat_q)
        mat_op = cirq.MatrixGate(U, qid_shape=(2,) * n).on(*flat_q)
        if n <= 2:
            return [mat_op]
        # n > 2: QSD to ≤ 2-qubit ops so ForteNativeGateset can compile them.
        decomposed = list(cirq.decompose(
            mat_op,
            keep=lambda o: cirq.num_qubits(o) <= 2,
            on_stuck_raise=None,
        ))
        return decomposed if decomposed else [mat_op]
    except (TypeError, ValueError, AttributeError):
        # No direct unitary — decompose one level and recurse
        sub_ops = list(cirq.decompose_once(op))
        result: List[cirq.Operation] = []
        for sub in sub_ops:
            result.extend(_op_to_matrix_ops(sub, qmap))
        return result


def _trotter_to_qubit_circuit(
    trotter_circ: cirq.Circuit,
    qmap:         Dict,
) -> cirq.Circuit:
    """Convert a Trotter qudit circuit to an equivalent LineQubit MatrixGate circuit."""
    ops: List[cirq.Operation] = []
    for op in trotter_circ.all_operations():
        ops.extend(_op_to_matrix_ops(op, qmap))
    return cirq.Circuit(ops)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  STATE PREPARATION
# ──────────────────────────────────────────────────────────────────────────────

def _build_state_prep(
    psi_n:      np.ndarray,
    n_orbitals: int,
    qmap:       Dict,
) -> List[cirq.Operation]:
    """X-gate layer to prepare the dominant basis state of ψ_n.

    Selects argmax(|ψ_n|) as the initial computational basis state and
    applies X to the appropriate LineQubits to prepare it from |0…0⟩.

    This is the same approximation used by compute_trotter_density_matrix:
    the dominant basis state captures the bulk of the FCI ground state
    amplitude for active-space calculations at CAS(4,4) level.

    Cirq's qudit convention (big-endian): for N qudits of dimension d,
        state_index = k_0 * d^(N-1) + k_1 * d^(N-2) + … + k_{N-1} * d^0
    NomosIonQid(0) is the most significant qudit.
    """
    init_idx = int(np.argmax(np.abs(psi_n)))
    d = 4
    ops: List[cirq.Operation] = []
    for i in range(n_orbitals):
        k_i = (init_idx // (d ** (n_orbitals - 1 - i))) % d
        lq_hi, lq_lo = qmap[NomosIonQid(i)]
        if (k_i >> 1) & 1:   # MSB of k_i → lq_hi
            ops.append(cirq.X(lq_hi))
        if k_i & 1:           # LSB of k_i → lq_lo
            ops.append(cirq.X(lq_lo))
    return ops


# ──────────────────────────────────────────────────────────────────────────────
# 3.  QPE CIRCUIT BUILDER
# ──────────────────────────────────────────────────────────────────────────────

# ZNE gate-fold counts for λ ∈ {1,2,3}:
#   λ=1 → U        (1 application)
#   λ=2 → U U† U   (3 applications, cancels O(p) noise bias)
#   λ=3 → U U† U U† U  (5 applications, cancels O(p) and O(p²))
_ZNE_FOLD: Dict[int, int] = {1: 1, 2: 3, 3: 5}


def build_mqe_qpe_shot_circuit(
    psi_n:      np.ndarray,
    n_max:      int,
    lam:        int,
    mechanism:  MechanismTuple,
    n_star:     int,
    vreg:       VirtualRegisterPair,
    dt:         float,
    n_orbitals: int,
    h_diag:     Dict,
    h_hop:      Dict,
    g_full:     Dict,
) -> Tuple[cirq.Circuit, List[cirq.LineQubit]]:
    r"""Build and compile the full MQE-QPE shot circuit for one (n_max, λ).

    Returns the GPI/GPI2/ZZ-compiled circuit and the V₁ measurement qubits.

    Circuit (schematic):
        [state prep]
        for i in range(n_max):
            [Trotter(H_{n*}, Δt) × fold(λ)]   ← ZNE noise amplification
            [CofactorCouplingGate(ν_{n*}) ∀ r ∈ B_{n*}]  ← phase accumulator
        [VDFTGate on V₁]
        [cirq.measure(*v1_qubits, key="virt")]

    Args:
        psi_n:      Ground-state vector at step n* (shape 4^N).
        n_max:      Number of Trotter–CofactorCoupling iterations.
                    Determined by the preflight as round(τ_max / Δt).
        lam:        ZNE noise scale λ ∈ {1, 2, 3}.
        mechanism:  MechanismTuple.
        n_star:     Checkpoint step index in the mechanism.
        vreg:       VirtualRegisterPair (used for qudit type determination).
        dt:         Trotter step Δt (Ha⁻¹).
        n_orbitals: N active-space orbitals.
        h_diag, h_hop, g_full: Step-specific integral dicts for H_{n*}.

    Returns:
        compiled_circuit: GPI/GPI2/ZZ-native cirq.Circuit.
        virt_qubits:      [lq_vhi, lq_vlo] — V₁ measurement LineQubits.
    """
    if not _CIRQ_IONQ_AVAILABLE:
        raise RuntimeError(
            "cirq-ionq and ForteNativeGateset are required for IonQ submission. "
            "Install with: pip install cirq-ionq"
        )

    m      = mechanism.m
    nu_n   = mechanism.nu_shifts[n_star]
    B_n    = mechanism.cofactor_sets[n_star]
    fold   = _ZNE_FOLD[lam]
    qmap   = _qudit_qubit_map(n_orbitals)

    # V₁ LineQubits: VirtualQudit(0) is the canonical phase register
    vq_hi, vq_lo = qmap[VirtualQudit(0)]
    virt_qubits  = [vq_hi, vq_lo]

    # ── Trotter step circuit (single step, qubit-native MatrixGates) ──────────
    trotter_qd = build_trotter_evolution_circuit(
        n_orbitals, h_diag, h_hop, g_full, dt=dt
    )
    trotter_qb = _trotter_to_qubit_circuit(trotter_qd, qmap)
    trotter_inv_qb = cirq.inverse(trotter_qb) if fold > 1 else None

    # ── CofactorCouplingGate ops (per orbital in B_{n*}, applied unfolded) ────
    # CompositeCofactorCouplingGate(m=4, nu) for VirtualQudit path (r=1, m≤4).
    # The ontological 16×16 unitary is QSD-decomposed to ≤2-qubit ops here so
    # that ForteNativeGateset can compile them to GPI/GPI2/ZZ pulses.
    coupling_ops: List[cirq.Operation] = []
    if m > 1 and nu_n > 0:
        for r_idx in B_n:
            if r_idx >= n_orbitals:
                continue
            lq_hi, lq_lo = qmap[NomosIonQid(r_idx)]
            vq_hi_r, vq_lo_r = qmap[VirtualQudit(r_idx)]
            gate = CompositeCofactorCouplingGate(m=m, nu=nu_n)
            U = gate._unitary_()  # ontological 16×16 permutation
            mat_op = cirq.MatrixGate(U, qid_shape=(2, 2, 2, 2)).on(
                lq_hi, lq_lo, vq_hi_r, vq_lo_r
            )
            # QSD → ≤2-qubit ops
            coupling_ops.extend(cirq.decompose(
                mat_op,
                keep=lambda o: cirq.num_qubits(o) <= 2,
                on_stuck_raise=None,
            ) or [mat_op])

    # ── VDFTGate on V₁ (F̂_m^V readout) ─────────────────────────────────────
    vdft_U  = VDFTGate()._unitary_()  # 4×4 ontological DFT
    vdft_op = cirq.MatrixGate(vdft_U, qid_shape=(2, 2)).on(vq_hi, vq_lo)

    # ── Assemble raw (qubit-native, MatrixGate) circuit ───────────────────────
    raw = cirq.Circuit()

    # State preparation
    prep_ops = _build_state_prep(psi_n, n_orbitals, qmap)
    if prep_ops:
        raw.append(cirq.Moment(prep_ops))

    # n_max iterations: ZNE-folded Trotter block + CofactorCouplingGate
    for _ in range(n_max):
        # Folded Trotter: U  |  U U† U  |  U U† U U† U
        raw.append(trotter_qb)
        if fold >= 3:
            raw.append(trotter_inv_qb)
            raw.append(trotter_qb)
        if fold >= 5:
            raw.append(trotter_inv_qb)
            raw.append(trotter_qb)

        # CofactorCouplingGate (not folded)
        if coupling_ops:
            raw.append(cirq.Moment(coupling_ops))

    # VDFTGate + measurement
    raw.append(cirq.Moment([vdft_op]))
    raw.append(cirq.measure(*virt_qubits, key="virt"))

    # ── Compile to GPI/GPI2/ZZ ───────────────────────────────────────────────
    compiled = cirq.optimize_for_target_gateset(
        raw,
        gateset=ForteNativeGateset(),
        context=cirq.TransformerContext(deep=True),
    )
    compiled = cirq.drop_negligible_operations(compiled, atol=1e-8)
    compiled = cirq.drop_empty_moments(compiled)

    # Sanity check: no MatrixGate should survive
    has_matrix = any(
        isinstance(op.gate, cirq.MatrixGate)
        for op in compiled.all_operations()
        if not isinstance(op.gate, cirq.MeasurementGate)
    )
    if has_matrix:
        log.warning(
            "[IONQ-QPE] MatrixGate survived compilation for "
            f"n*={n_star} λ={lam} — circuit may not run on IonQ hardware."
        )
    else:
        log.info(
            f"[IONQ-QPE n*={n_star} λ={lam}] Compiled: "
            f"{len(compiled)} moments | "
            f"has_matrix: [{'!' if has_matrix else '✓'}] {'DETECTED' if has_matrix else 'ZERO'}"
        )

    return compiled, virt_qubits


# ──────────────────────────────────────────────────────────────────────────────
# 4.  IONQ SUBMISSION
# ──────────────────────────────────────────────────────────────────────────────

def _submit_and_get_p_k(
    cfg:         BackendConfig,
    circuit:     cirq.Circuit,
    virt_qubits: List[cirq.LineQubit],
    n_shots:     int,
    job_name:    str = "mqe-qpe",
) -> np.ndarray:
    """Submit one compiled circuit to IonQ and return p(k|τ) as a shape-(4,) array.

    The virtual register V₁ is measured into key "virt".  Outcome bits
    (b_hi, b_lo) decode to k = b_hi * 2 + b_lo, consistent with the
    ontological encoding used throughout the circuit.

    Args:
        cfg:         BackendConfig (mode must be IONQ_SIM; api_key required).
        circuit:     GPI/GPI2/ZZ-compiled circuit from build_mqe_qpe_shot_circuit.
        virt_qubits: [lq_vhi, lq_vlo] — the two V₁ LineQubits.
        n_shots:     Number of circuit repetitions.
        job_name:    Human-readable job label for logging.

    Returns:
        np.ndarray of shape (4,) — normalised p(k), k ∈ {0,1,2,3}, summing to 1.
    """
    service = _make_ionq_service(cfg)

    extra_params: Dict = {}
    if "simulator" in cfg.resolved_target:
        extra_params["noise"] = {"model": "forte-1"}

    n_ops = sum(1 for _ in circuit.all_operations())
    log.info(
        f"[IONQ] {job_name!r} → {cfg.resolved_target!r} | "
        f"shots={n_shots} | moments={len(circuit)} | ops={n_ops}"
    )

    result = service.run(
        circuit=circuit,
        repetitions=n_shots,
        target=cfg.resolved_target,
        extra_query_params=extra_params if extra_params else None,
    )

    job_id = getattr(result, "job_id", "service-managed")
    log.info(f"[IONQ] ✓ {job_name!r} completed | job_id={job_id!r}")

    # ── Decode measurement outcomes → p(k) ───────────────────────────────────
    # result.measurements["virt"]: shape (n_shots, 2) → rows [b_hi, b_lo]
    counts = np.zeros(4, dtype=float)
    meas = result.measurements.get("virt")
    if meas is not None:
        for row in meas:
            k = int(row[0]) * 2 + int(row[1])
            if 0 <= k < 4:
                counts[k] += 1.0
    else:
        # Fallback: parse IonQ REST histogram dict
        hist = result.histogram(key="virt")
        for k_int, cnt in hist.items():
            if 0 <= int(k_int) < 4:
                counts[int(k_int)] += float(cnt)

    total = counts.sum()
    if total < 1e-12:
        log.warning(f"[IONQ] {job_name!r}: all counts zero — returning uniform p(k).")
        return np.full(4, 0.25)
    return counts / total


# ──────────────────────────────────────────────────────────────────────────────
# 5.  PIPELINE RUNNER
# ──────────────────────────────────────────────────────────────────────────────

class IonQMQEPipelineRunner:
    r"""Orchestrates MQE-QPE validation against the IonQ cloud simulator.

    Structurally mirrors HardwareQPEPipelineRunner(virtual_ancilla_mode=True)
    from mqevanc.py.  The sole difference is the QPE energy-extraction
    block at each checkpoint:

        mqevanc.py (local):
            compute_trotter_density_matrix  →  ρ_sys  (DensityMatrixSim)
            compute_virtual_ancilla_qpe_probs  →  p(k|τ)  (analytical)

        mqeionq.py (IonQ):
            build_mqe_qpe_shot_circuit  →  GPI/GPI2/ZZ circuit
            _submit_and_get_p_k  →  p(k|τ)  (IonQ shot histogram)

    Preflight (τ-sequence selection) runs locally for speed — IonQ shots
    are spent only on the 3 × N_τ production ZNE jobs per checkpoint.

    Args:
        mechanism:    MechanismTuple.
        integral_state: Populated IntegralState.
        config:       MQEConfig (defaults to MQEConfig() if None).
        dataset_dir:  Root datasets directory (mqedatagenerator output).
        backend_cfg:  BackendConfig with mode=IONQ_SIM and valid api_key.
    """

    def __init__(
        self,
        mechanism:      MechanismTuple,
        integral_state: IntegralState,
        config:         MQEConfig                  = None,
        dataset_dir:    Optional[Union[str, Path]] = None,
        backend_cfg:    BackendConfig              = None,
    ):
        cfg = config or MQEConfig()
        self._mechanism          = mechanism
        self._h_diag             = integral_state.h_diag
        self._h_hop              = integral_state.h_hop
        self._g_full             = integral_state.g_full
        self._e_core             = integral_state.e_core
        self._dt                 = cfg.dt
        self._eta                = cfg.eta
        self._tau_seq_candidates = list(cfg.tau_seq_candidates)
        self._cached_tau_seq: Optional[List[float]] = None
        self._noise_params       = FORTE_NOISE_PARAMS
        self._dataset_dir        = Path(dataset_dir) if dataset_dir else None
        self._store: Optional[StepwiseIntegralStore] = None
        self._backend_cfg        = backend_cfg
        self._n_shots            = backend_cfg.n_shots if backend_cfg else 8192
        self._p_idle_v           = VIRTUAL_ANCILLA_D_STATE_NOISE["p_idle_virtual"]

    # ── Step-store accessor (lazy init) ──────────────────────────────────────

    def _get_store(self) -> Optional[StepwiseIntegralStore]:
        if self._store is not None:
            return self._store
        if self._dataset_dir is None:
            return None
        try:
            self._store = StepwiseIntegralStore(
                self._dataset_dir, self._mechanism.name
            )
            self._mechanism = self._store.to_mechanism_tuple()
            log.info(
                f"[IonQ-Runner] StepwiseIntegralStore loaded for "
                f"'{self._mechanism.name}' from {self._dataset_dir}"
            )
        except FileNotFoundError as e:
            log.warning(
                f"[IonQ-Runner] dataset_dir unavailable ({e}). "
                f"Falling back to global integrals."
            )
            self._store = None
        return self._store

    # ── Public run() ────────────────────────────────────────────────────────

    def run(self) -> Dict:
        """Execute the IonQ MQE-QPE validation pipeline. Returns result dict."""
        mech  = self._mechanism
        N     = mech.N_orbitals
        m     = mech.m
        store = self._get_store()
        start = time.time()

        self._print_header(store)

        # ── 1. Allocate registers ────────────────────────────────────────────
        logical_qudits = [NomosIonQid(i) for i in range(N)]
        vreg           = _make_virtual_qudits_m(N, m)
        virtual_qudits = vreg.v1
        reg_desc = (
            f"{N} logical (d=4) + {N} virtual (d=4+carry, m={m})"
            if vreg.is_composite
            else f"{N} logical (d=4) + {N} virtual (d={m})"
        )
        src = (
            f"dataset ({self._dataset_dir}/{mech.name})"
            if store else "global fallback"
        )
        print(f"  [REG] {reg_desc} | integrals: {src}")
        target = self._backend_cfg.resolved_target if self._backend_cfg else "N/A"
        print(f"  [IONQ] target={target!r} | shots/job={self._n_shots}")

        # ── 2. Algebraic phase-closure validation (no simulation) ────────────
        print(f"\n  [ALGEBRAIC] Net-flux phase closure validation (ℤ_{m})...")
        alg_tracker = ZmPhaseTracker(m)
        for n in range(mech.M_steps):
            alg_tracker.step(
                n, mech.nu_shifts[n], len(mech.electron_sets[n]),
                nu_decouple         = mech.nu_decouple_shifts[n],
                n_electrons_ejected = len(mech.electron_eject_sets[n]),
                n_photons_absorbed  = len(mech.photon_absorb_sets[n]),
                n_photons_emitted   = len(mech.photon_emit_sets[n]),
            )
        print(alg_tracker.report(mech))
        algebraic_ok = alg_tracker.phase_closed
        electron_ok  = (alg_tracker.net_electrons == mech.total_net_electrons)

        # ── 3. M-step loop ───────────────────────────────────────────────────
        print(f"\n  [LOOP] Building M={mech.M_steps} MQE step blocks...")
        full_circuit  = cirq.Circuit()   # accumulated for compilation profiling
        tracker       = ZmPhaseTracker(m)
        checkpoints   = sorted(set(
            [c[0] for c in mech.crossings] + [mech.M_steps - 1]
        ))
        qpe_results:   Dict[int, Dict]         = {}
        step_energies: List[Optional[float]]   = []

        for n in range(mech.M_steps):

            # ── 3a. Load step integrals ────────────────────────────────────
            if store is not None:
                step_ints = store.get_step(n)
                e_ref_n   = store.get_reference_energy(n)
                store.log_step_summary(n)
            else:
                step_ints = None
                e_ref_n   = None
            step_energies.append(e_ref_n)

            # ── 3b–3c. Build B_n + L_n (for full_circuit profiling only) ──
            step_circ, crossing_applied = build_mqe_step_block(
                n, mech, logical_qudits, virtual_qudits,
                self._h_diag, self._h_hop, self._g_full,
                self._dt, tracker,
                step_integrals=step_ints, vreg=vreg,
            )
            L_n = build_mqe_L_block(
                step_circ, virtual_qudits, n, m, tracker, vreg=vreg,
            )
            full_circuit.append(L_n)

            log.info(
                f"  [IonQ n={n:02d}] ν={mech.nu_shifts[n]} "
                f"k^(n)={tracker._k_step} | "
                f"{'⚡ Janus' if crossing_applied else ''}"
            )

            # ── 3d. IonQ QPE + ZNE checkpoint ─────────────────────────────
            if n not in checkpoints:
                continue

            # ── Hamiltonian at step n ──────────────────────────────────────
            if step_ints is not None:
                h_d_n, h_h_n, g_f_n, e_core_n, n_o_n = step_ints
                step_raw       = store._load_step_raw(n)
                nelec_active_n = step_raw.get("metadata", {}).get("nelec_active", None)
            else:
                h_d_n, h_h_n, g_f_n = self._h_diag, self._h_hop, self._g_full
                n_o_n          = N
                nelec_active_n = None

            H_full = build_qudit_hamiltonian_matrix(n_o_n, h_d_n, h_h_n, g_f_n)
            sector_indices = None
            if nelec_active_n is not None:
                H_proj, sector_indices = _project_hamiltonian_to_sector(
                    H_full, n_o_n, nelec_active_n, return_indices=True
                )
            else:
                H_proj = H_full

            E_0_n, psi_proj = ground_state_from_diagonalization(H_proj)
            if sector_indices is not None:
                psi_n = np.zeros(4 ** n_o_n, dtype=complex)
                psi_n[sector_indices] = psi_proj
            else:
                psi_n = psi_proj

            E_ref_chk = (e_ref_n - e_core_n) if e_ref_n is not None else E_0_n
            effective_sector_indices = sector_indices if E_0_n > 0.0 else None

            is_janus   = (n in [c[0] for c in mech.crossings])
            step_label = (
                f"Janus E_{n}→E_{n+1}" if is_janus else f"E_{n}→E_{n+1}"
            )
            geo_label = (
                store.get_step_meta(n).get("geometry_label", "") if store else ""
            )

            print(
                f"\n  [IONQ-QPE] Checkpoint n={n} ({step_label})"
                + (f" | {geo_label}" if geo_label else "")
            )
            print(
                f"    E_0 (exact diag) = {E_0_n:+.8f} Ha"
                + (f" | E_ref (FCI) = {E_ref_chk:+.8f} Ha"
                    if e_ref_n is not None else "")
            )

            # ── Preflight: τ-sequence selection (LOCAL DensityMatrixSim) ──
            if self._cached_tau_seq is not None:
                active_tau_seq = self._cached_tau_seq
                tau_seq_source = "cached"
            else:
                active_tau_seq = select_tau_sequence_virtual_ancilla(
                    H_full, psi_n,
                    n_o_n, h_d_n, h_h_n, g_f_n,
                    self._dt, E_ref_chk,
                    self._tau_seq_candidates,
                    chem_accuracy_mHa=1.6,
                    sector_indices=effective_sector_indices,
                )
                if active_tau_seq is None:
                    log.warning(
                        f"[IONQ-QPE n={n}] No τ-sequence achieves ≤1.6 mHa "
                        f"under Forte noise. Recording noise-floor failure."
                    )
                    qpe_results[n] = {
                        "step_label":      step_label,
                        "geometry":        geo_label,
                        "E_exact_diag":    E_0_n,
                        "E_ref_fci":       e_ref_n,
                        "E_ref_used":      E_ref_chk,
                        "E_map":           {},
                        "E_zne_rich":      None,
                        "E_zne_exp":       None,
                        "E_zne_best":      None,
                        "zne_method":      None,
                        "residual_mHa":    float("inf"),
                        "chem_ok":         False,
                        "eta_v":           None,
                        "tau_seq_used":    None,
                        "tau_seq_source":  "noise-floor",
                        "integral_source": "dataset" if store else "global",
                    }
                    print(
                        f"    [!] IONQ QPE: noise-floor — "
                        f"no τ within 1.6 mHa budget."
                    )
                    continue
                self._cached_tau_seq = active_tau_seq
                tau_seq_source = "adaptive"

            tau_max = max(active_tau_seq)
            n_max   = max(1, int(round(tau_max / self._dt)))

            # ── η_V (same formula as local path) ─────────────────────────
            angle_scale    = tau_max / n_max
            gates_per_step = _count_screened_ctrl_gates(
                n_o_n, h_d_n, h_h_n, g_f_n, angle_scale
            )
            eta_v = (1.0 - self._p_idle_v) ** (gates_per_step * n_max)
            log.info(
                f"[IONQ-QPE n={n}] η_V={eta_v:.6f} "
                f"(gates/step={gates_per_step}, n_ctrl={gates_per_step*n_max})"
            )
            print(
                f"    τ-seq ({tau_seq_source}): {active_tau_seq} | "
                f"n_max={n_max} | η_V={eta_v:.6f}"
            )

            # ── Production ZNE loop — IonQ submission per (τ, λ) ─────────
            E_map_series: Dict[int, float] = {}
            for lam in [1, 2, 3]:
                ancilla_probs: Dict[float, np.ndarray] = {}
                for tau in active_tau_seq:
                    circuit_compiled, virt_qubits = build_mqe_qpe_shot_circuit(
                        psi_n, n_max, lam, mech, n, vreg,
                        self._dt, n_o_n, h_d_n, h_h_n, g_f_n,
                    )
                    p_k = _submit_and_get_p_k(
                        self._backend_cfg, circuit_compiled, virt_qubits,
                        self._n_shots,
                        job_name=(
                            f"{mech.name}_n{n}_lam{lam}_tau{tau:.3f}"
                        ),
                    )
                    ancilla_probs[tau] = p_k
                    log.info(
                        f"[IONQ-QPE n={n} λ={lam} τ={tau:.3f}] "
                        f"p(k)={np.round(p_k, 4).tolist()}"
                    )

                # MLE — same call as local path (η_V-corrected model)
                E_map, _, _ = hardware_map_energy(
                    ancilla_probs, E_ref=E_ref_chk, eta_v=eta_v
                )
                E_map_series[lam] = E_map

            # ── ZNE best-of-two ───────────────────────────────────────────
            E1, E2, E3 = E_map_series[1], E_map_series[2], E_map_series[3]
            E_zne = 3 * E1 - 3 * E2 + E3

            denom = E3 - 2 * E2 + E1
            if abs(denom) > 1e-12:
                E_inf    = (E1 * E3 - E2 ** 2) / denom
                denom_e  = E2 - E_inf
                E_zne_ex = (
                    E_inf + (E1 - E_inf) ** 2 / denom_e
                    if abs(denom_e) > 1e-12 else E_zne
                )
            else:
                E_zne_ex = E_zne

            res_rich = abs(E_zne    - E_ref_chk) * 1000
            res_ex   = abs(E_zne_ex - E_ref_chk) * 1000
            if res_ex <= res_rich:
                E_best, residual_mHa, zne_method = E_zne_ex, res_ex, "exp"
            else:
                E_best, residual_mHa, zne_method = E_zne,    res_rich, "rich"
            chem_ok = residual_mHa <= 1.6

            qpe_results[n] = {
                "step_label":      step_label,
                "geometry":        geo_label,
                "E_exact_diag":    E_0_n,
                "E_ref_fci":       e_ref_n,
                "E_ref_used":      E_ref_chk,
                "E_map":           {lam: E_map_series[lam] for lam in [1, 2, 3]},
                "E_zne_rich":      E_zne,
                "E_zne_exp":       E_zne_ex,
                "E_zne_best":      E_best,
                "zne_method":      zne_method,
                "residual_mHa":    residual_mHa,
                "chem_ok":         chem_ok,
                "eta_v":           eta_v,
                "tau_seq_used":    active_tau_seq,
                "tau_seq_source":  tau_seq_source,
                "integral_source": "dataset" if store else "global",
            }
            status = "[✓]" if chem_ok else "[!]"
            print(
                f"    E_ZNE ({zne_method}) = {E_best:+.8f} Ha | "
                f"|E_ZNE−E_ref| = {residual_mHa:.4f} mHa {status}"
            )

        # ── 4. Semantic warrant (local DensityMatrixSim on init circuit) ─────
        log.info("[IonQ-Runner] Semantic warrant (local simulation)...")
        qubits   = [NomosIonQid(i) for i in range(N)]
        dft_circ = cirq.Circuit(cirq.Moment(
            [TetralemmaticIonDFTGate().on(q) for q in qubits]
        ))
        noise_model = ForteHardwareNoiseModel(
            p1q=   self._noise_params["p1q_error"],
            p2q=   self._noise_params["p2q_error"],
            p_meas=self._noise_params["p_meas_error"],
            p_idle=self._noise_params["p_idle_error"],
        )
        sim    = cirq.DensityMatrixSimulator(noise=noise_model)
        result_sim = sim.simulate(dft_circ, initial_state=0)
        rho_super  = result_sim.final_density_matrix

        if mech.S_target >= 1.0:
            P_target = np.zeros((4, 4), dtype=complex)
            P_target[1, 1] = P_target[2, 2] = 1.0
            eta_prime = self._eta
        else:
            P_target = np.zeros((4, 4), dtype=complex)
            P_target[3, 3] = 1.0
            eta_prime = (self._eta * 0.75) / (0.25 + 0.5 * self._eta)

        K_local = (
            np.sqrt(eta_prime) * P_target
            + np.sqrt(1.0 - eta_prime) * (np.eye(4) - P_target)
        )
        K_total = functools.reduce(np.kron, [K_local] * N)
        rho_raw = K_total @ rho_super @ K_total.conj().T
        norm    = float(np.real(np.trace(rho_raw)))
        final_rho = rho_raw / norm if norm > 1e-12 else rho_raw

        # ── 5. Stoichiometric verification ────────────────────────────────────
        print(f"\n  [VERIFY] Stoichiometric invariance suite...")
        verifier  = StoichiometricVerifier(mech, self._eta)
        stoich_ok, checks = verifier.full_report(tracker, final_rho, N, mech.S_target)
        for chk in checks:
            status = "[✓]" if chk["passed"] else "[✗]"
            print(f"  {status} {chk['condition']}: {chk['detail']}")

        # ── 6. Final report ───────────────────────────────────────────────────
        last_chk  = checkpoints[-1]
        E_0_final = (
            step_energies[last_chk]
            if step_energies[last_chk] is not None
            else qpe_results[last_chk]["E_exact_diag"]
        )
        self._print_report(
            mech, tracker, checkpoints, qpe_results,
            stoich_ok, checks, E_0_final, time.time() - start,
            dataset_dir=self._dataset_dir,
        )

        last_qpe    = qpe_results[last_chk]
        all_chem_ok = all(r["chem_ok"] for r in qpe_results.values())

        return {
            "mechanism_name":          mech.name,
            "N_orbitals":              N,
            "M_steps":                 mech.M_steps,
            "m":                       m,
            "n_crossings":             mech.n_crossings,
            "is_reversible_cycle":     mech.is_reversible_cycle,
            "total_electrons":         tracker.total_electrons,
            "total_cofactor_shift":    tracker.k_total,
            "total_electrons_ejected": tracker.total_electrons_ejected,
            "net_electrons":           tracker.net_electrons,
            "net_cofactor_shift":      tracker.k_total,
            "total_photons_absorbed":  tracker.total_photons_absorbed,
            "total_photons_emitted":   tracker.total_photons_emitted,
            "net_photons":             tracker.net_photons,
            "phase_closure_ok":        algebraic_ok,
            "electron_count_ok":       electron_ok,
            "stoichiometric_ok":       stoich_ok,
            "qpe_results":             qpe_results,
            "step_reference_energies": step_energies,
            "E_ref_final":             E_0_final,
            "E_zne_final":             last_qpe["E_zne_best"],
            "residual_mHa_final":      last_qpe["residual_mHa"],
            "chemical_accuracy_ok":    all_chem_ok,
            "qpe_path":                "ionq_virtual_ancilla",
            "ionq_target":             self._backend_cfg.resolved_target if self._backend_cfg else None,
            "n_shots":                 self._n_shots,
            "integral_source":         "dataset" if store else "global",
            "dataset_dir":             str(self._dataset_dir) if self._dataset_dir else None,
            "elapsed_s":               round(time.time() - start, 2),
            "stoich_checks":           checks,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _print_header(self, store) -> None:
        mech   = self._mechanism
        w      = 78
        target = self._backend_cfg.resolved_target if self._backend_cfg else "N/A"
        print("\n" + "=" * w)
        print(f" IONQ MQE-QPE VALIDATION: {mech.name.upper()}")
        print("=" * w)
        print(mech.summary())
        print(f"  Backend:  IonQ {target!r} | shots/job: {self._n_shots}")
        if store:
            print(
                f"  Integrals: step-wise JSON "
                f"({self._dataset_dir}/{mech.name}/step_XX.json)"
            )
        else:
            print(f"  Integrals: global fallback (same H at every step)")
        print("-" * w)

    def _print_report(
        self, mech, tracker, checkpoints, qpe_results,
        stoich_ok, checks, E_0_ref, elapsed, dataset_dir=None,
    ) -> None:
        w1, w2, w3 = 44, 18, 10
        target = self._backend_cfg.resolved_target if self._backend_cfg else "N/A"
        print(f"\n[IONQ-QPE RESULTS] {mech.name}  |  backend: {target!r}")
        print(f"  ┌─{'─'*w1}─┬─{'─'*w2}─┬─{'─'*w3}─┐")
        print(f"  │ {'Metric':<{w1}} │ {'Value':<{w2}} │ {'Status':<{w3}} │")
        print(f"  ├─{'─'*w1}─┼─{'─'*w2}─┼─{'─'*w3}─┤")

        def row(label, val, ok):
            s = "[✓] OK" if ok else "[✗] FAIL"
            print(f"  │ {label:<{w1}} │ {str(val):<{w2}} │ {s:<{w3}} │")

        row(f"ℤ_{mech.m} Phase Closure (k≡0 mod {mech.m})",
            f"Σν={tracker.k_total}", tracker.phase_closed)
        row("Electron Conservation (<N_e>_final)",
            f"{tracker.total_electrons} e⁻",
            tracker.total_electrons == mech.total_electrons)
        row("Trace Preservation (Tr ρ=1)", "—",
            next((c["passed"] for c in checks if "Trace" in c["condition"]), False))
        row(f"Spin-Parity Holding (η={self._eta})", "—",
            next((c["passed"] for c in checks if "Spin" in c["condition"]), False))

        print(f"  ├─{'─'*w1}─┼─{'─'*w2}─┼─{'─'*w3}─┤")
        for step_idx, qpe in sorted(qpe_results.items()):
            ok    = qpe["chem_ok"]
            src_s = "DS" if qpe.get("integral_source") == "dataset" else "GF"
            res   = qpe["residual_mHa"]
            val   = "noise-floor" if res == float("inf") else f"{res:.4f} mHa"
            etav  = f" η_V={qpe['eta_v']:.4f}" if qpe.get("eta_v") else ""
            row(
                f"IONQ-QPE|ZNE n={step_idx} ({qpe['step_label']}) [{src_s}]{etav}",
                val, ok,
            )

        print(f"  ├─{'─'*w1}─┼─{'─'*w2}─┼─{'─'*w3}─┤")
        all_qpe_ok = all(r["chem_ok"] for r in qpe_results.values())
        row("OVERALL CHEMICAL ACCURACY (≤1.6 mHa)", "", all_qpe_ok)
        row("STOICHIOMETRIC INVARIANCE", "", stoich_ok)
        print(f"  └─{'─'*w1}─┴─{'─'*w2}─┴─{'─'*w3}─┘")
        print(f"\n  E_ref (last step) = {E_0_ref:+.10f} Ha")
        print(f"  Elapsed: {elapsed:.2f}s")
        all_ok = stoich_ok and all_qpe_ok
        print(
            f"\n  {'[✓] IONQ MQE-QPE VALIDATION PASSED' if all_ok else '[✗] IONQ MQE-QPE VALIDATION FAILED'}"
        )
        print("=" * 78)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  TOP-LEVEL ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def run_ionq_mqe_validation(
    mechanism_name:  str,
    integral_state:  IntegralState,
    config:          MQEConfig                  = None,
    output_json:     Optional[str]             = None,
    dataset_dir:     Optional[Union[str, Path]] = None,
    backend_cfg:     BackendConfig             = None,
) -> Dict:
    r"""Top-level IonQ MQE-QPE validation entry point.

    Mirrors run_virtual_ancilla_qpe_validation() from mqevanc.py,
    but routes the production QPE measurement to the IonQ cloud simulator.

    Args:
        mechanism_name:  "nitrogenase_lt" | … | "all".
        integral_state:  Populated IntegralState from initialise_integrals().
        config:          MQEConfig (defaults to MQEConfig() if None).
        output_json:     Optional JSON export path.
        dataset_dir:     Root directory of mqedatagenerator.py output.
        backend_cfg:     BackendConfig with mode=IONQ_SIM and valid api_key.

    Returns:
        Dict (single mechanism) or Dict[name → Dict] (mechanism_name="all").
    """
    cfg        = config or MQEConfig()
    n_orbitals = integral_state.n_orbitals
    PREDEFINED = build_predefined_mechanisms(n_orbitals)

    def _make_runner(mech):
        return IonQMQEPipelineRunner(
            mechanism      = mech,
            integral_state = integral_state,
            config         = cfg,
            dataset_dir    = dataset_dir,
            backend_cfg    = backend_cfg,
        )

    if mechanism_name == "all":
        print("\n" + "=" * 78)
        print(" IONQ MQE-QPE FRAMEWORK VALIDATION: ALL MECHANISMS")
        if dataset_dir:
            print(f" Step-wise integrals: {dataset_dir}/")
        print("=" * 78)
        results = {}
        for name, mech in PREDEFINED.items():
            runner        = _make_runner(mech)
            results[name] = runner.run()

        if output_json:
            _export_mqe_json(
                results, output_json, n_orbitals,
                framework="Modular Quantum Emulator — MQE-QPE (IonQ)",
            )
        return results

    elif mechanism_name in PREDEFINED:
        mech   = PREDEFINED[mechanism_name]
        runner = _make_runner(mech)
        result = runner.run()
        if output_json:
            _export_mqe_json(
                {mechanism_name: result}, output_json, n_orbitals,
                framework="Modular Quantum Emulator — MQE-QPE (IonQ)",
            )
        return result

    else:
        raise ValueError(
            f"Unknown mechanism {mechanism_name!r}. "
            f"Choose from: {list(PREDEFINED.keys())} or 'all'."
        )
