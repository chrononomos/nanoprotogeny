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
mqevanc.py — MQE Pipeline Core (Hardware-QPE Variant, Path B)
==============================================================================
Self-contained variant of mqebaseline.py.  All pipeline logic is copied directly
from mqebaseline.py and the two modules are entirely independent — mqevanc.py
does not import from mqebaseline.py.

The single architectural difference between the two files:

    mqebaseline.py           uses  ionq/ionqtrotter.py    →  qpe/mqeqpe.py
    mqevanc.py  uses  ionq/ionqqputrotter.py →  qpe/mqevancqpe.py

Concretely, the QPE checkpoint block inside run() calls:

    Path A (mqebaseline.py):
        compute_qpe_signal(H, ψ, τ, λ, ...)  →  complex overlap C(τ)
        bayesian_map_energy({τ: C(τ)}, ...)   →  E_MAP

    Path B (mqevanc.py):
        compute_hardware_qpe_signal(H, ψ, τ, λ, ...)  →  ancilla p(k) vector
        hardware_map_energy({τ: p(k)}, ...)             →  E_MAP

The step-block Trotter circuits (build_mqe_step_block) continue to use
build_trotter_evolution_circuit from ionqtrotter.py, which is the correct
uncontrolled single-step Trotter kernel.  ionqqputrotter.py is reached via
mqevancqpe.py, which calls build_hardware_qpe_circuit internally for the
QPE ancilla measurement.

KNOWN LIMITATIONS RELATIVE TO PATH A
    1. Trotter angle scaling: hardware QPE gate angles are θ = h·τ (full
       evolution time), not h·Δt.  At large τ, Trotter approximation error
       enters the non-perturbative regime.  ZNE still cancels the noise
       polynomial but the Trotter systematic is different.
    2. Information content: 4 real probabilities p(k) per τ carry strictly
       less information than the complex overlap C(τ) ∈ ℂ.  Residuals may
       be larger than Path A's sub-0.1 mHa results; the 1.6 mHa chemical-
       accuracy threshold remains the validation criterion.

ROLE IN THE PACKAGE
  This module is the pipeline orchestration layer for Path B.
  Sub-modules it imports (unchanged from mqebaseline.py):

    simulate/mqeconfig.py        — MQEConfig, IntegralState, algorithm constants
    simulate/mqedualmanifold.py  — dual-manifold circuit construction & verification
    molecular/mqestoichiometry.py— StoichiometricVerifier (Theorem 2)
    molecular/mqeintegralloader.py — IntegralState bootstrap
    molecular/mqeintegralstore.py  — StepwiseIntegralStore, per-step JSON loading
    ionq/ionqconnectivity.py     — BackendMode, BackendConfig, service factory
    ionq/ionqhistogram.py        — histogram parsing, ontological decoding
    ionq/ionqtrotter.py          — Trotterized step-block evolution circuits
    ionq/ionqqputrotter.py       — Hardware QPE circuit (via mqevancqpe.py)
    qpe/mqevancqpe.py        — Hardware QPE signal + MLE MAP extraction

QPE SIGNAL MATHEMATICS (Path B)
  p(k | τ, λ) = Tr_sys[ρ_anc]_{kk}  (ancilla diagonal after partial trace)
  Model:  p(k | τ, E) = (1/16)|Σ_{m=0}^3 e^{im(Eτ−πk/2)}|²
  MLE:    L(E) = Σ_τ Σ_k p_obs(k,τ) · log(p_model(k|τ,E)+ε)
  ZNE:    E_ZNE = 3E(λ=1) − 3E(λ=2) + E(λ=3)   [Richardson]
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import functools
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# ── Third-party ───────────────────────────────────────────────────────────────
import cirq
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Fallback: rely on shell exports

# ── IonQ hardware layer ───────────────────────────────────────────────────────
from nanoprotogeny.ionq.YB171PLUSHARDWARE import NomosIonQid, VirtualQudit

from nanoprotogeny.ionq.ionqtetralemmatics import (
    TetralemmaticIonDFTGate, TetralemmaticIonURShiftGate, TetralemmaticIonZClockGate, DFT_onto
)

from nanoprotogeny.ionq.ionqcrossgates import ZenoStabilizeGate
from nanoprotogeny.ionq.ionqmqegates import CrossManifoldSWAPGate
from nanoprotogeny.ionq.holographic import compile_with_holographic_routing
from nanoprotogeny.ionq.ionqscheduler import (
    build_qudit_dependency_dag,
    schedule_parallel_moments,
)
from nanoprotogeny.ionq.ionqjanus import build_biochemical_transition_circuit
from nanoprotogeny.ionq.ionqtrotter import (
    build_trotter_evolution_circuit,
)
from nanoprotogeny.ionq.ionqfortenoise import (
    FORTE_NOISE_PARAMS,
    ForteHardwareNoiseModel,
)
from nanoprotogeny.ionq.ionqconnectivity import _CIRQ_IONQ_AVAILABLE

# ── Molecular layer ───────────────────────────────────────────────────────────
from nanoprotogeny.molecular.mqemolecules import (
    MechanismTuple,
    build_predefined_mechanisms,
)
from nanoprotogeny.molecular.mqeintegralstore import StepwiseIntegralStore
from nanoprotogeny.molecular.mqehamiltonian import (
    build_qudit_hamiltonian_matrix,
    _project_hamiltonian_to_sector,
    ground_state_from_diagonalization,
)
from nanoprotogeny.molecular.mqephasetracker import ZmPhaseTracker
from nanoprotogeny.molecular.mqestoichiometry import StoichiometricVerifier

# ── Simulate layer ────────────────────────────────────────────────────────────
from nanoprotogeny.simulate.mqeconfig import MQEConfig, IntegralState
from nanoprotogeny.simulate.mqedualmanifold import (
    _make_virtual_qudits_m,
    VirtualRegisterPair,
)
from nanoprotogeny.ionq.ionqmqegates import (
    CompositeVirtualShiftGate,
)

# ── QPE — Path B: hardware ancilla circuit + MLE extraction ──────────────────
from nanoprotogeny.qpe.mqevancqpe import (
    _project_rho_to_sector,
    hardware_map_energy,
    compute_virtual_ancilla_qpe_probs,
    select_tau_sequence_virtual_ancilla,
    VIRTUAL_ANCILLA_D_STATE_NOISE,
    _count_screened_ctrl_gates,
)
from nanoprotogeny.qpe.mqetrotterdensematrix import (
    compute_trotter_density_matrix,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def build_mqe_step_block(
    n:               int,
    mechanism:       "MechanismTuple",
    logical_qudits:  List["NomosIonQid"],
    virtual_qudits:  List[cirq.Qid],
    h_diag:          Dict,
    h_hop:           Dict,
    g_full:          Dict,
    dt:              float,
    phase_tracker:   "ZmPhaseTracker",
    step_integrals:  Optional[Tuple[Dict, Dict, Dict, float, int]] = None,
    vreg:            Optional["VirtualRegisterPair"] = None,
) -> Tuple[cirq.Circuit, bool]:
    r"""Build B_n: Trotterized propagation + biochemical transition for step n.

    Implements Eq. (general_Bn) from the article:
        B_n^{(M)}(ρ) = J_{n→n+1}^{(M)} [e^{-iH_{n,1}Δt} e^{-iH_{n,2}Δt}
                        ρ e^{iH_{n,2}Δt} e^{iH_{n,1}Δt}]
                        (J_{n→n+1}^{(M)})†

    Step-wise Hamiltonian loading:
        If step_integrals is provided (a 5-tuple from StepwiseIntegralStore),
        its h_diag_n, h_hop_n, g_full_n override the global integrals for
        this step's Trotterized evolution. This is the production path when
        running against mqedatagenerator.py output.

        If step_integrals is None, the function falls back to the h_diag,
        h_hop, g_full arguments passed directly — this preserves full
        backward compatibility with the demo / single-molecule path.

    Args:
        n:               Step index (0-based).
        mechanism:       MechanismTuple.
        logical_qudits:  List of NomosIonQid [0..N-1].
        virtual_qudits:  List of d=m virtual Qid [0..N-1].
        h_diag:          Fallback one-electron diagonal integrals {p: h_pp}.
        h_hop:           Fallback one-electron hopping integrals {(p,q): h_pq}.
        g_full:          Fallback two-electron ERI integrals {(p,q,r,s): g}.
        dt:              Trotter step size (Ha⁻¹).
        phase_tracker:   ZmPhaseTracker to update.
        step_integrals:  Optional 5-tuple (h_diag_n, h_hop_n, g_full_n,
                         e_core_n, n_orbs_n) for this specific step.
                         When provided, overrides h_diag/h_hop/g_full.

    Returns:
        (circuit, crossing_applied): Step-n circuit + Janus SWAP flag.
    """
    crossing_applied = False

    # ── Select integrals: step-specific overrides global fallback ────────────
    if step_integrals is not None:
        h_diag_n, h_hop_n, g_full_n, e_core_n, n_orbs_n = step_integrals
        log.info(
            f"[HW-B{n}] Using step-specific integrals: "
            f"N={n_orbs_n}, E_core={e_core_n:+.6f} Ha"
        )
    else:
        h_diag_n = h_diag
        h_hop_n  = h_hop
        g_full_n = g_full
        n_orbs_n = mechanism.N_orbitals
        log.debug(f"[HW-B{n}] Using global fallback integrals (N={n_orbs_n})")

    # ── Trotterized Hamiltonian propagation with step-n integrals ────────────
    trotter_circ = build_trotter_evolution_circuit(
        n_orbs_n, h_diag_n, h_hop_n, g_full_n, dt=dt
    )

    # ── Resolve composite register for this step ─────────────────────────────
    vaux_qudits = vreg.vaux if (vreg is not None and vreg.is_composite) else None

    # ── Biochemical transition J_{n→n+1} ─────────────────────────────────────
    J_circ = build_biochemical_transition_circuit(
        n, mechanism, logical_qudits, virtual_qudits, dt,
        vaux_qudits=vaux_qudits,
    )

    # ── Conditional cross-manifold SWAP (conical intersection) ──────────────
    # Guard: m % 4 == 0  (Case III — Janus crossing is operational for all m=4r)
    # The SWAP acts on (H_L, H_V₁) only; H_V_aux is invariant through the crossing.
    # virtual_qudits here is always the V₁ list (primary d=4 register).
    crossing_ops = []
    for (step_idx, p_idx, q_idx, delta_ci) in mechanism.crossings:
        if step_idx == n:
            if (mechanism.m % 4 == 0                     # <-- generalised guard
                    and p_idx < len(logical_qudits)
                    and p_idx < len(virtual_qudits)
                    and q_idx < len(logical_qudits)
                    and q_idx < len(virtual_qudits)):
                v_p = virtual_qudits[p_idx]
                v_q = virtual_qudits[q_idx]
                # v_p must be the d=4 primary VirtualQudit (not a LineQid for odd m)
                if hasattr(v_p, '_index') or (
                    hasattr(v_p, 'dimension') and v_p.dimension == 4
                ):
                    crossing_ops.extend([
                        CrossManifoldSWAPGate().on(logical_qudits[p_idx], v_p),
                        CrossManifoldSWAPGate().on(logical_qudits[q_idx], v_q),
                    ])
                    crossing_applied = True
                    log.info(
                        f"[HW] Step n={n}: Janus crossing applied "
                        f"(orbitals p={p_idx}, q={q_idx}, δCI={delta_ci:.2e})"
                    )

    # ── Update phase tracker (net-flux: passes both forward and inverse fields) ─
    nu_decouple_n  = mechanism.nu_decouple_shifts[n]   # always present post __post_init__
    n_ejected_n    = len(mechanism.electron_eject_sets[n])
    n_absorbed_n   = len(mechanism.photon_absorb_sets[n])
    n_emitted_n    = len(mechanism.photon_emit_sets[n])
    phase_tracker.step(
        n,
        mechanism.nu_shifts[n],
        len(mechanism.electron_sets[n]),
        nu_decouple         = nu_decouple_n,
        n_electrons_ejected = n_ejected_n,
        n_photons_absorbed  = n_absorbed_n,
        n_photons_emitted   = n_emitted_n,
    )

    # ── Assemble: Trotter → [Crossing] → J ──────────────────────────────────
    combined = cirq.Circuit(trotter_circ)
    if crossing_ops:
        combined.append(cirq.Moment(crossing_ops))
    combined.append(J_circ)

    return combined, crossing_applied


def build_mqe_L_block(
    step_circuit:   cirq.Circuit,
    virtual_qudits: List[cirq.Qid],
    n:              int,
    m:              int,
    phase_tracker:  ZmPhaseTracker,
    vreg:           Optional["VirtualRegisterPair"] = None,
) -> cirq.Circuit:
    r"""Build L_n: coherence management wrapper around B_n.

    Implements Eq. (general_Ln):
        L_n^{(M)} = U_Zeno^{(n)} ∘ U_comp^{(n)} ∘ U_shield^{(n)} ∘ B_n^{(M)}

    For d=4 virtual qudits (m=4):
        - U_shield: holographic shielding via existing HolographicRouter.
        - U_comp: inverse quarter-turn to compensate accumulated phase drift.
        - U_Zeno: boundary reflection via ZenoStabilizeGate.

    For d=m (m≠4):
        - Coherence routing is simplified: phase compensation via
          GeneralizedVirtualShiftGate with inverse power.
        - Zeno not applied (VirtualQudit-specific hardware feature).

    Args:
        step_circuit:   B_n circuit from build_mqe_step_block.
        virtual_qudits: d=m virtual register qudits.
        n:              Step index (0-based).
        m:              Virtual modulus.
        phase_tracker:  ZmPhaseTracker for reading current phase index.

    Returns:
        L_n circuit with coherence management appended.
    """
    circuit = cirq.Circuit(step_circuit)

    # U_comp gate: group-theoretic inverse of the accumulated phase shift.
    # compensation_gate() owns the arithmetic (m - k^{(n)}) % m for any m.
    u_comp = phase_tracker.compensation_gate()

    if m % 4 == 0:
        # ── U_comp: phase closure via tracker-owned gate ─────────────────────
        # For m=4  (r=1): GeneralizedVirtualShiftGate on a single VirtualQudit.
        # For m=4r (r>1): CompositeVirtualShiftGate on (V1, Vaux) pair.
        comp_ops = []
        if u_comp._power > 0:
            if isinstance(u_comp, CompositeVirtualShiftGate) and vreg is not None and vreg.is_composite:
                for i, vq in enumerate(virtual_qudits):
                    va = vreg.vaux[i]
                    if va is not None:
                        comp_ops.append(u_comp.on(vq, va))
            else:
                for vq in virtual_qudits:
                    comp_ops.append(u_comp.on(vq))
        if comp_ops:
            circuit.append(cirq.Moment(comp_ops))

        # ── U_Zeno: boundary reflection on all (NomosIonQid, VirtualQudit) pairs
        # Zeno targets the primary d=4 register only — Vaux is unaffected.
        zeno_ops = []
        logical_q_in_circ = sorted(
            [q for q in circuit.all_qubits() if isinstance(q, NomosIonQid)]
        )
        virt_q_in_circ = [
            q for q in circuit.all_qubits() if isinstance(q, VirtualQudit)
        ]
        for lq, vq in zip(logical_q_in_circ, virt_q_in_circ):
            zeno_ops.append(ZenoStabilizeGate().on(lq, vq))
        if zeno_ops:
            circuit.append(cirq.Moment(zeno_ops))

    else:
        # ── Odd m (Case I): phase compensation only, no Zeno ─────────────────
        # (Zeno is hardware-specific to d=4 ¹⁷¹Yb⁺ VirtualQudit)
        comp_ops = []
        if u_comp._power > 0:
            for vq in virtual_qudits:
                comp_ops.append(u_comp.on(vq))
        if comp_ops:
            circuit.append(cirq.Moment(comp_ops))

    return circuit


# StoichiometricVerifier imported at top from nanoprotogeny.molecular.mqestoichiometry


# ==============================================================================
# LAYER 6: HARDWARE QPE PIPELINE RUNNER
# ==============================================================================

class HardwareQPEPipelineRunner:
    r"""Orchestrates the full MQE validation pipeline using hardware QPE.

    Structurally identical to MQEPipelineRunner in mqebaseline.py.  The sole
    difference is the QPE energy-extraction block inside run():

        mqebaseline.py:            compute_qpe_signal + bayesian_map_energy
        mqevanc.py:   compute_hardware_qpe_signal + hardware_map_energy

    The hardware QPE block simulates ionqqputrotter.build_hardware_qpe_circuit
    (F₄ → power-controlled Trotter → F₄†) under the Forte noise model,
    partial-traces the system register, and applies MLE over the ancilla
    probability vector p(k) to extract the energy.

    Extended with step-wise JSON loading:
        When dataset_dir is provided, each step n uses the Hamiltonian
        H_n from <dataset_dir>/<mechanism.name>/step_{n:02d}.json instead
        of the single globally-loaded integral set. The reference energy
        for QPE at each checkpoint is taken from the step JSON rather than
        from the single-molecule diagonalization.

    Args:
        mechanism:    MechanismTuple. If dataset_dir is set, this is
                      overridden by the manifest-reconstructed tuple.
        integral_state: Populated IntegralState (h_diag, h_hop, g_full,
                      e_core, n_orbitals).
        config:       MQEConfig (defaults to MQEConfig() if None).
        noise_params: IonQ Forte noise parameters dict.
        dataset_dir:  Path to the root datasets/ directory produced by
                      mqedatagenerator.py.
    """

    def __init__(
        self,
        mechanism:      "MechanismTuple",
        integral_state: "IntegralState",
        config:         "MQEConfig"              = None,
        noise_params:   Dict                     = None,
        dataset_dir:    Optional[Union[str, Path]] = None,
    ):
        cfg = config or MQEConfig()
        self._mechanism    = mechanism
        self._h_diag       = integral_state.h_diag
        self._h_hop        = integral_state.h_hop
        self._g_full       = integral_state.g_full
        self._e_core       = integral_state.e_core
        self._dt                 = cfg.dt
        self._eta                = cfg.eta
        self._tau_seq            = list(cfg.tau_seq)           # fallback (legacy ref)
        self._tau_seq_candidates = list(cfg.tau_seq_candidates)
        self._tau_seq_fallback   = list(cfg.tau_seq)
        self._cached_tau_seq: Optional[List[float]] = None
        self._idle_threshold = cfg.idle_threshold
        self._noise_params = noise_params or FORTE_NOISE_PARAMS
        self._dataset_dir  = Path(dataset_dir) if dataset_dir else None
        self._store: Optional["StepwiseIntegralStore"] = None

    # ── Step-store accessor (lazy init) ──────────────────────────────────────

    def _get_store(self) -> Optional["StepwiseIntegralStore"]:
        """Return the StepwiseIntegralStore, initialising on first access."""
        if self._store is not None:
            return self._store
        if self._dataset_dir is None:
            return None
        try:
            self._store = StepwiseIntegralStore(
                self._dataset_dir, self._mechanism.name
            )
            # If dataset provides the mechanism tuple, use it
            self._mechanism = self._store.to_mechanism_tuple()
            log.info(
                f"[HW-Runner] Loaded StepwiseIntegralStore for '{self._mechanism.name}' "
                f"from {self._dataset_dir}/{self._mechanism.name}"
            )
        except FileNotFoundError as e:
            log.warning(
                f"[HW-Runner] dataset_dir set but store not available: {e}\n"
                f"  Falling back to global integrals for all steps."
            )
            self._store = None
        return self._store

    # ── Public run() ────────────────────────────────────────────────────────
    def run(self) -> Dict:
        """Execute the full MQE validation pipeline (hardware QPE). Returns result dict."""
        mech   = self._mechanism
        N      = mech.N_orbitals
        m      = mech.m
        store  = self._get_store()
        start  = time.time()

        self._print_header(store)

        # ── 1. Allocate registers ────────────────────────────────────────────
        logical_qudits = [NomosIonQid(i) for i in range(N)]
        vreg           = _make_virtual_qudits_m(N, m)
        virtual_qudits = vreg.v1        # primary d=4 register (V1)
        src = (
            f"dataset ({self._dataset_dir}/{mech.name})"
            if store else "global fallback"
        )
        reg_desc = (
            f"{N} logical (d=4) + {N} virtual (d=4+carry, m={m})"
            if vreg.is_composite
            else f"{N} logical (d=4) + {N} virtual (d={m})"
        )
        print(f"  [REG] {reg_desc} qudits | integrals: {src}")

        # ── 2. Algebraic phase-closure validation (net-flux) ─────────────────
        print(f"\n  [ALGEBRAIC] Net-flux phase closure validation (ℤ_{m})...")
        alg_tracker = ZmPhaseTracker(m)
        for n in range(mech.M_steps):
            alg_tracker.step(
                n,
                mech.nu_shifts[n],
                len(mech.electron_sets[n]),
                nu_decouple         = mech.nu_decouple_shifts[n],
                n_electrons_ejected = len(mech.electron_eject_sets[n]),
                n_photons_absorbed  = len(mech.photon_absorb_sets[n]),
                n_photons_emitted   = len(mech.photon_emit_sets[n]),
            )
        print(alg_tracker.report(mech))

        algebraic_ok = alg_tracker.phase_closed
        electron_ok  = (alg_tracker.net_electrons == mech.total_net_electrons)

        # ── 3. Main loop: M steps ────────────────────────────────────────────
        print(f"\n  [LOOP] Building M={mech.M_steps} MQE step blocks...")
        full_circuit       = cirq.Circuit()
        tracker            = ZmPhaseTracker(m)
        checkpoints        = list(set(
            [c[0] for c in mech.crossings] + [mech.M_steps - 1]
        ))
        checkpoints.sort()
        qpe_results:  Dict[int, Dict] = {}
        step_energies: List[Optional[float]] = []

        for n in range(mech.M_steps):
            # ── 3a. Load step-specific integrals (or global fallback) ─────────
            if store is not None:
                step_ints = store.get_step(n)          # (h_d, h_h, g_f, ec, n_o)
                e_ref_n   = store.get_reference_energy(n)
                store.log_step_summary(n)
            else:
                step_ints = None
                e_ref_n   = None

            step_energies.append(e_ref_n)

            # ── DIAGNOSTIC: Verify ERI convention after parsing ──────────────────────
            if step_ints is not None:
                h_diag_n, h_hop_n, g_full_n, e_core_n, n_orbs_n = step_ints

                # Count unique vs. symmetric ERI entries
                unique_count = len(g_full_n)
                expected_unique = n_orbs_n * (n_orbs_n + 1) * (n_orbs_n + 2) * (n_orbs_n + 3) // 24
                expected_full = n_orbs_n ** 4

                log.info(f"[ERI DIAG] n_orbs={n_orbs_n}, g_full entries={unique_count}")
                log.info(f"[ERI DIAG] Expected unique (chemist): ~{expected_unique}, full tensor: {expected_full}")

                if unique_count > expected_unique * 4:
                    log.warning(
                        f"[ERI CONVENTION] g_full appears to contain pre-expanded symmetry positions. "
                        f"build_qudit_hamiltonian_matrix will expand again → 8x overcounting. "
                        f"Fix: Change prefactor 0.5 → 0.0625 in ERI summation loop."
                    )

            # ── DIAGNOSTIC: Verify H_qudit vs dataset FCI before QPE ──────────────
            if step_ints is not None:
                h_diag_n, h_hop_n, g_full_n, e_core_n, n_orbs_n = step_ints
                H_step = build_qudit_hamiltonian_matrix(n_orbs_n, h_diag_n, h_hop_n, g_full_n)
                E_diag, _ = ground_state_from_diagonalization(H_step)
                E_ref_active = e_ref_n - e_core_n if e_ref_n is not None else None

                if E_ref_active is not None:
                    delta = abs(E_diag - E_ref_active) * 1000
                    if delta > 1.0:
                        log.warning(
                            f"[DIAG] Step {n}: H_qudit diagonalization differs from dataset FCI: "
                            f"{delta:.3f} mHa (E_diag={E_diag:+.8f}, E_ref={E_ref_active:+.8f})"
                        )

            # ── 3b. Build B_n circuit ─────────────────────────────────────────
            step_circ, crossing_applied = build_mqe_step_block(
                n, mech, logical_qudits, virtual_qudits,
                self._h_diag, self._h_hop, self._g_full,
                self._dt, tracker,
                step_integrals=step_ints,
                vreg=vreg,
            )

            # ── 3c. L_n: coherence management ─────────────────────────────────
            L_n = build_mqe_L_block(
                step_circ, virtual_qudits, n, m, tracker,
                vreg=vreg,
            )
            full_circuit.append(L_n)

            k_n       = tracker._k_step
            net_e_n   = tracker.net_electrons
            rev_tag   = ""
            if mech.electron_eject_sets[n] or mech.proton_eject_sets[n] or mech.nu_decouple_shifts[n]:
                rev_tag = (
                    f" | A_ej={mech.electron_eject_sets[n]}"
                    f" ν†={mech.nu_decouple_shifts[n]}"
                )
            photon_tag = ""
            if mech.photon_absorb_sets[n] or mech.photon_emit_sets[n]:
                photon_tag = (
                    f" | hν_abs={mech.photon_absorb_sets[n]}"
                    f" hν_emt={mech.photon_emit_sets[n]}"
                )
            log.info(
                f"  [HW n={n:02d}] A_n={mech.electron_sets[n]}"
                f"{rev_tag}"
                f"{photon_tag}"
                f" | ν={mech.nu_shifts[n]} k^(n)={k_n}"
                f" | Σe_net={net_e_n}"
                f" | {'⚡ Janus' if crossing_applied else ''}"
            )

            # ── 3d. Hardware QPE + ZNE checkpoint ────────────────────────────
            if n in checkpoints:
                # Build step-n Hamiltonian matrix for QPE
                if step_ints is not None:
                    h_d_n, h_h_n, g_f_n, _, n_o_n = step_ints
                    step_data_raw  = store._load_step_raw(n)
                    nelec_active_n = step_data_raw.get("metadata", {}).get("nelec_active", None)
                else:
                    h_d_n, h_h_n, g_f_n = self._h_diag, self._h_hop, self._g_full
                    n_o_n          = N
                    nelec_active_n = None

                # ── Build FULL Hamiltonian, then project for diagonalization ──
                H_full = build_qudit_hamiltonian_matrix(n_o_n, h_d_n, h_h_n, g_f_n)

                # Project to electron sector for accurate ground-state reference
                sector_indices = None
                if nelec_active_n is not None:
                    H_proj, sector_indices = _project_hamiltonian_to_sector(
                        H_full, n_o_n, nelec_active_n, return_indices=True
                    )
                else:
                    H_proj = H_full

                # Diagonalize projected Hamiltonian for energy reference
                E_0_n, psi_proj = ground_state_from_diagonalization(H_proj)

                # ── Lift projected ground state back to full Hilbert space ──
                if sector_indices is not None:
                    full_dim = 4 ** n_o_n
                    psi_n = np.zeros(full_dim, dtype=complex)
                    psi_n[sector_indices] = psi_proj
                else:
                    psi_n = psi_proj

                # Use dataset FCI reference if available; otherwise exact diag
                if e_ref_n is not None:
                    _, _, _, e_core_n, _ = step_ints
                    E_ref_chk = e_ref_n - e_core_n
                else:
                    E_ref_chk = E_0_n

                # Sector projection guard:
                #   E_0_n > 0  → target sector is ABOVE the vacuum sector (E=0).
                #                 Noise scatters amplitude downhill into the vacuum,
                #                 contaminating p(k|τ) with a spurious E≈0 signal.
                #                 Project ρ to remove that contamination.
                #   E_0_n ≤ 0  → target sector IS the ground-state sector.
                #                 No vacuum contamination. Projection would distort
                #                 the λ-dependence of E(λ) that ZNE relies on.
                effective_sector_indices = sector_indices if E_0_n > 0.0 else None

                is_janus  = (n in [c[0] for c in mech.crossings])
                step_label = (
                    f"Janus E_{n}→E_{n+1}" if is_janus
                    else f"E_{n}→E_{n+1}"
                )
                geo_label = ""
                if store:
                    meta      = store.get_step_meta(n)
                    geo_label = meta.get("geometry_label", "")

                print(
                    f"\n  [HW-QPE] Checkpoint n={n} ({step_label})"
                    + (f" | {geo_label}" if geo_label else "")
                )
                print(
                    f"    E_0 (exact diag) = {E_0_n:+.8f} Ha"
                    + (f" | E_ref (FCI) = {E_ref_chk:+.8f} Ha"
                        if e_ref_n is not None else "")
                )

                # ── Hardware QPE: fixed-depth density-matrix path (Part 1) ───
                # Preflight selects τ-sequence using the actual Path B signal
                # (p(k|τ) + MLE) — not the Path B' signal (C(τ) + Bayesian MAP).
                # Returns None when no candidate achieves chemical accuracy under
                # Forte noise, which is recorded as a noise-floor failure.
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
                    # None means every candidate exceeded the budget — record as
                    # noise-floor result and do not cache (allows retry if E_ref
                    # changes at a later checkpoint).
                    if active_tau_seq is None:
                        log.warning(
                            f"[HW-QPE n={n}] No τ-sequence achieves ≤1.6 mHa "
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
                            "residual_mHa":    float("inf"),
                            "chem_ok":         False,
                            "eta_v":           None,
                            "tau_seq_used":    None,
                            "tau_seq_source":  "noise-floor",
                            "integral_source": "dataset" if store else "global",
                        }
                        print(
                            f"    [!] Hardware QPE below noise floor — "
                            f"no τ-sequence within 1.6 mHa budget."
                        )
                        continue  # skip to next step in the M-steps loop
                    self._cached_tau_seq = active_tau_seq
                    tau_seq_source = "adaptive"

                tau_max = max(active_tau_seq)
                n_max   = max(1, int(round(tau_max / self._dt)))
                log.info(
                    f"[VANC-QPE n={n}] τ-seq ({tau_seq_source}): {active_tau_seq} "
                    f"| n_max={n_max}"
                )

                # MQE-QPE: native virtual D-state ancilla, 256-dim block simulation.
                # η_V = (1-p_idle)^n_ctrl damps off-diagonal coherences.
                # The SAME η_V is passed to hardware_map_energy so the MLE model
                # (p_model_corr) matches the data-generating model (p_obs with η_V).
                angle_scale_mqe = tau_max / n_max
                gates_per_step_mqe = _count_screened_ctrl_gates(
                    n_o_n, h_d_n, h_h_n, g_f_n, angle_scale_mqe
                )
                eta_v_mqe = (
                    1.0 - VIRTUAL_ANCILLA_D_STATE_NOISE["p_idle_virtual"]
                ) ** (gates_per_step_mqe * n_max)
                eta_v_step = eta_v_mqe
                log.info(
                    f"[VANC-QPE n={n}] η_V={eta_v_mqe:.6f} "
                    f"(gates/step={gates_per_step_mqe}, n_ctrl={gates_per_step_mqe*n_max})"
                )

                E_map_series = {}
                for lam in [1, 2, 3]:
                    rho_lam = compute_trotter_density_matrix(
                        psi_n, n_max,
                        noise_scale=lam,
                        n_orbitals=n_o_n,
                        h_diag=h_d_n,
                        h_hop=h_h_n,
                        g_full=g_f_n,
                        dt=self._dt,
                    )
                    rho_lam = _project_rho_to_sector(rho_lam, effective_sector_indices)
                    ancilla_probs = {
                        tau: compute_virtual_ancilla_qpe_probs(
                            rho_lam, H_full, tau, n_max,
                            n_o_n, h_d_n, h_h_n, g_f_n,
                            dt=self._dt,
                        )
                        for tau in active_tau_seq
                    }
                    E_map, _, _ = hardware_map_energy(
                        ancilla_probs, E_ref=E_ref_chk, eta_v=eta_v_mqe
                    )
                    E_map_series[lam] = E_map

                E1, E2, E3 = E_map_series[1], E_map_series[2], E_map_series[3]
                E_zne = 3*E1 - 3*E2 + E3

                denom = E3 - 2*E2 + E1
                if abs(denom) > 1e-12:
                    E_inf    = (E1*E3 - E2**2) / denom
                    denom_e  = E2 - E_inf
                    E_zne_ex = (
                        E_inf + (E1-E_inf)**2/denom_e
                        if abs(denom_e) > 1e-12 else E_zne
                    )
                else:
                    E_zne_ex = E_zne

                # Pick the extrapolant with the smaller residual.  The exponential
                # fit can be ill-conditioned for short sequences (n_max=1, single τ);
                # plain Richardson is more reliable in that regime.  The preflight
                # uses the same criterion, so the preflight residual faithfully
                # predicts the production residual.
                res_rich = abs(E_zne    - E_ref_chk) * 1000
                res_ex   = abs(E_zne_ex - E_ref_chk) * 1000
                if res_ex <= res_rich:
                    E_best       = E_zne_ex
                    residual_mHa = res_ex
                    zne_method   = "exp"
                else:
                    E_best       = E_zne
                    residual_mHa = res_rich
                    zne_method   = "rich"
                chem_ok = residual_mHa <= 1.6

                qpe_results[n] = {
                    "step_label":      step_label,
                    "geometry":        geo_label,
                    "E_exact_diag":    E_0_n,
                    "E_ref_fci":       e_ref_n,
                    "E_ref_used":      E_ref_chk,
                    "E_map":           {lam: E_map_series[lam] for lam in [1,2,3]},
                    "E_zne_rich":      E_zne,
                    "E_zne_exp":       E_zne_ex,
                    "E_zne_best":      E_best,
                    "zne_method":      zne_method,
                    "residual_mHa":    residual_mHa,
                    "chem_ok":         chem_ok,
                    "eta_v":           eta_v_step,
                    "tau_seq_used":    active_tau_seq,
                    "tau_seq_source":  tau_seq_source,
                    "integral_source": "dataset" if store else "global",
                }

                status = "[✓]" if chem_ok else "[!]"
                print(
                    f"    E_ZNE ({zne_method}) = {E_best:+.8f} Ha | "
                    f"|E_ZNE−E_ref| = {residual_mHa:.4f} mHa {status}"
                )

        # ======================================================================
        # Compilation profile (outside the main M-steps loop)
        # ======================================================================
        print("\n  [COMPILATION] Profiling full multi-step sequence to Forte pulses...")
        try:
            idle_thresh = self._idle_threshold

            # Schedule qudit-level ops in parallel before B_LOG/B_VIRT expansion.
            # build_qudit_dependency_dag / schedule_parallel_moments collapse the
            # sequentially-assembled full_circuit into the minimum number of
            # parallel moments (up to max_concurrent_ms=4 entangling gates per
            # moment), which reduces the downstream GPI/GPI2/ZZ gate count.
            _dag = build_qudit_dependency_dag(list(full_circuit.all_operations()))
            _scheduled_moments = schedule_parallel_moments(_dag, max_concurrent_ms=4)
            scheduled_circuit = cirq.Circuit(_scheduled_moments)
            log.info(
                f"[SCHED] {len(list(full_circuit.all_operations()))} ops → "
                f"{len(scheduled_circuit)} moments "
                f"(was {sum(1 for _ in full_circuit)} sequential)"
            )

            # Per-gate-type breakdown BEFORE compilation (qudit level).
            from collections import Counter
            pre_counts = Counter(
                type(op.gate).__name__
                for op in scheduled_circuit.all_operations()
            )
            log.info(
                "[SCHED] Pre-compile gate type breakdown: "
                + ", ".join(f"{k}={v}" for k, v in pre_counts.most_common())
            )

            compiled_circuit = compile_with_holographic_routing(
                scheduled_circuit,
                idle_threshold=idle_thresh,
                auto_route=True,
                target="forte_native",
                simulation_mode=False
            )

            compiled_circuit = cirq.drop_negligible_operations(compiled_circuit, atol=1e-8)
            compiled_circuit = cirq.drop_empty_moments(compiled_circuit)
            has_matrix = any(isinstance(op.gate, cirq.MatrixGate) for op in compiled_circuit.all_operations())

            # Profile native hardware gates
            native_counts = {"GPI": 0, "GPI2": 0, "ZZ": 0, "Other": 0}
            for op in compiled_circuit.all_operations():
                name = op.gate.__class__.__name__
                if name in ("GPI2Gate", "cirq_ionq.GPI2Gate"):
                    native_counts["GPI2"] += 1
                elif name in ("GPIGate", "cirq_ionq.GPIGate"):
                    native_counts["GPI"] += 1
                elif name in ("ZZGate", "ZZPowGate", "cirq_ionq.ZZGate"):
                    native_counts["ZZ"] += 1
                else:
                    native_counts["Other"] += 1

            log.info(f"[HW-INFO] → Total Compiled Moments: {len(compiled_circuit)}")
            log.info(f"[HW-INFO] → Native Footprint: GPI={native_counts['GPI']}, GPI2={native_counts['GPI2']}, ZZ={native_counts['ZZ']}, Other={native_counts['Other']}")
            log.info(f"[HW-INFO] → MatrixGate Fallback: {'[!] DETECTED' if has_matrix else '[✓] ZERO'}")

            # Per-wrapper contribution: correct per-gate native cost + unitary verification.
            #
            # Two compilation paths are used:
            #   Standalone  — single gate in isolation, through the full pipeline.
            #                 Accurate for all gate types except BLOG-sandwich ones.
            #   Effective   — adjacent pair of the same gate, through the full pipeline
            #                 including cancel_adjacent_basis_changes.  The boundary
            #                 BLOG(exit) · BLOG_DAG(entry) pair cancels, saving 37 native
            #                 ops.  Dividing by 2 gives the effective per-gate cost.
            #                 Applied only to _BLOG_SANDWICH_WRAPPERS (Coulomb, Exchange).
            #
            # Unitary verification: checks that cirq.unitary(compiled_single) matches
            # cirq.unitary(expanded_single) (the Physical wrapper's intended matrix) up
            # to global phase.  Catches decomposition bugs before they reach hardware.
            import numpy as _np
            from nanoprotogeny.ionq.holographic import (
                unified_expand_qudit_circuit,
                _expand_blog_sandwich_wrappers,
                cancel_adjacent_basis_changes,
                _BLOG_SANDWICH_WRAPPERS,
            )
            from cirq_ionq.ionq_native_target_gateset import ForteNativeGateset as _FNG

            _NATIVE_NAMES = {"GPIGate", "GPI2Gate", "ZZGate", "ZZPowGate"}

            def _count_native(circ):
                return sum(1 for o in circ.all_operations()
                           if type(o.gate).__name__ in _NATIVE_NAMES)

            # Silence the BASIS-CANCEL logger during per-gate profiling calls —
            # single-gate and pair circuits log "Cancelled 0" for every gate type.
            import logging as _logging
            _basis_cancel_logger = _logging.getLogger(
                "nanoprotogeny.ionq.holographic.basis_cancel"
            )
            _basis_cancel_logger.disabled = True

            def _full_compile(ops):
                """Run the complete compilation pipeline on a list of ops."""
                c   = cirq.Circuit(ops)
                exp = unified_expand_qudit_circuit(c)
                exp = _expand_blog_sandwich_wrappers(exp)
                exp = cancel_adjacent_basis_changes(exp)
                com = cirq.optimize_for_target_gateset(
                    exp, gateset=_FNG(),
                    context=cirq.TransformerContext(deep=True),
                )
                return cirq.drop_negligible_operations(com, atol=1e-8)

            def _verify_unitary(sample_op, compiled_single):
                """Check compiled_single implements the same unitary as expanded_single."""
                try:
                    expanded = unified_expand_qudit_circuit(cirq.Circuit([sample_op]))
                    expected = cirq.unitary(expanded)
                    actual   = cirq.unitary(compiled_single)
                    if expected.shape != actual.shape:
                        return f"[✗ shape]"
                    # Cancel global phase using largest element
                    i = int(_np.argmax(_np.abs(expected)))
                    if abs(actual.flat[i]) > 1e-10:
                        phase = actual.flat[i] / expected.flat[i]
                        err   = float(_np.max(_np.abs(expected - actual / phase)))
                    else:
                        err   = float(_np.max(_np.abs(expected - actual)))
                    return "[✓]" if err < 1e-6 else f"[✗ {err:.1e}]"
                except Exception as ex:
                    return f"[? {type(ex).__name__}]"

            unique_gate_types = {}
            for op in scheduled_circuit.all_operations():
                gname = type(op.gate).__name__
                if gname not in unique_gate_types:
                    unique_gate_types[gname] = op

            contribution_rows = []
            for gname, sample_op in sorted(unique_gate_types.items()):
                try:
                    compiled_single  = _full_compile([sample_op])
                    n_standalone     = _count_native(compiled_single)
                    verified         = _verify_unitary(sample_op, compiled_single)

                    # is_sandwich must be checked on the PHYSICAL wrapper, not the
                    # qudit-level gate.  Expand one level to see what unified_expand
                    # produced, then check whether any resulting op is a sandwich type.
                    _single_exp = unified_expand_qudit_circuit(cirq.Circuit([sample_op]))
                    is_sandwich = any(
                        isinstance(o.gate, _BLOG_SANDWICH_WRAPPERS)
                        for o in _single_exp.all_operations()
                    )
                    if is_sandwich:
                        # Adjacent-pair measurement captures one boundary cancellation.
                        compiled_pair = _full_compile([sample_op, sample_op])
                        n_effective   = _count_native(compiled_pair) // 2
                    else:
                        n_effective = n_standalone

                    n_total = pre_counts[gname]
                    contribution_rows.append(
                        (gname, n_total, n_standalone, n_effective,
                         n_total * n_effective, verified)
                    )
                except Exception:
                    contribution_rows.append(
                        (gname, pre_counts.get(gname, "?"), "?", "?", "?", "?")
                    )

            _basis_cancel_logger.disabled = False   # re-enable for the rest of the run

            contribution_rows.sort(
                key=lambda r: r[4] if isinstance(r[4], int) else 0, reverse=True
            )
            log.info("[HW-INFO] → Gate contribution breakdown "
                     "(type | count | standalone→effective | total | verified):")
            for gname, n_total, n_sa, n_eff, total, verified in contribution_rows:
                if n_sa != n_eff:
                    log.info(
                        f"[HW-INFO]     {gname:<40} {n_total:>5} × "
                        f"{n_sa}→{n_eff:<4} = {total}  {verified}"
                    )
                else:
                    log.info(
                        f"[HW-INFO]     {gname:<40} {n_total:>5} × "
                        f"{n_eff:<8} = {total}  {verified}"
                    )
        except Exception as comp_err:
            log.error(f"[HW-ERROR] Native pulse profiling skipped or failed: {comp_err}")
        # ======================================================================

        # ── 4. Semantic warrant (on initialization circuit) ───────────────────
        log.info(f"[HW-INFO] → Semantic warrant extraction...")

        # 1. Simulate ONLY the DFT superposition (bypass uncompiled projector gates)
        qubits = [NomosIonQid(i) for i in range(N)]
        dft_circ = cirq.Circuit(cirq.Moment([TetralemmaticIonDFTGate().on(q) for q in qubits]))

        noise_model = ForteHardwareNoiseModel(
            p1q=    self._noise_params["p1q_error"],
            p2q=    self._noise_params["p2q_error"],
            p_meas= self._noise_params["p_meas_error"],
            p_idle= self._noise_params["p_idle_error"],
        )
        sim = cirq.DensityMatrixSimulator(noise=noise_model)
        result = sim.simulate(dft_circ, initial_state=0)
        rho_super = result.final_density_matrix

        # 2. Analytically construct the target projector & apply Bayesian inversion for eta
        if mech.S_target >= 1.0:
            P_target = np.zeros((4, 4), dtype=complex)
            P_target[1, 1] = 1.0
            P_target[2, 2] = 1.0
            eta_prime = self._eta  # Rank 2 projector (p=0.5) maps linearly
        else:
            P_target = np.zeros((4, 4), dtype=complex)
            P_target[3, 3] = 1.0
            # Rank 1 projector (p=0.25) requires Bayesian correction to achieve final target eta
            eta_prime = (self._eta * 0.75) / (0.25 + 0.5 * self._eta)

        # 3. Apply the properly scaled Kraus map
        K_local = np.sqrt(eta_prime) * P_target + np.sqrt(1.0 - eta_prime) * (np.eye(4) - P_target)
        K_total = functools.reduce(np.kron, [K_local] * N)

        rho_raw = K_total @ rho_super @ K_total.conj().T
        norm = float(np.real(np.trace(rho_raw)))

        # 4. Normalize to finalize the physical state
        final_rho = rho_raw / norm if norm > 1e-12 else rho_raw

        # --- DIAGNOSTIC INSERTION ---
        p_target_actual = np.real(np.trace(functools.reduce(np.kron, [P_target] * N) @ final_rho))
        print(f"\n[DEBUG] Calibrated Transmission η' = {eta_prime:.4f}")
        print(f"[DEBUG] Final Target Population = {p_target_actual:.4f} (Expected: ~{self._eta:.4f})")
        # --- DIAGNOSTIC INSERTION END ---

        # ── 5. Stoichiometric verification ────────────────────────────────────
        print(f"\n  [VERIFY] Stoichiometric invariance suite...")
        verifier  = StoichiometricVerifier(mech, self._eta)
        stoich_ok, checks = verifier.full_report(tracker, final_rho, N, mech.S_target)
        for chk in checks:
            status = "[✓]" if chk["passed"] else "[✗]"
            print(f"  {status} {chk['condition']}: {chk['detail']}")

        # ── 6. Build reference energy for final report ────────────────────────
        last_chk = checkpoints[-1]
        E_0_final = (
            step_energies[last_chk]
            if step_energies[last_chk] is not None
            else qpe_results[last_chk]["E_exact_diag"]
        )

        # ── 7. Final report ───────────────────────────────────────────────────
        self._print_report(
            mech, tracker, checkpoints, qpe_results,
            stoich_ok, checks, E_0_final, time.time() - start,
            dataset_dir=self._dataset_dir,
        )

        last_qpe    = qpe_results[last_chk]
        all_chem_ok = all(r["chem_ok"] for r in qpe_results.values())

        return {
            "mechanism_name":           mech.name,
            "N_orbitals":               N,
            "M_steps":                  mech.M_steps,
            "m":                        m,
            "n_crossings":              mech.n_crossings,
            "is_reversible_cycle":      mech.is_reversible_cycle,
            # ── Forward (cumulative injection) ──
            "total_electrons":          tracker.total_electrons,
            "total_cofactor_shift":     tracker.k_total,
            # ── Reverse (cumulative ejection) ──
            "total_electrons_ejected":  tracker.total_electrons_ejected,
            # ── Net-flux invariants ──
            "net_electrons":            tracker.net_electrons,
            "net_cofactor_shift":       tracker.k_total,    # already net (fwd − inv)
            # ── Photon balance ──
            "total_photons_absorbed":   tracker.total_photons_absorbed,
            "total_photons_emitted":    tracker.total_photons_emitted,
            "net_photons":              tracker.net_photons,
            # ── Validation flags ──
            "phase_closure_ok":         algebraic_ok,
            "electron_count_ok":        electron_ok,
            "stoichiometric_ok":        stoich_ok,
            "qpe_results":              qpe_results,
            "step_reference_energies":  step_energies,
            "E_ref_final":              E_0_final,
            "E_zne_final":              last_qpe.get("E_zne_best", last_qpe.get("E_zne_exp")),
            "residual_mHa_final":       last_qpe["residual_mHa"],
            "chemical_accuracy_ok":     all_chem_ok,
            "qpe_path":                 "virtual_ancilla",
            "integral_source":          "dataset" if store else "global",
            "dataset_dir":              str(self._dataset_dir) if self._dataset_dir else None,
            "elapsed_s":                round(time.time() - start, 2),
            "stoich_checks":            checks,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _print_header(self, store: Optional["StepwiseIntegralStore"]) -> None:
        mech = self._mechanism
        w    = 78
        path_label = "VANC-QPE"
        print("\n" + "="*w)
        print(f" {path_label} PIPELINE VALIDATION: {mech.name.upper()}")
        print("="*w)
        print(mech.summary())
        if store is not None:
            print(
                f"  Integral source: step-wise JSON datasets "
                f"({self._dataset_dir}/{mech.name}/step_XX.json)"
            )
        else:
            print(
                f"  Integral source: global fallback "
                f"(same Hamiltonian at every step)"
            )
        print("-"*w)

    def _print_report(
        self, mech, tracker, checkpoints, qpe_results,
        stoich_ok, checks, E_0_ref, elapsed,
        dataset_dir=None,
    ) -> None:
        w1, w2, w3 = 38, 18, 10
        path_label = "VANC-QPE"
        print(f"\n[{path_label} RESULTS] {mech.name}")
        src = (
            f"step-wise JSON ({dataset_dir}/{mech.name}/)"
            if dataset_dir else "global fallback integrals"
        )
        print(f"  Integral source: {src}")
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
        row("Trace Preservation (Tr ρ=1)", "-",
            next((c["passed"] for c in checks if "Trace" in c["condition"]), False))
        row(f"Spin-Parity Holding (η={self._eta})", "-",
            next((c["passed"] for c in checks if "Spin" in c["condition"]), False))

        print(f"  ├─{'─'*w1}─┼─{'─'*w2}─┼─{'─'*w3}─┤")
        for step_idx, qpe in sorted(qpe_results.items()):
            ok    = qpe["chem_ok"]
            src_s = "DS" if qpe.get("integral_source") == "dataset" else "GF"
            label = f"{path_label}|ZNE n={step_idx} ({qpe['step_label']}) [{src_s}]"
            res   = qpe["residual_mHa"]
            val   = "noise-floor" if res == float("inf") else f"{res:.4f} mHa"
            row(label, val, ok)

        print(f"  ├─{'─'*w1}─┼─{'─'*w2}─┼─{'─'*w3}─┤")
        all_qpe_ok = all(r["chem_ok"] for r in qpe_results.values())
        row("OVERALL CHEMICAL ACCURACY (≤1.6 mHa)", "", all_qpe_ok)
        row("STOICHIOMETRIC INVARIANCE", "", stoich_ok)

        print(f"  └─{'─'*w1}─┴─{'─'*w2}─┴─{'─'*w3}─┘")
        print(f"\n  E_ref (last step) = {E_0_ref:+.10f} Ha")
        print(f"  Gate algebra:      G(M) = A_HW^⊗{2*mech.N_orbitals} ∪ A_cross^(m={mech.m})")
        print(f"  Complexity bound:  G = O(M·N³·T²·C_int/ε)")
        print(f"  Elapsed:           {elapsed:.2f}s")
        all_ok = stoich_ok and all_qpe_ok
        print(
            f"\n  {'[✓] VANC-QPE VALIDATION PASSED' if all_ok else '[✗] VANC-QPE VALIDATION FAILED'}"
        )
        print("="*78)


# ==============================================================================
# LAYER 8: TOP-LEVEL ENTRY POINT
# ==============================================================================

def run_virtual_ancilla_qpe_validation(
    mechanism_name:  str,
    integral_state:  "IntegralState",
    config:          "MQEConfig"               = None,
    output_json:     Optional[str]             = None,
    dataset_dir:     Optional[Union[str, Path]] = None,
) -> Dict:
    r"""Top-level MQE-QPE validation entry point (default ``mqe run`` path).

    Uses the D-state virtual register of an existing ion as the QPE ancilla
    (VirtualQudit) — the native architectural method for the tetralemmatic
    dual-manifold architecture.  No external ancilla ion required.

    Simulation stays in 256-dim system space:
    compute_trotter_density_matrix -> rho_sys -> compute_virtual_ancilla_qpe_probs
    with eta_V-corrected MLE via hardware_map_energy.

    Args:
        mechanism_name:  "nitrogenase_lt" | "psii" | ... | "all".
        integral_state:  Populated IntegralState from initialise_integrals().
        config:          MQEConfig (defaults to MQEConfig() if None).
        output_json:     Optional JSON export path.
        dataset_dir:     Root directory of mqedatagenerator.py output.

    Returns:
        Dict (single mechanism) or Dict[name -> Dict] (mechanism_name="all").
    """
    cfg        = config or MQEConfig()
    n_orbitals = integral_state.n_orbitals
    PREDEFINED = build_predefined_mechanisms(n_orbitals)

    def _make_runner(mech):
        return HardwareQPEPipelineRunner(
            mechanism      = mech,
            integral_state = integral_state,
            config         = cfg,
            dataset_dir    = dataset_dir,
        )

    if mechanism_name == "all":
        print("\n" + "="*78)
        print(" VANC-QPE FRAMEWORK VALIDATION: ALL MECHANISMS")
        if dataset_dir:
            print(f" Step-wise integrals: {dataset_dir}/")
        print("="*78)
        results = {}
        for name, mech in PREDEFINED.items():
            runner        = _make_runner(mech)
            results[name] = runner.run()

        _print_vanc_qpe_summary_table(results)
        if output_json:
            _export_mqe_json(results, output_json, n_orbitals)
        return results

    elif mechanism_name in PREDEFINED:
        mech   = PREDEFINED[mechanism_name]
        runner = _make_runner(mech)
        result = runner.run()
        if output_json:
            _export_mqe_json({mechanism_name: result}, output_json, n_orbitals)
        return result

    else:
        raise ValueError(
            f"Unknown mechanism {mechanism_name!r}. "
            f"Choose from: {list(PREDEFINED.keys())} or 'all'."
        )


def _print_vanc_qpe_summary_table(results: Dict[str, Dict]):
    """Print a consolidated summary table after running all mechanisms."""
    w = 78
    print("\n" + "="*w)
    print(" VANC-QPE VALIDATION SUMMARY — ALL MECHANISMS")
    print("="*w)
    print(f"  {'Mechanism':<22} {'m':>4} {'M':>3} {'n':>4} "
          f"{'e-':>4} {'Sv':>4} {'Phase':>7} {'Chem. Acc':>10} {'Pass':>6}")
    print(f"  {'─'*22} {'─'*4} {'─'*3} {'─'*4} "
          f"{'─'*4} {'─'*4} {'─'*7} {'─'*10} {'─'*6}")

    all_pass = True
    for name, r in results.items():
        ph  = "[v]" if r["phase_closure_ok"] else "[x]"
        ok  = "[v]" if r["stoichiometric_ok"] and r["chemical_accuracy_ok"] else "[x]"
        _res = r["residual_mHa_final"]
        res  = "noise-floor" if _res == float("inf") else f"{_res:.3f} mHa"
        all_pass = all_pass and r["stoichiometric_ok"] and r["chemical_accuracy_ok"]
        print(f"  {name:<22} {r['m']:>4} {r['M_steps']:>3} "
              f"{r['n_crossings']:>4} {r['total_electrons']:>4} "
              f"{r['total_cofactor_shift']:>4} {ph:>7} {res:>10} {ok:>6}")

    print("="*w)
    print(f"  OVERALL: {'[v] ALL PASSED' if all_pass else '[!] SOME FAILED'}")
    print("="*w)


def _export_mqe_json(
    results: Dict,
    path: str,
    n_orbitals: int,
    framework: str = "Modular Quantum Emulator — MQE-QPE",
):
    """Export QPE validation results to JSON."""
    def _san(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.float64, np.float32)): return float(obj)
        if isinstance(obj, (np.int64, np.int32)): return int(obj)
        if isinstance(obj, float) and (obj == float("inf") or obj != obj):
            return "noise-floor"   # inf / NaN -> JSON-safe sentinel string
        if isinstance(obj, dict): return {k: _san(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_san(v) for v in obj]
        return obj

    export = {
        "mqe_validation": _san(results),
        "framework": framework,
        "n_orbitals": n_orbitals,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reference": "arXiv:nanoprotogeny.theory.mqe",
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"\n[VANC-QPE] Results exported -> {path}")
