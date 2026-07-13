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
mqeconfig.py — MQE Simulation Constants and State Dataclasses
==============================================================
Two responsibilities:

1. Pure compile-time constants (no I/O, no side-effects on import).
   Separated from mqe.py so algorithm parameters can be inspected and
   tuned without loading the full simulation stack.

2. ``MQEConfig`` and ``IntegralState`` dataclasses that replace the
   module-level mutable globals previously scattered across mqe.py.
   ``MQEConfig`` is immutable by convention (frozen=True).
   ``IntegralState`` is populated once by ``initialise_integrals()``
   and passed explicitly into the pipeline.

Usage
-----
    from nanoprotogeny.simulate.mqeconfig import (
        MQEConfig, IntegralState,
        ETA, IDLE_THRESHOLD, TAU_SEQ,
        N_STEPS, BASE_DT, DT, T_TOTAL, EPS_TROTTER_REF,
        SPIN_ACTIVE_CORNERS, SPIN_PARITY_SECTOR, UR_onto,
    )

    cfg   = MQEConfig()                     # defaults from module constants
    state = initialise_integrals(cfg, ...)  # returns populated IntegralState
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from nanoprotogeny.theory.algebra import Vertex

# ── Core Algorithmic Parameters ──────────────────────────────────────────────
ETA             = 0.90          # Semantic warrant threshold (spin-parity holding)
IDLE_THRESHOLD  = 8             # Max idle moments before Zeno stabilisation fires
P0_BASE         = 0.01          # Base leakage probability for error model

# ── Path A (mqe.py) fixed τ-sequence ─────────────────────────────────────────
# Used directly by MQEPipelineRunner (Path A) and as the fallback for Path B'.
# Empirically validated as the "Goldilocks zone": three τ points spanning a
# decade give bayesian_map_energy sufficient frequency resolution while keeping
# n_max = round(0.08/0.02) = 4 Trotter steps shallow enough for Richardson ZNE.
#
# Earlier experiments (preserved for reference):
#   [0.02, 0.04, 0.08, 0.16, 0.32]  n_max=16  too deep: non-monotone λ-series
#   [0.02, 0.04, 0.08, 0.16]        n_max=8   within budget but noisier
#   [0.02, 0.04, 0.08]              n_max=4   ← Goldilocks (current)
#   [0.02, 0.04]                    n_max=2   ethylene epoxidation proxy
#   [0.02]                          n_max=1   RNR radical proxy
TAU_SEQ: list = [0.02, 0.04, 0.08]

# ── Path B' (mqe_multitau.py) candidate pool ─────────────────────────────────
# select_tau_sequence() (qpe/mqemultitauqpe.py) probes these in descending
# order per checkpoint and returns the longest sequence whose Richardson
# denominator is well-conditioned.  The winning sequence determines n_max for
# that checkpoint's density-matrix simulation.
TAU_SEQ_CANDIDATES: list = [0.02, 0.04, 0.08, 0.16, 0.32]

SPIN_ACTIVE_CORNERS = [Vertex.AntiTh, Vertex.SynTh]  # Logical corners in spin sector
SPIN_PARITY_SECTOR  = [Vertex.AntiTh, Vertex.SynTh]  # Parity-even spin sector

# Ontological reflection operator  (ℤ_4 cyclic permutation on d=4 logical manifold)
UR_onto = np.array(
    [[0, 0, 0, 1],
     [1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0]],
    dtype=complex,
)

# ── Trotter Step Scaling Configuration ───────────────────────────────────────
N_STEPS       = 4                           # Number of Trotter steps per QPE call
BASE_DT       = 0.04                        # Base time step before Richardson scaling
DT            = BASE_DT / np.sqrt(N_STEPS)  # Effective step size after scaling
T_TOTAL       = N_STEPS * DT                # Total evolution time per QPE call
EPS_TROTTER_REF = 0.4                       # Trotter error reference threshold (mHa)


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MQEConfig:
    """Immutable algorithm configuration for the MQE pipeline.

    Defaults mirror the module-level constants above.  Override at the
    call site to experiment with different Trotter settings or thresholds
    without touching the source constants.

    Example::

        cfg = MQEConfig(eta=0.95, n_steps=8)
    """
    eta:                  float  = ETA
    idle_threshold:       int   = IDLE_THRESHOLD
    p0_base:              float = P0_BASE
    # Path A fixed sequence (also the Path B' fallback when all candidates fail)
    tau_seq:              tuple = tuple(TAU_SEQ)
    # Path B' candidate pool handed to select_tau_sequence() per checkpoint
    tau_seq_candidates:   tuple = tuple(TAU_SEQ_CANDIDATES)
    n_steps:              int   = N_STEPS
    base_dt:              float = BASE_DT
    dt:                   float = DT
    t_total:              float = T_TOTAL
    eps_trotter_ref:      float = EPS_TROTTER_REF

    # Derived convenience — not settable directly (use base_dt / n_steps)
    @property
    def tau_seq_list(self) -> List[float]:
        return list(self.tau_seq)


@dataclass
class IntegralState:
    """Populated by ``initialise_integrals()``.

    Replaces the module-level mutable globals (H_DIAG, H_HOP, G_FULL, …)
    that were previously scattered across mqe.py.  Pass this object
    explicitly into ``run_mqe_validation`` and ``MQEPipelineRunner``
    instead of threading the individual dicts.

    Fields
    ------
    h_diag      : {orbital_idx: energy}  — diagonal one-body integrals
    h_hop       : {(p,q): t_pq}          — off-diagonal hopping integrals
    g_full      : {(p,q,r,s): g_pqrs}    — full two-electron repulsion integrals
    g_coul      : {(p,r): J_pr}          — Coulomb integrals (diagonal ERI)
    e_core      : core energy (Ha)
    n_orbitals  : active-space orbital count N
    meta        : raw metadata dict from the integral JSON
    fci_reference : keys h1, eri, nalpha, nbeta, E_active, E_absolute
    integrals   : raw JSON dict (non-empty only for static load path)
    dataset_mode: True when StepwiseIntegralStore supplies per-step H_n
    """
    h_diag:       Dict[int, float]                        = field(default_factory=dict)
    h_hop:        Dict[Tuple[int, int], float]            = field(default_factory=dict)
    g_full:       Dict[Tuple[int, int, int, int], float]  = field(default_factory=dict)
    g_coul:       Dict[Tuple[int, int], float]            = field(default_factory=dict)
    e_core:       float                                   = 0.0
    n_orbitals:   int                                     = 4
    meta:         Dict                                    = field(default_factory=dict)
    fci_reference: Dict                                   = field(default_factory=dict)
    integrals:    Dict                                    = field(default_factory=dict)
    dataset_mode: bool                                    = False
