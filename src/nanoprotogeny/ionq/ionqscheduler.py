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
ionqscheduler.py — Qudit Circuit DAG Construction and Parallel Moment Scheduling
=================================================================================
General-purpose, IonQ-hardware-aware scheduling primitives for d=4 qudit circuits.
These functions are independent of any specific Hamiltonian or catalytic mechanism
and can be applied to any sequence of cirq operations.

Functions
---------
build_qudit_dependency_dag(circuit_ops)
    Builds a directed acyclic graph (DAG) over a flat list of cirq operations.
    Edges encode qudit-overlap and fold-atomic ordering constraints.
    Nodes are annotated with their qudit footprint and whether they are
    entangling (Mølmer-Sørensen / two-qudit) gates.

schedule_parallel_moments(G, max_concurrent_ms=4)
    Topological level-based scheduling with MS-gate budget enforcement.
    Packs non-conflicting operations into parallel cirq.Moment objects,
    prioritising entangling gates and respecting the hardware concurrency limit.

Dependencies: cirq, networkx, collections,
              nanoprotogeny.ionq.ionqsumgate,
              nanoprotogeny.ionq.ionqcnotgate,
              nanoprotogeny.ionq.ionqczgate,
              nanoprotogeny.ionq.ionqswapgate,
              nanoprotogeny.ionq.ionqcrossgates,
              nanoprotogeny.ionq.ionqparamgates,
              nanoprotogeny.ionq.ionqmqegates.
No simulate-layer imports.
"""

from __future__ import annotations

import cirq
import networkx as nx
from collections import defaultdict
from typing import List

from nanoprotogeny.ionq.ionqsumgate import TetralemmaticIonSUMGate
from nanoprotogeny.ionq.ionqcnotgate import TetralemmaticIonCNOTGate
from nanoprotogeny.ionq.ionqczgate import TetralemmaticIonCZGate
from nanoprotogeny.ionq.ionqswapgate import TetralemmaticIonSWAPGate
from nanoprotogeny.ionq.ionqcrossgates import PhaseSwapGate
from nanoprotogeny.ionq.ionqparamgates import ParamScatteringGate, ParamCoulombPhaseGate, PowerControlledGate
from nanoprotogeny.ionq.ionqmqegates import (
    CofactorCouplingGate,
    CofactorDecouplingGate,
    CrossManifoldSWAPGate,
)

def build_qudit_dependency_dag(circuit_ops: List[cirq.Operation]) -> nx.DiGraph:
    G = nx.DiGraph()
    qudit_last_use: dict[tuple, int | None] = defaultdict(lambda: None)

    for idx, op in enumerate(circuit_ops):
        # ── Robust Footprint Extraction (Duck-typed for any indexed Qid)
        support = tuple(sorted(
            (type(q).__name__, q._index)
            for q in op.qubits
            if hasattr(q, "_index")  # Safe against generic cirq.Qid variants
        ))

        is_entangling = isinstance(op.gate, (
            TetralemmaticIonSUMGate,
            TetralemmaticIonCNOTGate,
            TetralemmaticIonCZGate,
            TetralemmaticIonSWAPGate,
            ParamScatteringGate,
            ParamCoulombPhaseGate,
            PhaseSwapGate,
            PowerControlledGate,
            CofactorCouplingGate,      # Added for MQE multi-body
            CofactorDecouplingGate,    # Added for MQE multi-body
            CrossManifoldSWAPGate      # Added for MQE multi-body
        ))

        G.add_node(idx, op=op, support=support, is_entangling=is_entangling)

        for key in support:
            last = qudit_last_use[key]
            if last is not None:
                G.add_edge(last, idx, type="qudit_overlap")
            qudit_last_use[key] = idx

        if hasattr(op.gate, "_fold_group"):
            for prev_idx in range(idx):
                if (G.has_node(prev_idx)
                        and hasattr(G.nodes[prev_idx]["op"].gate, "_fold_group")
                        and G.nodes[prev_idx]["op"].gate._fold_group == op.gate._fold_group
                        and not G.has_edge(prev_idx, idx)):
                    G.add_edge(prev_idx, idx, type="fold_atomic")

    return G


def schedule_parallel_moments(
    G: nx.DiGraph,
    max_concurrent_ms: int = 4,
) -> List[cirq.Moment]:
    """
    Level-based topological scheduling with MS-gate budget enforcement.
    Returns a list of cirq.Moment objects ready for circuit assembly.
    """
    moments: List[cirq.Moment] = []
    scheduled: set[int] = set()
    remaining: set[int] = set(G.nodes())

    while remaining:
        # ── Ready set: all predecessors already scheduled
        ready = {
            n for n in remaining
            if all(p in scheduled for p in G.predecessors(n))
        }

        if not ready:
            # Should never happen in a valid DAG; guard against cycles
            raise RuntimeError("Cycle detected in dependency DAG — check ZNE fold edges.")

        current_ops: List[cirq.Operation] = []
        dispatched: set[int] = set()
        used_qudits: set[tuple] = set()
        ms_count = 0

        # ── Heuristic: prioritise entangling gates (higher MS cost) first
        ordered_ready = sorted(
            ready,
            key=lambda n: (not G.nodes[n]["is_entangling"], n)
        )

        for node in ordered_ready:
            support = set(G.nodes[node]["support"])
            is_entangling = G.nodes[node]["is_entangling"]

            fits_qudits   = support.isdisjoint(used_qudits)
            fits_ms_budget = (not is_entangling) or (ms_count < max_concurrent_ms)

            if fits_qudits and fits_ms_budget:
                current_ops.append(G.nodes[node]["op"])
                dispatched.add(node)
                used_qudits.update(support)
                if is_entangling:
                    ms_count += 1

        # ── Deadlock guard: if nothing fit, force-schedule highest-priority ready node
        if not current_ops:
            node = ordered_ready[0]
            current_ops.append(G.nodes[node]["op"])
            dispatched.add(node)

        moments.append(cirq.Moment(current_ops))
        scheduled.update(dispatched)
        remaining -= dispatched

    return moments

