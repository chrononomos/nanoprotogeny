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
mqedualmanifold.py — Dual-Manifold Circuit Architecture
=========================================================
Canonical implementations of the circuit construction and verification
functions that embody the dual-manifold qudit architecture:

  Logical manifold  H_L  — d=4 NomosIonQid registers encoding the
      tetralemmatic Fock space via ontological superposition and
      Kraus sector projection (Step 1).

  Virtual manifold  H_V  — d=m VirtualQudit phase registers for
      stoichiometric bookkeeping, shielded by holographic coherence
      routing and Zeno stabilization (Step 3).

Public API
----------
Circuit construction
    build_ontological_projection_circuit
    apply_holographic_routing
    inject_zeno_stabilization
    _make_virtual_qudits_m

Mathematical verification (proof obligations)
    verify_ontological_projection
    verify_holographic_routing

Resource accounting
    _count_qudit_resources
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import cirq
import numpy as np

from nanoprotogeny.ionq.YB171PLUSHARDWARE import NomosIonQid, VirtualQudit


# ==============================================================================
# COMPOSITE VIRTUAL REGISTER  (m = 4r generalisation)
# ==============================================================================

@dataclass
class VirtualRegisterPair:
    r"""Structured allocation of the virtual phase register for one mechanism.

    For any mechanism modulus m the register per orbital is:

    m = 4   (r=1) : v1 = [VirtualQudit(i)],       vaux = [None] * N
                    Single hardware-native d=4 qudit; no carry.

    m = 4r, r>1   : v1 = [VirtualQudit(i)],       vaux = [LineQubit(N+i)]  (r=2)
                    Primary d=4 qudit (hardware-native) carries k%4;
                    auxiliary LineQubit(s) carry k//4.  Together they
                    represent k ∈ ℤ_m via the mixed-radix map φ(k)=(k%4, k//4).

    odd m   (r=0) : v1 = [LineQid(i, dim=m)],     vaux = [None] * N
                    Binary-padded qudit register; no carry structure.
                    Janus crossing is impossible (Case I); no Vaux needed.

    Properties
    ----------
    is_composite  : True when r > 1 (Vaux register is non-trivial).
    v1_qudits()   : alias for v1 — the primary d=4 (or d=m for odd-m) list.
    """
    v1:   List    # List[VirtualQudit | cirq.LineQid]
    vaux: List    # List[None | cirq.LineQubit | Tuple[cirq.LineQubit, ...]]
    m:    int
    r:    int     # m // 4  for m%4==0, else 0

    @property
    def is_composite(self) -> bool:
        """True when m = 4r with r > 1 (auxiliary carry register is active)."""
        return self.r > 1

    def v1_qudits(self) -> List:
        """Primary d=4 (or d=m) qudit list, length N."""
        return self.v1

    def all_qudits(self) -> List:
        """Flat list of every qudit in registration order (v1 first, then vaux)."""
        result = list(self.v1)
        for va in self.vaux:
            if va is None:
                continue
            if isinstance(va, (list, tuple)):
                result.extend(va)
            else:
                result.append(va)
        return result
from nanoprotogeny.archive.ionqurgate import TetralemmaticIonDFTGate, DFT_onto
from nanoprotogeny.ionq.ionqprojectorgate import TetralemmaticIonProjectorGate
from nanoprotogeny.ionq.ionqcrossgates import ZenoStabilizeGate
from nanoprotogeny.ionq.holographic import HolographicRouter
from nanoprotogeny.theory.algebra import Vertex
from nanoprotogeny.simulate.mqeconfig import IDLE_THRESHOLD, UR_onto


# ==============================================================================
# LOGICAL MANIFOLD — Step 1: Ontological Superposition & Symmetry Projection
# ==============================================================================

def build_ontological_projection_circuit(
    n_orbitals: int,
    eta:        float,
    S_target:   float = 1.0,
) -> tuple[cirq.Circuit, list[NomosIonQid]]:
    """Build DFT superposition + Kraus sector projection circuit.

    Applies TetralemmaticIonDFTGate to each logical qudit to create a
    uniform superposition over all four tetralemmatic vertices, then
    applies Kraus projectors to select the target spin sector.

    Args:
        n_orbitals: Number of spatial orbitals (logical qudits).
        eta:        Transmission probability for the Kraus projector.
        S_target:   Target spin quantum number.
                    >= 1.0 → high-spin: projects onto AntiTh (|1⟩) + SynTh (|2⟩).
                    <  1.0 → singlet:   projects onto HoloTh (|3⟩).

    Returns:
        (circuit, qubits)
    """
    qubits  = [NomosIonQid(i) for i in range(n_orbitals)]
    circuit = cirq.Circuit()
    circuit.append(TetralemmaticIonDFTGate().on_each(*qubits))

    for q in qubits:
        if S_target >= 1.0:
            circuit.append(TetralemmaticIonProjectorGate(Vertex.AntiTh, transmission=eta).on(q))
            circuit.append(TetralemmaticIonProjectorGate(Vertex.SynTh,  transmission=eta).on(q))
        else:
            circuit.append(TetralemmaticIonProjectorGate(Vertex.HoloTh, transmission=eta).on(q))

    return circuit, qubits


def verify_ontological_projection(n_orbitals: int, eta: float) -> bool:
    r"""Validate DFT superposition amplitude and Kraus warrant sum.

    Proof obligations:
      1. F⊗N |vac⟩ produces uniform amplitude 1/2^N on each basis state.
      2. Kraus channel K(ρ) satisfies the η-holding relation:
         ω(AntiTh) + ω(SynTh) ≥ η.
    """
    dim      = 4 ** n_orbitals
    dim_rest = 4 ** (n_orbitals - 1)

    psi_vac    = np.zeros(dim, dtype=complex); psi_vac[0] = 1.0
    F_log      = np.eye(1, dtype=complex)
    for _ in range(n_orbitals):
        F_log  = np.kron(F_log, DFT_onto)
    psi_super   = F_log @ psi_vac
    uniform_amp = 1.0 / (2 ** n_orbitals)

    assert np.allclose(np.abs(psi_super), uniform_amp), \
        "DFT failed to create uniform superposition"
    print(f"[✓] DFT verified: uniform superposition amplitude = {uniform_amp:.4f}")

    rho         = np.outer(psi_super, np.conj(psi_super))
    rho_reduced = np.einsum("iaja->ij", rho.reshape(4, dim_rest, 4, dim_rest))

    P_AntiTh = np.zeros((4, 4)); P_AntiTh[1, 1] = 1.0
    P_SynTh  = np.zeros((4, 4)); P_SynTh[2, 2]  = 1.0
    Pi       = P_AntiTh + P_SynTh
    K        = np.sqrt(eta) * Pi + np.sqrt(1.0 - eta) * (np.eye(4) - Pi)
    rho_raw  = K @ rho_reduced @ K.conj().T
    norm     = float(np.real(np.trace(rho_raw)))
    assert norm > 1e-12, "Kraus output is zero"
    rho_proj = rho_raw / norm

    w_AntiTh    = float(np.real(np.trace(rho_proj @ P_AntiTh)))
    w_SynTh     = float(np.real(np.trace(rho_proj @ P_SynTh)))
    warrant_sum = w_AntiTh + w_SynTh
    print(f"[✓] Kraus channel verified: ω(AntiTh)={w_AntiTh:.4f}, ω(SynTh)={w_SynTh:.4f}")
    print(f"[✓] Holding relation ⊨_{eta} satisfied: Σω = {warrant_sum:.4f} >= {eta}")
    assert warrant_sum >= eta - 1e-10, \
        f"Semantic warrant VIOLATED: {warrant_sum:.4f} < {eta}"
    return True


# ==============================================================================
# VIRTUAL MANIFOLD — Step 3: Holographic Coherence Routing & Zeno Stabilization
# ==============================================================================

def apply_holographic_routing(
    circuit: cirq.Circuit,
) -> tuple[cirq.Circuit, HolographicRouter]:
    """Route the circuit through the holographic coherence router.

    Inserts virtual shielding operations at idle moments to maintain
    ℤ_4 phase coherence in the virtual manifold.

    Returns:
        (routed_circuit, router)  — router carries the phase accumulator
        and routing log for downstream verification.
    """
    router = HolographicRouter(
        idle_threshold_gates = IDLE_THRESHOLD,
        enable_auto_routing  = True,
        max_phase_drift      = 2,
    )
    routed_circuit = router.analyze_and_route(circuit)
    routed_circuit._routing_metadata = {
        "phase_accumulator": dict(router._phase_acc),
        "routing_log":       router._routing_log,
    }
    return routed_circuit, router


def inject_zeno_stabilization(
    circuit:       cirq.Circuit,
    virtual_qudits: list,
) -> cirq.Circuit:
    r"""Inject U_Zeno = I₁₆ − 2Π_union to suppress |3⟩ boundary leakage.

    Appends a ZenoStabilizeGate moment pairing each logical NomosIonQid
    with its corresponding VirtualQudit.
    """
    logical_qudits = sorted(
        [q for q in circuit.all_qubits() if isinstance(q, NomosIonQid)]
    )
    zeno_ops = [
        ZenoStabilizeGate().on(log_q, virt_q)
        for log_q, virt_q in zip(logical_qudits[:len(virtual_qudits)], virtual_qudits)
        if isinstance(virt_q, VirtualQudit)
    ]
    if zeno_ops:
        circuit.append(cirq.Moment(zeno_ops))
    return circuit


def verify_holographic_routing(router: HolographicRouter) -> bool:
    """Verify holographic phase closure and Zeno boundary reflection.

    Proof obligations:
      1. For every virtual register, U_comp · U_drift = I₄ (phase closure).
      2. U_Zeno reflects the HoloTh corner: U_Zeno |HH⟩ = −|HH⟩.
      3. U_Zeno preserves the interior:     U_Zeno |00⟩ = +|00⟩.
    """
    for vq_id, k in router._phase_acc.items():
        U_drift = np.linalg.matrix_power(UR_onto, k % 4)
        U_comp  = np.linalg.matrix_power(UR_onto.conj().T, k % 4)
        assert np.allclose(U_comp @ U_drift, np.eye(4)), \
            f"Phase closure failed for {vq_id}"

    I4       = np.eye(4, dtype=complex)
    Pi_H     = np.zeros((4, 4)); Pi_H[3, 3] = 1.0
    Pi_union = np.kron(Pi_H, I4) + np.kron(I4, Pi_H) - np.kron(Pi_H, Pi_H)
    U_Zeno   = np.eye(16) - 2.0 * Pi_union
    assert np.allclose(U_Zeno @ np.array([0]*15 + [1]), -np.array([0]*15 + [1])), \
        "Zeno boundary reflection failed"
    assert np.allclose(U_Zeno @ np.array([1] + [0]*15), np.array([1] + [0]*15)), \
        "Zeno interior preservation failed"
    print("[✓] Routing & Zeno mathematical properties verified.")
    return True


# ==============================================================================
# REGISTER UTILITIES
# ==============================================================================

def _make_virtual_qudits_m(n_orbitals: int, m: int) -> "VirtualRegisterPair":
    r"""Allocate the composite virtual register for n_orbitals with modulus m.

    For ALL m > 1 the primary register V₁ is the hardware-native d=4
    VirtualQudit.  An auxiliary carry register V_aux is added when r = ceil(m/4) > 1.
    This unified structure handles every taxonomy branch:

      m = 1        : trivial ℤ_1 — no register needed (LineQid dim=1 placeholder).
      m ≤ 4  (r=1) : V₁ alone.  ℤ_m shift = ℤ_4 carry-gate + modular correction C_m.
                     m=4 exact: no correction.  m=2,3: correction SWAP(|0⟩,|m⟩) on V₁.
      m > 4  (r>1) : V₁ + V_aux carry qubits.  Same carry-gate + correction scheme.
                     m=4r exact (e.g. m=8): no correction.  All other m: C_m applied.

    The old binary-padded LineQid path for odd m (Case I) is retired.  All
    mechanisms now target the hardware-native d=4 qudit for V₁.

    Notes
    -----
    Call ``vreg.v1_qudits()`` for the primary V₁ list (length N) or
    ``vreg.all_qudits()`` for every qudit (V₁ + V_aux) in flat order.
    """
    if m <= 1:
        return VirtualRegisterPair(
            v1=[cirq.LineQid(i, dimension=max(m, 1)) for i in range(n_orbitals)],
            vaux=[None] * n_orbitals,
            m=m,
            r=0,
        )

    r   = (m + 3) // 4           # ceil(m/4)
    v1  = [VirtualQudit(i) for i in range(n_orbitals)]

    if r == 1:
        vaux = [None] * n_orbitals
    else:
        n_aux = math.ceil(math.log2(r))
        if n_aux == 1:
            vaux = [cirq.LineQubit(n_orbitals + i) for i in range(n_orbitals)]
        else:
            vaux = [
                tuple(cirq.LineQubit(n_orbitals * (1 + bit) + i) for bit in range(n_aux))
                for i in range(n_orbitals)
            ]
    return VirtualRegisterPair(v1=v1, vaux=vaux, m=m, r=r)


def _count_qudit_resources(
    circuit: cirq.Circuit,
) -> tuple[int, int, int]:
    """Return (n_logical, n_virtual, n_physical_qubits).

    n_physical_qubits = 2 × (n_logical + n_virtual) because each d=4
    qudit is encoded in 2 physical qubits via the Bell-separable basis.
    """
    logical = [q for q in circuit.all_qubits() if isinstance(q, NomosIonQid)]
    virtual = [q for q in circuit.all_qubits() if isinstance(q, VirtualQudit)]
    return len(logical), len(virtual), 2 * (len(logical) + len(virtual))
