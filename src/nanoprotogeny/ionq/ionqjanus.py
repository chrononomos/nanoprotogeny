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
ionqjanus.py — Biochemical Transition Operator and Janus Crossing Circuit Builder
==================================================================================
Implements J_{n→n+1}^{(M)}, the per-step biochemical transition operator from
the MQE theory, as a native d=4 qudit cirq circuit.

The transition operator interleaves five forward gate layers with up to six
reverse/photon layers, realising the full generalised gate algebra G(M):

  Forward (biochemical reduction / PCET):
    1. Virtual cofactor shifts    (U_R^{V,m})^{ν_n}  on B_n
    2. Proton phase rotations     Ẑ_Clock(φ_H)        on P_n
    3. Electron injection         Û_R                  on A_n
    4. Conformational docking     S_dock               on D_n
    5. Cofactor coupling          Û_couple^{(m,ν)}     on (L_p, V_p) ∈ B_n

  Reverse (oxidation / thermodynamic reset):
    6. Cofactor decoupling        Û_couple†            on B_n_decouple
    7. Electron ejection          Û_R†                 on A_n_eject
    8. Deprotonation              Ẑ_Clock(−φ_H)        on P_n_eject
    9. Inverse virtual shift      (U_R^{V,m})^{m−ν}   on B_n_decouple

  Photon (photo-driven mechanisms):
   10. Photon absorption          Û_photon(φ_γ)        on Γ_n_abs
   11. Photon emission            Û_emission(φ_γ)      on Γ_n_emit

The Janus crossing — the non-adiabatic surface hop at the antipodal element
k* = m/2 of ℤ_m — is applied as a pair of CrossManifoldSWAPGate operations on
the hydride orbitals specified in mechanism.crossings.  It is activated
externally in build_mqe_step_block (mqe.py) at step n = n*.

Dependencies: cirq,
              nanoprotogeny.molecular.mqemolecules  (MechanismTuple),
              nanoprotogeny.ionq.YB171PLUSHARDWARE  (NomosIonQid),
              nanoprotogeny.ionq.ionqmqegates       (all UFC/MQE gate classes).
No simulate-layer imports.
"""

from __future__ import annotations

import cirq
from typing import List, Optional, Tuple

from nanoprotogeny.molecular.mqemolecules import MechanismTuple
from nanoprotogeny.ionq.YB171PLUSHARDWARE import NomosIonQid, VirtualQudit
from nanoprotogeny.ionq.ionqmqegates import (
    GeneralizedVirtualShiftGate,
    CompositeVirtualShiftGate,
    ElectronShiftGate,
    ElectronEjectGate,
    ProtonPhaseGate,
    CofactorCouplingGate,
    CofactorDecouplingGate,
    CompositeCofactorCouplingGate,
    CompositeCofactorDecouplingGate,
    ConformationalShiftGate,
    PhotonAbsorptionGate,
    PhotonEmissionGate,
)

def build_biochemical_transition_circuit(
    n:               int,
    mechanism:       MechanismTuple,
    logical_qudits:  List[NomosIonQid],
    virtual_qudits:  List[cirq.Qid],
    dt:              float,
    vaux_qudits:     Optional[List] = None,
) -> cirq.Circuit:
    r"""Build J_{n→n+1}: the biochemical transition operator circuit.

    Implements Eq. (general_transition_operator) from the article:

        J_{n→n+1}^{(M)} =
            U_cof^{(n)}            [cofactor ATP/photon coupling]
          · S_conf^{(n)}           [conformational docking shift]
          · ⊗_{p ∈ A_n} U_R^{(p)} [electron injection]
          · ⊗_{q ∈ P_n} Z_H^{(q)} [proton phase rotation]
          · ⊗_{r ∈ B_n} (U_R^{V,m})^{nu_n^{(r)}} [virtual cofactor shift]

    Gate ordering (right-to-left in composition = left-to-right in time):
      1. Virtual cofactor shifts on B_n (independent single-qudit, parallel).
      2. Proton phase rotations on P_n (single-qudit, parallel).
      3. Electron shifts on A_n (single-qudit, parallel).
      4. Conformational docking on D_n (single-qudit, parallel).
      5. Cofactor coupling (U_coupling) on (logical, virtual) pairs in B_n.

    Args:
        n:              Step index (0-based).
        mechanism:      MechanismTuple describing the full mechanism.
        logical_qudits: List of NomosIonQid objects [0..N-1].
        virtual_qudits: Primary virtual register — VirtualQudit list for m%4==0,
                        LineQid list for odd m (length N).
        dt:             Trotter step size (for conformational gate).
        vaux_qudits:    Auxiliary carry register for m=4r with r>1.
                        vaux_qudits[i] is the LineQubit (r=2) or tuple of
                        LineQubits (r>2) carrying k//4 for orbital i.
                        Pass None (default) for m=4 or odd-m mechanisms.

    Returns:
        cirq.Circuit: The J_{n→n+1} operator circuit.
    """
    m         = mechanism.m
    r         = (m + 3) // 4 if m > 1 else 0          # ceil(m/4); 0 for trivial m
    # Use composite gate for ALL m > 1 when V₁ is the hardware-native VirtualQudit.
    # v1_is_native: True means v1 list contains VirtualQudits (composite path).
    v1_is_native = (
        m > 1
        and bool(virtual_qudits)
        and isinstance(virtual_qudits[0], VirtualQudit)
    )
    # has_vaux: True when r>1 and the caller provided carry qubits.
    has_vaux  = v1_is_native and (r > 1) and (vaux_qudits is not None)
    composite = v1_is_native   # emit CompositeVirtualShiftGate for any m>1 with VirtualQudit

    nu_n  = mechanism.nu_shifts[n]
    A_n   = mechanism.electron_sets[n]
    P_n   = mechanism.proton_sets[n]
    B_n   = mechanism.cofactor_sets[n]
    phi_H = mechanism.phi_proton
    D_n   = mechanism.dock_orbitals[n] if mechanism.dock_orbitals else []

    ops_virt     = []   # Moment 1:  virtual cofactor shifts      (fwd)
    ops_proton   = []   # Moment 2:  proton phase rotations        (fwd)
    ops_elec     = []   # Moment 3:  electron injection            (fwd)
    ops_dock     = []   # Moment 4:  conformational docking        (fwd)
    ops_coupling = []   # Moment 5:  cofactor coupling U_coupling  (fwd)
    ops_photon_a = []   # Moment 10: photon absorption U_photon    (phot)
    ops_photon_e = []   # Moment 11: photon emission   U_emission  (phot)

    # ── 1. Virtual cofactor shifts: (U_R^{V,m})^{nu_n} on B_n ───────────────
    if m > 1 and nu_n > 0:
        for r_idx in B_n:
            if r_idx >= len(virtual_qudits):
                continue
            if composite:
                gate = CompositeVirtualShiftGate(m=m, power=nu_n)
                if has_vaux:
                    ops_virt.append(gate.on(virtual_qudits[r_idx], vaux_qudits[r_idx]))
                else:
                    ops_virt.append(gate.on(virtual_qudits[r_idx]))
            else:
                ops_virt.append(
                    GeneralizedVirtualShiftGate(m, power=nu_n)
                    .on(virtual_qudits[r_idx])
                )

    # ── 2. Proton phase rotations: Z_H^{(q)}(phi_H) on P_n ─────────────────
    for q_idx in P_n:
        if q_idx < len(logical_qudits):
            ops_proton.append(
                ProtonPhaseGate(phi_H).on(logical_qudits[q_idx])
            )

    # ── 3. Electron shifts: U_R^{(p)} on A_n ────────────────────────────────
    for p_idx in A_n:
        if p_idx < len(logical_qudits):
            ops_elec.append(
                ElectronShiftGate(power=1).on(logical_qudits[p_idx])
            )

    # ── 4. Conformational docking: S_dock^{(n)} on D_n ─────────────────────
    for d_idx in D_n:
        if d_idx < len(logical_qudits):
            ops_dock.append(
                ConformationalShiftGate(delta_h=0.01, dt=dt)
                .on(logical_qudits[d_idx])
            )

    # ── 5. Cofactor coupling on (logical_p, virtual_p) for p ∈ B_n ──────────
    if m > 1 and nu_n > 0:
        for r_idx in B_n:
            if r_idx >= len(logical_qudits) or r_idx >= len(virtual_qudits):
                continue
            if composite:
                gate = CompositeCofactorCouplingGate(m=m, nu=nu_n)
                if has_vaux:
                    ops_coupling.append(gate.on(
                        logical_qudits[r_idx],
                        virtual_qudits[r_idx],
                        vaux_qudits[r_idx],
                    ))
                else:
                    ops_coupling.append(gate.on(
                        logical_qudits[r_idx],
                        virtual_qudits[r_idx],
                    ))
            else:
                v_q = virtual_qudits[r_idx]
                if hasattr(v_q, '_index') or (
                        hasattr(v_q, 'dimension') and v_q.dimension == m):
                    ops_coupling.append(
                        CofactorCouplingGate(m=m, nu=nu_n)
                        .on(logical_qudits[r_idx], v_q)
                    )

    # ─────────────────────────────────────────────────────────────────────────
    # REVERSE moments
    # ─────────────────────────────────────────────────────────────────────────
    A_n_ej  = mechanism.electron_eject_sets[n]
    P_n_ej  = mechanism.proton_eject_sets[n]
    B_n_dec = mechanism.cofactor_decouple_sets[n]
    nu_dec  = mechanism.nu_decouple_shifts[n]

    Gamma_abs  = mechanism.photon_absorb_sets[n]
    Gamma_emit = mechanism.photon_emit_sets[n]
    phi_photon = mechanism.phi_photon

    ops_decoupling = []
    ops_elec_ej    = []
    ops_deproton   = []
    ops_virt_inv   = []

    # Moment 6 — cofactor decoupling
    if m > 1 and nu_dec > 0 and B_n_dec:
        for r_idx in B_n_dec:
            if r_idx >= len(logical_qudits) or r_idx >= len(virtual_qudits):
                continue
            if composite:
                gate = CompositeCofactorDecouplingGate(m=m, nu=nu_dec)
                if has_vaux:
                    ops_decoupling.append(gate.on(
                        logical_qudits[r_idx],
                        virtual_qudits[r_idx],
                        vaux_qudits[r_idx],
                    ))
                else:
                    ops_decoupling.append(gate.on(
                        logical_qudits[r_idx],
                        virtual_qudits[r_idx],
                    ))
            else:
                v_q = virtual_qudits[r_idx]
                if hasattr(v_q, '_index') or (
                        hasattr(v_q, 'dimension') and v_q.dimension == m):
                    ops_decoupling.append(
                        CofactorDecouplingGate(m=m, nu=nu_dec)
                        .on(logical_qudits[r_idx], v_q)
                    )

    # Moment 7 — electron ejection
    for p_idx in A_n_ej:
        if p_idx < len(logical_qudits):
            ops_elec_ej.append(
                ElectronEjectGate(power=1).on(logical_qudits[p_idx])
            )

    # Moment 8 — deprotonation
    for q_idx in P_n_ej:
        if q_idx < len(logical_qudits):
            ops_deproton.append(
                ProtonPhaseGate(-phi_H).on(logical_qudits[q_idx])
            )

    # Moment 9 — inverse virtual shift
    if m > 1 and nu_dec > 0 and B_n_dec:
        inv_power = (m - nu_dec) % m
        if inv_power > 0:
            for r_idx in B_n_dec:
                if r_idx >= len(virtual_qudits):
                    continue
                if composite:
                    gate = CompositeVirtualShiftGate(m=m, power=inv_power)
                    if has_vaux:
                        ops_virt_inv.append(gate.on(virtual_qudits[r_idx], vaux_qudits[r_idx]))
                    else:
                        ops_virt_inv.append(gate.on(virtual_qudits[r_idx]))
                else:
                    ops_virt_inv.append(
                        GeneralizedVirtualShiftGate(m, power=inv_power)
                        .on(virtual_qudits[r_idx])
                    )

    # Moment 10 — photon absorption
    for g_idx in Gamma_abs:
        if g_idx < len(logical_qudits):
            ops_photon_a.append(
                PhotonAbsorptionGate(phi_photon).on(logical_qudits[g_idx])
            )

    # Moment 11 — photon emission
    for g_idx in Gamma_emit:
        if g_idx < len(logical_qudits):
            ops_photon_e.append(
                PhotonEmissionGate(phi_photon).on(logical_qudits[g_idx])
            )

    # ── Assemble all moments ──────────────────────────────────────────────────
    moments = []
    for ops in [
        ops_virt,       #  1  fwd:  virtual cofactor shifts
        ops_proton,     #  2  fwd:  proton phase rotations
        ops_elec,       #  3  fwd:  electron injection
        ops_dock,       #  4  fwd:  conformational docking
        ops_coupling,   #  5  fwd:  cofactor coupling U_coupling
        ops_decoupling, #  6  rev:  cofactor decoupling U_coupling†
        ops_elec_ej,    #  7  rev:  electron ejection U_R†
        ops_deproton,   #  8  rev:  deprotonation Z_H(−φ)
        ops_virt_inv,   #  9  rev:  inverse virtual shift
        ops_photon_a,   # 10  phot: photon absorption U_photon
        ops_photon_e,   # 11  phot: photon emission   U_emission
    ]:
        if ops:
            moments.append(cirq.Moment(ops))

    return cirq.Circuit(moments)


