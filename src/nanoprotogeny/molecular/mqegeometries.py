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
mqegeometries.py — MQE Per-Step Nuclear Geometries (PySCF-Free)
===============================================================
Standalone geometry module for all MQE mechanisms.  Mirrors the geometry
layer of ``mqedatagenerator.py`` without importing PySCF, MQEStep, or
MQEMechanismSpec.

Public API
----------
BONDLENGTHS : Dict[str, List[float]]
    Per-step bond-length sequence for each mechanism (Angstrom).
    Keys match mechanism names used throughout the MQE pipeline.

Z3_COFACTOR_GEOM_PARAMS : List[Tuple[float, float]]
    (r12, r13) pairs for the H3+ triangle at each of the 3 z3_cofactor steps.

get_step_geometry(mechanism_name, step_n, bondlength=None)
    -> List[Tuple[str, float, float, float]]
    Return (symbol, x, y, z) in Angstroms for the requested step.
    If ``bondlength`` is None the value is taken from BONDLENGTHS.

get_janus_geometry(mechanism_name)
    -> List[Tuple[str, float, float, float]]
    Return the single Janus-point geometry (same as static registry in
    mqeprotogeny.py).

parse_atom_block(atom_str) -> List[Tuple[str, float, float, float]]
    Parse a PySCF-format whitespace-separated atom block string.

atom_block_to_str(atoms) -> str
    Inverse: convert tuple list back to PySCF-format string.

Dependencies
------------
numpy, math, typing, stdlib only.  No pyscf, no cirq, no ionq.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ===========================================================================
# SECTION 0 — UTILITIES
# ===========================================================================

def parse_atom_block(atom_str: str) -> List[Tuple[str, float, float, float]]:
    """Parse a PySCF-format atom block string to (symbol, x, y, z) tuples.

    Handles lines of the form ``Fe  0.000  1.350  0.000``.
    Blank lines and comment lines (starting with #) are skipped.
    """
    atoms: List[Tuple[str, float, float, float]] = []
    for line in atom_str.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 4:
            atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    return atoms


def atom_block_to_str(atoms: List[Tuple[str, float, float, float]]) -> str:
    """Convert (symbol, x, y, z) list to PySCF-format string."""
    return "\n".join(f"{sym}   {x:.6f}  {y:.6f}  {z:.6f}" for sym, x, y, z in atoms)


# ===========================================================================
# SECTION 1 — JANUS (SINGLE-POINT) GEOMETRY
# The Janus single-point geometry is derived from the per-step builder at the
# mechanism's janus_step, mirroring mqedatagenerator._atom_block_for_step
# evaluated at janus_step.  This guarantees the seed-free (mqegeometries) and
# PySCF (mqedatagenerator) layers use the SAME single-point geometry, and that
# the Janus point is always consistent with the per-step reaction coordinate.
# ===========================================================================

def _h4_janus() -> List[Tuple[str, float, float, float]]:
    """H₄ linear chain at 1.5 Å spacing — generic fallback for mechanisms with
    no per-step builder (i.e. names absent from :data:`BONDLENGTHS`)."""
    return [
        ("H", -2.25,  0.0,  0.0),
        ("H", -0.75,  0.0,  0.0),
        ("H",  0.75,  0.0,  0.0),
        ("H",  2.25,  0.0,  0.0),
    ]


def get_janus_geometry(
    mechanism_name: str,
) -> List[Tuple[str, float, float, float]]:
    """Return the single Janus-point geometry for *mechanism_name*.

    Derived from the per-step builder evaluated at the mechanism's
    ``janus_step`` (from :data:`ZETAZERO_SPECS`), or at step 0 when the
    mechanism is adiabatic (``janus_step=None``) or has no registered spec.
    This mirrors ``mqedatagenerator._atom_block_for_step(spec, janus_step)`` so
    the seed-free and PySCF dataset layers agree on the single-point geometry,
    and removes the previous static registry which had drifted from the
    per-step builders (e.g. psii→Mn₂O₂ vs Fe₂S₂, cu_co2rr's extra C atom,
    bent vs linear femon2).

    Mechanisms with no per-step builder (not in :data:`BONDLENGTHS`) fall back
    to the generic H₄ linear chain.
    """
    if mechanism_name not in BONDLENGTHS:
        return _h4_janus()
    spec = ZETAZERO_SPECS.get(mechanism_name)
    js = spec.janus_step if (spec is not None and spec.janus_step is not None) else 0
    n_steps = len(BONDLENGTHS[mechanism_name])
    if not (0 <= js < n_steps):
        js = 0
    return get_step_geometry(mechanism_name, js)


# ===========================================================================
# SECTION 2 — PER-STEP GEOMETRY BUILDER FUNCTIONS
# All functions return PySCF-format atom block strings (whitespace-separated).
# They take only plain scalar / integer arguments — no MQEStep objects.
# ===========================================================================

def _hchain_geometry(n: int, bondlength: float) -> str:
    """Linear H_n chain geometry (Angstrom), centred at origin."""
    total_len = (n - 1) * bondlength
    positions = [-total_len / 2 + i * bondlength for i in range(n)]
    lines = [f"H  {x:10.6f}  0.000000  0.000000" for x in positions]
    return "\n".join(lines)


def _fe2s2_geometry_at_bond(fe_s_distance: float) -> str:
    """[Fe₂(μ-S)₂] rhombic core with variable Fe–S distance (Angstrom).

    The Fe–Fe distance is held fixed at 2.70 Å; the Fe–S distance varies to
    model oxidation-state changes along the catalytic trajectory.

    Args:
        fe_s_distance: Fe–S bond length in Angstrom (nominally 2.26 Å resting).
    """
    d = fe_s_distance
    return (
        f"Fe   0.000000  1.350000  0.000000\n"
        f"Fe   0.000000 -1.350000  0.000000\n"
        f"S    {d:.6f}  0.000000  0.000000\n"
        f"S   -{d:.6f}  0.000000  0.000000"
    )


def _haber_bosch_fe2s2n2_geometry(step_n: int) -> str:
    """Fe₂S₂N₂ proxy geometry for Haber-Bosch N₂ activation.

    Fixed Fe₂S₂ rhombic core at the nitrogenase E4 Janus geometry (Fe–S =
    2.316 Å) with N₂ adsorbed axially above the Fe₂ centre.  The N–N bond
    elongates step-wise from 1.10 Å (step 0, chemisorbed N₂) to 1.52 Å
    (step 7, near-dissociation limit), modelling progressive activation.

    Args:
        step_n: Step index 0–7.
    """
    fe_s = 2.316
    nn   = 1.10 + step_n * 0.06      # 1.10 Å (n=0) → 1.52 Å (n=7)
    half = nn / 2.0
    z0   = 2.0                        # N₂ centre-of-mass elevation above plane
    return (
        f"Fe   0.000000  1.350000  0.000000\n"
        f"Fe   0.000000 -1.350000  0.000000\n"
        f"S    {fe_s:.6f}  0.000000  0.000000\n"
        f"S   -{fe_s:.6f}  0.000000  0.000000\n"
        f"N    0.000000  0.000000  {z0 + half:.6f}\n"
        f"N    0.000000  0.000000  {z0 - half:.6f}"
    )


def _psii_photo_geometry_at_step(step_n: int) -> str:
    """[Mn₂O₂] PSII photo proxy — shorter Mn–O bonds at higher oxidation state.

    Args:
        step_n: Step index 0–3.
    """
    d = 2.260 - step_n * 0.02
    return (
        f"Mn   0.000000  1.350000  0.000000\n"
        f"Mn   0.000000 -1.350000  0.000000\n"
        f"O    {d:.6f}  0.000000  0.000000\n"
        f"O   -{d:.6f}  0.000000  0.000000"
    )


def _h3plus_geometry(r_12: float, r_13: float) -> str:
    """H₃⁺ isoceles triangle allowing asymmetric deformation (z3_cofactor).

    Args:
        r_12: H₀–H₁ distance in Angstrom.
        r_13: H₀–H₂ distance in Angstrom.
    """
    x2 = r_12
    x3 = r_13 * 0.5
    y3 = r_13 * math.sqrt(3) / 2
    return (
        f"H   0.000000  0.000000  0.000000\n"
        f"H  {x2:.6f}  0.000000  0.000000\n"
        f"H  {x3:.6f}  {y3:.6f}  0.000000"
    )


def _h2_geometry(bondlength: float) -> str:
    """H₂ geometry at given bond length (Angstrom)."""
    half = bondlength / 2
    return (
        f"H  -{half:.6f}  0.000000  0.000000\n"
        f"H   {half:.6f}  0.000000  0.000000"
    )


def _fe4s4_geometry_at_step(step_n: int) -> str:
    """[Fe₄S₄] cubane geometry that expands with step_n.

    T_d symmetric-like cubane core.  The core volume expands by ~0.5% per
    reduction step to simulate the breathing mode of the cluster.

    Args:
        step_n: Step index 0–7.
    """
    d = 1.305 * (1.0 + step_n * 0.005)
    return (
        f"Fe   {d:.6f}  {d:.6f}  {d:.6f}\n"
        f"Fe  -{d:.6f} -{d:.6f}  {d:.6f}\n"
        f"Fe  -{d:.6f}  {d:.6f} -{d:.6f}\n"
        f"Fe   {d:.6f} -{d:.6f} -{d:.6f}\n"
        f"S   -{d:.6f} -{d:.6f} -{d:.6f}\n"
        f"S    {d:.6f}  {d:.6f} -{d:.6f}\n"
        f"S    {d:.6f} -{d:.6f}  {d:.6f}\n"
        f"S   -{d:.6f}  {d:.6f}  {d:.6f}"
    )


def _femo_proxy_atom_block(d: float) -> str:
    """Fe-Mo-S₂ proxy for Mo-nitrogenase (catalog entry 1, Group A).

    Fe(26)+Mo(14 ECP)+2×S(16)=72e (even), charge=0, spin_2S=4.
    Reaction coordinate: Fe-Mo bond compression 2.700→2.620 Å.
    """
    return (
        f"Fe  0.000000  0.000000  0.000000\n"
        f"Mo  0.000000  0.000000  {d:.6f}\n"
        f"S   0.000000  1.400000  {d / 2:.6f}\n"
        f"S   0.000000 -1.400000  {d / 2:.6f}"
    )


def _mo_nr_proxy_atom_block(d: float) -> str:
    """Mo-S₂-O₂ proxy for assimilatory nitrate reductase (catalog entry 7, Group A).

    Mo(14 ECP)+2×S(16)+2×O(8)=62e (even), charge=0, spin_2S=4.
    Reaction coordinate: Mo-S bond compression 2.420→2.340 Å.
    """
    return (
        f"Mo  0.000000  0.000000  0.000000\n"
        f"S   0.000000  {d:.6f}  0.000000\n"
        f"S   0.000000  {-d:.6f}  0.000000\n"
        f"O   1.800000  0.000000  0.000000\n"
        f"O  -1.800000  0.000000  0.000000"
    )


def _ti2n2_proxy_atom_block(d: float) -> str:
    """Ti₂N₂ proxy for photocatalytic N₂ fixation (catalog entry 14, Group A).

    2×Ti(22)+2×N(7)=58e (even), charge=0, spin_2S=4.
    Reaction coordinate: Ti-N bond compression 1.900→1.556 Å (N₂ side-on activation).
    """
    return (
        f"Ti  0.000000  0.000000  0.000000\n"
        f"Ti  2.960000  0.000000  0.000000\n"
        f"N   1.480000  {d:.6f}  0.000000\n"
        f"N   1.480000  {-d:.6f}  0.000000"
    )


def _v2s2_proxy_atom_block(d: float) -> str:
    """V₂S₂ proxy for V-nitrogenase (catalog entry 3, Group D).

    2×V(23)+2×S(16)=78e (even), charge=0, spin_2S=4.
    Reaction coordinate: V-S bond compression 2.350→2.258 Å.
    """
    return (
        f"V   0.000000  1.300000  0.000000\n"
        f"V   0.000000 -1.300000  0.000000\n"
        f"S   {d:.6f}  0.000000  0.000000\n"
        f"S  {-d:.6f}  0.000000  0.000000"
    )


def _cu3_proxy_atom_block(d: float) -> str:
    """Cu₃ equilateral trimer proxy for Cu CO₂RR (catalog entry 13, Group D).

    3×Cu(29), charge=−1 → 88e (even), spin_2S=0.
    Reaction coordinate: Cu-Cu bond compression 2.550→2.458 Å.
    """
    half   = d / 2.0
    height = d * 0.866025403784
    return (
        f"Cu  0.000000  0.000000  0.000000\n"
        f"Cu  {d:.6f}  0.000000  0.000000\n"
        f"Cu  {half:.6f}  {height:.6f}  0.000000"
    )


def _femon2_trimer_atom_block(d_nn: float) -> str:
    """Fe–Mo–N₂ trimer proxy (Group B, m=4).

    Fe(26)+Mo(ECP28→14val)+2×N(7)=54e (even), charge=0, spin_2S=4.
    Linear arrangement along z: Fe — Mo — N≡N (end-on binding).

    Fixed geometry:
        Fe–Mo = 2.700 Å
        Mo–N(prox) = 2.000 Å
    Reaction coordinate:
        N–N elongation d_nn = 1.10→1.52 Å (8 steps, 0.06 Å/step).

    Args:
        d_nn: N–N bond length in Angstrom.
    """
    d_femo = 2.700
    d_mon  = 2.000
    z_fe   = 0.0
    z_mo   = z_fe + d_femo
    z_n1   = z_mo + d_mon
    z_n2   = z_n1 + d_nn
    return (
        f"Fe  0.000000  0.000000  {z_fe:.6f}\n"
        f"Mo  0.000000  0.000000  {z_mo:.6f}\n"
        f"N   0.000000  0.000000  {z_n1:.6f}\n"
        f"N   0.000000  0.000000  {z_n2:.6f}"
    )


def _thymine_dimer_proxy_geometry(distance: float) -> str:
    """Two stacked ethylene molecules representing the [2+2] cycloaddition.

    Ethylene 1 is fixed at Z=0; ethylene 2 approaches from Z=distance.

    Args:
        distance: Inter-planar distance in Angstrom (2.80→1.50 Å).
    """
    z1 = 0.0
    z2 = distance
    eth1 = (
        f"C  -0.665000  0.000000  {z1:.6f}\n"
        f"C   0.665000  0.000000  {z1:.6f}\n"
        f"H  -1.230000  0.920000  {z1:.6f}\n"
        f"H  -1.230000 -0.920000  {z1:.6f}\n"
        f"H   1.230000  0.920000  {z1:.6f}\n"
        f"H   1.230000 -0.920000  {z1:.6f}"
    )
    eth2 = (
        f"C  -0.665000  0.000000  {z2:.6f}\n"
        f"C   0.665000  0.000000  {z2:.6f}\n"
        f"H  -1.230000  0.920000  {z2:.6f}\n"
        f"H  -1.230000 -0.920000  {z2:.6f}\n"
        f"H   1.230000  0.920000  {z2:.6f}\n"
        f"H   1.230000 -0.920000  {z2:.6f}"
    )
    return eth1 + "\n" + eth2


def _rnr_proxy_geometry(step_n: int) -> str:
    """Thiyl radical H-atom transfer proxy for RNR (4 steps, 0–3).

    Models HAT (S•—H···C) and subsequent C-O bond cleavage on a
    ribose-like cyclopentane scaffold.

    Args:
        step_n: Step index 0–3.
    """
    params = [
        (3.0, 1.1, 1.43),   # step 0: pre-reaction
        (1.6, 1.4, 1.60),   # step 1: HAT TS
        (1.35, 1.8, 2.20),  # step 2: post-HAT / dehydration TS
        (1.34, 2.5, 3.50),  # step 3: product-like
    ]
    d_SH, d_CH, d_CO = params[step_n]

    c3 = np.array([0.0, 0.0, 0.0])
    o  = np.array([d_CO, 0.0, 0.0])
    h  = np.array([0.0, 0.0, d_CH])
    s  = np.array([0.0, 0.0, d_CH + d_SH])
    s_methyl = np.array([0.0, 1.8, d_CH + d_SH])

    c1 = np.array([ 1.2, -0.8, -0.3])
    c2 = np.array([ 1.2,  0.8,  0.3])
    c4 = np.array([-1.2,  0.8,  0.3])
    c5 = np.array([-1.2, -0.8, -0.3])

    lines = [
        f"C   {c1[0]:.6f}  {c1[1]:.6f}  {c1[2]:.6f}",
        f"C   {c2[0]:.6f}  {c2[1]:.6f}  {c2[2]:.6f}",
        f"C   {c3[0]:.6f}  {c3[1]:.6f}  {c3[2]:.6f}",
        f"C   {c4[0]:.6f}  {c4[1]:.6f}  {c4[2]:.6f}",
        f"C   {c5[0]:.6f}  {c5[1]:.6f}  {c5[2]:.6f}",
        f"O   {o[0]:.6f}  {o[1]:.6f}  {o[2]:.6f}",
        f"H   {h[0]:.6f}  {h[1]:.6f}  {h[2]:.6f}",
        f"S   {s[0]:.6f}  {s[1]:.6f}  {s[2]:.6f}",
        f"C   {s_methyl[0]:.6f}  {s_methyl[1]:.6f}  {s_methyl[2]:.6f}",
        f"H   {s_methyl[0]:.6f}  {s_methyl[1]+1.0:.6f}  {s_methyl[2]:.6f}",
        f"H   {s_methyl[0]:.6f}  {s_methyl[1]-1.0:.6f}  {s_methyl[2]:.6f}",
    ]
    return "\n".join(lines)


def _atp_hydrolysis_geometry(bondlength: float) -> str:
    """H₃PO₄ + H₂O nucleophilic attack proxy (minimal, 60e).

    P at origin, tetrahedral H₃PO₄ (one axial P=O along +z, three equatorial
    P-OH), nucleophilic H₂O approaching along -z.

    Args:
        bondlength: P–O(water) distance in Angstrom (2.4→1.8 Å).
    """
    distance = bondlength
    r_po_dbl = 1.480
    r_po_sgl = 1.570
    r_oh     = 0.960

    ox1 =  r_po_sgl;             oy1 = 0.000;              oz1 = 0.0
    ox2 = -r_po_sgl * 0.5;       oy2 =  r_po_sgl * 0.866;  oz2 = 0.0
    ox3 = -r_po_sgl * 0.5;       oy3 = -r_po_sgl * 0.866;  oz3 = 0.0
    hx1 = ox1 + r_oh;            hy1 = 0.000;               hz1 = 0.0
    hx2 = ox2 - r_oh * 0.5;      hy2 = oy2 + r_oh * 0.866; hz2 = 0.0
    hx3 = ox3 - r_oh * 0.5;      hy3 = oy3 - r_oh * 0.866; hz3 = 0.0

    z_o_wat = -distance
    z_hw    = z_o_wat - r_oh * 0.611
    x_hw    = r_oh * 0.791

    return (
        f"P    0.000000  0.000000  0.000000\n"
        f"O    0.000000  0.000000  {r_po_dbl:.6f}\n"
        f"O    {ox1:.6f}  {oy1:.6f}  {oz1:.6f}\n"
        f"O    {ox2:.6f}  {oy2:.6f}  {oz2:.6f}\n"
        f"O    {ox3:.6f}  {oy3:.6f}  {oz3:.6f}\n"
        f"H    {hx1:.6f}  {hy1:.6f}  {hz1:.6f}\n"
        f"H    {hx2:.6f}  {hy2:.6f}  {hz2:.6f}\n"
        f"H    {hx3:.6f}  {hy3:.6f}  {hz3:.6f}\n"
        f"O    0.000000  0.000000  {z_o_wat:.6f}\n"
        f"H    {x_hw:.6f}  0.000000  {z_hw:.6f}\n"
        f"H   -{x_hw:.6f}  0.000000  {z_hw:.6f}\n"
    )


def _cyp450_geometry(bondlength: float) -> str:
    """CYP450 active-site proxy with proximal cysteine thiolate (78e).

    Fe at origin, O along +z, S(Cys) along -z, four equatorial N (porphyrin).

    Args:
        bondlength: Fe–O bond length in Angstrom (2.0→1.6→2.0 Å over 6 steps).
    """
    r_feo = bondlength
    return (
        f"Fe   0.000000  0.000000  0.000000\n"
        f"O    0.000000  0.000000  {r_feo:.6f}\n"
        f"S    0.000000  0.000000 -2.350000\n"
        f"N    2.000000  0.000000 -0.200000\n"
        f"N   -2.000000  0.000000 -0.200000\n"
        f"N    0.000000  2.000000 -0.200000\n"
        f"N    0.000000 -2.000000 -0.200000"
    )


def _ethylene_epoxidation_geometry(bondlength: float) -> str:
    """Ag₃/Cl/O/C₂H₄ ethylene epoxidation proxy (98e).

    Ag₃ triangle (surface proxy) + Cl promoter + electrophilic O + approaching
    ethylene. Reaction coordinate: C₂H₄ centre-of-mass to O distance.

    Args:
        bondlength: Ethylene-to-O distance in Angstrom (3.5→2.0 Å).
    """
    distance = bondlength
    ag1 = ( 0.000000,  1.670000, 0.000000)
    ag2 = (-1.450000, -0.835000, 0.000000)
    ag3 = ( 1.450000, -0.835000, 0.000000)
    cl  = ( 0.000000,  0.000000, -1.800000)
    o   = ( 0.000000,  1.670000, 1.900000)
    z_eth = 1.900000 + distance
    c1 = (-0.665000,  1.670000, z_eth)
    c2 = ( 0.665000,  1.670000, z_eth)
    h1 = (-1.230000,  2.590000, z_eth)
    h2 = (-1.230000,  0.750000, z_eth)
    h3 = ( 1.230000,  2.590000, z_eth)
    h4 = ( 1.230000,  0.750000, z_eth)
    return (
        f"Ag   {ag1[0]:.6f}  {ag1[1]:.6f}  {ag1[2]:.6f}\n"
        f"Ag   {ag2[0]:.6f}  {ag2[1]:.6f}  {ag2[2]:.6f}\n"
        f"Ag   {ag3[0]:.6f}  {ag3[1]:.6f}  {ag3[2]:.6f}\n"
        f"Cl   {cl[0]:.6f}   {cl[1]:.6f}   {cl[2]:.6f}\n"
        f"O    {o[0]:.6f}    {o[1]:.6f}    {o[2]:.6f}\n"
        f"C    {c1[0]:.6f}   {c1[1]:.6f}   {c1[2]:.6f}\n"
        f"C    {c2[0]:.6f}   {c2[1]:.6f}   {c2[2]:.6f}\n"
        f"H    {h1[0]:.6f}   {h1[1]:.6f}   {h1[2]:.6f}\n"
        f"H    {h2[0]:.6f}   {h2[1]:.6f}   {h2[2]:.6f}\n"
        f"H    {h3[0]:.6f}   {h3[1]:.6f}   {h3[2]:.6f}\n"
        f"H    {h4[0]:.6f}   {h4[1]:.6f}   {h4[2]:.6f}\n"
    )


# ===========================================================================
# SECTION 3 — STATIC GEOMETRY DICTIONARIES
# Mechanisms whose geometries are fully enumerated per-step rather than
# parameterised by a single bond length.
# ===========================================================================

HABER_BOSCH_GEOMETRIES: Dict[int, str] = {
    # Step 0: N₂ chemisorbed side-on bridge, N–N slightly elongated (1.15 Å)
    0: """Fe -1.200000  0.000000 -1.500000
          Fe  1.200000  0.000000 -1.500000
          N  -0.575000  0.000000  0.000000
          N   0.575000  0.000000  0.000000""",

    # Step 1: N₂ dissociated → 2 surface-bound N atoms (N–N ~ 2.4 Å)
    1: """Fe -1.200000  0.000000 -1.500000
          Fe  1.200000  0.000000 -1.500000
          N  -1.200000  0.000000  0.000000
          N   1.200000  0.000000  0.000000""",

    # Step 2: First hydrogenation (N + NH)
    2: """Fe -1.200000  0.000000 -1.500000
          Fe  1.200000  0.000000 -1.500000
          N  -1.200000  0.000000  0.000000
          N   1.200000  0.000000  0.000000
          H  -1.200000  0.900000  0.500000""",

    # Step 3: Second hydrogenation (NH + NH)
    3: """Fe -1.200000  0.000000 -1.500000
          Fe  1.200000  0.000000 -1.500000
          N  -1.200000  0.000000  0.000000
          N   1.200000  0.000000  0.000000
          H  -1.200000  0.900000  0.500000
          H   1.200000  0.900000  0.500000""",

    # Step 4: Further hydrogenation (NH₂ + NH₂)
    4: """Fe -1.200000  0.000000 -1.500000
          Fe  1.200000  0.000000 -1.500000
          N  -1.200000  0.000000  0.000000
          N   1.200000  0.000000  0.000000
          H  -1.200000  0.900000  0.500000
          H  -1.200000 -0.900000  0.500000
          H   1.200000  0.900000  0.500000
          H   1.200000 -0.900000  0.500000""",

    # Step 5: Final hydrogenation & desorption (2NH₃ moving away from surface)
    5: """Fe -1.200000  0.000000 -1.500000
          Fe  1.200000  0.000000 -1.500000
          N  -1.200000  0.000000  1.000000
          N   1.200000  0.000000  1.000000
          H  -1.200000  0.900000  1.500000
          H  -2.000000 -0.450000  1.500000
          H  -0.400000 -0.450000  1.500000
          H   1.200000  0.900000  1.500000
          H   2.000000 -0.450000  1.500000
          H   0.400000 -0.450000  1.500000""",
}

ANAMMOX_GEOMETRIES: Dict[int, str] = {
    # Step 0: Distant NH₂ fragments (pre-coupling, N–N ~ 2.6 Å)
    0: """Fe  0.000000  0.000000  0.000000
          N   0.000000  1.300000  1.500000
          H   0.800000  1.300000  2.000000
          H  -0.800000  1.300000  2.000000
          N   0.000000 -1.300000  1.500000
          H   0.800000 -1.300000  2.000000
          H  -0.800000 -1.300000  2.000000""",

    # Step 1: Fragments closer (N–N ~ 2.0 Å)
    1: """Fe  0.000000  0.000000  0.000000
          N   0.000000  1.000000  1.600000
          H   0.800000  1.100000  2.100000
          H  -0.800000  1.100000  2.100000
          N   0.000000 -1.000000  1.600000
          H   0.800000 -1.100000  2.100000
          H  -0.800000 -1.100000  2.100000""",

    # Step 2: Transition state proxy (N–N ~ 1.6 Å) — Janus step
    2: """Fe  0.000000  0.000000  0.000000
          N   0.000000  0.800000  1.800000
          H   0.800000  0.900000  2.300000
          H  -0.800000  0.900000  2.300000
          N   0.000000 -0.800000  1.800000
          H   0.800000 -0.900000  2.300000
          H  -0.800000 -0.900000  2.300000""",

    # Step 3: Bound hydrazine N₂H₄ (N–N ~ 1.44 Å)
    3: """Fe  0.000000  0.000000  0.000000
          N   0.000000  0.720000  2.000000
          H   0.800000  0.800000  2.500000
          H  -0.800000  0.800000  2.500000
          N   0.000000 -0.720000  2.000000
          H   0.800000 -0.800000  2.500000
          H  -0.800000 -0.800000  2.500000""",
}


# ===========================================================================
# Full FeMoco Cluster Geometry (19 atoms: Fe7MoS9C)
# ===========================================================================
# Derived from PDB 1M1N (Einsle et al., Science 2002) with the interstitial
# carbide (C) correctly identified (Spatzal et al., Science 2011).
# This is the resting state (E0) geometry. C3v-approximate symmetry.
FEMOCO_1M1N_GEOMETRY = """\
Fe    0.000000    0.000000    3.457475
Fe    0.000000    1.524205    1.294913
Fe   -1.320000   -0.762102    1.294913
Fe    1.320000   -0.762102    1.294913
Fe    1.320000    0.762102   -1.294913
Fe   -1.320000    0.762102   -1.294913
Fe    0.000000   -1.524205   -1.294913
Mo    0.000000    0.000000   -3.506624
S     0.000000    1.062102    2.376194
S    -0.919808   -0.531051    2.376194
S     0.919808   -0.531051    2.376194
S     1.060000    1.835974    0.000000
S    -2.120000    0.000000    0.000000
S     1.060000   -1.835974    0.000000
S     0.919808    0.531051   -2.400768
S    -0.919808    0.531051   -2.400768
S     0.000000   -1.062102   -2.400768
C     0.000000    0.000000    0.000000
"""

def _femoco_full_geometry() -> str:
    """Full 19-atom FeMoco cluster (Fe7MoS9C) resting state geometry.
    Returns PySCF-format atom block string.
    """
    return FEMOCO_1M1N_GEOMETRY

def _femoco_full_with_n2_geometry(step_n: int) -> str:
    """Full 19-atom FeMoco cluster + N2 substrate (21 atoms total).
    N2 binds end-on to a central belt Fe atom, pointing radially outward 
    from the interstitial carbide. The N-N bond elongates over 8 steps 
    to model progressive activation/reduction.
    """
    # 1. Parse the static 19-atom resting state
    base_atoms = parse_atom_block(FEMOCO_1M1N_GEOMETRY)
    
    # 2. Select a belt Fe atom (from the 1M1N geometry)
    # The vector from the origin (carbide) to this Fe is exactly the Fe-C bond.
    fe_pos = np.array([0.000000, 1.524205, 1.294913])
    radial_unit = fe_pos / np.linalg.norm(fe_pos)
    
    # 3. Get the N-N bond length for this specific step from the registry
    bls = BONDLENGTHS.get("nitrogenase_femoco")
    d_nn = bls[step_n] if bls and step_n < len(bls) else 1.10
    
    # 4. Place proximal N (N1) at ~2.0 Å from the Fe atom along the radial axis
    d_fe_n = 2.000
    n1_pos = fe_pos + (d_fe_n * radial_unit)
    
    # 5. Place distal N (N2) at d_nn distance from N1
    n2_pos = n1_pos + (d_nn * radial_unit)
    
    # 6. Append the N2 molecule to the cluster
    base_atoms.append(("N", float(n1_pos[0]), float(n1_pos[1]), float(n1_pos[2])))
    base_atoms.append(("N", float(n2_pos[0]), float(n2_pos[1]), float(n2_pos[2])))
    
    return atom_block_to_str(base_atoms)


def _benzoquinone_geometry(step_n: int) -> str:
    """1,4-benzoquinone / hydroquinone PCET proxy (14 atoms: C6 H4 O2 + 2H).
    Models the reversible 2e-/2H+ redox cycle: Q + 2H+ <-> QH2.
    
    Steps 0-2 (Forward): C=O double bond (1.22 A) stretches to C-O single (1.36 A).
                         Protons approach from 3.0 A to 0.96 A.
    Steps 3-5 (Reverse): C-O single bond compresses to C=O double.
                         Protons depart from 0.96 A to 3.0 A.
    """
    # Interpolate C-O bond length and O-H distance
    if step_n < 3:
        # Forward: Q -> QH2
        d_co = 1.22 + step_n * 0.07  # 1.22 -> 1.29 -> 1.36
        d_oh = 3.00 - step_n * 1.02  # 3.00 -> 1.98 -> 0.96
    else:
        # Reverse: QH2 -> Q
        rev_n = step_n - 3
        d_co = 1.36 - rev_n * 0.07   # 1.36 -> 1.29 -> 1.22
        d_oh = 0.96 + rev_n * 1.02   # 0.96 -> 1.98 -> 3.00

    # Hexagon ring (C-C ~ 1.40 A)
    r_cc = 1.40
    c1 = (0.0, r_cc, 0.0)
    c2 = (r_cc * 0.866, r_cc * 0.5, 0.0)
    c3 = (r_cc * 0.866, -r_cc * 0.5, 0.0)
    c4 = (0.0, -r_cc, 0.0)
    c5 = (-r_cc * 0.866, -r_cc * 0.5, 0.0)
    c6 = (-r_cc * 0.866, r_cc * 0.5, 0.0)

    # Oxygens attached to C1 and C4 along Y axis
    o1 = (0.0, r_cc + d_co, 0.0)
    o2 = (0.0, -r_cc - d_co, 0.0)

    # Ring Hydrogens (C-H ~ 1.09 A)
    d_ch = 1.09
    h2 = (c2[0] + d_ch * 0.866, c2[1] + d_ch * 0.5, 0.0)
    h3 = (c3[0] + d_ch * 0.866, c3[1] - d_ch * 0.5, 0.0)
    h5 = (c5[0] - d_ch * 0.866, c5[1] - d_ch * 0.5, 0.0)
    h6 = (c6[0] - d_ch * 0.866, c6[1] + d_ch * 0.5, 0.0)

    # Approaching Protons (along Y axis)
    h_prot1 = (0.0, o1[1] + d_oh, 0.0)
    h_prot2 = (0.0, o2[1] - d_oh, 0.0)

    atoms = [
        ("C", *c1), ("C", *c2), ("C", *c3), ("C", *c4), ("C", *c5), ("C", *c6),
        ("O", *o1), ("O", *o2),
        ("H", *h2), ("H", *h3), ("H", *h5), ("H", *h6),
        ("H", *h_prot1), ("H", *h_prot2)
    ]
    return atom_block_to_str(atoms)



# ── Add to Section 2 (Builder Functions) ───────────────────────────────
def _methanogenesis_proxy_geometry(bondlength: float) -> str:
    """Ni-CO-H2 proxy for methanogenesis / formate dehydrogenase.
    Ni at origin, CO along +z, H2 approaching along +x.
    Reaction coordinate: Ni-C distance (3.5 Å → 1.8 Å).
    Total electrons: Ni(28) + C(6) + O(8) + 2xH(1) = 44e (even).
    """
    d_nic = bondlength
    return (
        f"Ni  0.000000  0.000000  0.000000\n"
        f"C   0.000000  0.000000  {d_nic:.6f}\n"
        f"O   0.000000  0.000000  {d_nic + 1.150000:.6f}\n"
        f"H   {d_nic + 1.500000:.6f}  0.000000  0.000000\n"
        f"H   {d_nic + 1.500000:.6f}  0.740000  0.000000"
    )

def _complex_i_proxy_geometry(bondlength: float) -> str:
    """[2Fe-2S] rhombic proxy for Complex I terminal ET chain.
    The Fe–Fe distance (held fixed at 2.70 Å) contracts slightly as
    electrons transfer from NADH through the Fe-S chain to quinone,
    modelling the redox-driven conformational change at the TYKY
    module boundary.

    Args:
        bondlength: Fe–S distance in Angstrom (2.26 → 2.20 Å over 4 steps).
    """
    d = bondlength
    return (
        f"Fe   0.000000  1.350000  0.000000\n"
        f"Fe   0.000000 -1.350000  0.000000\n"
        f"S    {d:.6f}  0.000000  0.000000\n"
        f"S   -{d:.6f}  0.000000  0.000000"
    )

def _ni2s2_proxy_atom_block(d: float) -> str:
    """Ni₂S₂ proxy for CODH/ACS (catalog entry 9).
    
    2×Ni(28) + 2×S(16) + C(6) + O(8) = 102e (even), charge=0, spin_2S=0.
    Reaction coordinate: Ni–S bond compression 2.30 → 2.19 Å,
    modelling the C-cluster → A-cluster CO transfer and
    Ni_p(I)·CO formation (Janus step).
    
    CO ligand bound axially to Ni_p (proximal Ni) along Z-axis,
    perpendicular to the Ni–S–Ni plane, avoiding S–C steric clash.
    """
    import numpy as np
    y_ni = 1.300000
    x_s = np.sqrt(d**2 - y_ni**2)
    
    # CO geometry: Ni–C = 1.80 Å, C–O = 1.15 Å (standard Ni–CO values)
    r_ni_c = 1.800
    r_c_o  = 1.150
    z_c = r_ni_c
    z_o = r_ni_c + r_c_o
    
    return (
        f"Ni  0.000000  {y_ni:.6f}  0.000000\n"      # Ni_p (proximal, with CO)
        f"Ni  0.000000 {-y_ni:.6f}  0.000000\n"      # Ni_d (distal)
        f"S   {x_s:.6f}  0.000000  0.000000\n"
        f"S  {-x_s:.6f}  0.000000  0.000000\n"
        f"C   0.000000  {y_ni:.6f}  {z_c:.6f}\n"     # CO carbon, axial to Ni_p
        f"O   0.000000  {y_ni:.6f}  {z_o:.6f}"       # CO oxygen, linear Ni–C–O
    )

def _cyt_bd_proxy_geometry(bondlength: float) -> str:
    """Fe₂O₂ rhombic proxy for Cyt bd oxidase binuclear heme center.
    2×Fe(26) + 2×O(8) = 68e (even), charge=0, spin_2S=0.
    Reaction coordinate: Fe–O bond compression 2.30→2.20 Å, modelling
    reduction-driven contraction as hemes b558 + b595 become Fe(II).

    Args:
        bondlength: Fe–O distance in Angstrom (2.30 → 2.20 Å over 4 steps).
    """
    d = bondlength
    return (
        f"Fe   0.000000  1.350000  0.000000\n"
        f"Fe   0.000000 -1.350000  0.000000\n"
        f"O    {d:.6f}  0.000000  0.000000\n"
        f"O   -{d:.6f}  0.000000  0.000000"
    )

def _cyt_c_oxidase_proxy_geometry(d: float) -> str:
    """Fe-Cu-N-O binuclear proxy for Cyt c oxidase heme a3-CuB center (entry 11, Group C).
    Fe(26)+Cu(29)+N(7)+O(8)=70e (even), charge=0, spin_2S=0.
    Reaction coordinate: Fe-Cu compression 2.60→2.50 Å, modelling
    R state formation as binuclear center becomes fully reduced.
    """
    half_d = d / 2.0
    return (
        f"Fe   0.000000  0.000000  {half_d:.6f}\n"
        f"Cu   0.000000  0.000000 -{half_d:.6f}\n"
        f"N    1.800000  0.000000  0.000000\n"
        f"O   -1.800000  0.000000  0.000000"
    )

def _codh_acs_proxy_geometry(d: float) -> str:
    """Ni₂S₂-CO proxy for CODH/ACS (entry 9, Group B).
    2×Ni(28)+2×S(16)+C(6)+O(8)=102e (even), charge=0, spin_2S=0.
    Reaction coordinate: Ni-S compression 2.30→2.19 Å, modelling
    C-cluster → A-cluster CO transfer and Ni_p(I)·CO formation.
    CO ligand bound axially to Ni_p (proximal Ni) along Z-axis.
    """
    y_ni = 1.300000
    x_s = math.sqrt(max(d**2 - y_ni**2, 0.0))
    return (
        f"Ni   0.000000  {y_ni:.6f}  0.000000\n"
        f"Ni   0.000000 {-y_ni:.6f}  0.000000\n"
        f"S    {x_s:.6f}  0.000000  0.000000\n"
        f"S   -{x_s:.6f}  0.000000  0.000000\n"
        f"C    0.000000  {y_ni:.6f}  1.800000\n"
        f"O    0.000000  {y_ni:.6f}  2.950000"
    )

# ===========================================================================
# SECTION 4 — PER-STEP BOND LENGTH SEQUENCES
# These match the bondlength_angstrom fields set in mqedatagenerator.py specs.
# ===========================================================================

def _nitrogenase_closed_loop_bondlengths() -> List[float]:
    fwd = [2.260 - i * 0.014 for i in range(8)]        # compression
    return fwd + fwd[::-1]                              # mirror: 16 steps total


# Z3 cofactor requires 2 parameters per step (r12, r13 for H3+ triangle).
Z3_COFACTOR_GEOM_PARAMS: List[Tuple[float, float]] = [
    (0.874, 0.874),   # step 0: equilateral (equilibrium)
    (1.000, 0.874),   # step 1: isoceles (H elongated)
    (0.874, 0.874),   # step 2: equilateral (closure test)
]

BONDLENGTHS: Dict[str, List[float]] = {
    # ── Nitrogenase LT variants ──────────────────────────────────────────────
    "nitrogenase_lt":           [2.260 + i * 0.014 for i in range(8)],
    "nitrogenase_lt_m8":        [2.260 - i * 0.014 for i in range(8)],    # BUG1 FIX: compression
    "nitrogenase_lt_parallel":  [2.260 - i * 0.028 for i in range(4)],    # BUG1 FIX: compression
    "nitrogenase_lt_closed":    [2.260 - i * 0.014 for i in range(8)],    # alias for closed_loop fwd half
    "nitrogenase_closed_loop":  _nitrogenase_closed_loop_bondlengths(),    # 16 steps

    # ── Fe₄S₄ cubane ─────────────────────────────────────────────────────────
    "nitrogenase_fe4s4":        [2.260 + n * 0.015 for n in range(8)],

    # ── Catalog Group A ───────────────────────────────────────────────────────
    "nitrogenase_group_a":      [2.260 - i * 0.010 for i in range(8)],
    "mo_nitrogenase":           [round(2.700 - i * 0.010, 6) for i in range(8)],
    "assimilatory_nr":          [round(2.420 - i * (0.080 / 7), 6) for i in range(8)],
    "photocatalytic_n2":        [round(1.900 - i * (0.344 / 7), 6) for i in range(8)],

    # ── Catalog Group D (12-step) ─────────────────────────────────────────────
    "nitrogenase_group_d":      [round(2.260 - i * (0.088 / 11), 6) for i in range(12)],
    "v_nitrogenase":            [round(2.350 - i * (0.092 / 11), 6) for i in range(12)],
    "cu_co2rr":                 [round(2.550 - i * (0.092 / 11), 6) for i in range(12)],

    # ── Fe–Mo–N₂ trimer ───────────────────────────────────────────────────────
    "femon2_trimer":            [round(1.10 + i * 0.06, 6) for i in range(8)],
    "nitrogenase_femoco":       [round(1.10 + i * 0.06, 6) for i in range(8)],  # N–N stretch proxy

    # ── Haber-Bosch ───────────────────────────────────────────────────────────
    # primary: Fe₂S₂N₂ analytic geometry uses N–N stretch
    # step 0–5 also have fully-enumerated HABER_BOSCH_GEOMETRIES (override used)
    "haber_bosch":              [1.10 + n * 0.06 for n in range(8)],

    # ── PSII ──────────────────────────────────────────────────────────────────
    "psii":                     [2.260 - i * 0.015 for i in range(4)],
    "psii_photo":               [2.260 - n * 0.020 for n in range(4)],

    # ── Hydrogenase ───────────────────────────────────────────────────────────
    "hydrogenase":              [1.40, 0.742],
    "hydrogenase_oxidation":    [0.742, 1.40],

    # ── Small H-chain / organic ───────────────────────────────────────────────
    # z3_cofactor uses Z3_COFACTOR_GEOM_PARAMS (2 params per step); r12 is stored here
    "z3_cofactor":              [p[0] for p in Z3_COFACTOR_GEOM_PARAMS],
    "z5_cofactor":              [1.20, 1.05, 0.90, 0.75, 0.60],
    "reversible_quinone":       [1.2 + 0.1 * n for n in range(6)],

    # ── Anammox (static dict; bondlength here is N–N midpoint, informational) ─
    "anammox_proxy":            [2.6 - n * 0.38 for n in range(4)],

    # ── Thymine dimer ─────────────────────────────────────────────────────────
    "thymine_dimer_proxy":      [2.80, 2.54, 2.28, 2.02, 1.76, 1.50],

    # ── ATP hydrolysis ────────────────────────────────────────────────────────
    "atp_hydrolysis_proxy":     [2.4, 2.2, 2.0, 1.8],

    # ── RNR radical (4 fixed steps, bondlength = d_SH at that step) ──────────
    "rnr_radical_proxy":        [3.0, 1.6, 1.35, 1.34],

    # ── CYP450 ────────────────────────────────────────────────────────────────
    "cyp450_metabolism":        [2.0, 1.8, 1.7, 1.6, 1.65, 2.0],

    # ── Ethylene epoxidation ──────────────────────────────────────────────────
    "ethylene_epoxidation":     [3.5, 2.8, 2.4, 2.0],

    # ── Methanogenesis ──────────────────────────────────────────────────────
    "methanogenesis_proxy": [round(3.50 - n * (1.70 / 7), 6) for n in range(8)],

    # ── Oldform lifts (reuse parent bond-length sequences) ──────────────
    "mo_nitrogenase_m4":      [round(2.700 - i * 0.010, 6) for i in range(8)],
    "v_nitrogenase_m4":       [round(2.350 - i * (0.092 / 11), 6) for i in range(12)],
    "assimilatory_nr_m4":     [round(2.420 - i * (0.080 / 7), 6) for i in range(8)],
    "cu_co2rr_m4":            [round(2.550 - i * (0.092 / 11), 6) for i in range(12)],
    "photocatalytic_n2_m4":   [round(1.900 - i * (0.344 / 7), 6) for i in range(8)],
    # ── New unique entries ──────────────────────────────────────────────
    "complex_i":              [round(2.260 - i * 0.020, 6) for i in range(4)],
    "codh_acs":               [round(2.300 - i * (0.110 / 7), 6) for i in range(8)],
    "cyt_bd_oxidase":         [round(2.300 - i * (0.100 / 3), 6) for i in range(4)],
    "cyt_c_oxidase":          [round(2.600 - i * (0.100 / 3), 6) for i in range(4)],
}


# ===========================================================================
# SECTION 5 — UNIFIED ENTRY POINT
# ===========================================================================

def get_step_geometry(
    mechanism_name: str,
    step_n: int,
    bondlength: Optional[float] = None,
) -> List[Tuple[str, float, float, float]]:
    """Return per-step nuclear geometry for *mechanism_name* at *step_n*.

    Dispatches to the appropriate builder function for each mechanism.  If
    *bondlength* is None it is taken from :data:`BONDLENGTHS`.  Falls back to
    the static Janus geometry if the mechanism is not registered.

    Args:
        mechanism_name: Mechanism identifier string.
        step_n:         Step index (0-based).
        bondlength:     Override for the primary bond length [Å].  When None
                        the registered value for (mechanism_name, step_n) is used.

    Returns:
        List of (symbol, x, y, z) in Angstroms.
    """
    # Resolve bond length from registry when not overridden.
    if bondlength is None:
        bls = BONDLENGTHS.get(mechanism_name)
        if bls is not None and 0 <= step_n < len(bls):
            bondlength = bls[step_n]
        else:
            bondlength = 1.0   # sensible fallback for unknown mechanisms

    mech = mechanism_name

    # ── Nitrogenase LT variants (Fe₂S₂ rhombic core, variable Fe–S) ─────────
    if mech in ("nitrogenase_lt", "nitrogenase_lt_m8", "nitrogenase_lt_parallel",
                "nitrogenase_lt_closed", "nitrogenase_closed_loop",
                "nitrogenase_group_a", "nitrogenase_group_d", "psii"):
        atom_str = _fe2s2_geometry_at_bond(bondlength)

    # ── Fe₄S₄ cubane (breathing mode) ────────────────────────────────────────
    elif mech == "nitrogenase_fe4s4":
        atom_str = _fe4s4_geometry_at_step(step_n)

    # ── PSII photo (Mn₂O₂) ───────────────────────────────────────────────────
    elif mech == "psii_photo":
        atom_str = _psii_photo_geometry_at_step(step_n)

    # ── Hydrogenase (H₂) ─────────────────────────────────────────────────────
    elif mech in ("hydrogenase", "hydrogenase_oxidation"):
        atom_str = _h2_geometry(bondlength)

    # ── Z3 cofactor (H₃⁺ triangle — two bond-length parameters) ─────────────
    elif mech == "z3_cofactor":
        r12, r13 = Z3_COFACTOR_GEOM_PARAMS[step_n % len(Z3_COFACTOR_GEOM_PARAMS)]
        atom_str = _h3plus_geometry(r12, r13)

    # ── Z5 cofactor (H₅⁺ linear chain) ───────────────────────────────────────
    elif mech == "z5_cofactor":
        atom_str = _hchain_geometry(5, bondlength)

    # ── Reversible quinone / generic H₄ chain ────────────────────────────────
    elif mech == "reversible_quinone":
        atom_str = _benzoquinone_geometry(step_n)

    elif mech == "atp_hydrolysis_proxy":
        atom_str = _atp_hydrolysis_geometry(bondlength)

    # ── Haber-Bosch: prefer fully-enumerated dict for steps 0–5 ──────────────
    elif mech == "haber_bosch":
        if step_n in HABER_BOSCH_GEOMETRIES:
            atom_str = HABER_BOSCH_GEOMETRIES[step_n]
        else:
            atom_str = _haber_bosch_fe2s2n2_geometry(step_n)

    # ── Anammox (fully-enumerated dict) ──────────────────────────────────────
    elif mech == "anammox_proxy":
        if step_n in ANAMMOX_GEOMETRIES:
            atom_str = ANAMMOX_GEOMETRIES[step_n]
        else:
            atom_str = _fe2s2_geometry_at_bond(bondlength)   # fallback

    # ── Fe–Mo–N₂ trimer (N–N elongation, linear) ─────────────────────────────
    # elif mech in ("femon2_trimer", "nitrogenase_femoco"):
    #     atom_str = _femon2_trimer_atom_block(bondlength)

    elif mech == "femon2_trimer":
        atom_str = _femon2_trimer_atom_block(bondlength)

    elif mech == "nitrogenase_femoco":
        atom_str = _femoco_full_with_n2_geometry(step_n)

    # ── Mo-nitrogenase (Fe-Mo-S₂ proxy) ──────────────────────────────────────
    elif mech == "mo_nitrogenase":
        atom_str = _femo_proxy_atom_block(bondlength)

    # ── Assimilatory NR (Mo-S₂-O₂ proxy) ────────────────────────────────────
    elif mech == "assimilatory_nr":
        atom_str = _mo_nr_proxy_atom_block(bondlength)

    # ── Photocatalytic N₂ fixation (Ti₂N₂ proxy) ─────────────────────────────
    elif mech == "photocatalytic_n2":
        atom_str = _ti2n2_proxy_atom_block(bondlength)

    # ── V-nitrogenase (V₂S₂ proxy) ───────────────────────────────────────────
    elif mech == "v_nitrogenase":
        atom_str = _v2s2_proxy_atom_block(bondlength)

    # ── Cu CO₂RR (Cu₃ trimer proxy) ──────────────────────────────────────────
    elif mech == "cu_co2rr":
        atom_str = _cu3_proxy_atom_block(bondlength)

    # ── Thymine dimer proxy ───────────────────────────────────────────────────
    elif mech == "thymine_dimer_proxy":
        atom_str = _thymine_dimer_proxy_geometry(bondlength)

    # ── RNR radical proxy ─────────────────────────────────────────────────────
    elif mech == "rnr_radical_proxy":
        atom_str = _rnr_proxy_geometry(min(step_n, 3))

    # ── CYP450 (Fe-O + thiolate + porphyrin N proxy) ─────────────────────────
    elif mech == "cyp450_metabolism":
        atom_str = _cyp450_geometry(bondlength)

    # ── Ethylene epoxidation (Ag₃ + Cl + O + C₂H₄) ─────────────────────────
    elif mech == "ethylene_epoxidation":
        atom_str = _ethylene_epoxidation_geometry(bondlength)

    # ── Methanogenesis ──────────────────────────────────────────────────────
    elif mech == "methanogenesis_proxy":
        atom_str = _methanogenesis_proxy_geometry(bondlength)

    # ── Oldform lifts (reuse parent geometry builders) ──────────────────
    elif mech == "mo_nitrogenase_m4":
        atom_str = _femo_proxy_atom_block(bondlength)
    elif mech == "v_nitrogenase_m4":
        atom_str = _v2s2_proxy_atom_block(bondlength)
    elif mech == "assimilatory_nr_m4":
        atom_str = _mo_nr_proxy_atom_block(bondlength)
    elif mech == "cu_co2rr_m4":
        atom_str = _cu3_proxy_atom_block(bondlength)
    elif mech == "photocatalytic_n2_m4":
        atom_str = _ti2n2_proxy_atom_block(bondlength)

    # ── New unique entries ──────────────────────────────────────────────
    elif mech == "complex_i":
        atom_str = _complex_i_proxy_geometry(bondlength)
    elif mech == "codh_acs":
        atom_str = _codh_acs_proxy_geometry(bondlength)
    elif mech == "cyt_bd_oxidase":
        atom_str = _cyt_bd_proxy_geometry(bondlength)
    elif mech == "cyt_c_oxidase":
        atom_str = _cyt_c_oxidase_proxy_geometry(bondlength)

    # ── Unknown: fall back to static Janus geometry ───────────────────────────
    else:
        return get_janus_geometry(mechanism_name)

    return parse_atom_block(atom_str)


# ===========================================================================
# SECTION 7 — ZETAZERO BUILD SPECS
# ===========================================================================
#
# ZetaZeroSpec is the third mirror of the mechanism description:
#   mqemolecules.py    : MechanismTuple   — quantum circuit layer
#   mqedatagenerator.py: MQEMechanismSpec — PySCF dataset layer
#   mqegeometries.py   : ZetaZeroSpec     — seed-free pipeline layer
#
# All three must agree on M_steps, m_modulus, nu_n, and the Janus step.
# Keeping them here (rather than in mqeprotogeny.py) means geometry
# and build-spec drift is detectable in a single file.
# ===========================================================================

@dataclass(frozen=True)
class ZetaZeroSpec:
    r"""Seed-free pipeline build specification for one MQE mechanism.

    Fields
    ------
    n_electrons : int
        Number of active electrons for CAS diagonalisation (N_e).
        Standard: 4 for Fe/Mo/V/Ti/Cu/Mn clusters; 2 for H₂ / H₃⁺;
        3 for RNR doublet; 4 for H₅⁺ (z5_cofactor).
    n_orbitals : int
        Number of active spatial orbitals (N_orb). Standard: 4.
        Exceptions: hydrogenase (2), z3_cofactor (3).
    tower_p : int
        Prime base for the Iwasawa Kummer tower.
        p=2 for m∈{1,2,4,8}; p=3 for m∈{3,12}; p=5 for m=5.
    janus_step : Optional[int]
        Step index at which the Janus crossing fires (is_crossing=True).
        None for adiabatic mechanisms (no crossing).
    charge : int
        Molecular charge for the CAS atom block, matching
        ``_atom_block_for_step`` in mqedatagenerator.py.
    spin_2S : int
        2×S spin multiplicity (number of unpaired electrons), matching
        ``_atom_block_for_step`` in mqedatagenerator.py.
    expected_energy_ordering : str
        Qualitative E_seed trajectory description: "increasing",
        "decreasing", "monotone_increasing", "closure", or "none".
    spectral_group : str
        Spectral class: "A" (m=8,ν=2,n*=3), "B" (m=4,ν=2,n*=1),
        "C" (m=4,ν=1,n*=3), "D" (m=12,ν=2,n*=5), "F" (m=8,ν=1,n*=7),
        or "" for non-standard (adiabatic or prime-m cofactor scaffolds).
    """
    n_electrons:              int
    n_orbitals:               int
    tower_p:                  int
    janus_step:               Optional[int]
    charge:                   int
    spin_2S:                  int
    expected_energy_ordering: str
    spectral_group:           str


# ---------------------------------------------------------------------------
# ZETAZERO_SPECS registry — one entry per mechanism name
# ---------------------------------------------------------------------------
# Source cross-references (all values verified against mqedatagenerator.py):
#   janus_step   ← is_crossing=(n==X) in _build_*_spec
#   charge       ← _atom_block_for_step dispatch table
#   spin_2S      ← _atom_block_for_step dispatch table
#   tower_p      ← m_modulus: p=2 for m|8, p=3 for m|12 or m=3, p=5 for m=5
#   spectral_group ← mqeriemann._SPECTRAL_CLASSES (m, n*=m//ν−1)
# ---------------------------------------------------------------------------

ZETAZERO_SPECS: Dict[str, "ZetaZeroSpec"] = {

    # ── Group B (m=4, ν=2, n*=1, s=0.08115) ─────────────────────────────────
    # k^(n) = 2(n+1) mod 4; Janus when k_acc first equals m/2=2 at n=0,
    # but biological gate fires later by convention (nitrogenase: E4H4 state).

    "nitrogenase_lt": ZetaZeroSpec(
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=4,
        charge=0, spin_2S=4,
        expected_energy_ordering="decreasing", spectral_group="B",
    ),
    "nitrogenase_lt_parallel": ZetaZeroSpec(
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=2,
        charge=0, spin_2S=4,
        expected_energy_ordering="decreasing", spectral_group="B",
    ),
    "nitrogenase_closed_loop": ZetaZeroSpec(
        # Forward Janus at n=4; reverse Janus at n=11 (BUG8 FIX in spec).
        # janus_step records the forward crossing; reverse is n+7.
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=4,
        charge=0, spin_2S=4,
        expected_energy_ordering="none", spectral_group="B",
    ),
    "nitrogenase_lt_closed": ZetaZeroSpec(
        # Alias for the forward (compression) half of nitrogenase_closed_loop:
        # 8-step [Fe₂S₂] trajectory, same Group B parameters as nitrogenase_lt.
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=4,
        charge=0, spin_2S=4,
        expected_energy_ordering="decreasing", spectral_group="B",
    ),
    "nitrogenase_fe4s4": ZetaZeroSpec(
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=4,
        charge=4, spin_2S=4,
        expected_energy_ordering="none", spectral_group="B",
    ),
    "nitrogenase_femoco": ZetaZeroSpec(
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=4,
        charge=0, spin_2S=4,
        expected_energy_ordering="decreasing", spectral_group="B",
    ),
    "haber_bosch": ZetaZeroSpec(
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=4,
        charge=0, spin_2S=4,
        expected_energy_ordering="decreasing", spectral_group="B",
    ),
    "femon2_trimer": ZetaZeroSpec(
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=4,
        charge=0, spin_2S=4,
        expected_energy_ordering="decreasing", spectral_group="B",
    ),
    "thymine_dimer_proxy": ZetaZeroSpec(
        # ν=2, m=4; k^(1)=2=m/2 → Janus at n=1 (d=2.54 Å, pre-CI approach).
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=1,
        charge=0, spin_2S=0,
        expected_energy_ordering="decreasing", spectral_group="B",
    ),
    "reversible_quinone": ZetaZeroSpec(
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=3,
        charge=0, spin_2S=0,
        expected_energy_ordering="none", spectral_group="B",
    ),
    "cyp450_metabolism": ZetaZeroSpec(
        # Non-uniform ν (steps 0,3 have ν=2; others ν=0). n*=1 via first ν>0.
        # Janus at n=3: O–O cleavage conical intersection (Compound I surface).
        # spin_2S=2 (S=1, triplet): CYP450 Compound I is a doublet/quartet mix;
        # S=1 triplet adiabat chosen per Shaik et al. (Chem Rev 2005).
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=3,
        charge=0, spin_2S=2,
        expected_energy_ordering="decreasing", spectral_group="B",
    ),

    # ── Group A (m=8, ν=2, n*=3, s=0.04090) ─────────────────────────────────

    "nitrogenase_group_a": ZetaZeroSpec(
        # m=8, ν=2 → n*=3. Janus at n=3: k_acc=8=m (first full Z₈ revolution).
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=3,
        charge=0, spin_2S=4,
        expected_energy_ordering="none", spectral_group="A",
    ),
    "mo_nitrogenase": ZetaZeroSpec(
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=3,
        charge=0, spin_2S=4,
        expected_energy_ordering="none", spectral_group="A",
    ),
    "assimilatory_nr": ZetaZeroSpec(
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=3,
        charge=0, spin_2S=4,
        expected_energy_ordering="none", spectral_group="A",
    ),
    "photocatalytic_n2": ZetaZeroSpec(
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=3,
        charge=0, spin_2S=4,
        expected_energy_ordering="none", spectral_group="A",
    ),
    "ethylene_epoxidation": ZetaZeroSpec(
        # m=8, ν=2; Janus at n=2 (d=2.4 Å) by orbital-continuity convention
        # (one mo_cache propagation step before the S₁/S₀ approach region).
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=2,
        charge=0, spin_2S=0,
        expected_energy_ordering="decreasing", spectral_group="A",
    ),

    # ── Group C (m=4, ν=1, n*=3, s=0.04135) ─────────────────────────────────
    # k^(n)=n+1 mod 4; k*=m/2=2 first reached at n=1, but biological gate
    # fires at the mechanistically relevant midpoint.

    "psii": ZetaZeroSpec(
        # Kok S-state cycle; Janus at S2 (n=2), maximally oxidised state.
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=2,
        charge=0, spin_2S=0,
        expected_energy_ordering="none", spectral_group="C",
    ),
    "psii_photo": ZetaZeroSpec(
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=2,
        charge=0, spin_2S=0,
        expected_energy_ordering="monotone_increasing", spectral_group="C",
    ),
    "anammox_proxy": ZetaZeroSpec(
        # Anammox Fe-cluster; ν=1, m=4, Janus at n=2 (same midpoint convention).
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=2,
        charge=0, spin_2S=4,
        expected_energy_ordering="decreasing", spectral_group="C",
    ),
    "rnr_radical_proxy": ZetaZeroSpec(
        # Doublet (S=1/2, 2S=1): thiyl radical HAT. CAS(3,4): 3 electrons in
        # 4 orbitals (σ_SH, σ*_SH, p_S, p_C). Janus at n=1 (S···H···C TS).
        n_electrons=3, n_orbitals=4, tower_p=2, janus_step=1,
        charge=0, spin_2S=1,
        expected_energy_ordering="none", spectral_group="C",
    ),
    "atp_hydrolysis_proxy": ZetaZeroSpec(
        # H₃PO₄+H₂O minimal proxy; Janus at n=2 (P–O=2.0 Å, trigonal-bipyramidal TS).
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=2,
        charge=0, spin_2S=0,
        expected_energy_ordering="decreasing", spectral_group="C",
    ),

    # ── Group D (m=12, ν=2, n*=5, s=0.02743) ────────────────────────────────

    "nitrogenase_group_d": ZetaZeroSpec(
        # m=12, ν=2 → n*=5. Janus at n=5: k_acc=12=m (first full Z₁₂ revolution).
        n_electrons=4, n_orbitals=4, tower_p=3, janus_step=5,
        charge=0, spin_2S=4,
        expected_energy_ordering="none", spectral_group="D",
    ),
    "v_nitrogenase": ZetaZeroSpec(
        n_electrons=4, n_orbitals=4, tower_p=3, janus_step=5,
        charge=0, spin_2S=4,
        expected_energy_ordering="none", spectral_group="D",
    ),
    "cu_co2rr": ZetaZeroSpec(
        # Cu₃⁻ trimer proxy: charge=−1, singlet (88 valence electrons, even).
        n_electrons=4, n_orbitals=4, tower_p=3, janus_step=5,
        charge=-1, spin_2S=0,
        expected_energy_ordering="none", spectral_group="D",
    ),

    # ── Group F (m=8, ν=1, n*=7, s=0.02100) ─────────────────────────────────

    "nitrogenase_lt_m8": ZetaZeroSpec(
        # ℤ₈ phase variant (BUG3 FIX: ν=1 coprime with 8 → full Z₈ orbit).
        # n*=8/1−1=7. Janus at n=3: k_acc=4=m/2 (k* reached at step 3 with ν=1).
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=3,
        charge=0, spin_2S=4,
        expected_energy_ordering="decreasing", spectral_group="F",
    ),

    # ── Adiabatic (trivial group, no Janus crossing) ─────────────────────────

    "hydrogenase": ZetaZeroSpec(
        # 2H⁺ + 2e⁻ → H₂. CAS(2,2): σ and σ* of H₂. m=1 (Z₁ trivial).
        n_electrons=2, n_orbitals=2, tower_p=2, janus_step=None,
        charge=0, spin_2S=0,
        expected_energy_ordering="decreasing", spectral_group="",
    ),
    "hydrogenase_oxidation": ZetaZeroSpec(
        # H₂ → 2H⁺ + 2e⁻. Same CAS(2,2) proxy; reverse bond-length scan.
        n_electrons=2, n_orbitals=2, tower_p=2, janus_step=None,
        charge=0, spin_2S=0,
        expected_energy_ordering="increasing", spectral_group="",
    ),

    # ── Prime-m cofactor scaffolds (non-standard group) ──────────────────────

    "z3_cofactor": ZetaZeroSpec(
        # H₃⁺ triangle: 3H − charge(+1) = 2 electrons. CAS(2,3): all three
        # H₂⁺-like MOs (σ, σ*, nonbonding). m=3 (Z₃), p=3, no Janus.
        n_electrons=2, n_orbitals=3, tower_p=3, janus_step=None,
        charge=1, spin_2S=0,
        expected_energy_ordering="decreasing", spectral_group="",
    ),
    "z5_cofactor": ZetaZeroSpec(
        # H₅⁺ linear chain: 5H − charge(+1) = 4 electrons. CAS(4,4).
        # m=5 (Z₅), p=5, no Janus.
        n_electrons=4, n_orbitals=4, tower_p=5, janus_step=None,
        charge=1, spin_2S=0,
        expected_energy_ordering="decreasing", spectral_group="",
    ),

    "methanogenesis_proxy": ZetaZeroSpec(
        # Ni-CO-H2 proxy; 8-step PCET. Janus at n=4 (Ni-C ~ 2.65 Å).
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=4,
        charge=0, spin_2S=0,  # Singlet (44e, even)
        expected_energy_ordering="decreasing", spectral_group="B",
    ),

    # ── Oldform lifts (Group B) ─────────────────────────────────────────
    "mo_nitrogenase_m4": ZetaZeroSpec(
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=1,
        charge=0, spin_2S=4,
        expected_energy_ordering="decreasing", spectral_group="B",
    ),
    "v_nitrogenase_m4": ZetaZeroSpec(
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=1,
        charge=0, spin_2S=4,
        expected_energy_ordering="decreasing", spectral_group="B",
    ),
    "assimilatory_nr_m4": ZetaZeroSpec(
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=1,
        charge=0, spin_2S=4,
        expected_energy_ordering="decreasing", spectral_group="B",
    ),
    "cu_co2rr_m4": ZetaZeroSpec(
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=1,
        charge=-1, spin_2S=0,
        expected_energy_ordering="decreasing", spectral_group="B",
    ),
    "photocatalytic_n2_m4": ZetaZeroSpec(
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=1,
        charge=0, spin_2S=4,
        expected_energy_ordering="decreasing", spectral_group="B",
    ),
    # ── New unique entries ──────────────────────────────────────────────
    "complex_i": ZetaZeroSpec(
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=2,
        charge=0, spin_2S=0,
        expected_energy_ordering="none", spectral_group="C",
    ),
    "codh_acs": ZetaZeroSpec(
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=1,
        charge=0, spin_2S=0,
        expected_energy_ordering="decreasing", spectral_group="B",
    ),
    "cyt_bd_oxidase": ZetaZeroSpec(
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=2,
        charge=0, spin_2S=0,
        expected_energy_ordering="none", spectral_group="C",
    ),
    "cyt_c_oxidase": ZetaZeroSpec(
        n_electrons=4, n_orbitals=4, tower_p=2, janus_step=2,
        charge=0, spin_2S=0,
        expected_energy_ordering="none", spectral_group="C",
    ),
}


def get_zetazero_spec(mechanism_name: str) -> "ZetaZeroSpec":
    """Return the :class:`ZetaZeroSpec` for *mechanism_name*.

    Raises :exc:`ValueError` if the mechanism is not registered.
    """
    try:
        return ZETAZERO_SPECS[mechanism_name]
    except KeyError:
        raise ValueError(
            f"[mqegeometries] No ZetaZeroSpec registered for '{mechanism_name}'. "
            f"Available: {sorted(ZETAZERO_SPECS)}"
        ) from None
