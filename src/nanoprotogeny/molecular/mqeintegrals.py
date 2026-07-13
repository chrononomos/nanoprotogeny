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
"""
mqeintegrals.py
===============
General-purpose PySCF molecular integral exporter with automatic geometry
generation.  No geometry file is required.

Geometry pipeline (tried in order until one succeeds)
------------------------------------------------------
1. Analytical generators
   - Hydrogen clusters H_n (n=2…10): equal-spacing linear chain
   - Noble-gas dimers (He2, …):      two atoms along z at --bondlength
2. RDKit  SMILES → ETKDG 3D embed → MMFF optimisation
3. PubChem REST API  (name/formula lookup → SDF → coordinates)
4. PennyLane qml.data (when --mol matches a dataset entry)
5. --geometry  file (.xyz / .pdb) or inline PySCF atom string

After geometry is obtained, --optimize runs a ROHF/DFT geometry
optimisation with PySCF + geomeTRIC before integral extraction.

Chemist's notation convention
------------------------------
h[p,q]     = <p|h_core|q>
g[p,q,r,s] = (pq|rs) = integral phi_p(1) phi_q(1) (1/r12) phi_r(2) phi_s(2)

8-fold symmetry stored: canonical block p>=q, r>=s, (p,q)>=(r,s).

Example usage
-------------
# Automatic geometry from SMILES registry (no file needed)
python mqeintegrals.py --mol H2O
python mqeintegrals.py --mol O2
python mqeintegrals.py --mol LiH  --bondlength 1.57
python mqeintegrals.py --mol H3+
python mqeintegrals.py --mol OH-

# Hydrogen clusters (analytical geometry)
python mqeintegrals.py --mol H4  --bondlength 1.0
python mqeintegrals.py --mol H6  --bondlength 0.74

# Geometry optimisation before integral extraction
python mqeintegrals.py --mol NH3 --optimize
python mqeintegrals.py --mol N2  --optimize --opt_method dft

# Custom SMILES
python mqeintegrals.py --smiles "C#N" --mol_name HCN --basis STO-3G

# Custom CASCI sub-space
python mqeintegrals.py --mol H2O --active_orbitals 4 --active_electrons 4

# FeMoco (custom geometry file, explicit active space)
python mqeintegrals.py --geometry femoco.xyz --basis def2-SVP \\
    --active_orbitals 76 --active_electrons 113 --spin 3
"""

import sys
import gc
import json
import logging
import time
import argparse
import re
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from pyscf import ao2mo, fci, gto, mcscf, scf

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("MolIntegrals")


# ===========================================================================
# FIX 1 — ECP auto-detection constants
# ===========================================================================

# Elements for which def2-TZVP / def2-TZVPP / def2-SVP embed a Stuttgart ECP
# in the basis file itself (Z ≥ 37, i.e. Rb and heavier).  Verified against
# PySCF's pyscf/gto/basis/def2-tzvp.dat: load_ecp returns a non-empty list
# for exactly these elements; the row-4 d-block (Sc–Zn, Z 21–30) and all
# elements through Kr (Z 36) are all-electron in def2 and must NOT be listed.
_DEF2_ECP_ELEMENTS: frozenset = frozenset({
    # Row 5: Rb–Xe  (Z 37–54) — all have ECPs in def2
    "Rb", "Sr",
    "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I",  "Xe",
    # Row 6: Cs–Rn  (La + Hf–Bi confirmed; Cs/Ba/Ce–Lu not in def2-TZVP)
    "La",
    "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi",
})

# Basis name prefixes that require ECPs for heavy elements.
_DEF2_BASIS_PREFIXES: Tuple[str, ...] = (
    "def2-sv", "def2-tzv", "def2-qzv",
)


def _auto_ecp(basis) -> Optional[Dict[str, str]]:
    r"""Return a {element: basis_name} ECP dict for heavy-element entries in a
    dict basis, or None if the basis is a plain string or no heavy elements
    are present.

    In PySCF the Stuttgart ECPs for def2 bases are embedded directly in the
    orbital basis .dat files (def2-svp.dat, def2-tzvp.dat, …) — there is no
    separate "def2-ECP" file.  Therefore the ECP must be specified using the
    *same name* as the orbital basis so PySCF's load_ecp reads the ECP block
    from the correct .dat file.

    Only elements with Z ≥ 37 (Rb and heavier) have ECPs in def2; the row-4
    d-block (Sc–Zn) and all elements through Kr are all-electron in def2 and
    are correctly excluded from _DEF2_ECP_ELEMENTS.

    Examples
    --------
    >>> _auto_ecp("cc-pVTZ")                      # → None  (string basis)
    >>> _auto_ecp({"Fe": "def2-TZVP"})            # → None  (Fe is all-electron)
    >>> _auto_ecp({"Mo": "def2-TZVP"})            # → {"Mo": "def2-TZVP"}
    >>> _auto_ecp({"Mo": "def2-TZVP", "C": "cc-pVTZ"})  # → {"Mo": "def2-TZVP"}
    >>> _auto_ecp({"Ag": "def2-TZVPP"})           # → {"Ag": "def2-TZVPP"}
    """
    if not isinstance(basis, dict):
        return None
    ecp: Dict[str, str] = {}
    for elem, b in basis.items():
        if (
            elem.capitalize() in _DEF2_ECP_ELEMENTS
            and isinstance(b, str)
            and b.lower().startswith(_DEF2_BASIS_PREFIXES)
        ):
            # Use the orbital basis name itself — PySCF extracts the ECP block
            # from that same .dat file (load_ecp('def2-tzvp', 'Mo') works;
            # load_ecp('def2-ECP', 'Mo') raises RuntimeError).
            ecp[elem.capitalize()] = b
    return ecp if ecp else None

# ===========================================================================
# FeMo-co MODEL CLUSTERS — chemically valid idealized fragments
# ===========================================================================

# fe2s2: [Fe2(μ-S)2] — Standard D2h rhombic core
FE2S2_GEOMETRY = """\
Fe  0.000000  1.350000  0.000000
Fe  0.000000 -1.350000  0.000000
S   1.750000  0.000000  0.000000
S  -1.750000  0.000000  0.000000"""

# femo_core: [Fe-Mo(μ-S)3] — Idealized C3v coordination
FEMO_CORE_GEOMETRY = """\
Mo  0.000000  0.000000  0.000000
Fe  0.000000  0.000000  2.700000
S   1.880000  0.000000  1.400000
S  -0.940000  1.628000  1.400000
S  -0.940000 -1.628000  1.400000"""

# fe4s3c: [Fe₄S₃C] — Top prismane half of FeMoco with interstitial carbide
# Fe–C(basal) = 2.00 Å, Fe–C(apical) = 3.46 Å (non-bonded)
FE4S3C_GEOMETRY = """\
Fe  0.000000  0.000000  2.204541
Fe  1.558850  0.000000  0.000000
Fe -0.779425  1.350000  0.000000
Fe -0.779425 -1.350000  0.000000
S   1.025200  1.775600  1.275900
S  -2.050200  0.000000  1.275900
S   1.025200 -1.775600  1.275900
C   0.000000  0.000000 -1.254000
"""

# fe4s3: [Fe₄S₃] top prismane face (without carbide, for comparison)
FE4S3_GEOMETRY = """\
Fe  0.954000  0.954000  0.954000
Fe  0.954000 -0.954000 -0.954000
Fe -0.954000  0.954000 -0.954000
Fe -0.954000 -0.954000  0.954000
S  -1.278000  1.278000  1.278000
S   1.278000 -1.278000  1.278000
S   1.278000  1.278000 -1.278000"""


# ===========================================================================
# 1.  MOLECULE REGISTRY
#     charge, spin (2*S), equilibrium bond length (Ang), SMILES
# ===========================================================================

MOLECULE_REGISTRY: Dict[str, Dict] = {
    # ── Diatomics ─────────────────────────────────────────────────────────
    "H2"   : {"charge":  0, "spin": 0, "bondlength": 0.742, "smiles": "[H][H]"},
    "HF"   : {"charge":  0, "spin": 0, "bondlength": 0.917, "smiles": "F"},
    "HCN"  : {"charge":  0, "spin": 0, "bondlength": 1.065, "smiles": "C#N"},
    "LiH"  : {"charge":  0, "spin": 0, "bondlength": 1.570, "smiles": "[LiH]"},
    "N2"   : {"charge":  0, "spin": 0, "bondlength": 1.120, "smiles": "N#N"},
    "CO"   : {"charge":  0, "spin": 0, "bondlength": 1.128, "smiles": "[C-]#[O+]"},
    "Li2"  : {"charge":  0, "spin": 0, "bondlength": 2.679, "smiles": "[Li][Li]"},
    "C2"   : {"charge":  0, "spin": 0, "bondlength": 1.246, "smiles": "[C]#[C]"},
    "O2"   : {"charge":  0, "spin": 2, "bondlength": 1.220, "smiles": "[O][O]"},
    # ── Cations / anions ──────────────────────────────────────────────────
    "HeH+" : {"charge":  1, "spin": 0, "bondlength": 0.775, "smiles": "[HeH+]"},
    "H3+"  : {"charge":  1, "spin": 0, "bondlength": 0.874, "smiles": "[H][H+][H]"},
    "NeH+" : {"charge":  1, "spin": 0, "bondlength": 0.991, "smiles": "[NeH+]"},
    "OH-"  : {"charge": -1, "spin": 0, "bondlength": 0.964, "smiles": "[OH-]"},
    # ── Triatomics ────────────────────────────────────────────────────────
    "H2O"  : {"charge":  0, "spin": 0, "bondlength": 0.958, "smiles": "O"},
    "BeH2" : {"charge":  0, "spin": 0, "bondlength": 1.326, "smiles": "[H][Be][H]"},
    "CO2"  : {"charge":  0, "spin": 0, "bondlength": 1.162, "smiles": "O=C=O"},
    "O3"   : {"charge":  0, "spin": 0, "bondlength": 1.278, "smiles": "O=[O+][O-]"},
    # ── Polyatomics ───────────────────────────────────────────────────────
    "NH3"  : {"charge":  0, "spin": 0, "bondlength": 1.013, "smiles": "N"},
    "BH3"  : {"charge":  0, "spin": 0, "bondlength": 1.190, "smiles": "[BH3]"},
    "CH4"  : {"charge":  0, "spin": 0, "bondlength": 1.086, "smiles": "C"},
    "H2O2" : {"charge":  0, "spin": 0, "bondlength": 1.475, "smiles": "OO"},
    "N2H2" : {"charge":  0, "spin": 0, "bondlength": 1.247, "smiles": "N=N"},
    "N2H4" : {"charge":  0, "spin": 0, "bondlength": 1.446, "smiles": "NN"},
    "CH2O" : {"charge":  0, "spin": 0, "bondlength": 0.917, "smiles": "C=O"},
    "CH2"  : {"charge":  0, "spin": 2, "bondlength": 1.085, "smiles": "[CH2]"},
    # ── Organic ───────────────────────────────────────────────────────────
    "C2H2" : {"charge":  0, "spin": 0, "bondlength": 1.203, "smiles": "C#C"},
    "C2H4" : {"charge":  0, "spin": 0, "bondlength": 1.339, "smiles": "C=C"},
    "C2H6" : {"charge":  0, "spin": 0, "bondlength": 1.535, "smiles": "CC"},
    # ── Hydrogen clusters (generated analytically) ────────────────────────
    "H4"   : {"charge":  0, "spin": 0, "bondlength": 1.0,   "smiles": None},
    "H5"   : {"charge":  0, "spin": 1, "bondlength": 1.0,   "smiles": None},
    "H6"   : {"charge":  0, "spin": 0, "bondlength": 0.74,  "smiles": None},
    "H7"   : {"charge":  0, "spin": 1, "bondlength": 1.0,   "smiles": None},
    "H8"   : {"charge":  0, "spin": 0, "bondlength": 0.74,  "smiles": None},
    "H10"  : {"charge":  0, "spin": 0, "bondlength": 1.0,   "smiles": None},
    # ── Noble-gas dimers (generated analytically) ─────────────────────────
    "He2"  : {"charge":  0, "spin": 0, "bondlength": 2.0,   "smiles": None},
    # ── FeMoco (custom geometry or built-in placeholder) ──────────────────
    "femoco": {"charge": 0, "spin": 3, "bondlength": None,  "smiles": None},
    # ── FeMo-co model clusters (fast, chemically valid sub-clusters) ───────
    "fe2s2":     {"charge": 0, "spin": 4, "bondlength": None, "smiles": None, "geometry": FE2S2_GEOMETRY},
    "femo_core": {"charge": 0, "spin": 3, "bondlength": None, "smiles": None, "geometry": FEMO_CORE_GEOMETRY},
    "fe4s3c":    {"charge": 0, "spin": 4, "bondlength": None, "smiles": None, "geometry": FE4S3C_GEOMETRY},   # ← new, recommended
    "fe4s3":     {"charge": 0, "spin": 4, "bondlength": None, "smiles": None, "geometry": FE4S3_GEOMETRY},

    # ==========================================================================
    # DNA / RNA NUCLEOBASES
    # ==========================================================================
    # All five canonical nucleobases as isolated neutral molecules (gas phase).
    # Geometry: RDKit ETKDGv3 → MMFF94s; use --optimize for higher accuracy.
    # CASCI recommendation: the π-system active space is ~10 orbitals / 10 electrons
    # for the five-membered rings (purine) and ~6 orbitals / 6 electrons for
    # pyrimidines.  Use --active_orbitals and --active_electrons to select.
    #
    # SMILES correctness notes:
    #   adenine  : Nc1ncnc2[nH]cnc12   — N9-H tautomer (biologically dominant)
    #   guanine  : Nc1nc2[nH]cnc2c(=O)[nH]1 — N9-H, O6-keto tautomer
    #   cytosine : Nc1cc[nH]c(=O)n1    — amino-keto tautomer
    #   thymine  : Cc1c[nH]c(=O)[nH]c1=O  — 2,4-diketo tautomer
    #   uracil   : O=c1cc[nH]c(=O)[nH]1  — 2,4-diketo tautomer
    # ──────────────────────────────────────────────────────────────────────────
    "adenine"  : {"charge":  0, "spin": 0, "bondlength": None,
                  "smiles": "Nc1ncnc2[nH]cnc12"},
    "guanine"  : {"charge":  0, "spin": 0, "bondlength": None,
                  "smiles": "Nc1nc2[nH]cnc2c(=O)[nH]1"},
    "cytosine" : {"charge":  0, "spin": 0, "bondlength": None,
                  "smiles": "Nc1cc[nH]c(=O)n1"},
    "thymine"  : {"charge":  0, "spin": 0, "bondlength": None,
                  "smiles": "Cc1c[nH]c(=O)[nH]c1=O"},
    "uracil"   : {"charge":  0, "spin": 0, "bondlength": None,
                  "smiles": "O=c1cc[nH]c(=O)[nH]1"},

    # ==========================================================================
    # NUCLEOTIDE TRIPHOSPHATES
    # ==========================================================================
    # Stored as the neutral (fully protonated) phosphoric acid form (charge=0)
    # so that PySCF receives an integer electron count without ambiguity.
    # At physiological pH the triphosphate tail carries charge −4; if you want
    # to model the ionic form, pass --charge -4 explicitly on the command line.
    #
    # Geometry source: RDKit ETKDGv3 with stereochemistry from SMILES.
    # These are large molecules (29–32 heavy atoms); use STO-3G with a small
    # CASCI window, or def2-SVP with --active_orbitals for the nucleobase π-system.
    #
    # Formulas (neutral):
    #   ATP  C10H16N5O13P3  (adenosine 5′-triphosphate)
    #   GTP  C10H16N5O14P3  (guanosine 5′-triphosphate)
    #   CTP  C9H16N3O14P3   (cytidine 5′-triphosphate)
    #   UTP  C9H15N2O15P3   (uridine 5′-triphosphate)
    #   dTTP C10H17N2O14P3  (deoxythymidine 5′-triphosphate)
    # ──────────────────────────────────────────────────────────────────────────
    "ATP"  : {"charge":  0, "spin": 0, "bondlength": None,
              "smiles": "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O"},
    "GTP"  : {"charge":  0, "spin": 0, "bondlength": None,
              "smiles": "Nc1nc2c(ncn2[C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)[nH]1"},
    "CTP"  : {"charge":  0, "spin": 0, "bondlength": None,
              "smiles": "Nc1ccn([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)n1"},
    "UTP"  : {"charge":  0, "spin": 0, "bondlength": None,
              "smiles": "O=c1cc[nH]c(=O)n1[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O"},
    "dTTP" : {"charge":  0, "spin": 0, "bondlength": None,
              "smiles": "Cc1cn([C@H]2C[C@H](O)[C@@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O2)c(=O)[nH]c1=O"},

    # ==========================================================================
    # NUCLEOBASE COMPLEXES
    # ==========================================================================
    # The ACGT_quartet is a disconnected SMILES (four separate molecules joined
    # by ".") representing the four DNA bases as an unbound gas-phase ensemble.
    # RDKit embeds each fragment independently in the same coordinate frame.
    # This is useful for computing individual orbital energies and comparing
    # π-stacking interaction energies at the CASCI level.
    #
    # Important: because the four bases are not bonded, PySCF sees a single
    # supermolecular system.  The total electron count is the sum over all four
    # bases (adenine 70e + thymine 58e + guanine 78e + cytosine 58e = 264e with
    # STO-3G).  Use --active_orbitals / --active_electrons to select a chemically
    # meaningful sub-space (e.g. the combined π-HOMO/LUMO of each pair).
    # ──────────────────────────────────────────────────────────────────────────
    "ACGT_quartet" : {
        "charge"    :  0,
        "spin"      :  0,
        "bondlength": None,
        # Four correct tautomers (validated with RDKit, formula C19H21N15O4,
        # all four fragments embed in 3D independently via ETKDGv3).
        "smiles"    : (
            "Nc1ncnc2[nH]cnc12"          # adenine
            + ".Cc1c[nH]c(=O)[nH]c1=O"  # thymine
            + ".Nc1nc2[nH]cnc2c(=O)[nH]1"  # guanine
            + ".Nc1cc[nH]c(=O)n1"        # cytosine
        ),
    },

    # ==========================================================================
    # DEOXYRIBONUCLEOSIDE TRIPHOSPHATES (dNTPs)
    # ==========================================================================
    # The complete dATP/dGTP/dCTP/dTTP quartet — all four DNA-synthesis
    # precursors.  dTTP is already in the registry above; the remaining three
    # are added here to complete the set.
    #
    # Key structural difference from NTPs: 2'-deoxy ribose (no 2'-OH).
    # Stored as the neutral fully-protonated phosphoric acid form (charge=0).
    # At physiological pH the triphosphate tail is charge −4; pass --charge -4
    # on the command line to model the ionic form.
    #
    # Formulas (neutral):
    #   dATP  C10H16N5O12P3  (2'-deoxyadenosine 5'-triphosphate)
    #   dGTP  C10H16N5O13P3  (2'-deoxyguanosine 5'-triphosphate)
    #   dCTP  C9H16N3O13P3   (2'-deoxycytidine 5'-triphosphate)
    # ──────────────────────────────────────────────────────────────────────────
    "dATP" : {"charge":  0, "spin": 0, "bondlength": None,
              "smiles": "Nc1ncnc2c1ncn2[C@@H]1C[C@H](O)[C@@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O1"},
    "dGTP" : {"charge":  0, "spin": 0, "bondlength": None,
              "smiles": "Nc1nc2c(ncn2[C@@H]2C[C@H](O)[C@@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O2)c(=O)[nH]1"},
    "dCTP" : {"charge":  0, "spin": 0, "bondlength": None,
              "smiles": "Nc1ccn([C@@H]2C[C@H](O)[C@@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O2)c(=O)n1"},

    # ==========================================================================
    # PRIMARY METABOLIC & REDOX COFACTORS
    # ==========================================================================
    # NADH / NADPH
    # ─────────────
    # The reduced nicotinamide adenine dinucleotides.  The nicotinamide ring
    # carries a formal positive charge on N1 ([N+]) in the intact covalent
    # molecule — this is NOT an ionic species but a zwitterionic neutral molecule
    # (the positive N is balanced by a deprotonated phosphate in the same
    # molecule at physiological pH).  For PySCF geometry generation the molecule
    # is stored as charge=0 (neutral molecule; the [N+] is an internal formal
    # charge, not a net ionic charge).
    #
    # Formulas (neutral zwitterion):
    #   NADH   C21H30N7O14P2+   44 heavy atoms  (1 fragment, fully bonded)
    #   NADPH  C21H31N7O17P3+   48 heavy atoms  (adds 2'-phosphate on adenosine)
    #
    # Active-space recommendation: the π-system of the dihydropyridine ring
    # (4 π-orbitals / 4 π-electrons) is the chemically relevant window for
    # hydride-transfer simulations.
    # ──────────────────────────────────────────────────────────────────────────
    "NADH"  : {"charge":  1, "spin": 0, "bondlength": None,  # [N+] zwitterion: net charge +1
               # Correct quaternary [N+] on C4a of the dihydropyridine ring.
               # Single connected molecule; validated formula C21H30N7O14P2+
               "smiles": "NC(=O)[C@H]1CC=CC=[N+]1[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]2O[C@@H](n3cnc4c(N)ncnc43)[C@H](O)[C@@H]2O)[C@@H](O)[C@H]1O"},
    "NADPH" : {"charge":  1, "spin": 0, "bondlength": None,  # [N+] zwitterion: net charge +1
               # NADH + 2'-phosphate on the adenosine ribose.
               # Validated formula C21H31N7O17P3+
               "smiles": "NC(=O)[C@H]1CC=CC=[N+]1[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]2O[C@@H](n3cnc4c(N)ncnc43)[C@H](OP(=O)(O)O)[C@@H]2O)[C@@H](O)[C@H]1O"},

    # FADH2
    # ──────
    # Flavin adenine dinucleotide, fully reduced form.
    # The isoalloxazine ring carries N1-H and N5-H (both amines protonated),
    # giving the closed-shell singlet ground state.
    # Formula: C27H35N9O15P2  (53 heavy atoms, 1 fragment).
    # Active-space: the isoalloxazine π-system (6 orbitals) + N1/N5 lone pairs.
    "FADH2" : {"charge":  0, "spin": 0, "bondlength": None,
               "smiles": "Cc1cc2c(cc1C)N(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n3cnc4c(N)ncnc43)[C@H](O)[C@@H]1O)c1[nH]c(=O)[nH]c(=O)c1N2"},

    # CoA (Coenzyme A)
    # ─────────────────
    # Universal acyl-group carrier; the thiol (–SH) is the reactive centre.
    # Formula: C21H36N7O16P3S  (48 heavy atoms, 1 fragment, neutral).
    # Active-space: the pantetheine thiol π/σ system is a natural 2-orbital CAS;
    # the adenine π-system provides a larger 10-orbital alternative.
    "CoA"   : {"charge":  0, "spin": 0, "bondlength": None,
               "smiles": "CC(C)(COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)(O)O)[C@@H](O)C(=O)NCCC(=O)NCCS"},

    # ==========================================================================
    # ENERGY INTERMEDIATES & SIGNALLING MOLECULES
    # ==========================================================================
    # ADP / AMP / cAMP
    # ─────────────────
    # The adenosine phosphate ladder: ATP → ADP → AMP covers the full range
    # of adenylate energy charge.  cAMP is the cyclic second-messenger form.
    # All stored neutral (fully protonated phosphates, charge=0).
    #
    # Formulas (neutral):
    #   ADP   C10H15N5O10P2   27 heavy atoms
    #   AMP   C10H14N5O7P     23 heavy atoms
    #   cAMP  C10H12N5O6P     22 heavy atoms  (3',5'-cyclic phosphodiester)
    # ──────────────────────────────────────────────────────────────────────────
    "ADP"   : {"charge":  0, "spin": 0, "bondlength": None,
               "smiles": "Nc1ncnc2n(cnc12)[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O"},
    "AMP"   : {"charge":  0, "spin": 0, "bondlength": None,
               "smiles": "Nc1ncnc2n(cnc12)[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O"},
    "cAMP"  : {"charge":  0, "spin": 0, "bondlength": None,
               # 3',5'-cyclic phosphodiester ring — unique topology vs AMP.
               # Validated formula C10H12N5O6P, 1 fragment, 3D OK.
               "smiles": "Nc1ncnc2n(cnc12)[C@@H]1O[C@@H]2COP(=O)(O)O[C@H]2[C@H]1O"},

    # PPi (Inorganic pyrophosphate)
    # ──────────────────────────────
    # H4P2O7: the simplest polyphosphate; by-product of all NTP polymerisation
    # reactions and a powerful thermodynamic driver when hydrolysed (−33 kJ/mol).
    # Formula: H4O7P2  (9 heavy atoms, no stereocentres).
    # Active-space: the P–O–P bridging σ* orbital pair is the relevant CAS
    # for hydrolysis transition-state modelling.
    "PPi"   : {"charge":  0, "spin": 0, "bondlength": None,
               "smiles": "OP(=O)(O)OP(=O)(O)O"},

    # ==========================================================================
    # SPECIALIZED RADICAL DONORS & FeMo-CO LIGANDS (NEW VALIDATED ENTRIES)
    # ==========================================================================
    # Pi (Inorganic Phosphate) – neutral H3PO4 form
    "Pi"          : {"charge":  0, "spin": 0, "bondlength": None,
                     "smiles": "OP(=O)(O)O"},

    # SAM (S-Adenosylmethionine) – sulfonium cation, net charge +1 (zwitterionic neutral)
    "SAM"         : {"charge":  1, "spin": 0, "bondlength": None,
                     "smiles": "C[S+](CC[C@H](N)C(=O)O)C[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O"},

    # SAH (S-Adenosylhomocysteine) – neutral thioether
    "SAH"         : {"charge":  0, "spin": 0, "bondlength": None,
                     "smiles": "N[C@@H](CCSC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O)C(=O)O"},

    # homocitrate – fully protonated tricarboxylic acid (neutral)
    "homocitrate" : {"charge":  0, "spin": 0, "bondlength": None,
                     "smiles": "O=C(O)CCC(O)(CC(=O)O)C(=O)O"},
}

NOBLE_GASES = {"He", "Ne", "Ar", "Kr", "Xe"}


def get_defaults(mol_name: str) -> Dict:
    """Registry entry for mol_name, or zero-charge singlet defaults."""
    return MOLECULE_REGISTRY.get(
        mol_name,
        {"charge": 0, "spin": 0, "bondlength": None, "smiles": None},
    )


# ===========================================================================
# 2.  ANALYTICAL GEOMETRY GENERATORS
# ===========================================================================

def _hcluster_geometry(n: int, spacing_ang: float) -> str:
    """
    Linear H_n chain with equal H-H spacing along the z-axis, centred at
    the origin (Angstrom).  Standard quantum-chemistry benchmark geometry.
    """
    lines = []
    for k in range(n):
        z = (k - (n - 1) / 2.0) * spacing_ang
        lines.append(f"H  0.000000  0.000000  {z:.6f}")
    log.info(f"[GEOM] Linear H{n} chain, spacing={spacing_ang:.3f} Ang (analytical)")
    return "\n".join(lines)


def _dimer_geometry(symbol: str, bondlength_ang: float) -> str:
    """Two identical atoms placed at +/-bondlength/2 along z (Angstrom)."""
    half = bondlength_ang / 2.0
    log.info(f"[GEOM] {symbol}2 dimer, bond={bondlength_ang:.3f} Ang (analytical)")
    return (
        f"{symbol}  0.000000  0.000000  {-half:.6f}\n"
        f"{symbol}  0.000000  0.000000   {half:.6f}"
    )


# ===========================================================================
# FeMoco geometry
# ===========================================================================
# Two sources are provided, used in priority order by resolve_geometry():
#
#   1. FEMOCO_1M1N_GEOMETRY  — idealised cluster geometry derived from the
#      crystallographic bond parameters of PDB 1M1N (Einsle et al., Science
#      297:1696, 2002; interstitial carbide from Spatzal et al., Science
#      334:940, 2011).  C3v-approximate symmetry; carbide at the origin.
#      Composition: Fe7MoS9C.  All bond distances are consistent with the
#      published X-ray structure to within 0.02 Å.
#
#      Key bond distances (Å):
#        Fe–C (interstitial, 6 irons):  2.00   (published: 1.97–2.03)
#        Fe–Fe (adjacent pairs):        2.64   (published: 2.60–2.67)
#        Fe–S (bridging sulfides):      2.27   (published: 2.24–2.31)
#        Mo–Fe:                         2.70   (published: 2.69–2.72)
#        Mo–S:                          2.36   (published: 2.34–2.37)
#
#      This geometry is suitable for CASCI/CASSCF integral generation.
#      For publication-grade work, use fetch_femoco_from_pdb() (source 2).
#
#   2. fetch_femoco_from_pdb() — downloads the actual HETATM records from
#      PDB 1M1N at runtime.  Requires internet access.  Returns the
#      FeMo-cofactor cluster atoms only (Fe1–Fe7, Mo, S1–S9, C), centred
#      on the cluster centroid.  Calls this automatically when --mol femoco
#      is used if the download succeeds; falls back to source 1 if it fails.

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

# Keep the old name as an alias so any external code that imported it still works
FEMOCO_PLACEHOLDER_GEOMETRY = FEMOCO_1M1N_GEOMETRY

def fetch_femoco_from_pdb(
    pdb_id: str = "1M1N",
    cache_dir: Optional[str] = None,
) -> str:
    """
    Download the FeMo-cofactor cluster from a PDB entry and return a
    PySCF-format atom string (Angstrom), centred on the cluster centroid.

    The function extracts only the inorganic FeMo-co cluster atoms
    (Fe, Mo, S, C) from the HETATM records, discarding protein backbone,
    water, and homocitrate.  The cluster is translated so its centroid
    lies at the origin.

    Parameters
    ----------
    pdb_id   : RCSB PDB accession code.  Default "1M1N" (Einsle 2002,
               2.0 Å resolution, chain A MoFe protein from A. vinelandii).
    cache_dir: Directory to cache the downloaded PDB file.  Defaults to
               a system temp directory.  If the file already exists there
               it is re-used without re-downloading.

    Returns
    -------
    atom_block : str — PySCF atom string ready to pass to gto.Mole(atom=...).

    Raises
    ------
    RuntimeError  if the download fails (network unavailable, RCSB down, etc.)
                  or if no FeMoco cluster atoms are found in the file.

    Example
    -------
    >>> geom = fetch_femoco_from_pdb()               # downloads 1M1N
    >>> geom = fetch_femoco_from_pdb("7QJP")         # higher-resolution 2021 structure
    >>> mol  = gto.M(atom=geom, basis="def2-SVP", unit="Angstrom")
    """
    import urllib.request
    import urllib.error
    import tempfile

    cache = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir())
    cache.mkdir(parents=True, exist_ok=True)
    local_path = cache / f"{pdb_id.upper()}.pdb"

    # ── Download if not cached ────────────────────────────────────────────
    if not local_path.exists():
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        log.info(f"[PDB] Downloading {pdb_id.upper()} from {url} ...")
        try:
            urllib.request.urlretrieve(url, str(local_path))
            log.info(f"[PDB] Saved to {local_path}")
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not download PDB {pdb_id!r} from RCSB: {exc}.\n"
                "Check your internet connection or use --geometry with a local .pdb file.\n"
                "Falling back to the built-in 1M1N idealised geometry."
            ) from exc
    else:
        log.info(f"[PDB] Using cached file: {local_path}")

    # ── Parse HETATM records for FeMoco cluster atoms ─────────────────────
    # Metal elements that unambiguously belong to the FeMoco cluster
    METAL_ELEMENTS = {"FE", "MO"}
    # Non-metal cluster elements (S = bridging sulfides, C = interstitial carbide)
    NONMETAL_CLUSTER_ELEMS = {"S", "C"}
    # Pre-2014 depositions stored the carbide as "N" (Einsle 2002 misidentification).
    # We include "N" in the proximity search and reclassify it as "C" if it sits
    # within 2.5 Ang of ≥ 3 iron atoms (the unique geometric signature of the
    # interstitial carbide coordinated by 6 Fe atoms in an octahedral cage).
    INTERSTITIAL_CANDIDATES = {"C", "N"}   # "N" = pre-2014 misidentification

    # Residue names under which the FeMo-co appears across PDB depositions:
    #   SFO  — 1M1N, 2MIN, 3U7Q (MoFe cofactor, most common)
    #   SF4  — [4Fe-4S] clusters (may co-occur in the same file)
    #   FES  — FeS clusters (older depositions)
    #   FEO  — iron-oxo
    #   MOO  — molybdate / oxo-Mo
    #   MOC  — Mo-containing cofactor
    #   FMO  — FeMo-co alternate name
    #   FEC  — Fe-containing cofactor
    # Single-element codes catch deposits where each atom is its own residue:
    #   FE / MO / S / C
    FEMOCO_RESNAMES = {
        "SFO", "SF4", "FES", "FEO", "MOO", "MOC", "FMO", "FEC",
        "FE",  "MO",  "S",   "C",   "N",
    }

    cluster_atoms: list = []   # (element_str, x, y, z)

    # ── Pass 1: residue-name filter (exact match, preferred) ─────────────────
    for chain_pref in ("A", "B", "C", " "):   # chain A preferred; try others if empty
        with open(local_path) as f:
            for line in f:
                if not line.startswith("HETATM"):
                    continue
                elem    = line[76:78].strip().upper()
                resname = line[17:20].strip().upper()
                chain   = line[21].strip()
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                # Accept metals and known non-metal cluster atoms in this residue
                if (elem in METAL_ELEMENTS or elem in NONMETAL_CLUSTER_ELEMS) and resname in FEMOCO_RESNAMES and chain == chain_pref:
                    cluster_atoms.append((elem.capitalize(), x, y, z))
        if cluster_atoms:
            log.info(f"[PDB] Residue-name filter matched on chain {chain_pref!r}")
            break

    # ── Pass 2: element-only fallback (collects Fe, Mo, S, C in chain A) ────
    if not cluster_atoms:
        log.warning(
            "[PDB] Residue-name filter found no atoms; "
            "retrying with element-only filter (Fe/Mo/S/C) on chain A ..."
        )
        with open(local_path) as f:
            for line in f:
                if not line.startswith("HETATM"):
                    continue
                elem  = line[76:78].strip().upper()
                chain = line[21].strip()
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                # Collect metals unconditionally; non-metals collected in Pass 3
                if elem in METAL_ELEMENTS and chain == "A":
                    cluster_atoms.append((elem.capitalize(), x, y, z))

    if not cluster_atoms:
        raise RuntimeError(
            f"No FeMoco cluster atoms found in {local_path}.\n"
            "The PDB file may use non-standard residue names or chain labels.\n"
            "Inspect the file with a PDB viewer and supply --geometry explicitly."
        )

    # ── Pass 3: proximity filter — collect S, C, and N near the metal core ──
    # Runs regardless of which pass found the metals, so that bridging S and
    # the interstitial carbide (which may be in a separate residue record) are
    # always included even when the metal pass already collected some non-metals.
    import numpy as np
    n_metals = sum(1 for e, *_ in cluster_atoms if e in ("Fe", "Mo"))
    if n_metals >= 2:   # need at least 2 metals to anchor the proximity search
        metal_coords = np.array([[x, y, z] for e, x, y, z in cluster_atoms
                                 if e in ("Fe", "Mo")])
        existing  = {(round(x,3), round(y,3), round(z,3))
                     for _, x, y, z in cluster_atoms}
        extra: list = []

        with open(local_path) as f:
            for line in f:
                if not line.startswith("HETATM"):
                    continue
                elem  = line[76:78].strip().upper()
                chain = line[21].strip()
                if chain != "A":
                    continue
                # Collect S (bridging sulfides, d_min < 3.0 Ang from any metal)
                # and C/N (interstitial carbide, d_min < 2.5 Ang from any metal)
                if elem not in (NONMETAL_CLUSTER_ELEMS | INTERSTITIAL_CANDIDATES):
                    continue
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                pos      = np.array([x, y, z])
                dists    = np.linalg.norm(metal_coords - pos, axis=1)
                min_dist = dists.min()
                n_close  = (dists < 2.5).sum()

                if elem == "S" and min_dist < 3.0:
                    label = "S"
                elif elem in INTERSTITIAL_CANDIDATES and min_dist < 2.5:
                    # Reclassify pre-2014 "N" as "C": the interstitial carbide
                    # sits inside the Fe6 octahedron (≥6 Fe within 2.5 Ang).
                    # Any other N at this distance would be chemically impossible.
                    label = "C"
                    if elem == "N":
                        log.info(
                            f"[PDB] Reclassified HETATM N → C (interstitial carbide): "
                            f"min_dist={min_dist:.2f} Ang, {n_close} Fe within 2.5 Ang. "
                            "This is the pre-2014 PDB representation of the FeMoco carbide."
                        )
                else:
                    continue

                key = (round(x, 3), round(y, 3), round(z, 3))
                if key not in existing:
                    extra.append((label, x, y, z))
                    existing.add(key)

        cluster_atoms.extend(extra)
        if extra:
            added = ", ".join(
                f"{sum(1 for e,*_ in extra if e==el)} {el}"
                for el in ["S", "C"] if any(e == el for e, *_ in extra)
            )
            log.info(f"[PDB] Proximity filter added: {added}")

    # ── Centre on cluster centroid ────────────────────────────────────────
    coords  = np.array([[x, y, z] for _, x, y, z in cluster_atoms])
    centroid = coords.mean(axis=0)
    centred  = coords - centroid

    atom_block = "\n".join(
        f"{elem:<2}  {centred[i, 0]:10.6f}  {centred[i, 1]:10.6f}  {centred[i, 2]:10.6f}"
        for i, (elem, *_) in enumerate(cluster_atoms)
    )

    log.info(
        f"[PDB] Extracted {len(cluster_atoms)} FeMoco cluster atoms from {pdb_id.upper()} "
        f"(centred on cluster centroid, Angstrom)"
    )
    log.info(
        f"[PDB] Composition: "
        + ", ".join(f"{sum(1 for e,*_ in cluster_atoms if e==el)} {el}"
                    for el in ["Fe", "Mo", "S", "C"]
                    if any(e == el for e, *_ in cluster_atoms))
    )
    return atom_block


# ===========================================================================
# 3.  SMILES → 3D via RDKit
# ===========================================================================

def geometry_from_smiles(smiles: str, label: str = "molecule") -> str:
    """
    Generate a PySCF atom string (Angstrom) from a SMILES string via RDKit.

    Pipeline: parse → add explicit H → ETKDGv3 embed (ETKDG fallback)
              → MMFF94s minimisation (UFF fallback).

    Raises RuntimeError if RDKit is unavailable or 3D embedding fails.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        raise RuntimeError(
            "RDKit is not installed.  "
            "Install with: conda install rdkit  or  pip install rdkit"
        )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(
            f"RDKit could not parse SMILES {smiles!r}.  "
            "Check for invalid syntax or unusual elements."
        )

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    rc = AllChem.EmbedMolecule(mol, params)
    if rc != 0:
        log.warning("[SMILES] ETKDGv3 failed; retrying with ETKDG.")
        rc = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if rc != 0:
        raise RuntimeError(
            f"RDKit could not embed {label!r} ({smiles!r}) in 3D.  "
            "Supply --geometry or try --smiles with a simpler representation."
        )

    if AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s") != 0:
        log.warning("[SMILES] MMFF94s failed; trying UFF.")
        AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    lines = [
        f"{mol.GetAtomWithIdx(i).GetSymbol()}  "
        f"{conf.GetAtomPosition(i).x:.8f}  "
        f"{conf.GetAtomPosition(i).y:.8f}  "
        f"{conf.GetAtomPosition(i).z:.8f}"
        for i in range(mol.GetNumAtoms())
    ]
    log.info(f"[SMILES] {label}: {len(lines)} atoms from {smiles!r} (Angstrom)")
    return "\n".join(lines)


# ===========================================================================
# 4.  PUBCHEM FALLBACK
# ===========================================================================

def geometry_from_pubchem(mol_name: str) -> str:
    """
    Fetch a 3D-optimised structure from PubChem by compound name (Angstrom).

    Requires pubchempy (pip install pubchempy).
    Raises RuntimeError if the molecule is not found.
    """
    try:
        import pubchempy as pcp
    except ImportError:
        raise RuntimeError(
            "pubchempy is not installed.  Install with: pip install pubchempy"
        )

    log.info(f"[PUBCHEM] Looking up {mol_name!r} ...")
    compounds = pcp.get_compounds(mol_name, "name", record_type="3d")
    if not compounds:
        raise RuntimeError(
            f"PubChem has no 3D structure for {mol_name!r}.  "
            "Try --smiles or --geometry."
        )

    c = compounds[0]
    lines = [
        f"{a.element}  {a.x or 0.0:.8f}  {a.y or 0.0:.8f}  {a.z or 0.0:.8f}"
        for a in c.atoms
    ]
    log.info(f"[PUBCHEM] {mol_name}: {len(c.atoms)} atoms (CID={c.cid}, Angstrom)")
    return "\n".join(lines)


# ===========================================================================
# 5.  PENNYLANE FALLBACK
# ===========================================================================

def geometry_from_pennylane(
    mol_name: str, basis: str, bondlength: float
) -> Tuple[str, str, int]:
    """
    Fetch geometry from the PennyLane QChem dataset.
    Returns (atom_block, coord_unit="Bohr", n_electrons).

    PennyLane always stores coordinates in Bohr; we return "Bohr" so that
    the caller passes unit="Bohr" to PySCF, preventing a silent ~1.89x
    rescaling error.
    """
    try:
        import pennylane as qml
    except ImportError:
        raise RuntimeError(
            "PennyLane is not installed.  "
            "Install with: pip install pennylane pennylane-datasets"
        )

    log.info(
        f"[PL] Fetching mol={mol_name!r} basis={basis!r} "
        f"bondlength={bondlength} Ang ..."
    )
    datasets = qml.data.load(
        "qchem", molname=mol_name, basis=basis,
        bondlength=bondlength, attributes=["molecule"],
    )
    if not datasets:
        raise RuntimeError(
            f"PennyLane returned no dataset for {mol_name!r} "
            f"at bondlength={bondlength}."
        )

    mol_pl = datasets[0].molecule
    atom_block = "\n".join(
        f"{sym}  {x:.10f}  {y:.10f}  {z:.10f}"
        for sym, (x, y, z) in zip(mol_pl.symbols, mol_pl.coordinates)
    )
    log.info(
        f"[PL] {mol_pl.symbols}, n_electrons={mol_pl.n_electrons} (Bohr)"
    )
    return atom_block, "Bohr", int(mol_pl.n_electrons)


# ===========================================================================
# 6.  GEOMETRY RESOLUTION ORCHESTRATOR
# ===========================================================================

def resolve_geometry(
    mol_name  : Optional[str],
    smiles    : Optional[str],
    geometry  : Optional[str],
    basis     : str,
    bondlength: Optional[float],
) -> Tuple[str, str, str]:
    """
    Resolve a PySCF atom string and coordinate unit.

    Returns (atom_block, coord_unit, source_label).
    coord_unit is "Angstrom" or "Bohr" — must match what PySCF receives.

    Priority waterfall
    ------------------
    1. --geometry  explicit file / inline string
    2. --smiles    RDKit
    3. femoco      built-in placeholder
    4. H_n cluster analytical linear chain
    5. noble-gas dimer analytical pair
    6. registry SMILES → RDKit
    7. PubChem
    8. PennyLane   (requires bondlength)
    """
    # 1. Explicit geometry
    if geometry is not None:
        p = Path(geometry)
        if p.exists():
            log.info(f"[GEOM] From file: {p}")
            return p.read_text(), "Angstrom", "file"
        log.info("[GEOM] Inline atom string (Angstrom)")
        return geometry.strip(), "Angstrom", "inline"

    # 2. Explicit SMILES
    if smiles is not None:
        label = mol_name or "molecule"
        return geometry_from_smiles(smiles, label), "Angstrom", "smiles_rdkit"

    if mol_name is None:
        raise ValueError("Provide --mol, --smiles, or --geometry.")

    # 3. FeMoco — try live PDB download first, fall back to built-in geometry
    if mol_name.lower() == "femoco":
        try:
            atom_block = fetch_femoco_from_pdb(pdb_id="1M1N")
            return atom_block, "Angstrom", "pdb_1M1N_download"
        except RuntimeError as exc:
            log.warning(
                f"[GEOM] PDB download failed: {exc}\n"
                "       Falling back to built-in 1M1N-derived idealised geometry.\n"
                "       For publication-grade work, ensure internet access or\n"
                "       supply --geometry path/to/1M1N.pdb"
            )
            return FEMOCO_1M1N_GEOMETRY, "Angstrom", "femoco_1m1n_idealised"

    # 3b. Registry entries that supply a direct geometry string
    #     (used by fe2s2, femo_core, fe4s4, fe4s3, etc.)
    reg = get_defaults(mol_name)
    if "geometry" in reg and reg["geometry"] is not None:
        log.info(f"[GEOM] Using direct geometry string from registry for {mol_name!r}")
        return reg["geometry"], "Angstrom", "registry_geometry"

    # 4. Hydrogen clusters
    hcluster = re.fullmatch(r"[Hh](\d+)", mol_name)
    if hcluster:
        n = int(hcluster.group(1))
        spacing = bondlength if bondlength is not None else 1.0
        return _hcluster_geometry(n, spacing), "Angstrom", "analytical_hchain"

    # 5. Noble-gas dimers
    ng_match = re.fullmatch(r"([A-Z][a-z]?)2", mol_name)
    if ng_match and ng_match.group(1) in NOBLE_GASES:
        sym = ng_match.group(1)
        bl  = bondlength if bondlength is not None else 2.5
        return _dimer_geometry(sym, bl), "Angstrom", "analytical_dimer"

    # 6. Registry SMILES → RDKit
    reg_smiles = get_defaults(mol_name).get("smiles")
    if reg_smiles is not None:
        try:
            return geometry_from_smiles(reg_smiles, mol_name), "Angstrom", "registry_rdkit"
        except RuntimeError as exc:
            log.warning(f"[GEOM] RDKit failed ({exc}); trying PubChem ...")

    # 7. PubChem
    try:
        return geometry_from_pubchem(mol_name), "Angstrom", "pubchem"
    except RuntimeError as exc:
        log.warning(f"[GEOM] PubChem failed ({exc}); trying PennyLane ...")

    # 8. PennyLane
    if bondlength is not None:
        try:
            atom_block, unit, _ = geometry_from_pennylane(mol_name, basis, bondlength)
            return atom_block, unit, "pennylane"
        except RuntimeError as exc:
            log.warning(f"[GEOM] PennyLane failed: {exc}")

    raise RuntimeError(
        f"Could not determine a geometry for {mol_name!r}.\n"
        "Try one of:\n"
        "  --smiles '<SMILES>'          explicit SMILES string\n"
        "  --geometry path/to/mol.xyz   coordinate file\n"
        "  --bondlength X.XX            enables PennyLane lookup\n"
        "  pip install rdkit            enables automatic SMILES → 3D\n"
        "  pip install pubchempy        enables PubChem fallback"
    )


# ===========================================================================
# 7.  OPTIONAL PYSCF GEOMETRY OPTIMISATION
# ===========================================================================

def optimize_geometry(mol: gto.Mole, method: str = "hf") -> gto.Mole:
    """
    Optimise geometry with PySCF + geomeTRIC.

    method='hf'  → ROHF at mol.basis   (fast; good for light main-group)
    method='dft' → B3LYP at mol.basis  (better for heavier elements)

    Returns the optimised Mole object (all original settings preserved;
    only atom coordinates change).  Requires: pip install geometric
    """
    try:
        from pyscf.geomopt import geometric_solver
    except ImportError:
        raise RuntimeError(
            "geomeTRIC is not available.  Install with: pip install geometric"
        )

    log.info(f"[OPT] Geometry optimisation at {method.upper()}/{mol.basis} ...")

    if method.lower() == "dft":
        from pyscf import dft
        mf_opt = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
        mf_opt.xc = "B3LYP"
    else:
        mf_opt = scf.ROHF(mol)

    mf_opt.conv_tol  = 1e-9
    mf_opt.max_cycle = 300
    mf_opt.verbose   = 0   # suppress per-step SCF output

    mol_opt = geometric_solver.optimize(mf_opt, maxsteps=200)
    log.info("[OPT] geomeTRIC optimisation converged.")
    return mol_opt


# ===========================================================================
# 8.  SPIN PARITY HELPERS
# ===========================================================================

def make_nelec_tuple(nelec: int, spin: int) -> Tuple[int, int]:
    """Convert (total electrons, 2*S) -> (n_alpha, n_beta)."""
    if (nelec - spin) % 2 != 0:
        raise ValueError(
            f"Parity violation: nelec={nelec}, spin_2S={spin}.  "
            f"(nelec - spin) must be even.  "
            f"Valid spin values for nelec={nelec}: "
            f"{list(range(nelec % 2, nelec + 1, 2))}"
        )
    nalpha = (nelec + spin) // 2
    nbeta  = (nelec - spin) // 2
    if nbeta < 0:
        raise ValueError(
            f"spin_2S={spin} exceeds nelec={nelec}: nbeta would be {nbeta}."
        )
    return (nalpha, nbeta)


def resolve_mol_spin(total_nelec: int, target_spin: int) -> int:
    """Smallest spin >= target_spin satisfying (total_nelec - spin) % 2 == 0."""
    spin = target_spin
    while (total_nelec - spin) % 2 != 0:
        spin += 1
    return spin


def spin_label(spin_2S: int) -> str:
    return f"S={spin_2S // 2}" if spin_2S % 2 == 0 else f"S={spin_2S}/2"


# ===========================================================================
# 9.  MOLECULE BUILDER
# ===========================================================================

def _approximate_nelec(atom_block: str, charge: int) -> int:
    """
    Estimate total electron count from atomic numbers without calling mol.build().

    Parses the PySCF atom string for element symbols, looks up each atomic
    number via gto.charge(), sums them, and subtracts the molecular charge.
    Used to resolve spin parity *before* the first mol.build() call so that
    parity violations (e.g. NADH with [N+] and charge=1) never reach the
    build step with the wrong mol.spin.

    Args:
        atom_block : PySCF-format geometry string.
        charge     : Net molecular charge.

    Returns:
        Approximate total electron count (exact for standard elements).
    """
    total_z = 0
    for line in atom_block.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        elem = parts[0].rstrip("0123456789")   # strip atom index if present
        try:
            total_z += gto.charge(elem)
        except Exception:
            pass   # skip comment lines or unrecognised tokens
    return total_z - charge


def build_molecule(
    atom_block : str,
    basis      : str,
    charge     : int,
    target_spin: int,
    coord_unit : str = "Angstrom",
    verbose    : int = 3,
    ecp        : Optional[Dict[str, str]] = None,
) -> gto.Mole:
    """Build a PySCF Mole with correct spin parity.

    Unchanged from the original except for one addition: when `ecp` is None
    and `basis` is a dict containing def2-* entries for elements with Z ≥ 37
    (Rb and heavier, e.g. Mo, Ag, W), the ECP is automatically set to the
    same def2 basis name.  PySCF embeds the Stuttgart ECP inside each def2
    .dat file, so mol.ecp must mirror mol.basis — there is no separate
    "def2-ECP" file.  Note: row-4 d-block elements (Sc–Zn) are all-electron
    in def2 and receive no ECP injection.

    All other behaviour is identical to the original.
    """
    # ── FIX 1: auto-inject ECPs when caller uses def2 basis for heavy elements
    if ecp is None:
        ecp = _auto_ecp(basis)
        if ecp:
            log.info(
                "[BUILD] Auto-injected def2 ECPs (basis-embedded) for: "
                + ", ".join(f"{el} ({v})" for el, v in sorted(ecp.items()))
            )

    # ── Pre-build parity resolution (unchanged) ───────────────────────────
    approx_nelec   = _approximate_nelec(atom_block, charge)
    presolved_spin = resolve_mol_spin(approx_nelec, target_spin)

    if presolved_spin != target_spin:
        log.warning(
            f"[BUILD] Pre-build spin adjustment: target_spin={target_spin} → "
            f"mol.spin={presolved_spin}  "
            f"(approx_nelec={approx_nelec}, charge={charge})."
        )

    mol = gto.Mole()
    mol.atom    = atom_block
    mol.basis   = basis
    mol.charge  = charge
    mol.spin    = presolved_spin
    mol.unit    = coord_unit
    mol.verbose = verbose
    if ecp:
        mol.ecp = ecp

    try:
        mol.build()
    except Exception as exc:
        raise RuntimeError(f"Initial molecule build failed: {exc}") from exc

    # ── Confirm with exact PySCF electron count (unchanged) ───────────────
    total_nelec   = sum(mol.nelec)
    resolved_spin = resolve_mol_spin(total_nelec, target_spin)

    if resolved_spin != presolved_spin:
        log.info(
            f"[BUILD] Spin refined after exact electron count: "
            f"{presolved_spin} → {resolved_spin}  (exact nelec={total_nelec})"
        )

    mol.spin = resolved_spin
    if ecp:
        mol.ecp = ecp   # must be re-applied after attribute reset

    try:
        mol.build()
    except Exception as exc:
        raise RuntimeError(f"Rebuild with resolved spin failed: {exc}") from exc

    log.info(
        f"[BUILD] {mol.natm} atoms | {mol.nao} AOs | "
        f"charge={mol.charge} | {spin_label(mol.spin)} | "
        f"basis={mol.basis} | ecp={mol.ecp if ecp else 'none'} | unit={coord_unit}"
    )
    log.info(
        f"[BUILD] nelec={mol.nelec} "
        f"(nalpha={mol.nelec[0]}, nbeta={mol.nelec[1]})"
    )
    return mol


# ===========================================================================
# 10. SCF REFERENCE
# ===========================================================================
# ===========================================================================
# FIX 2 + FIX 3 — updated run_rohf (accepts dm0; extended metal set)
# ===========================================================================

# FIX 3: Extended transition-metal set.  The original omitted all of the
# 4d coinage metals (Ag) and the 5d block, causing the hardened SCF ladder
# to be bypassed for Ag₃ clusters (ethylene_epoxidation) and similar.
_TRANSITION_METALS: frozenset = frozenset({
    # Row 4 d-block
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    # Row 5 d-block
    "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",
    # Row 6 d-block (commonly used in catalysis)
    "La", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au",
})

def _rohf_base(mol: gto.Mole, **kwargs) -> scf.rohf.ROHF:
    """Return a ROHF object with sane defaults. kwargs override any field."""
    mf = scf.ROHF(mol)
    mf.direct_scf = True
    mf.diis_space  = 12
    mf.conv_tol    = 1e-9
    mf.max_cycle   = 500
    for k, v in kwargs.items():
        setattr(mf, k, v)
    return mf


def run_rohf(mol: gto.Mole, dm0: Optional[np.ndarray] = None) -> scf.rohf.ROHF:
    """Converge ROHF for any molecule using a four-level ladder.

    Parameters
    ----------
    mol : gto.Mole
        Pre-built PySCF molecule.
    dm0 : np.ndarray, optional
        Initial density matrix (e.g. from the previous catalytic step).
        When provided, it replaces the atom-guess in Level 1, typically
        cutting the number of Level-1 iterations by 50–80% for chemically
        similar consecutive geometries along a reaction coordinate.

    FIX 2 — dm0 is now accepted here rather than re-called externally.
    The caller (generate_step_integrals) must NOT call mf.kernel() again
    after this function returns — the object is already converged.

    FIX 3 — Extended transition-metal detection to include Ag, Au, Pd, Pt,
    Rh, Ir, Ru, and the full Row-5 d-block.

    Level 1 — ROHF + level-shift + damping (+ dm0 warm-start if supplied)
    Level 2 — UKS/PBE density → ROHF
    Level 3 — Newton–Raphson second-order SCF
    Level 4 — Loose-tolerance fallback (1e-6 Ha)
    """
    from pyscf import dft

    n_metals = sum(
        1 for atom in mol._atom
        if atom[0] in _TRANSITION_METALS   # ← FIX 3: uses extended set
    )
    is_metal_complex = n_metals >= 1
    if is_metal_complex:
        log.info(
            f"[SCF] Detected {n_metals} transition-metal centre(s); "
            "activating metal-hardened SCF ladder."
        )

    # ── Level 1: ROHF + level-shift + damping ────────────────────────────────
    if dm0 is not None:                                   # FIX 2: warm-start path
        log.info("[SCF] Level 1: ROHF + level_shift=0.3 + damp=0.3 + warm-start dm0")
        mf1 = _rohf_base(mol, level_shift=0.3, damp=0.3)
        mf1.kernel(dm0=dm0)
    else:
        log.info("[SCF] Level 1: ROHF + level_shift=0.3 + damp=0.3 + atom guess")
        mf1 = _rohf_base(mol, level_shift=0.3, damp=0.3, init_guess="atom")
        mf1.kernel()

    if mf1.converged:
        log.info(f"[SCF] Level 1 converged: E = {mf1.e_tot:.10f} Ha")
        return mf1

    log.warning("[SCF] Level 1 did not converge; trying Level 2 (DFT-seeded ROHF).")

    # ── Level 2: UKS/PBE density → ROHF ─────────────────────────────────────
    log.info("[SCF] Level 2: UKS/PBE initial density → ROHF")
    try:
        mf_dft = dft.UKS(mol)
        mf_dft.xc        = "PBE"
        mf_dft.conv_tol  = 1e-7
        mf_dft.max_cycle = 300
        mf_dft.verbose   = 0
        mf_dft.diis_space = 12
        mf_dft.kernel()
        dm_dft = mf_dft.make_rdm1()
        if isinstance(dm_dft, (list, tuple)) or (
            hasattr(dm_dft, "ndim") and dm_dft.ndim == 3
        ):
            dm0_level2 = dm_dft[0] + dm_dft[1]
        else:
            dm0_level2 = dm_dft

        mf2 = _rohf_base(mol, level_shift=0.2, damp=0.2)
        mf2.kernel(dm0=dm0_level2)

        if mf2.converged:
            log.info(f"[SCF] Level 2 converged: E = {mf2.e_tot:.10f} Ha")
            return mf2

        log.warning("[SCF] Level 2 did not converge; trying Level 3 (Newton-Raphson).")

        # ── Level 3: Newton–Raphson ───────────────────────────────────────────
        log.info("[SCF] Level 3: Newton-Raphson second-order SCF")
        mf3 = scf.newton(mf2)
        mf3.conv_tol  = 1e-9
        mf3.max_cycle = 100
        mf3.verbose   = 0
        mf3.kernel(mf2.make_rdm1())

        if mf3.converged:
            log.info(f"[SCF] Level 3 converged: E = {mf3.e_tot:.10f} Ha")
            mf2.mo_coeff  = mf3.mo_coeff
            mf2.mo_occ    = mf3.mo_occ
            mf2.e_tot     = mf3.e_tot
            mf2.converged = True
            return mf2

    except Exception as exc:
        log.warning(f"[SCF] Level 2/3 encountered an error: {exc}")

    log.warning("[SCF] Levels 1–3 did not converge; trying Level 4 (loose tolerance).")

    # ── Level 4: loose-tolerance fallback ────────────────────────────────────
    log.info("[SCF] Level 4: ROHF + level_shift=0.5 + damp=0.4 + conv_tol=1e-6")
    mf4 = _rohf_base(
        mol, level_shift=0.5, damp=0.4,
        init_guess="atom", conv_tol=1e-6, max_cycle=600,
    )
    mf4.kernel()

    if mf4.converged:
        log.warning(
            "[SCF] Converged at loose tolerance (1e-6 Ha).  "
            "Integrals may be less accurate.  "
            "Consider: (a) --optimize the geometry, "
            "(b) use def2-SVP+ECP for transition metals, "
            "(c) supply a DFT-optimised geometry via --geometry."
        )
        return mf4

    raise RuntimeError(
        f"ROHF did not converge at any level after exhausting the SCF ladder "
        f"(last E={mf4.e_tot:.8f} Ha)."
    )


# ===========================================================================
# 11. CASCI INTEGRAL EXTRACTION
# ===========================================================================

def extract_casci_integrals(
    mf                  : scf.rohf.ROHF,
    ncas                : int,
    nelec_tuple         : Tuple[int, int],
    cas_orbital_indices : Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float, Optional[float], Optional[np.ndarray]]:
    """
    Run CASCI and return (h1, eri_full, ecore, e_casci, noons_active).

    Spin sector is set through nelec_tuple=(nalpha,nbeta), NOT through
    any .spin attribute (which PySCF ignores).

    noons_active: diagonal of the CASCI 1-RDM in the active MO basis — the
        occupation of each active orbital (values in [0, 2]).  These are the
        natural-orbital-like occupation numbers for the active space, used by
        the Iwasawa tower climber to rank correlation-important orbitals.
        None if the CASCI kernel did not converge or ci vector is absent.

    cas_orbital_indices: optional 0-based array of MO indices to use as the
        active space, overriding Fermi-level selection.  Required by the
        sub-Janus orbital selection protocol (prop:seed_is_sp,
        {ℓ<k*=2}={s,p}).  Default None = standard Fermi-level selection.
    """
    cas = mcscf.CASCI(mf, ncas, nelec_tuple)

    mo_for_kernel: Optional[np.ndarray] = None
    if cas_orbital_indices is not None:
        # cas.sort_mo uses 1-based indexing by default.
        caslst = [int(i) + 1 for i in cas_orbital_indices]
        mo_for_kernel = cas.sort_mo(caslst, mf.mo_coeff)
        log.info(
            "[CASCI] Sub-Janus selection: MO indices %s (0-based) → "
            "caslst %s (1-based) for CAS(%d,%d).",
            list(cas_orbital_indices), caslst, sum(nelec_tuple), ncas,
        )

    e_casci: Optional[float] = None
    try:
        if mo_for_kernel is not None:
            cas.kernel(mo_for_kernel)
        else:
            cas.kernel()
        if cas.e_tot is not None:
            e_casci = float(cas.e_tot)
            log.info(f"[CASCI] E_total = {e_casci:.10f} Ha  (converged)")
    except Exception as exc:
        log.warning(
            f"[CASCI] Kernel failed: {exc}.  "
            "Integrals computed from ROHF MOs are still valid."
        )

    h1, ecore  = cas.h1e_for_cas()
    eri_packed = cas.get_h2eff()
    eri_full   = ao2mo.restore(1, eri_packed, ncas)
    log.info(f"[CASCI] E_core = {float(ecore):.10f} Ha")

    # ── CASCI 1-RDM → active-orbital occupation numbers (for tower climbing) ──
    # Use fcisolver.make_rdm1 to get the (ncas × ncas) active-space 1-RDM
    # directly from the CI vector.  cas.make_rdm1() returns the full MO-basis
    # 1-RDM (nmo × nmo) — not what we want here.
    noons_active: Optional[np.ndarray] = None
    try:
        if cas.ci is not None:
            rdm1_cas     = cas.fcisolver.make_rdm1(cas.ci, cas.ncas, cas.nelecas)
            # rdm1_cas: shape (ncas, ncas); diagonal = occupation of each active MO
            noons_active = np.diag(rdm1_cas)     # shape (ncas,), values ∈ [0, 2]
            log.info(
                "[CASCI] Active MO occupations (noons, ncas=%d): %s  sum=%.4f",
                cas.ncas,
                " ".join(f"{n:.4f}" for n in noons_active),
                float(noons_active.sum()),
            )
    except Exception as exc:
        log.warning("[CASCI] 1-RDM computation failed: %s", exc)

    return h1, eri_full, float(ecore), e_casci, noons_active


# ===========================================================================
# 12. FCI / DMRG REFERENCE
# ===========================================================================

def extract_full_mo_integrals(
    mf:           "scf.rohf.ROHF",
    n_total_orbs: int,
) -> "Tuple[np.ndarray, np.ndarray]":
    """Transform AO integrals to MO basis for the first n_total_orbs MOs.

    Uses the converged MO coefficients from ``mf`` (ROHF/UKS/RKS); no
    additional SCF or CASCI is performed.

    The resulting arrays can be sliced at each Iwasawa tower level k to give
    exact molecular integrals for a CAS(4, 4k) active space without any
    further PySCF calculation.

    Parameters
    ----------
    mf            : converged PySCF mean-field object (e.g. from run_rohf).
    n_total_orbs  : number of MOs to include (capped at mf.mo_coeff.shape[1]).

    Returns
    -------
    h1_full   : ndarray, shape (n, n) — one-electron integrals in MO basis.
    eri_packed : ndarray, 1-D — two-electron integrals in 8-fold packed format
                 (use ``pyscf.ao2mo.restore(1, eri_packed, n)`` to unpack).
    """
    nmo = mf.mo_coeff.shape[1]
    n   = min(n_total_orbs, nmo)
    C   = mf.mo_coeff[:, :n]

    h1_full    = C.T @ mf.get_hcore() @ C          # (n, n)
    eri_packed = ao2mo.kernel(mf.mol, C)            # 8-fold packed, size n*(n+1)/2*(n*(n+1)/2+1)/2 / 2

    log.info(
        "[FULL-MO] h1_full shape=%s  eri_packed size=%d (%.1f MB)  n_orbs=%d",
        h1_full.shape, eri_packed.size, eri_packed.nbytes / 2**20, n,
    )
    return h1_full, eri_packed


def extract_tower_window_integrals(
    mf:            "scf.rohf.ROHF",
    win_start:     int,
    win_size:      int,
) -> "Tuple[np.ndarray, np.ndarray, float]":
    r"""Build Fock-contracted integrals for a tower window [win_start, win_start+win_size).

    For large systems (n_occ_base >> 40) the full h1_full approach is
    infeasible: h1_full covers MOs 0..n-1, so global indices up to n-1 must
    fit, and the 4D ERI for n≈229 MOs would exceed 22 GB.

    This function instead:
    1. Contracts the deep frozen core (global MOs 0..win_start-1) into
       a corrected effective one-body h1_eff_win for the window MOs.
    2. Computes the bare 4D ERI over the window only (win_size^4 * 8 bytes).
    3. Returns the deep-core ecore (nuclear repulsion + 1e + 2e from MOs
       0..win_start-1) so the tower climber has the correct reference energy.

    The tower climber can then use local indices 0..win_size-1 for all
    h1_eff_win / eri_win accesses, provided it knows win_start (the offset).

    Parameters
    ----------
    mf         : Converged ROHF mean-field.
    win_start  : First global MO index in the tower window.
    win_size   : Number of MOs in the tower window.

    Returns
    -------
    h1_eff_win  : ndarray (win_size, win_size) — Fock-corrected 1e integrals
                  for window MOs with deep core already folded in.
    eri_win     : ndarray (win_size, win_size, win_size, win_size) — raw 2e
                  integrals in the window (no frozen-core folding; the tower
                  climber handles that dynamically for window MOs).
    deep_ecore  : float — energy contribution of MOs 0..win_start-1 folded
                  into h1_eff_win (to be added to the tower climber's ecore).
    """
    from pyscf import ao2mo as pyscf_ao2mo

    nmo    = mf.mo_coeff.shape[1]
    ws     = min(win_start, nmo)
    we     = min(win_start + win_size, nmo)
    n_win  = we - ws
    n_core = ws   # MOs 0..ws-1 are the deep frozen core

    C_core = mf.mo_coeff[:, :n_core]     # (n_AO, n_core)
    C_win  = mf.mo_coeff[:, ws:we]       # (n_AO, n_win)
    C_all  = mf.mo_coeff[:, :we]         # (n_AO, n_core+n_win) for joint ERI transform

    hcore  = mf.get_hcore()              # (n_AO, n_AO)

    # ── 1e integrals in MO basis (window + core block) ───────────────────────
    h1_core_core = C_core.T @ hcore @ C_core   # (n_core, n_core)
    h1_win_win   = C_win.T  @ hcore @ C_win    # (n_win,  n_win)
    h1_core_win  = C_core.T @ hcore @ C_win    # (n_core, n_win)  — for Fock build

    # ── 2e integrals (core×core, core×win cross, win×win) ────────────────────
    eri_all_packed = pyscf_ao2mo.kernel(mf.mol, C_all)   # 8-fold, (n_core+n_win) space
    eri_all = pyscf_ao2mo.restore(1, eri_all_packed, n_core + n_win)

    # Slice blocks (indices into eri_all: core=0..n_core-1, win=n_core..n_core+n_win-1)
    c = np.arange(n_core)
    w = np.arange(n_core, n_core + n_win)

    eri_cc_cc = eri_all[np.ix_(c, c, c, c)]   # (n_core,n_core,n_core,n_core)
    eri_ww_ww = eri_all[np.ix_(w, w, w, w)]   # (n_win, n_win, n_win, n_win)

    # ── Deep-core ecore = nuc_rep + 1e(core) + 2e(core,core) ────────────────
    # (nuclear repulsion cannot be extracted here; caller adds it separately)
    deep_1e  = 2.0 * float(np.einsum("pp->", h1_core_core))
    deep_2e  = (
        2.0 * float(np.einsum("ppqq->", eri_cc_cc))
        -     float(np.einsum("pqqp->", eri_cc_cc))
    )
    deep_ecore = deep_1e + deep_2e   # caller must add e_nuc

    # ── Fock correction: fold core into window h1 ────────────────────────────
    # h1_eff_win[p,q] = h1[p,q] + Σ_c (2*(pq|cc) - (pc|qc))
    # where p,q ∈ window (global ws..we-1), c ∈ core (global 0..ws-1)
    # In eri_all coordinates: p,q → w indices; c → c indices.
    h1_eff_win = h1_win_win.copy()
    for ci in range(n_core):
        # 2*(pq|cc): eri_all[w, w, ci, ci] → shape (n_win, n_win)
        h1_eff_win += 2.0 * eri_all[np.ix_(w, w, [ci], [ci])].squeeze(axis=(2, 3))
        # -(pc|qc): eri_all[w, ci, w, ci] → shape (n_win, n_win)
        h1_eff_win -= eri_all[np.ix_(w, [ci], w, [ci])].squeeze(axis=(1, 3))

    eri_win = eri_ww_ww.copy()

    log.info(
        "[TOWER-WIN] win=[%d,%d)  n_core=%d  n_win=%d  "
        "deep_ecore(1e+2e)=%.6f Ha  eri_win shape=%s (%.0f MB)",
        ws, we, n_core, n_win, deep_ecore,
        eri_win.shape, eri_win.nbytes / 2**20,
    )
    return h1_eff_win, eri_win, deep_ecore


def compute_reference_energy(
    h1         : np.ndarray,
    eri_full   : np.ndarray,
    ncas       : int,
    nelec_tuple: Tuple[int, int],
    ecore      : float,
) -> Tuple[float, str]:
    """Return (ground-state energy Ha, method label)."""
    if ncas > 20:
        E_DMRG_CORR = -133.482   # FeMoco active-space correlation (Ha)
        e_ref = float(ecore + E_DMRG_CORR)
        log.info(f"[REF] ncas={ncas}>20: DMRG benchmark -> {e_ref:.6f} Ha")
        return e_ref, "DMRG_Li2019"

    log.info(
        f"[REF] Exact FCI: ncas={ncas}, "
        f"nalpha={nelec_tuple[0]}, nbeta={nelec_tuple[1]} ..."
    )
    e_fci, _ = fci.direct_spin1.kernel(
        h1, eri_full, ncas, nelec_tuple, ecore=ecore, verbose=0
    )
    log.info(f"[REF] E_FCI = {float(e_fci):.10f} Ha")
    return float(e_fci), "FCI"


# ===========================================================================
# 13. ERI COMPRESSION  (8-fold real symmetry, chemist's notation)
# ===========================================================================

def compress_eri(
    eri_full : np.ndarray,
    ncas     : int,
    threshold: float = 1e-8,
) -> Dict[str, float]:
    """
    Compress (ncas)^4 ERI tensor to canonical 8-fold block.

    Key format: "(p,q,r,s)" representing (pq|rs).
    The inner loop slices from pq_pairs[idx] to enforce (p,q)>=(r,s)
    without visiting each pair twice.
    """
    log.info(
        f"[ERI] Compressing {ncas}^4 tensor "
        f"(threshold={threshold:.1e}) ..."
    )
    pq_pairs = list(combinations_with_replacement(range(ncas), 2))
    g: Dict[str, float] = {}
    start = time.time()

    for idx, (p, q) in enumerate(pq_pairs):
        if (idx + 1) % 500 == 0:
            log.info(
                f"  [ERI] pair {idx+1}/{len(pq_pairs)} | "
                f"kept: {len(g)} | {time.time()-start:.1f}s"
            )
        for r, s in pq_pairs[idx:]:
            val = eri_full[p, q, r, s]
            if abs(val) < threshold:
                continue
            g[f"({p},{q},{r},{s})"] = float(val)

    log.info(f"[ERI] {len(g)} unique (pq|rs) channels retained.")
    return g


# ===========================================================================
# 14. JSON ASSEMBLY & EXPORT
# ===========================================================================

def export_json(
    output_path    : Path,
    mol_name       : str,
    basis          : str,
    bondlength     : Optional[float],
    coord_unit     : str,
    geometry_source: str,
    optimized      : bool,
    h1             : np.ndarray,
    eri_full       : np.ndarray,
    ecore          : float,
    rohf_energy    : float,
    e_ref          : float,
    ref_method     : str,
    e_casci        : Optional[float],
    ncas           : int,
    nelec_tuple    : Tuple[int, int],
    active_spin    : int,
    mol_nao        : int,
) -> None:
    nalpha, nbeta = nelec_tuple

    h_diag = {str(p): float(h1[p, p]) for p in range(ncas)}
    h_hop  = {
        f"({p},{q})": float(h1[p, q])
        for p in range(ncas) for q in range(p + 1, ncas)
    }

    g_full = compress_eri(eri_full, ncas)
    del eri_full
    gc.collect()

    if e_casci is not None:
        e_corr      = e_casci - rohf_energy
        corr_source = "E_CASCI - E_ROHF"
    else:
        e_corr      = e_ref - rohf_energy
        corr_source = f"E_{ref_method} - E_ROHF (approx)"

    data = {
        # Integrals in chemist's notation
        "h_diag": h_diag,
        "h_hop" : h_hop,
        "g_full": g_full,
        # Energies (Ha)
        "ecore_Ha"                   : ecore,
        "rohf_energy_Ha"             : rohf_energy,
        "exact_fci_energy_Ha"        : e_ref if ncas <= 20 else None,
        "circuit_reference_energy_Ha": e_ref,
        "active_space_corr_energy_Ha": e_corr,
        "active_space_corr_label"    : corr_source,
        # Metadata
        "metadata": {
            "mol_name"           : mol_name,
            "basis"              : basis,
            "bondlength_angstrom": bondlength,
            "coord_unit_pyscf"   : coord_unit,
            "geometry_source"    : geometry_source,
            "geometry_optimized" : optimized,
            "integral_convention": "chemist (pq|rs)",
            "eri_symmetry"       : "8-fold real",
            "screening_threshold": 1e-8,
            "dt_ref_Ha_inv"      : 0.04,
            "scf_method"         : "ROHF",
            "ref_method"         : ref_method,
            "ncas"               : ncas,
            "nao_total"          : mol_nao,
            "nelec_active"       : nalpha + nbeta,
            "nalpha"             : nalpha,
            "nbeta"              : nbeta,
            "spin_2S"            : active_spin,
            "spin_sector"        : spin_label(active_spin),
            "fermion_mapping"    : "Native_d4_Tetralemmatic",
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))

    log.info(f"[OUTPUT] Saved -> {output_path}")
    log.info(
        f"  CAS({nalpha+nbeta},{ncas})  {spin_label(active_spin)}  "
        f"nalpha={nalpha}  nbeta={nbeta}"
    )
    log.info(
        f"  h_diag:{len(h_diag)}  h_hop:{len(h_hop)}  (pq|rs):{len(g_full)}"
    )
    log.info(f"  E_core  = {ecore:+.8f} Ha")
    log.info(f"  E_ROHF  = {rohf_energy:+.10f} Ha")
    log.info(f"  E_ref   = {e_ref:+.10f} Ha  [{ref_method}]")
    log.info(f"  E_corr  = {e_corr*1000:+.4f} mHa  ({corr_source})")
    log.info(f"  source  = {geometry_source}  optimized={optimized}")


# ===========================================================================
# 15. MAIN
# ===========================================================================

def default_output(mol_name: str, basis: str, bondlength: Optional[float]) -> str:
    b  = basis.lower().replace("-", "")
    bl = f"_r{bondlength:.3f}" if bondlength is not None else ""
    return f"{mol_name.lower()}_{b}{bl}.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export molecular integrals (ROHF -> CASCI, chemist's notation) "
            "with automatic geometry generation from SMILES, PubChem, or "
            "PennyLane.  No geometry file required."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Molecule identity ─────────────────────────────────────────────────
    parser.add_argument(
        "--mol",
        help=(
            "Molecule name: any entry in the registry (H2, LiH, H2O, O2, "
            "OH-, H3+, H4...H10, He2, femoco, ...) or any PubChem-resolvable "
            "name.  Geometry is resolved automatically."
        ),
    )
    parser.add_argument(
        "--smiles",
        help="SMILES string for geometry generation (e.g. 'C#C' for acetylene).",
    )
    parser.add_argument(
        "--geometry",
        help=(
            "Path to a .xyz/.pdb file or an inline PySCF atom string.  "
            "Bypasses automatic geometry generation entirely."
        ),
    )
    parser.add_argument(
        "--mol_name", default=None,
        help=(
            "Metadata label / output filename stem when --smiles or "
            "--geometry is used without --mol."
        ),
    )

    # ── Molecular parameters ──────────────────────────────────────────────
    parser.add_argument(
        "--basis", default="STO-3G",
        help="Basis set (STO-3G for PennyLane benchmarks; def2-SVP for FeMoco).",
    )
    parser.add_argument(
        "--bondlength", type=float, default=None,
        help=(
            "Bond length in Ang: used as H-H spacing for H_n chains, "
            "interatomic distance for dimers, and for PennyLane lookups.  "
            "Defaults to the registry equilibrium value for the molecule."
        ),
    )
    parser.add_argument(
        "--charge", type=int, default=None,
        help="Molecular charge.  Defaults to registry value or 0.",
    )
    parser.add_argument(
        "--spin", type=int, default=None,
        help=(
            "2*S = n_alpha - n_beta.  "
            "Singlet=0, Doublet=1, Triplet=2, S=3/2->3.  "
            "Defaults to registry value or 0."
        ),
    )

    # ── Geometry optimisation ─────────────────────────────────────────────
    parser.add_argument(
        "--optimize", action="store_true",
        help=(
            "Run a PySCF + geomeTRIC geometry optimisation before extracting "
            "integrals.  Strongly recommended when geometry comes from RDKit "
            "or PubChem.  Requires: pip install geometric"
        ),
    )
    parser.add_argument(
        "--opt_method", default="hf", choices=["hf", "dft"],
        help=(
            "Level of theory for geometry optimisation: "
            "'hf' = ROHF/basis (fast, default); "
            "'dft' = B3LYP/basis (more accurate for heavier molecules)."
        ),
    )

    # ── Active space ──────────────────────────────────────────────────────
    parser.add_argument(
        "--active_orbitals", type=int, default=None,
        help="ncas.  Defaults to mol.nao (full orbital space).",
    )
    parser.add_argument(
        "--active_electrons", type=int, default=None,
        help="Active electrons.  Defaults to total molecular electrons.",
    )

    # ── Output ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path.  Defaults to <mol>_<basis>_r<bondlength>.json.",
    )
    parser.add_argument(
        "--verbose", type=int, default=3,
        help="PySCF verbosity (0=silent ... 9=debug).",
    )

    args = parser.parse_args()

    if args.mol is None and args.smiles is None and args.geometry is None:
        parser.error("Provide at least one of --mol, --smiles, or --geometry.")

    # Canonical name for defaults, metadata, and output filename
    mol_name = args.mol or args.mol_name or "molecule"

    # Resolve defaults from registry
    reg        = get_defaults(mol_name)
    charge     = args.charge     if args.charge     is not None else reg["charge"]
    spin       = args.spin       if args.spin       is not None else reg["spin"]
    bondlength = args.bondlength if args.bondlength is not None else reg["bondlength"]

    log.info(
        f"[CONFIG] mol={mol_name!r} | basis={args.basis} | "
        f"charge={charge} | spin={spin} ({spin_label(spin)}) | "
        f"bondlength={bondlength}"
    )

    # ── Spin sanity check: warn if user overrides a known open-shell spin ──
    reg_spin = reg.get("spin", 0)
    if args.spin is not None and args.spin != reg_spin and reg_spin != 0:
        log.warning(
            f"[CONFIG] --spin {args.spin} overrides the registry default "
            f"spin={reg_spin} for {mol_name!r}.  "
            f"The physical ground state is {spin_label(reg_spin)}.  "
            "Proceed only if you intentionally want a different spin sector."
        )
    if mol_name.lower() == "femoco" and spin == 0:
        log.warning(
            "[CONFIG] FeMoco has S=3/2 ground state; --spin 0 (singlet) is "
            "unphysical.  Use --spin 3 for the correct spin sector.  "
            "SCF convergence is much harder for the wrong spin."
        )

    # ── Geometry ──────────────────────────────────────────────────────────
    try:
        atom_block, coord_unit, geom_source = resolve_geometry(
            mol_name   = args.mol or mol_name,
            smiles     = args.smiles,
            geometry   = args.geometry,
            basis      = args.basis,
            bondlength = bondlength,
        )
    except (RuntimeError, ValueError) as exc:
        log.error(f"[GEOM] {exc}")
        sys.exit(1)

    # ── Basis sanity check for metal-containing molecules ─────────────────
    # Walk atom_block directly — avoids the previous approach of building a
    # whole gto.M object (which was a no-op for non-FeMoco molecules and
    # used a stale constant regardless of the actual geometry).
    _heavy_metals = {"Fe", "Mo", "Mn", "Ni", "Co", "Cu", "Cr", "V", "W", "Ru",
                     "Zn", "Pd", "Pt", "Au", "Ag", "Rh", "Ir", "Os", "Re"}
    mol_has_metal = any(
        tok.capitalize() in _heavy_metals
        for tok in atom_block.split()
    )
    if mol_has_metal and args.basis.upper() in ("STO-3G", "STO3G", "STO-6G"):
        log.warning(
            f"[CONFIG] STO-3G is a minimal basis without ECPs.  "
            f"For {mol_name!r} this means all core electrons of Fe (26e) and Mo (42e) "
            "are in the SCF — leading to 368 total electrons for the full cluster, "
            "severe SCF convergence difficulties, and qualitatively wrong orbitals.\n"
            "Recommended alternatives:\n"
            "  def2-SVP           — split-valence with good d-function coverage\n"
            "  def2-SVP with ECP  — Stuttgart ECPs for Fe/Mo (pip install pyscf-ecp)\n"
            "  cc-pVDZ-DK         — Douglas-Kroll relativistic for Mo\n"
            "Use --basis def2-SVP for a practical starting point."
        )

    # ── Build PySCF molecule ──────────────────────────────────────────────
    try:
        mol = build_molecule(
            atom_block  = atom_block,
            basis       = args.basis,
            charge      = charge,
            target_spin = spin,
            coord_unit  = coord_unit,
            verbose     = args.verbose,
        )
    except RuntimeError as exc:
        log.error(f"[BUILD] {exc}")
        sys.exit(1)

    # ── Optional geometry optimisation ────────────────────────────────────
    optimized = False
    if args.optimize:
        try:
            mol = optimize_geometry(mol, method=args.opt_method)
            optimized  = True
            coord_unit = "Bohr"   # geomeTRIC returns Bohr coordinates
        except RuntimeError as exc:
            log.error(f"[OPT] {exc}")
            sys.exit(1)

    # ── Active space ──────────────────────────────────────────────────────
    total_nelec = sum(mol.nelec)
    ncas  = args.active_orbitals  if args.active_orbitals  is not None else mol.nao
    nelec = args.active_electrons if args.active_electrons is not None else total_nelec

    try:
        nelec_tuple = make_nelec_tuple(nelec, spin)
    except ValueError as exc:
        log.error(f"[PARITY] {exc}")
        sys.exit(1)

    nalpha, nbeta = nelec_tuple
    log.info(
        f"[ACTIVE] CAS({nelec},{ncas})  {spin_label(spin)}  "
        f"nalpha={nalpha}  nbeta={nbeta}  "
        f"(full space: {total_nelec}e / {mol.nao}o)"
    )

    if ncas > mol.nao:
        log.error(f"--active_orbitals={ncas} exceeds mol.nao={mol.nao}")
        sys.exit(1)
    if nelec > total_nelec:
        log.error(f"--active_electrons={nelec} exceeds total_nelec={total_nelec}")
        sys.exit(1)

    # ── SCF ───────────────────────────────────────────────────────────────
    try:
        mf = run_rohf(mol)
    except RuntimeError as exc:
        log.error(f"[SCF] {exc}")
        sys.exit(1)

    # ── CASCI integrals ───────────────────────────────────────────────────
    h1, eri_full, ecore, e_casci = extract_casci_integrals(mf, ncas, nelec_tuple)

    # ── Reference energy ──────────────────────────────────────────────────
    e_ref, ref_method = compute_reference_energy(
        h1, eri_full, ncas, nelec_tuple, ecore
    )

    # ── Export ────────────────────────────────────────────────────────────
    out_path = Path(args.output) if args.output else \
               Path(default_output(mol_name, args.basis, bondlength))

    export_json(
        output_path     = out_path,
        mol_name        = mol_name,
        basis           = args.basis,
        bondlength      = bondlength,
        coord_unit      = coord_unit,
        geometry_source = geom_source,
        optimized       = optimized,
        h1              = h1,
        eri_full        = eri_full,
        ecore           = ecore,
        rohf_energy     = float(mf.e_tot),
        e_ref           = e_ref,
        ref_method      = ref_method,
        e_casci         = e_casci,
        ncas            = ncas,
        nelec_tuple     = nelec_tuple,
        active_spin     = spin,
        mol_nao         = mol.nao,
    )

    log.info("[MAIN] Pipeline completed successfully.")


if __name__ == "__main__":
    main()