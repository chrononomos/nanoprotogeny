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
mqebasisloader.py — Basis Set Exchange JSON loader
===================================================

Reads basis sets from the local ``basis_set_exchange/`` mirror (downloaded
from https://www.basissetexchange.org/) and returns them in a format
compatible with ``mqeprotogeny._build_basis_shells``.

Public API
----------
.. code-block:: python

    from nanoprotogeny.basis.mqebasisloader import (
        load_bse_basis,
        has_am,
        list_available,
        BSE_DIR,
    )

    # Returns List[GeneralShellEntry] for Fe in def2-TZVP
    shells = load_bse_basis("def2-TZVP", "Fe")

``GeneralShellEntry`` format
----------------------------
Each entry is a 4-tuple::

    (angular_type: str,
     exponents:    Tuple[float, ...],
     coeffs_1:     Tuple[float, ...],
     coeffs_2:     Optional[Tuple[float, ...]])

* ``angular_type`` — ``"S"``, ``"P"``, ``"D"``, ``"F"``, ``"G"``, or
  ``"SP"`` (shared-exponent s+p shell, as in STO-3G for Li–Ar).
* ``coeffs_1`` — s-coefficients (or p-coefficients for pure P-shells,
  d-coefficients for pure D-shells, etc.).  For ``"SP"`` shells this is
  the s-contraction.
* ``coeffs_2`` — p-contraction for ``"SP"`` shells; ``None`` otherwise.

This matches the variable-length generalisation of ``mqebasis.ShellEntry``
(which uses fixed 3-tuples for STO-3G).

Basis set file naming
---------------------
BSE JSON files follow the pattern ``<name>.<revision>.json``.  A single
logical basis set may span several files (e.g. ``def2-tzvp.0.json`` for
main-group elements, ``def2-tzvp.1.json`` for transition metals).  This
module merges all revision files automatically.

The canonical BSE name is normalised to lower-case with ``*``, ``(``, ``)``
and spaces replaced by underscores/hyphens as stored on disk.  Star ``*``
maps to ``_st_`` following BSE convention.
"""

from __future__ import annotations

import glob
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

#: (angular_type, exponents, coeffs_1, coeffs_2_or_None)
#: All tuples have variable length — unlike the fixed-3-tuple ShellEntry in
#: mqebasis.py which is STO-3G-specific.
GeneralShellEntry = Tuple[
    str,
    Tuple[float, ...],
    Tuple[float, ...],
    Optional[Tuple[float, ...]],
]

# ---------------------------------------------------------------------------
# Path to the local BSE mirror
# ---------------------------------------------------------------------------

BSE_DIR: Path = Path(__file__).parent / "basis_set_exchange"

# ---------------------------------------------------------------------------
# Periodic table — atomic number ↔ symbol
# ---------------------------------------------------------------------------

_Z_TO_SYM: Dict[int, str] = {
    1: "H",   2: "He",  3: "Li",  4: "Be",  5: "B",   6: "C",   7: "N",
    8: "O",   9: "F",  10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si",
   15: "P",  16: "S",  17: "Cl", 18: "Ar", 19: "K",  20: "Ca", 21: "Sc",
   22: "Ti", 23: "V",  24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni",
   29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br",
   36: "Kr", 37: "Rb", 38: "Sr", 39: "Y",  40: "Zr", 41: "Nb", 42: "Mo",
   43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In",
   50: "Sn", 51: "Sb", 52: "Te", 53: "I",  54: "Xe", 55: "Cs", 56: "Ba",
   57: "La", 58: "Ce", 59: "Pr", 60: "Nd", 61: "Pm", 62: "Sm", 63: "Eu",
   64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb",
   71: "Lu", 72: "Hf", 73: "Ta", 74: "W",  75: "Re", 76: "Os", 77: "Ir",
   78: "Pt", 79: "Au", 80: "Hg", 81: "Tl", 82: "Pb", 83: "Bi", 84: "Po",
   85: "At", 86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac", 90: "Th", 91: "Pa",
   92: "U",  93: "Np", 94: "Pu", 95: "Am", 96: "Cm", 97: "Bk", 98: "Cf",
   99: "Es",100: "Fm",101: "Md",102: "No",103: "Lr",104: "Rf",105: "Db",
  106: "Sg",107: "Bh",108: "Hs",109: "Mt",110: "Ds",111: "Rg",112: "Cn",
  113: "Nh",114: "Fl",115: "Mc",116: "Lv",117: "Ts",118: "Og",
}

_SYM_TO_Z: Dict[str, int] = {v: k for k, v in _Z_TO_SYM.items()}

# ---------------------------------------------------------------------------
# Angular momentum integer list → shell type string
# ---------------------------------------------------------------------------

_AM_TO_STR: Dict[Tuple[int, ...], str] = {
    (0,):    "S",
    (1,):    "P",
    (2,):    "D",
    (3,):    "F",
    (4,):    "G",
    (5,):    "H",
    (0, 1):  "SP",
    (1, 0):  "SP",  # normalise reverse order
}


def _am_to_str(am_list: List[int]) -> str:
    key = tuple(am_list)
    result = _AM_TO_STR.get(key)
    if result is None:
        raise ValueError(f"[mqebasisloader] Unsupported angular_momentum list: {am_list}")
    return result


# ---------------------------------------------------------------------------
# BSE name → filename stem normalisation
# ---------------------------------------------------------------------------

def _normalise_name(name: str) -> str:
    """Normalise a user-supplied basis name to the BSE file stem convention.

    Examples::

        "def2-TZVP"  → "def2-tzvp"
        "6-31G*"     → "6-31g_st_"
        "6-31G**"    → "6-31g_st__st_"
        "cc-pVTZ"    → "cc-pvtz"
    """
    n = name.lower().strip()
    # Replace ** before * so we don't double-replace
    n = n.replace("**", "_st__st_")
    n = n.replace("*",  "_st_")
    n = n.replace("(", "_").replace(")", "_")
    n = n.replace(" ", "-")
    return n


# ---------------------------------------------------------------------------
# File discovery and element map loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _load_element_map(basis_stem: str) -> Dict[int, dict]:
    """Load and merge all revision files for *basis_stem*.

    Returns a dict mapping atomic number (int) to the BSE element dict
    (which contains ``electron_shells``).

    Raises ``FileNotFoundError`` if no files are found.
    """
    pattern = str(BSE_DIR / f"{basis_stem}.*.json")
    files = sorted(f for f in glob.glob(pattern)
                   if not f.endswith(".ref.bib"))
    if not files:
        available = list_available()
        close = [n for n in available if basis_stem in n]
        hint = f"  Similar names: {close[:8]}" if close else ""
        raise FileNotFoundError(
            f"[mqebasisloader] No BSE files found for basis '{basis_stem}'.\n"
            f"  Searched: {pattern}\n"
            f"  BSE_DIR: {BSE_DIR}\n"
            + hint
        )
    # Files are sorted by revision number (e.g. .0.json before .1.json).
    # Later revisions replace earlier ones for the same element — they are
    # independent complete definitions, not additive fragments.  Accumulating
    # shells across revision files for the same element produces duplicates.
    merged: Dict[int, dict] = {}
    for fpath in files:
        with open(fpath, "r") as fh:
            data = json.load(fh)
        for z_str, el_dict in data.get("elements", {}).items():
            z = int(z_str)
            # Last file wins: the highest revision number is the authoritative
            # definition for this element in this basis set.
            merged[z] = {
                "electron_shells": list(el_dict.get("electron_shells", []))
            }
    return merged


# ---------------------------------------------------------------------------
# Core public function
# ---------------------------------------------------------------------------

def load_bse_basis(basis_name: str, symbol: str) -> List[GeneralShellEntry]:
    """Return the shell list for *symbol* from the named BSE basis set.

    Parameters
    ----------
    basis_name:
        Human-readable basis name, e.g. ``"def2-TZVP"``, ``"6-31G*"``,
        ``"cc-pVTZ"``.  Case-insensitive; ``*`` characters are handled.
    symbol:
        Element symbol, e.g. ``"Fe"``, ``"S"``, ``"Mo"``.  Case-sensitive
        (first letter upper, rest lower — standard chemical notation).

    Returns
    -------
    List[GeneralShellEntry]
        One entry per contracted shell, in the order they appear in the BSE
        JSON files.  Each entry is::

            (angular_type, exponents, coeffs_1, coeffs_2_or_None)

        where all inner containers are ``tuple[float, ...]`` of variable
        length.

    Raises
    ------
    FileNotFoundError
        If no BSE files match *basis_name*.
    KeyError
        If *symbol* is not present in the merged basis set files.
    ValueError
        If an unsupported angular momentum type is encountered.
    """
    stem = _normalise_name(basis_name)
    Z    = _SYM_TO_Z.get(symbol)
    if Z is None:
        raise KeyError(f"[mqebasisloader] Unknown element symbol: '{symbol}'")

    el_map = _load_element_map(stem)
    el_data = el_map.get(Z)
    if el_data is None:
        raise KeyError(
            f"[mqebasisloader] Element '{symbol}' (Z={Z}) not found in "
            f"basis '{basis_name}' (stem='{stem}')."
        )

    shells: List[GeneralShellEntry] = []
    for sh in el_data["electron_shells"]:
        am     = list(sh["angular_momentum"])
        exps   = tuple(float(x) for x in sh["exponents"])
        coeffs = sh["coefficients"]   # list of columns (one per contraction / AM)

        if len(am) == 1:
            # Single angular momentum.  ``coefficients`` may hold MULTIPLE
            # columns — a *general contraction* (e.g. cc-pVTZ S: 10 primitives,
            # 4 contracted s-functions sharing those exponents).  Emit one
            # contracted shell per column so the full basis is preserved.
            # (def2 sets are segmented: one column → one shell, unchanged.)
            am_str = _am_to_str(am)
            for col in coeffs:
                shells.append((am_str, exps, tuple(float(c) for c in col), None))
        elif am == [0, 1] or am == [1, 0]:
            # Combined SP shell (shared exponents): col0 = s, col1 = p.
            shells.append((
                "SP", exps,
                tuple(float(c) for c in coeffs[0]),
                tuple(float(c) for c in coeffs[1]),
            ))
        else:
            # General combined L-shell (rare): pair am[i] with coefficients[i].
            for i, l in enumerate(am):
                shells.append((
                    _am_to_str([l]), exps,
                    tuple(float(c) for c in coeffs[i]), None,
                ))

    return shells


# ---------------------------------------------------------------------------
# Effective Core Potential (ECP) loading
# ---------------------------------------------------------------------------

#: ECP term: (n, zeta, d) for the radial function  d * r^{n-2} * exp(-zeta r^2).
EcpTerm  = Tuple[int, float, float]
#: ECP block: (l, [terms]).  The highest l is the LOCAL part U_L.
EcpBlock = Tuple[int, List[EcpTerm]]


def load_bse_ecp(basis_name: str, symbol: str) -> Tuple[int, List[EcpBlock]]:
    """Return ``(n_core, blocks)`` for *symbol*'s effective core potential.

    ``n_core`` = number of core electrons replaced by the ECP (``ecp_electrons``;
    0 if the element is all-electron in this basis).  ``blocks`` is a list of
    ``(l, [(n, zeta, d), ...])`` where each radial channel is

        U_l(r) = Σ_k d_k · r^{n_k - 2} · exp(-zeta_k r²)

    (BSE ``r_exponents`` = n_k; def2 ECPs use n_k = 2 → pure Gaussians).  The
    block with the **highest** l is the local part U_L; the lower-l blocks are
    the semilocal projectors.  Returns ``(0, [])`` when no ECP is present.
    """
    stem = _normalise_name(basis_name)
    Z = _SYM_TO_Z.get(symbol)
    if Z is None:
        raise KeyError(f"[mqebasisloader] Unknown element symbol: '{symbol}'")

    pattern = str(BSE_DIR / f"{stem}.*.json")
    files = sorted(f for f in glob.glob(pattern) if not f.endswith(".ref.bib"))
    n_core = 0
    blocks: List[EcpBlock] = []
    for fpath in files:
        with open(fpath, "r") as fh:
            data = json.load(fh)
        el = data.get("elements", {}).get(str(Z))
        if not el or "ecp_potentials" not in el:
            continue
        n_core = int(el.get("ecp_electrons", 0))
        blocks = []
        for pot in el["ecp_potentials"]:
            l = int(pot["angular_momentum"][0])
            n_list = pot["r_exponents"]
            z_list = pot["gaussian_exponents"]
            d_list = pot["coefficients"][0]
            terms = [
                (int(n), float(z), float(d))
                for n, z, d in zip(n_list, z_list, d_list)
            ]
            blocks.append((l, terms))
        break   # highest-revision file wins
    return n_core, blocks


def ecp_core_electrons(basis_name: str, symbol: str) -> int:
    """Number of core electrons replaced by the ECP (0 if all-electron)."""
    try:
        return load_bse_ecp(basis_name, symbol)[0]
    except (FileNotFoundError, KeyError):
        return 0


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def has_am(basis_name: str, symbol: str, angular_type: str) -> bool:
    """Return True if *symbol* has any shell of *angular_type* in *basis_name*.

    *angular_type* is one of ``"S"``, ``"P"``, ``"D"``, ``"F"``, ``"G"``,
    ``"SP"``.
    """
    try:
        shells = load_bse_basis(basis_name, symbol)
    except (FileNotFoundError, KeyError):
        return False
    return any(s[0] == angular_type for s in shells)


def max_am(basis_name: str, symbol: str) -> int:
    """Return the maximum angular momentum quantum number for *symbol*.

    Returns -1 if the element is not found.
    """
    _am_map = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4, "H": 5, "SP": 1}
    try:
        shells = load_bse_basis(basis_name, symbol)
    except (FileNotFoundError, KeyError):
        return -1
    return max((_am_map.get(s[0], 0) for s in shells), default=0)


@lru_cache(maxsize=1)
def list_available() -> List[str]:
    """Return a sorted list of all basis set stems available in BSE_DIR."""
    stems: set = set()
    pattern = str(BSE_DIR / "*.json")
    for fpath in glob.glob(pattern):
        fname = os.path.basename(fpath)
        m = re.match(r"^(.+)\.\d+\.json$", fname)
        if m:
            stems.add(m.group(1))
    return sorted(stems)


def symbol_in_basis(basis_name: str, symbol: str) -> bool:
    """Return True if *symbol* is covered by *basis_name* in the local BSE mirror."""
    stem = _normalise_name(basis_name)
    Z    = _SYM_TO_Z.get(symbol)
    if Z is None:
        return False
    try:
        el_map = _load_element_map(stem)
    except FileNotFoundError:
        return False
    return Z in el_map
