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
mqeconstants.py — Shared physical constants and element tables.
==============================================================
Single source of truth for physical constants, Cartesian angular-momentum
component tables, transition-metal / sulfur basis-routing sets, and nuclear
charges used across the seed-free Hamiltonian-from-Riemann-zeros pipeline.
"""

from typing import Dict, List, Tuple

import nanoprotogeny.basis.mqebasis as _mb

# ── Physical constants ──
_BOHR_PER_ANG = 1.8897259886          # Å → bohr
_KB_HA        = 3.166811e-6           # k_B in Ha/K
_H_S          = 6.62607015e-34        # Planck (J·s)
_KB_S         = 1.38064852e-23        # Boltzmann (J/K)
_HA_J         = 4.359744650e-18       # 1 Ha in Joules
_KCONV_PREF   = _KB_S / _H_S          # k_BT/h prefactor (s⁻¹/K)
_EPS_MILLI_HA = 1.6e-3                # 1.6 mHa convergence threshold
_K_BASE       = 2                     # CAS(4,4) seed tower level

# ── Basis-routing sets ──
#: Elements whose STO-3G contains a D shell — routed to Cartesian p/d integrals.
_HAS_D_BASIS: frozenset = _mb.HAS_D
#: Elements that receive additional p-type shells alongside the outer s-shell
#: in the Cartesian integral path (sulfur needs p-functions for S–H/S–C bonds).
_HAS_P_BASIS: frozenset = frozenset({"S"})

#: Cartesian angular momentum component tuples (lx, ly, lz) for l = 0..4.
#: s(1), p(3), d(6), f(10), g(15).  Used by the McMurchie-Davidson integral
#: engine; the recurrences (_build_e1d, _build_R, _norm_cart, Boys F_n) are
#: l-general, so f/g shells are supported once their components are listed here.
_CART_COMPS: Dict[int, List[Tuple[int, int, int]]] = {
    0: [(0, 0, 0)],
    1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    2: [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)],
    3: [(3, 0, 0), (0, 3, 0), (0, 0, 3), (2, 1, 0), (2, 0, 1),
        (1, 2, 0), (0, 2, 1), (1, 0, 2), (0, 1, 2), (1, 1, 1)],
    4: [(4, 0, 0), (0, 4, 0), (0, 0, 4), (3, 1, 0), (3, 0, 1),
        (1, 3, 0), (0, 3, 1), (1, 0, 3), (0, 1, 3), (2, 2, 0),
        (2, 0, 2), (0, 2, 2), (2, 1, 1), (1, 2, 1), (1, 1, 2)],
}

#: Nuclear charges Z by element symbol.
_NUCLEAR_CHARGES: Dict[str, float] = {
    "H": 1.0, "C": 6.0, "N": 7.0, "O": 8.0, "S": 16.0,
    "Fe": 26.0, "Mo": 42.0, "Mn": 25.0, "V": 23.0, "Cu": 29.0, "Ti": 22.0,
}
