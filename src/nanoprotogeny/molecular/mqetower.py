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
mqetower.py — Kummer / p-adic interpolation tower.
=================================================
Geometric p-adic interpolation of the seed energy toward E_∞ and the Kummer
tower sequence E^(k).
"""

from __future__ import annotations

from typing import List, Tuple

from nanoprotogeny.molecular.mqeconstants import _K_BASE

def padicinterp_energy(
    k: int, k_base: int, E_base: float, E_target: float, p: int
) -> float:
    r"""E(k) via geometric p-adic interpolation.

    E(k) = E_target + (E_base − E_target) · p^{−(k−k_base)}

    Consistent with thm:padicinterp in mqe-hilbert-polya.md.
    """
    return E_target + (E_base - E_target) * (p ** (-(k - k_base)))


# ===========================================================================
# SECTION 7 — KUMMER TOWER
# ===========================================================================

def compute_tower(
    E_inf:  float,
    E_seed: float,
    k_0:    int,
    K_max:  int = 12,
    p:      int = 2,
    k_base: int = _K_BASE,
) -> List[Tuple[int, float, float]]:
    """Kummer tower E^(k) for k = k_base, …, k_0+K_max.

    Per cor:sedfree_algorithm S7: the tower always starts at k=k_base
    (the CAS(4,4) seed level, default 2) and runs to k_0+K_max.
    k_0 is the convergence level (first k with Δ_k < ε); k_base is the
    fixed starting point — never k_0.

    Returns:
        List of (k, E_k, delta_k) where delta_k = |E_k − E_inf|.
    """
    tower = []
    k_end = max(k_0, k_base) + K_max
    for k in range(k_base, k_end + 1):
        E_k     = padicinterp_energy(
            k=k, k_base=k_base,
            E_base=E_seed, E_target=E_inf, p=p,
        )
        delta_k = abs(E_k - E_inf)
        tower.append((k, E_k, delta_k))
    return tower
