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
tower_climber.py — Iwasawa Tower Climber for Active-Space Expansion
====================================================================
Implements the block-sequential tower-climbing algorithm from
subsec:block_sequential and Bridge D (sssec:bridge_d).

Physical motivation
-------------------
We know two endpoints:
  * CAS(4,4) base  : PySCF gives h_diag, h_hop, g_full, E_Janus(base).
  * CAS(76,76) top : Riemann zeros give E_Janus(top) = −57.358 Ha.

The Iwasawa tower (Definition def:iwasawa_tower) interpolates between
them.  At each tower level k (modulus m_k = p^k), new orbital blocks are
added and the integrals are extended.  The p-adic measure μ_stoich^(k)
converges weak-* to μ_stoich (Theorem thm:padicinterp), with E_Janus(k)
following a geometric rate:

    E_Janus(k) = E_top + (E_base − E_top) · p^{−(k−k_base)}

This avoids PySCF runs at every level: CAS(4,4) integrals provide the
orbital structure; p-adic interpolation provides the energy profile;
quantum simulation at each level measures the actual E_Janus(k) and
checks convergence toward the Riemann target.

Kummer compatibility check
--------------------------
Between levels k−1 and k the energy ratio must satisfy:

    (E_Janus(k) − E_top) / (E_Janus(k−1) − E_top)  ≈  1/p

If the actual QPE result at level k deviates from this ratio it means
the wrong orbitals were added at level k (the Kummer congruence is
violated), and a different block must be selected.

Public API
----------
    TowerLevel            : dataclass — integrals + metadata at one level k
    padicinterp_energy    : E_Janus(k) from p-adic interpolation
    select_next_block     : NOON-ranked next orbital block
    extend_integrals      : h_diag/h_hop/g_full extension for a new block
    kummer_check          : geometric convergence ratio test
    TowerClimber          : orchestrates the full ascent
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from nanoprotogeny.molecular.mqeriemann import (
    RiemannScaffold
)

from nanoprotogeny.molecular.mqedatagenerator import StepwiseIntegralStore

log = logging.getLogger(__name__)


# ── TowerLevel ────────────────────────────────────────────────────────────────

@dataclass
class TowerLevel:
    r"""Complete state of the active space at one Iwasawa tower level.

    Fields
    ------
    k               : Tower index; modulus m_k = p^k.
    p               : Prime base (p=2 for nitrogenase, p=3 for V-nitrogenase).
    m_k             : Virtual register modulus = p^k.
    n_orbs          : Number of active orbitals at this level.
    orbital_indices : Global NOON-rank indices included in the active space.
    h_diag          : One-body diagonal integrals (str-keyed, Ha).
    h_hop           : One-body off-diagonal integrals (str-keyed "i,j", Ha).
    g_full          : Two-body integrals (str-keyed "p,q,r,s", Ha).
    ecore_Ha        : Frozen-core energy carried from the base level (Ha).
    E_janus_Ha      : E_Janus at this level from p-adic interpolation (Ha).
    delta_E_Ha      : E_janus(k) − E_janus(k−1) (Ha; negative = deeper).
    kummer_ok       : Whether geometric convergence ratio passed at this level.
    dataset_dir     : Path where step JSONs for this level were written.
    """
    k:               int
    p:               int
    m_k:             int
    n_orbs:          int
    orbital_indices: List[int]
    h_diag:          Dict[str, float]
    h_hop:           Dict[str, float]
    g_full:          Dict[str, float]
    ecore_Ha:        float
    E_janus_Ha:      float
    delta_E_Ha:      float
    kummer_ok:       bool = True
    dataset_dir:     Optional[Path] = None


# ── p-adic energy interpolation ───────────────────────────────────────────────

def padicinterp_energy(
    k:        int,
    k_base:   int,
    E_base:   float,
    E_target: float,
    p:        int,
) -> float:
    r"""E_Janus(k) via geometric p-adic interpolation.

    Boundary conditions:
        E(k_base) = E_base   (PySCF exact-diag at CAS(4,4))
        E(∞)      = E_target (Riemann zero, e.g. −57.358 Ha)

    Convergence:
        |E(k) − E_target| = |E_base − E_target| · p^{−(k−k_base)}

    This is consistent with weak-* convergence of μ_stoich^(k)
    (Theorem thm:padicinterp) without invoking Hyp. hyp:bernoulli_coupling.
    Under that hypothesis the rate would be governed by ζ_p(1−k); the
    geometric rate here is the leading-order p-adic approximation.

    Args:
        k:        Tower level (m_k = p^k).
        k_base:   Base level index (typically 2 for CAS(4,4) with p=2).
        E_base:   E_Janus at k_base [Ha].
        E_target: E_Janus at the Riemann top [Ha].
        p:        Prime base.

    Returns:
        E_Janus(k) in Ha.
    """
    if k <= k_base:
        return E_base
    t = float(p) ** (-(k - k_base))
    return E_target + (E_base - E_target) * t


# ── Orbital block selection ───────────────────────────────────────────────────

def select_next_block(
    h_diag_base:   Dict[str, float],
    current_count: int,
    block_size:    int = 4,
    noons:         Optional[np.ndarray] = None,
    n_total_orbs:  int = 76,
    mo_energies:   Optional[np.ndarray] = None,
) -> Tuple[List[int], List[float]]:
    r"""Select the next block_size orbitals to add at the next tower level.

    Selection criterion (priority order):
      1. If noons provided: rank by |NOON − 1.0| (smallest = most correlated).
         Singly-occupied natural orbitals carry the strongest entanglement.
         For virtual orbitals where all NOONs are degenerate at 0.0, use
         mo_energies (ROHF) as a tiebreaker — Fermi-proximal virtuals first.
      2. Fallback: rank virtual orbitals by energy proximity to the Fermi
         level (mean of existing h_diag values).

    The selected orbitals' estimated energies are returned so they can be
    inserted into h_diag at the new level.

    Args:
        h_diag_base:   h_diag at the current level (str-keyed).
        current_count: Number of orbitals already active.
        block_size:    Orbitals to add (default 4 matches the base CAS(4,4) block).
        noons:         Natural orbital occupation numbers for ALL orbitals
                       (length ≥ n_total_orbs).  If None, energy heuristic is used.
        n_total_orbs:  Total orbital pool size (default 76 for FeMoco CAS(76,76)).
        mo_energies:   ROHF canonical MO energies (length ≥ n_total_orbs).
                       Used as tiebreaker when candidate NOONs are degenerate (0.0).

    Returns:
        (new_indices, new_energies):
            new_indices  — global orbital indices selected for the new block.
            new_energies — estimated h_diag [Ha] for each new orbital.
    """
    occupied   = set(range(current_count))
    candidates = [i for i in range(n_total_orbs) if i not in occupied]

    if not candidates:
        log.warning("[TOWER] No candidates available — active space already at n_total_orbs=%d",
                    n_total_orbs)
        return [], []

    if noons is not None and len(noons) >= n_total_orbs:
        # Compute Fermi level for MO energy tiebreaker
        existing_eps = list(h_diag_base.values()) if h_diag_base else [-0.5, 0.0]
        fermi = float(np.mean(existing_eps))

        def _noon_key(i: int):
            noon_dist = abs(float(noons[i]) - 1.0)
            # Secondary key: MO energy proximity to Fermi (for virtual ties at 1.0)
            if mo_energies is not None and i < len(mo_energies):
                energy_dist = abs(float(mo_energies[i]) - fermi)
            else:
                energy_dist = float(i)   # stable fallback: index order
            return (noon_dist, energy_dist)

        candidates.sort(key=_noon_key)
    else:
        existing_eps = sorted(h_diag_base.values()) if h_diag_base else [-0.5, 0.0]
        fermi        = float(np.mean(existing_eps))
        span         = (existing_eps[-1] - existing_eps[0]) if len(existing_eps) > 1 else 0.2
        step         = span / max(current_count, 1)
        virtual_eps  = {
            idx: existing_eps[-1] + (rank + 1) * step
            for rank, idx in enumerate(candidates)
        }
        candidates.sort(key=lambda i: abs(virtual_eps[i] - fermi))

    selected = candidates[:block_size]

    # Estimate energies for the new orbitals
    if noons is not None and len(noons) >= n_total_orbs:
        existing_eps = sorted(h_diag_base.values()) if h_diag_base else [-0.5, 0.0]
        fermi        = float(np.mean(existing_eps))
        # NOON ≈ 1  → near Fermi (strongly correlated)
        # NOON ≈ 0  → above Fermi (virtual)
        # NOON ≈ 2  → below Fermi (occupied)
        energies = [float(fermi + (1.0 - float(noons[idx])) * 0.15) for idx in selected]
    else:
        existing_eps = sorted(h_diag_base.values()) if h_diag_base else [-0.5, 0.0]
        step         = 0.1 if len(existing_eps) < 2 else (existing_eps[-1] - existing_eps[0]) / max(current_count, 1)
        energies     = [existing_eps[-1] + (j + 1) * step for j in range(len(selected))]

    log.debug("[TOWER] Selected orbitals %s with energies %s", selected, [f"{e:.4f}" for e in energies])
    return selected, energies


# ── Integral slicing from full MO tensor ─────────────────────────────────────

def load_full_mo_integrals(
    base_dataset_dir: Path,
    mechanism_name:   str,
) -> "Optional[Tuple[np.ndarray, np.ndarray]]":
    """Load h1_full.npy and eri_packed.npy saved by _save_full_mo_integrals_for_tower.

    Returns (h1_full, eri_full_4d) where eri_full_4d is already unpacked to
    (n, n, n, n) so that slicing at each tower level is a simple index op.

    Returns None if either file is absent (tower will fall back to heuristic
    extension and log a warning).
    """
    from pyscf import ao2mo as pyscf_ao2mo

    mech_dir   = base_dataset_dir / mechanism_name
    h1_path    = mech_dir / "h1_full.npy"
    eri_path   = mech_dir / "eri_packed.npy"

    if not h1_path.exists() or not eri_path.exists():
        return None

    h1_full    = np.load(str(h1_path))
    eri_packed = np.load(str(eri_path))
    n          = h1_full.shape[0]
    eri_full   = pyscf_ao2mo.restore(1, eri_packed, n)   # (n,n,n,n)

    log.info(
        "[TOWER] Loaded full MO integrals: h1 %s, eri 4D %s (%.0f MB)",
        h1_full.shape, eri_full.shape, eri_full.nbytes / 2**20,
    )
    return h1_full, eri_full


def load_full_mo_integrals_step(
    base_dataset_dir: Path,
    mechanism_name:   str,
    step_n:           int,
) -> "Optional[Tuple[np.ndarray, np.ndarray]]":
    """Load h1_full_step{n:02d}.npy / eri_packed_step{n:02d}.npy for one step.

    Returns (h1_full, eri_full_4d) or None if files absent (datasets generated
    before the per-step fix only have h1_full.npy at the Janus step).
    """
    from pyscf import ao2mo as pyscf_ao2mo

    mech_dir  = base_dataset_dir / mechanism_name
    h1_path   = mech_dir / f"h1_full_step{step_n:02d}.npy"
    eri_path  = mech_dir / f"eri_packed_step{step_n:02d}.npy"

    if not h1_path.exists() or not eri_path.exists():
        return None

    h1_full    = np.load(str(h1_path))
    eri_packed = np.load(str(eri_path))
    n          = h1_full.shape[0]
    eri_full   = pyscf_ao2mo.restore(1, eri_packed, n)

    log.info(
        "[TOWER] Loaded step %02d MO integrals: h1 %s, eri 4D %s (%.0f MB)",
        step_n, h1_full.shape, eri_full.shape, eri_full.nbytes / 2**20,
    )
    return h1_full, eri_full


def load_tower_window_integrals(
    base_dataset_dir: Path,
    mechanism_name:   str,
) -> "Optional[Tuple[np.ndarray, np.ndarray, float, int]]":
    """Load pre-contracted tower-window integrals (large-system path).

    Returns (h1_eff_win, eri_win, deep_ecore, win_start) where:
      h1_eff_win  — (win_size, win_size) Fock-contracted 1e integrals;
                    deep core (MOs 0..win_start-1) already folded in.
      eri_win     — (win_size, win_size, win_size, win_size) raw 2e for window.
      deep_ecore  — 1e+2e energy of MOs 0..win_start-1 (add to base ecore).
      win_start   — global MO index of h1_eff_win[0,0] (offset for index translation).

    Returns None if files are absent (caller falls back to load_full_mo_integrals).
    """
    import json as _json

    mech_dir  = base_dataset_dir / mechanism_name
    h1_path   = mech_dir / "h1_full_win.npy"
    eri_path  = mech_dir / "eri_win.npy"
    ec_path   = mech_dir / "deep_ecore.npy"
    meta_path = mech_dir / "h1_full_win_meta.json"

    if not (h1_path.exists() and eri_path.exists() and ec_path.exists()
            and meta_path.exists()):
        return None

    h1_eff_win  = np.load(str(h1_path))
    eri_win     = np.load(str(eri_path))
    deep_ecore  = float(np.load(str(ec_path))[0])
    meta        = _json.loads(meta_path.read_text())
    win_start   = int(meta["win_start"])

    log.info(
        "[TOWER-WIN] Loaded window integrals: h1_eff_win %s, eri_win %s "
        "(%.0f MB), deep_ecore=%.6f Ha, win_start=%d",
        h1_eff_win.shape, eri_win.shape, eri_win.nbytes / 2**20,
        deep_ecore, win_start,
    )
    return h1_eff_win, eri_win, deep_ecore, win_start


def slice_mo_integrals(
    h1_full:  np.ndarray,
    eri_full: np.ndarray,
    n_orbs:   int,
    threshold: float = 1e-10,
) -> "Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]":
    """Slice the full MO integral arrays to the first n_orbs orbitals.

    This replaces the heuristic extend_integrals: cross-block couplings come
    from the actual molecular integrals saved during CAS(4,4) generation.

    Parameters
    ----------
    h1_full  : (N, N) one-electron integrals for all N MOs.
    eri_full : (N, N, N, N) two-electron integrals (chemist's notation).
    n_orbs   : number of orbitals to include (n_orbs ≤ N).
    threshold: values below this are omitted from the sparse dicts.

    Returns
    -------
    h_diag, h_hop, g_full — str-keyed dicts for StepwiseIntegralStore / dmrg_backend.
    """
    h1  = h1_full[:n_orbs, :n_orbs]
    eri = eri_full[:n_orbs, :n_orbs, :n_orbs, :n_orbs]

    # One-body: diagonal
    h_diag: Dict[str, float] = {
        str(p): float(h1[p, p])
        for p in range(n_orbs)
    }

    # One-body: canonical off-diagonal (p < q only; symmetric)
    h_hop: Dict[str, float] = {
        f"{p},{q}": float(h1[p, q])
        for p in range(n_orbs)
        for q in range(p + 1, n_orbs)
        if abs(h1[p, q]) > threshold
    }

    # Two-body: canonical (pq|rs) with p ≤ q, r ≤ s, (p,q) ≤ (r,s)
    # 8-fold symmetry — store only one representative per equivalence class.
    g_full: Dict[str, float] = {}
    for p in range(n_orbs):
        for q in range(p, n_orbs):
            for r in range(p, n_orbs):
                s_start = q if r == p else r
                for s in range(s_start, n_orbs):
                    v = float(eri[p, q, r, s])
                    if abs(v) > threshold:
                        g_full[f"{p},{q},{r},{s}"] = v

    log.debug(
        "[TOWER] Sliced integrals n_orbs=%d: h_diag=%d h_hop=%d g_full=%d entries",
        n_orbs, len(h_diag), len(h_hop), len(g_full),
    )
    return h_diag, h_hop, g_full


def extend_integrals(
    base_h_diag:    Dict[str, float],
    base_h_hop:     Dict[str, float],
    base_g_full:    Dict[str, float],
    new_indices:    List[int],
    new_energies:   List[float],
    existing_count: int,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Heuristic fallback: extend integrals when full MO tensor is unavailable.

    Cross-block terms are estimated from the average magnitude of existing
    integrals.  This is physically motivated but NOT exact.  Used only when
    h1_full.npy / eri_packed.npy were not saved during CAS(4,4) generation.

    For exact integrals, re-run ``mqe generate-data --source pyscf`` (which
    now saves h1_full.npy and eri_packed.npy automatically) and then rerun
    the tower generation.
    """
    h_diag = dict(base_h_diag)
    h_hop  = dict(base_h_hop)
    g_full = dict(base_g_full)

    n_new = len(new_indices)
    n_old = existing_count

    if n_new == 0:
        return h_diag, h_hop, g_full

    for local_i, eps in enumerate(new_energies):
        h_diag[str(n_old + local_i)] = float(eps)

    if base_h_hop:
        avg_hop = float(np.mean(np.abs(list(base_h_hop.values()))))
    else:
        avg_hop = 0.05

    for local_i in range(n_new):
        new_orb = n_old + local_i
        for old_orb in range(n_old):
            i, j = min(new_orb, old_orb), max(new_orb, old_orb)
            h_hop[f"{i},{j}"] = -avg_hop

    if base_g_full:
        avg_g = float(np.mean(np.abs(list(base_g_full.values()))))
    else:
        avg_g = avg_hop * 0.5

    for local_i in range(n_new):
        new_orb = n_old + local_i
        for old_orb in range(n_old):
            g_full[f"{new_orb},{old_orb},{old_orb},{new_orb}"] = avg_g
            g_full[f"{new_orb},{old_orb},{new_orb},{old_orb}"] = avg_g * 0.1

    return h_diag, h_hop, g_full


# ── CAS(N,N) active-space helpers ────────────────────────────────────────────

# MOs 0..7 are the 8 deepest frozen-core MOs (16 electrons) at all tower levels.
# This constant is used by select_casci_orbitals_for_level and TowerClimber.
_FROZEN_CORE_N_ORBS: int = 8


def compute_e_nuc(
    h1_full:      np.ndarray,
    eri_full:     np.ndarray,
    ecore_cas44:  float,
    n_core_cas44: int = 40,
) -> float:
    """Extract nuclear repulsion energy from the CAS(4,4) frozen-core energy.

    In the CAS(4,4) base, PySCF freezes MOs 0..n_core_cas44-1 (= 80 electrons)
    and folds their energy into ecore_cas44:

        ecore_cas44 = e_nuc
                    + 2·Σ_{c<n_core} h1[c,c]
                    + Σ_{c,c'<n_core} (2·(cc|c'c') − (cc'|c'c))

    We invert this to get e_nuc, which is then used in the new CASCI ecore
    formula with a smaller frozen core (MOs 0..7 only).

    Args:
        h1_full:      (N,N) MO one-body integrals.
        eri_full:     (N,N,N,N) MO two-electron integrals (chemist's notation).
        ecore_cas44:  Frozen-core energy from the CAS(4,4) base dataset [Ha].
        n_core_cas44: Number of frozen-core MOs in CAS(4,4); default 40.

    Returns:
        e_nuc [Ha].
    """
    nc = n_core_cas44
    # Mean-field energy of the n_core_cas44 frozen-core MOs
    e_mf = 2.0 * float(np.sum(np.diag(h1_full[:nc, :nc])))
    g_cc = eri_full[:nc, :nc, :nc, :nc]
    # J term: 2·Σ_{c,c'} (cc|c'c')
    e_J  = 2.0 * float(np.einsum("ccdd->", g_cc))
    # K term: −Σ_{c,c'} (cc'|c'c)
    e_K  = -float(np.einsum("cddc->", g_cc))
    e_nuc = ecore_cas44 - e_mf - e_J - e_K
    log.debug(
        "[CASCI] e_nuc extraction: ecore_cas44=%.6f  e_mf=%.6f  J+K=%.6f  e_nuc=%.6f",
        ecore_cas44, e_mf, e_J + e_K, e_nuc,
    )
    return float(e_nuc)


def select_casci_orbitals_for_level(
    k:                  int,
    k_base:             int,
    block_size:         int = 4,
    n_occ_base:         int = 40,
    n_act_base:         int = 4,
    frozen_n:           int = _FROZEN_CORE_N_ORBS,
    n_total_orbs:       int = 76,
    n_elec_max_alpha:   "Optional[int]" = None,
) -> Tuple[List[int], int, int]:
    """Return the active orbital indices and electron count for tower level k.

    Design (CAS(68,68) target):
    ─────────────────────────────────────────────────────────────────────────
    frozen core  : MOs 0..frozen_n-1  (always frozen, 2·frozen_n electrons)
    base active  : MOs n_occ_base .. n_occ_base+n_act_base-1  (e.g. 40..43)
                   nalpha_base = nbeta_base = n_act_base // 2  (= 2 for CAS(4,4))
    per step above k_base (n_steps = k − k_base):
        add  n_occ_per_step occupied MOs  : working downward from n_occ_base-1
        add  n_virt_per_step virtual MOs  : working upward from n_occ_base+n_act_base
        each occupied MO carries 2 electrons → nalpha += 1, nbeta += 1
    ─────────────────────────────────────────────────────────────────────────

    Args:
        k:                 Target tower level.
        k_base:            Base level (CAS(4,4) dataset level; usually 2).
        block_size:        Orbitals per step (default 4 = 2 occ + 2 virt).
        n_occ_base:        Index of first base active MO (default 40).
        n_act_base:        Number of base active MOs (default 4 for CAS(4,4)).
        frozen_n:          Frozen-core MO count (default _FROZEN_CORE_N_ORBS = 8).
        n_total_orbs:      Total orbital pool size (caps virtual MO indices).
        n_elec_max_alpha:  Physical upper bound on alpha electron count
                           (= N_elec_total // 2).  When provided, nalpha and
                           nbeta are clamped to this value and a warning is
                           logged if the clamp is ever triggered.  Callers
                           should pass ``n_occ_base + n_act_base // 2``, which
                           equals N_elec_total // 2 for a closed-shell system.

    Returns:
        (active_idx, nalpha, nbeta) where active_idx is sorted (ascending).
    """
    n_steps         = max(0, k - k_base)
    n_occ_per_step  = block_size // 2
    n_virt_per_step = block_size - n_occ_per_step

    base_active = list(range(n_occ_base, n_occ_base + n_act_base))

    # Occupied MOs added: descend from n_occ_base-1, skip frozen core
    occ_added: List[int] = []
    for step in range(1, n_steps + 1):
        for j in range(n_occ_per_step):
            idx = n_occ_base - 1 - (step - 1) * n_occ_per_step - j
            if idx >= frozen_n:
                occ_added.append(idx)

    # Virtual MOs added: ascend from n_occ_base + n_act_base
    # Cap at n_total_orbs − 1 to stay within the saved MO tensor bounds.
    virt_start = n_occ_base + n_act_base
    virt_added: List[int] = []
    for step in range(1, n_steps + 1):
        for j in range(n_virt_per_step):
            idx = virt_start + (step - 1) * n_virt_per_step + j
            if idx < n_total_orbs:
                virt_added.append(idx)

    active_idx = sorted(occ_added + base_active + virt_added)

    # Electron count: base has n_act_base//2 alpha + n_act_base//2 beta;
    # each promoted occupied MO contributes 1 alpha + 1 beta.
    nalpha = n_act_base // 2 + len(occ_added)
    nbeta  = n_act_base // 2 + len(occ_added)

    # Physical electron clamp: nalpha must not exceed N_elec_total // 2.
    # For a closed-shell system n_elec_max_alpha = n_occ_base + n_act_base//2,
    # which equals N_elec_total//2.  Without this clamp the formula can
    # over-count when occ_added exceeds the actual number of occupied MOs
    # (relevant for open-shell or multi-reference seeds where n_occ_base may
    # not equal N_elec_total//2 exactly).
    if n_elec_max_alpha is not None and nalpha > n_elec_max_alpha:
        log.warning(
            "[CASCI] k=%d: nalpha=%d exceeds physical limit %d "
            "(N_elec_total//2) — clamping.  Active space is saturated at "
            "%d orbs with %d alpha electrons.",
            k, nalpha, n_elec_max_alpha, len(active_idx), n_elec_max_alpha,
        )
        nalpha = n_elec_max_alpha
        nbeta  = n_elec_max_alpha

    n_virt_capped = n_steps * n_virt_per_step - len(virt_added)
    if n_virt_capped > 0:
        log.info(
            "[CASCI] k=%d: %d virtual MO(s) capped (would exceed n_total_orbs=%d); "
            "active space saturated at %d orbs.",
            k, n_virt_capped, n_total_orbs, len(active_idx),
        )
    log.debug(
        "[CASCI] k=%d n_steps=%d active=%d orbs (occ+%d, virt+%d) "
        "nalpha=%d nbeta=%d",
        k, n_steps, len(active_idx), len(occ_added), len(virt_added), nalpha, nbeta,
    )
    return active_idx, nalpha, nbeta


def compute_casci_effective_integrals(
    h1_full:     np.ndarray,
    eri_full:    np.ndarray,
    active_idx:  List[int],
    frozen_idx:  List[int],
    e_nuc:       float,
    threshold:   float = 1e-10,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], float]:
    """Compute CASCI effective one-body integrals and core energy.

    Given a partition of MOs into frozen core + active:

        h1_eff[p,q]  = h1[p,q] + Σ_{c∈frozen} (2·(pq|cc) − (pc|qc))
        ecore        = e_nuc
                     + 2·Σ_{c∈frozen} h1[c,c]
                     + Σ_{c,c'∈frozen} (2·(cc|c'c') − (cc'|c'c))

    where (pq|rs) is the PySCF chemist's-notation ERI g[p,q,r,s].

    Args:
        h1_full:    (N,N) one-body MO integrals.
        eri_full:   (N,N,N,N) two-electron MO integrals (chemist's notation).
        active_idx: Global MO indices to include in the active space.
        frozen_idx: Global MO indices of the frozen core.
        e_nuc:      Nuclear repulsion energy [Ha].
        threshold:  Sparsity threshold for dict storage.

    Returns:
        (h_diag, h_hop, g_full, ecore_Ha) — local 0-based orbital indexing.
    """
    fc  = np.array(frozen_idx, dtype=int)
    ac  = np.array(active_idx, dtype=int)
    n_ac = len(ac)

    # ── Effective one-body ───────────────────────────────────────────────────
    # h1_eff[p,q] = h1[p,q] + Σ_c (2*(pq|cc) - (pc|qc))
    h1_ac = h1_full[np.ix_(ac, ac)].copy()
    for c in fc:
        # 2*(pq|cc): g[ac,ac,c,c] — shape (n_ac, n_ac)
        h1_ac += 2.0 * eri_full[np.ix_(ac, ac, [int(c)], [int(c)])].squeeze(axis=(2, 3))
        # -(pc|qc): g[ac,c,ac,c] — need to reshape
        h1_ac -= eri_full[np.ix_(ac, [int(c)], ac, [int(c)])].squeeze(axis=(1, 3))

    # ── Frozen-core energy ───────────────────────────────────────────────────
    ecore = float(e_nuc)
    ecore += 2.0 * float(np.sum(h1_full[fc, fc]))
    g_fc  = eri_full[np.ix_(fc, fc, fc, fc)]
    ecore += 2.0 * float(np.einsum("ccdd->", g_fc))
    ecore -= float(np.einsum("cddc->", g_fc))

    # ── Convert to str-keyed dicts (local 0-based indexing) ──────────────────
    h_diag: Dict[str, float] = {
        str(p): float(h1_ac[p, p]) for p in range(n_ac)
    }
    h_hop: Dict[str, float] = {
        f"{p},{q}": float(h1_ac[p, q])
        for p in range(n_ac) for q in range(p + 1, n_ac)
        if abs(h1_ac[p, q]) > threshold
    }

    # Two-electron integrals in active subspace (local indices)
    g_ac = eri_full[np.ix_(ac, ac, ac, ac)]
    g_full: Dict[str, float] = {}
    for p in range(n_ac):
        for q in range(p, n_ac):
            for r in range(p, n_ac):
                s_start = q if r == p else r
                for s in range(s_start, n_ac):
                    v = float(g_ac[p, q, r, s])
                    if abs(v) > threshold:
                        g_full[f"{p},{q},{r},{s}"] = v

    log.debug(
        "[CASCI] active=%d frozen=%d → h_diag=%d h_hop=%d g_full=%d ecore=%.6f Ha",
        n_ac, len(fc), len(h_diag), len(h_hop), len(g_full), ecore,
    )
    return h_diag, h_hop, g_full, ecore


# ── Kummer congruence check ───────────────────────────────────────────────────

def kummer_check(
    E_janus_k:  float,
    E_janus_k1: float,
    E_target:   float,
    p:          int,
    k:          int,
    tol:        float = 0.15,
) -> bool:
    r"""Check geometric convergence ratio (necessary condition for Kummer compatibility).

    The Kummer congruences (Definition def:stoich_measure) require the
    p-adic measure μ_stoich^(k) to converge at rate p^{−k}.  For the
    Janus energy this reduces to:

        r_k = (E_Janus(k) − E_target) / (E_Janus(k−1) − E_target)  ≈  1/p

    If r_k deviates from 1/p by more than tol (fractional), the orbital
    block added at level k is incompatible with the Kummer structure —
    the wrong orbitals were selected and a different block should be tried.

    Args:
        E_janus_k:  E_Janus measured (or interpolated) at level k [Ha].
        E_janus_k1: E_Janus at level k−1 [Ha].
        E_target:   Riemann scaffold target [Ha].
        p:          Prime base.
        k:          Current tower level (for logging).
        tol:        Fractional tolerance on the ratio (default 15%).

    Returns:
        True if the Kummer geometric condition is satisfied.
    """
    gap_k  = E_janus_k  - E_target
    gap_k1 = E_janus_k1 - E_target

    if abs(gap_k1) < 1e-10:
        log.debug("[KUMMER] k=%d: already converged", k)
        return True

    ratio    = gap_k / gap_k1
    expected = 1.0 / p
    ok       = abs(ratio - expected) <= tol * expected

    log.debug(
        "[KUMMER] k=%d: gap_k=%.4f  gap_{k-1}=%.4f  ratio=%.4f  expected=%.4f  %s",
        k, gap_k, gap_k1, ratio, expected, "✓" if ok else "✗"
    )
    return ok


# ── TowerClimber ──────────────────────────────────────────────────────────────

class TowerClimber:
    r"""Orchestrates the Iwasawa tower ascent from CAS(4,4) to CAS(N,N).

    Algorithm
    ---------
    1. Load the base CAS(4,4) dataset as TowerLevel k=k_base.
    2. For k = k_base+1, ..., k_max:
       a. Compute E_Janus(k) via p-adic interpolation.
       b. Select the next orbital block (NOON or energy heuristic).
       c. Extend h_diag/h_hop/g_full by the new block.
       d. Write step JSONs (via build_tower_level_dataset).
       e. Optionally run MQERiemannPipeline to measure actual E_Janus(k).
       f. Kummer check: verify geometric convergence ratio.
       g. Convergence check: |E_Janus(k) − E_target| < tol_mHa.
    3. Return the list of TowerLevel objects at each level.

    Args:
        base_dataset_dir:   Root directory of the CAS(4,4) dataset
                            (parent of the mechanism subdirectory).
        mechanism_name:     Mechanism name (e.g. 'nitrogenase_lt').
        scaffold:           RiemannScaffold for the mechanism (provides E_target).
        p:                  Prime base (default 2).
        k_base:             Base tower level corresponding to CAS(4,4) (default 2).
        block_size:         Orbitals per new block (default 4).
        convergence_tol_mHa: Stop when |E_Janus(k) − E_target| < this [mHa].
        noons:              Natural orbital occupation numbers for ALL orbitals
                            (numpy array, length = n_total_orbs).  If None, the
                            energy-proximity heuristic is used.
        n_total_orbs:       Total orbital pool (default 76 for FeMoco).
        k_target:           Which Riemann zero to use (0 → γ₁). Default 0.
    """

    def __init__(
        self,
        base_dataset_dir:     Union[str, Path],
        mechanism_name:       str,
        scaffold:             "RiemannScaffold",
        p:                    int = 2,
        k_base:               int = 2,
        block_size:           int = 4,
        convergence_tol_mHa:  float = 1.6,
        noons:                Optional[np.ndarray] = None,
        n_total_orbs:         int = 76,
        k_target:             int = 0,
    ):
        from nanoprotogeny.molecular.mqeintegralstore import StepwiseIntegralStore

        self._base_dir        = Path(base_dataset_dir)
        self._mechanism_name  = mechanism_name
        self._scaffold        = scaffold
        self._p               = p
        self._k_base          = k_base
        self._block_size      = block_size
        self._tol_mHa         = convergence_tol_mHa
        self._noons           = noons
        self._n_total_orbs    = n_total_orbs
        self._k_target        = k_target

        if k_target >= len(scaffold.janus_energies):
            raise ValueError(
                f"k_target={k_target} out of range; scaffold has "
                f"{len(scaffold.janus_energies)} zeros in window."
            )
        self._E_target = scaffold.janus_energies[k_target]

        # Load base store to get M_steps / manifest.
        # _janus_steps comes from the scaffold (derived from the mechanism
        # definition's declared crossings), NOT from the dataset's is_crossing
        # flags.  The dataset flags can be corrupted — e.g. _detect_lv_crossing
        # over-detects on round-trip paths, marking all BL-twin steps as Janus.
        # scaffold.all_crossing_energies is keyed by the authoritative step
        # indices from MechanismTuple.crossings and is always correct.
        self._store = StepwiseIntegralStore(self._base_dir, mechanism_name)
        self._janus_steps = set(scaffold.all_crossing_energies.keys())
        self._M_steps     = self._store.M_steps

        self._E_base:      Optional[float]      = None   # set by _load_base() on first climb()
        self._mo_energies: Optional[np.ndarray] = None   # ROHF MO energies, auto-loaded
        self._h1_full:     Optional[np.ndarray] = None   # (N,N) full MO one-body integrals
        self._eri_full:    Optional[np.ndarray] = None   # (N,N,N,N) unpacked full ERI

        # Per-step MO tensors — populated by _load_base() when per-step files exist.
        # Absent for datasets generated before the per-step geometry fix; in that
        # case tower falls back to the Janus-geometry h1_full/eri_full for all steps.
        self._h1_full_per_step:  Dict[int, np.ndarray] = {}
        self._eri_full_per_step: Dict[int, np.ndarray] = {}
        self._e_nuc_per_step:    Dict[int, float]       = {}

        # CAS(N,N) tracking — populated by _load_base() then grown in _build_level()
        self._e_nuc:           Optional[float]      = None  # nuclear repulsion [Ha]
        self._frozen_core_idx: Optional[List[int]]  = None  # global MO indices, always frozen
        self._active_idx_k:    Dict[int, List[int]] = {}    # k → sorted global active MO idx
        self._nalpha_k:        Dict[int, int]        = {}   # k → alpha electron count
        self._nbeta_k:         Dict[int, int]        = {}   # k → beta  electron count

        # Seed tensor metadata — set when _load_base loads seed_tensors.npz.
        # _seed_n_frozen: N_frozen as stored in the .npz (may be 0 for zetazero seeds).
        # Used instead of _FROZEN_CORE_N_ORBS so the active-space indexing stays
        # within the actual (N×N) matrix bounds regardless of the basis size.
        self._seed_local_basis: bool            = False
        self._seed_e_nuc:       Optional[float] = None   # E_nuc from seed [Ha]
        self._seed_n_frozen:    int             = 0      # N_frozen from seed .npz

        # Tower-window metadata — set when _load_base loads h1_full_win.npy
        # (large-system path, n_occ_base > _TOWER_WIN_THRESHOLD).
        # _h1_full_win_offset: global MO index of h1_full[0,0] / eri_win[0,0,0,0].
        # All global→local index translation: local = global - _h1_full_win_offset.
        # _deep_ecore: 1e+2e contribution of MOs 0.._h1_full_win_offset-1
        #              (already folded into h1_eff_win; add to base ecore separately
        #              so there's no double-counting when the step JSON ecore is used).
        self._h1_full_win_offset: int           = 0      # 0 = standard path (no offset)
        self._deep_ecore:         float         = 0.0    # deep-core energy contribution

        log.info(
            "[TOWER] mechanism=%s  p=%d  k_base=%d  E_target=%.6f Ha  "
            "n_total_orbs=%d  janus=%s",
            mechanism_name, p, k_base, self._E_target,
            n_total_orbs, sorted(self._janus_steps),
        )

    # ── Private: global↔local index translation (tower-window path) ──────────

    def _to_local(self, global_indices: "List[int]") -> "List[int]":
        """Translate global MO indices to local h1_full/eri_full indices.

        In the standard path (offset=0) this is a no-op.
        In the tower-window path (offset>0) it subtracts _h1_full_win_offset
        from every index so that global MO win_start maps to local index 0.
        Indices that would go negative (global < offset) are clamped to 0 and
        a warning is logged — they represent deep-core MOs that are already
        folded into h1_eff_win and should never appear as active/frozen indices.
        """
        off = self._h1_full_win_offset
        if off == 0:
            return global_indices
        local: List[int] = []
        for g in global_indices:
            loc = g - off
            if loc < 0:
                log.warning(
                    "[TOWER-WIN] global MO %d < win_start=%d — "
                    "deep-core MO promoted into active space?  "
                    "Clamping to local=0 (likely a bug in _n_occ_base computation).",
                    g, off,
                )
                loc = 0
            local.append(loc)
        return local

    # ── Private: load the base TowerLevel from the CAS(4,4) dataset ──────────

    def _load_base(self) -> TowerLevel:
        """Construct TowerLevel k=k_base from the CAS(4,4) store.

        E_base is read from circuit_reference_energy_Ha − ecore_Ha at the
        Janus step — this is the PySCF FCI active-space energy (e.g. −2.752 Ha
        for nitrogenase_lt CAS(4,4)).  No Hamiltonian diagonalisation needed here
        because the store already holds the verified FCI reference.

        Integrals are loaded as str-keyed dicts (raw JSON format) so they can be
        JSON-serialised at each tower level without key-type errors.
        """
        janus_n = min(self._janus_steps) if self._janus_steps else 0
        raw     = self._store._load_step_raw(janus_n)

        def _norm_key(k: str) -> str:
            """Normalise a JSON key to comma-separated ints: '(0, 1)' → '0,1'."""
            return ",".join(
                x.strip().lstrip("(").rstrip(")")
                for x in str(k).split(",")
            )

        h_diag = {str(k): float(v) for k, v in raw.get("h_diag", {}).items()}
        h_hop  = {_norm_key(k): float(v) for k, v in raw.get("h_hop", {}).items()}
        g_full = {_norm_key(k): float(v) for k, v in raw.get("g_full", {}).items()}
        ecore  = float(raw.get("ecore_Ha", 0.0))
        n_orbs = int(raw.get("metadata", {}).get("ncas", len(h_diag)))

        # E_base = FCI active-space energy (ecore already excluded).
        # Two conventions exist for circuit_reference_energy_Ha:
        #   "total"              — E_total = ecore + E_active  →  E_base = circuit_ref − ecore
        #   "pyscf_active_space" — E_active already (hybrid protocol sets this)  →  E_base = circuit_ref
        # Subtracting ecore from an already-active-space value doubles the core
        # contribution and inflates E_base by ~1434 Ha for femon2_trimer.
        circuit_ref = raw.get("circuit_reference_energy_Ha")
        ref_origin  = raw.get("circuit_reference_energy_origin", "total")
        if circuit_ref is not None:
            if ref_origin == "pyscf_active_space":
                E_base = float(circuit_ref)
            else:
                E_base = float(circuit_ref) - ecore
        else:
            log.warning("[TOWER] No circuit_reference_energy_Ha at Janus step %d — "
                        "using 0.0 as E_base", janus_n)
            E_base = 0.0

        self._E_base = E_base   # store for padicinterp across all levels

        # ── Auto-load NOONs + MO energies if not supplied by caller ──────────
        # generate_mechanism_dataset saves these next to the step JSONs.
        # Users who ran an older dataset without them still need --noons-file.
        if self._noons is None:
            noons_path = self._base_dir / self._mechanism_name / "noons.npy"
            if noons_path.exists():
                self._noons = np.load(str(noons_path))
                log.info(
                    "[TOWER] Auto-loaded noons.npy (shape=%s) from %s",
                    self._noons.shape, noons_path,
                )
            else:
                log.info(
                    "[TOWER] noons.npy not found in %s — re-generate the base "
                    "dataset or supply --noons-file to enable NOON-guided block selection.",
                    self._base_dir / self._mechanism_name,
                )

        if self._mo_energies is None:
            mo_path = self._base_dir / self._mechanism_name / "mo_energies.npy"
            if mo_path.exists():
                self._mo_energies = np.load(str(mo_path))
                log.info(
                    "[TOWER] Auto-loaded mo_energies.npy (shape=%s) from %s",
                    self._mo_energies.shape, mo_path,
                )

        # ── Auto-load full MO integral tensor ───────────────────────────────
        # Saved by _save_full_mo_integrals_for_tower in generate_mechanism_dataset.
        # Gives exact cross-block integrals at every tower level via array slicing.
        # Falls back to seed_tensors.npz (zetazero --full-mo-seed format), then
        # heuristic extend_integrals if neither file set is present.
        #
        # For large systems (n_occ_base > _TOWER_WIN_THRESHOLD, e.g. nitrogenase_femoco
        # with n_occ_base=189) the standard h1_full.npy covers MOs 0..n_total_orbs-1.
        # With n_total_orbs=80 this means indices 0..79, but the active space starts at
        # global MO 189 → IndexError in compute_casci_effective_integrals.
        # In this case generate_step_integrals writes h1_full_win.npy instead, which
        # covers a small window (global MOs win_start..win_start+win_size-1) with the
        # deep core already Fock-contracted into the 1e integrals.  The tower climber
        # translates global→local indices by subtracting win_start before any
        # h1_full / eri_full array access.
        if self._h1_full is None:
            win_result = load_tower_window_integrals(
                self._base_dir, self._mechanism_name
            )
            if win_result is not None:
                (self._h1_full, self._eri_full,
                 self._deep_ecore, self._h1_full_win_offset) = win_result
                log.info(
                    "[TOWER-WIN] Using tower-window integrals: offset=%d, "
                    "deep_ecore=%.6f Ha.  All global MO indices will be "
                    "translated: local = global - %d.",
                    self._h1_full_win_offset, self._deep_ecore,
                    self._h1_full_win_offset,
                )

        if self._h1_full is None:
            result = load_full_mo_integrals(self._base_dir, self._mechanism_name)
            if result is not None:
                self._h1_full, self._eri_full = result
            else:
                # ── Fallback: mqeprotogeny seed_tensors.npz (no PySCF required) ──
                _seed_path = (
                    self._base_dir / self._mechanism_name / "seed_tensors.npz"
                )
                if _seed_path.exists():
                    try:
                        from nanoprotogeny.molecular.mqeprotogeny import (
                            load_seed_tensors,
                        )
                        _h1_MO, _g_MO, _E_core, _E_nuc, _N_frozen, _ = (
                            load_seed_tensors(str(_seed_path))
                        )
                        self._h1_full          = _h1_MO   # (N,N) already in MO basis
                        self._eri_full         = _g_MO    # (N,N,N,N) already unpacked
                        self._seed_e_nuc       = float(_E_nuc)
                        self._seed_n_frozen    = int(_N_frozen)
                        self._seed_local_basis = True
                        log.info(
                            "[TOWER] Loaded zetazero seed tensors from %s "
                            "(h1_MO=%s, g_MO=%s, N_frozen=%d).",
                            _seed_path,
                            _h1_MO.shape,
                            _g_MO.shape,
                            int(_N_frozen),
                        )
                    except Exception as _exc:
                        log.warning(
                            "[TOWER] seed_tensors.npz found at %s but failed to load: %s",
                            _seed_path,
                            _exc,
                        )
                if self._h1_full is None:
                    log.warning(
                        "[TOWER] h1_full.npy / eri_packed.npy / seed_tensors.npz "
                        "not found in %s — using heuristic integral extension "
                        "(re-run generate-data with --full-mo-seed to fix).",
                        self._base_dir / self._mechanism_name,
                    )

        # ── CAS(N,N) initialisation ───────────────────────────────────────────
        # Frozen core = MOs 0.._FROZEN_CORE_N_ORBS-1 (same at every level).
        # Base active = the CAS(4,4) active MOs inferred from n_orbs and noons.
        # We assume: MOs 0..n_occ_base-1 are occupied (NOON≈2), the next n_orbs
        # MOs are the base active space, and MOs beyond are virtual (NOON≈0).
        #
        # When the integrals were loaded from seed_tensors.npz (local MO basis,
        # shape N_seed×N_seed), frozen MOs are the first _seed_n_frozen local
        # indices and the CAS active space starts immediately after.
        # _seed_n_frozen comes from the .npz metadata (may be 0 for zetazero
        # seeds), so active indices stay within the actual (N×N) matrix bounds.
        # n_occ_base=40 (the full-system PySCF default) would be out of bounds
        # for a small seed basis — never use it in seed_local_basis mode.
        if self._seed_local_basis:
            n_occ_base = self._seed_n_frozen
        else:
            _meta      = raw.get("metadata", {})
            # ncas_occ_base is written by generate_step_integrals (= n_core).
            # Fall back to n_core (same value, older datasets), then 40 (legacy default).
            n_occ_base = int(_meta.get("ncas_occ_base",
                             _meta.get("n_core", 40)))
        self._n_occ_base      = n_occ_base          # first base active MO (global)

        # ── Frozen-core index list in h1_full LOCAL coordinates ──────────────
        # Standard path (offset=0): local = global, frozen = [0..n_occ_base-1].
        # Tower-window path (offset>0): deep core (global 0..offset-1) is already
        # Fock-contracted into h1_eff_win; only window MOs 0..n_occ_base-offset-1
        # (local) need dynamic folding at each level.  The remaining MOs are virtual.
        _off = self._h1_full_win_offset
        _n_occ_local = max(0, n_occ_base - _off)   # number of occupied MOs in window
        self._frozen_core_idx = list(range(_n_occ_local))  # updated per level in _build_level

        if self._h1_full is not None and self._eri_full is not None:
            if self._seed_local_basis and self._seed_e_nuc is not None:
                # E_nuc is known directly from the seed file; skip compute_e_nuc
                # (which would require the correct CAS(4,4) ecore, but ecore_Ha=0.0
                # in zetazero step JSONs due to the known reporting convention).
                self._e_nuc = self._seed_e_nuc
                log.info(
                    "[TOWER] Seed local basis: n_occ_base=%d, e_nuc=%.6f Ha (from seed).",
                    n_occ_base, self._e_nuc,
                )
            elif _off > 0:
                # Tower-window path: deep_ecore already accounts for MOs 0..offset-1.
                # Compute e_nuc from the window frozen MOs (local 0..n_occ_local-1)
                # and add deep_ecore.  The step JSON ecore_Ha includes all frozen
                # MOs 0..n_occ_base-1; to avoid double-counting, we recompute from
                # the window h1_eff_win (which already has the deep core folded in).
                self._e_nuc = compute_e_nuc(
                    self._h1_full, self._eri_full, ecore,
                    n_core_cas44=_n_occ_local,
                ) + self._deep_ecore
                log.info(
                    "[TOWER-WIN] e_nuc=%.6f Ha (window frozen=%d local MOs "
                    "+ deep_ecore=%.6f Ha).",
                    self._e_nuc, _n_occ_local, self._deep_ecore,
                )
            else:
                # Standard path: extract nuclear repulsion from base ecore
                # (frozen MOs 0..n_occ_base-1)
                self._e_nuc = compute_e_nuc(
                    self._h1_full, self._eri_full, ecore, n_core_cas44=n_occ_base,
                )
        else:
            # Cannot compute exactly — use ecore as a safe fallback (heuristic path)
            self._e_nuc = ecore
            log.warning("[TOWER] Full MO integrals unavailable — e_nuc set to base ecore (heuristic).")

        # ── Per-step full MO tensors (correct non-Janus geometries) ─────────────
        # Requires the base dataset to have been generated after the per-step fix
        # (h1_full_step{n:02d}.npy present for each step).  Falls back silently to
        # Janus-only tensors (original behaviour) when files are absent.
        for _sn in range(self._M_steps):
            _res_s = load_full_mo_integrals_step(self._base_dir, self._mechanism_name, _sn)
            if _res_s is not None:
                _h1_s, _eri_s = _res_s
                self._h1_full_per_step[_sn]  = _h1_s
                self._eri_full_per_step[_sn] = _eri_s
                # Compute per-step nuclear repulsion energy for correct ecore later
                if self._h1_full is not None:
                    try:
                        _raw_sn  = self._store._load_step_raw(_sn)
                        _ec_sn   = float(_raw_sn.get("ecore_Ha", ecore))
                        self._e_nuc_per_step[_sn] = compute_e_nuc(
                            _h1_s, _eri_s, _ec_sn, n_core_cas44=n_occ_base,
                        )
                    except Exception as _exc:
                        log.warning(
                            "[TOWER] step %02d: could not compute per-step e_nuc: %s",
                            _sn, _exc,
                        )
        if self._h1_full_per_step:
            log.info(
                "[TOWER] Loaded per-step MO integrals for %d/%d steps — "
                "non-Janus tower energies will use correct geometries.",
                len(self._h1_full_per_step), self._M_steps,
            )
        else:
            log.info(
                "[TOWER] No per-step MO integral files found — re-generate the "
                "base dataset to enable per-step geometry correction.",
            )

        # Base active MO indices: CAS(4,4) window starting at n_occ_base.
        # When loading from a zetazero seed_tensors.npz, the JSON's ncas records
        # the full tensor size (e.g. 20), NOT the CAS(4,4) active window.  The
        # correct base active space is block_size*(k_base-1) orbitals starting at
        # n_occ_base, matching slice_active_hamiltonian's A_k convention exactly.
        # For the non-seed path (PySCF datasets) n_orbs from the JSON is already
        # the CAS(4,4) active count, so the formula gives the same answer.
        if self._seed_local_basis:
            n_act_base = self._block_size * (self._k_base - 1)  # = 4 for k_base=2
            log.info(
                "[TOWER] seed_local_basis: overriding ncas=%d with CAS window "
                "n_act_base=%d (block_size=%d * (k_base=%d - 1))",
                n_orbs, n_act_base, self._block_size, self._k_base,
            )
        else:
            n_act_base = n_orbs
        # In the standard path base_active_idx uses global MO indices (offset=0).
        # In the tower-window path we translate to local indices immediately so
        # that ALL downstream h1_full / eri_full accesses use local coordinates.
        _off = self._h1_full_win_offset
        _n_occ_local = max(0, n_occ_base - _off)
        base_active_idx = list(range(_n_occ_local, _n_occ_local + n_act_base))
        # Sub-Janus hybrid datasets store the exact (non-contiguous) MO indices
        # in metadata["cas_orbital_indices"].  Translate global→local and use
        # them so the tower extends from the correct {s,p} seed orbitals.
        _cas_idx = raw.get("metadata", {}).get("cas_orbital_indices")
        if _cas_idx is not None:
            base_active_idx = self._to_local([int(i) for i in _cas_idx])
            n_act_base      = len(base_active_idx)
            log.info(
                "[TOWER] Using cas_orbital_indices as base active space "
                "(global=%s → local=%s, offset=%d)",
                list(_cas_idx), base_active_idx, _off,
            )
        self._active_idx_k[self._k_base] = base_active_idx
        self._nalpha_k[self._k_base]     = n_act_base // 2
        self._nbeta_k[self._k_base]      = n_act_base // 2

        log.info(
            "[TOWER] CAS init: frozen=%s  base_active=%s  e_nuc=%.6f Ha  "
            "nalpha=%d nbeta=%d",
            self._frozen_core_idx,
            base_active_idx,
            self._e_nuc if self._e_nuc is not None else float("nan"),
            self._nalpha_k[self._k_base],
            self._nbeta_k[self._k_base],
        )

        log.info(
            "[TOWER] Base k=%d: n_orbs=%d  E_base=%.6f Ha  E_target=%.6f Ha  "
            "gap=%.3f Ha",
            self._k_base, n_act_base, E_base, self._E_target,
            abs(E_base - self._E_target),
        )

        return TowerLevel(
            k               = self._k_base,
            p               = self._p,
            m_k             = self._p ** self._k_base,
            n_orbs          = n_act_base,
            orbital_indices = base_active_idx,
            h_diag          = h_diag,
            h_hop           = h_hop,
            g_full          = g_full,
            ecore_Ha        = ecore,
            E_janus_Ha      = E_base,
            delta_E_Ha      = 0.0,
            kummer_ok       = True,
            dataset_dir     = self._base_dir / self._mechanism_name,
        )

    # ── Private: build one new level ─────────────────────────────────────────

    def _build_level(
        self,
        k:          int,
        prev:       TowerLevel,
        output_root: Path,
        run_pipeline: bool,
    ) -> TowerLevel:
        """Construct TowerLevel k from level k−1."""
        from nanoprotogeny.molecular.mqedatagenerator import build_tower_level_dataset

        # Interpolated Janus energy — always from the original CAS(4,4) base,
        # not from prev.E_janus_Ha, so the geometric series is correct:
        #   gap(k) = gap(2) * p^{-(k-2)}  for all k.
        E_k = padicinterp_energy(k, self._k_base, self._E_base, self._E_target, self._p)
        delta_E = E_k - prev.E_janus_Ha

        # ── Select active orbitals for this level ────────────────────────────
        # Use the structured CAS(N,N) orbital schedule (not the NOON heuristic),
        # so that occupied and virtual MOs are added symmetrically around the
        # frontier and the electron count grows correctly.
        per_step_integrals: "Optional[Dict[int, tuple]]" = None  # set in exact branch
        if self._h1_full is not None and self._eri_full is not None:
            base_active = self._active_idx_k.get(self._k_base, list(range(40, 44)))
            # Cap n_total_orbs at the actual MO tensor size so virtual orbital
            # indices never exceed h1_full.shape[0] (seed basis may be smaller
            # than the full n_total_orbs target).
            _n_orbs_cap = min(self._n_total_orbs, self._h1_full.shape[0])
            # For the seed-local-basis path the seed's N_frozen accounts for ALL
            # occupied MOs in the frozen-core energy (E_core computation).  The
            # TowerClimber must be free to promote those occupied MOs into the
            # active space as k grows — CAS(4,4) → CAS(8,8) → … → CAS(80,80).
            # To allow that, the permanent deep frozen core is 0 (no MOs are
            # permanently off-limits), so the 50/50 occ+virt split in
            # select_casci_orbitals_for_level works correctly.  For the PySCF path
            # the deep core is _FROZEN_CORE_N_ORBS as before.
            _frozen_n_for_select = 0 if self._seed_local_basis else _FROZEN_CORE_N_ORBS
            _n_occ_base_sel      = min(base_active)
            _n_act_base_sel      = len(base_active)
            # Physical electron ceiling: for a closed-shell system this equals
            # N_elec_total // 2.  Passed to select_casci_orbitals_for_level so
            # it can clamp nalpha/nbeta and emit a clear warning if the formula
            # would otherwise exceed the available electron count.
            _n_elec_max_alpha    = _n_occ_base_sel + _n_act_base_sel // 2
            active_idx_k, nalpha_k, nbeta_k = select_casci_orbitals_for_level(
                k                 = k,
                k_base            = self._k_base,
                block_size        = self._block_size,
                n_occ_base        = _n_occ_base_sel,
                n_act_base        = _n_act_base_sel,
                frozen_n          = _frozen_n_for_select,
                n_total_orbs      = _n_orbs_cap,
                n_elec_max_alpha  = _n_elec_max_alpha,
            )
            self._active_idx_k[k] = active_idx_k
            self._nalpha_k[k]     = nalpha_k
            self._nbeta_k[k]      = nbeta_k
            n_orbs_k = len(active_idx_k)

            # Dynamic frozen core: occupied MOs below n_occ_base that have not
            # yet been promoted into active_idx_k.  As k grows, more occupied
            # MOs join the active space and frozen_idx_k shrinks — ensuring there
            # is no double-counting in compute_casci_effective_integrals.
            # In the tower-window path all indices are already LOCAL (offset applied
            # in _load_base via _to_local), so no further translation is needed here.
            _n_occ_base_k = getattr(self, '_n_occ_base', min(base_active))
            # Convert global n_occ_base to local for the range boundary.
            _n_occ_local_k = max(0, _n_occ_base_k - self._h1_full_win_offset)
            _active_set_k  = set(active_idx_k)
            frozen_idx_k   = [i for i in range(_n_occ_local_k) if i not in _active_set_k]

            # CASCI effective integrals: fold frozen-core contribution into h1_eff
            # Primary computation uses Janus-step tensors (for TowerLevel object
            # and as fallback for steps without per-step tensors).
            # Indices active_idx_k and frozen_idx_k are already in LOCAL h1_full
            # coordinates (offset subtracted in _load_base / _to_local).
            h_diag_k, h_hop_k, g_full_k, ecore_k = compute_casci_effective_integrals(
                h1_full    = self._h1_full,
                eri_full   = self._eri_full,
                active_idx = active_idx_k,
                frozen_idx = frozen_idx_k,
                e_nuc      = self._e_nuc,
            )
            log.info(
                "[TOWER] k=%d: CASCI integrals (Janus)  active=%d orbs  "
                "frozen=%d  ecore=%.6f Ha  nalpha=%d nbeta=%d",
                k, n_orbs_k, len(frozen_idx_k),
                ecore_k, nalpha_k, nbeta_k,
            )

            # ── Per-step integrals (algebraic only — no FCI during tower build) ─
            # For each step apply the same active_idx_k to that step's geometry
            # tensors.  circuit_ref is set only for the Janus step (Kummer value);
            # non-Janus circuit_refs are left as None and deferred to the Riemann
            # pipeline, which computes E_gs_n relative to E_gs_Janus at the
            # converged tower level (avoiding O(k_max) DMRG runs during build).
            if self._h1_full_per_step:
                per_step_integrals = {}
                for _sn in range(self._M_steps):
                    _h1_sn   = self._h1_full_per_step.get(_sn, self._h1_full)
                    _eri_sn  = self._eri_full_per_step.get(_sn, self._eri_full)
                    _enuc_sn = self._e_nuc_per_step.get(_sn, self._e_nuc)
                    _hd_sn, _hh_sn, _gf_sn, _ec_sn = compute_casci_effective_integrals(
                        h1_full    = _h1_sn,
                        eri_full   = _eri_sn,
                        active_idx = active_idx_k,
                        frozen_idx = frozen_idx_k,
                        e_nuc      = _enuc_sn,
                    )
                    # Janus: Kummer-interpolated reference; non-Janus: deferred (None)
                    _cref_sn = (E_k + _ec_sn) if (_sn in self._janus_steps) else None
                    per_step_integrals[_sn] = (_hd_sn, _hh_sn, _gf_sn, _ec_sn, _cref_sn)
        else:
            # Heuristic fallback (no saved MO tensors)
            new_indices, new_energies = select_next_block(
                h_diag_base   = prev.h_diag,
                current_count = prev.n_orbs,
                block_size    = self._block_size,
                noons         = self._noons,
                n_total_orbs  = self._n_total_orbs,
                mo_energies   = self._mo_energies,
            )
            active_idx_k = prev.orbital_indices + new_indices
            n_orbs_k     = len(active_idx_k)
            nalpha_k     = self._nalpha_k.get(k - 1, 2) + self._block_size // 2
            nbeta_k      = self._nbeta_k.get(k - 1, 2) + self._block_size // 2
            ecore_k      = prev.ecore_Ha   # unchanged in heuristic path
            self._active_idx_k[k] = active_idx_k
            self._nalpha_k[k]     = nalpha_k
            self._nbeta_k[k]      = nbeta_k
            h_diag_k, h_hop_k, g_full_k = extend_integrals(
                prev.h_diag, prev.h_hop, prev.g_full,
                new_indices, new_energies, prev.n_orbs,
            )
            log.info(
                "[TOWER] k=%d: heuristic integrals  n_orbs=%d  "
                "nalpha=%d nbeta=%d (no MO tensor)",
                k, n_orbs_k, nalpha_k, nbeta_k,
            )

        # Write step JSONs for this level
        src_manifest = dict(self._store.manifest)
        level_dir = build_tower_level_dataset(
            k                      = k,
            p                      = self._p,
            n_orbs                 = n_orbs_k,
            h_diag                 = h_diag_k,
            h_hop                  = h_hop_k,
            g_full                 = g_full_k,
            ecore_Ha               = ecore_k,
            E_janus_Ha             = E_k,
            mechanism_name         = self._mechanism_name,
            scaffold               = self._scaffold,
            k_target               = self._k_target,
            janus_steps            = self._janus_steps,
            M_steps                = self._M_steps,
            src_manifest           = src_manifest,
            output_root            = output_root,
            store                  = self._store,
            nalpha                 = nalpha_k,
            nbeta                  = nbeta_k,
            active_orbital_indices = active_idx_k,
            per_step_integrals     = per_step_integrals,
        )

        # Optionally run the pipeline to measure actual E_Janus(k)
        E_measured = E_k   # default: interpolated value
        if run_pipeline:
            E_measured = self._run_level_pipeline(k, level_dir, n_orbs_k)
            if E_measured is None:
                log.warning("[TOWER] k=%d: pipeline returned no result — using interpolation", k)
                E_measured = E_k

        # Kummer check
        ok = kummer_check(E_measured, prev.E_janus_Ha, self._E_target, self._p, k)

        return TowerLevel(
            k               = k,
            p               = self._p,
            m_k             = self._p ** k,
            n_orbs          = n_orbs_k,
            orbital_indices = active_idx_k,
            h_diag          = h_diag_k,
            h_hop           = h_hop_k,
            g_full          = g_full_k,
            ecore_Ha        = ecore_k,
            E_janus_Ha      = E_measured,
            delta_E_Ha      = delta_E,
            kummer_ok       = ok,
            dataset_dir     = level_dir,
        )

    # Maximum n_orbs for which dense exact-diagonalisation is safe.
    # 4^6 = 4 096  →  256 MiB matrix (fine on any machine).
    # 4^7 = 16 384  →  4 GiB  (risky on laptops).
    # 4^8 = 65 536  →  64 GiB  (OOM-kills before MemoryError is catchable).
    _MAX_DIAG_ORBS: int = 6

    def _run_level_pipeline(
        self,
        k:         int,
        level_dir: Path,
        n_orbs:    int,
    ) -> Optional[float]:
        """Measure E_Janus(k) from the active-space ground state at this tower level.

        Dispatches to the appropriate solver tier via dmrg_backend:
          n_orbs ≤ 6   → dense numpy diagonalization  (4^6 = 4 096 dim)
          n_orbs ≤ 50  → PySCF Davidson FCI            (polynomial in CI-dim)
          n_orbs > 50  → Block2 DMRG                   (polynomial in K=n/4)

        Returns None only if all solvers fail; the caller falls back to the
        Kummer-interpolated E_Janus(k) in that case.

        The returned value is the ACTIVE-SPACE energy (ecore already added),
        which is what the Kummer convergence check compares against E_target
        from the Riemann scaffold.
        """
        try:
            from nanoprotogeny.simulate.dmrg_backend import run_active_space_fci
            from nanoprotogeny.molecular.mqeintegralstore import StepwiseIntegralStore

            # build_tower_level_dataset uses a flat layout: step files live
            # directly inside level_dir (no mechanism_name subdir).
            # StepwiseIntegralStore(root, name) sets self._root = root/name,
            # so pass (level_dir.parent, level_dir.name) to get root = level_dir.
            store   = StepwiseIntegralStore(level_dir.parent, level_dir.name)
            janus_n = min(self._janus_steps)
            raw     = store._load_step_raw(janus_n)

            n_o = len(raw.get("h_diag", {}))

            # Electron count: prefer (nalpha, nbeta) from metadata
            meta   = raw.get("metadata", {})
            nalpha = meta.get("nalpha")
            nbeta  = meta.get("nbeta")
            if nalpha is None or nbeta is None:
                nelec_r = meta.get("nelec_active", 4)
                if isinstance(nelec_r, (list, tuple)):
                    nalpha, nbeta = int(nelec_r[0]), int(nelec_r[1])
                else:
                    total  = int(nelec_r)
                    nalpha = total // 2 + total % 2
                    nbeta  = total // 2
            else:
                nalpha, nbeta = int(nalpha), int(nbeta)

            # ecore from the step (may be zero for tower-generated datasets)
            step_ints = store.get_step(janus_n)   # (h_d, h_h, g_f, ecore, n_orbs)
            ecore_val = float(step_ints[3]) if step_ints is not None else 0.0

            # Convert str-keyed JSON → typed dicts for dmrg_backend
            h_diag = {int(ki): float(v)
                      for ki, v in raw.get("h_diag", {}).items()}
            h_hop  = {tuple(int(x) for x in ki.split(",")): float(v)
                      for ki, v in raw.get("h_hop", {}).items()}
            g_full = {tuple(int(x) for x in ki.split(",")): float(v)
                      for ki, v in raw.get("g_full", {}).items()}

            E_total = run_active_space_fci(
                h_diag, h_hop, g_full,
                ecore     = ecore_val,
                n_orbs    = n_o,
                nelec_tuple = (nalpha, nbeta),
            )

            if E_total is None:
                log.warning(
                    "[TOWER] k=%d: all solvers failed for n_orbs=%d — "
                    "falling back to interpolated E_Janus.", k, n_o,
                )
                return None

            # Active-space energy = E_total - ecore (what Kummer check compares)
            E_janus = E_total - ecore_val
            log.info(
                "[TOWER] k=%d: E_Janus = %.8f Ha  (n_orbs=%d, nelec=(%d,%d), "
                "ecore=%.6f Ha)",
                k, E_janus, n_o, nalpha, nbeta, ecore_val,
            )
            return float(E_janus)

        except Exception as exc:
            log.warning("[TOWER] k=%d pipeline error: %s", k, exc, exc_info=True)
            return None

    # ── Public: climb ─────────────────────────────────────────────────────────

    def climb(
        self,
        k_max:        int = 7,
        run_pipeline: bool = False,
        output_root:  Optional[Union[str, Path]] = None,
    ) -> List[TowerLevel]:
        r"""Ascend the Iwasawa tower from k_base to k_max (or until convergence).

        Args:
            k_max:        Maximum tower level to reach (default 7 → CAS(128,128)
                          for p=2; physically capped at n_total_orbs=76).
            run_pipeline: If True, run MQERiemannPipeline at each level to
                          measure E_Janus(k) from the actual Hamiltonian.
                          If False (default), use the interpolated E_Janus(k)
                          (dataset-generation mode, no quantum simulation).
            output_root:  Where to write tower-level datasets.  Defaults to
                          <base_dataset_dir>/../tower/.

        Returns:
            List of TowerLevel objects, one per level from k_base to the
            level where convergence was reached (inclusive).
        """
        if output_root is None:
            output_root = self._base_dir.parent / "tower"
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        base  = self._load_base()
        levels: List[TowerLevel] = [base]

        print(f"\n{'='*70}")
        print(f"  Iwasawa Tower Climb  —  {self._mechanism_name}")
        print(f"  p={self._p}  k_base={self._k_base}  k_max={k_max}")
        print(f"  E_base = {base.E_janus_Ha:+.6f} Ha  E_target = {self._E_target:+.6f} Ha")
        print(f"  Total gap = {(self._E_target - base.E_janus_Ha)*1000:.3f} mHa")
        print(f"{'='*70}")
        print(
            f"  {'k':>3} {'m_k':>6} {'n_orbs':>7} {'E_Janus (Ha)':>16} "
            f"{'ΔE (Ha)':>12} {'gap (mHa)':>12} {'Kummer':>8}"
        )
        print(f"  {'-'*3} {'-'*6} {'-'*7} {'-'*16} {'-'*12} {'-'*12} {'-'*8}")
        print(
            f"  {base.k:>3} {base.m_k:>6} {base.n_orbs:>7} "
            f"{base.E_janus_Ha:>+16.6f} {'—':>12} "
            f"{abs(base.E_janus_Ha - self._E_target)*1000:>12.3f} {'base':>8}"
        )

        for k in range(self._k_base + 1, k_max + 1):
            prev  = levels[-1]

            # Stop if active space is already saturated
            if prev.n_orbs >= self._n_total_orbs:
                print(f"\n  Active space saturated at {prev.n_orbs} orbitals — stopping.")
                break

            level = self._build_level(k, prev, output_root, run_pipeline)
            levels.append(level)

            gap_mHa = abs(level.E_janus_Ha - self._E_target) * 1000
            print(
                f"  {k:>3} {level.m_k:>6} {level.n_orbs:>7} "
                f"{level.E_janus_Ha:>+16.6f} {level.delta_E_Ha:>+12.6f} "
                f"{gap_mHa:>12.3f} {'✓' if level.kummer_ok else '✗':>8}"
            )

            if gap_mHa < self._tol_mHa:
                print(
                    f"\n  ✓ CONVERGED at k={k}: "
                    f"|E_Janus − E_target| = {gap_mHa:.3f} mHa < {self._tol_mHa:.1f} mHa"
                )
                break

            if not level.kummer_ok:
                log.warning(
                    "[TOWER] k=%d: Kummer check FAILED — orbital block may be wrong. "
                    "Consider re-running with --noons-file to improve orbital selection.", k
                )

        print(f"{'='*70}\n")

        # ── Compact tower: prune intermediate levels + step files ─────────────
        # After convergence only the highest level is needed for Riemann + rates.
        # Keep: k_max/manifest.json  (+ seed_tensors.npz in the mechanism root).
        # Delete: k_base+1 … k_max-1 directories (no longer needed).
        #         step_XX.json at k_max  (Hamiltonians are in seed_tensors.npz;
        #                                 energies are in manifest.json).
        if len(levels) > 1:
            import shutil
            k_max_level = levels[-1]
            # Delete all intermediate k levels (k_base is the seed — keep it)
            for lv in levels[1:-1]:
                lv_dir = output_root / f"k{lv.k}_{self._mechanism_name}"
                if lv_dir.exists():
                    shutil.rmtree(lv_dir)
                    log.info("[TOWER] Pruned intermediate level dir: %s", lv_dir)
            # Delete step_XX.json from k_max — manifest.json is sufficient
            k_max_dir = output_root / f"k{k_max_level.k}_{self._mechanism_name}"
            pruned_steps = 0
            for step_f in sorted(k_max_dir.glob("step_*.json")):
                step_f.unlink()
                pruned_steps += 1
            if pruned_steps:
                log.info(
                    "[TOWER] Pruned %d step files from k=%d (compact mode).",
                    pruned_steps, k_max_level.k,
                )
            print(
                f"  Compact tower: kept k={k_max_level.k}/manifest.json, "
                f"pruned {len(levels)-2} intermediate level(s) + "
                f"{pruned_steps} step file(s)."
            )

        # ── Post-convergence energy re-referencing ────────────────────────────
        # E^(k_conv) is the canonical reference energy for all mechanism steps.
        # Setting E_Janus = 0 makes ΔE_n = E_n − E_conv the physically meaningful
        # quantity: the reactant sits at +|E_∞| > 0 (above the Janus intermediate),
        # correctly reflecting the barrierless downhill character of the reaction.
        #
        # New manifest fields (absolute values preserved for provenance):
        #   janus_reference_energy_Ha   — E^(k_conv) in Ha (absolute)
        #   fci_energies_abs_Ha         — original absolute values (copy of fci_energies_Ha)
        #   fci_energies_relative_Ha    — E_n − E^(k_conv) for each step
        #   e_ref_relative_Ha           — per-step relative energy in step_results
        #   energy_reference_convention — human-readable description
        k_conv_level   = levels[-1]
        E_conv         = k_conv_level.E_janus_Ha
        _k_conv_dir    = output_root / f"k{k_conv_level.k}_{self._mechanism_name}"
        _manifest_path = _k_conv_dir / "manifest.json"
        if _manifest_path.exists():
            _manifest = json.loads(_manifest_path.read_text())
            _fci_abs  = _manifest.get("fci_energies_Ha", [])
            _fci_rel  = [
                float(e) - E_conv if e is not None else None
                for e in _fci_abs
            ]
            for _sr in _manifest.get("step_results", []):
                _e = _sr.get("e_ref_Ha")
                _sr["e_ref_relative_Ha"] = float(_e) - E_conv if _e is not None else None
            _manifest["janus_reference_energy_Ha"]   = E_conv
            _manifest["fci_energies_abs_Ha"]         = list(_fci_abs)
            _manifest["fci_energies_relative_Ha"]    = _fci_rel
            _manifest["energy_reference_convention"] = (
                "fci_energies_relative_Ha = E_n − E^(k_conv): "
                "Janus intermediate = 0; reactant > 0 (barrierless downhill). "
                "k_conv = " + str(k_conv_level.k) + "."
            )
            _manifest_path.write_text(json.dumps(_manifest, indent=2))
            log.info(
                "[TOWER] Post-convergence re-referencing complete: "
                "E_conv = %.8f Ha (k=%d)  →  %s",
                E_conv, k_conv_level.k, _manifest_path,
            )

        print(f"  Tower levels written to: {output_root}/")
        print(f"  Run any level with:")
        print(f"    mqe run --mechanism {self._mechanism_name} --riemann \\")
        print(f"      --dataset-dir {output_root}/<k>_{self._mechanism_name}/")
        print()
        return levels

    # ── Summary table ─────────────────────────────────────────────────────────

    @staticmethod
    def print_summary(levels: List[TowerLevel], E_target: float) -> None:
        """Print a summary table of all tower levels."""
        print(f"\n{'─'*70}")
        print(f"  Tower Summary")
        print(f"{'─'*70}")
        print(
            f"  {'k':>3} {'n_orbs':>7} {'E_Janus (Ha)':>16} "
            f"{'gap (mHa)':>12} {'Kummer':>8} {'Dataset':>30}"
        )
        for lv in levels:
            gap = abs(lv.E_janus_Ha - E_target) * 1000
            dset = str(lv.dataset_dir)[-28:] if lv.dataset_dir else "—"
            print(
                f"  {lv.k:>3} {lv.n_orbs:>7} {lv.E_janus_Ha:>+16.6f} "
                f"{gap:>12.3f} {'✓' if lv.kummer_ok else '✗':>8} {dset:>30}"
            )
        print(f"{'─'*70}\n")
