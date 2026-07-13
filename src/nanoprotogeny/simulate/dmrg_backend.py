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
r"""Active-space FCI / DMRG solver for the Iwasawa tower climber.

Provides polynomial-scaling ground-state energy calculation for dilute
active spaces (nelec ≪ n_orbs) arising from the block-sequential MPO
construction described in subsec:block_sequential.

Scaling for CAS(4,4)-seeded tower levels (nelec=4):
  k=7  → n_orbs=24,  CI-dim ≈ 76 k,    CI-vec ~  0.6 MB   (milliseconds)
  k=18 → n_orbs=68,  CI-dim ≈ 5.2 M,   CI-vec ~   42 MB   (few seconds)
  k=30 → n_orbs=116, CI-dim ≈ 44.5 M,  CI-vec ~  356 MB   (~30 s)
  k=50 → n_orbs=196, CI-dim ≈ 365 M,   CI-vec ~ 2.9 GB    (Block2 DMRG)

The solver hierarchy:
  1. Dense numpy diag          — n_orbs ≤ DENSE_MAX (6)
  2. PySCF Davidson FCI        — n_orbs ≤ FCI_MAX (50, configurable)
  3. Block2 DMRG via pyscf     — n_orbs > FCI_MAX (if block2 installed)
  [4. Fallback: returns None   — if neither PySCF nor Block2 available]

The UJCT topological confinement (prop:ujct_mps_invariance) holds at
every solver tier: Re(s)=1/2 placement depends only on phase closure
and Z_m group structure, not on the bond dimension or FCI truncation.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np


log = logging.getLogger(__name__)

# ── Tier thresholds ───────────────────────────────────────────────────────────
DENSE_MAX:   int = 6    # n_orbs ≤ DENSE_MAX  → dense numpy diag (4^6 = 4096)
FCI_MAX:     int = 50   # n_orbs ≤ FCI_MAX    → PySCF Davidson FCI
# n_orbs > FCI_MAX → Block2 DMRG (chi=200 default) or raises RuntimeError


# ── Integral reconstruction ───────────────────────────────────────────────────

def reconstruct_integrals(
    h_diag: Dict[int, float],
    h_hop:  Dict[Tuple[int, int], float],
    g_full: Dict[Tuple[int, int, int, int], float],
    n_orbs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct (h1, eri) arrays from the JSON-encoded sparse dicts.

    h_diag : {p: h_pp}
    h_hop  : {(p,q): h_pq}   one canonical entry per pair (p < q)
    g_full : {(p,q,r,s): val} ERI in chemist's notation (pq|rs);
             may contain only canonical entries — the 8-fold symmetry is
             applied here to fill the full tensor.

    Returns
    -------
    h1  : (n_orbs, n_orbs) float64 — one-electron integrals (symmetric)
    eri : (n_orbs, n_orbs, n_orbs, n_orbs) float64 — two-electron integrals
          in chemist's notation, 8-fold symmetry filled
    """
    h1 = np.zeros((n_orbs, n_orbs), dtype=float)
    for p, h_pp in h_diag.items():
        h1[int(p), int(p)] = float(h_pp)
    for key, h_pq in h_hop.items():
        if isinstance(key, str):
            p, q = (int(x) for x in key.split(","))
        else:
            p, q = int(key[0]), int(key[1])
        h1[p, q] = float(h_pq)
        h1[q, p] = float(h_pq)   # symmetric for real orbitals

    eri = np.zeros((n_orbs, n_orbs, n_orbs, n_orbs), dtype=float)
    for key, val in g_full.items():
        if isinstance(key, str):
            p, q, r, s = (int(x) for x in key.split(","))
        else:
            p, q, r, s = (int(x) for x in key)
        v = float(val)
        # 8-fold symmetry of real-orbital chemist's-notation ERIs
        eri[p, q, r, s] = v; eri[q, p, s, r] = v
        eri[r, s, p, q] = v; eri[s, r, q, p] = v
        eri[p, q, s, r] = v; eri[q, p, r, s] = v
        eri[r, s, q, p] = v; eri[s, r, p, q] = v

    return h1, eri


# ── Tier-2: PySCF Davidson FCI ────────────────────────────────────────────────

def _run_pyscf_fci(
    h1:      np.ndarray,
    eri:     np.ndarray,
    n_orbs:  int,
    nelec:   Tuple[int, int],
    ecore:   float = 0.0,
    max_memory_mb: int = 8000,
    conv_tol:      float = 1e-10,
    max_cycle:     int = 200,
) -> float:
    """Run PySCF Davidson FCI and return the active-space ground-state energy.

    The active-space energy does NOT include ecore; add it back at the
    call site if the total energy is needed.

    Parameters
    ----------
    h1            : (n, n) one-electron integrals
    eri           : (n, n, n, n) two-electron integrals, chemist's notation
    n_orbs        : number of spatial orbitals in the active space
    nelec         : (n_alpha, n_beta) tuple
    ecore         : core energy (added to the returned value)
    max_memory_mb : memory limit for PySCF FCI in MB
    conv_tol      : energy convergence threshold for Davidson
    max_cycle     : maximum Davidson iterations

    Returns
    -------
    E_gs : float — active-space ground-state energy (includes ecore)
    """
    from pyscf import fci as pyscf_fci

    # Pack ERI to 8-fold symmetry (reduces memory ~8×)
    try:
        from pyscf import ao2mo
        eri_packed = ao2mo.restore(8, eri, n_orbs)
    except Exception:
        eri_packed = eri   # fall back to full tensor

    cisolver = pyscf_fci.direct_spin1.FCI()
    cisolver.max_memory = max_memory_mb
    cisolver.conv_tol   = conv_tol
    cisolver.max_cycle  = max_cycle
    cisolver.verbose    = 0   # suppress PySCF stdout

    e, _ = cisolver.kernel(h1, eri_packed, n_orbs, nelec, ecore=ecore)
    return float(e)


# ── Tier-3: Block2 DMRG via PySCF ─────────────────────────────────────────────

def _run_block2_dmrg(
    h1:        np.ndarray,
    eri:       np.ndarray,
    n_orbs:    int,
    nelec:     Tuple[int, int],
    ecore:     float = 0.0,
    chi_max:   int = 500,
    n_sweeps:  int = 20,
    max_memory_mb: int = 16000,
) -> float:
    """Run Block2 DMRG via pyscf.dmrgscf and return the ground-state energy.

    Requires the `block2` Python package (pip install block2).
    Falls back gracefully with a RuntimeError if not available.

    chi_max   : maximum bond dimension (500 is safe for nelec=4 area-law systems)
    n_sweeps  : number of DMRG sweeps
    """
    try:
        from pyscf import dmrgscf, gto, scf, mcscf
    except ImportError as e:
        raise RuntimeError(
            "block2 / pyscf.dmrgscf not available. "
            "Install with: pip install block2"
        ) from e

    # Build a minimal dummy molecule to host the CASCI object
    mol       = gto.Mole()
    mol.nelectron = sum(nelec)
    mol.spin  = nelec[0] - nelec[1]   # 2S
    mol.verbose = 0
    mol.build(dump_input=False, parse_arg=False)

    # Identity SCF object (no mean field needed; supply integrals directly)
    mf        = scf.RHF(mol)
    mf.e_tot  = ecore
    mf.mo_coeff = np.eye(n_orbs)
    mf.mo_occ   = np.zeros(n_orbs)
    mf.mo_energy = np.diag(h1).copy()

    mc        = mcscf.CASCI(mf, n_orbs, nelec)
    mc.fcisolver = dmrgscf.DMRG(mol, maxM=chi_max, num_thrds=1)
    mc.fcisolver.max_memory   = max_memory_mb
    mc.fcisolver.num_sweeps   = n_sweeps
    mc.fcisolver.twopdm       = False
    mc.fcisolver.verbose      = 0

    # Supply integrals directly
    mc.ao2mo = lambda *a, **kw: None
    mc._scf.with_df = None

    # Run DMRG
    e_tot, e_cas, _, _, _ = mc.kernel(
        mo_coeff=np.eye(n_orbs),
        ci0=None,
        h1eff=h1,
        fcivec=None,
    )
    return float(e_tot)


# ── Public entry point ────────────────────────────────────────────────────────

def run_active_space_fci(
    h_diag:        Dict[int, float],
    h_hop:         Dict[Tuple[int, int], float],
    g_full:        Dict[Tuple[int, int, int, int], float],
    ecore:         float,
    n_orbs:        int,
    nelec_tuple:   Tuple[int, int],
    *,
    fci_max:       int = FCI_MAX,
    chi_max:       int = 500,
    max_memory_mb: int = 8000,
) -> Optional[float]:
    """Compute the active-space ground-state energy for a tower level.

    Dispatches to the appropriate solver tier based on n_orbs:
      ≤ DENSE_MAX  → dense numpy diagonalization (via mqehamiltonian)
      ≤ fci_max    → PySCF Davidson FCI
      > fci_max    → Block2 DMRG (if available)

    The returned energy includes ecore (total active-space energy).
    Returns None only if all solvers fail.

    Parameters
    ----------
    h_diag, h_hop, g_full : sparse integral dicts from StepwiseIntegralStore
    ecore         : core energy from the step file
    n_orbs        : number of active-space spatial orbitals
    nelec_tuple   : (n_alpha, n_beta)
    fci_max       : orbital count above which Block2 DMRG replaces PySCF FCI
    chi_max       : maximum DMRG bond dimension
    max_memory_mb : memory ceiling forwarded to PySCF/Block2
    """
    nalpha, nbeta = nelec_tuple
    nelec_total   = nalpha + nbeta

    log.info(
        "[DMRG] k-level solver: n_orbs=%d, nelec=(%d,%d), ecore=%.6f Ha",
        n_orbs, nalpha, nbeta, ecore,
    )

    # ── Tier 1: dense diag for tiny systems ──────────────────────────────────
    if n_orbs <= DENSE_MAX:
        from nanoprotogeny.molecular.mqehamiltonian import (
            build_qudit_hamiltonian_matrix,
            _project_hamiltonian_to_sector,
            ground_state_from_diagonalization,
        )
        H_full    = build_qudit_hamiltonian_matrix(n_orbs, h_diag, h_hop, g_full)
        H_proj, _ = _project_hamiltonian_to_sector(
            H_full, n_orbs, nelec_total, return_indices=True
        )
        E_gs, _   = ground_state_from_diagonalization(H_proj)
        E_total   = float(E_gs) + ecore
        log.info("[DMRG] Tier 1 (dense): E_gs = %.8f Ha  (ecore added)", E_total)
        return E_total

    # ── Reconstruct integrals ─────────────────────────────────────────────────
    try:
        h1, eri = reconstruct_integrals(h_diag, h_hop, g_full, n_orbs)
    except Exception as exc:
        log.warning("[DMRG] Integral reconstruction failed: %s", exc)
        return None

    # ── Tier 2: PySCF Davidson FCI ────────────────────────────────────────────
    if n_orbs <= fci_max:
        try:
            # CI vector size estimate
            from math import comb
            ci_dim = comb(n_orbs, nalpha) * comb(n_orbs, nbeta)
            ci_mb  = ci_dim * 8 / 2**20
            log.info(
                "[DMRG] Tier 2 (PySCF FCI): CI-dim = C(%d,%d)×C(%d,%d) = %d "
                "(%.1f MB CI-vec)",
                n_orbs, nalpha, n_orbs, nbeta, ci_dim, ci_mb,
            )
            E_total = _run_pyscf_fci(
                h1, eri, n_orbs, (nalpha, nbeta),
                ecore=ecore,
                max_memory_mb=max_memory_mb,
            )
            log.info("[DMRG] Tier 2 (PySCF FCI): E_gs = %.8f Ha", E_total)
            return E_total
        except Exception as exc:
            log.warning("[DMRG] PySCF FCI failed (n_orbs=%d): %s — "
                        "trying Block2 DMRG.", n_orbs, exc)

    # ── Tier 3: Block2 DMRG ──────────────────────────────────────────────────
    try:
        log.info(
            "[DMRG] Tier 3 (Block2 DMRG): n_orbs=%d, chi_max=%d",
            n_orbs, chi_max,
        )
        E_total = _run_block2_dmrg(
            h1, eri, n_orbs, (nalpha, nbeta),
            ecore=ecore,
            chi_max=chi_max,
            max_memory_mb=max_memory_mb,
        )
        log.info("[DMRG] Tier 3 (Block2 DMRG): E_gs = %.8f Ha", E_total)
        return E_total
    except RuntimeError as exc:
        log.error("[DMRG] Block2 DMRG unavailable: %s", exc)
    except Exception as exc:
        log.warning("[DMRG] Block2 DMRG failed: %s", exc)

    log.error(
        "[DMRG] All solvers failed for n_orbs=%d. "
        "Install block2: pip install block2",
        n_orbs,
    )
    return None
