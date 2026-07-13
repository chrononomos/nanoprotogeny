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
mqeintegralstore.py — MQE Integral Data Loading and Dataset Infrastructure
===========================================================================
Generic data-loading layer for the MQE framework.  Handles all I/O
between on-disk JSON integral datasets (produced by mqedatagenerator.py)
and the typed Python dicts consumed by build_trotter_evolution_circuit
and build_qudit_hamiltonian_matrix.

Functions / Classes
-------------------
_generate_fallback_hamiltonian()
    Generates a minimal physically valid 4-orbital CAS Hamiltonian for
    local testing when no JSON dataset is available.

_parse_step_integrals(data)
    Deserialises one step_XX.json dict into typed integral dicts
    (h_diag, h_hop, g_full, e_core, n_orbitals).  Handles both string-
    keyed and tuple-keyed formats via ast.literal_eval.

StepwiseIntegralStore
    Loads and caches the per-step JSON datasets produced by mqedatagenerator.py
    for one catalytic mechanism.  Validates all step files at construction
    time and exposes a clean API for integral retrieval and MechanismTuple
    reconstruction.

_manifest_to_mechanism_tuple(manifest, store)
    Reconstructs a MechanismTuple from a manifest.json and the corresponding
    StepwiseIntegralStore, assembling per-step A_n/P_n/B_n/nu_n arrays from
    each step_XX.json "mqe_step" block.

Dependencies: pathlib, json, ast, logging, numpy (fallback only),
              nanoprotogeny.molecular.mqemolecules (MechanismTuple).
No cirq, ionq, pyscf, or simulate-layer imports.
"""

from __future__ import annotations

import ast
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from nanoprotogeny.molecular.mqemolecules import MechanismTuple

log = logging.getLogger(__name__)

def _generate_fallback_hamiltonian() -> Tuple[Dict, Dict, Dict, Dict, float, Dict]:
    """Generate a simple but physically valid 4-orbital Hamiltonian for local testing."""
    n = 4
    h_diag = {i: -0.5 + 0.1*i for i in range(n)}
    h_hop = {(0,1): -0.3, (1,2): -0.25, (2,3): -0.2, (0,2): -0.1}
    h_hop.update({(q,p): v for (p,q), v in list(h_hop.items())})
    
    # Simple density-density + exchange-like terms
    g_full = {}
    for p in range(n):
        for r in range(p+1, n):
            g_full[(p,p,r,r)] = 0.5
            g_full[(p,r,r,p)] = 0.1  # exchange
    g_full.update({(s,r,q,p): v for (p,q,r,s), v in list(g_full.items())})
    
    g_coul = {(p,r): v for (p,p,r,r), v in g_full.items() if p < r}
    ecore = -2.5
    metadata = {"ncas": 4, "nelec_active": 4, "nalpha": 2, "nbeta": 2, "mol_name": "fallback_mqe_test"}
    return {}, h_diag, h_hop, g_full, g_coul, ecore, metadata


def _parse_step_integrals(data: Dict) -> Tuple[Dict, Dict, Dict, float, int]:
    r"""Deserialize one step_XX.json into typed integral dicts.

    Returns:
        (h_diag, h_hop, g_full, e_core, n_orbitals)
    """
    # Parse h_diag
    h_diag = {int(k): float(v) for k, v in data["h_diag"].items()}

    # Parse h_hop with robust tuple conversion
    h_hop: Dict[Tuple[int, int], float] = {}
    for k, v in data["h_hop"].items():
        try:
            if isinstance(k, str):
                cleaned = k.strip().replace(" ", "")
                p, q = ast.literal_eval(cleaned)
            else:
                p, q = k
            h_hop[(int(p), int(q))] = float(v)
        except (ValueError, SyntaxError, TypeError) as e:
            log.warning(f"[PARSE] Failed h_hop key {k!r}: {e}")
            continue

    # Parse g_full with robust tuple conversion + diagnostic logging
    g_full: Dict[Tuple[int, int, int, int], float] = {}
    sample_count = 0
    for k, v in data["g_full"].items():
        try:
            if isinstance(k, str):
                cleaned = k.strip().replace(" ", "")
                p, q, r, s = ast.literal_eval(cleaned)
            else:
                p, q, r, s = k
            key = (int(p), int(q), int(r), int(s))
            g_full[key] = float(v)
            # Log first 3 entries for verification
            if sample_count < 3:
                log.debug(f"[PARSE] g_full sample: {key} = {float(v):.8f}")
                sample_count += 1
        except (ValueError, SyntaxError, TypeError) as e:
            log.warning(f"[PARSE] Failed g_full key {k!r}: {e}")
            continue

    e_core = float(data.get("ecore_Ha", 0.0))
    n_orbs = int(data.get("metadata", {}).get("ncas", len(h_diag)))
    
    log.info(f"[PARSE] Loaded: {len(h_diag)} h_diag, {len(h_hop)} h_hop, {len(g_full)} g_full")
    return h_diag, h_hop, g_full, e_core, n_orbs


# ──────────────────────────────────────────────────────────────────────────────
# A2. StepwiseIntegralStore
# ──────────────────────────────────────────────────────────────────────────────

class StepwiseIntegralStore:
    r"""Loads and caches the per-step integral JSON datasets produced by
    mqedatagenerator.py for one mechanism.

    Expected directory layout:
        <dataset_dir>/<mechanism_name>/
            manifest.json           ← mechanism summary + step list
            step_00.json            ← H_{0,1}, H_{0,2}, mqe_step metadata
            step_01.json
            ...
            step_{M-1:02d}.json

    Each step_XX.json is a standard mqeintegrals.py export extended with
    a "mqe_step" block containing step-specific MQE parameters (A_n, P_n,
    B_n, nu_n, is_crossing, etc.).

    On construction the manifest and all step files are validated; missing
    steps raise FileNotFoundError immediately so errors surface before the
    pipeline executes.

    Usage:
        store = StepwiseIntegralStore("mqe_datasets", "nitrogenase_lt")
        h_diag, h_hop, g_full, e_core, n_orbs = store.get_step(n=3)
        mqe_step_meta = store.get_step_meta(n=3)   # raw "mqe_step" dict
        mechanism = store.to_mechanism_tuple()      # rebuild MechanismTuple
    """

    def __init__(self, dataset_dir: Union[str, Path], mechanism_name: str):
        _base = Path(dataset_dir)
        # Prefer <dataset_dir>/<mechanism_name>/manifest.json (zetazeros tower layout).
        # Fall back to <dataset_dir>/manifest.json when the mechanism subdir is absent
        # (hybrid tower layout: manifest lives directly in the level directory).
        if not (_base / mechanism_name / "manifest.json").exists() and \
                (_base / "manifest.json").exists():
            self._root = _base
        else:
            self._root = _base / mechanism_name
        self._name      = mechanism_name
        self._manifest  = self._load_manifest()
        self._M         = self._manifest["M_steps"]
        self._steps: Dict[int, Dict] = {}          # cache: step_n → raw dict
        self._integrals: Dict[int, tuple] = {}     # cache: step_n → parsed

        # Compact mode: step_XX.json files have been pruned after tower build.
        # In compact mode all energy/ecore data is served from manifest.json;
        # Hamiltonian integrals (h_diag/h_hop/g_full) are unavailable (H_AB=0).
        missing = [
            str(self._root / f"step_{n:02d}.json")
            for n in range(self._M)
            if not (self._root / f"step_{n:02d}.json").exists()
        ]
        self._compact = len(missing) == self._M   # all absent → compact
        if missing and not self._compact:
            raise FileNotFoundError(
                f"StepwiseIntegralStore: {len(missing)} step file(s) missing "
                f"for mechanism '{mechanism_name}':\n  "
                + "\n  ".join(missing)
            )
        if self._compact:
            log.info(
                "[StepStore] Compact mode for '%s': step files absent, "
                "serving energies from manifest.json (H_AB unavailable).",
                mechanism_name,
            )
        log.info(
            f"[StepStore] Loaded manifest for '{mechanism_name}': "
            f"M={self._M} steps, m={self._manifest['m_modulus']} (ℤ_{self._manifest['m_modulus']})"
        )

    # ── Public API ───────────────────────────────────────────────────────────

    @property
    def M_steps(self) -> int:
        return self._M

    @property
    def mechanism_name(self) -> str:
        return self._name

    @property
    def manifest(self) -> Dict:
        return self._manifest

    def get_step(self, n: int) -> Tuple[Dict, Dict, Dict, float, int]:
        r"""Return parsed integrals for step n.

        Returns:
            (h_diag, h_hop, g_full, e_core, n_orbitals)
            Types match the signatures expected by build_trotter_evolution_circuit.
        In compact mode (step files pruned) returns empty integral dicts with
        ecore and n_orbs derived from manifest.json.
        """
        if self._compact:
            ecore_n = self._ecore_from_manifest(n)
            n_orbs  = int(self._manifest.get("n_orbitals", 4))
            return {}, {}, {}, ecore_n, n_orbs
        if n not in self._integrals:
            raw = self._load_step_raw(n)
            self._integrals[n] = _parse_step_integrals(raw)
        return self._integrals[n]

    def get_step_meta(self, n: int) -> Dict:
        r"""Return the raw 'mqe_step' metadata dict for step n.

        Contains: step_n, nu_n, A_n, P_n, B_n, is_crossing,
                  delta_CI_Ha, crossing_orbitals, phase_index_k,
                  cumulative_electrons, geometry_label, bondlength_angstrom.
        In compact mode assembles a minimal dict from manifest janus_crossings
        and step_summary.
        """
        if self._compact:
            return self._meta_from_manifest(n)
        raw = self._load_step_raw(n)
        return raw.get("mqe_step", {})

    def get_reference_energy(self, n: int) -> Optional[float]:
        r"""Return the total reference energy for step n (Ha), or None.

        In compact mode derived from manifest step_summary:
          Janus step:     E_janus_Ha + E_core_Ha   (tower Kummer energy + frozen core)
          Non-Janus step: E_seed_Ha                (total CAS energy from seed geometry)
        """
        if self._compact:
            return self._ref_energy_from_manifest(n)
        raw = self._load_step_raw(n)
        e   = raw.get("circuit_reference_energy_Ha")
        return float(e) if e is not None and not isinstance(e, str) else None

    def to_mechanism_tuple(self) -> "MechanismTuple":
        r"""Reconstruct a MechanismTuple from the manifest + step metadata.

        This allows the MQE pipeline to consume the dataset-driven mechanism
        without separately specifying the mechanism parameters.
        """
        return _manifest_to_mechanism_tuple(self._manifest, self)

    def n_orbitals(self) -> int:
        r"""Return the active-space orbital count N from the manifest."""
        return int(self._manifest.get("n_orbitals", 4))

    def log_step_summary(self, n: int) -> None:
        r"""Print a one-line summary of step n's geometry and energy."""
        meta  = self.get_step_meta(n)
        e_ref = self.get_reference_energy(n)
        e_str = f"{e_ref:+.8f} Ha" if e_ref is not None else "N/A"
        label = meta.get("geometry_label", f"step {n}")
        log.info(
            f"  [StepStore] n={n:02d}: {label} | "
            f"E_ref={e_str} | "
            f"ν={meta.get('nu_n', 0)} | "
            f"k^(n)={meta.get('phase_index_k', 0)}"
        )

    # ── Private helpers ──────────────────────────────────────────────────────

    def _seed_ecore(self) -> float:
        """Fixed ecore_Ha used in all step files at this tower level.

        Step files store a single ecore_Ha (= nuclear repulsion at the seed
        geometry, step 0) rather than per-step values.  We recover it from
        the step_summary for step 0 since that geometry IS the seed.
        """
        ss = self._manifest.get("step_summary", [])
        if ss:
            return float(ss[0].get("E_nuc_Ha", 0.0))
        return 0.0

    def _ecore_from_manifest(self, n: int) -> float:
        """Fixed ecore_Ha for any step n — same value for all steps (seed geometry)."""
        return self._seed_ecore()

    def _ref_energy_from_manifest(self, n: int) -> Optional[float]:
        """Total circuit_reference_energy_Ha for step n from manifest.

        Reproduces what build_tower_level_dataset writes into the step file:
          Janus step n:     E_janus_Ha + seed_ecore
          Non-Janus step n: E_seed_Ha(n) + seed_ecore
        so that E_tower = ref - ecore = E_seed_Ha(n) or E_janus_Ha exactly.

        Hybrid tower fallback: when step_summary is absent (compact hybrid
        manifest), returns E_janus_Ha for the Janus step and weyl_pes_Ha[n]
        for non-Janus steps — both already in the Riemann/Weyl energy scale
        (ecore = 0 in compact mode).
        """
        ss        = self._manifest.get("step_summary", [])
        janus_set = set(self._manifest.get("janus_steps", []))
        if not ss:
            # Hybrid tower layout: no step_summary; use hybrid-specific keys.
            return self._ref_energy_hybrid(n, janus_set)
        if n >= len(ss):
            return None
        ecore  = self._seed_ecore()
        rec    = ss[n]
        if n in janus_set:
            e_janus = self._manifest.get("E_janus_Ha")
            if e_janus is None:
                return None
            return float(e_janus) + ecore
        seed = rec.get("E_seed_Ha")
        return (float(seed) + ecore) if seed is not None else None

    def _ref_energy_hybrid(self, n: int, janus_set: set) -> Optional[float]:
        """Reference energy for compact manifests without step_summary.

        Covers three manifest layouts (all compact / no step_summary):

        1. Hybrid tower (has hybrid_protocol):
              Janus step     → E_janus_Ha
              Non-Janus step → weyl_pes_Ha[n] from step0_algebraic
        2. Tower-scaffold / seed-only (has step_results + fci_energies_Ha,
           no hybrid_protocol — generated by write_zetazero_dataset):
              Janus step     → E_janus_Ha (Kummer-converged tower energy)
              Non-Janus step → step_results[n]["e_ref_Ha"]  (Weyl PES / FCI)
        3. Last-resort: fci_energies_Ha[n] directly.

        ecore = 0 for all compact-mode steps (absorbed into tower energy).
        """
        if n in janus_set:
            e = self._manifest.get("E_janus_Ha")
            return float(e) if e is not None else None

        # ── Hybrid tower: weyl_pes_Ha in step0_algebraic ─────────────────
        step0_alg = (self._manifest.get("hybrid_protocol") or {}).get(
            "step0_algebraic", {}
        )
        weyl_pes = step0_alg.get("weyl_pes_Ha", [])
        if n < len(weyl_pes):
            return float(weyl_pes[n])

        # ── Tower-scaffold (seed-only) manifests: step_results[n].e_ref_Ha
        sr = self._manifest.get("step_results", [])
        if n < len(sr):
            e = sr[n].get("e_ref_Ha")
            if e is not None:
                return float(e)

        # ── Last resort: fci_energies_Ha list ────────────────────────────
        fci = self._manifest.get("fci_energies_Ha", [])
        if n < len(fci):
            return float(fci[n])

        return None

    def _meta_from_manifest(self, n: int) -> Dict:
        """Minimal mqe_step dict for step n assembled from manifest (compact mode).

        Supports both zetazeros-tower manifests (step_summary + janus_crossings)
        and compact hybrid-tower manifests (step_results + janus_steps +
        hybrid_protocol).
        """
        ss   = self._manifest.get("step_summary", [])
        jcx  = self._manifest.get("janus_crossings", [])
        rec  = ss[n] if n < len(ss) else {}
        cx   = next((c for c in jcx if c.get("step_n") == n), {})

        # Janus detection: zetazeros tower uses rec["is_janus"]; hybrid tower
        # uses the top-level "janus_steps" array.
        janus_set   = set(self._manifest.get("janus_steps", []))
        is_crossing = bool(rec.get("is_janus", False)) or (n in janus_set)

        # Crossing orbitals: prefer janus_crossings entry; fall back to the
        # hybrid_protocol sub_janus_selection (first two CAS orbital indices).
        if is_crossing and not cx:
            step1  = (self._manifest.get("hybrid_protocol") or {}).get(
                "step1_janus_pyscf", {}
            )
            cas_idx        = step1.get("sub_janus_selection", {}).get(
                "cas_orbital_indices", []
            )
            crossing_orbs  = cas_idx[:2] if len(cas_idx) >= 2 else [0, 1]
        else:
            crossing_orbs = cx.get("crossing_orbitals", [])

        # Bond length / geometry label: zetazeros stores bondlength_angstrom in
        # step_summary; hybrid tower stores geometry strings in step_results.
        bondlen  = rec.get("bondlength_angstrom")
        geo_str  = ""
        if not bondlen:
            sr      = self._manifest.get("step_results", [])
            sr_rec  = sr[n] if n < len(sr) else {}
            geo_str = sr_rec.get("geometry", "")
            # Geometry strings like "FeMoN2_trimer_n04_1.340A" → extract "1.340"
            parts = geo_str.rsplit("_", 1)
            if len(parts) == 2 and parts[1].endswith("A"):
                try:
                    bondlen = float(parts[1][:-1])
                except ValueError:
                    pass

        # nu_n: zetazeros step_summary stores it per-step; hybrid tower stores
        # total_nu in stoichiometry.phase_closure — distribute uniformly.
        nu_n = int(rec.get("nu_n", 0))
        if nu_n == 0 and not ss:
            stoich    = self._manifest.get("stoichiometry") or {}
            pc        = stoich.get("phase_closure", {})
            total_nu  = int(pc.get("total_nu", 0))
            M         = int(self._manifest.get("M_steps", 1))
            nu_n      = total_nu // M  # uniform distribution

        return {
            "step_n":              n,
            "bondlength_angstrom": bondlen,
            "is_crossing":         is_crossing,
            "crossing_orbitals":   crossing_orbs,
            "delta_CI_Ha":         cx.get("delta_CI_mHa", 1.6) / 1000.0,
            "geometry_label": (
                f"BL={bondlen} Å" if bondlen else (geo_str or f"step {n}")
            ),
            "nu_n":                nu_n,
        }

    def _load_manifest(self) -> Dict:
        mp = self._root / "manifest.json"
        if not mp.exists():
            raise FileNotFoundError(
                f"StepwiseIntegralStore: manifest not found at {mp}. "
                f"Run mqedatagenerator.py first:\n"
                f"  python mqedatagenerator.py "
                f"--mechanism {self._name} --output_dir {self._root.parent}"
            )
        with open(mp) as f:
            return json.load(f)

    def _load_step_raw(self, n: int) -> Dict:
        if n not in self._steps:
            p = self._root / f"step_{n:02d}.json"
            with open(p) as f:
                self._steps[n] = json.load(f)
        return self._steps[n]

# ──────────────────────────────────────────────────────────────────────────────
# A3. MechanismLoader — reconstruct MechanismTuple from a StepwiseIntegralStore
# ──────────────────────────────────────────────────────────────────────────────

def _manifest_to_mechanism_tuple(
    manifest: Dict,
    store:    "StepwiseIntegralStore",
) -> "MechanismTuple":
    r"""Reconstruct a MechanismTuple from a manifest.json + step metadata.

    The manifest produced by mqe_dataset.py dataset generator and stores the algebraic
    stoichiometry (M, m, expected_electrons, phase_closure) but not the
    per-step A_n/P_n/B_n/nu_n arrays. Those live in each step_XX.json
    under the "mqe_step" key.

    This function reads all step metadata files and assembles the full
    MechanismTuple so the MQE pipeline can run against dataset integrals
    without separately specifying the mechanism.

    Returns:
        MechanismTuple fully populated from the dataset.
    """
    M    = int(manifest["M_steps"])
    m    = int(manifest["m_modulus"])
    N    = int(manifest["n_orbitals"])
    S    = float(manifest.get("S_target", 0.0))
    name = manifest["mechanism"]
    desc = manifest.get("description", "")

    electron_sets:       List[List[int]] = []
    proton_sets:         List[List[int]] = []
    cofactor_sets:       List[List[int]] = []
    nu_shifts:           List[int]       = []
    crossings:           List[Tuple[int, int, int, float]] = []
    electron_eject_sets:    List[List[int]] = []
    proton_eject_sets:      List[List[int]] = []
    cofactor_decouple_sets: List[List[int]] = []
    nu_decouple_shifts:     List[int]       = []
    photon_absorb_sets:     List[List[int]] = []
    photon_emit_sets:       List[List[int]] = []
    dock_orbitals: List[List[int]] = []

    for n in range(M):
        meta = store.get_step_meta(n)

        electron_sets.append(list(meta.get("A_n", [])))
        proton_sets.append(list(meta.get("P_n", [])))
        cofactor_sets.append(list(meta.get("B_n", [])))
        nu_shifts.append(int(meta.get("nu_n", 0)))
        dock_orbitals.append([])  # not stored in dataset; empty default

        # Reverse / ejection fields (bidirectional PCET)
        electron_eject_sets.append(list(meta.get("A_n_eject", [])))
        proton_eject_sets.append(list(meta.get("P_n_eject", [])))
        cofactor_decouple_sets.append(list(meta.get("B_n_decouple", [])))
        nu_decouple_shifts.append(int(meta.get("nu_decouple_n", 0)))
        photon_absorb_sets.append(list(meta.get("Gamma_n_abs", [])))
        photon_emit_sets.append(list(meta.get("Gamma_n_emit", [])))

        if meta.get("is_crossing", False):
            orbs       = meta.get("crossing_orbitals") or [0, 1]
            delta_ci   = float(meta.get("delta_CI_Ha") or 1.6e-3)
            crossings.append((n, int(orbs[0]), int(orbs[1]), delta_ci))

    return MechanismTuple(
        name                    = name,
        N_orbitals              = N,
        M_steps                 = M,
        m                       = m,
        S_target                = S,
        electron_sets           = electron_sets,
        proton_sets             = proton_sets,
        cofactor_sets           = cofactor_sets,
        nu_shifts               = nu_shifts,
        crossings               = crossings,
        dock_orbitals           = dock_orbitals if any(dock_orbitals) else None,
        electron_eject_sets     = electron_eject_sets,
        proton_eject_sets       = proton_eject_sets,
        cofactor_decouple_sets  = cofactor_decouple_sets,
        nu_decouple_shifts      = nu_decouple_shifts,
        photon_absorb_sets      = photon_absorb_sets,
        photon_emit_sets        = photon_emit_sets,
        description             = desc,
    )


