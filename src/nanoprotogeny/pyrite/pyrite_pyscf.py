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
pyrite_pyscf.py
Exports PySCF molecular integrals to femocosim-compatible JSON format.

Usage: poetry run python pyrite_pyscf.py --basis cc-pVDZ --active_orbitals 4 --active_electrons 4
"""
import numpy as np
import json
import argparse
from pyscf import gto, scf, mcscf, ao2mo, fci
from typing import Dict, Tuple

# ==============================================================================
# 1. EXACT FCI SOLVER
# ==============================================================================
def compute_fci_energy(
    h1e: np.ndarray, eri: np.ndarray,
    ncas: int, nelec: int, ecore: float = 0.0
) -> float:
    """Compute exact FCI energy using PySCF's direct_spin1 kernel."""
    e_fci, _ = fci.direct_spin1.kernel(
        h1e, eri, ncas, nelec, ecore=ecore, verbose=0
    )
    return float(e_fci)

# ==============================================================================
# 2. FULL ERI COMPRESSION (8-FOLD SYMMETRY + THRESHOLD SCREENING)
# ==============================================================================
def compress_eri_full(
    eri_full: np.ndarray, ncas: int, threshold: float = 1e-8
) -> Dict[str, float]:
    """
    Compresses the full 4-index ERI tensor using real-orbital 8-fold symmetry.
    Applies magnitude screening and stores only unique canonical keys.
    """
    g_full = {}
    for p in range(ncas):
        for q in range(p, ncas):
            for r in range(ncas):
                for s in range(ncas):
                    val = eri_full[p, q, r, s]
                    if abs(val) < threshold:
                        continue

                    p1, q1 = (p, q) if p <= q else (q, p)
                    r1, s1 = (r, s) if r <= s else (s, r)
                    if (p1, q1) > (r1, s1):
                        p1, q1, r1, s1 = r1, s1, p1, q1

                    g_full[f"({p1},{q1},{r1},{s1})"] = float(eri_full[p, q, r, s])
    return g_full

# ==============================================================================
# 3. CIRCUIT-MATCHED REFERENCE
# ==============================================================================
def compute_circuit_reference_full(
    h_diag: Dict, h_hop: Dict, g_full: Dict,
    ecore: float, dt: float, ncas: int, nelec: int
) -> float:
    """
    Computes the exact FCI ground-state energy using real-valued integrals
    reconstructed from the exported dictionaries, for use as the QPE reference.

    Fixes applied vs. previous version:
      1. dtype=float throughout — PySCF direct_spin1 raises NotImplementedError
         for complex Hamiltonians; the CASSCF integrals are always real.
      2. nelec is passed explicitly — the old code hardcoded 4, breaking any
         active space other than 4 electrons.
      3. h1 off-diagonal fill is now a plain float assignment — the previous
         version assigned into a complex array which triggered the PySCF error
         even when the imaginary parts were all zero.
      4. dt is retained as a parameter for the Trotter error log and for future
         Trotterized propagator extension; it is not silently ignored.
    """
    # --- Reconstruct real-valued 1e matrix from exported dicts ---------------
    # FIX 1 & 3: dtype=float — PySCF fci.direct_spin1 requires real arrays.
    h1  = np.zeros((ncas, ncas), dtype=float)
    eri = np.zeros((ncas, ncas, ncas, ncas), dtype=float)

    for k, v in h_diag.items():
        h1[int(k), int(k)] = float(v)

    for pq_str, val in h_hop.items():
        p, q = map(int, pq_str.strip("()").split(","))
        h1[p, q] = float(val)
        h1[q, p] = float(val)

    # --- Reconstruct real-valued 2e tensor from compressed g_full ------------
    for key, val in g_full.items():
        p, q, r, s = map(int, key.strip("()").split(","))
        v = float(val)
        # Fill all 8 symmetry-equivalent positions
        for (a, b, c, d) in [
            (p, q, r, s), (q, p, s, r),
            (r, s, p, q), (s, r, q, p),
            (p, q, s, r), (q, p, r, s),
            (r, s, q, p), (s, r, p, q),
        ]:
            eri[a, b, c, d] = v

    # --- Exact FCI solve ------------------------------------------------------
    # FIX 2: use the caller-supplied nelec, not the hardcoded literal 4.
    E_exact, _ = fci.direct_spin1.kernel(
        h1, eri, ncas, nelec, ecore=ecore, verbose=0
    )

    # Log Trotter discretisation bound for diagnostics.
    # Article Proposition 6.1: epsilon_Trotter <= 0.4 mHa for dt <= 0.05 Ha^-1
    print(f"[REFERENCE] dt = {dt:.4f} Ha^-1 | "
          f"Theoretical Trotter bound: <= 0.4 mHa  (Prop. 6.1)")

    return float(E_exact)

# ==============================================================================
# 4. MAIN EXPORT PIPELINE
# ==============================================================================
def export_integrals(
    mol, ncas: int, nelec: int,
    filename: str = "femoco_integrals.json"
):
    """Run CASSCF, extract full integrals, and export validation-ready JSON."""
    print(f"[EXPORT] Initializing CASSCF({nelec},{ncas}) for {mol.nao} AO basis...")

    # 1. RHF reference
    mf = scf.RHF(mol).run()

    # 2. CASSCF optimisation
    cas = mcscf.CASSCF(mf, ncas, nelec)
    try:
        cas.kernel()
    except Exception as e:
        print(f"[ERROR] CASSCF failed: {e}")
        raise

    E_corr_active = cas.e_tot - mf.e_tot
    print(f"[EXPORT] RHF energy:                  {mf.e_tot:.10f} Ha")
    print(f"[EXPORT] CASSCF energy:                {cas.e_tot:.10f} Ha")
    print(f"[EXPORT] Active-space corr energy:     {E_corr_active:.10f} Ha")

    # 3. Extract integrals
    h1, ecore  = cas.h1e_for_cas()
    eri_packed = cas.get_h2eff()
    eri_full   = ao2mo.restore(1, eri_packed, ncas)  # (ncas, ncas, ncas, ncas)

    h_diag = {str(i): float(h1[i, i]) for i in range(ncas)}
    h_hop  = {
        f"({i},{j})": float(h1[i, j])
        for i in range(ncas) for j in range(i + 1, ncas)
    }

    # 4. Full ERI compression
    g_full = compress_eri_full(eri_full, ncas, threshold=1e-8)
    print(f"[EXPORT] Full ERI channels exported: {len(g_full)}  (screened @ 1e-8 Ha)")

    # 5. Circuit-matched reference
    # dt=0.04 is within the article's epsilon_Trotter <= 0.4 mHa budget.
    dt_ref = 0.04
    print(f"\n[REFERENCE] Computing exact circuit-ground energy (dt={dt_ref} Ha^-1)...")
    E_circuit_ref = compute_circuit_reference_full(
        h_diag, h_hop, g_full, float(ecore), dt_ref, ncas, nelec   # FIX 2: pass nelec
    )
    print(f"[REFERENCE] Exact FCI reference energy:  {E_circuit_ref:.10f} Ha")

    # 6. Full-ERI diagnostics (independent cross-check)
    E_fci_full = compute_fci_energy(h1, eri_full, ncas, nelec, ecore=float(ecore))
    print(f"\n[DIAGNOSTIC] Full-ERI FCI energy:        {E_fci_full:.10f} Ha")
    print(f"[DIAGNOSTIC] FCI vs CASSCF delta:        {E_fci_full - cas.e_tot:.2e} Ha  (should be ~0)")
    print(f"[DIAGNOSTIC] Reference vs full-ERI FCI:  {E_circuit_ref - E_fci_full:.2e} Ha  (should be ~0)")

    # 7. JSON export
    data = {
        "h_diag":  h_diag,
        "h_hop":   h_hop,
        "g_full":  g_full,
        "ecore_Ha": float(ecore),
        "exact_fci_energy_Ha":        E_fci_full,
        "circuit_reference_energy_Ha": E_circuit_ref,
        "active_space_corr_energy_Ha": E_corr_active,
        "metadata": {
            "dt_ref_Ha_inv":      dt_ref,
            "screening_threshold": 1e-8,
            "symmetry":           "8-fold real",
            "ncas":               ncas,
            "nelec":              nelec,
            "spin_sector":        "S=0" if (nelec % 2 == 0) else "S=1/2"
        }
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n[SUCCESS] Saved to {filename}")
    print(f"Orbitals: {ncas} | Diagonal: {len(h_diag)} | "
          f"Hopping: {len(h_hop)} | Full ERI: {len(g_full)}")

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Pyrite integrals via PySCF")
    parser.add_argument("--basis",             default="cc-pVDZ")
    parser.add_argument("--active_orbitals",   type=int, default=4)
    parser.add_argument("--active_electrons",  type=int, default=4)
    parser.add_argument("--output",            default="pyrite_integrals.json")
    args = parser.parse_args()

    mol = gto.Mole()
    mol.atom = """
    Fe  0.0  0.0  0.0
    S   1.8  0.0  0.0
    S  -0.9  1.5  0.0
    """
    mol.basis   = args.basis
    mol.charge  = 0
    mol.spin    = 0
    mol.verbose = 3
    mol.build()

    export_integrals(
        mol,
        ncas=args.active_orbitals,
        nelec=args.active_electrons,
        filename=args.output
    )