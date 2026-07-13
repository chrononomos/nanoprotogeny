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
mqeatomicweights.py — Per-atom spectral weights, seed residuals, bond decay
constants, and catalog entry data for the Molecular Arithmetic Protocol (MAP,
alg:molecular_arithmetic).

Sources (tabs/props/defs reference theory documents):
  mqe-heterogeneous-hamiltonian-euler.md:
    tab:per_atom_clock           — j_0^A, m^A, k*, k_min^A per element
    tab:per_atom_kmin            — δ_0^A seed residuals and tower heights
    tab:coupling_qab             — Q_AB, e^{-Q_AB} for FeMoco bond types
    def:per_atom_spectral_weight — χ^A = δ_0^A / δ_0^mol  (eq 713)
    prop:convergence_decomposition — δ_0^mol = Σ_A n_A δ_0^A + δ_0^coupling
  mqe-global-langlands.md:
    subsec:psii_langlands        — PSII χ^A from Boys-function integrals
                                   (thm:analytic_completeness)
  mqe-foundations.md:
    cor:case_iii                 — 4 | m^A for all atoms listed here

Notation for data provenance in this file:
  [THEORY]    — value from theory documents (tab:per_atom_kmin etc.)
  [BOYS]      — from Boys-function nuclear attraction integrals (thm:analytic_completeness)
  [ESTIMATE]  — estimated; requires Boys-function integrals for exact value

All quantities at tower prime p = 2 (Case III).
"""
from math import exp, ceil, log
from typing import Dict, List, NamedTuple, Optional, Tuple

# ---------------------------------------------------------------------------
# Bond key helper — canonical alphabetical pair for symmetric lookup
# Must be defined before BOND_DECAY_Q so call-time evaluation works.
# ---------------------------------------------------------------------------

def _bond_key(a: str, b: str) -> Tuple[str, str]:
    """Canonical alphabetical pair for symmetric bond lookup."""
    return (a, b) if a <= b else (b, a)


# ---------------------------------------------------------------------------
# Per-atom seed residuals δ_0^A (Ha) — tab:per_atom_kmin
#
# δ_0^A = ‖FCI_{CAS(4,4)}(A) − FCI_{full}(A)‖  (Boys-function integrals)
# Error of the seed CAS relative to full active-space FCI for isolated atom A
# (rem:frozen_core_exact).
#
# Provenance column:
#   [THEORY]    — from tab:per_atom_kmin (FeMoco atoms); exact Boys-fn values
#   [ESTIMATE]  — scaled from Boys-fn integrals; update when computed
# ---------------------------------------------------------------------------
ATOMIC_SEED_RESIDUALS_HA: Dict[str, float] = {
    # --- FeMoco cluster atoms [THEORY] ---
    'Fe': 3.5,    # [Ar]3d^6 4s^2  frozen {1s,2s,2p}
    'Mo': 5.0,    # [Kr]4d^5 5s^1  ECP_28
    'S':  0.75,   # [Ne]3s^2 3p^4  frozen {1s,2s,2p}
    'C':  0.12,   # [He]2s^2 2p^2  frozen {1s}
    'N':  0.18,   # [He]2s^2 2p^3  frozen {1s}
    # --- 3d transition metals [ESTIMATE] ---
    # Scale: δ_0^A ∝ n_d^{active} × Z^{1/3}; calibrated against Fe/Mo
    'Ti': 1.6,    # [Ar]3d^2 4s^2  2 active d electrons; k_min=12
    'V':  2.0,    # [Ar]3d^3 4s^2  3 active d electrons; k_min=13
    'Mn': 3.2,    # [Ar]3d^5 4s^2  half-filled, exchange-reduced; k_min=13
    'Co': 3.8,    # [Ar]3d^7 4s^2  7 active d electrons; k_min=14
    'Ni': 4.0,    # [Ar]3d^8 4s^2  near-closed 3d; k_min=14
    'Cu': 1.5,    # [Ar]3d^10 4s^1 closed d, residual 4s correlation; k_min=12
    # --- light non-metals [ESTIMATE] ---
    'O':  0.22,   # [He]2s^2 2p^4  frozen {1s}; slightly > N (0.18); k_min=10
    'H':  0.02,   # 1s^1  minimal correlation; k_min=6
    # --- alkaline earth [ESTIMATE] ---
    'Ca': 0.20,   # [Ar]4s^2  minimal 3d virtual contribution; k_min=9
    # --- heavy metal [ESTIMATE] ---
    'Bi': 3.5,    # [Xe]4f^14 5d^10 6s^2 6p^3  ECP_60; relativistic corr; k_min=14
}

# ---------------------------------------------------------------------------
# Per-atom Frobenius traces / spectral weights χ^A at p = 2
#   χ^A = tr ρ_A(Frob_2) = δ_0^A / δ_0^mol  (def:per_atom_spectral_weight)
#
# IMPORTANT: χ^A is molecule-specific (normalised by that molecule's δ_0^mol).
#   - FeMoco values below are χ^A = δ_0^A / 54.6 [THEORY]
#   - PSII values are from Boys-function integrals directly [BOYS]
#     (subsec:psii_langlands); they are PSII-specific — not portable to other
#     systems containing Mn/Ca/O.
#   - For any system not listed: use compute_spectral_weights().
# ---------------------------------------------------------------------------
ATOMIC_SPECTRAL_WEIGHTS_P2: Dict[str, float] = {
    # FeMoco system [THEORY] — χ^A = δ_0^A / 54.6
    'Fe': 0.064,   # 3.5  / 54.6
    'Mo': 0.092,   # 5.0  / 54.6
    'S':  0.014,   # 0.75 / 54.6
    'C':  0.002,   # 0.12 / 54.6
    'N':  0.003,   # 0.18 / 54.6 (≈)
    # PSII OEC system [BOYS] — subsec:psii_langlands
    # These χ^A are PSII-specific. Do not use for non-PSII systems.
    'Mn': 0.061,
    'Ca': 0.018,
    'O':  0.011,
}

# ---------------------------------------------------------------------------
# Per-atom Iwasawa tower data — tab:per_atom_clock
#
# j_0^A  : first active principal quantum number
# m_A    : local virtual modulus; 4 | m^A certifies Case III (cor:case_iii)
#          = 4 for 3d metals (s,p,d,f active subshells)
#          = 8 for 4d and heavy metals (g orbital enters virtual space)
# k_star : inner Janus level = j_0^A for 3d/4d transition metals
#          (k_star = m_A // 2 holds for Mo; for 3d metals k_star = j_0 = 3)
# k_min  : minimum Iwasawa tower level for Kummer convergence within ε
#          eq:kmin_explicit: k_min^A = 2 + ⌈log_p(δ_0^A / ε)⌉, p=2, ε=1.6 mHa
#
# Provenance: [THEORY] from tab:per_atom_clock; [ESTIMATE] derived from
# ATOMIC_SEED_RESIDUALS_HA via eq:kmin_explicit with estimated δ_0^A.
# ---------------------------------------------------------------------------
class PerAtomTowerData(NamedTuple):
    j0:     int    # first active principal quantum number
    m_A:    int    # local virtual modulus (4 | m_A for all Case III atoms)
    k_star: int    # inner Janus step
    k_min:  int    # minimum Iwasawa tower level (Kummer convergence criterion)


PER_ATOM_TOWER: Dict[str, PerAtomTowerData] = {
    # [THEORY] — tab:per_atom_clock / tab:per_atom_kmin
    'Fe': PerAtomTowerData(j0=3, m_A=4, k_star=3, k_min=14),
    'Mo': PerAtomTowerData(j0=4, m_A=8, k_star=4, k_min=14),  # 5g outer Janus
    'S':  PerAtomTowerData(j0=3, m_A=4, k_star=3, k_min=11),
    'C':  PerAtomTowerData(j0=2, m_A=4, k_star=2, k_min=9),
    'N':  PerAtomTowerData(j0=2, m_A=4, k_star=2, k_min=9),
    # [ESTIMATE] — k_min from eq:kmin_explicit with estimated δ_0^A above
    'Ti': PerAtomTowerData(j0=3, m_A=4, k_star=3, k_min=12),
    'V':  PerAtomTowerData(j0=3, m_A=4, k_star=3, k_min=13),
    'Mn': PerAtomTowerData(j0=3, m_A=4, k_star=3, k_min=13),
    'Co': PerAtomTowerData(j0=3, m_A=4, k_star=3, k_min=14),
    'Ni': PerAtomTowerData(j0=3, m_A=4, k_star=3, k_min=14),
    'Cu': PerAtomTowerData(j0=3, m_A=4, k_star=3, k_min=12),  # closed 3d
    'Ca': PerAtomTowerData(j0=3, m_A=4, k_star=3, k_min=9),
    'O':  PerAtomTowerData(j0=2, m_A=4, k_star=2, k_min=10),
    'H':  PerAtomTowerData(j0=1, m_A=4, k_star=1, k_min=6),
    'Bi': PerAtomTowerData(j0=5, m_A=8, k_star=5, k_min=14),  # 6p active + ECP_60
}

# ---------------------------------------------------------------------------
# Bond Gaussian decay exponents Q_AB — tab:coupling_qab
#
# Q_AB = (α_A · α_B) / (α_A + α_B) · |R_A − R_B|²  (Gaussian product theorem)
# See prop:coupling_decay and eq:bond_euler for role in L^mol(s).
#
# Keys are canonical (alphabetical) pairs via _bond_key().
#
# Effective Gaussian exponents α_eff (bohr⁻²) extracted from FeMoco Q values:
#   Fe=2.0, Mo=1.0, S=0.58, C=1.05, N=0.66
# New-element estimates (same methodology):
#   V=1.9, Mn=2.0, Co=2.0, Ni=2.0, Cu=1.8, Ti=1.7, Ca=0.30, O=0.90, Bi=0.60
#
# Provenance: [THEORY] from tab:coupling_qab; [BOYS] from subsec:psii_langlands;
#             [ESTIMATE] from Gaussian product theorem with estimated α_eff.
# ---------------------------------------------------------------------------
BOND_DECAY_Q: Dict[Tuple[str, str], float] = {
    # --- FeMoco bonds [THEORY] — tab:coupling_qab ---
    _bond_key('Fe', 'S'):   7.99,   # bridging sulfide        e^{-Q} ≈ 3.4e-4
    _bond_key('Fe', 'C'):   9.53,   # interstitial carbide    e^{-Q} ≈ 7.2e-5
    _bond_key('Fe', 'N'):   8.17,   # protein-N ligation      e^{-Q} ≈ 2.8e-4
    _bond_key('S',  'S'):  13.10,   # through-space S…S       e^{-Q} ≈ 2.0e-6
    _bond_key('Fe', 'Fe'): 24.10,   # cubane superexchange    e^{-Q} ≈ 3.4e-11
    _bond_key('Fe', 'Mo'): 17.30,   # Fe–Mo superexchange     e^{-Q} ≈ 3.1e-8
    # --- PSII OEC bonds [BOYS] — subsec:psii_langlands ---
    _bond_key('Mn', 'O'):   3.22,   # μ-oxo bridges (dominant) e^{-Q} ≈ 4.0e-2
    _bond_key('Ca', 'O'):   5.12,   # Ca–O coordinative        e^{-Q} ≈ 6.0e-3
    # --- FeVco bonds [ESTIMATE] — α_V=1.9 bohr^{-2} ---
    _bond_key('Fe', 'V'):  24.40,   # Fe–V in FeVco cofactor  e^{-Q} ≈ 2.5e-11
    _bond_key('V',  'S'):   8.02,   # bridging sulfide (V)    e^{-Q} ≈ 3.3e-4
    _bond_key('V',  'C'):  10.13,   # V–carbide analogue      e^{-Q} ≈ 4.0e-5
    _bond_key('V',  'N'):   8.48,   # V–N protein ligation    e^{-Q} ≈ 2.1e-4
    # --- CODH/ACS A-cluster [ESTIMATE] — α_Ni=2.0 bohr^{-2} ---
    _bond_key('Ni', 'Fe'): 24.10,   # Ni–[4Fe-4S] A-cluster  e^{-Q} ≈ 3.4e-11
    _bond_key('Ni', 'S'):   7.79,   # Ni–S in A-cluster       e^{-Q} ≈ 4.1e-4
    # --- Corrinoid Co-N [ESTIMATE] — α_Co=2.0, α_N=0.66, R=1.9 Å ---
    _bond_key('Co', 'N'):   6.39,   # corrinoid Co–N (pyrrole) e^{-Q} ≈ 1.7e-3
    # --- Cyt c oxidase / Cu electrocatalyst [ESTIMATE] ---
    _bond_key('Cu', 'Cu'): 18.55,   # CuA binuclear Cu–Cu     e^{-Q} ≈ 8.7e-9
    _bond_key('Cu', 'O'):   8.57,   # CuB–peroxo / CO2RR O    e^{-Q} ≈ 1.9e-4
    # --- Photocatalytic N2 fixation (TiO2) [ESTIMATE] — α_Ti=1.7 bohr^{-2} ---
    _bond_key('Ti', 'O'):   8.01,   # TiO2 Ti–O surface bond  e^{-Q} ≈ 3.3e-4
    _bond_key('Ti', 'N'):   7.50,   # Ti–N2 adsorption        e^{-Q} ≈ 5.5e-4
    # --- BiOBr photocatalyst [ESTIMATE] — α_Bi=0.60 bohr^{-2} ---
    _bond_key('Bi', 'O'):   6.50,   # Bi–O in BiOBr layer     e^{-Q} ≈ 1.5e-3
    # --- Assimilatory NR Mo-cofactor [ESTIMATE] — α_Mo=1.0 bohr^{-2} ---
    _bond_key('Mo', 'O'):   4.88,   # Mo=O oxo bond in Moco   e^{-Q} ≈ 7.6e-3
    _bond_key('Mo', 'N'):   6.27,   # Mo–N (NR or photocatal) e^{-Q} ≈ 1.9e-3
    _bond_key('Mo', 'S'):   7.57,   # Mo–S (Moco dithiolate)  e^{-Q} ≈ 5.2e-4
}

# Pre-computed bond amplitude regulators e^{-Q_AB}
BOND_DECAY_FACTOR: Dict[Tuple[str, str], float] = {
    pair: exp(-q) for pair, q in BOND_DECAY_Q.items()
}

# Threshold below which bond amplitude is negligible (Euler factor → 1)
BOND_NEGLIGIBLE_THRESHOLD: float = 1.0e-6

# ---------------------------------------------------------------------------
# FeMoco molecular constants — prop:convergence_decomposition
#   δ_0^mol = Σ_A n_A · δ_0^A  +  δ_0^coupling
#   54.6    =     36.4          +    18.2
# ---------------------------------------------------------------------------
FEMOCO_DELTA0_PER_ATOM: float = 36.4   # Σ_A n_A · δ_0^A  (Ha)
FEMOCO_DELTA0_COUPLING: float = 18.2   # inter-atomic bonding correlation (Ha)
FEMOCO_DELTA0_MOL:      float = 54.6   # δ_0^mol (Ha)

FEMOCO_ELEMENTS: List[Tuple[str, int]] = [('Fe', 7), ('Mo', 1), ('S', 9), ('C', 1)]

# PSII OEC cluster — subsec:psii_langlands
PSII_KMIN_MOL: int = 4
PSII_ELEMENTS: List[Tuple[str, int]] = [('Mn', 4), ('Ca', 1), ('O', 5)]

# ---------------------------------------------------------------------------
# Photocatalytic N2 fixation — alternative active-site models
# Entry 14/15 in catalog.md covers a class of catalysts; here we provide
# representative active-site compositions for TiO2, MoS2, and BiOBr classes.
# Choose via k_0_override and element_multiset substitution in run_map_entry().
# ---------------------------------------------------------------------------
PHOTO_N2_ELEMENTS_TIO2:  List[Tuple[str, int]] = [('Ti', 2), ('O', 4), ('N', 2)]
PHOTO_N2_ELEMENTS_MOS2:  List[Tuple[str, int]] = [('Mo', 2), ('S', 4), ('N', 2)]
PHOTO_N2_ELEMENTS_BIOBR: List[Tuple[str, int]] = [('Bi', 2), ('O', 2), ('N', 2)]

# ---------------------------------------------------------------------------
# Global MQE parameters
# ---------------------------------------------------------------------------
TOWER_PRIME: int = 2
PRECISION_BUDGET_MHA: float = 1.6
PRECISION_BUDGET_HA: float = PRECISION_BUDGET_MHA * 1.0e-3   # 1.6 mHa

# ---------------------------------------------------------------------------
# Catalog entries — all 15 entries from catalog.md
#
# Keys per entry:
#   name               : human-readable label
#   Ne, Mcof           : stoichiometric parameters for Phase 1 (alg:molecular_arithmetic)
#   element_multiset   : [(element, count), ...] for Phase 2–3
#   bond_network       : [(elem_a, elem_b), ...] bond types present in active site
#   delta0_coupling_ha : δ_0^coupling (Ha) for δ_0^mol = Σ n_A δ_0^A + δ_0^coupling
#                        (prop:convergence_decomposition); macroscopic quantity
#   coupling_k_increment : integer tower-level increment due to coupling in k_min^mol
#                          k_min^mol = max_A(k_min^A) + coupling_k_increment
#                          [THEORY] from cor:kmin_femoco where available;
#                          [ESTIMATE] otherwise — override via k_0_override
#   k_0_override       : exact k_0 if known from QPE or theory; None = compute
#   k_cat_verified     : turnover frequency (s^{-1}) confirmed by QPE/experiment
#   description        : winding, group, spectral class, notes
#
# Note on secondary-mode entries (2, 4, 8, 12, 15):
#   These entries use a sub-maximal admissible modulus (e.g. m=4 when g=8 or g=12).
#   phase1_stoich() always returns m = max{m': 4|m', m'|g}, which is the PRIMITIVE mode.
#   To force a secondary mode, pass m_override (not yet implemented; use run_map()
#   directly and construct Phase1Result manually).  Secondary-mode entries are
#   lower priority (catalog.md; entries 2 and 4 explicitly lower priority).
#
# IMPORTANT — delta0_coupling_ha vs coupling_k_increment:
#   These are SEPARATE quantities on different scales.
#   delta0_coupling_ha is in Ha (macroscopic, ~Ha for metalloenzymes).
#   coupling_k_increment is a small integer (2–5 typically).
#   Do NOT substitute one for the other; compute_kmin_mol() uses the integer.
# ---------------------------------------------------------------------------
CATALOG_ENTRIES: Dict[int, Dict] = {
    1: {
        'name': 'Mo-nitrogenase (m=8)',
        'Ne': 8, 'Mcof': 16,
        'element_multiset': [('Fe', 7), ('Mo', 1), ('S', 9), ('C', 1)],
        'bond_network': [
            ('Fe', 'S'), ('Fe', 'C'), ('Fe', 'Fe'), ('Fe', 'Mo'),
        ],
        'delta0_coupling_ha': 18.2,       # [THEORY] prop:convergence_decomposition
        'coupling_k_increment': 4,         # [THEORY] cor:kmin_femoco: 14 + 4 = 18
        'k_0_override': 18,               # [THEORY] cor:kmin_femoco
        'k_cat_verified': 1.807e12,        # compact pipeline 2026-06-11 (s^{-1})
        'description': (
            'FeMoco primitive mode; winding (1,2); non-dihedral; '
            'Group A, spectral class A; r_selmer=1 (prop:femoco_selmer)'
        ),
    },
    2: {
        'name': 'Mo-nitrogenase (m=4)',
        'Ne': 8, 'Mcof': 16,
        'element_multiset': [('Fe', 7), ('Mo', 1), ('S', 9), ('C', 1)],
        'bond_network': [
            ('Fe', 'S'), ('Fe', 'C'), ('Fe', 'Fe'), ('Fe', 'Mo'),
        ],
        'delta0_coupling_ha': 18.2,
        'coupling_k_increment': 4,
        'k_0_override': 18,
        'k_cat_verified': None,
        'description': (
            'FeMoco secondary mode (m=4); winding (2,4); non-dihedral; '
            'Group A non-primitive; lower priority (catalog entry 2)'
        ),
    },
    3: {
        'name': 'V-nitrogenase (m=12)',
        'Ne': 12, 'Mcof': 24,
        'element_multiset': [('Fe', 7), ('V', 1), ('S', 8), ('C', 1), ('N', 1)],
        'bond_network': [
            ('Fe', 'S'), ('Fe', 'C'), ('Fe', 'N'), ('Fe', 'Fe'),
            ('Fe', 'V'), ('V', 'S'),
        ],
        'delta0_coupling_ha': 17.5,       # [ESTIMATE] similar topology to FeMoco
        'coupling_k_increment': 4,         # [ESTIMATE] same cubane topology → same increment
        'k_0_override': None,
        'k_cat_verified': None,
        'description': (
            'FeVco primitive mode; winding (1,2); non-dihedral; '
            'Group A extended (m=12); Ne=12 Mcof=24'
        ),
    },
    4: {
        'name': 'V-nitrogenase (m=4)',
        'Ne': 12, 'Mcof': 24,
        'element_multiset': [('Fe', 7), ('V', 1), ('S', 8), ('C', 1), ('N', 1)],
        'bond_network': [
            ('Fe', 'S'), ('Fe', 'C'), ('Fe', 'N'), ('Fe', 'Fe'),
            ('Fe', 'V'), ('V', 'S'),
        ],
        'delta0_coupling_ha': 17.5,
        'coupling_k_increment': 3,         # [ESTIMATE] lower-order mode
        'k_0_override': None,
        'k_cat_verified': None,
        'description': (
            'FeVco secondary mode (m=4); winding (3,6); non-dihedral; '
            'lower priority (catalog entry 4)'
        ),
    },
    5: {
        'name': 'PSII/OEC',
        'Ne': 4, 'Mcof': 4,
        'element_multiset': [('Mn', 4), ('Ca', 1), ('O', 5)],
        'bond_network': [('Mn', 'O'), ('Ca', 'O')],
        'delta0_coupling_ha': 2.0,         # [ESTIMATE] μ-oxo + Ca-O cluster coupling
        'coupling_k_increment': 2,          # [ESTIMATE] (k_min^OEC ≈ 4 from PSII_KMIN_MOL)
        'k_0_override': 4,                  # [BOYS] PSII_KMIN_MOL (subsec:psii_langlands)
        'k_cat_verified': 1.1e3,            # O2 evolution rate ≈ 1100 s^{-1}
        'description': (
            'Mn4Ca-O5 cubane-like OEC; winding (1,1); dihedral; '
            'Group A (r_selmer=2); symmetric winding'
        ),
    },
    6: {
        'name': 'Complex I ([4Fe-4S] N2 cluster)',
        'Ne': 4, 'Mcof': 0,
        'element_multiset': [('Fe', 4), ('S', 4)],
        'bond_network': [('Fe', 'S'), ('Fe', 'Fe')],
        'delta0_coupling_ha': 4.5,         # [ESTIMATE] single [4Fe-4S] cubane
        'coupling_k_increment': 3,          # [ESTIMATE]
        'k_0_override': None,
        'k_cat_verified': None,
        'description': (
            'Complex I [4Fe-4S] cluster (N2/N3 site); Ne=4 Mcof=0; '
            'winding (1,0); GL_1; Group A'
        ),
    },
    7: {
        'name': 'Assimilatory NR (m=8)',
        'Ne': 8, 'Mcof': 0,
        'element_multiset': [('Mo', 1), ('Fe', 4), ('S', 4)],
        'bond_network': [
            ('Mo', 'O'), ('Mo', 'S'), ('Fe', 'S'), ('Fe', 'Fe'),
        ],
        'delta0_coupling_ha': 5.0,         # [ESTIMATE] Mo-pterin + [4Fe-4S] coupling
        'coupling_k_increment': 3,          # [ESTIMATE]
        'k_0_override': None,
        'k_cat_verified': None,
        'description': (
            'Assimilatory nitrate reductase, m=8 primitive mode; '
            'winding (1,0); GL_1; Group A; Ne=8 Mcof=0'
        ),
    },
    8: {
        'name': 'Assimilatory NR (m=4)',
        'Ne': 8, 'Mcof': 0,
        'element_multiset': [('Mo', 1), ('Fe', 4), ('S', 4)],
        'bond_network': [
            ('Mo', 'O'), ('Mo', 'S'), ('Fe', 'S'), ('Fe', 'Fe'),
        ],
        'delta0_coupling_ha': 5.0,
        'coupling_k_increment': 3,
        'k_0_override': None,
        'k_cat_verified': None,
        'description': (
            'Assimilatory NR secondary mode (m=4); winding (2,0); GL_1'
        ),
    },
    9: {
        'name': 'CODH/ACS',
        'Ne': 8, 'Mcof': 4,
        'element_multiset': [('Ni', 1), ('Fe', 4), ('S', 5), ('Co', 1)],
        'bond_network': [
            ('Ni', 'Fe'), ('Ni', 'S'), ('Fe', 'S'), ('Co', 'N'),
        ],
        'delta0_coupling_ha': 6.0,         # [ESTIMATE] A-cluster + corrinoid coupling
        'coupling_k_increment': 3,          # [ESTIMATE]
        'k_0_override': None,
        'k_cat_verified': None,
        'description': (
            'CODH/ACS: A-cluster Ni-[4Fe-4S] + corrinoid Co; '
            'Ne=8 Mcof=4; winding (2,1); non-dihedral; r_selmer=1'
        ),
    },
    10: {
        'name': 'Cyt bd oxidase',
        'Ne': 4, 'Mcof': 0,
        'element_multiset': [('Fe', 3)],
        'bond_network': [],                 # heme b558/b595/d — no direct Fe-Fe bond
        'delta0_coupling_ha': 0.5,         # [ESTIMATE] through-space heme-heme
        'coupling_k_increment': 2,          # [ESTIMATE]
        'k_0_override': None,
        'k_cat_verified': None,
        'description': (
            'Cytochrome bd oxidase (heme b558/b595/d); Ne=4 Mcof=0; '
            'winding (1,0); GL_1; Group A'
        ),
    },
    11: {
        'name': 'Cyt c oxidase',
        'Ne': 4, 'Mcof': 0,
        'element_multiset': [('Cu', 2), ('Fe', 2)],
        'bond_network': [('Cu', 'Cu'), ('Cu', 'O')],
        'delta0_coupling_ha': 3.0,         # [ESTIMATE] CuA binuclear + heme a3-CuB
        'coupling_k_increment': 3,          # [ESTIMATE]
        'k_0_override': None,
        'k_cat_verified': None,
        'description': (
            'Cytochrome c oxidase: CuA/CuB binuclear + heme a/a3; '
            'Ne=4 Mcof=0; winding (1,0); GL_1'
        ),
    },
    12: {
        'name': 'Cu CO2RR (m=4)',
        'Ne': 12, 'Mcof': 0,
        'element_multiset': [('Cu', 4)],
        'bond_network': [('Cu', 'Cu')],
        'delta0_coupling_ha': 2.0,         # [ESTIMATE] 4-Cu surface cluster
        'coupling_k_increment': 3,          # [ESTIMATE]
        'k_0_override': None,
        'k_cat_verified': None,
        'description': (
            'Cu electrocatalyst CO2→C2H4, m=4 mode; Ne=12 Mcof=0; '
            'winding (3,0); GL_1; Group D (purely topological, w_LZ~10^{-13})'
        ),
    },
    13: {
        'name': 'Cu CO2RR (m=12)',
        'Ne': 12, 'Mcof': 0,
        'element_multiset': [('Cu', 4)],
        'bond_network': [('Cu', 'Cu')],
        'delta0_coupling_ha': 2.0,
        'coupling_k_increment': 3,
        'k_0_override': None,
        'k_cat_verified': None,
        'description': (
            'Cu electrocatalyst CO2→C2H4, m=12 primitive mode; Ne=12 Mcof=0; '
            'winding (1,0); GL_1; Group D'
        ),
    },
    14: {
        'name': 'Photocatalytic N2 fixation (m=8)',
        'Ne': 8, 'Mcof': 8,
        # Default: TiO2 representative.
        # Alternatives: PHOTO_N2_ELEMENTS_MOS2, PHOTO_N2_ELEMENTS_BIOBR
        'element_multiset': PHOTO_N2_ELEMENTS_TIO2,
        'bond_network': [('Ti', 'O'), ('Ti', 'N')],
        'delta0_coupling_ha': 2.5,         # [ESTIMATE] surface-N2 coupling, TiO2
        'coupling_k_increment': 3,          # [ESTIMATE]
        'k_0_override': None,
        'k_cat_verified': None,
        'description': (
            'Photocatalytic N2 fixation (TiO2 class, representative); '
            'Ne=8 Mcof=8; winding (1,1); dihedral; r_selmer=2; m=8 mode'
        ),
    },
    15: {
        'name': 'Photocatalytic N2 fixation (m=4)',
        'Ne': 8, 'Mcof': 8,
        'element_multiset': PHOTO_N2_ELEMENTS_TIO2,
        'bond_network': [('Ti', 'O'), ('Ti', 'N')],
        'delta0_coupling_ha': 2.5,
        'coupling_k_increment': 2,          # [ESTIMATE] lower m mode
        'k_0_override': None,
        'k_cat_verified': None,
        'description': (
            'Photocatalytic N2 fixation (TiO2 class, representative); '
            'Ne=8 Mcof=8; winding (2,2); dihedral; r_selmer=2; m=4 mode'
        ),
    },
}

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def compute_q_ab(alpha_A: float, alpha_B: float, dist_bohr: float) -> float:
    """
    Gaussian product theorem bond decay exponent (prop:coupling_decay):

        Q_AB = (α_A · α_B) / (α_A + α_B) · |R_A − R_B|²

    Parameters
    ----------
    alpha_A, alpha_B : float  — primitive Gaussian exponents (bohr⁻²)
    dist_bohr : float         — interatomic distance (bohr)
    """
    return (alpha_A * alpha_B) / (alpha_A + alpha_B) * dist_bohr ** 2


def get_bond_factor(elem_a: str, elem_b: str) -> float:
    """
    Bond amplitude e^{-Q_AB} for a bonded pair.
    Falls back to BOND_NEGLIGIBLE_THRESHOLD for unknown pairs
    (appropriate for non-bonded / super-exchange-negligible pairs).
    """
    key = _bond_key(elem_a, elem_b)
    return BOND_DECAY_FACTOR.get(key, BOND_NEGLIGIBLE_THRESHOLD)


def get_bond_q(elem_a: str, elem_b: str) -> float:
    """
    Q_AB for a bonded pair. Returns a large sentinel (-log threshold)
    for pairs not in BOND_DECAY_Q (negligible coupling).
    """
    key = _bond_key(elem_a, elem_b)
    if key in BOND_DECAY_Q:
        return BOND_DECAY_Q[key]
    return -log(BOND_NEGLIGIBLE_THRESHOLD)


def compute_delta0_mol(
    element_multiset: List[Tuple[str, int]],
    delta0_coupling_ha: float,
) -> float:
    """
    Molecular seed residual decomposition (prop:convergence_decomposition):

        δ_0^mol = Σ_A n_A · δ_0^A  +  δ_0^coupling

    Parameters
    ----------
    element_multiset : list of (str, int)
    delta0_coupling_ha : float  — inter-atomic coupling term (Ha)
    """
    per_atom_sum = sum(
        count * ATOMIC_SEED_RESIDUALS_HA[elem]
        for elem, count in element_multiset
    )
    return per_atom_sum + delta0_coupling_ha


def compute_spectral_weights(
    element_multiset: List[Tuple[str, int]],
    delta0_coupling_ha: float,
) -> Dict[str, float]:
    """
    Per-element Frobenius spectral weights χ^A = δ_0^A / δ_0^mol
    (def:per_atom_spectral_weight, eq 713).

    Returns {element_symbol: χ^A} for each unique element in the multiset.
    """
    delta0_mol = compute_delta0_mol(element_multiset, delta0_coupling_ha)
    return {
        elem: ATOMIC_SEED_RESIDUALS_HA[elem] / delta0_mol
        for elem, _ in element_multiset
    }


def compute_kmin_mol(
    element_multiset: List[Tuple[str, int]],
    delta0_coupling_ha: float,
    coupling_k_increment: Optional[int] = None,
    epsilon_ha: float = PRECISION_BUDGET_HA,
    p: int = TOWER_PRIME,
) -> int:
    """
    Minimum Iwasawa tower level for the molecule (cor:kmin_femoco):

        k_min^mol = max_A(k_min^A) + coupling_k_increment

    Parameters
    ----------
    element_multiset : list of (str, int)
    delta0_coupling_ha : float
        Macroscopic coupling term δ_0^coupling (Ha).  Used ONLY when
        coupling_k_increment is None (rough heuristic fallback).
    coupling_k_increment : int or None
        Integer tower-level increment from coupling.  Preferred when known
        from theory (e.g. 4 for FeMoco via cor:kmin_femoco).
        When None, falls back to a bond-count heuristic.
    epsilon_ha : float — precision budget (Ha)
    p : int — tower prime

    Returns
    -------
    int  k_min^mol
    """
    elements_in_tower = [
        elem for elem, _ in element_multiset if elem in PER_ATOM_TOWER
    ]
    if not elements_in_tower:
        raise ValueError(
            f"No elements in PER_ATOM_TOWER for {element_multiset}. "
            "Add missing elements to PER_ATOM_TOWER."
        )
    k_max_atom = max(PER_ATOM_TOWER[elem].k_min for elem in elements_in_tower)
    if coupling_k_increment is not None:
        return k_max_atom + coupling_k_increment
    # Heuristic fallback: crude approximation, use coupling_k_increment from
    # CATALOG_ENTRIES where available rather than this path.
    if delta0_coupling_ha <= 0:
        return k_max_atom
    increment = max(1, ceil(log(delta0_coupling_ha / (16 * epsilon_ha)) / log(p)))
    return k_max_atom + increment


def compute_kmin_from_entry(entry: Dict) -> int:
    """
    Compute k_min^mol from a CATALOG_ENTRIES dict entry, using
    coupling_k_increment when available, otherwise heuristic.

    Returns k_0_override directly if it is set.
    """
    if entry.get('k_0_override') is not None:
        return entry['k_0_override']
    return compute_kmin_mol(
        element_multiset=entry['element_multiset'],
        delta0_coupling_ha=entry['delta0_coupling_ha'],
        coupling_k_increment=entry.get('coupling_k_increment'),
    )
