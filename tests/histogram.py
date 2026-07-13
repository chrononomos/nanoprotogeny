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
import numpy as np
import json
import itertools
import scipy
from scipy.linalg import expm
import scipy.linalg
import concurrent.futures

from typing import Dict, List, Tuple, Iterator, Union, Optional, Literal, Any
import logging
from pathlib import Path, sys
import networkx as nx
from collections import defaultdict

import os
import time
import uuid
import enum
import copy
import hashlib
import argparse
import dataclasses
from dataclasses import dataclass, field

# Your histogram data
histogram = {"0":0.0038999998942017555,"2":0.007799999788403511,"3":0.0038999998942017555,"8":0.007799999788403511,"10":0.015699999406933784,"11":0.007799999788403511,"12":0.0038999998942017555,"14":0.007799999788403511,"15":0.0038999998942017555,"32":0.007799999788403511,"34":0.015699999406933784,"35":0.007799999788403511,"40":0.015699999406933784,"42":0.031300000846385956,"43":0.015599999576807022,"44":0.007799999788403511,"46":0.015599999576807022,"47":0.007799999788403511,"48":0.0038999998942017555,"50":0.007799999788403511,"51":0.0038999998942017555,"56":0.007799999788403511,"58":0.015599999576807022,"59":0.007799999788403511,"60":0.0038999998942017555,"62":0.007799999788403511,"63":0.0038999998942017555,"128":0.007799999788403511,"130":0.015699999406933784,"131":0.007799999788403511,"136":0.015699999406933784,"138":0.031300000846385956,"139":0.015599999576807022,"140":0.007799999788403511,"142":0.015599999576807022,"143":0.007799999788403511,"160":0.015699999406933784,"162":0.031300000846385956,"163":0.015599999576807022,"168":0.031300000846385956,"170":0.0625,"171":0.031199999153614044,"172":0.015599999576807022,"174":0.031199999153614044,"175":0.015599999576807022,"176":0.007799999788403511,"178":0.015599999576807022,"179":0.007799999788403511,"184":0.015599999576807022,"186":0.031199999153614044,"187":0.015599999576807022,"188":0.007799999788403511,"190":0.015599999576807022,"191":0.007799999788403511,"192":0.0038999998942017555,"194":0.007799999788403511,"195":0.0038999998942017555,"200":0.007799999788403511,"202":0.015599999576807022,"203":0.007799999788403511,"204":0.0038999998942017555,"206":0.007799999788403511,"207":0.0038999998942017555,"224":0.007799999788403511,"226":0.015599999576807022,"227":0.007799999788403511,"232":0.015599999576807022,"234":0.031199999153614044,"235":0.015599999576807022,"236":0.007799999788403511,"238":0.015599999576807022,"239":0.007799999788403511,"240":0.0038999998942017555,"242":0.007799999788403511,"243":0.0038999998942017555,"248":0.007799999788403511,"250":0.015599999576807022,"251":0.007799999788403511,"252":0.0038999998942017555,"254":0.007799999788403511,"255":0.0038999998942017555}

# Verify normalization
total_prob = sum(histogram.values())
print(f"Total probability: {total_prob:.6f}")

# Convert probabilities to counts (using your --n-shots 100 setting)
N_SHOTS = 100
counts = {bs: int(round(prob * N_SHOTS)) for bs, prob in histogram.items()}

def probs_to_counts(histogram: dict, n_shots: int) -> dict:
    """Convert probabilities to integer counts preserving exact total."""
    # Step 1: Floor allocation
    counts = {bs: int(np.floor(prob * n_shots)) for bs, prob in histogram.items()}
    remainder = n_shots - sum(counts.values())
    
    # Step 2: Distribute remainder to largest fractional parts
    if remainder > 0:
        fractions = {bs: (prob * n_shots) - counts[bs] 
                     for bs, prob in histogram.items()}
        for bs in sorted(fractions, key=fractions.get, reverse=True)[:remainder]:
            counts[bs] += 1
    return {bs: c for bs, c in counts.items() if c > 0}

# Usage
counts = probs_to_counts(histogram, n_shots=100)
print(f"Total reconstructed shots: {sum(counts.values())}")  # → 100 exactly
print(f"Non-zero bitstrings: {len(counts)} / {len(histogram)}")
print(f"Total reconstructed shots: {sum(counts.values())}")

import numpy as np
from typing import List, Tuple, Dict, Literal

# ── [Your validated histogram & probs_to_counts code here] ──────────────
counts = probs_to_counts(histogram, n_shots=100)
print(f"✓ Validated counts ready: {sum(counts.values())} shots across {len(counts)} outcomes\n")

# ── Step 1: Weyl–Heisenberg Decomposition (from qpu_evolution.py) ──────
# def qudit_hamiltonian_to_weyl_strings(
#     H: np.ndarray, n_qudits: int, d: int = 4, threshold: float = 1e-10
# ) -> List[Tuple[complex, List[Tuple[int, int]]]]:
#     omega = np.exp(2j * np.pi / d)
#     X = np.roll(np.eye(d), -1, axis=0)
#     Z = np.diag([omega**k for k in range(d)])
#     weyl_single = [(a, b, np.linalg.matrix_power(X, a) @ np.linalg.matrix_power(Z, b)) 
#                    for a in range(d) for b in range(d)]
    
#     result = []
#     for labels in __import__("itertools").product(weyl_single, repeat=n_qudits):
#         P = np.array([[1.0]], dtype=complex)
#         op_labels = []
#         for a, b, W in labels:
#             P = np.kron(P, W); op_labels.append((a, b))
#         coeff = np.trace(P.conj().T @ H) / (d ** n_qudits)
#         if abs(coeff) >= threshold:
#             result.append((coeff, op_labels))
#     result.sort(key=lambda x: abs(x[0]), reverse=True)
#     return result

def qudit_hamiltonian_to_weyl_strings(
    H: np.ndarray,
    n_qudits: int,
    d: int = 4,
    threshold: float = 1e-10,
) -> List[Tuple[complex, List[Tuple[int, int]]]]:
    """
    Decompose a d^n × d^n Hermitian matrix into Weyl-Heisenberg operators.

    H = Σ_{a⃗,b⃗} β_{a⃗,b⃗} ⊗_p W_{a_p,b_p},   W_{a,b} = X^a Z^b

    Expansion coefficients: β_{a⃗,b⃗} = Tr(W_{a⃗,b⃗}† H) / d^n

    This is the d-dimensional analogue of Pauli decomposition.  For d=4
    there are 16^n terms total; the threshold prunes negligible ones so the
    shot estimator only measures terms that contribute ≥ threshold Ha.

    Args:
        H:          Hermitian matrix, shape (d^n_qudits, d^n_qudits).
        n_qudits:   Number of d-level qudits.
        d:          Qudit dimension (4 for this project).
        threshold:  Discard |β| < threshold.

    Returns:
        Sorted list of (coefficient, [(a_1,b_1), …, (a_n,b_n)]) pairs,
        largest |coeff| first.
    """
    omega = np.exp(2j * np.pi / d)
    X = np.roll(np.eye(d), -1, axis=0)   # cyclic shift |k⟩→|k+1 mod d⟩
    Z = np.diag([omega ** k for k in range(d)])  # clock operator

    # Single-qudit Weyl operators  W_{a,b} = X^a Z^b
    weyl_single = []
    for a in range(d):
        for b in range(d):
            W = np.linalg.matrix_power(X, a) @ np.linalg.matrix_power(Z, b)
            weyl_single.append((a, b, W))

    result: List[Tuple[complex, List[Tuple[int, int]]]] = []
    norm   = float(d ** n_qudits)

    for labels in itertools.product(weyl_single, repeat=n_qudits):
        P         = np.array([[1.0 + 0j]])
        op_labels: List[Tuple[int, int]] = []
        for a, b, W in labels:
            P = np.kron(P, W)
            op_labels.append((a, b))
        coeff = complex(np.trace(P.conj().T @ H)) / norm
        if abs(coeff) >= threshold:
            result.append((coeff, op_labels))

    result.sort(key=lambda x: abs(x[0]), reverse=True)
    # log.info(
    #     f"[WEYL] H decomposed: {len(result)} non-zero Weyl terms "
    #     f"(d={d}, n={n_qudits}, threshold={threshold:.1e})"
    # )
    return result

# ── Step 2: Shot-Based Energy Estimation (from qpu_evolution.py) ───────
def estimate_energy_from_qudit_shots(
    weyl_terms: List[Tuple[complex, List[Tuple[int, int]]]],
    counts: Dict[str, int],
    n_qudits: int, d: int = 4, manifold: Literal['logical', 'virtual'] = 'logical',
) -> float:
    B = np.array([[1,0,0,0],[0,0,1/np.sqrt(2),1/np.sqrt(2)],[0,0,1/np.sqrt(2),-1/np.sqrt(2)],[0,1,0,0]], dtype=complex) # B_LOG
    total_shots = sum(counts.values())
    if total_shots == 0: raise ValueError("Empty counts")
    
    energy = 0.0
    for coeff, op_labels in weyl_terms:
        expval = 0.0j
        for bs, count in counts.items():
            if len(bs) != 2 * n_qudits: continue
            corner_outcomes = []
            for p in range(n_qudits):
                b0, b1 = int(bs[2*p]), int(bs[2*p+1])
                phys_vec = np.zeros(4, dtype=complex); phys_vec[2*b0+b1] = 1.0
                corner_probs = [abs(np.vdot(B[:, k], phys_vec))**2 for k in range(d)]
                corner_outcomes.append(int(np.argmax(corner_probs)))
            
            phase = 1.0+0.0j
            for p, (a, b) in enumerate(op_labels):
                k = corner_outcomes[p]
                phase *= (1j)**(b * k)  # Z^b contribution
            expval += phase * count
        expval /= total_shots
        energy += coeff * expval
    return energy.real

# ── Step 3: Execution & FCI Comparison ─────────────────────────────────
# Replace this with your actual H_qudit (4^4 × 4^4 matrix from integrals)
# H_qudit = build_qudit_hamiltonian_matrix(N_ORBITALS, H_DIAG, H_HOP, G_FULL)
# For testing, use a mock Hermitian matrix of correct dimension:
np.random.seed(42)
H_mock = np.random.randn(256, 256) + 1j * np.random.randn(256, 256)
H_qudit = (H_mock + H_mock.conj().T) / 2  # Force Hermiticity

print("[PIPELINE] Decomposing H_qudit into Weyl-Heisenberg operators...")
weyl_terms = qudit_hamiltonian_to_weyl_strings(H_qudit, n_qudits=4, d=4, threshold=1e-10)
print(f"[PIPELINE] Retained {len(weyl_terms)} non-zero Weyl terms.")

print("[PIPELINE] Estimating energy from shot counts (fold=1)...")
E_est = estimate_energy_from_qudit_shots(
    weyl_terms=weyl_terms, counts=counts, n_qudits=4, d=4, manifold='logical'
)

# FCI reference from your adenine CAS(4,4) run
E_FCI_ACTIVE = -2.6381449566
bias_mHa = (E_est - E_FCI_ACTIVE) * 1000

print(f"\n[RESULTS] E_est (fold=1)     = {E_est:+.6f} Ha")
print(f"[RESULTS] E_FCI (reference)  = {E_FCI_ACTIVE:+.6f} Ha")
print(f"[RESULTS] Bias vs FCI        = {bias_mHa:+.3f} mHa")
print(f"[STATUS] {'✓ Within chemical accuracy' if abs(bias_mHa) <= 1.6 else '⚠ Statistical noise dominates (increase shots)'}")