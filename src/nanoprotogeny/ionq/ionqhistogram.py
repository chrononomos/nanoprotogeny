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
ionqhistogram.py — IonQ Measurement Result Parsing and Ontological Decoding
============================================================================
Post-processing layer for IonQ circuit measurement results.

Responsibilities
----------------
_parse_histogram_to_counts
    Unified parser for IonQ API histogram dicts and cirq.Result objects.
    Handles the various key formats returned by the IonQ REST API
    (integer keys, hex strings, binary strings) and normalises them to
    zero-padded binary strings with integer counts.  Optionally decodes
    physical bitstrings to ontological corner probabilities.

_decode_physical_to_ontological
    Decodes a physical qubit bitstring to per-corner probabilities in the
    tetralemmatic d=4 encoding using the Bell-separable basis transform
    (B_LOG for the logical manifold, B_VIRT for the virtual manifold).
    Specific to the ¹⁷¹Yb⁺ NomosIonQid encoding on IonQ hardware.

Separation from ionqconnectivity.py
------------------------------------
Connectivity (auth, service factory, job manifests) is independent of
result parsing.  This file has no dependency on the IonQ REST API or
cirq_ionq — only on cirq.Result, numpy, and the ionq-layer basis matrices.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Union

import numpy as np

try:
    import cirq
    _CIRQ_AVAILABLE = True
except ImportError:
    _CIRQ_AVAILABLE = False
    cirq = None  # type: ignore

from nanoprotogeny.ionq.ionqmqegates import B_LOG, B_VIRT


# ──────────────────────────────────────────────────────────────────────────────
# HISTOGRAM PARSER
# ──────────────────────────────────────────────────────────────────────────────

def _parse_histogram_to_counts(
    histogram_data:   Union[Dict[str, float], "cirq.Result"],
    n_phys_qubits:    int,
    decode_to_qudit:  bool                      = False,
    manifold:         Literal["logical", "virtual"] = "logical",
    n_logical_qudits: Optional[int]             = None,
    expected_shots:   int                       = 10000,
) -> Dict[str, Union[int, Dict[int, float]]]:
    """Parse IonQ API histograms or cirq.Result objects into normalised counts.

    Args:
        histogram_data:   IonQ API histogram dict or cirq.Result.
        n_phys_qubits:    Number of physical qubits in the circuit.
        decode_to_qudit:  When True, decode each bitstring to ontological
                          corner probabilities via _decode_physical_to_ontological.
        manifold:         Basis to use for decoding: 'logical' (B_LOG) or
                          'virtual' (B_VIRT).
        n_logical_qudits: Number of logical qudits (required when decode_to_qudit).
        expected_shots:   Shot count for scaling probability-valued histograms.

    Returns:
        Dict mapping zero-padded binary bitstrings to integer counts, or
        to ontological probability dicts when decode_to_qudit is True.
    """
    B      = B_VIRT if manifold == "virtual" else B_LOG
    d      = 4
    counts: Dict[str, Union[int, Dict[int, float]]] = {}

    # ── Case 1: IonQ API histogram dict ───────────────────────────────────────
    if isinstance(histogram_data, dict):
        if not histogram_data:
            raise ValueError("Received empty histogram from IonQ API.")

        vals     = list(histogram_data.values())
        is_prob  = (
            all(isinstance(v, (int, float)) for v in vals)
            and max(vals) <= 1.0
            and abs(sum(vals) - 1.0) < 0.05
        )
        scale = expected_shots if is_prob else 1.0

        for key, val in histogram_data.items():
            # Normalise key — cirq_ionq.Service can return integer keys
            if isinstance(key, int):
                key = format(key, f"0{n_phys_qubits}b")
            elif not isinstance(key, str):
                key = str(key)

            count = int(round(float(val) * scale))
            if count == 0 and is_prob:
                count = 1  # Preserve rare outcomes

            # Normalise to zero-padded binary string
            if key.startswith("0b"):
                clean = key[2:].zfill(n_phys_qubits)
            elif key.startswith("0x"):
                clean = format(int(key, 16), f"0{n_phys_qubits}b")
            elif key.replace(" ", "").isdigit():
                clean = format(int(key), f"0{n_phys_qubits}b")
            else:
                clean = key.lstrip("0b").zfill(n_phys_qubits)

            if decode_to_qudit and n_logical_qudits is not None:
                counts[clean] = _decode_physical_to_ontological(clean, B, d, n_logical_qudits)
            else:
                counts[clean] = count

    # ── Case 2: cirq.Result object ─────────────────────────────────────────────
    elif _CIRQ_AVAILABLE and isinstance(histogram_data, cirq.Result):
        shots = histogram_data.measurements.get(
            "result", np.empty((0, n_phys_qubits), dtype=int)
        )
        for row in shots:
            bs = "".join(str(int(b)) for b in row[:n_phys_qubits])
            if decode_to_qudit and n_logical_qudits is not None:
                corner_probs = _decode_physical_to_ontological(bs, B, d, n_logical_qudits)
                if bs in counts:
                    for k, v in corner_probs.items():
                        counts[bs][k] = counts[bs].get(k, 0.0) + v  # type: ignore[index]
                else:
                    counts[bs] = corner_probs
            else:
                counts[bs] = counts.get(bs, 0) + 1  # type: ignore[assignment]
    else:
        raise TypeError(
            f"histogram_data must be dict or cirq.Result, got {type(histogram_data)}"
        )

    return counts


# ──────────────────────────────────────────────────────────────────────────────
# ONTOLOGICAL DECODER
# ──────────────────────────────────────────────────────────────────────────────

def _decode_physical_to_ontological(
    bitstring: str,
    B:         np.ndarray,
    d:         int,
    n_qudits:  int,
) -> Dict[int, float]:
    """Decode a physical bitstring to ontological corner probabilities.

    Each pair of physical qubits encodes one d=4 NomosIonQid qudit via the
    Bell-separable basis B.  The function projects each pair onto the four
    tetralemmatic corners {Th=0, AntiTh=1, SynTh=2, HoloTh=3} and averages
    across qudits to give per-corner occupation probabilities.

    Args:
        bitstring: Binary string of length 2 × n_qudits.
        B:         Basis matrix — B_LOG (logical) or B_VIRT (virtual).
        d:         Qudit dimension (always 4 for tetralemmatic encoding).
        n_qudits:  Number of qudits.

    Returns:
        Dict mapping corner index (0–3) to normalised probability.
    """
    if len(bitstring) != 2 * n_qudits:
        raise ValueError(
            f"bitstring length {len(bitstring)} != 2×n_qudits={2*n_qudits}"
        )

    corner_probs: Dict[int, float] = {}
    for p in range(n_qudits):
        b0       = int(bitstring[2 * p])
        b1       = int(bitstring[2 * p + 1])
        phys_idx = 2 * b0 + b1
        phys_vec = np.zeros(d, dtype=complex)
        phys_vec[phys_idx] = 1.0
        for k in range(d):
            prob = np.abs(np.vdot(B[:, k], phys_vec)) ** 2
            corner_probs[k] = corner_probs.get(k, 0.0) + prob / n_qudits

    total = sum(corner_probs.values())
    if total > 1e-12:
        corner_probs = {k: v / total for k, v in corner_probs.items()}

    return corner_probs


__all__ = [
    "_parse_histogram_to_counts",
    "_decode_physical_to_ontological",
]
