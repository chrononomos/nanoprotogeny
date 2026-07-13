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
nanoprotogeny.ionq.ionqconnectivity
IonQ v0.4 API Connectivity and Backend Configuration.

Provides backend configuration, service factory, job manifest utilities,
and connectivity probing for IonQ hardware integration.

Histogram parsing and ontological decoding live in ionqhistogram.py.
"""
from __future__ import annotations
import os
import time
import json
import hashlib
import enum
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, field

try:
    import cirq
    import cirq_ionq
    _CIRQ_IONQ_AVAILABLE = True
except ImportError:
    _CIRQ_IONQ_AVAILABLE = False
    cirq = None  # type: ignore
    cirq_ionq = None  # type: ignore

import numpy as np
import logging

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# BACKEND CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

class BackendMode(enum.Enum):
    """Execution backend modes."""
    LOCAL    = "local"      # Local cirq DensityMatrixSimulator — no API key required
    IONQ_SIM = "ionq-sim"   # IonQ cloud simulator via REST
    IONQ_QPU = "ionq-qpu"   # Real IonQ QPU

    @classmethod
    def from_str(cls, s: str) -> "BackendMode":
        for m in cls:
            if m.value == s:
                return m
        raise ValueError(f"Unknown backend {s!r}. Choose: {[m.value for m in cls]}")


@dataclass
class BackendConfig:
    """All parameters needed to configure and authenticate the IonQ backend."""
    
    mode: BackendMode = BackendMode.IONQ_SIM
    
    # IonQ REST API credentials & targeting
    api_key: Optional[str] = None
    api_url: str = "https://api.ionq.co/v0.4"
    qpu_target: str = "qpu.forte-1"
    sim_target: str = "simulator"
    
    # Shot budget
    n_shots: int = 8192
    zne_folds: List[int] = field(default_factory=lambda: [1, 3, 5])
    
    # Job management
    job_timeout_s: int = 7200
    poll_interval: float = 15.0
    resume_job_id: Optional[str] = None
    
    # Reproducibility
    manifest_path: str = "ionq_job_manifest.json"
    
    @property
    def effective_api_key(self) -> Optional[str]:
        return self.api_key or os.environ.get("IONQ_API_KEY")
    
    @property
    def resolved_target(self) -> str:
        if self.mode == BackendMode.LOCAL:
            return "local"
        return self.qpu_target if self.mode == BackendMode.IONQ_QPU else self.sim_target
    
    def validate(self) -> None:
        """Validate configuration for the selected backend mode."""
        if self.mode == BackendMode.LOCAL:
            return  # No credentials or external packages required
        if self.mode in (BackendMode.IONQ_SIM, BackendMode.IONQ_QPU):
            if not self.effective_api_key:
                raise RuntimeError(
                    "IonQ API key is required for ionq-sim and ionq-qpu backends. "
                    "Pass --api-key or set the IONQ_API_KEY environment variable."
                )
            if self.mode == BackendMode.IONQ_QPU and not _CIRQ_IONQ_AVAILABLE:
                raise RuntimeError(
                    "cirq-ionq is required for QPU submission. "
                    "Install with: pip install cirq-ionq"
                )
        
        # Validate ZNE folds
        for f in self.zne_folds:
            if f < 1 or f % 2 == 0:
                raise ValueError(
                    f"All --zne-folds values must be positive odd integers; got {f}."
                )
        if len(self.zne_folds) != 3:
            import warnings
            warnings.warn(
                f"--zne-folds has {len(self.zne_folds)} values; "
                "exactly 3 values are needed for standard Richardson extrapolation.",
                stacklevel=2,
            )


# ──────────────────────────────────────────────────────────────────────────────
# SERVICE FACTORY
# ──────────────────────────────────────────────────────────────────────────────

def _make_ionq_service(cfg: BackendConfig) -> "cirq_ionq.Service":
    """Instantiate a cirq_ionq.Service from a BackendConfig."""
    if not _CIRQ_IONQ_AVAILABLE:
        raise RuntimeError(
            "cirq-ionq is required for IonQ backends. "
            "Install with: pip install cirq-ionq"
        )
    cfg.validate()
    return cirq_ionq.Service(
        api_key=cfg.effective_api_key,
        remote_host=cfg.api_url,
        default_target=cfg.resolved_target,
        max_retry_seconds=cfg.job_timeout_s,
    )


# ──────────────────────────────────────────────────────────────────────────────
# RETRY SESSION HELPER
# ──────────────────────────────────────────────────────────────────────────────

def _get_retry_session() -> requests.Session:
    """Create a session with robust retry logic for transient network errors."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504, 520],
        allowed_methods=["POST", "GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# Histogram parsing moved to ionqhistogram.py — import here for probe use
from nanoprotogeny.ionq.ionqhistogram import _parse_histogram_to_counts


# ──────────────────────────────────────────────────────────────────────────────
# JOB MANIFEST
# ──────────────────────────────────────────────────────────────────────────────

def _save_job_manifest(
    job_id: str,
    circuit: cirq.Circuit,
    n_shots: int,
    name: str,
    cfg: BackendConfig,
) -> None:
    """Write a JSON manifest so interrupted QPU runs can be resumed.
    
    Args:
        job_id: IonQ job identifier.
        circuit: The submitted Cirq circuit.
        n_shots: Number of measurement shots.
        name: Human-readable job name.
        cfg: BackendConfig used for submission.
    """
    circuit_digest = hashlib.sha256(str(circuit).encode()).hexdigest()[:16]
    manifest = {
        "job_id": job_id,
        "circuit_digest": circuit_digest,
        "name": name,
        "n_shots": n_shots,
        "target": cfg.resolved_target,
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend_mode": cfg.mode.value,
    }
    path = cfg.manifest_path
    try:
        if Path(path).exists():
            with open(path) as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = [existing]
        else:
            existing = []
        existing.append(manifest)
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)
        log.info(f"[MANIFEST] Job manifest saved to {path!r}")
    except Exception as exc:
        log.warning(f"[MANIFEST] Could not save manifest: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# EXPORTS
# ──────────────────────────────────────────────────────────────────────────────

def probe_ionq_service(cfg: "BackendConfig") -> None:
    """Submit a trivial √X test circuit to verify IonQ service connectivity.

    Sends one qubit through X**0.5 + measure to confirm the backend is
    reachable before committing a real job.  No-ops when cfg.resume_job_id
    is set (the service is assumed live from the previous submission).

    Args:
        cfg: BackendConfig describing the target backend and shot count.
    """
    if cfg.resume_job_id:
        log.info("[IONQ-PROBE] Resuming existing job — skipping connectivity probe.")
        return

    service = _make_ionq_service(cfg)
    q = cirq.LineQubit(0)
    test_circuit = cirq.Circuit(cirq.X(q) ** 0.5, cirq.measure(q, key="result"))

    log.info(f"[IONQ-PROBE] Submitting connectivity probe to {cfg.resolved_target!r}...")
    result = service.run(
        circuit=test_circuit,
        repetitions=min(cfg.n_shots, 100),
        target=cfg.resolved_target,
        extra_query_params=(
            {"noise": {"model": "forte-1"}}
            if "simulator" in cfg.resolved_target
            else {}
        ),
    )
    counts = _parse_histogram_to_counts(
        histogram_data=result.histogram(key="result"),
        n_phys_qubits=1,
        decode_to_qudit=False,
        expected_shots=min(cfg.n_shots, 100),
    )
    log.info(
        f"[IONQ-PROBE] ✓ Connectivity confirmed. "
        f"Sample counts: {dict(list(counts.items())[:3])}"
    )


__all__ = [
    # Configuration
    "BackendMode",
    "BackendConfig",

    # Service factory
    "_make_ionq_service",

    # Connectivity probe
    "probe_ionq_service",

    # Utilities
    "_get_retry_session",
    "_save_job_manifest",

    # Availability flag
    "_CIRQ_IONQ_AVAILABLE",
]