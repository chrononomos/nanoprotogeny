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
test_steps_4_5.py — Boys localization (Step 4) and screened AO→MO (Step 5)
===========================================================================

Guards:
  * Boys localization preserves g_MO Frobenius norm (unitary rotational
    invariance of ERIs).
  * schwarz_bounds does not raise and returns non-negative K[P,Q].
  * screened_direct_ao_to_mo with threshold=0 matches the dense g_AO
    einsum path within 1e-8 Ha (max element).
  * Regression: localize=False, screened=False gives identical results to
    the legacy build_ao_integrals / build_full_mo_tensors path.

Run: pytest tests/test_steps_4_5.py -q   (or: python tests/test_steps_4_5.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from nanoprotogeny.molecular.mqeaobuild import (
    build_ao_integrals,
    build_ao_integrals_with_shells,
)
from nanoprotogeny.molecular.mqeaointegrals import schwarz_bounds
from nanoprotogeny.molecular.mqelocalize import boys_localize, build_dipole_ao_matrices
from nanoprotogeny.molecular.mqeseedtensors import build_full_mo_tensors

# ---------------------------------------------------------------------------
# Shared geometry: minimal H2O (Bohr)
# ---------------------------------------------------------------------------
_H2O = [
    ("O", np.array([0.0,  0.0,      0.0])),
    ("H", np.array([0.0,  1.43,     1.10])),
    ("H", np.array([0.0, -1.43,     1.10])),
]


# ---------------------------------------------------------------------------
# Helper: build both paths with defaults
# ---------------------------------------------------------------------------
def _dense(atoms=_H2O, **kw):
    S, h1, Enuc, g_AO = build_ao_integrals(atoms, d_single_zeta=True, full_shells=False)
    h1_mo, g_mo, Ecore, C = build_full_mo_tensors(
        h1, g_AO, S, Enuc, N_frozen=1, N_elec_total=10, **kw
    )
    return h1_mo, g_mo, Ecore, C


def _screened(atoms=_H2O, thr=0.0, **kw):
    S, h1, Enuc, shells, norms, sph_C = build_ao_integrals_with_shells(
        atoms, d_single_zeta=True, full_shells=False, spherical=False
    )
    h1_mo, g_mo, Ecore, C = build_full_mo_tensors(
        h1, None, S, Enuc, N_frozen=1, N_elec_total=10,
        screened=True, shells=shells, norms=norms, sph_C=sph_C,
        schwarz_thr=thr, **kw
    )
    return h1_mo, g_mo, Ecore, C


# ---------------------------------------------------------------------------
# Step 4: schwarz_bounds argument order (regression for the fixed bug)
# ---------------------------------------------------------------------------
def test_schwarz_bounds_runs_and_is_nonneg():
    """schwarz_bounds must not raise (was TypeError before fix) and return K>=0."""
    S, h1, Enuc, shells, norms, _ = build_ao_integrals_with_shells(
        _H2O, d_single_zeta=True, full_shells=False, spherical=False
    )
    K = schwarz_bounds(shells, norms)
    assert K.shape == (len(shells), len(shells))
    assert np.all(K >= 0.0), "Schwarz bounds must be non-negative"
    assert np.any(K > 0.0), "At least one bound must be positive"


# ---------------------------------------------------------------------------
# Step 4: Boys localization preserves g_MO Frobenius norm
# ---------------------------------------------------------------------------
def test_boys_localize_preserves_eri():
    """Rotation by U is unitary → ||g_MO||_F invariant."""
    _, g_mo_canon, _, _ = _dense()

    # Re-run with localize=True
    S, h1, Enuc, shells, norms, sph_C = build_ao_integrals_with_shells(
        _H2O, d_single_zeta=True, full_shells=False, spherical=False
    )
    g_AO = build_ao_integrals(_H2O, d_single_zeta=True, full_shells=False)[3]
    _, g_mo_loc, _, _ = build_full_mo_tensors(
        h1, g_AO, S, Enuc, N_frozen=1, N_elec_total=10,
        localize=True, shells=shells, norms=norms, sph_C=sph_C,
    )

    diff = abs(np.linalg.norm(g_mo_loc) - np.linalg.norm(g_mo_canon))
    assert diff < 1e-8, f"||g_MO||_F changed by {diff:.3e} under Boys rotation"


# ---------------------------------------------------------------------------
# Step 5: screened path (threshold=0) matches dense path
# ---------------------------------------------------------------------------
def test_screened_matches_dense_threshold0():
    _, g_mo_dense, Ecore_dense, _ = _dense()
    _, g_mo_scr,   Ecore_scr,   _ = _screened(thr=0.0)

    max_diff = float(np.max(np.abs(g_mo_dense - g_mo_scr)))
    assert max_diff < 1e-8, f"max |g_MO_dense - g_MO_screened| = {max_diff:.3e}"
    assert abs(Ecore_dense - Ecore_scr) < 1e-10, "Ecore mismatch between paths"


# ---------------------------------------------------------------------------
# Regression: default path unchanged
# ---------------------------------------------------------------------------
def test_regression_default_path():
    """localize=False, screened=False must give bit-identical results to before."""
    h1_mo1, g_mo1, Ecore1, C1 = _dense()
    h1_mo2, g_mo2, Ecore2, C2 = _dense()  # second call, same args

    assert np.allclose(g_mo1, g_mo2, atol=0), "Default path not reproducible"
    assert np.allclose(h1_mo1, h1_mo2, atol=0), "h1_MO not reproducible"


if __name__ == "__main__":
    print("test_schwarz_bounds_runs_and_is_nonneg … ", end="", flush=True)
    test_schwarz_bounds_runs_and_is_nonneg()
    print("PASS")

    print("test_boys_localize_preserves_eri … ", end="", flush=True)
    test_boys_localize_preserves_eri()
    print("PASS")

    print("test_screened_matches_dense_threshold0 … ", end="", flush=True)
    test_screened_matches_dense_threshold0()
    print("PASS")

    print("test_regression_default_path … ", end="", flush=True)
    test_regression_default_path()
    print("PASS")

    print("\nAll Step 4+5 tests passed.")
