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
test_ecp.py — effective-core-potential (ECP) support (Step 3)
=============================================================
Self-consistency validation of the numerical ECP engine (no external
reference needed):
  * loader parses the def2 ECP (n_core, radial blocks);
  * the C-centred grid + GTO evaluation reproduces the analytic overlap S
    (local form with U=1);
  * the angular projectors are complete: Σ_l P_l = 1 reproduces S;
  * Z_eff bookkeeping (Mo → 14, all-electron atoms unchanged, no ECP centre);
  * V^ECP is symmetric, finite, and core-repulsive (positive diagonal).

The absolute ECP magnitude/convention (r^{n-2}, sign) is NOT validated here —
it should be cross-checked once against a PySCF def2-TZVP+ECP reference before
publication use.  The sign (repulsive core) and structure are checked.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import nanoprotogeny.molecular.mqeaointegrals as I        # noqa: E402
import nanoprotogeny.molecular.mqeecp as E                # noqa: E402
from nanoprotogeny.basis import mqebasisloader as L       # noqa: E402
from nanoprotogeny.molecular.mqeaobuild import _effective_charges  # noqa: E402


def _smooth_basis():
    O = np.zeros(3)
    shells = [(O, (0, 0, 0), [0.6], [1.0]), (O, (0, 0, 0), [0.25], [1.0]),
              (O, (1, 0, 0), [0.5], [1.0]), (O, (0, 1, 0), [0.5], [1.0]),
              (O, (0, 0, 1), [0.5], [1.0]), (O, (2, 0, 0), [0.4], [1.0]),
              (O, (0, 2, 0), [0.4], [1.0]), (O, (1, 1, 0), [0.4], [1.0])]
    norms = [I._contracted_norm_cart(a, c, *ang) for (_, ang, a, c) in shells]
    n = len(shells)
    S = np.zeros((n, n))
    for i, (A, la, am, dm) in enumerate(shells):
        for j, (B, lb, an, dn) in enumerate(shells):
            v = sum(da * db * I._overlap_prim_cart(a, A, la, b, B, lb)
                    for a, da in zip(am, dm) for b, db in zip(an, dn))
            S[i, j] = v / (norms[i] * norms[j])
    return shells, norms, S


def test_loader_parses_def2_ecp():
    n_core, blocks = L.load_bse_ecp("def2-TZVP", "Mo")
    assert n_core == 28
    ls = [l for l, _ in blocks]
    assert max(ls) == 3                       # local part is the highest l
    assert set(ls) == {0, 1, 2, 3}
    # no ECP for a light all-electron element
    assert L.ecp_core_electrons("cc-pVTZ", "C") == 0


def test_grid_reproduces_overlap():
    shells, norms, S = _smooth_basis()
    O = np.zeros(3)
    Sg = E.ecp_matrix(shells, norms, [(O, 0, [(0, [(2, 1.0, 1.0)])])],
                      nr=80, ntheta=20, nphi=40, rm=1.2, _unit_potential=True)
    assert np.max(np.abs(Sg - S)) < 1e-9


def test_angular_projectors_complete():
    shells, norms, S = _smooth_basis()
    O = np.zeros(3)
    r, wr = E._radial_grid(80, 1.2)
    dirs, wang = E._angular_grid(20, 40)
    pts = O[None, None, :] + r[:, None, None] * dirs[None, :, :]
    chi = np.array([E._eval_shell(shells[i], norms[i], pts.reshape(-1, 3))
                    for i in range(len(shells))]).reshape(len(shells), len(r), len(dirs))
    Sp = np.zeros_like(S)
    for l in range(4):
        Yl = E._sph_on_grid(l, dirs, wang)
        for k in range(len(r)):
            ck = chi[:, k, :]
            proj = (ck * wang) @ Yl.T
            Sp += wr[k] * (proj @ proj.T)
    assert np.max(np.abs(Sp - S)) < 1e-9


def test_zeff_bookkeeping():
    ze, cen = _effective_charges([("Mo", 0, 0, 0)], {"Mo": "def2-TZVP"})
    assert ze == [14.0] and len(cen) == 1 and cen[0][1] == 28
    zf, cf = _effective_charges([("Fe", 0, 0, 0), ("S", 2.2, 0, 0)],
                                {"Fe": "def2-TZVP", "S": "def2-TZVP"})
    assert zf == [26.0, 16.0] and cf == []
    # ecp disabled → no Z_eff reduction, no centres
    zn, cn = _effective_charges([("Mo", 0, 0, 0)], None)
    assert zn == [42.0] and cn == []


def test_vecp_symmetric_and_repulsive():
    atoms = [("Mo", 0.0, 0.0, 0.0)]
    shells = I._build_basis_shells(atoms, basis_spec={"Mo": "def2-TZVP"}, full_shells=True)
    norms = [I._contracted_norm_cart(a, c, *ang) for (_, ang, a, c) in shells]
    _, cen = _effective_charges(atoms, {"Mo": "def2-TZVP"})
    V = E.ecp_matrix(shells, norms, cen)
    assert np.allclose(V, V.T) and np.isfinite(V).all()
    assert np.mean(np.diag(V)) > 0           # core-repulsive


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    bad = 0
    for f in fns:
        try:
            f(); print("PASS ", f.__name__)
        except AssertionError as e:
            bad += 1; print("FAIL ", f.__name__, e)
    print(f"\n{len(fns)-bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)
