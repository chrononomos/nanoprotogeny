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
test_full_basis_integrals.py — full untruncated basis + f/g integral engine
===========================================================================
Guards Step 1 of the non-classical full-Hamiltonian build:
  * _CART_COMPS covers l=0..4 (s,p,d,f,g);
  * the McMurchie-Davidson primitives are correct for f-shells
    (normalization; rotational/translational invariance of the spectrum;
    batch == scalar ERI);
  * the BSE loader expands general contractions (cc-pVTZ) — no column dropped;
  * full_shells=True yields the genuine full basis count (def2 and cc);
  * the legacy truncated path is unchanged.

Run: pytest tests/test_full_basis_integrals.py -q   (or: python tests/...py)
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import nanoprotogeny.molecular.mqeaointegrals as I            # noqa: E402
from nanoprotogeny.basis import mqebasisloader as L           # noqa: E402
from nanoprotogeny.molecular import mqegeometries as G        # noqa: E402


def test_cart_comps_cover_l0_to_l4():
    assert len(I._CART_COMPS[3]) == 10
    assert len(I._CART_COMPS[4]) == 15
    for l in range(5):
        for (a, b, c) in I._CART_COMPS[l]:
            assert a + b + c == l


def test_self_overlap_normalized_all_l():
    O = np.zeros(3)
    for l in range(5):
        for la in I._CART_COMPS[l]:
            s = I._overlap_prim_cart(0.8, O, la, 0.8, O, la)
            assert abs(s - 1.0) < 1e-9, f"l={l} {la}: <f|f>={s}"


def _build_1e_fshells(centers, alpha=0.9, Z=3.0):
    """S, h1 for one uncontracted f-shell on each centre (isolates l=3 code)."""
    shells = [(np.array(c), ang, [alpha], [1.0])
              for c in centers for ang in I._CART_COMPS[3]]
    nuc = [(Z, np.array(c)) for c in centers]
    norms = [I._contracted_norm_cart(a, co, *ang) for (_, ang, a, co) in shells]
    n = len(shells)
    S = np.zeros((n, n)); H = np.zeros((n, n))
    for mu, (A, la, am, dm) in enumerate(shells):
        for nu, (B, lb, an, dn) in enumerate(shells):
            s, t, v = I._build_1e_contracted_cart(am, dm, A, la, an, dn, B, lb, nuc)
            nn = norms[mu] * norms[nu]
            S[mu, nu] = s / nn; H[mu, nu] = (t + v) / nn
    return S, H


def test_f_shell_rotation_translation_invariance():
    C0 = [(0.0, 0.0, 0.0), (0.0, 0.0, 2.1)]
    S, H = _build_1e_fshells(C0)
    assert np.linalg.eigvalsh(S).min() > 1e-10          # positive-definite
    ev = eigh(H, S, eigvals_only=True)
    # rotation
    rng = np.random.default_rng(7)
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3))); Q *= np.sign(np.linalg.det(Q))
    Sr, Hr = _build_1e_fshells([tuple(Q @ np.array(c)) for c in C0])
    assert np.max(np.abs(ev - eigh(Hr, Sr, eigvals_only=True))) < 1e-10
    # translation
    St, Ht = _build_1e_fshells([(c[0]+2.3, c[1]-1.1, c[2]+0.7) for c in C0])
    assert np.max(np.abs(ev - eigh(Ht, St, eigvals_only=True))) < 1e-10


def test_batch_eri_matches_scalar_with_f():
    rng = np.random.default_rng(0)

    def rcomp():
        l = int(rng.integers(0, 4)); cs = I._CART_COMPS[l]
        return cs[int(rng.integers(0, len(cs)))]

    maxerr = 0.0
    for _ in range(25):
        la, lb, lc, ld = rcomp(), rcomp(), rcomp(), rcomp()
        a, b, c, d = (float(rng.uniform(0.4, 2.0)) for _ in range(4))
        A, B, C, D = (rng.uniform(-1, 1, 3) for _ in range(4))
        scal = I._eri_prim_cart(a, A, la, b, B, lb, c, C, lc, d, D, ld)
        bat = I._eri_prim_cart_batch(
            la, lb, lc, ld,
            np.array([a]), A[None, :], np.array([b]), B[None, :],
            np.array([c]), C[None, :], np.array([d]), D[None, :])[0]
        maxerr = max(maxerr, abs(scal - bat))
    assert maxerr < 1e-10


def test_loader_expands_general_contractions():
    # cc-pVTZ C is generally contracted: 4 s-columns, 3 p, 2 d, 1 f.
    kinds = Counter(s[0] for s in L.load_bse_basis("cc-pVTZ", "C"))
    assert kinds == Counter({"S": 4, "P": 3, "D": 2, "F": 1})


def _nao_full(atoms, spec):
    return len(I._build_basis_shells(atoms, basis_spec=spec, full_shells=True))


def test_full_basis_counts():
    fe2s2 = G.get_step_geometry("nitrogenase_lt", 4)
    # def2-TZVP is segmented → loader change is a no-op → 188 cartesian.
    assert _nao_full(fe2s2, {"Fe": "def2-TZVP", "S": "def2-TZVP"}) == 188
    # cc-pVTZ C expands to 4s+3p+2d+1f = 4+9+12+10 = 35 cartesian.
    assert _nao_full([("C", 0, 0, 0)], {"C": "cc-pVTZ"}) == 35


def test_legacy_truncated_path_unchanged():
    fe2s2 = G.get_step_geometry("nitrogenase_lt", 4)
    n = len(I._build_basis_shells(fe2s2, basis_spec={"Fe": "def2-TZVP", "S": "def2-TZVP"}))
    assert n == 20   # outermost-s/p + single-zeta-d, as before


def test_spherical_harmonic_dims_and_orthonormality():
    for l, dim in [(2, 5), (3, 7), (4, 9)]:
        assert I._harmonic_null(l).shape[1] == dim
        T = I._shell_sph_T(l, [0.9, 0.3], [0.6, 0.5])
        S0 = I._cart_self_overlap([0.9, 0.3], [0.6, 0.5], I._CART_COMPS[l])
        assert T.shape == (len(I._CART_COMPS[l]), dim)
        assert np.allclose(T.T @ S0 @ T, np.eye(dim), atol=1e-10)


def test_spherical_count_fe2s2():
    sh = I._build_basis_shells(G.get_step_geometry("nitrogenase_lt", 4),
                               basis_spec={"Fe": "def2-TZVP", "S": "def2-TZVP"},
                               full_shells=True)
    C = I.cart_to_sph_transform(sh)
    assert C.shape == (188, 164)   # cartesian → spherical (24 contaminants removed)


def test_spherical_pipeline_rotation_invariance():
    from nanoprotogeny.molecular.mqeaobuild import build_ao_integrals
    spec = {"C": "def2-SVP", "O": "def2-SVP"}
    atoms = [("C", 0, 0, 0), ("O", 0, 0, 1.13)]
    S, h1, g, _ = build_ao_integrals(atoms, basis_spec=spec, full_shells=True, spherical=True)
    assert np.allclose(np.diag(S), 1.0, atol=1e-8)
    assert np.linalg.eigvalsh(S).min() > 1e-8
    ev = eigh(h1, S, eigvals_only=True)
    rng = np.random.default_rng(5)
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3))); Q *= np.sign(np.linalg.det(Q))
    S2, h2, _, _ = build_ao_integrals([(s, *(Q @ np.array([x, y, z]))) for s, x, y, z in atoms],
                                      basis_spec=spec, full_shells=True, spherical=True)
    assert np.max(np.abs(ev - eigh(h2, S2, eigvals_only=True))) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for f in fns:
        try:
            f(); print("PASS ", f.__name__)
        except AssertionError as e:
            bad += 1; print("FAIL ", f.__name__, e)
    print(f"\n{len(fns)-bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)
