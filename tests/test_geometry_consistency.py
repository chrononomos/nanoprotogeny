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
test_geometry_consistency.py — Geometry invariants for mqegeometries.py
=======================================================================
Guards against the classes of geometry/label drift found in the 2026-06-12
audit.  Each test encodes one invariant; failures point directly at the
mechanism and the broken property.

Run under pytest::

    pytest tests/test_geometry_consistency.py -q

or standalone::

    python tests/test_geometry_consistency.py
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

# Allow running both as a module (pytest) and as a standalone script.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from nanoprotogeny.molecular import mqegeometries as G  # noqa: E402


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _composition(atoms):
    return dict(Counter(a[0] for a in atoms))


def _dist(a, b):
    return math.dist(a[1:], b[1:])


def _min_pair_distance(atoms, e1, e2):
    """Minimum distance between an e1 atom and an e2 atom (e1 may equal e2)."""
    ds = []
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n):
            pair = {atoms[i][0], atoms[j][0]}
            if (e1 != e2 and pair == {e1, e2}) or (
                e1 == e2 and atoms[i][0] == e1 and atoms[j][0] == e1
            ):
                ds.append(_dist(atoms[i], atoms[j]))
    return min(ds) if ds else None


_TOL = 1e-4


# --------------------------------------------------------------------------
# 1. Every per-step geometry builds, parses, and is finite
# --------------------------------------------------------------------------

def test_every_step_builds_and_is_finite():
    for name, bls in G.BONDLENGTHS.items():
        for step_n in range(len(bls)):
            atoms = G.get_step_geometry(name, step_n)
            assert atoms, f"{name} step {step_n}: empty geometry"
            for sym, x, y, z in atoms:
                assert sym and isinstance(sym, str), f"{name} s{step_n}: bad symbol"
                for c in (x, y, z):
                    assert math.isfinite(c), f"{name} s{step_n}: non-finite coord"
            # No two atoms coincident (would break integrals).
            for i in range(len(atoms)):
                for j in range(i + 1, len(atoms)):
                    assert _dist(atoms[i], atoms[j]) > 1e-3, (
                        f"{name} s{step_n}: coincident atoms {i},{j}"
                    )


# --------------------------------------------------------------------------
# 2. Spec / registry coverage and janus_step ranges
# --------------------------------------------------------------------------

def test_every_spec_has_bondlengths_and_resolves():
    for name in G.HAMZEROS_SPECS:
        assert name in G.BONDLENGTHS, f"{name}: spec present but no BONDLENGTHS"
        # get_hamzeros_spec must not raise for any registered mechanism.
        assert G.get_hamzeros_spec(name) is G.HAMZEROS_SPECS[name]


def test_janus_step_in_range():
    for name, spec in G.HAMZEROS_SPECS.items():
        if spec.janus_step is None:
            continue
        n = len(G.BONDLENGTHS[name])
        assert 0 <= spec.janus_step < n, (
            f"{name}: janus_step={spec.janus_step} out of range [0,{n})"
        )


def test_nitrogenase_lt_closed_spec_present():
    # Regression: this alias was in BONDLENGTHS but had no HamZerosSpec.
    assert "nitrogenase_lt_closed" in G.HAMZEROS_SPECS
    G.get_hamzeros_spec("nitrogenase_lt_closed")


# --------------------------------------------------------------------------
# 3. Janus single-point geometry == per-step builder at janus_step
#    (the delegation invariant that replaced the divergent static registry)
# --------------------------------------------------------------------------

def test_janus_geometry_matches_per_step_builder():
    for name, spec in G.HAMZEROS_SPECS.items():
        js = spec.janus_step if spec.janus_step is not None else 0
        n = len(G.BONDLENGTHS[name])
        if not (0 <= js < n):
            js = 0
        janus = G.get_janus_geometry(name)
        per_step = G.get_step_geometry(name, js)
        assert janus == per_step, (
            f"{name}: janus geometry diverges from per-step builder at step {js}"
        )


def test_unknown_mechanism_falls_back_without_recursion():
    atoms = G.get_janus_geometry("definitely_not_a_mechanism")
    assert _composition(atoms) == {"H": 4}


# --------------------------------------------------------------------------
# 4. Colinear builders: the passed bondlength IS the named primary bond
#    (these builders place the two atoms so distance == bondlength exactly)
# --------------------------------------------------------------------------

def test_colinear_builders_honor_bondlength():
    cases = [
        # (mechanism, element_a, element_b, bondlength)
        ("hydrogenase",    "H",  "H",  0.900),   # H–H == bondlength
        ("mo_nitrogenase", "Fe", "Mo", 2.650),   # Fe–Mo == bondlength
        ("femon2_trimer",  "N",  "N",  1.300),   # N–N  == bondlength
        ("cu_co2rr",       "Cu", "Cu", 2.500),   # Cu–Cu (min) == bondlength
        ("assimilatory_nr","Mo", "S",  2.380),   # Mo–S == bondlength
    ]
    for mech, ea, eb, bl in cases:
        atoms = G.get_step_geometry(mech, 0, bondlength=bl)
        d = _min_pair_distance(atoms, ea, eb)
        assert d is not None, f"{mech}: no {ea}-{eb} pair found"
        assert abs(d - bl) < _TOL, (
            f"{mech}: {ea}-{eb} = {d:.6f} Å but bondlength was {bl:.6f} Å"
        )


def test_step_builders_respond_to_bondlength_where_expected():
    # Builders parameterised by bondlength must change when it changes.
    for mech in ["nitrogenase_lt", "hydrogenase", "mo_nitrogenase",
                 "cu_co2rr", "femon2_trimer", "v_nitrogenase"]:
        a = G.get_step_geometry(mech, 0, bondlength=1.234)
        b = G.get_step_geometry(mech, 0, bondlength=2.345)
        assert a != b, f"{mech}: geometry ignores the bondlength argument"


# --------------------------------------------------------------------------
# Standalone runner
# --------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
