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
nanoprotogeny.theory.algebra

Implements the Tetralemmatic Algebra 𝔏 = ⟨Λ, ⊕, ⊗, Δ⟩ and derived operations
(Subtraction ⊖, Division ⊘, Exponentiation ^) as defined in sparkChap2B.md.
"""
from enum import Enum, auto
from typing import Set, List, Tuple, Callable
import itertools

class Vertex(Enum):
    r"""The four primitive ontological vertices of the Tetralemmatic Lens Λ."""
    Th     = 0
    AntiTh = 1
    SynTh  = 2
    HoloTh = 3

    def __repr__(self) -> str:
        return self.name

# =============================================================================
# CORE ALGEBRAIC OPERATIONS
# =============================================================================

def oplus(x: Vertex, y: Vertex) -> Vertex:
    r"""Logical Addition (⊕): Commutative, idempotent join.
    - Identity: HoloTh ⊕ x = x
    - Absorption: SynTh ⊕ x = SynTh
    - Dyadic: Th ⊕ AntiTh = SynTh
    """
    if x == Vertex.HoloTh: return y
    if y == Vertex.HoloTh: return x
    if x == Vertex.SynTh or y == Vertex.SynTh:
        return Vertex.SynTh
    if {x, y} == {Vertex.Th, Vertex.AntiTh}:
        return Vertex.SynTh
    # Idempotence for identical inputs
    if x == y: return x
    raise ValueError("Undefined logical addition.")

def otimes(x: Vertex, y: Vertex) -> Vertex:
    r"""Logical Multiplication (⊗): Commutative, idempotent meet.
    - Identity: SynTh ⊗ x = x
    - Absorption: HoloTh ⊗ x = HoloTh
    - Orthogonality: Th ⊗ AntiTh = HoloTh
    - Filtering: Pole ⊗ SynTh = Pole
    """
    if x == Vertex.SynTh: return y
    if y == Vertex.SynTh: return x
    if x == Vertex.HoloTh or y == Vertex.HoloTh:
        return Vertex.HoloTh
    if {x, y} == {Vertex.Th, Vertex.AntiTh}:
        return Vertex.HoloTh
    # Idempotence/Filtering for identical inputs
    if x == y: return x
    raise ValueError("Undefined logical multiplication.")

def delta(x: Vertex) -> Vertex:
    r"""Duality Involution (Δ): Structure-preserving automorphism.
    Δ(Th) ↔ AntiTh, Δ(SynTh) = SynTh, Δ(HoloTh) = HoloTh
    """
    if x == Vertex.Th: return Vertex.AntiTh
    if x == Vertex.AntiTh: return Vertex.Th
    return x  # SynTh and HoloTh are self-dual

# =============================================================================
# DERIVED OPERATIONS
# =============================================================================

def ominus(x: Vertex, y: Vertex) -> Vertex:
    r"""Logical Subtraction (⊖): Contextual exclusion.
    x ⊖ y := x ⊗ Δ(y)
    """
    return otimes(x, delta(y))

def leq(x: Vertex, y: Vertex) -> bool:
    r"""Partial Order (≤): Induced by the join operation.
    x ≤ y ⇔ x ⊕ y = y
    """
    return oplus(x, y) == y

def oslash(x: Vertex, y: Vertex) -> Vertex:
    r"""Logical Division/Residuation (⊘): Material implication.
    x ⊘ y := max{ z ∈ Λ : z ⊗ y ≤ x }
    """
    candidates = [z for z in Vertex if leq(otimes(z, y), x)]
    if not candidates:
        raise ValueError("Residuation undefined for given operands.")
    # Find maximal element under ≤
    def is_maximal(c: Vertex) -> bool:
        return all(leq(other, c) for other in candidates)
    # In this finite diamond lattice, the maximal element is unique
    return next(c for c in candidates if is_maximal(c))

def power(x: Vertex, y: Vertex) -> Vertex:
    r"""Logical Exponentiation (^): Entailment stability.
    x^y := Δ(y) ⊕ x
    """
    return oplus(delta(y), x)

# =============================================================================
# ALGEBRAIC VERIFICATION SUITE
# =============================================================================

def verify_closure() -> bool:
    r"""Proposition: Closure and Atomicity. 𝔏 is closed under ⊕, ⊗, Δ."""
    for x, y in itertools.product(Vertex, repeat=2):
        if oplus(x, y) not in Vertex: return False
        if otimes(x, y) not in Vertex: return False
        if delta(x) not in Vertex: return False
    return True

def verify_duality_symmetry() -> bool:
    r"""Proposition: Duality Symmetry. Δ distributes over ⊕ and ⊗."""
    for x, y in itertools.product(Vertex, repeat=2):
        # Δ(x ⊕ y) = Δ(x) ⊕ Δ(y)
        if delta(oplus(x, y)) != oplus(delta(x), delta(y)):
            return False
        # Δ(x ⊗ y) = Δ(x) ⊗ Δ(y)
        if delta(otimes(x, y)) != otimes(delta(x), delta(y)):
            return False
    return True

def verify_subtraction_properties() -> Tuple[bool, List[str]]:
    r"""Proposition: Properties of Logical Subtraction."""
    failures = []
    # (i) Polar Annihilation: x ⊖ x = HoloTh for poles
    for x in [Vertex.Th, Vertex.AntiTh]:
        if ominus(x, x) != Vertex.HoloTh:
            failures.append(f"Polar Annihilation failed for {x}")
    # (ii) Synthesis Persistence: SynTh ⊖ SynTh = SynTh
    if ominus(Vertex.SynTh, Vertex.SynTh) != Vertex.SynTh:
        failures.append("Synthesis Persistence failed")
    # (iii) Duality Symmetry: Δ(x ⊖ y) = Δ(x) ⊖ Δ(y)
    for x, y in itertools.product(Vertex, repeat=2):
        if delta(ominus(x, y)) != ominus(delta(x), delta(y)):
            failures.append(f"Subtraction Duality failed for {x}, {y}")
    # (iv) Thesis-Antithesis Identity
    if ominus(Vertex.Th, Vertex.AntiTh) != Vertex.Th:
        failures.append("Th ⊖ AntiTh != Th")
    if ominus(Vertex.AntiTh, Vertex.Th) != Vertex.AntiTh:
        failures.append("AntiTh ⊖ Th != AntiTh")
    return len(failures) == 0, failures

def verify_division_properties() -> Tuple[bool, List[str]]:
    r"""Proposition: Properties of Logical Division."""
    failures = []
    # (i) Reflexivity: x ⊘ x = SynTh
    for x in Vertex:
        if oslash(x, x) != Vertex.SynTh:
            failures.append(f"Reflexivity failed for {x}")
    # (ii) Top Element: SynTh ⊘ x = SynTh
    for x in Vertex:
        if oslash(Vertex.SynTh, x) != Vertex.SynTh:
            failures.append(f"Top Element failed for {x}")
    # (iii) Bottom Element: x ⊘ HoloTh = SynTh
    for x in Vertex:
        if oslash(x, Vertex.HoloTh) != Vertex.SynTh:
            failures.append(f"Bottom Element failed for {x}")
    # (iv) Implication Duality: Δ(x ⊘ y) = Δ(y) ⊘ Δ(x)
    for x, y in itertools.product(Vertex, repeat=2):
        if delta(oslash(x, y)) != oslash(delta(y), delta(x)):
            failures.append(f"Division Duality failed for {x}, {y}")
    return len(failures) == 0, failures

def verify_exponentiation_properties() -> Tuple[bool, List[str]]:
    r"""Proposition: Properties of Logical Exponentiation."""
    failures = []
    # (i) Self-Entailment: x^x = SynTh for poles
    for x in [Vertex.Th, Vertex.AntiTh]:
        if power(x, x) != Vertex.SynTh:
            failures.append(f"Self-Entailment failed for {x}")
    # (ii) Holothesis Transparency
    for x in Vertex:
        if power(x, Vertex.HoloTh) != x:
            failures.append(f"Holothesis Transparency (x^HoloTh) failed for {x}")
        if power(Vertex.HoloTh, x) != delta(x):
            failures.append(f"Holothesis Transparency (HoloTh^x) failed for {x}")
    # (iii) Synthesis Limit
    for y in Vertex:
        if power(Vertex.SynTh, y) != Vertex.SynTh:
            failures.append(f"Synthesis Limit failed for {y}")
    return len(failures) == 0, failures

# =============================================================================
# ERGONOMIC OPERATOR OVERLOADING (Optional)
# =============================================================================

class _TetralemmaticOps:
    """Namespace mapping standard operators to logical operations for ergonomic use."""
    @staticmethod
    def __add__(x: Vertex, y: Vertex) -> Vertex: return oplus(x, y)
    @staticmethod
    def __mul__(x: Vertex, y: Vertex) -> Vertex: return otimes(x, y)
    @staticmethod
    def __invert__(x: Vertex) -> Vertex: return delta(x)
    @staticmethod
    def __sub__(x: Vertex, y: Vertex) -> Vertex: return ominus(x, y)
    @staticmethod
    def __truediv__(x: Vertex, y: Vertex) -> Vertex: return oslash(x, y)
    @staticmethod
    def __pow__(x: Vertex, y: Vertex) -> Vertex: return power(x, y)
    @staticmethod
    def __le__(x: Vertex, y: Vertex) -> bool: return leq(x, y)

# Bind dunder methods to Vertex for seamless algebraic notation
for name in ["__add__", "__mul__", "__invert__", "__sub__", "__truediv__", "__pow__", "__le__"]:
    setattr(Vertex, name, getattr(_TetralemmaticOps, name))

# =============================================================================
# MODULE EXPORTS
# =============================================================================
__all__ = [
    "Vertex",
    "oplus",
    "otimes",
    "delta",
    "ominus",
    "leq",
    "oslash",
    "power",
    "verify_closure",
    "verify_duality_symmetry",
    "verify_subtraction_properties",
    "verify_division_properties",
    "verify_exponentiation_properties"
]