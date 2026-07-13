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
nanoprotogeny.ionq.ionqurgate  — BACKWARD-COMPATIBILITY SHIM
=============================================================
This module has been superseded by the split:

    tetralemmatics.py      — platform-independent mathematical kernel
    ionqtetralemmatics.py  — IonQ Forte-specific compiled sequences

All symbols previously defined here are now re-exported from those two
modules.  Existing import sites continue to work without modification.

New code should import directly from the canonical source:

    from nanoprotogeny.ionq.tetralemmatics    import B_LOG, B_VIRT, ...
    from nanoprotogeny.ionq.ionqtetralemmatics import B_LOG_OPS, ...
    from nanoprotogeny.ionq.ionqBLOGgate       import BLOG, BLOG_DAG
    from nanoprotogeny.ionq.ionqBVIRTgate      import BVIRT, BVIRT_DAG
"""

# Re-export everything from the canonical modules
from nanoprotogeny.ionq.ionqtetralemmatics import *          # noqa: F401, F403
from nanoprotogeny.ionq.ionqtetralemmatics import (          # explicit for linters
    B_LOG, B_VIRT,
    U_R_shift_onto, Z_clock_onto, DFT_onto,
    UR_phys_log, Z_phys_log, DFT_phys_log,
    UR_phys_virt, Z_phys_virt, DFT_phys_virt,
    DFT_phys_log_inv, DFT_phys_virt_inv,
    get_physical_matrix, apply_basis_ops,
    B_LOG_DAG_OPS, B_LOG_OPS,
    B_VIRT_DAG_OPS, B_VIRT_OPS,
    PhysicalURWrapper, PhysicalZClockWrapper,
    PhysicalDFTWrapper, InversePhysicalDFTWrapper,
    TetralemmaticIonURShiftGate, TetralemmaticIonZClockGate,
    TetralemmaticIonDFTGate, TetralemmaticIonInverseDFTGate,
    expand_qudit_circuit, compile_tetralemmatic_ionq,
    TetralemmaticIonURgates, AnyQudit,
    omega,
)

# Preserve DFT_onto under its legacy alias used by ionqparamgates.py
DFT_onto = DFT_onto  # noqa: F811  (already imported above via *)
