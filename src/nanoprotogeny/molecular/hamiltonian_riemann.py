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
hamiltonian_riemann.py — compatibility shim
===========================================
This module is superseded by :mod:`nanoprotogeny.molecular.mqeprotogeny`.
All public names are re-exported here for backwards compatibility.

.. deprecated::
    Import from ``nanoprotogeny.molecular.mqeprotogeny`` directly.
"""

from nanoprotogeny.molecular.mqeprotogeny import (  # noqa: F401
    ZetaZeroResult,
    run_zetazero_pipeline,
    run_zetazero_for_spec,
    run_zetazero_all,
    write_zetazero_dataset,
    save_seed_tensors,
    load_seed_tensors,
    _detect_lv_crossing,
    _dense_to_sparse_integrals,
    _save_result,
    _get_geometry,
    _parse_atom_block,
    _get_step_geometry,
)
