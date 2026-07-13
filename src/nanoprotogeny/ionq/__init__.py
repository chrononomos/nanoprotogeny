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
nanoprotogeny.ionq
Tetralemmatic IonQ Interface Package.
Provides Bell-separable encoding, hardware-specific compilation,
and semantic evaluation layers for ¹⁷¹Yb⁺ trapped-ion systems.
All gates now support both Logical Manifold (NomosIonQid) and
Virtual Phase Register (VirtualQudit).

NOTE: The legacy `ionqurgate` module has been deprecated. Core mathematical 
operators and physical matrices are now imported from `tetralemmatics` and 
`ionqtetralemmatics`, while the fundamental basis-change gates are provided 
by `ionqBLOGgate` and `ionqBVIRTgate`.
"""
# =============================================================================
# 1. SUBMODULE IMPORTS
# =============================================================================
from antlr4.atn import ATNConfigSet
from antlr4.atn import ATNConfigSet
from nanoprotogeny.ionq.ionqmqegates import PhysicalPhotonEmissionWrapper
from nanoprotogeny.ionq import YB171PLUSHARDWARE
from nanoprotogeny.ionq import holographic
from nanoprotogeny.ionq import ionqpauligates
from nanoprotogeny.ionq import ionqphasegates
from nanoprotogeny.ionq import ionqhadamardgate
from nanoprotogeny.ionq import ionqtetralemmatics  # Replaces legacy ionqurgate
from nanoprotogeny.ionq import ionqBLOGgate        # Logical basis-change
from nanoprotogeny.ionq import ionqBVIRTgate       # Virtual basis-change
from nanoprotogeny.ionq import ionqcnotgate
from nanoprotogeny.ionq import ionqczgate
from nanoprotogeny.ionq import ionqswapgate
from nanoprotogeny.ionq import ionqcurgate
from nanoprotogeny.ionq import ionqsumgate
from nanoprotogeny.ionq import ionqtoffoligate
from nanoprotogeny.ionq import ionqvirtualgates
from nanoprotogeny.ionq import ionqcrossgates
from nanoprotogeny.ionq import ionqprojectorgate
from nanoprotogeny.ionq import ionqsemantics
from nanoprotogeny.ionq import ionqmqegates

# Pipeline & infrastructure modules
from nanoprotogeny.ionq import ionqconnectivity
from nanoprotogeny.ionq import ionqhistogram
from nanoprotogeny.ionq import ionqfortenoise
from nanoprotogeny.ionq import ionqscheduler
from nanoprotogeny.ionq import ionqjanus
from nanoprotogeny.ionq import ionqparamgates
from nanoprotogeny.ionq import ionqtrotter

# =============================================================================
# 2. HARDWARE ABSTRACTION LAYER
# =============================================================================
from nanoprotogeny.ionq.YB171PLUSHARDWARE import (
    IonManifold,
    NomosIonQid,
    NomosState,
    VirtualQudit,
    PHYS_TO_VIRTUAL_MAP,
    VIRTUAL_TO_PHYS_MAP,
    LOGICAL_LEVELS,
    AUXILIARY_LEVELS,
    VIRTUAL_OFFSET,
)

# =============================================================================
# 3. GATE IMPLEMENTATIONS & MATRICES
# =============================================================================
#----- SUM (Generalised CNOT) -----
from nanoprotogeny.ionq.ionqsumgate import (
    AnyQudit as SUMAnyQudit,
    B_LOG as SUM_B_LOG,
    B_VIRT as SUM_B_VIRT,
    PhysicalSUMWrapper,
    SUM_onto,
    SUM_phys_log,
    SUM_phys_virt,
    SUM_phys_log_inv,
    SUM_phys_virt_inv,
    TetralemmaticIonSUMGate,
    TetralemmaticIonInverseSUMGate,
)

#----- CNOT -----
from nanoprotogeny.ionq.ionqcnotgate import (
    AnyQudit,
    B_LOG,
    B_VIRT,
    CNOT_onto,
    CNOT_phys_log,
    CNOT_phys_virt,
    I4,
    P_anti_onto,
    PhysicalCNOTWrapper,
    TetralemmaticIonCNOTGate,
    TetralemmaticIonCNOTGates,
    X_onto,
)

#----- Controlled-U_R -----
from nanoprotogeny.ionq.ionqcurgate import (
    AnyQudit as CURAnyQudit,
    CONTROL_IDX,
    CUR_onto,
    CUR_phys_log,
    CUR_phys_virt,
    ControlledURIonGate,
    PhysicalCURWrapper,
    TetralemmaticIonCURGates,
    UR_onto,
)

#----- CZ -----
from nanoprotogeny.ionq.ionqczgate import (
    AnyQudit as CZAnyQudit,
    CZ_onto,
    CZ_phys_log,
    CZ_phys_virt,
    PhysicalCZWrapper,
    TetralemmaticIonCZGate,
    TetralemmaticIonCZGates,
)

#----- Hadamard -----
from nanoprotogeny.ionq.ionqhadamardgate import (
    AnyQudit as HadAnyQudit,
    H_onto,
    H_phys_log,
    H_phys_virt,
    PhysicalHadamardWrapper,
    TetralemmaticIonHadamard,
    TetralemmaticIonHadamardGate,
)

#----- Pauli Gates -----
from nanoprotogeny.ionq.ionqpauligates import (
    AnyQudit as PauliAnyQudit,
    DFT_onto,
    DFT_phys,
    DFT_virt_phys,
    PhysicalLogicalGateWrapper,
    PhysicalVirtualGateWrapper,
    TetralemmaticIonDFTGate,
    TetralemmaticIonGates,
    TetralemmaticIonURGate,
    TetralemmaticIonXGate,
    TetralemmaticIonYGate,
    TetralemmaticIonZGate,
    X_phys,
    X_virt_phys,
    Y_onto,
    Y_phys,
    Y_virt_phys,
    Z_onto,
    Z_phys,
    Z_virt_phys,
)

#----- Phase Gates -----
from nanoprotogeny.ionq.ionqphasegates import (
    AnyQudit as PhaseAnyQudit,
    PhysicalPhaseGateWrapper,
    S_onto,
    S_phys_log,
    S_phys_virt,
    T_onto,
    T_phys_log,
    T_phys_virt,
    TetralemmaticIonPhaseGates,
    TetralemmaticIonSGate,
    TetralemmaticIonTGate,
    Z_phys_log,
    Z_phys_virt,
)

#----- Projector & Algebra Gates -----
from nanoprotogeny.ionq.ionqprojectorgate import (
    AnyQudit as ProjAnyQudit,
    LOGICAL_DIM,
    TetralemmaticIonAlgebraGate,
    TetralemmaticIonProjectorGate,
    TetralemmaticIonProjectorGates,
    logical_to_physical_matrix,
)

#----- SWAP -----
from nanoprotogeny.ionq.ionqswapgate import (
    AnyQudit as SWAPAnyQudit,
    PhysicalSWAPWrapper,
    SWAP_onto,
    SWAP_phys_log,
    SWAP_phys_virt,
    TetralemmaticIonSWAPGate,
    TetralemmaticIonSWAPGates,
)

#----- Toffoli -----
from nanoprotogeny.ionq.ionqtoffoligate import (
    AnyQudit as TofAnyQudit,
    PhysicalToffoliWrapper,
    TetralemmaticIonToffoliGate,
    TetralemmaticIonToffoliGates,
    Toffoli_onto,
    Toffoli_phys_log,
    Toffoli_phys_virt,
    term1,
    term2,
)

#----- U_R / DFT / Z_Clock (Refactored from legacy ionqurgate) -----
from nanoprotogeny.ionq.ionqtetralemmatics import (
    AnyQudit as UrgAnyQudit,
    DFT_phys_log,
    DFT_phys_virt,
    PhysicalDFTWrapper,
    PhysicalURWrapper,
    PhysicalZClockWrapper,
    TetralemmaticIonDFTGate as UrDFTGate,
    TetralemmaticIonURShiftGate,
    TetralemmaticIonURgates,
    TetralemmaticIonZClockGate,
    UR_phys_log,
    UR_phys_virt,
    U_R_shift_onto,
    Z_clock_onto,
    omega,
    get_physical_matrix,
    apply_basis_ops,
    compile_tetralemmatic_ionq,
    B_LOG_OPS,
    B_LOG_DAG_OPS,
    B_VIRT_OPS,
    B_VIRT_DAG_OPS,
)

#----- Logical Basis-Change Gates -----
from nanoprotogeny.ionq.ionqBLOGgate import (
    BLOGGate,
    BLOGDagGate,
    BLOG,
    BLOG_DAG,
    wrap_logical,
)

#----- Virtual Basis-Change Gates -----
from nanoprotogeny.ionq.ionqBVIRTgate import (
    BVIRTGate,
    BVIRTDagGate,
    BVIRT,
    BVIRT_DAG,
    wrap_virtual,
    wrap_cross,
)

# =============================================================================
# 4. VIRTUAL & CROSS-MANIFOLD GATES
# =============================================================================
from nanoprotogeny.ionq.ionqvirtualgates import (
    AnyQudit as VirtAnyQudit,
    PhysicalPhaseCompensateWrapper,
    PhysicalProjectorWrapper,
    PhysicalUnitaryWrapper,
    VDFTGate,
    VDFT_onto,
    VDFT_phys_log,
    VDFT_phys_virt,
    VPhaseCompensateGate,
    VProjectorGate,
    VURShiftGate,
    VUR_onto,
    VUR_phys_log,
    VUR_phys_virt,
    VZClockGate,
    VZ_onto,
    VZ_phys_log,
    VZ_phys_virt,
    VirtualGateFactory,
    to_physical,
    compile_to_ionq_native,
)

from nanoprotogeny.ionq.ionqcrossgates import (
    B_total,
    CrossManifoldGateFactory,
    HoloAmplifyGate,
    HoloPhaseGate,
    I16,
    PhaseInterferenceGate,
    PhaseSwapGate,
    U_R_PhaseCtrlGate,
    ZenoStabilizeGate,
    logical_cross_to_physical,
    physical_to_logical,
)

# =============================================================================
# 5. MQE EXTENSION GATES (NEW)
# =============================================================================
from nanoprotogeny.ionq.ionqmqegates import (
    # Basis matrices (re-exported for convenience)
    B_LOG as MQE_B_LOG,
    B_VIRT as MQE_B_VIRT,
    B_total as MQE_B_total,
    # Utility functions
    _get_physical_1q,
    _get_physical_2q,
    _cyclic_shift_onto,
    _diagonal_phase_onto,
    _cofactor_coupling_onto,
    _cross_manifold_swap_onto,
    _n_qubits_for_m,
    _cyclic_shift_binary_padded,
    # Precomputed physical matrices
    cross_swap_phys,
    _electron_shift_phys,
    _proton_phase_phys,
    _conformational_shift_phys,
    _gen_virt_shift_phys_d4,
    _cofactor_coupling_phys_d4,
    _cofactor_coupling_binary_padded,
    # Physical wrapper classes (single-qudit, d=4 logical)
    PhysicalElectronShiftWrapper,
    PhysicalProtonPhaseWrapper,
    PhysicalPhotonAbsorptionWrapper,
    PhysicalPhotonEmissionWrapper,
    PhysicalConformationalShiftWrapper,
    # Physical wrapper classes (virtual, d=4 and general d=m)
    PhysicalGenVirtShiftWrapper_d4,
    PhysicalGenVirtShiftWrapper_general,
    # Physical wrapper classes (cross-manifold)
    PhysicalCofactorCouplingWrapper_d4,
    PhysicalCofactorCouplingWrapper_general,
    PhysicalCrossManifoldSWAPWrapper,
    # Compilation pipeline
    expand_mqe_qudit_circuit,
    compile_mqe_gates,
    # Verification suite
    verify_mqe_gate_compilation,
    print_verification_report,
    # Factory
    MQEGateFactory,
)

# =============================================================================
# 6. HOLOGRAPHIC ROUTING & COMPILATION
# =============================================================================
from nanoprotogeny.ionq.holographic import (
    RoutingDirective,
    HolographicRouter,
    compile_with_holographic_routing,
)

# Unified global compilation functions
from nanoprotogeny.ionq.ionqsumgate import (
    expand_qudit_circuit,
)

# =============================================================================
# 7. SEMANTIC OBSERVER
# =============================================================================
from nanoprotogeny.ionq.ionqsemantics import (
    P_PHYS_LOG,
    P_PHYS_VIRT,
    SemanticObserver,
    Status,
)

# Backwards Compatibility
B = B_LOG
P_PHYS = P_PHYS_LOG

# =============================================================================
# 8. PIPELINE & INFRASTRUCTURE MODULES (added during refactor)
# =============================================================================
#----- Backend configuration & connectivity -----
from nanoprotogeny.ionq.ionqconnectivity import (
    BackendMode,
    BackendConfig,
    _make_ionq_service,
    _get_retry_session,
    _save_job_manifest,
    probe_ionq_service,
    _CIRQ_IONQ_AVAILABLE,
)

#----- Histogram parsing & ontological decoding -----
from nanoprotogeny.ionq.ionqhistogram import (
    _parse_histogram_to_counts,
    _decode_physical_to_ontological,
)

#----- IonQ Forte 1 noise model -----
from nanoprotogeny.ionq.ionqfortenoise import (
    QuditDepolarizingChannel,
    ForteHardwareNoiseModel,
    build_forte_noise_model,
    FORTE_NOISE_PARAMS,
    USE_FORTE_NOISE_MODEL,
    FALLBACK_DEPOL_P,
)

#----- Qudit DAG scheduler -----
from nanoprotogeny.ionq.ionqscheduler import (
    build_qudit_dependency_dag,
    schedule_parallel_moments,
)

#----- Biochemical transition (Janus) -----
from nanoprotogeny.ionq.ionqjanus import build_biochemical_transition_circuit

#----- Parametrised Trotter gates -----
from nanoprotogeny.ionq.ionqparamgates import (
    ParamZClockGate,
    ParamURShiftGate,
    ParamCoulombPhaseGate,
    ParamExchangeGate,
    ParamScatteringGate,
    TetralemmaticIonInverseDFTGate,
)

#----- Power-conditioned control gate (canonical module) -----
from nanoprotogeny.ionq.ionqpowercontrolgate import PowerControlledGate

#----- Trotterised evolution -----
from nanoprotogeny.ionq.ionqtrotter import (
    build_trotter_evolution_circuit,
    validate_trotter_structure,
)

# =============================================================================
# 9. PACKAGE EXPORTS
# =============================================================================
__all__ = [
    # Submodules — gate layer
    "YB171PLUSHARDWARE", "holographic", "ionqcnotgate", "ionqcrossgates",
    "ionqcurgate", "ionqczgate", "ionqhadamardgate", "ionqpauligates",
    "ionqphasegates", "ionqprojectorgate", "ionqsemantics", "ionqswapgate",
    "ionqtoffoligate", "ionqvirtualgates", "ionqsumgate", "ionqmqegates",
    "ionqtetralemmatics", "ionqBLOGgate", "ionqBVIRTgate",
    # Submodules — pipeline & infrastructure
    "ionqconnectivity", "ionqhistogram", "ionqfortenoise",
    "ionqscheduler", "ionqjanus", "ionqparamgates", "ionqtrotter",
    # Hardware abstraction
    "IonManifold", "NomosIonQid", "NomosState", "VirtualQudit",
    "PHYS_TO_VIRTUAL_MAP", "VIRTUAL_TO_PHYS_MAP", "LOGICAL_LEVELS",
    "AUXILIARY_LEVELS", "VIRTUAL_OFFSET",
    # Core matrices & constants
    "AnyQudit", "B", "B_LOG", "B_VIRT", "B_total", "I4", "I16",
    "X_onto", "X_phys", "X_virt_phys", "Z_onto", "Z_phys", "Z_phys_log",
    "Z_phys_virt", "Z_virt_phys", "Y_onto", "Y_phys", "Y_virt_phys",
    "UR_onto", "UR_phys", "UR_phys_log", "UR_phys_virt", "UR_virt_phys",
    "U_R_shift_onto", "DFT_onto", "DFT_phys", "DFT_phys_log", "DFT_phys_virt",
    "DFT_virt_phys", "S_onto", "S_phys_log", "S_phys_virt", "T_onto",
    "T_phys_log", "T_phys_virt", "CNOT_onto", "CNOT_phys_log", "CNOT_phys_virt",
    "CZ_onto", "CZ_phys_log", "CZ_phys_virt", "SWAP_onto", "SWAP_phys_log",
    "SWAP_phys_virt", "Toffoli_onto", "Toffoli_phys_log", "Toffoli_phys_virt",
    "SUM_onto", "SUM_phys_log", "SUM_phys_virt", "SUM_phys_log_inv", "SUM_phys_virt_inv",
    "CUR_onto", "CUR_phys_log", "CUR_phys_virt", "CONTROL_IDX", "P_anti_onto", "omega",
    # MQE extension: basis matrices (namespaced to avoid collision)
    "MQE_B_LOG", "MQE_B_VIRT", "MQE_B_total",
    # MQE extension: utility functions
    "_get_physical_1q", "_get_physical_2q", "_cyclic_shift_onto",
    "_diagonal_phase_onto", "_cofactor_coupling_onto", "_cross_manifold_swap_onto",
    "_n_qubits_for_m", "_cyclic_shift_binary_padded",
    # MQE extension: precomputed physical matrices
    "cross_swap_phys", "_electron_shift_phys", "_proton_phase_phys",
    "_conformational_shift_phys", "_gen_virt_shift_phys_d4",
    "_cofactor_coupling_phys_d4", "_cofactor_coupling_binary_padded",
    # Physical wrapper classes (existing gates)
    "PhysicalCNOTWrapper", "PhysicalCURWrapper", "PhysicalCZWrapper",
    "PhysicalHadamardWrapper", "PhysicalLogicalGateWrapper",
    "PhysicalVirtualGateWrapper", "PhysicalPhaseGateWrapper",
    "PhysicalSWAPWrapper", "PhysicalToffoliWrapper", "PhysicalURWrapper",
    "PhysicalZClockWrapper", "PhysicalDFTWrapper", "PhysicalUnitaryWrapper",
    "PhysicalProjectorWrapper", "PhysicalPhaseCompensateWrapper",
    "PhysicalSUMWrapper", "PhysicalPhotonAbsorptionWrapper",
    "PhysicalPhotonEmissionWrapper",
    # Physical wrapper classes (MQE extension gates)
    "PhysicalElectronShiftWrapper",
    "PhysicalProtonPhaseWrapper",
    "PhysicalConformationalShiftWrapper",
    "PhysicalGenVirtShiftWrapper_d4",
    "PhysicalGenVirtShiftWrapper_general",
    "PhysicalCofactorCouplingWrapper_d4",
    "PhysicalCofactorCouplingWrapper_general",
    "PhysicalCrossManifoldSWAPWrapper",
    # Gate classes (existing)
    "TetralemmaticIonCNOTGate", "TetralemmaticIonCNOTGates", "ControlledURIonGate",
    "TetralemmaticIonCURGates", "TetralemmaticIonCZGate", "TetralemmaticIonCZGates",
    "TetralemmaticIonHadamardGate", "TetralemmaticIonHadamard",
    "TetralemmaticIonXGate", "TetralemmaticIonYGate", "TetralemmaticIonZGate",
    "TetralemmaticIonURGate", "TetralemmaticIonDFTGate", "TetralemmaticIonGates",
    "TetralemmaticIonSGate", "TetralemmaticIonTGate", "TetralemmaticIonPhaseGates",
    "TetralemmaticIonProjectorGate", "TetralemmaticIonAlgebraGate",
    "TetralemmaticIonProjectorGates", "TetralemmaticIonSWAPGate",
    "TetralemmaticIonSWAPGates", "TetralemmaticIonToffoliGate",
    "TetralemmaticIonToffoliGates", "TetralemmaticIonURShiftGate",
    "TetralemmaticIonZClockGate", "TetralemmaticIonURgates",
    "TetralemmaticIonSUMGate", "TetralemmaticIonInverseSUMGate",
    # Virtual gate classes
    "VURShiftGate", "VZClockGate", "VDFTGate", "VProjectorGate",
    "VPhaseCompensateGate", "VirtualGateFactory", "VUR_onto", "VUR_phys_log",
    "VUR_phys_virt", "VZ_onto", "VZ_phys_log", "VZ_phys_virt", "VDFT_onto",
    "VDFT_phys_log", "VDFT_phys_virt",
    # Cross-manifold gate classes
    "PhaseSwapGate", "U_R_PhaseCtrlGate", "HoloAmplifyGate",
    "PhaseInterferenceGate", "HoloPhaseGate", "ZenoStabilizeGate",
    "CrossManifoldGateFactory",
    # Holographic routing & semantic observer
    "RoutingDirective", "HolographicRouter",
    "SemanticObserver", "Status", "P_PHYS", "P_PHYS_LOG", "P_PHYS_VIRT",
    # Compilation & expansion functions
    "compile_to_ionq_native", "expand_qudit_circuit", "compile_with_holographic_routing",
    "get_physical_matrix", "logical_to_physical_matrix", "logical_cross_to_physical",
    "physical_to_logical", "to_physical", "compile_tetralemmatic_ionq",
    # MQE extension: compilation pipeline
    "expand_mqe_qudit_circuit",
    "compile_mqe_gates",
    # MQE extension: verification
    "verify_mqe_gate_compilation",
    "print_verification_report",
    # MQE extension: factory
    "MQEGateFactory",
    # Basis-change sequences & helpers
    "B_LOG_OPS", "B_LOG_DAG_OPS", "B_VIRT_OPS", "B_VIRT_DAG_OPS",
    "apply_basis_ops",
    # Basis-change gate classes & singletons
    "BLOGGate", "BLOGDagGate", "BLOG", "BLOG_DAG", "wrap_logical",
    "BVIRTGate", "BVIRTDagGate", "BVIRT", "BVIRT_DAG", "wrap_virtual", "wrap_cross",
    # Miscellaneous
    "term1", "term2",
    # Backend configuration & connectivity
    "BackendMode", "BackendConfig",
    "_make_ionq_service", "_get_retry_session", "_save_job_manifest",
    "probe_ionq_service", "_CIRQ_IONQ_AVAILABLE",
    # Histogram parsing & ontological decoding
    "_parse_histogram_to_counts", "_decode_physical_to_ontological",
    # Forte noise model
    "QuditDepolarizingChannel", "ForteHardwareNoiseModel", "build_forte_noise_model",
    "FORTE_NOISE_PARAMS", "USE_FORTE_NOISE_MODEL", "FALLBACK_DEPOL_P",
    # Scheduler
    "build_qudit_dependency_dag", "schedule_parallel_moments",
    # Janus / biochemical transition
    "build_biochemical_transition_circuit",
    # Parametrised gates
    "ParamZClockGate", "ParamURShiftGate", "ParamCoulombPhaseGate",
    "ParamExchangeGate", "ParamScatteringGate",
    "TetralemmaticIonInverseDFTGate", "PowerControlledGate",
    # Trotterised evolution
    "build_trotter_evolution_circuit", "validate_trotter_structure",
]