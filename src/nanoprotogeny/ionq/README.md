# Quantum Logic Gates — Tetralemmatic IonQ Forte

This directory implements a **d = 4 qudit** gate set for trapped-ion quantum computing on the **IonQ Forte** platform using **¹⁷¹Yb⁺** ions. Each abstract qudit is encoded into **two physical qubits** via a Bell-separable mapping, and every gate follows the **sandwich pattern**:

$$U_{\text{phys}} = B^{\dagger} \; U_{\text{onto}} \; B$$

where $B$ is a basis-change matrix and $U_{\text{onto}}$ is the abstract ontological operation.

---

## Two Manifolds

The framework maintains two parallel d = 4 register types:

| Manifold | Class | Levels | Physical encoding |
|---|---|---|---|
| **Logical** | `NomosIonQid` | Th, AntiTh, SynTh, HoloTh | S₁/₂ hyperfine clock states |
| **Virtual** | `VirtualQudit` | F, P, M, R | Metastable D₃/₂ shelving states |

The basis-change matrices that map abstract ontological states to two-qubit Bell-separable physical states are:

$$B_{\text{LOG}} = \begin{pmatrix} 1 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ 0 & 1 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ 0 & 1 & -\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ 1 & 0 & -\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \end{pmatrix}, \quad B_{\text{VIRT}} = \begin{pmatrix} \frac{1}{\sqrt{2}} & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ -\frac{1}{\sqrt{2}} & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ 0 & 1 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ 0 & -1 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \end{pmatrix}$$

The columns correspond to the Bell states $\{|00\rangle, |11\rangle, |\Psi^+\rangle, |\Psi^-\rangle\}$.

---

## Fundamental Generators

Three unitary operators generate the full single-qudit gate algebra on d = 4:

### Quarter-Turn Cyclic Shift $U_R$

The cyclic permutation $|k\rangle \mapsto |(k+1) \bmod 4\rangle$:

$$U_R = \begin{pmatrix} 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{pmatrix}, \qquad U_R^4 = I$$

This is the fundamental generator of the d = 4 cyclic group. It cycles the four ontological vertices: $\text{Th} \to \text{SynTh} \to \text{AntiTh} \to \text{HoloTh} \to \text{Th}$.

### Clock Phase Operator $Z_{\text{clock}}$

$$Z_{\text{clock}} = \text{diag}(1,\; i,\; -1,\; -i)$$

Satisfies the **Weyl commutation relation** with $U_R$:

$$Z_{\text{clock}} \, U_R = i \, U_R \, Z_{\text{clock}}$$

### Discrete Fourier Transform $\mathcal{F}$

$$\mathcal{F} = \frac{1}{2}\begin{pmatrix} 1 & 1 & 1 & 1 \\ 1 & i & -1 & -i \\ 1 & -1 & 1 & -1 \\ 1 & -i & -1 & i \end{pmatrix}$$

The DFT diagonalises the cyclic shift:

$$\mathcal{F} \; U_R \; \mathcal{F}^{\dagger} = Z_{\text{clock}}$$

and is itself of order 4: $\mathcal{F}^4 = I$.

---

## Basis-Change Gates

### $B_{\text{LOG}}$ Gate (`ionqBLOGgate.py`)

Maps the logical ontological basis to the physical Bell-separable encoding:

$$\text{Th} \mapsto |00\rangle, \quad \text{AntiTh} \mapsto |11\rangle, \quad \text{SynTh} \mapsto |\Psi^+\rangle, \quad \text{HoloTh} \mapsto |\Psi^-\rangle$$

Its adjoint $B_{\text{LOG}}^{\dagger}$ performs the inverse mapping. Both are pre-compiled into IonQ Forte native gates (GPI, GPI2, ZZ), each requiring exactly **2 ZZ(¼) entangling gates** — the KAK decomposition minimum for a general two-qubit unitary.

### $B_{\text{VIRT}}$ Gate (`ionqBVIRTgate.py`)

Maps the virtual ontological basis to the physical Bell-separable encoding:

$$F \mapsto |\Psi^-\rangle, \quad P \mapsto |00\rangle, \quad M \mapsto |\Psi^+\rangle, \quad R \mapsto |11\rangle$$

Like $B_{\text{LOG}}$, it is pre-compiled to native Forte operations.

---

## Single-Qudit Gates

### Hadamard Gate (`ionqhadamardgate.py`)

The tetralemmatic Hadamard acts on the **polar subspace** (Th, AntiTh) as the standard Hadamard, while acting as the identity on the **non-polar subspace** (SynTh, HoloTh):

$$H_{\text{onto}} = \begin{pmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} & 0 & 0 \\ \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

**Properties:** Unitary, self-inverse ($H^2 = I$). Maps $\text{Th} \mapsto \frac{1}{\sqrt{2}}(\text{Th} + \text{AntiTh})$ and $\text{AntiTh} \mapsto \frac{1}{\sqrt{2}}(\text{Th} - \text{AntiTh})$, while fixing SynTh and HoloTh.

The physical gate is $H_{\text{phys}} = B \, H_{\text{onto}} \, B^{\dagger}$, a 4×4 unitary acting on two qubits.

### Pauli-X: Duality Involution (`ionqpauligates.py`)

Swaps Th ↔ AntiTh, fixes SynTh and HoloTh:

$$X_{\text{onto}} = \begin{pmatrix} 0 & 1 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

**Properties:** Unitary, Hermitian, involutory ($X^2 = I$).

### Pauli-Z: Polar Discrimination (`ionqpauligates.py`)

Assigns +1 to Th and −1 to AntiTh, annihilating the non-polar states:

$$Z_{\text{onto}} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

**Properties:** Hermitian but **non-unitary** — it is a partial isometry. Implemented as a Kraus channel with operators $K_0 = Z_{\text{onto}}$ and $K_1 = \sqrt{I - Z_{\text{onto}}^2}$.

### Pauli-Y (`ionqpauligates.py`)

Defined as $Y_{\text{onto}} = i \, X_{\text{onto}} \, Z_{\text{onto}}$. Non-unitary; annihilates the Bell states (SynTh, HoloTh).

### Phase Gates (`ionqphasegates.py`)

**S gate** (π/4 phase):

$$S_{\text{onto}} = \begin{pmatrix} e^{-i\pi/4} & 0 & 0 & 0 \\ 0 & e^{i\pi/4} & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

**T gate** (π/8 phase):

$$T_{\text{onto}} = \begin{pmatrix} e^{-i\pi/8} & 0 & 0 & 0 \\ 0 & e^{i\pi/8} & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

Both act as geometric phase rotations on the polar subspace and identity on the non-polar subspace. They satisfy $T^2 = S$ and commute with $Z_{\text{onto}}$.

---

## Two-Qudit Entangling Gates

All two-qudit gates are 16×16 unitaries acting on the tensor product $\mathbb{C}^4 \otimes \mathbb{C}^4$.

### CNOT (`ionqcnotgate.py`)

The controlled-NOT, with control on AntiTh (index 1) and target applying the duality involution $X$:

$$\text{CNOT}_{\text{onto}} = I \otimes I + |1\rangle\langle 1| \otimes (X_{\text{onto}} - I)$$

In block form, the 16×16 matrix is the identity except that the block corresponding to control = AntiTh applies $X_{\text{onto}}$ to the target. Reduces to the standard CNOT on the polar subspace.

**Decomposition:** $B^{\dagger}$ on control and target → controlled-$X_{\text{onto}}$ → $B$ on control and target.

### CZ (`ionqczgate.py`)

The controlled-Z phase gate:

$$\text{CZ}_{\text{onto}} = I_{16} - 2\,|1,1\rangle\langle 1,1|$$

The 16×16 identity matrix with a single −1 at the position where both qudits are in AntiTh. Hermitian and involutory ($\text{CZ}^2 = I$).

**Decomposition:** $\text{CZ} = (I \otimes H_{\text{onto}}) \cdot \text{CNOT} \cdot (I \otimes H_{\text{onto}})$.

### SWAP (`ionqswapgate.py`)

Full d = 4 state exchange between two qudits:

$$\text{SWAP}_{\text{onto}} |j\rangle \otimes |k\rangle = |k\rangle \otimes |j\rangle$$

A 16×16 permutation matrix with $\text{SWAP}[i \cdot 4 + j,\; j \cdot 4 + i] = 1$. Unitary, Hermitian, involutory.

**Decomposition:** Three tetralemmatic CNOTs: $\text{CNOT}(c,t) \cdot \text{CNOT}(t,c) \cdot \text{CNOT}(c,t)$.

### CUR: Controlled $U_R$ (`ionqcurgate.py`)

Applies the quarter-turn shift $U_R$ to the target qudit when the control qudit is in a specific state (AntiTh for logical, HoloTh for virtual):

$$\text{CUR}_{\text{onto}} = \sum_{k \neq c} |k\rangle\langle k| \otimes I + |c\rangle\langle c| \otimes U_R$$

A 16×16 block-diagonal matrix. This gate generates entanglement between the polar and non-polar subspaces — it has no analogue in standard qubit computing.

### SUM: Generalised CNOT (`ionqsumgate.py`)

The full d = 4 cyclic addition gate:

$$\text{SUM}_{\text{onto}} = \sum_{k=0}^{3} |k\rangle\langle k| \otimes U_R^k$$

Acts as $|k\rangle_c |j\rangle_t \mapsto |k\rangle_c |(j + k) \bmod 4\rangle_t$. The 16×16 matrix is block-diagonal with blocks $I, U_R, U_R^2, U_R^3$.

**Note:** CNOT is the special case of SUM restricted to the polar subspace.

**Decomposition:** Uses an optimized 2-bit binary adder with CCZ for carry generation and CNOT for bit-wise addition, wrapped in basis-change operations.

The inverse gate $\text{SUM}^{\dagger}$ applies $U_R^{-k}$ instead.

### Toffoli (`ionqtoffoligate.py`)

Doubly-controlled NOT with both controls on AntiTh:

$$\text{Toffoli}_{\text{onto}} = I^{\otimes 3} + (|1\rangle\langle 1| \otimes |1\rangle\langle 1|) \otimes (X_{\text{onto}} - I)$$

A 64×64 unitary acting on three qudits (six physical qubits). Unitary and involutory.

**Decomposition:** $B^{\dagger}$ on all three qudits → four CCX (Toffoli) chains on the physical qubits → $B$ on all three qudits.

---

## Projector and Algebra Gates (`ionqprojectorgate.py`)

### Sharp / Unsharp Projector

A Kraus channel implementing a measurement-like projection onto a chosen ontological vertex $v$:

$$K_0 = B \, \sqrt{E} \, B^{\dagger}, \qquad K_1 = B \, \sqrt{I - E} \, B^{\dagger}$$

where $E$ is a diagonal matrix with `transmission` at vertex $v$ and zero elsewhere. For a sharp projector, `transmission` = 1; for unsharp, $0 < \text{transmission} < 1$.

The completeness relation $K_0^{\dagger} K_0 + K_1^{\dagger} K_1 = I$ is guaranteed by construction.

### Generic Algebra Gate

Accepts an arbitrary 4×4 matrix. Automatically detects whether the matrix is unitary (provides `_unitary_`) or non-unitary (provides `_kraus_`), enabling both coherent gates and dissipative channels through a single interface.

---

## Virtual Phase Register Gates (`ionqvirtualgates.py`)

These gates operate on the virtual d = 4 register and mirror the fundamental generators:

| Gate | Ontological matrix | Description |
|---|---|---|
| **VURShift** | $U_R$ | Cyclic quarter-turn on virtual register |
| **VZClock** | $\text{diag}(1, i, -1, -i)$ | Phase clock, order 4 |
| **VDFT** | $\mathcal{F}$ | Discrete Fourier transform |
| **VProjector** | Kraus channel | Sharp/unsharp projection on virtual index |
| **VPhaseCompensate**($k$) | $(U_R^{\dagger})^k$ | Phase compensation by $k$ quarter-turns |

The phase compensation gate is used to correct accumulated phase errors before measurement.

---

## Cross-Manifold Gates (`ionqcrossgates.py`)

These gates bridge the logical and virtual manifolds, operating on the joint space $\mathcal{H}_L \otimes \mathcal{H}_V$ via the combined basis change $B_{\text{total}} = B_{\text{LOG}} \otimes B_{\text{VIRT}}$.

### PhaseSwap

Full SWAP between the logical and virtual d = 4 registers:

$$\text{PhaseSwap} \; |j\rangle_L |k\rangle_V = |k\rangle_L |j\rangle_V$$

### Controlled-$U_R$ Phase Gate

The virtual register state $k$ controls $U_R^k$ on the logical register:

$$U_{R,\text{PhaseCtrl}} = \sum_{k=0}^{3} |k\rangle\langle k|_V \otimes U_R^k{}_L$$

A 16×16 block-diagonal matrix with blocks $I, U_R, U_R^2, U_R^3$.

### HoloAmplify

Transfers "warrant" between manifolds by swapping the SynTh–F cross term:

$$|{\text{SynTh}}\rangle_L |F\rangle_V \longleftrightarrow |F\rangle_L |{\text{SynTh}}\rangle_V$$

### PhaseInterference($\theta$)

Applies a parametric phase $e^{i\theta}$ to the Synthesis subspace of the logical register, conditioned on the cross-manifold state.

### HoloPhase($k$)

Applies $U_R^{-k}$ on the virtual register for holographic phase correction.

### ZenoStabilize

A quantum Zeno-effect stabilizer. The reflection operator:

$$M = I - 2\,\Pi_{\text{union}}$$

where $\Pi_{\text{union}}$ projects onto states where **either** register is in HoloTh. Involutory ($M^2 = I$). Repeated application freezes the system in the HoloTh-free subspace.

---

## Parametrized Trotter Gates (`ionqparamgates.py`)

These gates encode the terms of a second-quantised Hamiltonian for quantum chemistry simulation via Suzuki–Trotter decomposition.

### ParamZClockGate($\theta$)

$$\text{ParamZClock}(\theta) = \text{diag}(1,\; e^{i\theta},\; e^{2i\theta},\; e^{3i\theta})$$

Encodes the **on-site energy** term $h_{pp} \, n_p$.

### ParamURShiftGate($\theta$, inverse)

Diagonal phase encoding the **hopping** integral $h_{pq}$:

$$U = \text{diag}(1,\; e^{i\theta},\; e^{2i\theta},\; e^{3i\theta})$$

The inverse variant negates all phases.

### ParamCoulombPhaseGate($\phi$)

A 16×16 identity with a single phase at $|3,3\rangle$:

$$U[15,15] = e^{i\phi}$$

Encodes the **density–density Coulomb** interaction $\frac{1}{2} g_{pp,rr} \, n_p \, n_r$.

### ParamExchangeGate($\phi$)

A beam-splitter on the $|1,2\rangle \leftrightarrow |2,1\rangle$ subspace (indices 6 and 9):

$$U = \begin{pmatrix} \cos\phi & -i\sin\phi \\ -i\sin\phi & \cos\phi \end{pmatrix} \quad \text{on } \{|1,2\rangle, |2,1\rangle\}$$

Encodes the **exchange integral** $\frac{1}{2} g_{pq,qp}$.

### ParamScatteringGate($\phi$, indices)

General four-centre scattering integral $g_{pqrs}$. Decomposed as a sequence of SUM, inverse-SUM, and Coulomb phase gates.

---

## Power-Controlled Gate (`ionqpowercontrolgate.py`)

Given a base gate $U$ and an ancilla qudit of dimension $d$:

$$U_{\text{power}} = \sum_{m=0}^{d-1} |m\rangle\langle m|_{\text{ancilla}} \otimes U^m$$

Realized by stacking $d - 1$ threshold-controlled gates: for each threshold $t \in \{1, \ldots, d-1\}$, apply $U$ conditioned on the ancilla being in state $\geq t$. For d = 4:

$$|0\rangle \to U^0 = I, \quad |1\rangle \to U^1, \quad |2\rangle \to U^2, \quad |3\rangle \to U^3$$

---

## MQE: Modular Quantum Emulator Gates (`ionqmqegates.py`)

The biochemical quantum chemistry extension. These gates model molecular processes on the tetralemmatic qudit register.

### Single-Qudit MQE Gates

| Gate | Matrix | Physical process |
|---|---|---|
| **ElectronShift**($p$) | $U_R^p$ | Electron injection into orbital |
| **ElectronEject**($p$) | $(U_R^{\dagger})^p$ | Electron ejection from orbital |
| **ProtonPhase**($\phi$) | $\text{diag}(1, e^{i\phi}, e^{2i\phi}, e^{3i\phi})$ | Protonation state rotation |
| **PhotonAbsorption**($\phi$) | Phase gate | Photo-driven excitation |
| **PhotonEmission**($\phi$) | Phase gate | Photo-driven relaxation |
| **ConformationalShift**($\delta_h, \Delta t$) | Parametric | Conformational docking |

### Cross-Manifold MQE Gates

**CofactorCoupling**($m, \nu$): Couples logical and virtual registers with modular arithmetic:

$$|k\rangle_L |j\rangle_V \mapsto |k\rangle_L |(j + \nu k) \bmod m\rangle_V$$

Encodes ATP / cofactor stoichiometry. The parameter $\nu$ is the coupling strength (stoichiometric coefficient).

**CofactorDecoupling**($m, \nu$): The inverse operation:

$$|k\rangle_L |j\rangle_V \mapsto |k\rangle_L |(j - \nu k) \bmod m\rangle_V$$

**CrossManifoldSWAP**: Full 16×16 SWAP between logical and virtual d = 4 registers.

**GeneralizedVirtualShift**($m, p$): $U_R^p$ on a d = $m$ virtual register — cyclic permutation of order $m$.

### Composite Gates (m > 4)

For virtual register dimensions $m > 4$, the register is decomposed via mixed-radix encoding $\varphi(k) = (k \bmod 4,\; \lfloor k/4 \rfloor)$ into a primary virtual qudit plus auxiliary qudits. Composite gates dispatch to sequences of singly- and doubly-controlled shifts.

---

## Trotter Evolution (`ionqtrotter.py`)

### First-Order Suzuki–Trotter

Maps molecular integrals to gate sequences for one time step $\Delta t$:

| Integral | Gate |
|---|---|
| $h_{pp}$ (on-site energy) | `ParamZClockGate`$(h_{pp} \, \Delta t)$ |
| $h_{pq}$ (hopping) | `ParamURShiftGate`$(h_{pq} \, \Delta t)$ |
| $\frac{1}{2}g_{pp,rr}$ (Coulomb) | `ParamCoulombPhaseGate`$(\frac{1}{2}g_{pp,rr} \, \Delta t)$ |
| $\frac{1}{2}g_{pq,qp}$ (exchange) | `ParamExchangeGate`$(\frac{1}{2}g_{pq,qp} \, \Delta t)$ |
| $g_{pqrs}$ (scattering) | `ParamScatteringGate`$(g_{pqrs} \, \Delta t)$ |

Error: $\mathcal{O}(\Delta t^2)$.

### Second-Order (Strang) Splitting

Symmetric palindrome $S_1(\Delta t/2) \, S_2(\Delta t) \, S_1(\Delta t/2)$ with leapfrog merging of adjacent half-steps. Error: $\mathcal{O}(\Delta t^3)$.

---

## Biochemical Transition Operator (`ionqjanus.py`)

The Janus operator $J_{n \to n+1}$ implements a single biochemical transition as an 11-layer circuit:

1. **Virtual cofactor shift** — $(U_R^{V,m})^{\nu_n}$
2. **Proton phase rotation** — $Z_{\text{clock}}(\phi_H)$
3. **Electron injection** — $U_R$
4. **Conformational docking** — $S_{\text{dock}}$
5. **Cofactor coupling** — $U_{\text{coupling}}$
6. **Cofactor decoupling** — $U_{\text{coupling}}^{\dagger}$
7. **Electron ejection** — $U_R^{\dagger}$
8. **Deprotonation** — $Z_{\text{clock}}(-\phi_H)$
9. **Inverse virtual shift** — $(U_R^{V,m})^{-\nu_n}$
10. **Photon absorption** — $U_{\text{abs}}(\phi)$
11. **Photon emission** — $U_{\text{em}}(\phi)$

---

## Key Algebraic Relations

1. **Weyl relation:** $Z_{\text{clock}} \, U_R = i \, U_R \, Z_{\text{clock}}$
2. **Fourier diagonalisation:** $\mathcal{F} \, U_R \, \mathcal{F}^{\dagger} = Z_{\text{clock}}$
3. **Cyclic order:** $U_R^4 = I$, $\mathcal{F}^4 = I$, $Z_{\text{clock}}^4 = I$
4. **CNOT ⊂ SUM:** CNOT is SUM restricted to the polar subspace
5. **CZ via Hadamard:** $\text{CZ} = (I \otimes H) \cdot \text{CNOT} \cdot (I \otimes H)$
6. **SWAP via CNOT:** $\text{SWAP} = \text{CNOT}_{12} \cdot \text{CNOT}_{21} \cdot \text{CNOT}_{12}$
7. **Sandwich universality:** Every physical gate is $B^{\dagger} \, U_{\text{onto}} \, B$

---

## Noise Model (`ionqfortenoise.py`)

Calibrated to IonQ Forte 1 error rates:

| Error channel | Rate |
|---|---|
| Single-qubit depolarising | 0.0026 |
| Two-qubit depolarising | 0.0068 |
| Measurement error | 0.0050 |
| Idle error | 0.00005 |

The qudit depolarising channel uses the Weyl operator basis $\{W_i\}$ for d = 4:

$$\mathcal{E}(\rho) = \left(1 - p + \frac{p}{d^2}\right)\rho + \frac{p}{d^2}\sum_{i} W_i \, \rho \, W_i^{\dagger}$$

---

## Compilation Pipeline

The full compilation flow from abstract qudit circuit to IonQ Forte native gates:

```
Abstract qudit circuit
    │
    ▼  (holographic routing)
Gate dispatch → physical wrappers
    │
    ▼  (basis-change sandwich)
2-qubit unitary matrices
    │
    ▼  (KAK / CZ decomposition)
Forte native gates: GPI, GPI2, ZZ(¼)
    │
    ▼  (optional ZNE)
Zero-noise extrapolation folds
```

The `compile_with_holographic_routing` function orchestrates this pipeline, including virtual-register allocation for idle logical qudits and phase compensation before measurement.
