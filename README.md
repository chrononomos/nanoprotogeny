
# nanoprotogeny

**A Polynomial-Complexity Modular Quantum Emulator for Multi-Step Fermionic Catalysis**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21348354.svg)](https://doi.org/10.5281/zenodo.21348354)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

`nanoprotogeny` is the open-source implementation of the **Modular Quantum Emulator (MQE)**, a quantum architecture designed for the chemically accurate, dynamical emulation of multi-step fermionic catalytic mechanisms. Built upon Google Cirq, the IonQ API, and PySCF, this pipeline shifts the computational paradigm from static variational minimization to exact, polynomially-bounded trajectory emulation on near-term trapped-ion hardware.

## Universal Fock Isomorphism and JW Elimination

The architectural cornerstone of MQE is the **Universal Fock Isomorphism** $\iota_p : \mathcal{F}_p \xrightarrow{\,\sim\,} \mathbb{C}^4$, a site-local bijection mapping the four-dimensional Fock space of a single spin-$\frac{1}{2}$ orbital onto a $d=4$ qudit register. The tetralemmatic encoding is defined as:

$$
\begin{aligned}
|0\rangle                    &\mapsto |\mathbf{Th}\rangle     = |00\rangle, \\
|\uparrow\rangle             &\mapsto |\mathbf{AntiTh}\rangle = |11\rangle, \\
|\downarrow\rangle           &\mapsto |\mathbf{SynTh}\rangle  = |\Psi^+\rangle, \\
|\uparrow\downarrow\rangle   &\mapsto |\mathbf{HoloTh}\rangle = |\Psi^-\rangle,
\end{aligned}
$$

where $|\Psi^\pm\rangle=(|01\rangle\pm|10\rangle)/\sqrt{2}$. Because $\iota_p$ is site-local and unitary, the fermionic exchange sign is absorbed into the local antisymmetric Bell state $|\mathbf{HoloTh}\rangle = |\Psi^-\rangle$ at each site. **No non-local parity string is required; the Jordan-Wigner string is eliminated entirely.**

### Hopping Term Elimination

The one-electron hopping term maps to a strictly local two-qudit operator with circuit depth $\mathcal{O}(1)$:

$$
\hat{a}_{p\sigma}^\dagger\hat{a}_{q\sigma}+\mathrm{h.c.}
\;\xmapsto{\iota_p\otimes\iota_q}\;
\hat{U}_{R,\sigma}^{(p)}\otimes\hat{U}_{R,\sigma}^{\dagger(q)},
$$

acting exclusively on $\mathcal{H}_p\otimes\mathcal{H}_q$. No ancillary site $r\notin\{p,q\}$ participates. The quarter-turn phase $\omega_4^k=i^k$ embedded in the tetralemmatic basis tracks the local parity at each site independently, reproducing the sign that JW would assign via a $\hat{Z}$-string over all intermediate sites.

### Coulomb Term Elimination

The Trotter factor for the two-electron repulsion is realized by a constant-depth shift-sandwich circuit:

$$
e^{i\theta_{pq}\hat{n}_p\hat{n}_q}
\;=\;
\hat{S}_{p\to q}\cdot\hat{C}_{p,q}(\theta_{pq})\cdot\hat{S}_{p\to q}^\dagger,
$$

where $\hat{S}_{p\to q}:|k\rangle_p|j\rangle_q\mapsto|k\rangle_p|j{+}k\bmod4\rangle_q$ and $\hat{C}_{p,q}(\theta_{pq})=I_p\otimes\operatorname{diag}(1,e^{i\theta_{pq}}, e^{2i\theta_{pq}},e^{3i\theta_{pq}})_q$. Each of $\hat{S}$, $\hat{C}$, $\hat{S}^\dagger$ is a single two-qudit gate; total depth is $\mathcal{O}(1)$ per pair.

### Resource Reduction

For an active space with $N$ spatial orbitals:

| Resource | Jordan-Wigner | MQE (Tetralemmatic) |
| :--- | :--- | :--- |
| Entangling gate count | $\mathcal{O}(N^4)$ | $\mathcal{O}(N^2)$ |
| Circuit depth per term | $\mathcal{O}(N)$ | $\mathcal{O}(1)$ |
| Total Trotter-step depth | $\mathcal{O}(N^2)$ | $\mathcal{O}(N)$ (all-to-all) |

For the FeMoco $(113e, 76o)$ active space, this reduces $\sim 10^7$ Coulomb parity strings to $\sim 5.8\times10^3$ local modular-addition gates.

## Zero-Overhead Quantum Phase Estimation

The dual-manifold architecture repurposes the virtual register $\mathcal{H}_V^{(m)}$ (required for stoichiometric cofactor phase tracking) as the native QPE clock. The cofactor coupling gate acts as:

$$
\hat{U}_{\mathrm{couple}}^{(m,\nu)}
= \sum_{k=0}^{3}|k\rangle\langle k|_L\otimes\bigl(\hat{U}_R^{V,m}\bigr)^{\nu k},
$$

and conjugation by the virtual discrete Fourier transform $\hat{F}_m^V$ converts it into the QPE controlled-phase gate:

$$
\hat{F}_m^V\,\hat{U}_{\mathrm{couple}}^{(m,\nu)}\,(\hat{F}_m^V)^\dagger
= \sum_{k=0}^{3}|k\rangle\langle k|_L\otimes
\bigl(\hat{Z}_m^{\mathrm{comp}}\bigr)^{\nu k}.
$$

The phase accumulation isomorphism identifies the virtual winding number $k^{(n)}\equiv n\nu\pmod{m}$ with the standard QPE phase:

$$
\frac{k^{(n)}}{m} \equiv \frac{n\,E\tau}{2\pi} \pmod{1},
\qquad
\nu \;\longleftrightarrow\; \frac{mE\tau}{2\pi}.
$$

The two group actions are not analogous—they are identical. Ancilla overhead is identically zero. For composite $m=4r$, precision is $(2+\lceil\log_2 r\rceil)$ bits at zero additional hardware cost.

## Universal Polynomial Complexity

For any fermionic catalytic mechanism $\mathfrak{M}$ with $N$ active orbitals, $M$ discrete steps, $n_\mathrm{cross}$ non-adiabatic crossings, Trotter order $T$, interaction constant $C_\mathrm{int}$, and target accuracy $\epsilon$:

$$
\begin{aligned}
G(\mathfrak{M}) &= \mathcal{O}\left(\frac{MN^3T^2C_\mathrm{int}}{\epsilon}\right), \\
D(\mathfrak{M}) &= \mathcal{O}\left(\frac{MN^2T^2C_\mathrm{int}}{\epsilon} + n_\mathrm{cross}\right), \\
N_\mathrm{shots} &= \mathcal{O}\left(\frac{1}{\epsilon^2}\right).
\end{aligned}
$$

All three bounds are strictly polynomial in $N$, $M$, $T$, $\epsilon^{-1}$, $n_\mathrm{cross}$, and $C_\mathrm{int}$. The virtual register dimension $m$ contributes only a constant multiplicative factor $\mathcal{O}(\log m)$ via Solovay-Kitaev decomposition; it does not alter the asymptotic class.

The quantum advantage over classical FCI grows super-exponentially:

$$
\frac{\text{Classical FCI}}{\text{MQE}}
= \Omega\left(\frac{e^{2N}}{P\cdot MN^3T^2C_{\mathrm{int}}/\epsilon}\right).
$$

For the FeMoco active space ($N=76$, $N_e=113$), this ratio is approximately $5\times10^{27}$.

## Validation

### Mechanism Encoding Survey

| Mechanism | $M$ | $m$ | Br. | $N$ | $S$ | $N_e$ | $\Sigma\nu$ | $n_\mathrm{cr}$ | $n_\gamma$ | $E_0$ (Ha) | $\delta E$ (mHa) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Case I** ($m=1$, virtual register trivial) | | | | | | | | | | | |
| Hydrogenase (H₂ form.) | 2 | 1 | I | 2 | 0 | +2 | 0 | 0 | 0 | −1.0441 | 90.3 |
| Hydrogenase (H₂ ox.) | 2 | 1 | I | 2 | 0 | −2 | 0 | 0 | 0 | −1.1344 | 90.3 |
| ATP Hydrolysis | 5 | 1 | I | 4 | 0 | 0 | 0 | 0 | 0 | −641.2045 | 7.2 |
| Haber–Bosch | 6 | 1 | I | 4 | 0 | 0 | 0 | 0 | 0 | −2633.2110 | 3766.1 |
| Thymine Dimer | 6 | 1 | I | 4 | 0 | 0 | 0 | 1³ | 0 | −156.1219 | 295.4 |
| RNR Radical Relay † | 4 | 1 | I | 4 | ½ | 0 | 0 | 0 | 0 | −698.7776 | 116043 † |
| Ethylene Epoxidation † | 4 | 1 | I | 4 | ½ | 0 | 0 | 0 | 0 | −591.2101 | 92689 † |
| **Case II** ($m \equiv 2 \pmod{4}$, topological Janus only) | | | | | | | | | | | |
| Cytochrome P450 | 6 | 2 | II | 4 | 5/2 | 0 | 2 | 1³ | 0 | −1951.0655 | 576.8 |
| Quinone Q-cycle | 6 | 2 | II | 4 | ½ | 0 | 3 | 0 | 0 | −2.1133 | 428.4 |
| **Case III** ($4 \mid m$, operational Janus SWAP) | | | | | | | | | | | |
| PSII (Kok cycle) | 4 | 4 | III | 4 | 0 | +4 | 4 | 0 | 0 | −3319.5958 | 12.1 |
| Anammox | 4 | 4 | III | 4 | 0 | −4 | 4 | 0 | 0 | −1373.3352 | 1417.1 |
| PSII (photoexcited) | 4 | 4 | III | 4 | ½ | +4 | 4 | 0 | 4 | −2448.4208 | 2131.6 |
| Nitrogenase LT | 8 | 4 | III | 4 | 2 | +8 | 16 | 1⁴ | 0 | −3319.6800 | 11.5 |
| Nitrogenase LT ($m=8$) | 8 | 8 | III | 4 | 2 | +8 | 8 | 1³ | 0 | −3319.6800 | 9.0 |
| Nitrogenase LT (parallel) | 4 | 4 | III | 4 | 2 | +8 | 8 | 1² | 0 | −3319.6800 | 7.9 |
| Nitrogenase [Fe₄S₄] | 8 | 4 | III | 4 | 0 | +8 | 16 | 1⁴ | 0 | −6639.5563 | 58.1 |
| Nitrogenase (closed loop) | 16 | 4 | III | 4 | 2 | 0 | 16 | 2⁴˒¹¹ | 0 | −3319.6800 | 9.0 |

> **Notes:**
> -   † Active-space proxy: inter-step $|\Delta E_\mathrm{corr}| > 10^2$ Ha; MQE stoichiometric invariants verified but $\delta E$ is not a physical energy range.
> -   $\Sigma\nu = \sum_n \nu_n$ counts forward clock advances; for cyclic mechanisms $\Sigma\nu \bmod m = 0$ (phase closure).
> -   Case III crossings at step $n$ (superscript) are operational Janus SWAPs; the nitrogenase closed-loop entry has two Janus intermediates at steps $n=4$ (forward) and $n=11$ (reverse).

---

### MQE-QPE Energy-Extraction Accuracy

| Mechanism | Br. | $N_e^\text{net}$ | $k_\text{net}$ | $S$ | $\eta_V$ | $N_\tau$ | QPE steps | $\|\delta E\|_\text{final}$ (mHa) | $\|\delta E\|_\text{J}$ (mHa) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Case I** (odd $m$ or $m=1$) | | | | | | | | | |
| Hydrogenase (H₂ form.) | I | +2 | 0 | 0 | 0.9976 | 5ᵉ | 1 | 0.611 | — |
| Hydrogenase (H₂ ox.) | I | −2 | 0 | 0 | 0.9976 | 5ᵉ | 1 | 1.263 | — |
| ATP Hydrolysis | I | 0 | 0 | 0 | 0.9900 | 4ᵈ | 4 | 0.384 | — |
| Haber–Bosch | I | 0 | 0 | 0 | 0.9950 | 3 | 5 | 0.420 | — |
| Thymine Dimer | I | 0 | 0 | 0 | 0.9976 | 3 | 3, 5 | 0.564 | — |
| RNR Radical † | I | 0 | 0 | ½ | 0.9988 | 1ᵃ‡ | 3 | 0.178 | — |
| Ethylene Epox. † | I | 0 | 0 | ½ | 0.9975 | 2ᵇ‡ | 3 | 0.350 | — |
| **Case II** ($m \equiv 2 \pmod{4}$) | | | | | | | | | |
| Cytochrome P450 | II | 0 | 0 | 5/2 | 0.9950 | 3 | 3, 5 | 0.681 | — |
| Quinone Q-cycle | II | 0 | 0 | ½ | 0.9966 | 3 | 5 | 0.601 | — |
| **Case III** ($4 \mid m$) | | | | | | | | | |
| PSII (Kok cycle) | III | +4 | 4 | 0 | 0.9950 | 3 | 3 | 0.640 | — |
| Anammox | III | −4 | 4 | 0 | 0.9983 | 2ᵇ | 3 | 0.080 | — |
| PSII (photoexcited) | III | +4 | 4 | ½ | 0.9958 | 3 | 3 | 0.650 | — |
| Nitrogenase LT | III | +8 | 16 | 2 | 0.9952 | 4ᵈ | 4, 7 | 0.595 | 0.640 |
| Nitrogenase LT ($m=8$) | III | +8 | 8 | 2 | 0.9952 | 4ᵈ | 3, 7 | 0.742 | 0.534 |
| Nitrogenase LT (parallel) | III | +8 | 8 | 2 | 0.9952 | 4ᵈ | 2, 3 | 0.611 | 0.636 |
| Nitrogenase [Fe₄S₄] | III | +8 | 16 | 0 | 0.9950 | 3 | 4, 7 | 0.524 | 0.583 |
| Nitrogenase CL | III | 0 | 0 | 2 | 0.9952 | 4ᵈ | 4, 11, 15 | 0.539 | 0.636, 0.493 |
| | | | | | | | | | |
| **All 17 mechanisms, all 4 stoichiometric invariants** | | | | | | | | **pass universally** | |
| **All 25 QPE measurements < 1.6 mHa threshold** | | | | | | | | **chemical accuracy** | |

> **Notes:**
> -   † Proxy dataset; stoichiometric invariants verified but inter-step $|\Delta E_\mathrm{corr}| > 10^2$ Ha renders absolute $\delta E$ non-physical.
> -   $\eta_V = (1 - p_{\mathrm{idle},V})^{n_\mathrm{ctrl}}$: D-state idle decoherence factor recorded at the adaptive (first) QPE checkpoint.
> -   $N_\tau$: number of $\tau$ values in the MLE sequence selected by adaptive preflight.
>     -   ᵃ $N_\tau=1$: $\tau\text{-seq}=\{0.02\}$ Ha⁻¹
>     -   ᵇ $N_\tau=2$: $\tau\text{-seq}=\{0.02, 0.04\}$ Ha⁻¹
>     -   Unmarked ($N_\tau=3$): $\tau\text{-seq}=\{0.02, 0.04, 0.08\}$ Ha⁻¹
>     -   ᵈ $N_\tau=4$: $\tau\text{-seq}=\{0.02, 0.04, 0.08, 0.16\}$ Ha⁻¹
>     -   ᵉ $N_\tau=5$: $\tau\text{-seq}=\{0.02, 0.04, 0.08, 0.16, 0.32\}$ Ha⁻¹
> -   ‡ Richardson ZNE extrapolant selected by best-of-two criterion; all other steps use exponential ZNE.
> -   $\|\delta E\|_\text{J}$: QPE residual at the operational Janus SWAP step; — for Case I, II, and Case III with $n_\mathrm{cr}=0$. For Nitrogenase CL, two values correspond to forward (step 4) and reverse (step 11) Janus intermediates.
> -   **Global MQE-QPE residual range:** 0.080–1.263 mHa; **mean:** 0.548 mHa; **total wall time:** 1763 s (29.4 min).

## Software Stack

`nanoprotogeny` integrates three foundational open-source libraries:

-   **[PySCF](https://pyscf.org/):** Generation of ROHF/FCI active-space integrals ($h_{pq}$, $g_{pqrs}$) and molecular orbital optimization.
-   **[Google Cirq](https://quantumai.google/cirq):** Circuit construction, $d=4$ qudit gate synthesis, and noise-aware compilation.
-   **[IonQ API](https://ionq.com/):** Access to realistic Forte hardware noise models and density matrix simulation backends.

## Installation

```bash
git clone https://github.com/chrononomos/nanoprotogeny.git
cd nanoprotogeny
python -m venv .venv
source .venv/bin/activate  # Linux/Mac; .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

# If you use nanoprotogeny in your research, please cite:

```bibtex
@misc{borom_mqe_2026,
  author       = {Borom, Santos C.},
  title        = {chrononomos/nanoprotogeny: {M}odular {Q}uantum {E}mulator {I}nitial {Z}enodo {R}elease ({V}ersion v0.1.0-mqe-zetazeros-zenodo)},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.21348354},
  url          = {https://doi.org/10.5281/zenodo.21348354},
  note         = {Computer software}
}
```

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21348354.svg)](https://doi.org/10.5281/zenodo.21348354)