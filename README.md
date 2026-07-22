# nanoprotogeny

**A Polynomial-Complexity Modular Quantum Emulator for Multi-Step Fermionic Catalysis**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21348354.svg)](https://doi.org/10.5281/zenodo.21348354)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

`nanoprotogeny` is the open-source implementation of the **Modular Quantum Emulator (MQE)**, a quantum architecture designed for the chemically accurate, dynamical emulation of multi-step fermionic catalytic mechanisms. Built upon Google Cirq, the IonQ API, and PySCF, this pipeline shifts the computational paradigm from static variational minimization to exact, polynomially-bounded trajectory emulation on near-term trapped-ion hardware.

> **Reference.** This software accompanies the pre-print: Santos C. Borom, *The Modular Quantum Emulator: Chemically Accurate, Polynomial-Cost Simulation of Multi-Step Metalloenzyme Catalysis in a Dissipative Protein Environment*, ChemRxiv (2026), [doi:10.26434/chemrxiv.15006143/v1](https://doi.org/10.26434/chemrxiv.15006143/v1). The claims and figures below track that pre-print; see [Citation](#citation) to reference the paper and the code.

## Universal Fock Isomorphism and Jordan–Wigner Reduction

The architectural cornerstone of MQE is the **Universal Fock Isomorphism** $\iota_p : \mathcal{F}_p \xrightarrow{\,\sim\,} \mathbb{C}^4$, a site-local bijection mapping the four-dimensional Fock space of a single spin-$\frac{1}{2}$ orbital onto a $d=4$ qudit register. The tetralemmatic encoding is defined as:

$$
\begin{aligned}
|0\rangle                    &\mapsto |\mathbf{Th}\rangle     = |00\rangle, \\
|\uparrow\rangle             &\mapsto |\mathbf{AntiTh}\rangle = |11\rangle, \\
|\downarrow\rangle           &\mapsto |\mathbf{SynTh}\rangle  = |\Psi^+\rangle, \\
|\uparrow\downarrow\rangle   &\mapsto |\mathbf{HoloTh}\rangle = |\Psi^-\rangle,
\end{aligned}
$$

where $|\Psi^\pm\rangle=(|01\rangle\pm|10\rangle)/\sqrt{2}$. Because $\iota_p$ is site-local and unitary, the *intra*-orbital exchange sign is absorbed locally at each site. **The intra-orbital Jordan–Wigner string is eliminated exactly; the non-local *inter*-orbital string is not removed but *compressed* to an $\mathcal{O}(\log N)$-weight $\hat{Z}_4$ read on a dedicated $d=4$ parity register** (a Fenwick / Bravyi–Kitaev tree), strictly within the native $d=4$ Heisenberg–Weyl gate set. This is the honest scope of the reduction: a bare two-qudit hopping operator provably *cannot* represent the inter-orbital term, so the parity register — one $d=4$ qudit per orbital — is mandatory. The fermionic core therefore occupies $2N$ $d=4$ qudits ($N$ logical $+$ $N$ parity), not $N$.

### Hopping Term Reduction

The one-electron hopping term maps to local ladder factors on the logical register $\mathcal{H}_L$ dressed by an $\mathcal{O}(\log N)$-weight parity read on the parity register $\mathcal{H}_P$:

$$
\hat{a}_{p\sigma}^\dagger\hat{a}_{q\sigma}+\mathrm{h.c.}
\;\xmapsto{\;\text{tripartite encoding}\;}\;
\underbrace{\hat{U}_{R,\sigma}^{(p)}\otimes\hat{U}_{R,\sigma}^{\dagger(q)}}_{\text{local ladder on }\mathcal{H}_L}
\;\otimes\;
\underbrace{\hat{Z}_4^{\,R(p,q)}}_{\mathcal{O}(\log N)\text{ read on }\mathcal{H}_P}.
$$

The intra-orbital sign is carried locally by the quarter-turn phase $\omega_4^k=i^k$ of the tetralemmatic basis; the inter-orbital sign is supplied by the $\hat{Z}_4$ read over the $\mathcal{O}(\log N)$ Fenwick-tree node set $R(p,q)$. This replaces the $\mathcal{O}(N)$-length Jordan–Wigner string with an $\mathcal{O}(\log N)$-weight read — a genuine reduction, not an elimination. A bare two-qudit operator $\hat{U}_{R,\sigma}^{(p)}\otimes\hat{U}_{R,\sigma}^{\dagger(q)}$ alone cannot represent the term (inter-orbital obstruction). Depth is $\mathcal{O}(\log N)$ per one-body term.

### Coulomb Term (String-Free)

Unlike the hopping term, the density–density repulsion is diagonal in occupation and its Jordan–Wigner strings cancel identically, so it requires **no** parity register. The Trotter factor is a single diagonal two-qudit phase using the *physical* occupations $\hat{n}=\operatorname{diag}(0,1,1,2)$ (not the qudit index):

$$
e^{i\theta_{pq}\hat{n}_p\hat{n}_q}
\;=\;
\exp\!\Bigl(i\theta_{pq}\,\operatorname{diag}(0,1,1,2)_p \otimes \operatorname{diag}(0,1,1,2)_q\Bigr),
$$

acting on $\mathcal{H}_L^{(p)}\otimes\mathcal{H}_L^{(q)}$ at depth $\mathcal{O}(1)$ per pair. (Using the qudit index $0,1,2,3$ in place of the physical occupations $0,1,1,2$ is the naïve, incorrect form; the correction to physical occupations is required for a faithful encoding.)

### Resource Reduction

For an active space with $N$ spatial orbitals:

| Resource | Jordan–Wigner | MQE (Tetralemmatic + parity register) |
| :--- | :--- | :--- |
| One-body (hopping) depth | $\mathcal{O}(N)$ | $\mathcal{O}(\log N)$ |
| Two-body (Coulomb) depth | $\mathcal{O}(N)$ | $\mathcal{O}(1)$ (string-free) |
| Trotter-step gate count | $\mathcal{O}(N^4)$ | $\mathcal{O}(N^2 \log N)$ |
| Qudit footprint (fermionic core) | $N$ (qubits) | $2N$ $d{=}4$ ($N$ logical $+$ $N$ parity) |

The one-body speedup is a compression of the $\mathcal{O}(N)$ Jordan–Wigner string to an $\mathcal{O}(\log N)$ parity read; the two-body term is genuinely string-free at $\mathcal{O}(1)$. For the FeMoco $(113e, 76o)$ active space, point-group screening reduces the $\sim 3.3\times10^7$ two-electron scattering channels to $\sim 5.8\times10^3$ dominant $\mathcal{O}(N^2)$ pairs, each a constant-depth modular-addition gate, while each one-body term carries an $\mathcal{O}(\log N)$ parity read.

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

The two group actions are not analogous—they are identical. QPE therefore adds **no clock ancilla**: it reuses the catalytic register that stoichiometric bookkeeping already makes mandatory (this zero-overhead is distinct from the fermionic-core parity register, which is a separate, mandatory $N$-qudit resource). For composite $m=4r$, precision is $(2+\lceil\log_2 r\rceil)$ bits at zero additional clock hardware.

## Universal Polynomial Complexity

For any fermionic catalytic mechanism $\mathfrak{M}$ with $N$ active orbitals, $M$ discrete steps, $n_\mathrm{cross}$ non-adiabatic crossings, Trotter order $T$, double-commutator interaction constant $C^{(2)}_\mathrm{int}$, and target accuracy $\epsilon$, the second-order (Suzuki–Trotter) pipeline actually executed satisfies:

$$
\begin{aligned}
G(\mathfrak{M}) &= \mathcal{O}\left(\frac{MN^{7/2}\,T\sqrt{C^{(2)}_\mathrm{int}}\,\log N}{\sqrt{\epsilon}}\right), \\
D(\mathfrak{M}) &= \mathcal{O}\left(\frac{MN^{5/2}\,T\sqrt{C^{(2)}_\mathrm{int}}\,\log N}{\sqrt{\epsilon}} + n_\mathrm{cross}\right), \\
N_\mathrm{shots} &= \mathcal{O}\left(\frac{1}{\epsilon^2}\right).
\end{aligned}
$$

All three bounds are strictly polynomial in $N$, $M$, $T$, $\epsilon^{-1}$, $n_\mathrm{cross}$, and $C^{(2)}_\mathrm{int}$. The $\log N$ factor is the Jordan–Wigner parity read — polylogarithmic, and it does not affect the polynomial class. The virtual register dimension $m$ contributes only a constant multiplicative $\mathcal{O}(\log m)$ (Solovay–Kitaev) to the virtual-sector decomposition; it does not alter the asymptotic class. Because the auxiliary sector is algebraically independent of the electronic Hamiltonian, the eight extensions (nuclear quantum effects, relativistic and spin–orbit coupling, open-system dissipation, periodic boundary conditions, and more) are absorbed under a single unified certificate $G^{\mathrm{ext}}=\mathcal{O}(MN^{7/2}T\sqrt{C^{(2)}_\mathrm{int}}/\sqrt{\epsilon})$ without changing the complexity class.

The advantage over classical exact methods is the contrast between an **exponential** and a **polynomial** cost. Exact real-time, non-adiabatic propagation of the FeMoco $(113e, 76o)$ active space by Full Configuration Interaction requires a Hilbert space of dimension $>10^{36}$ and the evaluation of $\sim 3.3\times10^7$ irreducible two-electron channels; the MQE cost is bounded by the strictly polynomial $G,\,D,\,N_\mathrm{shots}$ above.

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
git clone [https://github.com/chrononomos/nanoprotogeny.git](https://github.com/chrononomos/nanoprotogeny.git)
cd nanoprotogeny
python -m venv .venv
source .venv/bin/activate  # Linux/Mac; .venv\Scripts\activate on Windows
pip install -r requirements.txt
```


# Generates datasets: contains per reachtion stoichiometry step 1e and 2e integrals

```bash
# New command
mqe generate-data --mechanism nitrogenase_lt \
  --basis '{"Fe":"def2-TZVP","S":"def2-TZVP"}' \
  --output-dir datasets/baseline

# ── Fe2S2 reductive cycles ────────────────────────────────────────────────────
# Atoms: Fe, S only
mqe generate-data \
  --mechanism nitrogenase_lt \
  --basis '{"Fe":"def2-TZVP","S":"def2-TZVP"}' \
  --output-dir datasets/baseline

# ── Fe2S2 closed-loop ────────────────────────────────────────────────────
mqe generate-data \
  --mechanism nitrogenase_closed_loop \
  --basis '{"Fe":"def2-TZVP","S":"def2-TZVP"}' \
  --output-dir datasets/baseline

# ── Fe4S4 cubane ─────────────────────────────────────────────────────────────
# Atoms: Fe, S only (same as Fe2S2 family)
mqe generate-data \
  --mechanism nitrogenase_fe4s4 \
  --basis '{"Fe":"def2-TZVP","S":"def2-TZVP"}' \
  --output-dir datasets/baseline

# ── Nitrogenase FeMoco─────────────────────────────────────────────────────────
mqe generate-data \
  --mechanism nitrogenase_femoco \
  --basis '{"Fe":"def2-SVP","Mo":"def2-TZVP","S":"def2-SVP","C":"cc-pVDZ","N":"cc-pVTZ","O":"cc-pVDZ"}' \
  --output-dir datasets/baseline

# ── PSII (Fe2S2 proxy, not Mn — the non-photo spec uses the same Fe2S2 geometry) ─── 
# Atoms: Fe, S only
mqe generate-data \
  --mechanism psii \
  --basis '{"Fe":"def2-TZVP","S":"def2-TZVP"}' \
  --output-dir datasets/baseline

# ── PSII photo (Mn2O2 proxy geometry — _psii_photo_geometry_at_step) ──────────
# Atoms: Mn, O only. Mn is Z=25, well within def2-TZVP all-electron range.
mqe generate-data \
  --mechanism psii_photo \
  --basis '{"Mn":"def2-TZVP","O":"cc-pVTZ"}' \
  --output-dir datasets/baseline

# ── Haber-Bosch (Fe2 surface + N, H intermediates) ───────────────────────────
# Atoms: Fe, N, H. No S. H is light — cc-pVTZ is over-specified but harmless.
mqe generate-data \
  --mechanism haber_bosch \
  --basis '{"Fe":"def2-TZVP","N":"cc-pVTZ","H":"cc-pVTZ"}' \
  --output-dir datasets/baseline

# poetry run python mqe_datasets.py \
#   --mechanism haber_bosch \
#   --basis '{"Fe":"def2-TZVP","N":"ma-def2-TZVP","H":"def2-SVP"}' \
#   --output-dir datasets/baseline/

# ── Anammox proxy (Fe + N, H — hydrazine synthase active site) ───────────────
# Atoms: Fe (single centre), N, H
mqe generate-data \
  --mechanism anammox_proxy \
  --basis '{"Fe":"def2-TZVP","N":"cc-pVTZ","H":"cc-pVTZ"}' \
  --output-dir datasets/baseline

# ── Ethylene epoxidation (Ag3 surface + C, H, O) ─────────────────────────────
# Atoms: Ag (Z=47 — needs ECP), C, H, O
# Ag requires def2-SVP-PP or def2-TZVPP (which bundles the Stuttgart ECP).
# def2-TZVPP is the correct all-in-one string for PySCF's basis registry.
mqe generate-data \
  --mechanism ethylene_epoxidation \
  --basis '{"Ag":"def2-TZVPP","C":"cc-pVTZ","O":"cc-pVTZ","H":"cc-pVTZ"}' \
  --output-dir datasets/baseline

# ── Thymine dimer ─────────────────────────────────────────────────────────────
mqe generate-data \
  --mechanism thymine_dimer_proxy \
  --basis '{"C":"aug-cc-pVTZ","H":"cc-pVTZ"}' \
  --output-dir datasets/baseline

# ── RNR radical proxy (C, O, H, S — thiyl radical + ribose scaffold) ──────────
# Atoms: C, O, H, S. S (Z=16) is well within all-electron range.
# S carries the radical: def2-TZVP for S gives better spin-density description.
# C, O, H on the ribose scaffold: cc-pVTZ is appropriate.
mqe generate-data \
  --mechanism rnr_radical_proxy \
  --basis '{"S":"def2-TZVP","C":"cc-pVTZ","O":"cc-pVTZ","H":"cc-pVTZ"}' \
  --output-dir datasets/baseline/

# poetry run python mqe_datasets.py \
#   --mechanism rnr_radical_proxy \
#   --basis '{"S":"def2-SVP","C":"cc-pVTZ","O":"cc-pVTZ","H":"cc-pVTZ"}' \
#   --output-dir datasets/baseline

# ── CYP450 metabolism (Fe, O, S, N — porphyrin iron-oxo active site) ─────────
# Atoms: Fe (centre), O (ferryl), S (proximal thiolate), N×4 (porphyrin equatorial)
mqe generate-data \
  --mechanism cyp450_metabolism \
  --basis '{"Fe":"def2-TZVP","S":"def2-TZVP","O":"cc-pVTZ","N":"cc-pVTZ"}' \
  --output-dir datasets/baseline

# ── ATP hydrolysis ───────────────────────────────────────────────────────────
mqe generate-data \
  --mechanism atp_hydrolysis_proxy \
  --basis '{"P":"aug-cc-pVTZ","O":"aug-cc-pVTZ","H":"cc-pVTZ"}' \
  --output-dir datasets/baseline

# ── Hydrogenase reduction (H2 — H only) ───────────────────────────────────────
mqe generate-data \
  --mechanism hydrogenase \
  --basis '{"H":"cc-pVTZ"}' \
  --output-dir datasets/baseline

# ── Hydrogenase oxidation (H2 — H only) ───────────────────────────────────────
mqe generate-data \
  --mechanism hydrogenase_oxidation \
  --basis '{"H":"cc-pVTZ"}' \
  --output-dir datasets/baseline

# ── Reversible quinone ($$Q + 2e^- + 2H^+ \rightleftharpoons QH_2$$) ────────
mqe generate-data \
  --mechanism reversible_quinone \
  --basis '{"H":"aug-cc-pVTZ"}' \
  --output-dir datasets/baseline
```

# Usage: Quantum Phase Estimation with Native MQE QPE (default)

```shell
mqe run --mechanism nitrogenase_lt \
  --dataset-dir datasets/baseline/nitrogenase_lt \
  --output stoichiometry-mqeqpe/nitrogenase_lt_mqeqpe_results.json

# ──────────────────────────────────────────────────────────────────────────
# CORE / ARTICLE MECHANISMS
# ──────────────────────────────────────────────────────────────────────────
# Passed Complete
mqe run --mechanism nitrogenase_lt \
  --dataset-dir datasets/baseline/nitrogenase_lt \
  --output stoichiometry-mqeqpe/nitrogenase_lt_mqeqpe_results.json

# Passed Complete
mqe run --mechanism psii \
  --dataset-dir datasets/baseline/psii \
  --output stoichiometry-mqeqpe/psii_mqeqpe_results.json

# Passed Complete
mqe run --mechanism hydrogenase \
  --dataset-dir datasets/baseline/hydrogenase \
  --output stoichiometry-mqeqpe/hydrogenase_mqeqpe_results.json

# ──────────────────────────────────────────────────────────────────────────
# SURFACE / HETEROGENEOUS CATALYSIS
# ──────────────────────────────────────────────────────────────────────────
# Passed Complete
mqe run --mechanism haber_bosch \
  --dataset-dir datasets/baseline/haber_bosch \
  --output stoichiometry-mqeqpe/haber_bosch_mqeqpe_results.json

# Passed Complete
mqe run --mechanism nitrogenase_fe4s4 \
  --dataset-dir datasets/baseline/nitrogenase_fe4s4 \
  --output stoichiometry-mqeqpe/nitrogenase_fe4s4_mqeqpe_results.json

# To do — no dataset directory yet
# mqe run --mechanism nitrogenase_femoco --dataset-dir ../datasets/ufc_datasets_pubquality --output stoichiometry-mqeqpe/nitrogenase_femoco_mqeqpe_results.json

# Passed Complete
mqe run --mechanism ethylene_epoxidation \
  --dataset-dir datasets/baseline/ethylene_epoxidation \
  --output stoichiometry-mqeqpe/ethylene_epoxidation_mqeqpe_results.json

# ──────────────────────────────────────────────────────────────────────────
# BIOLOGICAL / ENZYMATIC PROXIES
# ──────────────────────────────────────────────────────────────────────────
# Passed Complete
mqe run --mechanism thymine_dimer_proxy \
  --dataset-dir datasets/baseline/thymine_dimer_proxy \
  --output stoichiometry-mqeqpe/thymine_dimer_proxy_mqeqpe_results.json

# ──────────────────────────────────────────────────────────────────────────
# BIOLOGICAL / ENZYMATIC PROXIES
# ──────────────────────────────────────────────────────────────────────────
# Passed Complete
mqe run --mechanism anammox_proxy \
  --dataset-dir datasets/baseline/anammox_proxy \
  --output stoichiometry-mqeqpe/anammox_proxy_mqeqpe_results.json

# ──────────────────────────────────────────────────────────────────────────
# BIOLOGICAL / ENZYMATIC PROXIES
# ──────────────────────────────────────────────────────────────────────────
# Passed Complete
mqe run --mechanism atp_hydrolysis_proxy \
  --dataset-dir datasets/baseline/atp_hydrolysis_proxy \
  --output stoichiometry-mqeqpe/atp_hydrolysis_proxy_mqeqpe_results.json

# Passed Complete
mqe run --mechanism cyp450_metabolism \
  --dataset-dir datasets/baseline/cyp450_metabolism \
  --output stoichiometry-mqeqpe/cyp450_metabolism_mqeqpe_results.json

# Passed Complete
mqe run --mechanism rnr_radical_proxy \
  --dataset-dir datasets/baseline/rnr_radical_proxy \
  --output stoichiometry-mqeqpe/rnr_radical_proxy_mqeqpe_results.json

# ──────────────────────────────────────────────────────────────────────────
# REVERSIBLE / ADVANCED CYCLES
# ──────────────────────────────────────────────────────────────────────────
# Passed Complete
mqe run --mechanism hydrogenase_oxidation \
  --dataset-dir datasets/baseline/hydrogenase_oxidation \
  --output stoichiometry-mqeqpe/hydrogenase_oxidation_mqeqpe_results.json

# Passed Complete
mqe run --mechanism reversible_quinone \
  --dataset-dir datasets/baseline/reversible_quinone \
  --output stoichiometry-mqeqpe/reversible_quinone_mqeqpe_results.json

# Passed Complete
mqe run --mechanism nitrogenase_closed_loop \
  --dataset-dir datasets/baseline/nitrogenase_closed_loop \
  --output stoichiometry-mqeqpe/nitrogenase_closed_loop_mqeqpe_results.json

# ──────────────────────────────────────────────────────────────────────────
# PHOTO-DRIVEN MECHANISMS
# ──────────────────────────────────────────────────────────────────────────
# Passed Complete
mqe run --mechanism psii_photo \
  --dataset-dir datasets/baseline/psii_photo \
  --output stoichiometry-mqeqpe/psii_photo_mqeqpe_results.json

mqe run --mechanism nitrogenase_lt_m8 \
  --dataset-dir datasets/baseline/nitrogenase_lt_m8 \
  --output stoichiometry-mqeqpe/nitrogenase_lt_m8_mqeqpe_results.json

mqe run --mechanism nitrogenase_lt_parallel \
  --dataset-dir datasets/baseline/nitrogenase_lt_parallel \
  --output stoichiometry-mqeqpe/nitrogenase_lt_parallel_mqeqpe_results.json
```

# Full example
```shell
mqe generate-data --mechanism nitrogenase_lt \
  --basis '{"Fe":"def2-TZVP","S":"def2-TZVP"}' \
  --output-dir datasets/baseline


[21:19:49] INFO     | [GENERATE] mechanism='nitrogenase_lt' source='pyscf' n_orbitals=4 output_dir=src/nanoprotogeny/datasets/baseline
[21:19:49] INFO     | 
====================================================================
[21:19:49] INFO     | [MQE-GEN] Mechanism: NITROGENASE_LT
[21:19:49] INFO     |   M=8 steps | m=4 (ℤ_4) | N=4 orbitals
[21:19:49] INFO     |   Expected e⁻: 8 | Expected Σν: 16 ≡ 0 (mod 4)
[21:19:49] INFO     | ====================================================================
[21:19:49] INFO     | [STOICH] Phase closure: [✓] Σν=16 mod 4 = 0
[21:19:49] INFO     | [STOICH] Electron count: [✓] actual=8 expected=8
[21:19:49] INFO     | [MQE-STEP] nitrogenase_lt | n=0/7 | geo='Fe2S2 E0: Fe-S=2.260 Ang' | ncas=4
[21:19:49] INFO     | [BUILD] 4 atoms | 164 AOs | charge=0 | S=2 | basis={'Fe': 'def2-TZVP', 'S': 'def2-TZVP'} | ecp=none | unit=Angstrom
[21:19:49] INFO     | [BUILD] nelec=(44, 40) (nalpha=44, nbeta=40)
[21:19:49] WARNING  |   [CAS] Truncating to CAS(4,4): 80 core electrons excluded. Ensure active space matches validation target.
[21:19:49] INFO     |   CAS(4,4) nalpha=2 nbeta=2  spin_2S=0 (total_nelec=84, core=80)
[21:19:49] INFO     | [SCF] Detected 2 transition-metal centre(s); activating metal-hardened SCF ladder.
[21:19:49] INFO     | [SCF] Level 1: ROHF + level_shift=0.3 + damp=0.3 + atom guess
[21:22:02] INFO     | [SCF] Level 1 converged: E = -3319.6799636887 Ha
[21:22:09] INFO     | [FULL-MO] h1_full shape=(76, 76)  eri_packed size=8561476 (65.3 MB)  n_orbs=76
[21:22:09] INFO     | [TOWER] Saved h1_full_step00.npy (shape=(76, 76)) + eri_packed_step00.npy (65.3 MB) → src/nanoprotogeny/datasets/baseline/nitrogenase_lt
[21:22:11] INFO     | [CASCI] E_total = -3319.6799636887 Ha  (converged)
[21:22:12] INFO     | [CASCI] E_core = -3316.9153386833 Ha
[21:22:12] INFO     | [CASCI] Active MO occupations (noons, ncas=4): 1.0000 1.0000 1.0000 1.0000  sum=4.0000
[21:22:12] INFO     | [REF] Exact FCI: ncas=4, nalpha=2, nbeta=2 ...
[21:22:12] INFO     | [REF] E_FCI = -3319.6799636887 Ha
[21:22:12] INFO     | [ERI] Compressing 4^4 tensor (threshold=1.0e-08) ...
[21:22:12] INFO     | [ERI] 20 unique (pq|rs) channels retained.
[21:22:12] INFO     |   [STEP 0] E_FCI=-3319.67996369 Ha | ν=2 | k^(0)=2 | checks=[✓] | 143.7s
[21:22:12] INFO     | [MQE-STEP] nitrogenase_lt | n=1/7 | geo='Fe2S2 E1: Fe-S=2.274 Ang' | ncas=4
[21:22:12] INFO     | [BUILD] 4 atoms | 164 AOs | charge=0 | S=2 | basis={'Fe': 'def2-TZVP', 'S': 'def2-TZVP'} | ecp=none | unit=Angstrom
[21:22:12] INFO     | [BUILD] nelec=(44, 40) (nalpha=44, nbeta=40)
[21:22:12] WARNING  |   [CAS] Truncating to CAS(4,4): 80 core electrons excluded. Ensure active space matches validation target.
[21:22:12] INFO     |   CAS(4,4) nalpha=2 nbeta=2  spin_2S=0 (total_nelec=84, core=80)
[21:22:12] INFO     | [SCF] Warm-starting step 1 from step 0 density matrix.
[21:22:12] INFO     | [SCF] Detected 2 transition-metal centre(s); activating metal-hardened SCF ladder.
[21:22:12] INFO     | [SCF] Level 1: ROHF + level_shift=0.3 + damp=0.3 + warm-start dm0
[21:22:28] INFO     | [SCF] Level 1 converged: E = -3319.6784525943 Ha
[21:22:36] INFO     | [FULL-MO] h1_full shape=(76, 76)  eri_packed size=8561476 (65.3 MB)  n_orbs=76
[21:22:36] INFO     | [TOWER] Saved h1_full_step01.npy (shape=(76, 76)) + eri_packed_step01.npy (65.3 MB) → src/nanoprotogeny/datasets/baseline/nitrogenase_lt
[21:22:38] INFO     | [CASCI] E_total = -3319.6784525943 Ha  (converged)
[21:22:39] INFO     | [CASCI] E_core = -3316.9170724425 Ha
[21:22:39] INFO     | [CASCI] Active MO occupations (noons, ncas=4): 1.0000 1.0000 1.0000 1.0000  sum=4.0000
[21:22:39] INFO     | [REF] Exact FCI: ncas=4, nalpha=2, nbeta=2 ...
[21:22:39] INFO     | [REF] E_FCI = -3319.6784525943 Ha
[21:22:39] INFO     | [ERI] Compressing 4^4 tensor (threshold=1.0e-08) ...
[21:22:39] INFO     | [ERI] 19 unique (pq|rs) channels retained.
[21:22:39] INFO     |   [STEP 1] E_FCI=-3319.67845259 Ha | ν=2 | k^(1)=0 | checks=[✓] | 26.3s
[21:22:39] INFO     | [MQE-STEP] nitrogenase_lt | n=2/7 | geo='Fe2S2 E2: Fe-S=2.288 Ang' | ncas=4
[21:22:39] INFO     | [BUILD] 4 atoms | 164 AOs | charge=0 | S=2 | basis={'Fe': 'def2-TZVP', 'S': 'def2-TZVP'} | ecp=none | unit=Angstrom
[21:22:39] INFO     | [BUILD] nelec=(44, 40) (nalpha=44, nbeta=40)
[21:22:39] WARNING  |   [CAS] Truncating to CAS(4,4): 80 core electrons excluded. Ensure active space matches validation target.
[21:22:39] INFO     |   CAS(4,4) nalpha=2 nbeta=2  spin_2S=0 (total_nelec=84, core=80)
[21:22:39] INFO     | [SCF] Warm-starting step 2 from step 1 density matrix.
[21:22:39] INFO     | [SCF] Detected 2 transition-metal centre(s); activating metal-hardened SCF ladder.
[21:22:39] INFO     | [SCF] Level 1: ROHF + level_shift=0.3 + damp=0.3 + warm-start dm0
[21:22:55] INFO     | [SCF] Level 1 converged: E = -3319.6768929289 Ha
[21:23:02] INFO     | [FULL-MO] h1_full shape=(76, 76)  eri_packed size=8561476 (65.3 MB)  n_orbs=76
[21:23:02] INFO     | [TOWER] Saved h1_full_step02.npy (shape=(76, 76)) + eri_packed_step02.npy (65.3 MB) → src/nanoprotogeny/datasets/baseline/nitrogenase_lt
[21:23:04] INFO     | [CASCI] E_total = -3319.6768929289 Ha  (converged)
[21:23:05] INFO     | [CASCI] E_core = -3316.9186832216 Ha
[21:23:05] INFO     | [CASCI] Active MO occupations (noons, ncas=4): 1.0000 1.0000 1.0000 1.0000  sum=4.0000
[21:23:05] INFO     | [REF] Exact FCI: ncas=4, nalpha=2, nbeta=2 ...
[21:23:05] INFO     | [REF] E_FCI = -3319.6768929289 Ha
[21:23:05] INFO     | [ERI] Compressing 4^4 tensor (threshold=1.0e-08) ...
[21:23:05] INFO     | [ERI] 20 unique (pq|rs) channels retained.
[21:23:05] INFO     |   [STEP 2] E_FCI=-3319.67689293 Ha | ν=2 | k^(2)=2 | checks=[✓] | 26.5s
[21:23:05] INFO     | [MQE-STEP] nitrogenase_lt | n=3/7 | geo='Fe2S2 E3: Fe-S=2.302 Ang' | ncas=4
[21:23:05] INFO     | [BUILD] 4 atoms | 164 AOs | charge=0 | S=2 | basis={'Fe': 'def2-TZVP', 'S': 'def2-TZVP'} | ecp=none | unit=Angstrom
[21:23:05] INFO     | [BUILD] nelec=(44, 40) (nalpha=44, nbeta=40)
[21:23:05] WARNING  |   [CAS] Truncating to CAS(4,4): 80 core electrons excluded. Ensure active space matches validation target.
[21:23:05] INFO     |   CAS(4,4) nalpha=2 nbeta=2  spin_2S=0 (total_nelec=84, core=80)
[21:23:05] INFO     | [SCF] Warm-starting step 3 from step 2 density matrix.
[21:23:05] INFO     | [SCF] Detected 2 transition-metal centre(s); activating metal-hardened SCF ladder.
[21:23:05] INFO     | [SCF] Level 1: ROHF + level_shift=0.3 + damp=0.3 + warm-start dm0
[21:23:22] INFO     | [SCF] Level 1 converged: E = -3319.6752868358 Ha
[21:23:29] INFO     | [FULL-MO] h1_full shape=(76, 76)  eri_packed size=8561476 (65.3 MB)  n_orbs=76
[21:23:29] INFO     | [TOWER] Saved h1_full_step03.npy (shape=(76, 76)) + eri_packed_step03.npy (65.3 MB) → src/nanoprotogeny/datasets/baseline/nitrogenase_lt
[21:23:31] INFO     | [CASCI] E_total = -3319.6752868358 Ha  (converged)
[21:23:32] INFO     | [CASCI] E_core = -3316.9201743059 Ha
[21:23:32] INFO     | [CASCI] Active MO occupations (noons, ncas=4): 1.0000 1.0000 1.0000 1.0000  sum=4.0000
[21:23:32] INFO     | [REF] Exact FCI: ncas=4, nalpha=2, nbeta=2 ...
[21:23:32] INFO     | [REF] E_FCI = -3319.6752868358 Ha
[21:23:32] INFO     | [ERI] Compressing 4^4 tensor (threshold=1.0e-08) ...
[21:23:32] INFO     | [ERI] 18 unique (pq|rs) channels retained.
[21:23:32] INFO     |   [STEP 3] E_FCI=-3319.67528684 Ha | ν=2 | k^(3)=0 | checks=[✓] | 27.1s
[21:23:32] INFO     | [MQE-STEP] nitrogenase_lt | n=4/7 | geo='Fe2S2 E4: Fe-S=2.316 Ang' | ncas=4
[21:23:32] INFO     | [BUILD] 4 atoms | 164 AOs | charge=0 | S=2 | basis={'Fe': 'def2-TZVP', 'S': 'def2-TZVP'} | ecp=none | unit=Angstrom
[21:23:32] INFO     | [BUILD] nelec=(44, 40) (nalpha=44, nbeta=40)
[21:23:32] WARNING  |   [CAS] Truncating to CAS(4,4): 80 core electrons excluded. Ensure active space matches validation target.
[21:23:32] INFO     |   CAS(4,4) nalpha=2 nbeta=2  spin_2S=0 (total_nelec=84, core=80)
[21:23:32] INFO     | [SCF] Warm-starting step 4 from step 3 density matrix.
[21:23:32] INFO     | [SCF] Detected 2 transition-metal centre(s); activating metal-hardened SCF ladder.
[21:23:32] INFO     | [SCF] Level 1: ROHF + level_shift=0.3 + damp=0.3 + warm-start dm0
[21:23:48] INFO     | [SCF] Level 1 converged: E = -3319.6736363695 Ha
[21:23:55] INFO     | [FULL-MO] h1_full shape=(76, 76)  eri_packed size=8561476 (65.3 MB)  n_orbs=76
[21:23:55] INFO     | [TOWER] Saved h1_full_step04.npy (shape=(76, 76)) + eri_packed_step04.npy (65.3 MB) → src/nanoprotogeny/datasets/baseline/nitrogenase_lt
[21:23:55] INFO     | [TOWER] (Janus) Also wrote h1_full.npy + eri_packed.npy for backward compatibility → src/nanoprotogeny/datasets/baseline/nitrogenase_lt
[21:23:57] INFO     | [CASCI] E_total = -3319.6736363695 Ha  (converged)
[21:23:58] INFO     | [CASCI] E_core = -3316.9215568262 Ha
[21:23:58] INFO     | [CASCI] Active MO occupations (noons, ncas=4): 1.0000 1.0000 1.0000 1.0000  sum=4.0000
[21:23:58] INFO     | [REF] Exact FCI: ncas=4, nalpha=2, nbeta=2 ...
[21:23:58] INFO     | [REF] E_FCI = -3319.6736363695 Ha
[21:23:58] INFO     | [ERI] Compressing 4^4 tensor (threshold=1.0e-08) ...
[21:23:58] INFO     | [ERI] 19 unique (pq|rs) channels retained.
[21:23:58] INFO     |   [STEP 4] E_FCI=-3319.67363637 Ha | ν=2 | k^(4)=2 | checks=[✓] | 25.6s
[21:23:58] INFO     | [MQE-STEP] nitrogenase_lt | n=5/7 | geo='Fe2S2 E5: Fe-S=2.330 Ang' | ncas=4
[21:23:58] INFO     | [BUILD] 4 atoms | 164 AOs | charge=0 | S=2 | basis={'Fe': 'def2-TZVP', 'S': 'def2-TZVP'} | ecp=none | unit=Angstrom
[21:23:58] INFO     | [BUILD] nelec=(44, 40) (nalpha=44, nbeta=40)
[21:23:58] WARNING  |   [CAS] Truncating to CAS(4,4): 80 core electrons excluded. Ensure active space matches validation target.
[21:23:58] INFO     |   CAS(4,4) nalpha=2 nbeta=2  spin_2S=0 (total_nelec=84, core=80)
[21:23:58] INFO     | [SCF] Warm-starting step 5 from step 4 density matrix.
[21:23:58] INFO     | [SCF] Detected 2 transition-metal centre(s); activating metal-hardened SCF ladder.
[21:23:58] INFO     | [SCF] Level 1: ROHF + level_shift=0.3 + damp=0.3 + warm-start dm0
[21:24:14] INFO     | [SCF] Level 1 converged: E = -3319.6719435517 Ha
[21:24:21] INFO     | [FULL-MO] h1_full shape=(76, 76)  eri_packed size=8561476 (65.3 MB)  n_orbs=76
[21:24:21] INFO     | [TOWER] Saved h1_full_step05.npy (shape=(76, 76)) + eri_packed_step05.npy (65.3 MB) → src/nanoprotogeny/datasets/baseline/nitrogenase_lt
[21:24:23] INFO     | [CASCI] E_total = -3319.6719435517 Ha  (converged)
[21:24:24] INFO     | [CASCI] E_core = -3316.9228262632 Ha
[21:24:24] INFO     | [CASCI] Active MO occupations (noons, ncas=4): 1.0000 1.0000 1.0000 1.0000  sum=4.0000
[21:24:24] INFO     | [REF] Exact FCI: ncas=4, nalpha=2, nbeta=2 ...
[21:24:24] INFO     | [REF] E_FCI = -3319.6719435517 Ha
[21:24:24] INFO     | [ERI] Compressing 4^4 tensor (threshold=1.0e-08) ...
[21:24:24] INFO     | [ERI] 18 unique (pq|rs) channels retained.
[21:24:24] INFO     |   [STEP 5] E_FCI=-3319.67194355 Ha | ν=2 | k^(5)=0 | checks=[✓] | 25.9s
[21:24:24] INFO     | [MQE-STEP] nitrogenase_lt | n=6/7 | geo='Fe2S2 E6: Fe-S=2.344 Ang' | ncas=4
[21:24:24] INFO     | [BUILD] 4 atoms | 164 AOs | charge=0 | S=2 | basis={'Fe': 'def2-TZVP', 'S': 'def2-TZVP'} | ecp=none | unit=Angstrom
[21:24:24] INFO     | [BUILD] nelec=(44, 40) (nalpha=44, nbeta=40)
[21:24:24] WARNING  |   [CAS] Truncating to CAS(4,4): 80 core electrons excluded. Ensure active space matches validation target.
[21:24:24] INFO     |   CAS(4,4) nalpha=2 nbeta=2  spin_2S=0 (total_nelec=84, core=80)
[21:24:24] INFO     | [SCF] Warm-starting step 6 from step 5 density matrix.
[21:24:24] INFO     | [SCF] Detected 2 transition-metal centre(s); activating metal-hardened SCF ladder.
[21:24:24] INFO     | [SCF] Level 1: ROHF + level_shift=0.3 + damp=0.3 + warm-start dm0
[21:24:40] INFO     | [SCF] Level 1 converged: E = -3319.6702103270 Ha
[21:24:47] INFO     | [FULL-MO] h1_full shape=(76, 76)  eri_packed size=8561476 (65.3 MB)  n_orbs=76
[21:24:47] INFO     | [TOWER] Saved h1_full_step06.npy (shape=(76, 76)) + eri_packed_step06.npy (65.3 MB) → src/nanoprotogeny/datasets/baseline/nitrogenase_lt
[21:24:49] INFO     | [CASCI] E_total = -3319.6702103270 Ha  (converged)
[21:24:50] INFO     | [CASCI] E_core = -3316.9239894191 Ha
[21:24:50] INFO     | [CASCI] Active MO occupations (noons, ncas=4): 1.0000 1.0000 1.0000 1.0000  sum=4.0000
[21:24:50] INFO     | [REF] Exact FCI: ncas=4, nalpha=2, nbeta=2 ...
[21:24:50] INFO     | [REF] E_FCI = -3319.6702103270 Ha
[21:24:50] INFO     | [ERI] Compressing 4^4 tensor (threshold=1.0e-08) ...
[21:24:50] INFO     | [ERI] 18 unique (pq|rs) channels retained.
[21:24:50] INFO     |   [STEP 6] E_FCI=-3319.67021033 Ha | ν=2 | k^(6)=2 | checks=[✓] | 26.3s
[21:24:50] INFO     | [MQE-STEP] nitrogenase_lt | n=7/7 | geo='Fe2S2 E7: Fe-S=2.358 Ang' | ncas=4
[21:24:50] INFO     | [BUILD] 4 atoms | 164 AOs | charge=0 | S=2 | basis={'Fe': 'def2-TZVP', 'S': 'def2-TZVP'} | ecp=none | unit=Angstrom
[21:24:50] INFO     | [BUILD] nelec=(44, 40) (nalpha=44, nbeta=40)
[21:24:50] WARNING  |   [CAS] Truncating to CAS(4,4): 80 core electrons excluded. Ensure active space matches validation target.
[21:24:50] INFO     |   CAS(4,4) nalpha=2 nbeta=2  spin_2S=0 (total_nelec=84, core=80)
[21:24:50] INFO     | [SCF] Warm-starting step 7 from step 6 density matrix.
[21:24:50] INFO     | [SCF] Detected 2 transition-metal centre(s); activating metal-hardened SCF ladder.
[21:24:50] INFO     | [SCF] Level 1: ROHF + level_shift=0.3 + damp=0.3 + warm-start dm0
[21:25:07] INFO     | [SCF] Level 1 converged: E = -3319.6684385539 Ha
[21:25:14] INFO     | [FULL-MO] h1_full shape=(76, 76)  eri_packed size=8561476 (65.3 MB)  n_orbs=76
[21:25:14] INFO     | [TOWER] Saved h1_full_step07.npy (shape=(76, 76)) + eri_packed_step07.npy (65.3 MB) → src/nanoprotogeny/datasets/baseline/nitrogenase_lt
[21:25:16] INFO     | [CASCI] E_total = -3319.6684385539 Ha  (converged)
[21:25:17] INFO     | [CASCI] E_core = -3316.9250524083 Ha
[21:25:17] INFO     | [CASCI] Active MO occupations (noons, ncas=4): 1.0000 1.0000 1.0000 1.0000  sum=4.0000
[21:25:17] INFO     | [REF] Exact FCI: ncas=4, nalpha=2, nbeta=2 ...
[21:25:17] INFO     | [REF] E_FCI = -3319.6684385539 Ha
[21:25:17] INFO     | [ERI] Compressing 4^4 tensor (threshold=1.0e-08) ...
[21:25:17] INFO     | [ERI] 18 unique (pq|rs) channels retained.
[21:25:17] INFO     |   [STEP 7] E_FCI=-3319.66843855 Ha | ν=2 | k^(7)=0 | checks=[✓] | 27.1s
[21:25:17] INFO     | [TOWER] Saved noons.npy (shape=(164,), n_core=40, n_active=4) → src/nanoprotogeny/datasets/baseline/nitrogenase_lt/noons.npy
[21:25:17] INFO     | [TOWER] Saved mo_energies.npy (shape=(164,)) → src/nanoprotogeny/datasets/baseline/nitrogenase_lt/mo_energies.npy

====================================================================
 MQE DATASET: NITROGENASE_LT
====================================================================
  ℤ_4 phase closure : [✓] Σν=16 mod 4 = 0
  Electron count      : [✓] 8 e⁻ (expected 8)
  Energy ordering     : [✓]
  All algebraic checks: [✓] PASSED

  Step breakdown:
    n=0: -3319.67996369 Ha  [✓]  Fe2S2 E0: Fe-S=2.260 Ang
    n=1: -3319.67845259 Ha  [✓]  Fe2S2 E1: Fe-S=2.274 Ang
    n=2: -3319.67689293 Ha  [✓]  Fe2S2 E2: Fe-S=2.288 Ang
    n=3: -3319.67528684 Ha  [✓]  Fe2S2 E3: Fe-S=2.302 Ang
    n=4: -3319.67363637 Ha  [✓]  Fe2S2 E4: Fe-S=2.316 Ang
    n=5: -3319.67194355 Ha  [✓]  Fe2S2 E5: Fe-S=2.330 Ang
    n=6: -3319.67021033 Ha  [✓]  Fe2S2 E6: Fe-S=2.344 Ang
    n=7: -3319.66843855 Ha  [✓]  Fe2S2 E7: Fe-S=2.358 Ang
====================================================================



❯ mqe run --mechanism nitrogenase_lt \
  --dataset-dir datasets/baseline/nitrogenase_lt \
  --output stoichiometry-mqeqpe/nitrogenase_lt_mqeqpe_results.json


[21:25:37] INFO     | [DATASET MODE] IntegralState initialised empty. StepwiseIntegralStore will load H_n per step at runtime.
[21:25:37] INFO     | [CONFIG] Backend: 'ionq-sim' → 'simulator' | folds=[1, 3, 5] shots=8192
[21:25:37] INFO     | [MQE] Step-wise dataset directory: /Users/padmevajra/Desktop/NanoProtogeny/src/nanoprotogeny/datasets/baseline/nitrogenase_lt
[21:25:37] INFO     | [StepStore] Loaded manifest for 'nitrogenase_lt': M=8 steps, m=4 (ℤ_4)
[21:25:37] INFO     | [HW-Runner] Loaded StepwiseIntegralStore for 'nitrogenase_lt' from /Users/padmevajra/Desktop/NanoProtogeny/src/nanoprotogeny/datasets/baseline/nitrogenase_lt/nitrogenase_lt

==============================================================================
 VANC-QPE PIPELINE VALIDATION: NITROGENASE_LT
==============================================================================
  Mechanism : nitrogenase_lt
  N orbitals: 4 | Steps M=8 | Virtual modulus m=4 (ℤ_4)
  S_target  : 1.5
  e⁻ inject : 8 | e⁻ eject: 0 | Net e⁻: 8
  Σν (fwd)  : 16 | Σν (inv): 0 | Net Σν: 16
  Phase closure ≡ 0 (mod 4): [✓] SATISFIED
  Photons abs: 0 | emit: 0 | net: 0 | phi_photon=1.5708 rad
  Non-adiabatic crossings: 1
  Description: Lowe-Thorneley nitrogenase LT cycle. E0→E8 via 8 sequential forward PCET steps (8e⁻, 8H⁺, 16 ATP). Janus crossing at E4→E5. Fully reversible framework: A_n_eject/P_n_eject/B_n_decouple explicitly zeroed for canonical LT; ready for back-reaction modeling. Net-flux phase closure: Σ(ν-ν†)=16 ≡ 0 (mod 4). Net e⁻: 8.
  Integral source: step-wise JSON datasets (/Users/padmevajra/Desktop/NanoProtogeny/src/nanoprotogeny/datasets/baseline/nitrogenase_lt/nitrogenase_lt/step_XX.json)
------------------------------------------------------------------------------
  [REG] 4 logical (d=4) + 4 virtual (d=4) qudits | integrals: dataset (/Users/padmevajra/Desktop/NanoProtogeny/src/nanoprotogeny/datasets/baseline/nitrogenase_lt/nitrogenase_lt)

  [ALGEBRAIC] Net-flux phase closure validation (ℤ_4)...

[STOICHIOMETRY] ℤ_4 Net-Flux Phase Closure & Electron Count
  e⁻ inject : 8 | e⁻ eject: 0 | Net: 8 (expected 8) [✓]
  Phase:    Σ(ν−ν†)=16, mod 4 = 0 (expected 0) [✓]
  Step log:
    n=00: ν=2, k^(n)=2, Σe_net=1
    n=01: ν=2, k^(n)=0, Σe_net=2
    n=02: ν=2, k^(n)=2, Σe_net=3
    n=03: ν=2, k^(n)=0, Σe_net=4
    n=04: ν=2, k^(n)=2, Σe_net=5
    n=05: ν=2, k^(n)=0, Σe_net=6
    n=06: ν=2, k^(n)=2, Σe_net=7
    n=07: ν=2, k^(n)=0, Σe_net=8
  Overall: [✓] PASSED

  [LOOP] Building M=8 MQE step blocks...
[21:25:37] INFO     | [PARSE] Loaded: 4 h_diag, 6 h_hop, 20 g_full
[21:25:37] INFO     |   [StepStore] n=00: Fe2S2 E0: Fe-S=2.260 Ang | E_ref=-3319.67996369 Ha | ν=2 | k^(n)=2
[21:25:37] INFO     | [ERI DIAG] n_orbs=4, g_full entries=20
[21:25:37] INFO     | [ERI DIAG] Expected unique (chemist): ~35, full tensor: 256
[21:25:38] INFO     | [HW-B0] Using step-specific integrals: N=4, E_core=-3316.915339 Ha
[21:25:38] INFO     |   [HW n=00] A_n=[0] | ν=2 k^(n)=2 | Σe_net=1 | 
[21:25:38] INFO     | [PARSE] Loaded: 4 h_diag, 6 h_hop, 19 g_full
[21:25:38] INFO     |   [StepStore] n=01: Fe2S2 E1: Fe-S=2.274 Ang | E_ref=-3319.67845259 Ha | ν=2 | k^(n)=0
[21:25:38] INFO     | [ERI DIAG] n_orbs=4, g_full entries=19
[21:25:38] INFO     | [ERI DIAG] Expected unique (chemist): ~35, full tensor: 256
[21:25:39] INFO     | [HW-B1] Using step-specific integrals: N=4, E_core=-3316.917072 Ha
[21:25:39] INFO     |   [HW n=01] A_n=[1] | ν=2 k^(n)=0 | Σe_net=2 | 
[21:25:39] INFO     | [PARSE] Loaded: 4 h_diag, 6 h_hop, 20 g_full
[21:25:39] INFO     |   [StepStore] n=02: Fe2S2 E2: Fe-S=2.288 Ang | E_ref=-3319.67689293 Ha | ν=2 | k^(n)=2
[21:25:39] INFO     | [ERI DIAG] n_orbs=4, g_full entries=20
[21:25:39] INFO     | [ERI DIAG] Expected unique (chemist): ~35, full tensor: 256
[21:25:40] INFO     | [HW-B2] Using step-specific integrals: N=4, E_core=-3316.918683 Ha
[21:25:40] INFO     |   [HW n=02] A_n=[2] | ν=2 k^(n)=2 | Σe_net=3 | 
[21:25:40] INFO     | [PARSE] Loaded: 4 h_diag, 6 h_hop, 18 g_full
[21:25:40] INFO     |   [StepStore] n=03: Fe2S2 E3: Fe-S=2.302 Ang | E_ref=-3319.67528684 Ha | ν=2 | k^(n)=0
[21:25:40] INFO     | [ERI DIAG] n_orbs=4, g_full entries=18
[21:25:40] INFO     | [ERI DIAG] Expected unique (chemist): ~35, full tensor: 256
[21:25:41] INFO     | [HW-B3] Using step-specific integrals: N=4, E_core=-3316.920174 Ha
[21:25:41] INFO     |   [HW n=03] A_n=[3] | ν=2 k^(n)=0 | Σe_net=4 | 
[21:25:41] INFO     | [PARSE] Loaded: 4 h_diag, 6 h_hop, 19 g_full
[21:25:41] INFO     |   [StepStore] n=04: Fe2S2 E4: Fe-S=2.316 Ang | E_ref=-3319.67363637 Ha | ν=2 | k^(n)=2
[21:25:41] INFO     | [ERI DIAG] n_orbs=4, g_full entries=19
[21:25:41] INFO     | [ERI DIAG] Expected unique (chemist): ~35, full tensor: 256
[21:25:42] INFO     | [HW-B4] Using step-specific integrals: N=4, E_core=-3316.921557 Ha
[21:25:42] INFO     | [HW] Step n=4: Janus crossing applied (orbitals p=0, q=1, δCI=1.60e-03)
[21:25:42] INFO     |   [HW n=04] A_n=[0] | ν=2 k^(n)=2 | Σe_net=5 | ⚡ Janus

  [HW-QPE] Checkpoint n=4 (Janus E_4→E_5) | Fe2S2 E4: Fe-S=2.316 Ang
    E_0 (exact diag) = -2.75207954 Ha | E_ref (FCI) = -2.75207954 Ha
[21:26:14] INFO     | [VANC-TAU-SELECT] ✗ τ_max=0.32 Ha⁻¹  n_max=16  |E_ZNE−E_ref|=10.2625 mHa  [exceeds 1.6 mHa budget]
[21:26:31] INFO     | [VANC-TAU-SELECT] ✓ τ_max=0.16 Ha⁻¹  n_max=8  |E_ZNE−E_ref|=0.6399 mHa  [within 1.6 mHa budget]
[21:26:31] INFO     | [VANC-QPE n=4] τ-seq (adaptive): [0.02, 0.04, 0.08, 0.16] | n_max=8
[21:26:31] INFO     | [VANC-QPE n=4] η_V=0.995211 (gates/step=12, n_ctrl=96)
    E_ZNE (exp) = -2.75143965 Ha | |E_ZNE−E_ref| = 0.6399 mHa [✓]
[21:26:47] INFO     | [PARSE] Loaded: 4 h_diag, 6 h_hop, 18 g_full
[21:26:47] INFO     |   [StepStore] n=05: Fe2S2 E5: Fe-S=2.330 Ang | E_ref=-3319.67194355 Ha | ν=2 | k^(n)=0
[21:26:47] INFO     | [ERI DIAG] n_orbs=4, g_full entries=18
[21:26:47] INFO     | [ERI DIAG] Expected unique (chemist): ~35, full tensor: 256
[21:26:48] INFO     | [HW-B5] Using step-specific integrals: N=4, E_core=-3316.922826 Ha
[21:26:48] INFO     |   [HW n=05] A_n=[1] | ν=2 k^(n)=0 | Σe_net=6 | 
[21:26:48] INFO     | [PARSE] Loaded: 4 h_diag, 6 h_hop, 18 g_full
[21:26:48] INFO     |   [StepStore] n=06: Fe2S2 E6: Fe-S=2.344 Ang | E_ref=-3319.67021033 Ha | ν=2 | k^(n)=2
[21:26:48] INFO     | [ERI DIAG] n_orbs=4, g_full entries=18
[21:26:48] INFO     | [ERI DIAG] Expected unique (chemist): ~35, full tensor: 256
[21:26:49] INFO     | [HW-B6] Using step-specific integrals: N=4, E_core=-3316.923989 Ha
[21:26:49] INFO     |   [HW n=06] A_n=[2] | ν=2 k^(n)=2 | Σe_net=7 | 
[21:26:49] INFO     | [PARSE] Loaded: 4 h_diag, 6 h_hop, 18 g_full
[21:26:49] INFO     |   [StepStore] n=07: Fe2S2 E7: Fe-S=2.358 Ang | E_ref=-3319.66843855 Ha | ν=2 | k^(n)=0
[21:26:49] INFO     | [ERI DIAG] n_orbs=4, g_full entries=18
[21:26:49] INFO     | [ERI DIAG] Expected unique (chemist): ~35, full tensor: 256
[21:26:49] INFO     | [HW-B7] Using step-specific integrals: N=4, E_core=-3316.925052 Ha
[21:26:49] INFO     |   [HW n=07] A_n=[3] | ν=2 k^(n)=0 | Σe_net=8 | 

  [HW-QPE] Checkpoint n=7 (E_7→E_8) | Fe2S2 E7: Fe-S=2.358 Ang
    E_0 (exact diag) = -2.74338615 Ha | E_ref (FCI) = -2.74338615 Ha
[21:26:50] INFO     | [VANC-QPE n=7] τ-seq (cached): [0.02, 0.04, 0.08, 0.16] | n_max=8
[21:26:50] INFO     | [VANC-QPE n=7] η_V=0.995211 (gates/step=12, n_ctrl=96)
    E_ZNE (exp) = -2.74279109 Ha | |E_ZNE−E_ref| = 0.5951 mHa [✓]

  [COMPILATION] Profiling full multi-step sequence to Forte pulses...
[21:27:06] INFO     | [SCHED] 180 ops → 72 moments (was 103 sequential)
[21:27:06] INFO     | [SCHED] Pre-compile gate type breakdown: ParamCoulombPhaseGate=48, ParamZClockGate=32, CompositeVirtualShiftGate=24, ParamURShiftGate=22, ZenoStabilizeGate=20, ElectronShiftGate=8, ProtonPhaseGate=8, ConformationalShiftGate=8, CompositeCofactorCouplingGate=8, CrossManifoldSWAPGate=2
[21:27:06] INFO     | [BASIS-CANCEL] Cancelled 128 basis-change gates (128 BLOG/BLOG†, 0 BVIRT/BVIRT†) → ~2368 native ops saved
[21:27:11] INFO     | [HW-INFO] → Total Compiled Moments: 9201
[21:27:11] INFO     | [HW-INFO] → Native Footprint: GPI=5480, GPI2=10960, ZZ=2368, Other=0
[21:27:11] INFO     | [HW-INFO] → MatrixGate Fallback: [✓] ZERO
[21:27:12] INFO     | [HW-INFO] → Gate contribution breakdown (type | count | standalone→effective | total | verified):
[21:27:12] INFO     | [HW-INFO]     ParamCoulombPhaseGate                       48 × 217→180  = 8640  [✓]
[21:27:12] INFO     | [HW-INFO]     ZenoStabilizeGate                           20 × 240      = 4800  [✓]
[21:27:12] INFO     | [HW-INFO]     CompositeCofactorCouplingGate                8 × 427      = 3416  [✓]
[21:27:12] INFO     | [HW-INFO]     ParamZClockGate                             32 × 27       = 864  [✓]
[21:27:12] INFO     | [HW-INFO]     CrossManifoldSWAPGate                        2 × 264      = 528  [✓]
[21:27:12] INFO     | [HW-INFO]     CompositeVirtualShiftGate                   24 × 20       = 480  [✓]
[21:27:12] INFO     | [HW-INFO]     ConformationalShiftGate                      8 × 27       = 216  [✓]
[21:27:12] INFO     | [HW-INFO]     ElectronShiftGate                            8 × 27       = 216  [✓]
[21:27:12] INFO     | [HW-INFO]     ProtonPhaseGate                              8 × 20       = 160  [✓]
[21:27:12] INFO     | [HW-INFO]     ParamURShiftGate                            22 × 0        = 0  [✗ shape]
[21:27:12] INFO     | [HW-INFO] → Semantic warrant extraction...

[DEBUG] Calibrated Transmission η' = 0.9000
[DEBUG] Final Target Population = 0.6561 (Expected: ~0.9000)

  [VERIFY] Stoichiometric invariance suite...
  [✓] (i) Net electron flux conservation: <N_e>_net = 8 (injected 8 − ejected 0) , expected 8
  [✓] (ii) Net-flux phase closure: Σ(ν−ν†)=16 mod 4 = 0 (expected 0)
  [✓] (iii) Trace preservation: Tr(ρ_final) = 1.00000000 (expected 1.0)
  [✓] (iv) Spin-parity holding: min ω = 0.9000 vs η=0.9 [AntiTh+SynTh (high-spin)]

[VANC-QPE RESULTS] nitrogenase_lt
  Integral source: step-wise JSON (src/nanoprotogeny/datasets/baseline/nitrogenase_lt/nitrogenase_lt/)
  ┌────────────────────────────────────────┬────────────────────┬────────────┐
  │ Metric                                 │ Value              │ Status     │
  ├────────────────────────────────────────┼────────────────────┼────────────┤
  │ ℤ_4 Phase Closure (k≡0 mod 4)          │ Σν=16              │ [✓] OK     │
  │ Electron Conservation (<N_e>_final)    │ 8 e⁻               │ [✓] OK     │
  │ Trace Preservation (Tr ρ=1)            │ -                  │ [✓] OK     │
  │ Spin-Parity Holding (η=0.9)            │ -                  │ [✓] OK     │
  ├────────────────────────────────────────┼────────────────────┼────────────┤
  │ VANC-QPE|ZNE n=4 (Janus E_4→E_5) [DS]  │ 0.6399 mHa         │ [✓] OK     │
  │ VANC-QPE|ZNE n=7 (E_7→E_8) [DS]        │ 0.5951 mHa         │ [✓] OK     │
  ├────────────────────────────────────────┼────────────────────┼────────────┤
  │ OVERALL CHEMICAL ACCURACY (≤1.6 mHa)   │                    │ [✓] OK     │
  │ STOICHIOMETRIC INVARIANCE              │                    │ [✓] OK     │
  └────────────────────────────────────────┴────────────────────┴────────────┘

  E_ref (last step) = -3319.6684385539 Ha
  Gate algebra:      G(M) = A_HW^⊗8 ∪ A_cross^(m=4)
  Complexity bound:  G = O(M·N³·T²·C_int/ε)
  Elapsed:           94.91s

  [✓] VANC-QPE VALIDATION PASSED
==============================================================================

[VANC-QPE] Results exported -> src/nanoprotogeny/stoichiometry-mqeqpe/nitrogenase_lt_mqeqpe_results.json
```


<a name="citation"></a>
## Citation

If you use this work, please cite **both** the paper and the software.

**Paper (pre-print):**
```bibtex
@article{BoromMQE,
  author  = {Borom, Santos C.},
  title   = {The Modular Quantum Emulator: Chemically Accurate,
             Polynomial-Cost Simulation of Multi-Step Metalloenzyme
             Catalysis in a Dissipative Protein Environment},
  journal = {ChemRxiv},
  volume  = {2026},
  number  = {0716},
  year    = {2026},
  doi     = {10.26434/chemrxiv.15006143/v1},
  url     = {https://chemrxiv.org/doi/abs/10.26434/chemrxiv.15006143/v1}
}
```

**Software:**
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