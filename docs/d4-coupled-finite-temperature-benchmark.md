# D4 coupled finite-temperature lattice-bath benchmark

## Scope

D4 closes the finite-temperature **classical lattice bath** layer for the one-carrier Holstein-Peierls Ehrenfest dynamics. The electronic state remains coherently propagated with the deterministic CF4-Lanczos method selected in D1-D3. Electronic decoherence or explicit electronic thermalization is not introduced here and remains a separate D5 physics decision.

The production candidate is a BAOAB Langevin lattice split with an exact Ornstein-Uhlenbeck velocity substep, persistent seeded NumPy RNG, and the D2/D3 moving-lattice electronic propagation. The two uniform intermolecular zero modes are handled explicitly through a `zero_mode_policy`; the validated production control below uses `project`.

## D4a prerequisite

The isolated thermostat layer was first validated locally because GitHub Actions minutes were unavailable. The D4a package passed all eight dedicated gates, the complete repository test suite, D0a, and all six S0 relaxation branches. The local run used Python 3.12.3, NumPy 2.5.2, SciPy 1.18.1 and single-thread BLAS/OpenMP settings.

Because that validation was run from a GitHub ZIP rather than a Git checkout, its local metadata reported `git_commit=unknown` and `git_branch=unknown`. The code contents corresponded to the D4 branch ZIP supplied for validation; this provenance limitation is documented rather than hidden.

## D4b structural gates

The coupled bath implementation adds the following requirements:

1. zero-friction BAOAB/Ehrenfest reduction to the validated D2 zero-field trajectory;
2. zero-friction reduction to the validated D3 field-aware trajectory when a field is present;
3. electronic norm preservation under CF4-Lanczos without renormalization;
4. reproducible stochastic trajectories for a fixed persistent RNG seed;
5. explicit `retain` versus `project` treatment of the two uniform `vx`/`vy` modes;
6. `3N-2` kinetic degrees of freedom when those two modes are projected;
7. zero-field stochastic energy accounting through

   `Delta E_matter = Q_bath + numerical splitting error`;

8. field-driven stochastic accounting through

   `Delta E_matter = Q_bath + W_field + numerical splitting error`.

The D4-focused local test gate passed, and the full repository suite closed with **265 passed**.

## Local validation environment

- platform: Linux / WSL2, x86_64;
- Python: 3.12.3;
- NumPy: 2.5.2;
- SciPy: 1.18.1;
- `OPENBLAS_NUM_THREADS=1`;
- `OMP_NUM_THREADS=1`;
- `MKL_NUM_THREADS=1`.

As with the D4a validation, the run was executed from a ZIP and therefore recorded Git commit/branch metadata as `unknown`.

## 20x20 finite-temperature control

The control uses the relaxed 20x20 polaron, a 300 K lattice bath, `gamma_u = gamma_v = 0.01 fs^-1`, RNG seed `20260903`, projected intermolecular zero modes, and CF4-Lanczos with Krylov dimension 6. These damping values are numerical bath controls and are not promoted as material-specific parameters.

### Timestep comparison: 2 ps

| dt [fs] | <Tkin> [K] | std(T) [K] | max norm error | max |Delta E-Qbath| [eV] | final Delta E-Qbath [eV] | max zero-mode mean |
|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 300.932 | 10.487 | 1.047e-12 | 1.415e-05 | -1.175e-05 | 4.996e-18 |
| 0.1 | 298.483 | 11.244 | 2.321e-12 | 3.708e-06 | -3.004e-06 | 5.551e-18 |

The maximum bath-energy residual falls by a factor of **3.82** when the step is halved, and the final residual falls by a factor of **3.91**, consistent with the expected second-order global splitting error. The temperature estimates remain statistically consistent with 300 K.

At `dt=0.2 fs`, the measured mean lattice kinetic energy is 15.53345 eV; at `dt=0.1 fs` it is 15.40704 eV. For `3N-2 = 1198` projected kinetic degrees of freedom, equipartition predicts about 15.48535 eV at 300 K, placing both measurements within the expected finite-sampling fluctuations.

## 10 ps / 50,000-step stability gate

The final stability control uses `dt=0.2 fs`, 10 ps total time, 2 ps burn-in, and 4,000 post-burn-in samples.

Measured values:

- mean kinetic temperature: **299.97369 K**;
- temperature standard deviation: **11.84826 K**;
- mean lattice kinetic energy: **15.48399 eV**;
- expected equipartition kinetic energy for 1198 DOF: **15.48535 eV**;
- accumulated bath heat: **30.56880190 eV**;
- matter-energy change: **30.56878574 eV**;
- final `Delta E_matter - Q_bath`: **-1.61619e-05 eV**;
- maximum sampled `|Delta E_matter-Q_bath|`: **1.64911e-05 eV**;
- final residual relative to accumulated bath heat: **5.29e-7**;
- maximum electronic norm error: **1.643e-12**;
- maximum projected zero-mode mean: **6.66e-18**;
- final IPR: **0.0150896**, from initial **0.4477811**;
- population L2 change: **0.60894**;
- maximum lattice-coordinate excursion: **0.76356 angstrom**.

The thermal energy scale is internally consistent: the measured long-run kinetic energy agrees with the exact equipartition prediction to roughly `9e-5` relative. The energy transferred by the stochastic bath is also reproduced by the matter-energy change to sub-ppm relative accuracy. No secular electronic-norm instability or zero-mode drift is observed.

The substantial IPR and population change demonstrates a genuinely non-stationary finite-temperature electronic trajectory. It must **not** be interpreted as proof that coherent Ehrenfest dynamics samples the correct electronic thermal distribution; that is precisely the D5 question.

## D4 decision

D4 is numerically closed for the one-carrier lattice bath.

- **Production finite-temperature lattice scheme:** BAOAB with exact OU thermostat + moving-lattice CF4-Lanczos (`m=6`).
- **Validated control timestep:** `dt=0.2 fs` for the present 20x20 legacy-parameter, 300 K, `gamma=0.01 fs^-1` control only; it is not a universal timestep.
- **Preferred zero-mode policy for production controls:** explicitly project the two uniform `vx`/`vy` modes and use `3N-2` kinetic degrees of freedom.
- **RNG:** persistent caller-owned `numpy.random.Generator` with stored seed.
- **Zero-field energy diagnostic:** `Delta E_matter - Q_bath`.
- **Field-driven energy diagnostic:** `Delta E_matter - Q_bath - W_field`.

D5 must now determine whether pure coherent mean-field Ehrenfest dynamics gives an acceptable electronic equilibrium/transport description in the intended regime, or whether a controlled decoherence/thermalization extension is required. No D4 result by itself justifies a mobility or electronic-equilibrium claim.
