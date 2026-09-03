# D5a local validation and physical diagnosis — 2026-09-03

D5a was executed locally from a downloaded ZIP of branch `d5-electronic-thermalization-decoherence` under Python 3.12.3, NumPy 2.5.2, SciPy 1.18.1, WSL2 Linux, with OpenBLAS/OpenMP/MKL restricted to one thread.

Because a ZIP rather than a Git clone was used, the runner recorded `git_commit`, `git_branch`, and `git_status` as `unknown`. The intended source head before the run was `8f604b3a7ffab8c83d0f5dc3ecef8173c05db351`.

## Regression status

D0a and all six S0 relaxation regressions passed. The D5a ensemble benchmark also completed successfully.

Two pytest gates were initially marked failed solely because one test compared the analytically exact participation number of a uniform five-state distribution using strict binary equality: floating-point evaluation returned `4.999999999999999` instead of `5.0`. No model or integrator quantity failed. The test was corrected to use a roundoff-level `np.isclose` assertion in commit `4ffdd53`.

## D5a control

- lattice: 20x20
- bath temperature: 300 K
- gamma_u = gamma_v = 0.01 fs^-1
- dt = 0.2 fs
- final time = 10 ps
- burn-in = 2 ps
- diagnostic interval = 100 fs
- zero-mode policy: projected
- electronic propagator: CF4-Lanczos, Krylov dimension 6
- four independent Langevin seeds: 20260903–20260906

The calculation comprised 200,000 coupled propagation steps and 320 post-burn-in electronic diagnostic snapshots.

## Numerical integrity

- maximum electronic norm error: 2.434e-12
- maximum sampled |Delta E - Q_lattice_bath|: 1.835e-05 eV
- mean lattice temperature over trajectories: 298.779 K
- trajectory-to-trajectory standard deviation of mean lattice temperature: 1.915 K

The classical thermal bath therefore remains correctly controlled while the electronic diagnostic evolves.

## Electronic thermalization diagnosis

Ensemble means over four trajectories:

- mean heating coordinate: 0.2726
- early heating coordinate: 0.0315
- late heating coordinate: 0.5401
- mean heating-coordinate slope: +0.09339 ps^-1
- mean beta_eff / beta_bath over finite-beta samples: 0.5694
- fraction of non-positive beta_eff samples: 0
- mean TV distance to canonical occupations: 0.5190
- mean Jensen-Shannon distance to canonical occupations: 0.4733
- mean TV distance to uniform occupations: 0.7643
- fraction of samples closer to uniform than to canonical: 0.4094
- mean normalized adiabatic-occupation entropy: 0.4036
- mean adiabatic occupation participation number: 35.84
- mean propagated ground-manifold population: 0.4527
- mean instantaneous canonical ground-manifold population: 0.7264

Every trajectory had a positive heating-coordinate slope. Late heating coordinates for seeds 20260903–20260906 were 0.448, 0.706, 0.771, and 0.236, respectively.

## Interpretation

The result is not yet the infinite-temperature limit: occupations remain, on average, farther from the uniform distribution than from the canonical distribution, and beta_eff remains positive. However, the early-to-late drift is systematic and large, while the lattice temperature remains correctly thermalized. Thus coherent Ehrenfest dynamics exhibits progressive electronic overheating in this Holstein-Peierls control.

This is sufficient evidence to proceed to D5b. D5b must benchmark explicit decoherence/energy-relaxation extensions rather than assuming a literature correction is automatically valid.
