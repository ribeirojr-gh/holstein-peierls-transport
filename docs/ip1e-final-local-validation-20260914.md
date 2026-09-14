# IP1e final local validation — 2026-09-14

## Status

IP1e is numerically closed after the no-dynamics size-aware energy-balance recheck.

Local environment:

- Python 3.12.3
- NumPy 2.5.3
- SciPy 1.18.1
- WSL2 Linux
- `OPENBLAS_NUM_THREADS=1`
- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`

Validation:

- focused size-aware tests: PASS;
- full regression suite: **405 passed**;
- pycompile: PASS;
- size-aware posthoc recheck: PASS;
- failed gates: none;
- rechecked numerical pass: true.

## Energy-balance gate correction

The original IP1e attempt reused the absolute `5e-5 eV` balance threshold calibrated on a 20x20 lattice for the 40x40 controls. The generalized balance residual is extensive, so the finite-size recheck preserves the 20x20 threshold exactly and scales it linearly with site count:

`tol(N) = 5e-5 eV * N / 400`.

Results:

| protocol | sites | max residual [eV] | residual/site [eV] | size-aware tolerance [eV] | status |
|---|---:|---:|---:|---:|---:|
| 20x20, gamma_v=0.01 | 400 | 1.51860856e-05 | 3.79652140e-08 | 5.0e-05 | PASS |
| 40x40, gamma_v=0.01 | 1600 | 5.33097094e-05 | 3.33185684e-08 | 2.0e-04 | PASS |
| 40x40, gamma_v=0.002 | 1600 | 5.26707978e-05 | 3.29192486e-08 | 2.0e-04 | PASS |

The larger cells therefore have no degradation in residual per site. No trajectory was changed or rerun by the recheck.

## Physical sensitivity result at 300 K

For the isotropic condition `J0y/J0x=1.0`:

| protocol | complete events | <Delta q matched> | positive q advantage | current positive +/-100 fs | future-bond top1 -100:-80 fs |
|---|---:|---:|---:|---:|---:|
| 20x20, gamma_v=0.01 | 22 | 0.2465 | 0.818 | 0.955 | 0.799 |
| 40x40, gamma_v=0.01 | 30 | 0.1721 | 0.800 | 0.933 | 0.797 |
| 40x40, gamma_v=0.002 | 33 | 0.1468 | 0.697 | 0.909 | 0.480 |

The direction-specific lattice/current correlation survives both doubling the cell and reducing the intermolecular damping. The main qualitative change under weaker `gamma_v` is that the future direction is less often represented by a single locally dominant transfer bond, while the full lattice response and probability current remain directionally correlated.

This is consistent with a more collective, longer-memory intermolecular response when the Peierls bath is less strongly damped. It is not a calibrated material-lifetime result because neither `gamma_v` value is material fitted.

## Interpretation boundary

IP1e does not establish a production hopping rate, activation energy, diffusion coefficient, or mobility. It establishes that the IP1d mechanism is not simply a 20x20 finite-size artifact and that intermolecular phonon memory changes the microscopic local-bond signature without destroying the broader carrier-lattice correlation.

The next stage is IP1f: direct carrier-centered characterization of the emitted lattice wake, with explicit lattice-energy current and longitudinal/transverse mode decomposition.
