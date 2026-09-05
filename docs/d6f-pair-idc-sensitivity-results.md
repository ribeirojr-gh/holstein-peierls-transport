# D6f pair IDC decoherence-interval sensitivity results

## Scope

D6f compares destruction-of-phase (DP) and Boltzmann-modified (BM) instantaneous decoherence controls for the finite-temperature D6 pair sectors over a numerical-control interval sweep. The lattice is 4x4, the bath is 300 K with gamma_u = gamma_v = 0.01 fs^-1, dt = 0.2 fs, the trajectory length is 4 ps with 1 ps burn-in, CF4-Lanczos uses m = 8, and four stochastic seeds are used for each point. The decoherence intervals are 50, 100, 180, 250 and 500 fs.

The interval is a phenomenological model parameter. D6f does not calibrate a material-specific decoherence time.

## Numerical validation

The local runner completed successfully:

- 39/39 focused D6 tests passed;
- 320 total pytest tests passed;
- all 80 trajectories completed;
- all pre-registered temperature, generalized-energy-balance, norm and sector-constraint gates passed;
- the largest generalized balance residual remained of order 2.2e-6 eV;
- norm and sector constraints remained at roundoff.

The local calculation used Python 3.12.3 under WSL2 with NumPy 2.5.2 and SciPy 1.18.1, with OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1. The source tree was downloaded as a ZIP, so git commit/branch metadata in the local artifact are `unknown`.

## Bipolaron

DP is excellent at short intervals but is not robust across the full sweep. Ensemble mean TV distance to the instantaneous canonical distribution is approximately 2.1e-5 at 50 fs and 3.6e-5 at 100 fs. At longer intervals a seed-dependent failure appears. For seed 20260907 the mean canonical TV grows to about 9.5e-3 at 180 fs, 2.42e-2 at 250 fs and 0.732 at 500 fs. The corresponding ensemble mean canonical TV reaches 0.183 at 500 fs.

BM remains close to canonical throughout the full 50--500 fs interval range. Ensemble mean canonical TV stays between about 1.5e-5 and 4.6e-5, with ground-manifold mismatch at the same scale. Mean pre-collapse heating remains of order 1e-5 or smaller. BM therefore removes the long-interval DP instability in this control.

BM has a more negative electronic-environment exchange rate than DP, as expected from the explicit thermal reweighting. This exchange is recorded explicitly and is not repaired by hidden lattice-velocity rescaling.

**D6f bipolaron selection:** BM is the downstream reference scheme. DP remains a useful short-interval control but is not selected as the robust production candidate.

## Exciton

DP remains very close to the instantaneous canonical reference over the entire 50--500 fs sweep. Ensemble mean canonical TV is approximately 2.4e-6, 4.7e-6, 1.24e-5, 1.09e-5 and 1.40e-5 for 50, 100, 180, 250 and 500 fs, respectively. Ground-manifold mismatch follows the same scale.

BM provides no systematic improvement large enough to justify the extra thermal reweighting. At 180, 250 and 500 fs the DP and BM trajectories coincide for the sampled random streams because the same collapse states are selected. The exciton therefore does not require BM in this control.

**D6f exciton selection:** DP is the downstream reference scheme because it is the more parsimonious correction and remains robust across the full tested interval range.

## Downstream control interval

For the final D6 closure, `t_d = 100 fs` is selected as a **common numerical-control interval** for both sectors. It lies inside the robust region of both selected schemes and allows a clean sector-to-sector comparison. It is not a material-specific decoherence time and must remain user-visible/configurable in later production calculations.

The final closure therefore uses:

- bipolaron: BM, t_d = 100 fs;
- distinguishable e-h exciton: DP, t_d = 100 fs.

A 10 ps ensemble closure is required before D6 is declared complete. The final gate must also rerun the repository-wide D0a and six S0 numerical closure checks.
