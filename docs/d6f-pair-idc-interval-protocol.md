# D6f pair IDC decoherence-interval sensitivity protocol

## Motivation

D6d established systematic electronic overheating for both coherent finite-temperature pair sectors. D6e then showed that at the single numerical-control interval `t_d = 100 fs`, both destruction-of-phase coherence (DP) and Boltzmann-modified (BM) IDC suppress that overheating almost completely. The D6e result is not sufficient to select a production model because a phenomenological decoherence method should not be accepted solely because it performs well at one tuned interval.

D6f therefore measures sensitivity to the decoherence interval without changing the validated D6e dynamics kernel.

## Candidates

Primary candidates:

- **DP**: collapse probability equals the instantaneous adiabatic population. This is the least thermodynamically prescriptive correction because it destroys phase coherence without an explicit Boltzmann reweighting.
- **BM**: adiabatic populations are reweighted by a Boltzmann factor before collapse. This is retained as the detailed-balance control.

MA is not included in the primary D6f sweep. D6e found a large bipolaron outlier for MA at 100 fs, whereas DP and BM were both stable. MA remains implemented and may be revisited if later evidence warrants it.

## Physical sectors

The two sectors remain separate:

1. **Bipolaron** — only the symmetric spatial singlet subspace is used for instantaneous diagonalization and collapse.
2. **Exciton** — the distinguishable electron-hole state uses the full ordered `N^2` basis.

No antisymmetric bipolaron state may enter the canonical reference or stochastic collapse.

## Numerical control

The pre-registered D6f control is:

- lattice: 4x4;
- temperature: 300 K;
- `gamma_u = gamma_v = 0.01 fs^-1`;
- `dt = 0.2 fs`;
- total time: 4 ps;
- burn-in: 1 ps;
- projected intermolecular zero modes;
- CF4-Lanczos, Krylov dimension 8;
- four lattice seeds: 20260905–20260908;
- schemes: DP and BM;
- decoherence intervals: **50, 100, 180, 250, 500 fs**.

This produces 80 trajectories: `2 sectors x 2 schemes x 5 intervals x 4 seeds`.

The interval grid is a numerical sensitivity grid, not a set of material parameters.

## Recorded diagnostics

For every sector/scheme/interval ensemble, D6f records:

- lattice temperature;
- mean, early, and late pre-collapse heating coordinate;
- heating-coordinate slope in `ps^-1`;
- total-variation distance to the instantaneous canonical distribution;
- total-variation distance to the uniform distribution;
- absolute ground-manifold population mismatch;
- effective-beta ratio as a secondary nonlinear diagnostic;
- expected and realized post-collapse heating;
- electronic-environment exchange rate in `eV/ps`;
- generalized energy-balance residual;
- norm error;
- pair-sector constraint error.

The generalized zero-field balance remains

`Delta E_matter ~= Q_lattice_bath + Q_electronic_environment`.

No lattice-velocity rescaling or hidden energy repair is applied.

## Numerical gates

For every sector/scheme/interval aggregate:

- mean lattice temperature must lie in 240–360 K;
- maximum generalized balance residual must remain below `1e-4 eV`;
- maximum norm error must remain below `1e-10`;
- maximum sector-constraint error must remain below `1e-10`.

These gates establish numerical execution only.

## Physical decision rule

There is deliberately **no weighted scalar score**.

A stronger downstream candidate should remain close to the canonical reference across a nontrivial interval range while avoiding systematic heating/cooling trends and unnecessary electronic-environment energy exchange.

Particular attention will be paid to:

1. robustness of the late heating coordinate and slope across `t_d`;
2. TV distance to canonical and ground-manifold mismatch;
3. whether DP remains competitive with BM as `t_d` increases;
4. the magnitude and sign of the electronic exchange rate;
5. seed-to-seed variability and isolated outliers.

If DP remains comparably canonical over a broad interval range, it will be preferred on parsimony grounds because it introduces less explicit thermal bias. If DP deteriorates while BM remains robust, BM will be the stronger candidate for that sector. The two pair sectors may select different downstream models or intervals.

D6f does not calibrate a material-specific decoherence time. Any selected numerical-control interval must subsequently undergo a longer closure trajectory before D6 can be considered complete.
