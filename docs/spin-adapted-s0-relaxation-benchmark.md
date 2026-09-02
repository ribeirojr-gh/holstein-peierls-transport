# S0 isotropic singlet/triplet relaxation benchmark

## Purpose

This checkpoint validates the first fully coupled stationary spin-adapted
calculation after the neutral-reference and open-shell orbital layers were
established.

The benchmark remains deliberately material agnostic. It uses the canonical
coding control selected for the new implementation:

- `J1 = J2 = 0.100 eV`;
- `alpha1 = alpha2 = 3.0 eV/A`;
- `K1 = 16.51 eV/A^2`;
- `K2 = 0.51 eV/A^2`.

The simple density-density validation kernel initially uses `U = 0.525 eV` and
nearest-neighbour `V = 0.08 eV`. These interaction values are regression
controls, not a material parameterization.

## Half-filled many-electron reference

The spin-adapted Miranda-style model is a many-electron pi-lattice problem, not
the previously validated distinguishable electron-hole pair model. For the S0
control we therefore use one neutral pi electron per site. On an even `N`-site
lattice the neutral state contains `N/2` doubly occupied spatial orbitals.
After a HOMO-to-LUMO excitation,

`n_closed = N/2 - 1`,

with the two frontier electrons coupled either as an open-shell singlet or as a
high-spin triplet.

For the historical 20x20 lattice this convention corresponds to
`n_closed = 199` and 400 neutral pi electrons.

## Why the relaxation benchmark needs a gap

For the periodic nearest-neighbour isotropic square lattice,

`epsilon(k) = -2 J [cos(kx) + cos(ky)]`.

At half filling every even-site periodic rectangular cell contains exact
zero-energy states. The original 4x4 control, for example, has six degenerate
one-particle states at the Fermi level. A HOMO-to-LUMO label is therefore not
unique and the reoptimized excited-state Born-Oppenheimer surface develops
state crossings/cusps. This was observed numerically: both neutral and excited
orbital problems could satisfy the strict `1e-8` orbital-gradient gate while
the structural line search approached a non-smooth point with a large residual
lattice gradient.

This is a property of the deliberately idealized half-filled control, not a
reason to relax the convergence criteria.

The production S0 validation therefore adds a checkerboard site-energy term

`epsilon_i = +/- Delta/2`,

with default full one-particle gap `Delta = 0.8 eV`. The checkerboard term:

- preserves `Jx = Jy` and `alpha_x = alpha_y`;
- preserves the x/y square-lattice symmetry used by the bond-seed regression;
- converts the half-filled control into a well-conditioned band insulator;
- has no elastic energy or structural force of its own;
- is a validation control only and is not a material parameter.

The gap is implemented internally as a static electronic site-energy offset.
The physical lattice coordinates returned by the solver remain the true
`u`, `vx`, and `vy` coordinates.

## Structural optimizer

The legacy component-wise RPROP update is retained for the historical carrier
solver but is not used for this spin-adapted excited-state benchmark. The
harmonic lattice Hessian is known exactly:

- `K1 I` for the local Holstein coordinate `u`;
- `K2 Lx` for each periodic `vx` row;
- `K2 Ly` for each periodic `vy` column.

The S0 solver therefore constructs the exact fixed-electronic-state Newton /
preconditioned-gradient direction. The periodic Laplacians are inverted by FFT
with their translational zero modes fixed to zero mean through the
Moore-Penrose pseudoinverse. An Armijo line search then accepts steps only on
the fully reoptimized neutral-referenced Born-Oppenheimer energy.

If an electronic warm start fails, deterministic canonical/reseeded recovery is
performed at the same geometry before any structural step may be accepted.
Thus unconverged electronic forces never move the lattice.

## Symmetry-breaking seeds

A perfectly isotropic periodic lattice can retain a translationally symmetric
stationary solution if started exactly at zero distortion. The benchmark
therefore uses only tiny deterministic structural perturbations:

1. `onsite`: a zero-mean local Holstein displacement;
2. `bond_x`: a zero-net-displacement x-bond Peierls seed;
3. `bond_y`: the exact square-lattice partner of `bond_x`.

The seed amplitude defaults to `1e-3 A`. It is a numerical symmetry breaker,
not a physical parameter. In the isotropic model, converged x- and y-seeded
states should be degenerate whenever they represent the same basin.

## Acceptance criteria

For every promoted branch we require:

- neutral orbital optimization converged;
- excited orbital optimization converged;
- orbital-rotation gradient below `1e-8`;
- maximum structural update below `1e-8 A`;
- maximum structural gradient below `1e-6 eV/A`;
- `Tr(Delta gamma)` numerically zero;
- consistency of symmetry-related x/y seeds;
- independent singlet and triplet minimizations.

The permanent benchmark driver is
`experiments/spin_adapted_isotropic_relaxation.py`.

The first promoted numerical results will be added only after the complete
branch matrix has converged. No dynamics, thermostat, field, GPU backend, or
material-specific fit belongs to this checkpoint.
