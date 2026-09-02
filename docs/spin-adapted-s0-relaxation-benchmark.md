# S0 isotropic singlet/triplet relaxation benchmark

## Purpose

This checkpoint validates the first fully coupled stationary spin-adapted
calculation after the neutral-reference and open-shell orbital layers were
established.

The benchmark remains deliberately material agnostic.  It uses the canonical
coding control selected for the new implementation:

- `J1 = J2 = 0.100 eV`;
- `alpha1 = alpha2 = 3.0 eV/A`;
- `K1 = 16.51 eV/A^2`;
- `K2 = 0.51 eV/A^2`.

The simple density-density validation kernel initially uses `U = 0.525 eV` and
nearest-neighbour `V = 0.08 eV`.  These interaction values are regression
controls, not a material parameterization.

## Half-filled many-electron reference

The spin-adapted Miranda-style model is a many-electron pi-lattice problem, not
the previously validated distinguishable electron-hole pair model.  For the S0
control we therefore use one neutral pi electron per site.  On an even `N`-site
lattice the neutral state contains `N/2` doubly occupied spatial orbitals.
After a HOMO-to-LUMO excitation,

`n_closed = N/2 - 1`,

with the two frontier electrons coupled either as an open-shell singlet or as a
high-spin triplet.

For the historical 20x20 lattice this convention corresponds to
`n_closed = 199` and 400 neutral pi electrons.

## Symmetry-breaking seeds

A perfectly isotropic periodic lattice can retain a translationally symmetric
stationary solution if started exactly at zero distortion.  The benchmark
therefore uses only tiny deterministic structural perturbations:

1. `onsite`: a zero-mean local Holstein displacement;
2. `bond_x`: a zero-net-displacement x-bond Peierls seed;
3. `bond_y`: the exact square-lattice partner of `bond_x`.

The seed amplitude defaults to `1e-3 A`.  It is a numerical symmetry breaker,
not a physical parameter.  In the isotropic model, converged x- and y-seeded
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
branch matrix has converged.  No dynamics, thermostat, field, GPU backend, or
material-specific fit belongs to this checkpoint.
