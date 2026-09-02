# S0 spin-adapted static foundation: isotropic control

## Scope decision

The spin-adapted implementation will be developed first in the smallest
parameter space possible.  The legacy/reference Holstein-Peierls parameter
naming is mapped to the research notation as follows:

- `J1 -> j0x`;
- `J2 -> j0y`;
- `alpha1 -> alpha_intra` (Holstein coupling);
- `alpha2 -> alpha_interx = alpha_intery` (Peierls coupling).

For S0 coding and validation we impose

- `J1 = J2 = 0.100 eV`, using the current/reference `J1` value;
- `alpha1 = alpha2 = 3.0 eV/A`, using the current/reference `alpha1` value.

This is a deliberately symmetric **numerical control**, not a material fit.
The existing legacy defaults remain unchanged outside the S0 control adapter.

`K1 = 16.51 eV/A^2` and `K2 = 0.51 eV/A^2` are retained independently.  They
are not made equal because that was not part of the present simplification
request and because they describe physically different intra- and
intermolecular lattice modes in the historical model.

The canonical mapping is implemented by
`spin_adapted.IsotropicControlParameters`.  It can generate both
`StaticPolaronParameters` and `ExcitonParameters`, so every S0 comparison uses
one explicit source of truth.

## Why the static spin-adapted layer is split into two controls

The existing `exciton` package is a distinguishable electron-hole pair model.
It is useful for direct Coulomb binding and for matrix-free numerical tests, but
its electronic Hilbert space does not by itself contain the open-shell
many-electron exchange structure required by the Miranda MCTDHF formalism.
Therefore S0 is being built in two complementary pieces rather than by silently
adding a `singlet`/`triplet` label to the old solver.

### S0a: general open-shell functional

The new `spin_adapted.open_shell` module implements the stationary energy and
shell-Fock functional corresponding to the general open-shell equations used by
Miranda et al. for a site-basis density-density interaction kernel.

The following state definitions are encoded explicitly:

1. closed-shell singlet: occupation `(2)`;
2. open-shell singlet: occupations `(2,1,1)` with
   `a=1` and
   `b=[[1,1,1],[1,2,-2],[1,-2,2]]`;
3. high-spin triplet: occupations `(2,1)` with
   `a=[[1,1],[1,1]]` and `b=[[1,1],[1,2]]`.

This is the physical core that will be used by the orbital optimizer and later
by the MCTDHF orbital generator.

The first exact algebra controls are already defined:

- the noninteracting singlet/triplet limit is degenerate;
- the expected `<S^2>` values are `0` and `2`;
- the closed-shell Fock matrix reduces to `T + 2J - K`;
- a two-site Hubbard open-shell control gives the expected exchange-driven
  singlet-triplet splitting.

### S0b-control: reduced electron-hole exchange bridge

A separate `spin_adapted.pair_control` module adds a frozen short-range
exchange kernel to the validated distinguishable pair Hamiltonian.  This is
**not** the production MCTDHF model and its exchange values are not material
parameters.  Its purpose is to validate the software/physics bridge:

- zero exchange must recover the existing spin-blind exciton solver;
- positive exchange must split the two spin sectors with the standard
  two-open-shell sign convention;
- the pair RDMs must continue to drive the correct Holstein-Peierls
  Hellmann-Feynman forces;
- singlet/triplet fixed-lattice and relaxed-lattice code paths can be tested
  before the more expensive orbital self-consistency is introduced.

In the one-site atomic control,

`E_S = E_direct + K`,

`E_T = E_direct - K`,

and therefore `E_S - E_T = 2 K`.

This reduced bridge will remain a regression/reference backend after full
open-shell SCF/MCTDHF is available.

## Next S0 implementation step

The next code block is the state-specific **orbital optimization** for the
open-shell functional.  It should use orbital rotations rather than independent
diagonalization of each shell Fock matrix, because the Miranda variational
condition couples different occupation shells and contains gauge freedom within
each shell.

The first optimizer validation sequence will be:

1. closed-shell small-lattice convergence;
2. open-shell singlet and triplet on a tiny isotropic lattice;
3. finite-difference derivative of the energy with respect to orbital rotations;
4. comparison with direct determinant/configuration construction on very small
   Hubbard-like systems;
5. exchange-off singlet/triplet degeneracy;
6. exchange-on state ordering and `<S^2>` checks;
7. only then, simultaneous electronic + lattice relaxation using the canonical
   isotropic `J1=J2=0.100`, `alpha1=alpha2=3.0` control.

No dynamics propagator, thermostat, field, GPU path, or material-specific
parameterization is introduced in S0.
