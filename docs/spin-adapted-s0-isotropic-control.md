# S0 spin-adapted static foundation: isotropic control

## Scope decision

The spin-adapted implementation is developed first in the smallest parameter
space possible.  The legacy/reference Holstein-Peierls parameter naming is
mapped to the research notation as follows:

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
Therefore S0 is built in two complementary pieces rather than by silently
adding a `singlet`/`triplet` label to the old solver.

### S0a: general open-shell functional

The `spin_adapted.open_shell` module implements the stationary energy and
shell-Fock functional corresponding to the general open-shell equations used by
Miranda et al. for a site-basis density-density interaction kernel.

The following state definitions are encoded explicitly:

1. closed-shell singlet: occupation `(2)`;
2. open-shell singlet: occupations `(2,1,1)` with
   `a=1` and
   `b=[[1,1,1],[1,2,-2],[1,-2,2]]`;
3. high-spin triplet: occupations `(2,1)` with
   `a=[[1,1],[1,1]]` and `b=[[1,1],[1,2]]`.

The validation gates now include:

- noninteracting singlet/triplet degeneracy;
- `<S^2>=0` for the singlet and `<S^2>=2` for the triplet;
- reduction of the closed-shell Fock matrix to `T + 2J - K`;
- an exchange-driven Hubbard-dimer singlet-triplet splitting;
- direct comparison of both the open-shell singlet and high-spin triplet
  energies with an independently constructed fixed-particle Fock-space
  Hamiltonian on a tiny system.

### State-specific orbital optimization

`spin_adapted.orbital_optimization` now provides the first stationary orbital
optimizer.  It uses a complete orthonormal orbital matrix and exponential
orbital rotations.  The analytic anti-symmetric gradient is

`M = sum_mu n_mu [P_mu, F_mu]`.

Only rotations between orbital subspaces with different occupation numbers are
allowed.  Rotations within one shell are gauge degrees of freedom.  Rotations
between the two different singly occupied singlet shells are also held fixed,
consistent with the minimal fixed-coefficient Miranda variational restriction.

The analytic orbital-rotation derivative has been checked against central
finite differences.  Closed-shell optimization decreases the energy while
preserving orthonormality, and the interaction-free optimized singlet and
triplet controls remain degenerate.

### S0b-control: reduced electron-hole exchange bridge

A separate `spin_adapted.pair_control` module adds a frozen short-range
exchange kernel to the validated distinguishable pair Hamiltonian.  This is
**not** the production MCTDHF model and its exchange values are not material
parameters.  Its purpose is to validate the software/physics bridge:

- zero exchange recovers the existing spin-blind exciton solver;
- positive exchange splits the two spin sectors with the standard two-open-shell
  sign convention;
- the pair RDMs continue to drive the Holstein-Peierls Hellmann-Feynman forces;
- the structural gradient agrees with finite differences in the presence of the
  exchange control.

In the one-site atomic control,

`E_S = E_direct + K`,

`E_T = E_direct - K`,

and therefore `E_S - E_T = 2 K`.

This reduced bridge will remain a regression/reference backend after full
open-shell SCF/MCTDHF is available.

## Remaining S0 physics before coupled lattice relaxation

The next step is **not** to couple the many-electron open-shell density directly
to the old one-carrier Holstein force.  Doing that would incorrectly let the
entire neutral electronic background drive the molecular deformation.

Before simultaneous open-shell electronic + lattice relaxation, the excited
state must be referenced consistently to the neutral closed-shell background.
For the molecular-crystal problem this requires an explicit excitation-density
(or equivalent two-band HOMO/LUMO) construction so that the lattice couples to
the electronic change associated with the excitation rather than to all
occupied electrons.

The remaining S0 sequence is therefore:

1. define the neutral closed-shell reference and excitation-density convention;
2. validate the corresponding electronic energy difference against tiny exact
   controls;
3. derive the Holstein and Peierls structural derivatives for the excitation
   energy;
4. verify all structural derivatives by central finite differences;
5. perform the first simultaneous singlet/triplet + lattice relaxations using
   the canonical isotropic `J1=J2=0.100`, `alpha1=alpha2=3.0` control;
6. only after these gates pass, promote S0 and move to O0 occupations/yields.

No dynamics propagator, thermostat, electric field, GPU path, or
material-specific parameterization is introduced in S0.
