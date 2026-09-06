# IP0c — constrained relaxed one-site translation profile

## Motivation

IP0a showed that the frozen one-site translation barrier along the easy x direction remains near 11–13 meV while transfer isotropy is approached. IP0b showed that the corresponding lattice reaction-coordinate inertia grows by about 2.46x and the associated local harmonic frequency scale falls by about 39%.

Those results rule out a simple explanation based only on a growing frozen potential barrier. They also show that the frozen midpoint is not generally the relaxed transition configuration.

IP0c therefore asks:

> How much of the frozen barrier survives when every lattice degree of freedom orthogonal to a controlled translation coordinate is allowed to relax?

## Reaction coordinate

For a relaxed source lattice state `q_A` and its exact one-site translation `q_B`, define

`d = q_B - q_A`

and the dimensionless scalar coordinate

`s(q) = [(q-q_A)·d]/[d·d]`.

At each prescribed `s`, the energy is minimized subject to this linear projection being fixed.

Uniform `vx` and `vy` shifts are exact gauge/zero modes because the Hamiltonian and elastic energy depend only on displacement differences. Those modes are explicitly projected out during optimization.

## What is relaxed

At fixed `s`, all components of `u`, `vx` and `vy` orthogonal to the endpoint translation vector are optimized on the instantaneous electronic ground-state Born-Oppenheimer surface.

The electronic ground state is re-solved at each optimizer evaluation and analytic Hellmann-Feynman lattice gradients are used.

The constrained minimizer uses projected L-BFGS. No empirical force or post-hoc energy correction is introduced.

## What IP0c is not

The path is a set of independently relaxed hyperplane minima. It is more physical than the frozen linear interpolation, but it is not yet claimed to be:

- a nudged elastic band minimum-energy path;
- a string-method MEP;
- a finite-temperature potential of mean force;
- an Arrhenius activation energy;
- a hopping rate;
- a mobility.

A branch switch between local minima is possible and will be diagnosed rather than hidden.

## Control scan

To keep this first constrained calculation focused, IP0c uses:

- 20x20 periodic lattice;
- `J0x = 0.100 eV`;
- `J0y/J0x = 0.15, 0.50, 1.00`;
- both `+x` and `+y` translation directions;
- 7 fixed reaction-coordinate images including endpoints;
- sparse electronic ground-state solver;
- optimized analytic gradient;
- projected-gradient target 2e-6 eV/A;
- maximum 300 L-BFGS iterations per interior image.

The 0.15 and 1.00 cases compare the previously mobile anisotropic control with the isotropic limit; 0.50 provides an intermediate point.

## Recorded observables

For every image IP0c stores:

- relaxed adiabatic energy;
- frozen energy at the same `s`;
- energy lowering due to orthogonal relaxation;
- electronic eigenvalue;
- IPR and participation number;
- source and target populations;
- projected-gradient residual;
- reaction-coordinate error;
- optimizer iterations and success state.

For each profile it reports the relaxed barrier, frozen barrier, barrier reduction and saddle fraction.

## Numerical gates

IP0c requires:

1. all static endpoint relaxations converged;
2. all constrained images meet the projected-gradient and coordinate criteria;
3. translated endpoints agree in energy within 1e-8 eV;
4. the fixed reaction coordinate is preserved within 1e-9;
5. maximum projected gradient is below 1e-5 eV/A;
6. a relaxed image never lies above its corresponding frozen image by more than 1e-8 eV;
7. all relaxed barriers are finite and non-negative within numerical tolerance.

No barrier magnitude or anisotropy trend is pre-registered as a physical PASS condition.

## Decision after IP0c

Three qualitatively different outcomes are possible.

1. **The isotropic barrier remains small after relaxation.** Then static pinning is even less likely to explain the old immobility, and the next priority is finite-temperature dynamical hopping, reaction-coordinate inertia and transient symmetry breaking.
2. **The isotropic relaxed path develops a larger barrier than the frozen path suggested.** Then the independent-hyperplane relaxation has found a different branch and a true MEP/string treatment becomes necessary before temperature scans.
3. **The constrained path becomes discontinuous or difficult to converge.** That itself signals multiple competing polaron branches; IP0d would then use a coupled string/NEB treatment with continuity control.

Only after a stable relaxed path is established will the project extract finite-temperature hopping rates or compare activation scales.