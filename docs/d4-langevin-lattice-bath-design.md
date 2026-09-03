# D4 finite-temperature lattice bath: BAOAB design

## Scope

D4 introduces a classical finite-temperature bath for the Holstein-Peierls lattice only. Electronic propagation remains coherent Ehrenfest dynamics; electronic decoherence or thermalization is deliberately deferred to D5.

The archived code already intended to use classical Markovian Langevin dynamics. D4 therefore preserves that physical model while replacing the fragile discrete implementation by a modern, explicitly validated splitting.

## Legacy issues that are not reproduced

The dynamics audit identified several finite-temperature problems in the archived implementation:

- random-number streams are repeatedly created and destroyed with seeds tied to integer simulation time;
- separate random arrays are generated on both half steps without a clearly documented discrete Langevin scheme;
- multiple damping constants are mixed between force and kick expressions;
- some intermolecular damping terms appear to use the wrong loop index;
- the active temperature diagnostic is `3 K/(N k_B)`, whereas equipartition for three unconstrained classical coordinates per site requires `2 K/(3 N k_B)`.

These are implementation artifacts, not physical requirements.

## Continuous Langevin equation

Each lattice velocity component obeys

\[
dv = \left(\frac{F(q)}{m}-\gamma v\right)dt
+\sqrt{\frac{2\gamma k_B T}{m}}\,dW.
\]

The project uses energies in eV, positions in angstrom, and time in fs. The previously validated mass conversion gives effective masses in `eV fs^2 / angstrom^2`; therefore `gamma` is expressed in `fs^-1` and the velocity noise has units `angstrom/fs`.

D4 defines separate numerical friction controls for the intramolecular `u` coordinate and the two intermolecular coordinate fields `vx`/`vy`. No material-specific damping value is promoted by this checkpoint.

## Exact Ornstein-Uhlenbeck substep

For a thermostat interval `dt`, the stochastic O step is integrated analytically:

\[
v' = c v + \sigma R,
\qquad c=e^{-\gamma dt},
\]

\[
\sigma^2=(1-c^2)\frac{k_B T}{m},
\]

with `R` an independent unit normal variate. This form satisfies the fluctuation-dissipation relation exactly for the isolated velocity process and has stationary variance

\[
\langle v^2\rangle = \frac{k_B T}{m}.
\]

A persistent caller-owned `numpy.random.Generator` supplies all random numbers. The implementation consumes no random number when the noise amplitude is exactly zero, making the frictionless limit strictly deterministic and zero-temperature damping reproducible without hidden RNG-state changes.

## BAOAB splitting

The initial D4 classical integrator is

`B(dt/2) A(dt/2) O(dt) A(dt/2) B(dt/2)`.

Here:

- `B`: deterministic force kick;
- `A`: coordinate drift;
- `O`: exact Ornstein-Uhlenbeck velocity update.

When both friction coefficients are zero, `O` is the identity and the two half drifts combine exactly. BAOAB therefore reduces to the D2 velocity-Verlet classical step, providing a strong deterministic regression gate.

BAOAB is preferred over blindly reproducing the archived BBK-like arithmetic because it separates force, drift, damping, and randomization into individually testable operations and is well established for configurational sampling of Langevin systems.

## Correct temperature diagnostic

For `N_dof` unconstrained classical velocity degrees of freedom,

\[
T_{kin}=\frac{2K}{N_{dof}k_B}.
\]

For the current unconstrained three-coordinate-per-site representation, `N_dof=3N`. If D4 later removes collective zero modes from the intermolecular coordinates, the reduced degree-of-freedom count must be supplied explicitly.

## Intermolecular zero modes

The elastic energy contains only spatial differences of `vx` and `vy`. Uniform offsets of either field are therefore exact zero modes. An unconstrained Langevin bath will make these physically redundant collective coordinates diffuse.

The first D4 checkpoint deliberately does **not** silently remove these modes. The thermostat primitives are validated first. Before long full-lattice finite-temperature production runs, D4 must explicitly compare two documented choices:

1. retain the unconstrained zero modes, matching the mathematical coordinates of the legacy model; or
2. project the uniform `vx` and `vy` collective modes out of both velocities and stochastic impulses and use the correspondingly reduced kinetic degrees of freedom.

This choice must be made from stability/observable tests rather than hidden inside the thermostat.

## Initial D4a validation gates

The isolated classical bath must pass all of the following before electronic coupling is added:

1. exact analytical OU damping factor and FDT noise variance;
2. frictionless O-step identity without consuming RNG state;
3. zero-temperature finite-friction deterministic exponential damping;
4. fixed-seed reproducibility with statistically independent coordinate channels;
5. correct `T=2K/(3Nk_B)` kinetic-temperature diagnostic;
6. exact frictionless BAOAB reduction to velocity Verlet;
7. stationary kinetic variance `m<v^2>=k_B T` in a large fixed-seed OU ensemble;
8. harmonic-oscillator equipartition, `<K>=<V>=k_B T/2`, using the actual legacy mass and spring-unit conventions.

## D4b coupled-bath gates

Only after D4a passes will the bath be coupled to the electronic-lattice dynamics. That stage must establish:

- a clearly defined BAOAB/Ehrenfest operator ordering;
- zero-friction reduction to D2 at zero field and to D3 when a field is enabled;
- electronic norm preservation under the same CF4-Lanczos propagator;
- thermal equilibrium statistics for an `E=0` 20x20 lattice control;
- thermostat reproducibility from a stored seed;
- temperature and energy-distribution convergence with timestep;
- explicit treatment of the two intermolecular collective zero modes;
- a multi-ps 300 K stability gate before any transport observable is interpreted.

D4 validates the classical bath only. Whether coherent mean-field Ehrenfest dynamics yields the correct electronic equilibrium distribution remains a separate D5 question.