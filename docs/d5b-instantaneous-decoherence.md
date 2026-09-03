# D5b instantaneous decoherence and energy-relaxation controls

## Motivation from D5a

The validated D5a 20x20, 300 K control showed progressive electronic overheating while the D4 lattice remained correctly thermalized. The ensemble heating coordinate increased from about 0.03 early in the post-burn-in window to about 0.54 late in the 10 ps trajectory, with positive slopes for all four seeds. D5b therefore benchmarks explicit electronic-environment corrections.

## Reference algorithm

The initial D5b controls follow Wei Si and Chang-Qin Wu, J. Chem. Phys. 143, 024103 (2015), DOI 10.1063/1.4926534. Their instantaneous decoherence correction (IDC) divides coherent quantum-classical dynamics into finite segments and performs a measurement-like collapse in the instantaneous adiabatic basis after each decoherence interval.

If the pre-collapse adiabatic populations are

`p_mu = |<E_mu|psi>|^2`,

D5b implements three schemes:

- IDC-DP: `q_mu = p_mu`;
- IDC-BM: `q_mu proportional to p_mu exp(-E_mu / k_B T)`;
- IDC-MA: `q_mu proportional to p_mu exp[-max(E_mu-Ebar,0)/(k_B T)]`, where `Ebar = sum_mu p_mu E_mu`.

The formulas are evaluated with stable energy shifts and normalized explicitly before stochastic selection.

## Energy bookkeeping

A collapse occurs at fixed lattice coordinates and velocities, so it introduces a discontinuous electronic-energy change

`Q_electronic = E_selected - <psi|H|psi>`.

This is recorded explicitly and is not hidden by a velocity rescaling. At zero external field the numerical energy-balance diagnostic is

`Delta E_matter ~= Q_lattice_bath + Q_electronic`.

Positive `Q_electronic` means the model electronic environment injected energy into the matter subsystem; negative values mean it removed energy.

## Independent random streams

The classical Langevin bath and the electronic collapse use independent caller-owned NumPy generators. This prevents a change in decoherence scheme from merely shifting the sequence of random numbers used by the D4 thermostat.

## Initial validation control

The first scientific D5b benchmark uses:

- 20x20 lattice;
- 300 K;
- gamma_u = gamma_v = 0.01 fs^-1;
- dt = 0.2 fs;
- 10 ps total time;
- 2 ps burn-in;
- projected vx/vy collective zero modes;
- CF4-Lanczos with Krylov dimension 6;
- four independent lattice seeds;
- all three schemes DP, BM, and MA;
- a fixed decoherence interval of 100 fs.

The 100 fs interval is a numerical control only. Si and Wu used about 180 fs for a different one-dimensional parameter set. Their value is not transferred as a pentacene or Holstein-Peierls material parameter here. Decoherence-time sensitivity is a separate downstream gate.

## Correct equilibrium diagnostic for collapsed trajectories

A single post-collapse realization is an adiabatic pure state by construction, so its occupation vector should not be compared directly with a canonical mixed distribution. D5b therefore records:

1. the D5a equilibrium diagnostics immediately before every IDC event after burn-in;
2. the expected post-collapse electronic energy from the stochastic collapse distribution;
3. the actually selected post-collapse energy;
4. lattice temperature and electronic norm;
5. the generalized matter-energy residual including both lattice-bath heat and collapse energy exchange.

The preferred correction must reduce or eliminate the D5a late-time overheating without destabilizing the lattice bath, violating norm preservation, or producing unacceptable energy bookkeeping. No scheme is selected in advance.
