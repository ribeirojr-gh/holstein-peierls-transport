# IP1q final local validation — mode-resolved autonomous post-hop memory

Date: 2026-09-14  
Branch: `isotropic-polaron-barrier`  
Validated commit: `b390403366879910246bffed7a617a699897e7ca`  
Local artifact: `ip1q-local-validation/20260914T203122Z`

## Numerical closure

IP1q is numerically closed.

- runner status: PASS;
- failed gates: none;
- pycompile: PASS;
- focused tests: PASS;
- full suite: **454/454 PASS**;
- 40x40 zero-damping PBC recurrence preflight for the complete driven + 10 ps frozen continuation: PASS;
- selected natural event: 820 -> 819 (-x), transition start 2826 fs, accepted/switch at 2874 fs;
- frozen electronic state: bitwise unchanged;
- fixed-surface equilibrium force residual: `9.71445e-17`;
- maximum frozen-surface energy drift: `1.43043e-10 eV`;
- maximum modal/real-space Parseval error: `6.93889e-18 eV`;
- maximum surface-excitation closure error: `6.97359e-16 eV`;
- maximum zero-mode modal energy: `6.20368e-32 eV`;
- polarization and traveling-wave modal energies remain conserved within the preregistered numerical tolerances;
- q-omega ridges for vx and vy and the Holstein peak are within one spectral-frequency bin of the analytic harmonic frequencies.

The local working tree contained only untracked helper/editor files (`git-run.sh`, `.git-run.sh.swp`); the tracked validated source matched the commit above.

## Fixed-surface excitation energy

The autonomous excitation above the exact frozen-electronic equilibrium is

`E_exc = 0.011502317612246332 eV`.

At the switch this partitions as

- Holstein `u`: `1.42375347e-4 eV` = **1.2378%**;
- Peierls `vx`: `2.11093707e-3 eV` = **18.3523%**;
- Peierls `vy`: `9.24900519e-3 eV` = **80.4099%**;
- total Peierls fraction: **98.7622%**.

Thus the autonomous post-hop lattice excitation is overwhelmingly **Peierls/intermolecular**, not Holstein/intramolecular.

## Wave-number content

The Peierls excitation is concentrated in long-wavelength sectors.

For `vx`, 72.50% of its energy lies in the lowest nonzero quarter of the x Brillouin zone. The five largest |qx| sectors are

1. |q| = 0.47124 rad/site, wavelength 13.33 sites, 19.33%;
2. |q| = 0.62832 rad/site, wavelength 10 sites, 18.24%;
3. |q| = 0.31416 rad/site, wavelength 20 sites, 14.64%;
4. |q| = 0.78540 rad/site, wavelength 8 sites, 12.55%;
5. |q| = 0.15708 rad/site, wavelength 40 sites, 7.74%.

For `vy`, 90.85% lies in the lowest nonzero quarter of the y Brillouin zone. The five largest |qy| sectors are

1. |q| = 0.15708 rad/site, wavelength 40 sites, 29.82%;
2. |q| = 0.31416 rad/site, wavelength 20 sites, 24.70%;
3. |q| = 0.47124 rad/site, wavelength 13.33 sites, 18.04%;
4. |q| = 0.62832 rad/site, wavelength 10 sites, 11.62%;
5. |q| = 0.78540 rad/site, wavelength 8 sites, 6.67%.

The q-omega ridge agrees with the exact harmonic dispersion for all eligible resolved sectors within one FFT frequency bin (`Delta omega = 6.27691e-4 fs^-1`). The lowest-q sectors are not all independently frequency-resolved by a 10 ps FFT, so the ridge test must not be presented as resolving every dominant low-q component. Their modal energies and analytic frequencies are nonetheless exact on the frozen harmonic surface.

The Holstein spectral peak is at `0.0150646 fs^-1` versus analytic `0.0148369 fs^-1`, also within one frequency bin.

## Direction-resolved vx modes

Because the carrier hop is -x, laboratory +x traveling vx modes are retrograde relative to carrier motion.

At the switch:

- retrograde vx energy: `1.06227978e-3 eV`;
- co-moving vx energy: `1.04726689e-3 eV`;
- nondirectional/special vx energy: `1.39040e-6 eV`;
- 99.934% of vx energy is direction-resolved;
- retrograde fraction of direction-resolved vx: **50.3558%**;
- co-moving fraction: **49.6442%**;
- retrograde energy-weighted harmonic group speed: **1.63887 sites/ps**.

Therefore IP1q does **not** show a globally retrograde-dominated x-Peierls energy reservoir. The two traveling directions carry nearly equal global vx energy. The retrograde component is only about **9.24% of the total autonomous excitation energy**.

## Critical interpretation: total modal energy versus the trailing jx observable

The large `vy` energy fraction and the local trailing x-current observable answer different questions.

The harmonic intermolecular x-current `jx` used in IP1f-IP1o depends directly on the `vx` coordinate/velocity chains. Consequently, the fact that `vy` stores 80.41% of total excitation energy does **not** mean `vy` carries the previously identified trailing x-current pattern.

An exploratory exact decomposition of the saved IP1q trajectory into +x and -x traveling `vx` components resolves the apparent contradiction between the almost 50/50 global direction-resolved vx energy and the strong local trailing pattern:

- the full real-space `jx` current is reconstructed to machine precision from retrograde, co-moving, special and interference terms;
- over the late window (`t - ts >= 1 ps`), the projection of the local B-centered trailing d=1..4 current vector onto the full pattern is approximately:
  - retrograde traveling vx: **94.83%**;
  - co-moving traveling vx: **3.92%**;
  - counterpropagating/interference remainder: **1.24%**;
  - special q sectors: negligible;
- the corresponding squared-norm ratio of the retrograde component is ~93.63% of the full late trailing-pattern norm.

This posthoc result is not promoted as an IP1q preregistered gate. It motivates the next deterministic algebraic checkpoint.

## Scientific closure

IP1q establishes that the autonomous isotropic post-hop memory is a long-wavelength, Peierls-dominated harmonic lattice excitation on the frozen electronic surface.

The correct physical picture is now two-level:

1. **global stored excitation:** overwhelmingly Peierls, with the largest reservoir in transverse `vy` and nearly balanced +/-x traveling `vx` energies;
2. **local trailing x-current pattern:** generated by `vx` and apparently phase-selected so that the local late B-centered observable is dominated by the retrograde traveling component even though the global vx energy is almost direction-balanced.

This distinction prevents a local retrograde wake signature from being misreported as globally retrograde energy transport.

## Guardrails

- `vy` dominance refers to total stored excitation energy, not to the x-directed trailing current.
- The ~50/50 vx traveling-energy split rules out a claim of globally retrograde-dominated vx energy.
- The harmonic group speeds are model diagnostics, not mobility or a calibrated material transport velocity.
- The undamped frozen continuation does not define a material phonon lifetime.
- The 10 ps spectral window does not resolve the very lowest-q frequencies individually.
- No mobility, hopping rate, activation energy or threshold field is inferred.

## Next stage

Proceed to **IP1r: exact traveling-wave attribution of the local trailing x-current pattern**. The next stage should reconstruct the full `jx` field from retrograde (+x), co-moving (-x), nondirectional and cross/interference vx contributions, verify exact current closure, and quantify their contribution to the B-centered d=1..4 trailing pattern as a function of time. This directly tests whether the persistent local wake is genuinely selected by retrograde vx phases despite the nearly balanced global vx energy reservoir.
