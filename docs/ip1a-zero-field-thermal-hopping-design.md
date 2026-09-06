# IP1a — zero-field thermal hopping and transient-anisotropy screening

## Scientific question

IP0a–IP0d showed that the 20x20 isotropic model does not possess an anomalously large static nearest-neighbour translation barrier. The frozen shared-charge barrier is about 12 meV in either lattice direction, while IP0b showed a slower and heavier collective intermolecular reorganization coordinate as isotropy is approached.

IP1a therefore moves from static barriers to dynamics and asks:

> Does a finite-temperature isotropic polaron undergo persistent spontaneous nearest-neighbour hops at zero electric field, and are accepted hops preceded by transient local anisotropy of the instantaneous Peierls-modulated transfer integrals?

This is a screening calculation. It does not pre-register a nonzero hopping rate and does not force an Arrhenius interpretation.

## Dynamics model

The trajectory uses the already validated one-carrier finite-temperature stack:

- 20x20 periodic lattice;
- BAOAB Langevin thermostat;
- projected uniform intermolecular zero modes;
- moving-lattice CF4-Lanczos electronic propagation;
- IDC-BM electronic decoherence with `td = 180 fs`;
- independent lattice and decoherence random-number streams;
- `dt = 0.2 fs`;
- zero electric field.

`td = 180 fs` and the Langevin friction are numerical/model controls inherited from D5; they are not material calibrations.

## Screening conditions

Primary isotropic scan:

`J0y/J0x = 1.0` at `T = 100, 200, 300, 400, 500 K`.

Positive/reference control:

`J0y/J0x = 0.15` at `T = 300 K`.

Each condition uses four independent lattice/decoherence seed pairs. The default trajectory length is 20 ps with 2 ps burn-in. This is intentionally several times the ~6.8 ps frozen-path collective period found for the isotropic IP0b control, while remaining a screening rather than a rare-event production calculation.

## Persistent-hop definition

The propagated electronic density is sampled every 2 fs. The instantaneous maximum-population molecule is **not** automatically treated as a hop.

A candidate residence site is accepted only when:

1. it is the dominant molecular population;
2. the maximum molecular population is at least 0.10;
3. its population exceeds the second-largest population by at least 0.02;
4. it remains the same candidate for 20 fs (10 consecutive samples).

Short source/target flicker near a shared-charge configuration is therefore rejected.

Every accepted residence-site change is classified as:

- `+x`, `-x`, `+y`, or `-y` nearest-neighbour hop under PBC; or
- nonlocal transition.

No physical gate requires any hop to occur.

## IDC causality tag

IDC can instantaneously change the electronic state. Therefore every persistent transition records whether its **first persistent candidate sample** occurred immediately after an IDC event. IP1a reports the fraction of persistent transitions that are IDC-associated.

A large IDC-associated fraction would not invalidate the calculation, but it would trigger a later coherent/IDC-scheme sensitivity study before interpreting a hopping rate as material physics.

## Transient local anisotropy

At the currently accepted residence site, the four instantaneous transfer magnitudes are

`|J(+x)|, |J(-x)|, |J(+y)|, |J(-y)|`,

including the current Peierls lattice distortion.

The local anisotropy diagnostic is

`A_local = (J_max - J_min)/(J_max + J_min)`.

For each accepted nearest-neighbour hop, IP1a also evaluates the prospective hop bond relative to the other three bonds,

`B_dir = (J_hop - mean(J_other))/(J_hop + mean(J_other))`.

The preceding 100 fs history is retained. For every accepted hop the output records the pre-hop mean/maximum `A_local` and the mean/maximum `B_dir` in the direction that is subsequently selected.

The baseline `A_local` distribution is also sampled throughout the post-burn trajectory. A higher pre-hop anisotropy than the trajectory baseline would support the hypothesis of transient symmetry breaking, but this trend is **not** a numerical PASS criterion.

## Additional diagnostics

Each trajectory records:

- IPR and participation-number statistics;
- maximum molecular population;
- lattice kinetic temperature;
- persistent residence times;
- nearest-neighbour and nonlocal transition counts;
- direction-resolved hop counts;
- net accepted-site displacement in lattice units (diagnostic only);
- instantaneous IDC dominant-site changes, whether or not persistent;
- generalized energy-balance residual;
- electronic norm error;
- projected zero-mode residual.

The accepted-site displacement is not substituted for TP1 current integration and is not used to claim mobility.

## Numerical gates

IP1a numerical PASS requires only:

1. all static initial polarons converge;
2. all requested trajectories complete;
3. each trajectory has the exact expected IDC event count;
4. ensemble lattice temperatures remain within a broad 30% window around their target temperatures;
5. maximum generalized zero-field energy-balance residual < `1e-4 eV`;
6. maximum electronic norm error < `1e-10`;
7. maximum projected zero-mode magnitude < `1e-12`;
8. all hop/an\-isotropy diagnostics are finite and internally consistent.

There is deliberately **no** PASS requirement on hop count, hopping-rate monotonicity, anisotropic-vs-isotropic ordering, IDC association, or transient-anisotropy enhancement.

## Decision after IP1a

If the isotropic scan produces enough persistent hops at several temperatures, IP1b will increase the ensemble/trajectory length and estimate residence-time distributions and temperature-dependent rates with uncertainty. An Arrhenius fit will only be attempted if the data support an activated regime.

If hops remain rare or absent, the result will define a lower bound on residence time over the simulated window and IP1b will extend the time horizon and/or temperature range rather than inventing a rate.

If most accepted hops are IDC-associated, the next gate will explicitly compare coherent, IDC-BM and alternative IDC controls before assigning a microscopic interpretation.
