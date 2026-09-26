# T=0 one-polaron validated baseline — freeze record (2026-09-26)

## Scope

This record freezes the validated **single-polaron zero-temperature dynamics baseline** after completion of IP1 and the positive IP2b deterministic ensemble sensitivity study.

This baseline is narrower than the repository as a whole. The repository also contains static bipolaron, exciton and spin-adapted modules with their own validation records. This freeze does **not** certify those sectors for the same dynamical claims.

## Frozen scientific status

The following one-polaron T=0 numerical/dynamical foundations are closed:

- static isotropic reference and event-generation dynamics;
- electric-field Peierls phase implementation and gauge-continuous held-phase release;
- energy/work balance, electronic norm and lattice zero-mode diagnostics;
- finite-size PBC recurrence preflight;
- persistent post-hop lattice memory (IP1o);
- modal decomposition and long-wavelength Peierls character (IP1q);
- exact local trailing x-current attribution (IP1r);
- prospectively negative single-trajectory commitment controls (IP1s, IP1t, IP1u);
- deterministic 32-state Peierls preparation construction (IP2a-1);
- refined production timestep selection, dt=0.10 fs (IP2a-3);
- full refined calibration and production preparation energy selection, 1.0e-5 eV (IP2a-4);
- complete 32-member paired counterfactual ensemble (IP2b).

## Locked production numerical settings

For the validated one-polaron T=0 production path:

- lattice: 40x40 periodic square lattice;
- isotropic hopping for the validated causal ensemble, J0y/J0x = 1;
- temperature: 0 K;
- no thermostat and no IDC;
- time step: **0.10 fs**;
- electronic propagator: CF4-Lanczos;
- Krylov dimension: **6**;
- field-driven event generation: +10 mV/A along x;
- deterministic preparation energy for IP2 ensemble work: **1.0e-5 eV**;
- event sampling: 2 fs;
- energy diagnostics: 10 fs;
- branch release: gauge-continuous held Peierls phase with zero phase rate/external power;
- PBC recurrence diagnostic: stationary harmonic full-wrap ~21.693 ps for the 40x40, a=3 A control.

These settings are validated for the present model/protocol. They are not universal defaults for every future carrier sector.

## Final IP2b result retained with scope

IP2b passed all five preregistered finite-design criteria on 32/32 valid deterministic preparations.

The supported conclusion is that energy-preserving reversal of the non-special x-Peierls traveling phase at a matched post-hop state reproducibly changes the subsequent coupled electron-lattice electronic trajectory across the complete tested design grid.

The baseline does **not** infer:
- hopping probability or rate;
- mobility;
- diffusion coefficient;
- activation energy;
- threshold field;
- material phonon lifetime;
- population-level confidence intervals or p-values from the deterministic grid.

IP1s-IP1u remain formally negative under their own preregistered discrete-event commitment criteria.

## Reproducibility records

Key closure documents:

- `docs/ip1o-final-local-validation-20260914.md`;
- `docs/ip1q-final-local-validation-20260914.md`;
- `docs/ip1r-final-local-validation-20260914.md`;
- `docs/ip1s-final-local-validation-20260914.md`;
- `docs/ip1t-final-local-validation-20260923.md`;
- `docs/ip1u-final-local-validation-20260923.md`;
- `docs/ip1o-ip1u-scientific-synthesis-20260923.md`;
- `docs/ip2a1-final-local-validation-20260923.md`;
- `docs/ip2a2-final-local-validation-20260923.md`;
- `docs/ip2a3-final-local-validation-20260923.md`;
- `docs/ip2a4-final-github-validation-20260925.md`;
- `docs/ip2b-final-github-validation-20260925.md`.

The final IP2b GitHub Actions production run was `36198826221`, validating commit `cddfb1d45f7b71f2bc4911c8304b75f7fee48ef5`. The scientific closure commit was `07a8b7a1588e04ebe5078192c817031ce4be3764`.

## Freeze mechanism

A dedicated immutable-working-reference branch is created from the commit containing this record:

`t0-polaron-baseline-20260926`

Future development continues on `isotropic-polaron-barrier` or successor branches. Do not rewrite the frozen branch to include later temperature, pair-dynamics, exciton or bilayer changes.

No merge to `main` is implied by this freeze.

## Relationship to software version

The package metadata currently remains `0.7.0a1`, whose historical release note describes the earlier static-exciton milestone. This T=0 freeze is therefore a **scientific baseline branch**, not a claim that the package semantic version has already been bumped or that a new public software release has been cut.

A later release-version decision should be made only after the static Paper 1 readiness audit and any chosen code consolidation are complete.
