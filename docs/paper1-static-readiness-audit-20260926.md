# Paper 1 static extended Holstein–Peierls model — readiness audit (2026-09-26)

## Target paper

The first publication after the T=0 one-polaron code freeze is the **static extended Holstein–Peierls model paper**. Its intended scientific content is:

- one-polaron stationary reference;
- explicit electron-electron interaction parameters `U` and `V`;
- correlated singlet bipolarons;
- distinguishable electron-hole excitons as a direct-Coulomb reference sector;
- spin-adapted open-shell singlet and triplet neutral excitations;
- stationary lattice relaxation only; no dynamics;
- RPROP-based structural results wherever RPROP is the declared production optimizer.

The paper must clearly distinguish the physical Hilbert spaces. A spin-blind electron-hole pair is not renamed a singlet or triplet, and the spin-adapted open-shell model is not silently treated as identical to the distinguishable-pair exciton model.

## Current code/readiness matrix

| Sector | Physics implemented | Structural optimizer | Numerical status | Paper-1 readiness |
| --- | --- | --- | --- | --- |
| One polaron | Holstein + Peierls | legacy-compatible RPROP | validated | **Ready** |
| Correlated bipolaron | singlet two-equal-charge state; onsite U; nearest-neighbour V; screened long-range/shell-resolved V | two-particle RPROP | strict 20x20/40x40 validations, analytic/FD gates | **Ready** |
| Reference electron-hole exciton | distinguishable e-h pair; direct attraction; separate e/h parameters | exciton RPROP | strict 6x6/10x10/20x20 validation, gradients/branches | **Ready as spin-blind reference only** |
| Spin-adapted singlet/triplet | neutral-referenced open-shell orbital functional with exchange and spin purity controls | harmonic-preconditioned Newton/Armijo in promoted benchmark | all 4x4 gapped-control branches converge; exact tiny-system and spin gates passed | **Framework ready; optimizer/method bridge still needed for Paper 1 RPROP claim** |
| Generic spin-adapted RPROP path | neutral-referenced spin-adapted excitation | component-wise non-backtracking RPROP exists in `excitation_reference.py` | not the promoted canonical gapped benchmark; lacks the robust recovery/gap path used by the accepted S0 benchmark | **Not yet production-ready** |

## What is already publishable

### Polaron

The static one-polaron solver and the Holstein/Peierls structural relaxation are established. Paper 1 should use this as the one-carrier reference and define all later binding/relaxation conventions consistently against it.

### Bipolaron U/V sector

The static singlet bipolaron sector is the most mature new extension.

Validated content includes:
- exact noninteracting and atomic Holstein-Hubbard limits;
- onsite Hubbard `U`;
- nearest-neighbour `V1`;
- screened minimum-image long-range repulsion with explicit `a_x,a_y,epsilon_r`;
- shell-resolved short-range replacements that avoid double counting;
- finite-difference structural gradients;
- strict 40x40 stationary convergence;
- finite-size binding/dissociation controls;
- isotropic and anisotropic stationary phase/topology changes;
- controlled linear-Peierls diagnostics.

This is sufficient for Paper 1 provided the final figures are generated from a frozen, coherent parameter campaign rather than assembled from heterogeneous historical exploratory scans.

### Spin-blind electron-hole reference

The distinguishable e-h solver is numerically mature and can supply:
- Frenkel/local versus CT/separated topology;
- direct Coulomb binding;
- electron/hole RDMs and densities;
- lattice relaxation;
- binding and self-trapping conventions.

It must be labeled explicitly as a **spin-blind direct-Coulomb/reference exciton sector**. Its energies cannot be reported as singlet/triplet energies.

### Spin-adapted open-shell sector

The stationary spin-adapted framework already has:
- explicit singlet/triplet state definitions;
- spin-purity gates;
- exchange-driven tiny-system splitting;
- exact Fock-space controls;
- state-specific orbital optimization;
- neutral-reference excitation-density forces;
- finite-difference derivative validation;
- fully coupled 4x4 singlet/triplet lattice relaxation on a gapped isotropic control;
- six converged spin/seed branches.

For the current 4x4 control, the promoted lowest converged energies give `E_S-E_T = +1.231314 meV`. This is a regression/control result only, not a material prediction.

## Critical method issue for the planned paper

The currently promoted spin-adapted S0 relaxation uses an exact harmonic lattice preconditioner plus Armijo line search. It does **not** use the same component-wise RPROP algorithm as the polaron, bipolaron and reference-exciton production calculations.

A generic spin-adapted RPROP implementation exists, but it is not yet the robust gapped/recovery-aware canonical S0 path. Therefore Paper 1 must not currently state that all MCHF/spin-adapted static results were obtained with RPROP.

The clean solution is to validate a **gapped, recovery-aware RPROP structural option** for the spin-adapted control and compare its stationary states against the already accepted harmonic-preconditioned reference. If both converge to the same stationary basin/energy within preregistered tolerances, RPROP results can be used consistently in the static paper while the preconditioned solver remains an independent reference.

## Terminology guard

The present spin-adapted implementation is a fixed-coefficient open-shell/multiconfigurational orbital framework inspired by the general open-shell/MCTDHF formulation. Paper text should use precise terms such as:

- “spin-adapted open-shell multiconfigurational model”;
- “state-specific orbital optimization”;
- “fixed-coefficient spin-adapted manifold”.

Use the broad label **MCHF** only where the implemented functional and variational degrees of freedom match the claimed MCHF definition. Do not imply a general configuration-interaction expansion with freely optimized time-dependent coefficients if it is not present.

## Paper-1 production gates

Before figure production, complete the following in order:

1. **S1 — spin-adapted RPROP bridge**
   - implement RPROP on the same checkerboard-gapped canonical S0 control;
   - retain electronic warm-start recovery;
   - require strict electronic and structural convergence;
   - compare stationary energies, gradients, spin splitting and lattice fields with the accepted preconditioned reference.

2. **S2 — static-sector unified regression**
   - run one GitHub Actions workflow containing representative strict polaron, bipolaron, spin-blind exciton and spin-adapted S/T stationary controls;
   - freeze package/commit provenance and optimizer labels;
   - fail if any sector silently changes its binding/sign/normalization convention.

3. **S3 — Paper-1 parameter campaign**
   - choose a compact generic model grid for U/V and coupling strength;
   - use the same declared lattice geometry and screening convention within each compared phase map;
   - run multiple branch seeds;
   - retain all converged minima and classify by observables, not seed names;
   - apply finite-size and linear-Peierls filters before promoting boundaries.

4. **S4 — paper data freeze**
   - immutable JSON/NPZ/CSV data products;
   - figure-generation scripts tied to one commit and one parameter manifest;
   - explicit distinction between validation controls and physical/model trends.

## Recommended Paper-1 figure architecture

1. Hamiltonian/model schematic showing P, BP, direct e-h reference, and spin-adapted S/T sectors.
2. Static one-polaron distortion/localization reference.
3. Bipolaron U-V structural phase map with onsite/axial/diagonal/separated states.
4. Representative RPROP-relaxed lattice and pair-density maps for BP states.
5. Spin-blind exciton Frenkel/CT topology and binding/self-trapping controls.
6. Spin-adapted singlet/triplet stationary densities/distortions plus `E_S-E_T` on the validated control.
7. Optional summary diagram connecting direct Coulomb reference excitons to the spin-adapted exchange-resolved sector.

## Immediate next stage

Proceed with **S1**, a prospective numerical validation of the spin-adapted RPROP bridge. No Paper-1 production parameter sweep should start until S1 closes.
