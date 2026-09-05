# D6e pair IDC screening protocol

## Motivation

D6d found systematic electronic overheating under coherent finite-temperature pair Ehrenfest dynamics in both physical sectors while the D6c lattice bath remained correctly thermalized.  D6e therefore screens explicit instantaneous-decoherence controls without assuming that the one-polaron D5 selection transfers to pair dynamics.

## Physical sectors

- **Bipolaron:** collapse is restricted to the symmetric spatial singlet sector, dimension `N(N+1)/2`. Antisymmetric spatial eigenstates are never admitted into the stochastic collapse distribution.
- **Exciton:** the electron and hole are distinguishable and the full ordered `N^2` pair sector is used. This direct e-h model is spin-blind and is not labeled singlet/triplet.

## IDC schemes

D6e reuses the already validated D5 definitions:

- `DP`: collapse probabilities equal the pre-collapse adiabatic populations;
- `BM`: `p_mu exp(-beta E_mu)` after stable energy shifting;
- `MA`: uphill states above the pre-collapse mean energy receive `exp[-beta(E_mu-Ebar)]` suppression.

These remain stochastic model extensions, not exact consequences of Ehrenfest dynamics.

## Energy accounting

An IDC event holds lattice coordinates and velocities fixed.  Its energy jump

`Q_e = E_selected - E_before`

is recorded explicitly.  The zero-field finite-temperature balance is

`Delta E_matter ~= Q_lattice_bath + Q_electronic_environment`.

No lattice-velocity rescaling or hidden post-collapse repair is applied.

## Screening control

- lattice: `4x4`;
- bath: 300 K;
- `gamma_u = gamma_v = 0.01 fs^-1`;
- `dt = 0.2 fs`;
- total time: 4 ps;
- burn-in: 1 ps;
- projected intermolecular zero modes;
- CF4-Lanczos `m=8`;
- schemes: DP, BM, MA;
- common numerical-control decoherence interval: `t_d = 100 fs`;
- four lattice seeds per sector and scheme;
- independent RNG streams for lattice and IDC collapse.

The 100 fs interval is deliberately a **numerical control**, not a material parameter and not inherited from the one-polaron D5 production closure.

## Metrics

Pre-collapse metrics after burn-in:

- mean, early and late heating coordinate;
- heating slope in `ps^-1`;
- TV distance to instantaneous canonical and uniform references;
- absolute ground-manifold population mismatch;
- effective inverse-temperature ratio as a secondary diagnostic;
- expected and realized post-collapse heating coordinates.

Numerical controls:

- lattice temperature;
- generalized energy balance;
- electronic norm;
- bipolaron exchange/RDM or exciton RDM constraints;
- accumulated electronic-environment exchange rate.

No weighted scalar selection score is used.

## Decision logic

D6e is a first screening gate only.  A scheme is not promoted merely because the runner passes.  Physical comparison emphasizes canonical proximity, removal of positive heating drift, ground-manifold agreement, stable lattice temperature, and explicit energy accounting.

If BM and MA both perform acceptably, or their ranking depends materially on the common 100 fs interval, D6f must perform an interval-sensitivity sweep before selection.  DP is retained as the phase-destruction-only control.  No field or mobility calculation is allowed in D6e.