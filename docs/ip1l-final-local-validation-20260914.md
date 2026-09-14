# IP1l final local validation — inter-hop wake memory and renewed radiation

Date: 2026-09-14
Branch: `isotropic-polaron-barrier`
Validated commit: `247c09cda453082456b0a7f4dff756d9d31932a3`
Local artifact: `ip1l-local-validation/20260914T184221Z`

## Numerical closure

IP1l is numerically closed.

- runner status: PASS;
- failed gates: none;
- pycompile: PASS;
- focused tests: PASS;
- full suite: **436/436 PASS**;
- 40x40 zero-damping PBC recurrence preflight for the 5 ps trajectory: PASS;
- static relaxation, complete requested integration, size-aware field-work balance, electronic norm, projected intermolecular zero modes, exact two-event chain, finite memory diagnostics and full-current serialization: PASS.

The local working tree contained one unrelated untracked file (`git-run.sh`). The tracked source matched the validated commit.

## Production trajectory

- cell: 40x40 PBC;
- `J0y/J0x = 0.15`;
- T = 0 K;
- field: +10 mV/A along x;
- no thermostat;
- no IDC;
- dt = 0.2 fs;
- final time = 5 ps;
- harmonic intermolecular x-current sampled every 2 fs;
- CF4-Lanczos with Krylov dimension 6.

The trajectory contains exactly two persistent -x nearest-neighbor relocations:

1. 820 -> 819 at 2496 fs;
2. 819 -> 818 at 4108 fs.

The inter-event separation is 1612 fs. TP1 displacement over the full run is -7.15776 A.

The maximum field-work residual is `3.95905e-6 eV`, well below the size-aware 40x40 tolerance of `2.0e-4 eV`. The maximum norm error is `5.31e-12` and the maximum projected zero-mode mean is `1.66e-17`.

## First-hop backward packet

In the fixed B-residence frame, the first natural hop launches a qualified backward d1->d2 packet:

- lag: 608 fs;
- packet speed: 1.64474 sites/ps;
- normalized correlation: 0.925366;
- internal-lag gate: PASS;
- harmonic-speed gate: PASS.

This independently reproduces the natural retrograde packet identified in IP1j with the refined IP1l baseline/frame convention.

## Inter-hop lattice memory

Relative to the pre-first-hop field-only reference, the backward d=2 signal is strongly non-monotonic. It is positive immediately after the first event, reverses sign around roughly 0.45-0.75 ps after the event, and then re-emerges strongly from about 0.85 ps onward.

The preregistered late-memory gate closes:

- five late 100-fs bins have centers more than 1 ps after the first event;
- all five lie above the 95th percentile of the pre-first field-only background in both backward positive energy and backward peak flux;
- late passing fraction: 1.000;
- latest jointly passing bin center: 3946 fs, i.e. 1450 fs after event 1;
- late inter-hop backward positive energy: `9.89552e-4 eV`;
- corresponding positive forward energy in the same late interval: 0 within the adopted outward-flux convention;
- late directionality: +1.0.

This is not a periodic-boundary wrap artifact. The zero-damping 40x40 ballistic wrap time is 21.693 ps, more than four times the complete simulated duration and far beyond the observed 0.85-1.45 ps re-emergence interval.

The non-monotonic time structure must nevertheless be treated carefully: IP1l demonstrates long-lived coherent lattice-memory/current relative to the field-only background, but does not yet establish that one compact packet remains continuously attached to the carrier throughout the residence. Oscillatory lattice-memory/ringing and propagation through the local frame can contribute.

## Incremental second-hop radiation

Using the local pre-second-hop wake state (3408-4008 fs) as the baseline, event 2 launches a renewed backward pulse:

- backward positive excess energy: `7.87059e-4 eV`;
- backward peak excess flux: `3.06649e-6 eV/fs`;
- peak percentile versus local pre-second 100-fs fluctuations: 100.0;
- peak / local-pre maximum: 4.9793;
- backward d1->d2 lag: 612 fs;
- packet speed: 1.63399 sites/ps;
- normalized correlation: 0.999760;
- packet gate: PASS;
- incremental second-hop gate: PASS.

The total second-event radiation remains strongly forward dominated: forward positive excess energy is `1.85763e-2 eV`, giving d=2 positive-energy directionality `-0.918706`. Thus renewed retrograde radiation is a real but subdominant component.

## Physical conclusion

Both preregistered IP1l physical gates pass. The current working mechanism is therefore:

**history-dependent hopping with a long-lived retrograde lattice-memory component plus renewed retrograde radiation at the subsequent carrier relocation.**

This strengthens the mechanistic interpretation developed from IP1h-IP1k without changing the formally negative two-event IP1k background-separation result.

## Interpretation boundaries

- IP1k remains formally non-replicated under its original joint percentile rule.
- IP1l demonstrates memory and renewed radiation, not a calibrated phonon lifetime.
- The late signal is non-monotonic and should not yet be described as a single continuously attached wake packet.
- d1->d2 speeds are packet-propagation diagnostics, not unique normal-mode group velocities.
- The 10 mV/A field is a protocol control, not a material threshold field.
- No mobility, hopping rate, activation energy or transport coefficient is inferred.

## Next stage

The highest-value next control is the natural **isotropic single-hop** trajectory already identified by IP1i/IP1j at 10 mV/A. It provides a clean test of long-lived wake memory without a second persistent hop inside 5 ps. This directly addresses the main isotropic-polaron question and removes the most important ambiguity in the anisotropic late-memory trace: contamination by an approaching second relocation.

Proceed to IP1m: isotropic single-hop wake persistence and spatial-envelope control.