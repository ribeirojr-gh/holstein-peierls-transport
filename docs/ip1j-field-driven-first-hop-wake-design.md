# IP1j — first natural field-driven hop wake

## Purpose

IP1j asks a narrower question than IP1f-IP1h: during a self-consistent field-driven carrier relocation, does a propagating intermolecular lattice-energy branch move opposite to the carrier direction?

IP1i selected the 40x40, T=0, 10 mV/A anisotropic control as a moving protocol and retained the isotropic 10 mV/A case as a near-pinned comparison. Both have at least one persistent -x nearest-neighbor event within 5 ps.

## Protocol

- cell: 40x40 PBC;
- ratios: J0y/J0x = 1.0 and 0.15;
- field: +10 mV/A along x;
- carrier response expected under the established electron-like convention: -x;
- T=0, no thermostat, no IDC;
- deterministic D3 coupled field dynamics;
- dt = 0.2 fs;
- final time = 5 ps;
- harmonic intermolecular current sampled every 2 fs;
- energy-work balance sampled every 10 fs;
- persistent electronic residence tracker: 50 fs.

The current field itself is stored only transiently. The artifact retains the event-aligned, baseline-subtracted post-event longitudinal profiles needed for audit and reanalysis.

## Event isolation

Only the first persistent NN x event is used for the primary wake diagnostic. Its transition-start time defines t=0 for the wake analysis.

The pre-event baseline is -500 to -100 fs. The post-event interval is capped at 1500 fs and is also truncated 50 fs before any second persistent x event. This keeps the anisotropic first-event analysis inside the 1612 fs IP1i inter-event interval.

Because a ~1.6-1.7 site/ps lattice packet travels only ~2.4-2.6 sites in 1.5 ps, the primary natural-event directionality is evaluated at d=2. Longer-range d=4-6 claims are deliberately deferred.

## Observable

The first hop source->target is rotated so the carrier direction is +s even when the laboratory event is -x. Harmonic Peierls energy current is projected onto this event axis and summed only within |p| <= 3 sites.

At a fixed boundary d:

- backward outward current is -j(s=-d,t);
- forward outward current is +j(s=+d,t).

The pre-event mean current profile is subtracted before post-event integration. Positive outward energy directionality is

D(d) = [E_back^+(d)-E_front^+(d)]/[E_back^+(d)+E_front^+(d)].

A positive D means the backward branch carries more positive outward energy; D<0 means the forward branch dominates.

Packet propagation from d=1 to d=2 is diagnosed by normalized time-delay correlation over 300-1000 fs. A branch is called validated only if correlation >=0.80 and inferred packet speed does not exceed 1.05 times the harmonic maximum group velocity.

If the backward branch passes this propagation gate, then by the event-aligned definition it directly realizes v_phonon dot v_carrier < 0 for the observed first carrier relocation. This establishes existence of a retrograde branch, not necessarily dominance.

## Interpretation guards

- the electric field remains on during the wake window;
- pre-event subtraction does not uniquely separate bound distortion from free phonons;
- d1->d2 correlation speed is a packet speed, not a unique normal-mode group velocity;
- 10 mV/A is a numerical protocol control, not a material-calibrated threshold;
- no mobility, hopping rate, activation energy or phonon lifetime is inferred;
- the isotropic comparison is not transport-qualified under the IP1i criterion, even though it has one persistent event.
