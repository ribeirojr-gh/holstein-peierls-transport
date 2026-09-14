# IP1m final local validation — 2026-09-14

IP1m is numerically closed from the user-local artifact `ip1m-local-validation/20260914T190240Z` produced on branch `isotropic-polaron-barrier` at commit `f70d62a7838a1ff614813f4ad5891fe994ce1525`.

## Numerical closure

- runner status: PASS;
- failed gates: none;
- pycompile: PASS;
- focused IP1m tests: PASS;
- full suite: **442/442 PASS**;
- 40x40 zero-damping PBC recurrence preflight for 5 ps: PASS;
- static relaxation, requested integration, energy-work balance, electronic norm, projected intermolecular zero modes, finite memory/envelope diagnostics and full current-trajectory serialization: PASS.

The working tree contained only two unrelated untracked local files, `.git-run.sh.swp` and `git-run.sh`; the tracked validation source matched the recorded commit.

The recurrence audit gives a 40x40 stationary-carrier ballistic wrap time of 21.693 ps, far beyond the 5 ps production trajectory.

## Production event

The deterministic isotropic control (`J0y/J0x = 1.0`, `E = +10 mV/A`, T=0, no thermostat, no IDC) contains exactly one persistent nearest-neighbor x relocation:

- 820 -> 819 (-x);
- transition start: 2826 fs;
- persistent acceptance: 2874 fs;
- no second persistent x hop before 5 ps.

The post-hop residence frame used by the preregistered IP1m analysis is therefore 819 -> 818, with carrier motion defined as +s and the trailing direction as -s.

## Preregistered IP1m physical decision

The preregistered physical gates do **not** close.

### Post-hop B-centered d1 -> d2 test

- best allowed lag: 300 fs, exactly at the lower search boundary;
- inferred speed: 3.3333 sites/ps, above the harmonic limit;
- correlation: -0.984791;
- backward packet gate: FAIL.

### d2 late-memory test

- late bins (>1 ps after the hop): 10;
- jointly above the 95th-percentile energy and peak backgrounds: 0;
- latest passing center: none;
- late-memory gate: FAIL;
- sustained-memory gate: FAIL;
- late positive backward energy at d2: 0;
- late d2 positive-energy directionality: -1.0.

Thus IP1m does **not** establish a long-lived retrograde wake that propagates outward through the first two boundaries behind the *post-hop carrier*.

## Spatial structure in the B-centered residence frame

The negative d2 result is not equivalent to absence of a lattice response. The post-hop field is strongly spatially phase-structured:

- d=1: all 20 post-hop 100 fs bins are background-separated; outward current remains positive and grows to a peak near 3.51e-6 eV/fs;
- d=2: no post-hop bin is background-separated; outward current remains negative throughout the analyzed trajectory;
- d=3: six early bins are background-separated, after which the current changes sign;
- d=4: all 20 post-hop bins are background-separated, with a broad positive response that later decays.

The ordered d=1..3 envelope fit is therefore unavailable. This is not the morphology of one simple outward-moving packet centered on the new carrier site.

## Posthoc frame audit

The complete saved IP1m current trajectory permits a direct frame audit without rerunning dynamics.

When the same trajectory is reprojected into the older event-source frame 820 -> 819 used by IP1j/IP1k, the previous isotropic IP1k delay is reproduced exactly:

- lag: 634 fs;
- speed: 1.577287 sites/ps;
- correlation: 0.999996709.

This proves that the difference between IP1k and IP1m is not numerical nondeterminism.

The coordinate mapping explains the result. For a -x hop, source-frame backward d=1 and d=2 correspond to B-centered backward d=2 and d=3, respectively. In the IP1m trajectory:

- B-centered d2 is inward over the complete first 800 fs after the event;
- B-centered d3 is outward over that early window;
- the demeaned d2 -> d3 correlation nevertheless peaks at 634 fs with the same 0.999996709 value.

Therefore the old source-frame isotropic correlation should **not** be promoted to evidence that positive outward lattice energy physically crossed both neighboring backward boundaries. The high normalized correlation tracks a phase relation between two signals with opposite mean flow signs.

This refinement does not reverse the formal IP1k conclusion: the isotropic event was already *not* background-separated under the preregistered IP1k joint criterion. It sharpens the physical interpretation of why the correlation alone is insufficient.

For completeness, B-centered d3 -> d4 also gives a high demeaned correlation (lag 602 fs, speed 1.66113 sites/ps, correlation 0.99999823), while both boundaries carry positive outward current during the early window. This confirms that a spatially propagating component exists farther in the trailing lattice, but it is embedded in a phase-structured pattern rather than a monotonic carrier-centered outward front.

## Scientific conclusion

IP1m provides a useful negative control for the central isotropic-polaron problem:

1. there is no evidence for the preregistered **carrier-centered d1 -> d2 long-lived retrograde wake**;
2. the isotropic hop nevertheless leaves a strong, long-lived and spatially structured lattice-current reorganization behind it;
3. the older source-frame delay correlation is real and reproducible, but correlation alone does not establish outward transported energy across every intervening boundary;
4. the post-hop pattern can reflect a mixture of emitted lattice excitation, coherent spatial phase structure and continuing field-driven response.

The next causal question is therefore whether this phase-structured post-hop lattice current survives when the external electric field is removed without a gauge discontinuity.

## Interpretation guards

- Do not reinterpret the failed IP1m d1 -> d2 packet gate as an absence of all lattice memory.
- Do not use demeaned delay correlation alone as proof of positive outward energy transport.
- Do not call the B-centered pattern a single compact phonon wake.
- The field remains on throughout IP1m, so long-lived post-hop current can include continued field support.
- No mobility, hopping rate, activation energy, threshold field or calibrated phonon lifetime is inferred.

## Next stage

Proceed to IP1n: a gauge-continuous field-release audit. The driven isotropic trajectory should be propagated through the natural hop, then branched after persistent acceptance into (i) the normal field-on continuation and (ii) a continuation with the Peierls phase held fixed so that the electric field is zero while the vector potential remains continuous. Persistence of the trailing lattice-current structure in the zero-power branch would establish stored lattice memory rather than a response requiring continued external work.