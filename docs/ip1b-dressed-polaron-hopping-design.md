# IP1b — dressed-polaron hopping validation

## Motivation

IP1a showed many persistent changes of the dominant electronic population site in the zero-field isotropic model, with a monotonic increase of the nearest-neighbour transition rate from 100 to 500 K.  Those events are promising, but a dominant-site change is not sufficient to demonstrate translation of the full Holstein-Peierls polaron.  In particular, the broad electronic cloud can change its maximum while the lattice distortion remains centered elsewhere.

IP1b therefore asks a stricter mechanistic question:

> When the electronic residence site changes, does the **lattice deformation dressing the charge** translate to the same neighbouring molecular site on a comparable timescale?

No nonzero dressed-hop count is required for numerical PASS.

## Lattice-distortion matched filter

Let the relaxed static polaron define the template.  The physical lattice features are

\[
q_u=\sqrt{K_1}\,u,
\qquad
q_x=\sqrt{K_2}\,[v_x(i+\hat x)-v_x(i)],
\qquad
q_y=\sqrt{K_2}\,[v_y(i+\hat y)-v_y(i)].
\]

Using bond differences for `vx` and `vy` removes their exact uniform gauge modes.  The square-root stiffness factors put all three components in elastic-energy coordinates.

For every periodic translation R of the relaxed template, IP1b evaluates the cross-correlation

\[
C(R)=q_u(t)\cdot T_R q_u^{(0)}
    +q_x(t)\cdot T_R q_x^{(0)}
    +q_y(t)\cdot T_R q_y^{(0)}.
\]

The translation with maximum `C(R)` defines the instantaneous **lattice-distortion center**.  FFT periodic correlation is used, so the cost scales approximately as `N log N` per sampled frame rather than evaluating all `N` translations explicitly.

For an exact translated copy of the static template the normalized amplitude is one.  At finite temperature the amplitude and the gap between the best and second-best translations are diagnostics, not probabilities.

## Independent electronic and lattice trackers

The propagated electronic state is tracked with the already validated persistent dominant-site detector.  The lattice template center is tracked independently with a persistent template-center detector.

To test event-definition sensitivity without rerunning the expensive dynamics, both trackers are evaluated simultaneously at persistence windows

- 20 fs,
- 50 fs,
- 100 fs.

The **50 fs** definition is the primary mechanistic diagnostic; 20 and 100 fs are sensitivity checks.

A lattice template sample is considered usable by the persistence tracker when

- template amplitude >= 0.05, and
- relative best-versus-second template gap >= 0.01.

These permissive thresholds are numerical confidence controls, not physical calibration.  The fraction of usable samples is reported explicitly.

## Dressed-event matching

For each persistence definition, nearest-neighbour electronic and lattice-center transitions are matched only when they have the same source and target sites.  The signed lag is

\[
\tau_{\rm lag}=t_{\rm lattice}-t_{\rm electronic}.
\]

Rather than preregistering a single arbitrary synchronization time, IP1b reports the matched fraction for symmetric lag windows

- 100 fs,
- 250 fs,
- 500 fs,
- 1000 fs.

A positive lag means the lattice distortion follows the electronic site change; a negative lag means the lattice reorganization leads it.

No one lag window is promoted to a material parameter in IP1b.

## Direction-selection test

IP1a found positive transfer bias along the eventual hop direction, but lacked a matched directional null.  For every primary (50 fs) nearest-neighbour electronic event, IP1b evaluates the rank of the future hop bond among

\[
|J_{+x}|,|J_{-x}|,|J_{+y}|,|J_{-y}|
\]

throughout the 100 fs before the transition begins.

The key diagnostic is the fraction of pre-hop samples for which the **future hop direction is the strongest local bond**.  In an isotropic system with no predictive relation the natural directional reference is 1/4.  IP1b reports this quantity and the mean prospective-bond rank, but does not use either as a numerical PASS gate.

## Production screen

- lattice: 20x20 periodic;
- zero external field;
- BAOAB + CF4-Lanczos, `m=6`;
- IDC-BM with `td=180 fs` as the validated one-polaron numerical control, not material calibration;
- `dt=0.2 fs`;
- final time 20 ps;
- burn-in 2 ps;
- diagnostics sampled every 2 fs;
- isotropic `J0y/J0x=1.0` at 100, 300, and 500 K;
- anisotropic reference `J0y/J0x=0.15` at 300 K;
- four independent seed pairs per condition.

This requires 16 trajectories, reducing cost relative to IP1a while retaining low-, room-, and high-temperature endpoints plus the established anisotropic reference.

## Numerical gates

IP1b numerical PASS requires:

1. syntax checks pass;
2. focused tests pass;
3. full pytest suite passes;
4. all static polaron relaxations converge;
5. all 16 requested trajectories complete;
6. exact IDC event counts;
7. ensemble lattice temperatures remain close to their targets;
8. maximum zero-field generalized energy residual < `5e-5 eV`;
9. maximum electronic norm error < `1e-10`;
10. projected zero modes < `1e-12`;
11. static template self-match returns the correct center with amplitude approximately one;
12. finite electronic/lattice persistence and matched-event diagnostics.

There is deliberately **no physical gate requiring a dressed hop**.

## Interpretation logic

- If electronic transitions survive all persistence windows and are accompanied by same-site lattice-template translations, IP1a is upgraded to evidence for dressed-polaron hopping.
- If electronic transitions remain but lattice-center transitions do not follow, IP1a primarily detected electronic redistribution/flicker inside a largely stationary distortion cloud.
- If dressed events appear only above a temperature range, that temperature dependence becomes the basis for IP1c kinetic/activation analysis.
- If the eventual hop direction becomes predictably the strongest local bond shortly before dressed events, the transient-local-anisotropy hypothesis receives direct microscopic support.
- If lattice reorganization systematically leads the electronic transition, the mechanism is naturally interpreted as a thermally prepared structural gateway rather than the charge dragging a rigid distortion after the fact.

No diffusion coefficient, mobility, or activation energy will be claimed in IP1b.
