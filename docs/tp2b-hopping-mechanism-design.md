# TP2b — hopping-mechanism validation and field-onset characterization

## Motivation

TP2a validated the periodic current, unwrapped displacement, field work, and paired +/-E bookkeeping, but a single linear-response fit across 0.5, 1.0, and 2.0 mV/A was statistically and physically ambiguous.

The underlying Holstein-Peierls physics indicates that this ambiguity is expected. Charge transport is mediated by site-to-site hopping of a lattice-dressed polaron. In increasingly isotropic two-dimensional systems the carrier can become effectively immobile because motion requires the intra-site distortion and the intermolecular x/y deformation pattern to reorganize and travel with the charge. The anisotropic legacy control `J0x = 100 meV`, `J0y = 15 meV` is instead a transport-favorable hopping regime.

TP2b therefore characterizes the hopping mechanism before any further mobility inference.

## Frozen validated dynamics

- one-polaron Holstein-Peierls dynamics
- 20 x 20 periodic lattice
- 300 K
- BAOAB lattice bath
- `gamma_u = gamma_v = 0.01 fs^-1`
- field-aware IDC-BM
- `t_d = 180 fs` as a numerical control only
- `dt = 0.2 fs`
- CF4-Lanczos, Krylov dimension 6
- projected intermolecular zero modes
- electron-like Peierls field convention

## Hopping observables

The primary mechanistic observables are discrete and site resolved.

1. **Normalized site population** `rho_i = |psi_i|^2 / <psi|psi>`.
2. **Dominant site** and its population.
3. **IPR** `sum_i rho_i^2` and participation number `1/IPR`.
4. **Persistent dominant-site transitions** with a short persistence filter to reject one-step numerical flicker.
5. **Periodic nearest-neighbour hop vector**: `(+/-1,0)` or `(0,+/-1)` with boundary wrapping handled exactly.
6. **Hop counts and directional rates** `k_{+x}, k_{-x}, k_{+y}, k_{-y}`.
7. **Residence times** between confirmed hops.
8. **Event-based unwrapped displacement**, compared with the independently validated current-integrated displacement.
9. **Transition localization signature**: IPR/dominant population immediately before, during, and after a hop, to detect compression -> sharing -> recompression.
10. **Lattice-following diagnostics**: local intra-site deformation and neighbouring x/y bond distortions around the confirmed polaron site before and after a hop.

A continuous current remains a valid exact quantum observable, but it is not interpreted as evidence of band-like transport. In the hopping regime it is an independent displacement/work observable against which discrete hop bookkeeping can be checked.

## Two physical controls

### A. Anisotropic hopping control

Use the current legacy values:

- `J0x = 100 meV`
- `J0y = 15 meV`
- isotropic intermolecular e-ph coupling in x/y

This control is expected to permit hopping under sufficiently strong field.

### B. Isotropic transport-suppression control

Use an explicitly documented isotropic parameter set with `J0x = J0y` and matching x/y e-ph coupling. The exact isotropic transfer-integral magnitude must be selected from the validated literature/control set rather than invented.

This is a **negative transport control**: a near-zero net displacement or absence of confirmed hops is an acceptable and potentially expected physical result. TP2b must never fail simply because transport is absent in this control.

## Field-onset characterization

Do not fit one linear mobility across a field interval before the hopping-onset regime is known.

For the anisotropic control, scan a field grid that resolves the weak-motion/onset region. The literature control suggests that fields around the 1-2 mV/A region deserve specific resolution. The exact final grid will be frozen before execution.

For each field, report:

- hop count per trajectory;
- total hop rate;
- directional hop-rate asymmetry;
- net event displacement;
- current-integrated displacement;
- IPR/localization statistics;
- lattice-following diagnostics;
- temperature and complete energy closure.

The purpose is to identify whether the response is:

- no-hopping/locked;
- rare activated hopping;
- biased hopping;
- or a regime where a linear drift approximation is actually defensible.

## Numerical gates

All prior TP0/TP1 numerical gates remain mandatory:

- static initial state converged;
- temperature controlled;
- complete energy balance;
- electronic norm;
- zero modes;
- IDC event counts;
- field work vs `-E Delta x`;
- instantaneous field-power/velocity identity.

Additional hopping-observable gates will include:

- exact periodic nearest-neighbour mapping on synthetic traces;
- correct boundary-crossing unwrapping;
- global-phase invariance of site localization/IPR;
- no false hop for a stationary localized state;
- persistence-filter rejection of one-sample site flicker;
- correct +/-x and +/-y hop counts for constructed traces.

## Interpretation rules

- **No hopping in an isotropic control is not a failure.** It can be the expected high-polaron-mass result.
- **A non-zero continuous current without persistent site relocation is not by itself classified as charge transport.** It may reflect intra-polaron oscillatory/breather-like motion.
- **Mobility is not reported until the field regime is identified.** If the low-field response is locked or activated/nonlinear, report field-dependent hop rates/drift instead of forcing a linear mobility.
- If a genuine weak-field linear hopping regime exists, drift mobility may later be compared with an independent zero-field diffusion/Einstein estimate.

## Next stage after TP2b

TP2c will be chosen from the TP2b mechanism results:

1. zero-field long-time diffusion and Einstein consistency if sufficient unbiased hopping occurs;
2. field-biased hopping-rate analysis if transport is activated/nonlinear;
3. conventional paired-field mobility only if a real linear regime is demonstrated.

No material-calibrated mobility claim, pair transport, CPU threading, or GPU claim is part of TP2b.
