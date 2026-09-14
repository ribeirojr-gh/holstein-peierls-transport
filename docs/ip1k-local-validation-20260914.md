# IP1k local validation — natural-hop wake replication and matched background

Date: 2026-09-14
Branch: `isotropic-polaron-barrier`
Validated commit: `e2adf84f179347d489f1fad2a42c089f0546a72e`
Local artifact: `ip1k-local-validation/20260914T175210Z`

## Numerical closure

IP1k is numerically closed.

- runner status: PASS;
- failed gates: none;
- pycompile: PASS;
- focused tests: PASS;
- full suite: **431/431 PASS**;
- 40x40 zero-damping PBC recurrence preflight for the 5 ps trajectory: PASS;
- static relaxation, requested time integration, energy-work balance, electronic norm, projected intermolecular zero modes, finite event/background metrics and profile serialization: PASS.

The local working tree reported one unrelated untracked file (`git-run.sh`). The validated tracked source itself matched commit `e2adf84f179347d489f1fad2a42c089f0546a72e`.

## Production protocol

- cell: 40x40 PBC;
- T = 0 K;
- field: +10 mV/A along x;
- deterministic coupled D3 dynamics;
- no thermostat;
- no IDC;
- dt = 0.2 fs;
- final time = 5 ps;
- harmonic intermolecular current sampled every 2 fs;
- persistent nearest-neighbor residence criterion: 50 fs;
- backward/forward d1->d2 delay search: 300-700 fs;
- residence-matched pseudo-events every 20 fs;
- pseudo windows restricted to the same persistent-x residence interval as each real event source.

The event direction defines +s. A backward packet is therefore retrograde relative to carrier motion.

## Main physical result

The preregistered IP1k replication gate **does not close**. The anisotropic trajectory contains two complete natural x-hops, but only the first is background-separated under the joint packet + integrated-energy + peak-flux rule.

### Isotropic control, J0y/J0x = 1.0

One complete event occurs at 2826 fs (820 -> 819, -x).

- backward lag: 634 fs;
- packet speed: 1.57729 sites/ps;
- delay correlation: 0.9999967;
- packet gate: PASS;
- backward positive energy percentile: 100.0;
- backward peak-flux percentile: 89.55;
- background-separated: NO.

Thus the isotropic event contains a very coherent backward-propagating component and unusually large integrated backward energy, but its peak amplitude does not exceed the preregistered 95th-percentile background threshold. This must not be translated into absence of a retrograde branch.

### Anisotropic control, J0y/J0x = 0.15 — event 1

First event at 2496 fs (820 -> 819, -x).

- backward lag: 562 fs;
- packet speed: 1.77936 sites/ps;
- delay correlation: 0.9964545;
- packet gate: PASS;
- backward positive energy: 3.94319e-4 eV;
- backward energy percentile: 100.0;
- backward peak flux: 8.17738e-7 eV/fs;
- backward peak percentile: 100.0;
- background-separated: YES.

This is the strongest natural event-associated retrograde result so far: both the coherent packet diagnostic and both amplitude observables lie above every residence-matched pseudo-event control.

### Anisotropic control, J0y/J0x = 0.15 — event 2

Second event at 4108 fs (819 -> 818, -x).

- backward lag: 612 fs;
- packet speed: 1.63399 sites/ps;
- delay correlation: 0.9997598;
- packet gate: PASS;
- backward positive energy: 7.87059e-4 eV;
- backward energy percentile: 0.0;
- backward peak flux: 3.06649e-6 eV/fs;
- backward peak percentile: 100.0;
- background-separated: NO.

The apparent failure is physically structured rather than a loss of backward propagation. All six residence-matched pseudo-events between 3200 and 3300 fs already contain a coherent backward packet, with backward positive energy between 1.021e-3 and 1.146e-3 eV and delay correlations between 0.99979 and 0.99999. Their d2 directionality is +1.0. These controls occur after the first natural hop and before the second.

Therefore the second event is born on top of a substantial **pre-existing retrograde lattice-energy wake** left during the inter-hop residence. Its integrated backward energy is lower than this inherited background, while its instantaneous backward peak is higher than every matched pseudo-event by roughly 28% relative to the largest control peak. This is consistent with a new sharp backward impulse superposed on a long-lived residual wake.

## Interpretation

IP1k must remain formally classified as **no replicated background-separated evidence under the preregistered joint rule**: two complete anisotropic events exist, but only one passes both amplitude percentiles.

However, the reason the second event fails the energy percentile exposes a new physical feature that is more informative than a simple null result: the inter-hop state is not a neutral background. It carries a coherent retrograde lattice-energy current from the preceding relocation for at least several hundred femtoseconds and into the residence interval before the next hop.

This observation connects three previous results:

1. IP1h showed long-range retrograde radiation after a controlled relocation in the anisotropic system;
2. IP1j established a backward branch during the first natural field-driven hop;
3. IP1k shows that after the first natural hop this backward branch remains present during the subsequent residence and contaminates the nominal background of the second hop.

The physically appropriate next question is therefore no longer simple event-vs-background replication. It is whether the first-hop wake persists quantitatively across the complete inter-hop interval and whether the second hop launches an **incremental** backward pulse on top of that memory.

## Guardrails

- Do not relax the IP1k 95th-percentile joint criterion post hoc; the formal replication outcome remains NO.
- Do not interpret residence-matched pseudo windows after the first hop as unperturbed background.
- The d1->d2 speed is a flux-packet propagation speed, not a unique normal-mode material group velocity.
- The 10 mV/A field remains a protocol control, not a calibrated threshold field.
- No mobility, hopping rate, activation energy or material phonon lifetime is inferred from IP1k.

## Next stage

Proceed to **IP1l: inter-hop wake memory and incremental second-hop radiation**. The primary tasks are to track the first-hop backward packet through the 2496-4108 fs interval, quantify its persistence behind the moving carrier, and test whether the second relocation adds a new backward flux impulse relative to the immediately preceding wake state rather than relative to an event-free zero-memory background.
