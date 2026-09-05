# TP2b — superseded linear-response convergence protocol

> **STATUS: SUPERSEDED BEFORE EXECUTION. DO NOT USE THIS RUNNER AS A PHYSICAL MOBILITY GATE.**

The original TP2b protocol was written after TP2a as a long-trajectory paired-field convergence study. It assumed that the main unresolved issue was statistical convergence of an approximately linear drift response.

That assumption is not appropriate for the physical regime being modeled.

## Why this protocol was superseded

The current legacy control is strongly anisotropic:

- `J0x = 0.100 eV`
- `J0y = 0.015 eV`
- `J0y/J0x = 0.15`

This is a hopping-favorable anisotropic regime studied in the underlying Holstein-Peierls work. The same body of work shows that:

1. transport is mediated by site-to-site hopping rather than a smooth band-like drift;
2. the hopping event is accompanied by charge compression/delocalization/recompression and lattice reorganization;
3. increasingly isotropic two-dimensional systems can become effectively immobile because the polaron must carry intra-site and both x/y intermolecular distortions;
4. for the `J0x = 100 meV`, `J0y = 15 meV` control, weak fields can produce little motion while stronger fields activate visible transport.

Therefore a single through-origin linear fit across 0.5, 1.0 and 2.0 mV/A can straddle different hopping regimes. A low R^2 or a confidence interval including zero is not automatically a sampling failure and must not trigger progressively longer simulations designed to force a non-zero mobility.

## Replacement

The active replacement is `docs/tp2b-hopping-mechanism-design.md`.

The replacement protocol first validates the hopping mechanism itself, including localization/IPR, nearest-neighbour transitions, residence times, field-biased hop counts, unwrapped hop displacement, and the correlated motion of the charge and lattice distortion. Only after the hopping regime and field-onset window are established will a diffusion or mobility estimator be selected.

The previous TP2b experiment/runner may remain in the repository for provenance, but it is not an active closure gate.
