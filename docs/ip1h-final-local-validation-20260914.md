# IP1h final local validation — 2026-09-14

IP1h is numerically closed from the user-local artifact `ip1h-local-validation/20260914T153833Z` produced on branch `isotropic-polaron-barrier` at commit `a1e556e8ea6649ae786c65760843f9aa3f709a30`.

## Numerical closure

- runner status: PASS;
- failed gates: none;
- pycompile: PASS;
- focused IP1h tests: PASS;
- full test suite: PASS;
- source IP1g numerical status: PASS;
- finite fixed-boundary flux and propagation-delay diagnostics: PASS.

The reanalysis is posthoc: no dynamics were rerun.

## Validated packet propagation

The harmonic upper group-velocity scale for the 40x40 control is 1.84391 sites/ps. Fixed-boundary cross-correlations give physically admissible packet speeds below this scale.

For J0y/J0x = 1.0:

- backward median packet speed: 1.64767 sites/ps, four validated boundary pairs, minimum correlation 0.99879;
- forward median packet speed: 1.67793 sites/ps, four validated boundary pairs, minimum correlation 0.99954.

For J0y/J0x = 0.15:

- backward median packet speed: 1.67316 sites/ps, four validated boundary pairs, minimum correlation 0.99873;
- forward median packet speed: 1.62894 sites/ps, four validated boundary pairs, minimum correlation 0.83140.

Thus the controlled IP1g relocation launches genuine propagating lattice-energy packets in both directions. These are flux-packet speeds, not a unique normal-mode group velocity.

## Directionality

Using positive outward energy through matched fixed boundaries,

D(d) = (E_back^+(d) - E_front^+(d)) / (E_back^+(d) + E_front^+(d)).

The isotropic control weakly favors the backward side near the source and becomes nearly symmetric by six sites: D(2)=0.1206, D(4)=0.0386, D(6)=0.01245.

The anisotropic control changes character with distance: D(2)=-0.3263, D(3)=0.1207, D(4)=0.7410, D(5)=0.7891, D(6)=0.8452. The long-range packet is therefore strongly retrograde relative to the imposed +x relocation in this controlled quench.

## Interpretation boundary

IP1h validates directional lattice-radiation propagation after the controlled IP1g electronic-relocation impulse. It does **not** establish that the same asymmetry accompanies a naturally field-driven moving polaron. It also does not establish mobility, a hopping rate, a calibrated phonon lifetime, or a material group velocity.

The next stage must first identify a clean self-consistent field-driven transport protocol. Only after that screening should the fixed-boundary wake diagnostic be applied to actual carrier motion.
