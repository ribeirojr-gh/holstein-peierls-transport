# IP0a local validation — 2026-09-06

## Execution provenance

Local ZIP validation was produced from a GitHub branch ZIP, so git commit/branch/status are `unknown` by design and accepted for this validation path.

Environment:

- Python 3.12.3
- WSL2 Linux 6.18.33.2
- NumPy 2.5.2
- SciPy 1.18.1
- `OPENBLAS_NUM_THREADS=1`
- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`

## Numerical closure

All IP0a gates passed:

- `py_compile`: PASS
- focused pytest: PASS (13 tests)
- full pytest: PASS (345 tests)
- 20x20 frozen translation benchmark: PASS
- all static relaxations converged
- maximum translated-endpoint energy mismatch: `4.44e-16 eV`
- maximum explicit energy-decomposition error: `7.77e-16 eV`
- maximum electronic norm error: `7.77e-16`

IP0a is therefore numerically closed.

## Frozen-path barriers

The path is a linear interpolation of the classical lattice between a relaxed polaron and its exact one-site translation, with the electronic ground state re-solved at every image. These values are upper-bound path diagnostics, not MEP barriers or activation energies.

| J0y/J0x | static participation number | +x barrier [meV] | +y barrier [meV] |
|---:|---:|---:|---:|
| 0.15 | 2.2332 | 13.0462 | 101.5223 |
| 0.30 | 2.3187 | 12.4968 | 77.4475 |
| 0.50 | 2.5320 | 11.3112 | 51.2417 |
| 0.70 | 2.7019 | 11.3564 | 25.0048 |
| 1.00 | 3.2020 | 12.0140 | 12.0140 |

Every maximum occurs at the midpoint of the interpolation (`s=0.5`). The isotropic x/y profiles are equal to numerical accuracy.

## Main physical result

The original working hypothesis that the immobile isotropic polaron is pinned by a *larger static translation-energy barrier* is not supported by IP0a.

The easy-direction (+x) frozen barrier remains nearly constant, approximately 11--13 meV, across the whole anisotropy scan. In the transverse (+y) direction, the frozen barrier instead decreases strongly as isotropy is approached, from about 102 meV at `J0y/J0x=0.15` to the same approximately 12 meV value at isotropy.

At 300 K, `k_B T` is about 25.9 meV. Since the isotropic frozen-path barrier is already only about 12 meV and a true relaxed MEP cannot be higher than this particular upper-bound path if the same adiabatic ground-state surface is followed continuously, a large static potential-energy barrier by itself is unlikely to explain the previously observed absence of isotropic transport.

This does **not** yet establish a hopping rate, because kinetic/collective inertia and electronic-state continuity can still suppress translation even when the adiabatic energy profile is shallow.

## Energy decomposition at the midpoint

For the isotropic +x path, the approximately 12.014 meV total barrier results from a large cancellation:

- lattice contribution: `-63.245 meV`
- electronic contribution: `+75.259 meV`

The main positive electronic changes are the Holstein term (`+54.486 meV`) and the y-Peierls term (`+39.201 meV`), partially compensated by more favorable bare transfer and x-Peierls contributions. Thus the two-dimensional distortion participates materially in the translation path even though the total barrier is small.

For the strongly anisotropic `J0y/J0x=0.15` +x path, the corresponding total barrier is 13.046 meV, again from cancellation between a favorable lattice change (`-73.213 meV`) and an unfavorable electronic change (`+86.259 meV`).

## Charge sharing / branch diagnostic

At the midpoint of the isotropic paths, the source and target molecular populations are equal (approximately 0.312 each), and the participation number increases from approximately 3.20 at the relaxed endpoint to approximately 4.68 at the path midpoint. This is consistent with transient charge expansion/sharing during translation.

In contrast, for the low-anisotropy +x paths the midpoint electronic ground-state output is strongly asymmetric between the two symmetry-related sites despite a symmetric lattice interpolation. This is a warning that the lowest two electronic states may be nearly degenerate and that a solver can select a localized member of an almost-degenerate doublet. The associated low-state splitting must be measured before treating the adiabatic path as a single smooth reaction surface.

## Revised interpretation

IP0a shifts the next question from

> "Does isotropy create a larger potential-energy barrier?"

to

> "Does isotropy create a dynamically heavier collective translation coordinate, a different electronic avoided crossing, or both?"

This is closer to the effective-mass interpretation of the earlier work. A polaron can have a shallow translational potential barrier but still move slowly if translating it requires coherent motion of a large mass-weighted lattice deformation, which lowers the attempt frequency.

## Next stage: IP0b

IP0b should therefore quantify, before a full NEB calculation:

1. the mass-weighted lattice displacement between equivalent translated polarons, decomposed into `u`, `vx`, and `vy` contributions;
2. a collective reaction-coordinate inertia using the already validated legacy masses converted to eV fs^2/A^2;
3. the harmonic curvature of the frozen path near each minimum and the corresponding collective attempt-frequency scale;
4. the lowest electronic energy gap at the symmetric midpoint, to detect near-degenerate/branch-switching cases;
5. x/y equivalence at isotropy and anisotropy trends without pre-registering a desired physical outcome.

Only after these diagnostics should constrained relaxation/MEP be added. The temperature hypothesis remains viable, but IP0a indicates that temperature may help primarily through activation of a heavy collective coordinate and transient symmetry breaking rather than by overcoming a large static energy barrier.

## Scope limits

IP0a does not establish:

- a minimum-energy path;
- a Peierls-Nabarro activation barrier;
- a finite-temperature free-energy barrier;
- a complete polaron effective mass;
- a hopping rate;
- a mobility;
- a material-specific prediction.
