# CPU Optimization and Validation — v0.2.0a1

## Scope

Version `0.2.0a1` optimizes the static Holstein–Peierls polaron solver while
preserving an explicit reference path for regression against the archived
Fortran implementation. No time-dependent dynamics are introduced in this
version.

The scientific model, periodic lattice, RPROP update equations, and legacy
input format remain unchanged. The changes in this version target algorithmic
cost, memory use, and reproducibility of the numerical implementation.

## Optimizations

### Vectorized Hamiltonian construction

The dense Hamiltonian is assembled without Python site loops. The sparse
Hamiltonian is constructed directly as a CSR matrix instead of first allocating
a dense `N x N` matrix and converting it afterwards.

For the nearest-neighbour two-dimensional model, the sparse representation has
O(N) non-zero elements rather than O(N²) storage.

### Ground-state-only electronic solution

The `dense_lowest` path computes only the lowest electronic eigenpair. The
`sparse` path uses `scipy.sparse.linalg.eigsh` and therefore avoids full-spectrum
diagonalization.

During RPROP, the electronic ground state calculated after a lattice update is
reused for the gradient at the beginning of the following iteration. This
removes one redundant electronic solution per RPROP step.

For the sparse solver, the previous ground-state wavefunction is also used as
the initial vector for the next iterative eigensolution. The first sparse solve
uses a vector localized at the same `polaron_position` used to seed the legacy
lattice distortion.

### O(N) analytical gradient

The reference implementation forms the complete one-particle density matrix

`rho[i,j] = psi[i] * psi[j]`.

The lattice gradient requires only diagonal and nearest-neighbour elements.
For a real one-particle wavefunction, the electronic contribution can therefore
be written directly in terms of local products of `psi`. For example,

`rho[i,left] + rho[left,i] - rho[right,i] - rho[i,right]`

is exactly

`2 * psi[i] * (psi[left] - psi[right])`.

The optimized gradient consequently uses O(N) temporary storage and fully
vectorized periodic-neighbour operations. The original density-matrix and loop
ordering remains available through `gradient_mode="reference"` or the command
line option `--gradient reference`.

## Regression modes

The closest modern analogue of the archived static code is

```bash
hp-polaron \
  --parameters examples/static_polaron/parameters1.inc \
  --solver dense_full \
  --gradient reference \
  --legacy-convergence \
  --output run-reference
```

The recommended optimized calculation is

```bash
hp-polaron \
  --parameters examples/static_polaron/parameters1.inc \
  --solver sparse \
  --gradient optimized \
  --output run-optimized
```

The CLI retains `dense_lowest` as its default eigensolver in `0.2.0a1` so that
switching from the reference release does not silently introduce an iterative
solver. Users who want the best CPU scaling should request `--solver sparse`
explicitly.

## Physical equivalence and lattice symmetries

A periodic translationally invariant lattice has multiple representations of the
same localized polaron. Two converged calculations can therefore place the
same density profile on different periodic cells while having identical energy
and physical observables.

The Peierls coordinates have an additional zero mode: adding a constant to all
`vx` values, or independently to all `vy` values, does not change the model
because the Hamiltonian and lattice energy depend on coordinate differences.
Consequently, validation must not compare absolute `vx` or `vy` values before
removing these physically irrelevant freedoms.

The `20 x 20` validation therefore uses the following policy:

1. compare scalar energies and charge normalization directly;
2. align charge density and the Holstein coordinate by the best periodic lattice
   translation;
3. compare Peierls bond distortions `vx(x+1)-vx(x)` and `vy(y+1)-vy(y)` after the
   same periodic alignment;
4. report the mean absolute-coordinate offsets separately as zero-mode
   diagnostics.

## 20 x 20 full-relaxation validation

Input: `examples/static_polaron/parameters1.inc`.

The legacy-compatible reference calculation (`dense_full`, reference gradient,
legacy u-only stopping rule) gives a formation energy of
`0.6354590577000392 eV` on the GitHub Actions runner. The independently known
legacy value for this input is approximately `0.6354590577418 eV`.

The optimized legacy-stop calculations agree as follows:

| Quantity | dense_lowest optimized | sparse optimized |
| --- | ---: | ---: |
| Formation-energy difference from reference | 2.92e-11 eV | 2.21e-11 eV |
| Periodic alignment shift (y, x) | (0, 0) | (0, 1) |
| Max aligned charge-density difference | 5.50e-7 | 1.52e-6 |
| Max aligned `u` difference | 8.34e-8 | 3.25e-7 |
| Max x-bond distortion difference | 3.03e-6 | 3.80e-6 |
| Max y-bond distortion difference | 1.05e-6 | 8.99e-7 |

The modern all-coordinate stopping rule converges `u`, `vx`, and `vy` after
523 iterations on this runner and gives a formation energy of
`0.6354590577561128 eV`, differing from the legacy-stop reference by about
`5.6e-11 eV`.

## CPU microbenchmarks

The following numbers are medians of three measurements on a GitHub-hosted
Ubuntu runner with `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and
`MKL_NUM_THREADS=1`. They are implementation benchmarks, not portable hardware
performance claims.

| Lattice | Kernel | Reference / dense | Optimized / sparse | Speedup |
| --- | --- | ---: | ---: | ---: |
| 20 x 20 | analytical gradient | 1.285 ms | 0.0963 ms | 13.4x |
| 40 x 40 | analytical gradient | 6.519 ms | 0.109 ms | 59.8x |
| 80 x 80 | analytical gradient | 44.248 ms | 0.147 ms | 300.8x |
| 20 x 20 | electronic ground state | dense_lowest 6.656 ms | sparse 3.749 ms | 1.78x |
| 40 x 40 | electronic ground state | dense_lowest 228.092 ms | sparse 3.748 ms | 60.9x |

Dense eigensolution was deliberately omitted for `80 x 80` in this benchmark to
avoid allocating and diagonalizing the full `6400 x 6400` matrix.

A representative complete `20 x 20` relaxation on the same hosted runner took
approximately 4.57 s for the strict dense reference, 1.41 s for optimized
`dense_lowest`, and 0.33 s for optimized sparse with the legacy stopping rule.
These wall times are more variable than the kernel microbenchmarks because they
include different iteration histories and shared-runner effects.

## Reproducible benchmark commands

```bash
python benchmarks/static_cpu.py \
  --sizes 20 40 80 \
  --solvers dense_lowest sparse \
  --repeats 3 \
  --dense-max-sites 1600 \
  --output benchmark-results
```

Full static validation:

```bash
python benchmarks/validate_static_20x20.py \
  --parameters examples/static_polaron/parameters1.inc \
  --output validation-results
```

The GitHub workflows `benchmark-cpu.yml` and `validation-cpu.yml` provide manual
release checks using the same commands.
