# Holstein–Peierls Transport

Modern Python implementation of a semiclassical Holstein–Peierls model for charge transport in two-dimensional molecular organic semiconductors.

## Scope

The project is being modernized from a legacy Fortran workflow. Development is intentionally split into two scientific stages:

1. **Static polaron solver** — simultaneous convergence of the excess-charge ground state and lattice deformation using RPROP.
2. **Polaron dynamics** — time propagation of the electronic wavefunction and classical lattice degrees of freedom, including external electric fields and later extensions for disorder and temperature.

The first implementation milestone is a numerically faithful CPU reference version of the static solver. Performance optimizations, sparse eigensolvers, multithreading, and GPU acceleration will be introduced only after regression tests against the legacy implementation are established.

## Legacy archive

The original Fortran sources and reconstructed legacy input parameter sets are preserved in the project Google Drive under:

`CODIGOS/holstein-peierls-transport/legacy/`

The Drive archive contains SHA-256 checksums and is treated as immutable reference material. Original legacy files must not be edited in place.

## Development principles

- Scientific code, documentation, variable names, and comments are written in English.
- Physics and numerical algorithms are kept modular and independently testable.
- Every scientific change is validated against the legacy reference or published benchmarks.
- CPU and GPU implementations must share the same high-level solver API.
- Reproducible runs record parameters, software version, Git commit, numerical backend, and hardware information.
- Publication results will be tied to tagged releases and archived input/output datasets.

## Planned package layout

```text
src/holstein_peierls/
    parameters.py
    lattice.py
    hamiltonian.py
    energy.py
    gradients.py
    rprop.py
    polaron.py
    observables.py
    io.py
    plotting.py
    backends/
        numpy_backend.py
        cupy_backend.py

tests/
examples/
benchmarks/
docs/
```

## Current status

**Phase 0 — legacy audit and scientific specification.**

The complete legacy workflow has been identified and archived. The next milestone is the CPU reference implementation of the static RPROP polaron solver together with regression and physics tests.
