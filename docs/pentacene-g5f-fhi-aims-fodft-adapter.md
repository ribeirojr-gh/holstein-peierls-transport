# Pentacene G5f: documented FHI-aims H2n@DA FO-DFT adapter

## Purpose

G5f connects the backend-independent rigid-dimer finite-difference pipeline from
G5e to the documented fragment-orbital DFT workflow in FHI-aims.  It does not
run FHI-aims, bundle species defaults, choose HOMO state indices, or parse an
undocumented output format.

The implementation follows the current FHI-aims manual, section 3.48,
"Fragment molecular orbital DFT calculations":

`https://fhi-aims.org/uploads/manual/Ch3/S48.html`

The target flavour is the classic H2n@DA scheme used in the pentacene reference
protocol.  The manual specifies neutral fragment calculations with the default
FO-DFT options for this flavour.

## Three-calculation workflow

A single transfer-integral evaluation requires at least three FHI-aims
calculations:

```text
job_root/
  frag1/
    geometry.in
    control.in + G5f fragment block
  frag2/
    geometry.in
    control.in + G5f fragment block
  dimer/
    geometry.in
    control.in + G5f final block
```

The manual requires all three calculation folders to share one root and requires
the combined-system `geometry.in` to mirror the actual fragmentation: the atom
ordering must remain consistent between the dimer and each fragment.

G5f enforces this by rendering the dimer as all molecule-A atoms followed by all
molecule-B atoms.  The fragment renderers preserve exactly the corresponding
atomic order.

## Why G5f writes `fodft.control`, not a complete `control.in`

FHI-aims needs much more than the FO-DFT tags: XC settings, species defaults,
basis functions, SCF convergence controls, relativistic settings when relevant,
and installation-specific basis/species paths.  G5f does not have enough
information to generate those safely.

The `FodftInputBundle.file_map()` therefore uses the name `fodft.control` for the
FO-DFT-specific snippets.  A later execution environment must combine these
snippets with a validated common FHI-aims control template and the appropriate
species defaults.  Renaming the snippets to `control.in` without that composition
would create an incomplete input and is intentionally avoided.

## Fragment block

For H2n@DA the explicit reproducibility block is

```text
fo_dft fragment
fo_flavour default
```

`fo_flavour default` is technically the default, but G5f writes it explicitly so
that the intended H2n@DA occupation convention is visible in every archived job.

`fo_deltaplus` is never emitted by this adapter.

The fragment calculation writes the `restart.frag` and `info.frag` information
needed by the final combination step, according to the FHI-aims manual.

## Final block

For hole transfer the rendered final block has the form

```text
fo_dft final
fo_flavour default
fo_folders frag1 frag2
fo_orbitals STATE1 STATE2 1 1 hole
fo_verbosity 1
```

`STATE1` and `STATE2` are required positive integers supplied by the caller.  G5f
does not infer HOMO indices from electron counts because occupation, charge,
spin, and FHI-aims state indexing must be verified from the actual fragment
calculations.

The manual documents `hole`/`dn` for the spin-down Hamiltonian and `elec`/`up`
for spin-up, and permits positive state ranges.  G5f validates these documented
choices.

`fo_verbosity 1` is selected because the manual states that it writes the
calculated couplings and the full Hab matrix to `full_hab_submatrix`.

## Output parser boundary

The current manual documents the **name** and purpose of
`full_hab_submatrix`, but not its exact textual layout.  Public documentation
searches performed during G5f did not locate an authoritative format example.

Therefore G5f intentionally does **not** implement a parser for this file or for
a guessed FHI-aims stdout line.  A parser will be added only after one of the
following is available:

1. an official documented output example;
2. a small validated FHI-aims H2n@DA run whose output can become a regression
   fixture; or
3. authoritative FHI-aims source/documentation defining the serialization.

This boundary is represented by `g5f_output_parser_status()`.

## G5e scan manifest

`build_fodft_scan_manifest()` consumes one G5e `RigidDimerReference`, a
`FiniteDifferenceScanPlan`, and explicit fragment-state selection.  Each
perturbation receives a deterministic path-safe root name containing:

- oriented bond family;
- coordinate label;
- plus/minus sign;
- perturbation magnitude and unit.

For the initial pentacene G5e grid, one oriented bond contains 36 displaced
transfer-integral jobs.  Without making fragment-reuse assumptions, each coupling
job consists of three FHI-aims calculations, so the conservative cost is

```text
36 coupling jobs x 3 FHI-aims calculations = 108 calculations per oriented bond.
```

For all six explicit herringbone bond families this becomes

```text
216 coupling jobs x 3 = 648 FHI-aims calculations,
```

plus any equilibrium reference jobs used to establish state indices and orbital
gauge continuity.

This is a conservative bookkeeping count, not a claim that no fragments can ever
be reused.  Reuse will only be introduced after demonstrating that it preserves
the required orbital representation and FHI-aims fragment data for translated
or rotated geometries.

## Folder/path safety

Job and calculation folder names are restricted to letters, digits, underscore,
dot, and hyphen.  Path traversal, slashes, whitespace, `.`/`..`, and duplicate
fragment/final folder names are rejected.  This makes manifests safe to
materialize later in local/HPC workflows.

## Relation to the published pentacene protocol

The parameter record from G5e continues to describe the electronic-coupling
reference used by Neef et al.:

- H2n@DA fragment-orbital DFT;
- FHI-aims;
- PBE;
- tier2 numeric atom-centered basis;
- tight integration grids;
- electronic-level convergence below 1e-6 eV;
- no vdW correction for static isolated-dimer coupling calculations.

G5f does not repeat those general DFT keywords because it only owns the FO-DFT
fragment/final layer.  A future execution template must combine both pieces and
be regression-tested before numerical pentacene derivatives are promoted.

## Current acceptance gates

G5f is considered complete when:

1. fragment and dimer atom ordering is identical to the documented fragmentation;
2. the fragment block contains `fo_dft fragment` and explicit
   `fo_flavour default`;
3. the final block contains `fo_dft final`, explicit default flavour, custom
   fragment folders, validated `fo_orbitals`, and verbosity 1;
4. no delta-plus tag is generated;
5. invalid state indices and unsafe folder names fail early;
6. G5e perturbations map deterministically to unique FHI-aims job roots;
7. one oriented G5e scan produces exactly 36 coupling jobs / 108 conservative
   FHI-aims calculations;
8. no output parser is invented before an authoritative output format exists.

No numerical transfer integral or Peierls derivative is produced by this stage.
