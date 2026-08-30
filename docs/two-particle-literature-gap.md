# Two-particle Holstein-Peierls literature gap

## Scope

This note records the initial literature map for extending the static two-dimensional molecular-crystal Holstein-Peierls model from one carrier to correlated two-particle states. It is a working research note, not yet a systematic review.

Consensus search quota was exhausted during this stage, so the initial survey used SciSpace and direct publisher/index searches in addition to the papers already archived with the project.

## 1. What the current project already establishes

Mozafari and Stafstrom (Physics Letters A 376, 1807-1811, 2012; DOI 10.1016/j.physleta.2012.04.004) studied static polaron stability in a two-dimensional molecular crystal with simultaneous semiclassical Holstein and Peierls coupling. They found that the combined local and nonlocal lattice response can stabilize polarons non-additively and identified molecular-centered and intermolecular-centered localized solutions.

Mozafari and Stafstrom (Journal of Chemical Physics 138, 184104, 2013; DOI 10.1063/1.4803691) developed the associated two-dimensional polaron dynamics. Later work from Ribeiro Junior and collaborators explored coupling symmetry, anisotropy, and material-oriented extensions.

The present Python project reproduces and optimizes the static one-carrier problem and therefore offers a validated structural starting point for the two-particle extension.

## 2. Bipolarons: established literature

### 2.1 Two-dimensional Holstein-Hubbard bipolarons already exist

Proville and Aubry studied the adiabatic two-dimensional Holstein-Hubbard model (Eur. Phys. J. B 11, 41-58, 1999; DOI 10.1007/s100510050915). Their phase diagram contains unbound carriers, onsite bipolarons, nearest-neighbour/intersite bipolarons, and a quadrisinglet state. This is particularly relevant because their treatment is adiabatic/semiclassical and two-dimensional, making it the closest conceptual benchmark for the Holstein-only limit of our proposed solver.

A companion work examined quantum corrections and bipolaron mobility near the adiabatic limit.

Macridin, Sawatzky, and Jarrell later studied the two-dimensional Hubbard-Holstein bipolaron with diagrammatic Monte Carlo (Phys. Rev. B 69, 245111, 2004; DOI 10.1103/PhysRevB.69.245111), again finding onsite and nearest-neighbour bipolaron regimes as Coulomb repulsion and electron-phonon coupling are varied.

Therefore, a claim of novelty based simply on "two-dimensional Holstein bipolarons" would be incorrect.

### 2.2 Peierls coupling can stabilize bipolarons

Sous, Chakraborty, Krems, and Berciu demonstrated strongly bound, relatively light bipolarons stabilized by Peierls electron-phonon coupling (Phys. Rev. Lett. 121, 247001, 2018; DOI 10.1103/PhysRevLett.121.247001). The important mechanism in that quantum model is an effective phonon-mediated pair-hopping interaction rather than a conventional density-density attraction. The resulting bipolarons can remain stable against substantial Coulomb repulsion.

This work establishes that Peierls coupling is not merely a perturbation to bipolaron physics. However, its model and objective differ from the present semiclassical molecular-crystal problem.

Related work on two-dimensional Peierls/SSH polarons (Phys. Rev. B 104, 035143, 2021; DOI 10.1103/PhysRevB.104.035143) shows that the character of displacement-modulated hopping matters strongly in higher dimensions.

Therefore, a claim of novelty based simply on "Peierls bipolarons" would also be incorrect.

## 3. Excitons in pentacene and molecular crystals: established literature

Excitons in pentacene have been studied extensively with many-body first-principles and vibronic Hamiltonian approaches.

Cudazzo et al. (Phys. Rev. B 86, 195307, 2012; DOI 10.1103/PhysRevB.86.195307) showed from Bethe-Salpeter calculations that the absorption onset in pentacene has strong charge-transfer exciton character.

Sharifzadeh et al. (J. Phys. Chem. Lett. 4, 2197-2201, 2013; DOI 10.1021/jz401069f) quantified electron-hole separation and charge-transfer character in solid pentacene from first principles.

Beljonne et al. (Phys. Rev. Lett. 110, 226402, 2013; DOI 10.1103/PhysRevLett.110.226402) combined quantum chemistry with a Frenkel-Holstein model and showed that charge-transfer admixture is essential to the lowest singlet excitations and singlet-fission physics.

Tempelaar and Reichman (J. Chem. Phys. 148, 244701, 2018; DOI 10.1063/1.5031778) treated vibronic singlet-fission dynamics in pentacene with strong Holstein coupling and approximate Peierls effects.

Alvertis et al. (Phys. Rev. Lett. 130, 086401, 2023; DOI 10.1103/PhysRevLett.130.086401) used first-principles GW-BSE and nuclear quantum/anharmonic methods to show strong phonon-induced localization of excitons in pentacene.

Other molecular-exciton studies explicitly demonstrate that nonlocal Peierls phonons can strongly affect exciton transfer and localization.

Therefore, a claim that phonon-coupled or Holstein-Peierls excitons in pentacene are generally unexplored would be too broad.

## 4. More defensible research gap

The initial search did not identify a study matching the following complete combination:

- a two-dimensional anisotropic molecular crystal represented at the molecular-site level;
- simultaneous semiclassical local Holstein and nonlocal Peierls lattice coordinates;
- explicit structural relaxation of both intra- and intermolecular coordinates;
- a correlated two-identical-carrier wavefunction for bipolaron formation, including Coulomb repulsion;
- systematic mapping of bound/unbound, onsite/intersite, and structurally distinct bipolaron states in parameter ranges motivated by organic molecular semiconductors;
- direct continuity with a validated one-polaron model for the same molecular lattice.

For excitons, the potentially distinctive combination is:

- a correlated electron-hole real-space wavefunction on a two-dimensional organic molecular crystal;
- separate HOMO-like and LUMO-like Holstein and Peierls parameters;
- explicit optimization of both intramolecular and intermolecular lattice coordinates around a neutral electron-hole pair;
- simultaneous characterization of Frenkel/charge-transfer character, electron-hole binding, and lattice self-trapping within the same structural model;
- direct comparison with the single electron and single hole polaron limits.

These statements are deliberately narrower than "first Holstein-Peierls bipolaron" or "first phonon-coupled exciton" and should remain hypotheses until a more exhaustive literature review is complete.

## 5. Why the problem remains scientifically interesting despite prior work

The current single-polaron literature already shows a strongly non-additive Holstein-Peierls stabilization in molecular crystals. A two-particle state adds a new competition:

- kinetic delocalization;
- local Holstein relaxation;
- nonlocal Peierls relaxation;
- Coulomb attraction or repulsion;
- anisotropic molecular transfer integrals;
- possible multiple structurally distinct local minima.

For bipolarons, this means that a bound state may emerge through mechanisms qualitatively different from either a pure Holstein-Hubbard onsite bipolaron or a purely quantum Peierls pair-hopping bipolaron.

For excitons, overall charge neutrality does not eliminate lattice coupling. Electron and hole Holstein/Peierls couplings are orbital-energy and transfer-integral derivatives and therefore need not cancel. This creates the possibility of a structurally self-trapped neutral state whose Frenkel/charge-transfer character changes with lattice relaxation.

## 6. Initial research questions

### Bipolaron

1. Does simultaneous Holstein and Peierls relaxation increase or decrease the bipolaron binding region relative to the 2D Holstein-Hubbard limit?
2. Can an intermolecular-centered or bond-centered bipolaron become stable even when onsite pairing is suppressed by U?
3. How does anisotropy J_y/J_x modify onsite, nearest-neighbour, and extended pair states?
4. Are there metastable structures analogous to the molecular-centered and intermolecular-centered single-polaron minima already known in the one-particle model?
5. Does Peierls coupling qualitatively change the critical Coulomb repulsion at which binding disappears?
6. Can physically relevant single-polaron formation energies (roughly 50-100 meV) coexist with a positive two-polaron binding energy?

### Exciton

1. Under what relations between A_e and A_h does lattice relaxation reinforce or cancel for a neutral electron-hole pair?
2. Can Peierls relaxation shift the balance between Frenkel-like and charge-transfer-like excitons?
3. Is there a self-trapping transition controlled by the same Holstein/Peierls parameters that control single-polaron stability?
4. How do electron-hole separation and exciton binding change when the lattice is allowed to relax compared with a rigid lattice?
5. Can the model reproduce qualitative trends from first-principles pentacene studies after HOMO/LUMO parameters are independently calibrated?

## 7. Proposed publication strategy

A first paper should focus only on the static bipolaron problem. The cleanest narrative is:

1. validate the new two-particle solver in the Holstein-Hubbard limit against established 2D phase behaviour;
2. switch on Peierls coupling while preserving the molecular-crystal lattice geometry;
3. map the new phase diagram in Holstein, Peierls, Coulomb, and anisotropy parameters;
4. identify structural signatures and binding energies of onsite, intersite, and unbound states;
5. then restrict attention to parameter ranges where the underlying single-polaron stabilization is physically plausible for organic molecular crystals.

The exciton problem should follow as a separate study because it requires new HOMO/LUMO parameterization, electron-hole screening, and possibly exchange. Combining both in the first implementation would make numerical validation and physical interpretation unnecessarily difficult.

## 8. Key references for the working bibliography

- E. Mozafari and S. Stafstrom, Polaron stability in molecular crystals, Physics Letters A 376, 1807-1811 (2012). DOI: 10.1016/j.physleta.2012.04.004.
- E. Mozafari and S. Stafstrom, Polaron dynamics in a two-dimensional Holstein-Peierls system, J. Chem. Phys. 138, 184104 (2013). DOI: 10.1063/1.4803691.
- L. Proville and S. Aubry, Small bipolarons in the 2-dimensional Holstein-Hubbard model. I. The adiabatic limit, Eur. Phys. J. B 11, 41-58 (1999). DOI: 10.1007/s100510050915.
- A. Macridin, G. A. Sawatzky, and M. Jarrell, Two-dimensional Hubbard-Holstein bipolaron, Phys. Rev. B 69, 245111 (2004). DOI: 10.1103/PhysRevB.69.245111.
- J. Sous, M. Chakraborty, R. V. Krems, and M. Berciu, Light Bipolarons Stabilized by Peierls Electron-Phonon Coupling, Phys. Rev. Lett. 121, 247001 (2018). DOI: 10.1103/PhysRevLett.121.247001.
- C. Zhang, N. Prokof'ev, and B. Svistunov, Peierls/Su-Schrieffer-Heeger polarons in two dimensions, Phys. Rev. B 104, 035143 (2021). DOI: 10.1103/PhysRevB.104.035143.
- P. Cudazzo, M. Gatti, and A. Rubio, Excitons in molecular crystals from first-principles many-body perturbation theory: Picene versus pentacene, Phys. Rev. B 86, 195307 (2012). DOI: 10.1103/PhysRevB.86.195307.
- S. Sharifzadeh et al., Low-Energy Charge-Transfer Excitons in Organic Solids from First Principles: The Case of Pentacene, J. Phys. Chem. Lett. 4, 2197-2201 (2013). DOI: 10.1021/jz401069f.
- D. Beljonne et al., Charge-Transfer Excitations Steer the Davydov Splitting and Mediate Singlet Exciton Fission in Pentacene, Phys. Rev. Lett. 110, 226402 (2013). DOI: 10.1103/PhysRevLett.110.226402.
- R. Tempelaar and D. R. Reichman, Vibronic exciton theory of singlet fission. III, J. Chem. Phys. 148, 244701 (2018). DOI: 10.1063/1.5031778.
- A. M. Alvertis et al., Phonon-Induced Localization of Excitons in Molecular Crystals from First Principles, Phys. Rev. Lett. 130, 086401 (2023). DOI: 10.1103/PhysRevLett.130.086401.
