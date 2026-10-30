# ACE Workflow

ACE inhibitor discovery workflow reconstructed from the DDR1 notebook sequence.
It leaves the original DDR1 notebooks and all source datasets untouched.

All inputs, models, reports, poses, and figures live in **Google My Drive**:
`My Drive/ACE_Workflow`. There are no `D:`-drive data fallbacks.

## Run in Colab CPU or VS Code

**Colab CPU:** upload the complete `ACE_Workflow` directory to the top of My
Drive, open its notebooks there, and run notebook 0. Set
`INSTALL_COLAB_CPU_DEPENDENCIES = True` once per fresh runtime: it installs
`requirements.txt` and downloads the official AutoDock Vina 1.2.5 binary. All
artifacts are written back to My Drive. No GPU is requested anywhere.

**VS Code:** use Google Drive for desktop to sync `My Drive/ACE_Workflow`, then
open that synced folder in VS Code and select its Python/Jupyter interpreter.
Install `requirements.txt`; install AutoDock Vina **>= 1.2** separately and expose
it as `vina` (or `vina.exe`) on the system path (Vina 1.1.x lacks options used here).
If VS Code starts the notebook from another folder, set `ACE_WORKFLOW_ROOT` to
the synced `ACE_Workflow` directory.

## Inputs

Place every input below in `My Drive/ACE_Workflow/data/raw/` before running:

- `CHEMBL1808.csv`, `.tsv`, or `.xlsx` — the only training database.
- `CHEMBL2096989.csv`, `.tsv`, or `.xlsx` — reserved exclusively for external validation.
- `FDA_DrugBank_cleaned.csv`, `Experimental_ziai_cleaned.csv`,
  `NPASS_cleaned.csv`, `Plants_CMAUP_cleaned.csv`, and
  `glaxo_structural_alerts.csv`.
- `1o86.pdb` — ACE receptor for redocking and docking.

ChEMBL exports need a SMILES column and either pChEMBL or IC50 standard values.
For the biochemical-assay filter they should also include `assay_type`
(otherwise the code falls back to BAO format, then to description keywords, and
prints a warning). Comma-/semicolon-delimited CSV, TSV and Excel are supported;
wrapping quotes around values (`'IC50'`, `'='`) are removed.

## Data-selection rules (defined once in `ace_config.py`)

- **Activity:** active if IC50 <= **25 nM** (`IC50_CUTOFF_NM`); the regressor/drugability
  cutoff is the same value expressed as pChEMBL (9 - log10(25) = 7.60).
- **Assays:** biochemical / binding only (`BIOCHEMICAL_ONLY`, ChEMBL `assay_type == "B"`).
- **Censored data:** `>` / `>=` IC50 records >= 10,000 nM are kept as *inactive*
  (`CENSORED_INACTIVE_MIN_NM`); `<` records and smaller `>` bounds are dropped.
  Censored compounds have no pChEMBL and are not used by the regressor.
- **Prodrugs:** ester / thioester / phosphinate-ester ACE-inhibitor prodrugs
  (e.g. enalapril, ramipril, fosinopril, zofenopril) are removed from training,
  external and screening sets (`EXCLUDE_PRODRUGS`); removed rows are saved as CSV.
- Model selection uses scaffold-grouped cross-validation within CHEMBL1808 only;
  CHEMBL2096989 is never used for tuning.

## Notebook order

1. `notebooks/0-ACE-CONFIG.ipynb`
2. `notebooks/1-ACE-MODEL-IC50-CUTOFF.ipynb`
3. `notebooks/2-ACE-MODEL-REGRESSOR.ipynb` (writes `ace_tanimoto.py` to the project root)
4. `notebooks/3-ACE-DRUGABILITY.ipynb`
5. `notebooks/4-ACE-REDOCK-1O86.ipynb`
6. `notebooks/5-ACE-DOCK-1O86.ipynb` (flexible Vina; redocking-failed protocol, kept for reference)
   `notebooks/5b-ACE-DOCK-ENSEMBLE.ipynb` (conformer-ensemble protocol validated in notebook 4; docks the experiment plan from notebook 9)
7. `notebooks/6-ACE-VIZ.ipynb`
8. `notebooks/7-ACE-DOCK-SCREENING-POWER.ipynb` (does the Vina score separate measured actives from inactives?)
9. `notebooks/8-ACE-ML-EVIDENCE.ipynb` (calibrated probabilities, conformal sets/intervals, applicability domain)
10. `notebooks/9-ACE-EVIDENCE-TABLE.ipynb` (evidence tiers, zinc-binding annotation, experimental shortlist with controls)

`3-ACE-DRUGABILITY` calculates Lipinski/Veber descriptors and Glaxo SMARTS
alerts. They are transparent prioritization rules, not clinical ADMET or toxicity
claims; check its positive-control table (known ACE drugs) before interpreting
the pass list. `4-ACE-REDOCK-1O86` must reproduce the lisinopril pose within 2 A
for at least 2 of 3 seeds before candidate docking is allowed to run.

## Docking limitations

Vina scores are not binding affinities. Vina has no explicit zinc-coordination
term, ligands are docked in their neutral SMILES form with undefined stereocentres
embedded arbitrarily, and the receptor is rigid. Treat docking and PLIP contacts
as prioritisation evidence only; experimental ACE inhibition and ADMET testing are required.

Crystal structures often contain residues with unresolved side-chain atoms that Meeko
cannot parameterize. `4-ACE-REDOCK-1O86` lists them (`1o86_incomplete_residues.csv`),
deletes only those far outside the docking box, and stops if an incomplete residue is
near the pocket. The full Meeko log is saved next to the receptor files.

## Evidence workflow (how candidates are supported)

Redocking of lisinopril did not reproduce the crystal pose with the flexible Vina protocol, so docking scores are **not** treated as confirmation. Evidence comes from:

1. **Validated ML models** (scaffold-grouped CV + external check), turned into per-candidate evidence in notebook 8: calibrated probability, class-conditional conformal prediction sets, conformal pChEMBL/IC50 intervals widened by applicability-domain bin.
2. **Zinc-binding-group annotation** (`ace_config.zinc_binding_groups`): carboxylic acid, thiol, phosphorus acid, hydroxamic acid, tetrazole, plus catechol as a weak chelator. It is shown as interpretable evidence and never used as a filter.
3. **Docking**, reported with its validation status. Candidate docking uses the conformer-ensemble protocol (notebook 5b), which reproduced the lisinopril crystal pose in 2 of 3 redocking seeds with thin score margins (weakly validated, pose reproduction only); it writes `ace_ensemble_dock.py` next to `ace_config.py` so notebooks 5b and 7 run identical code. Notebook 7 (which automatically uses the same protocol) tests whether the Vina score separates measured actives from size-matched measured inactives (ROC AUC with bootstrap CI, against heavy-atom, ligand-efficiency, size-adjusted and ZBG baselines). Notebook 9 counts docking as a supporting signal only if that test reports `USEFUL_BEYOND_SIZE`.

Notebook 9 assigns explicit evidence tiers (A, B, exploratory, unsupported) and proposes a diverse shortlist for an ACE inhibition assay with positive controls (captopril, lisinopril) and size-matched negative controls from ChEMBL. Tiers describe computational support, not activity.

## Repository hygiene

Do not commit `__pycache__/`. Check licences before publishing third-party raw data
(ChEMBL, DrugBank, NPASS, CMAUP). A suitable `.gitignore` is included.
