# ACE Workflow

ACE inhibitor discovery workflow reconstructed from the DDR1 notebook sequence.
It leaves the original DDR1 notebooks and all source datasets untouched.

All inputs, models, reports, poses, and figures live in **Google My Drive**:
`My Drive/ACE_Workflow`. There are no `D:`-drive data fallbacks.

## Run in Colab CPU or VS Code

**Colab CPU:** upload the complete `ACE_Workflow` directory to the top of My
Drive, open its notebooks there, and run notebook 0. Its optional setup cell
installs dependencies and AutoDock Vina in the temporary CPU runtime; all
artifacts are written back to My Drive. No GPU is requested anywhere.

**VS Code:** use Google Drive for desktop to sync `My Drive/ACE_Workflow`, then
open that synced folder in VS Code and select its Python/Jupyter interpreter.
Install `requirements.txt`; install AutoDock Vina separately and expose it as
`vina` (or `vina.exe`) on the system path. If VS Code starts the notebook from
another folder, set `ACE_WORKFLOW_ROOT` to the synced `ACE_Workflow` directory.
This gives both environments the exact same files and paths.

## Inputs

Place every input below in `My Drive/ACE_Workflow/data/raw/` before running:

- `CHEMBL1808.csv`, `.tsv`, or `.xlsx` — the only training database.
- `CHEMBL2096989.csv`, `.tsv`, or `.xlsx` — reserved exclusively for external validation.
- `FDA_DrugBank_cleaned.csv`, `Experimental_ziai_cleaned.csv`,
  `NPASS_cleaned.csv`, `Plants_CMAUP_cleaned.csv`, and
  `glaxo_structural_alerts.csv`.
- `1o86.pdb` — ACE receptor for redocking and docking.

ChEMBL inputs need a SMILES column plus either pChEMBL values or IC50 / standard
values. If standard type and relation columns are present, only equality IC50
measurements are retained.

## Notebook order

1. `notebooks/0-ACE-CONFIG.ipynb`
2. `notebooks/1-ACE-MODEL-IC50-CUTOFF.ipynb`
3. `notebooks/2-ACE-MODEL-REGRESSOR.ipynb`
4. `notebooks/3-ACE-DRUGABILITY.ipynb`
5. `notebooks/4-ACE-REDOCK-1O86.ipynb`
6. `notebooks/5-ACE-DOCK-1O86.ipynb`
7. `notebooks/6-ACE-VIZ.ipynb`

The default binary activity definition is IC50 <= 1,000 nM. Change it only
before fitting and record the rationale. Model selection uses scaffold-grouped
cross-validation within CHEMBL1808; CHEMBL2096989 is never used for tuning.

`3-ACE-DRUGABILITY` calculates Lipinski/Veber ADME-relevant descriptors and
Glaxo SMARTS alerts. They are transparent prioritization rules, not clinical
ADMET or toxicity claims. `4-ACE-REDOCK-1O86` must pass an RMSD review before
candidate docking results are interpreted.
