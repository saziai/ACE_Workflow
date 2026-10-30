"""Shared configuration and chemistry helpers for the ACE inhibitor workflow.

The notebooks import this module after running ``0-ACE-CONFIG.ipynb``.  It does
not alter source datasets: all derived files are written under ``ACE_Workflow``.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


PROJECT_NAME = "ACE_Workflow"
RANDOM_STATE = 42
FP_RADIUS = 2
FP_SIZE = 2048
IC50_CUTOFF_NM = 25.0
# --- Data-selection rules (single source of truth; used by notebooks 1-3) ---------------
BIOCHEMICAL_ONLY = True               # keep only biochemical / binding assays (ChEMBL assay_type "B")
CENSORED_INACTIVE_MIN_NM = 10_000.0   # IC50 ">" / ">=" records at or above this value are kept as inactive
EXCLUDE_PRODRUGS = True               # drop ester / thioester / phosphinate-ester ACE-inhibitor prodrugs
SCREENING_CHUNK_SIZE = 50_000
COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive")
CPU_WORKERS = max(1, int(os.environ.get("ACE_CPU_WORKERS", min(os.cpu_count() or 2, 4))))


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def mount_google_drive() -> Path:
    """Mount the user's Google Drive when running in Colab; no-op in VS Code."""
    if not is_colab():
        return Path(os.environ.get("ACE_WORKFLOW_ROOT", Path(__file__).resolve().parent))
    from google.colab import drive
    if not COLAB_DRIVE_ROOT.exists():
        drive.mount("/content/drive", force_remount=False)
    return COLAB_DRIVE_ROOT


def workflow_root() -> Path:
    """Return the ACE_Workflow folder in Google My Drive.

    In Colab this is always ``/content/drive/MyDrive/ACE_Workflow``.  In VS
    Code, open the Google Drive for desktop-synced copy of that same folder, or
    set ``ACE_WORKFLOW_ROOT`` to it before launching the notebook kernel.
    """
    env_root = os.environ.get("ACE_WORKFLOW_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    if is_colab():
        return (mount_google_drive() / PROJECT_NAME).resolve()
    return Path(__file__).resolve().parent


def setup_paths() -> dict[str, Path]:
    """Create only ACE_Workflow result directories and return their locations."""
    root = workflow_root()
    p = {
        "PROJECT": root,
        "NOTEBOOKS": root / "notebooks",
        "DATA": root / "data",
        "DATA_RAW": root / "data" / "raw",
        "DATA_PROCESSED": root / "data" / "processed",
        "ML": root / "ml_model",
        "ML_RESULTS": root / "ml_model" / "results",
        "ML_CLASSIFICATION": root / "ml_model" / "results" / "ic50_classifier",
        "ML_REGRESSION": root / "ml_model" / "results" / "pchembl_regressor",
        "ML_MODELS": root / "ml_model" / "saved_models",
        "DRUGABILITY": root / "drugability",
        "DRUGABILITY_REPORTS": root / "drugability" / "reports",
        "REDOCKING": root / "redocking_1o86",
        "REDOCKING_RAW": root / "redocking_1o86" / "raw_structure",
        "REDOCKING_RECEPTOR": root / "redocking_1o86" / "receptor",
        "REDOCKING_LIGAND": root / "redocking_1o86" / "ligand",
        "REDOCKING_DOCKING": root / "redocking_1o86" / "docking",
        "REDOCKING_ANALYSIS": root / "redocking_1o86" / "analysis",
        "DOCKING": root / "candidate_docking",
        "DOCKING_RECEPTOR": root / "candidate_docking" / "receptor",
        "DOCKING_LIGANDS": root / "candidate_docking" / "ligands",
        "DOCKING_PDBQT": root / "candidate_docking" / "ligands" / "pdbqt",
        "DOCKING_RUNS": root / "candidate_docking" / "runs",
        "DOCKING_POSES": root / "candidate_docking" / "representative_poses",
        "DOCKING_COMPLEXES": root / "candidate_docking" / "complexes",
        "DOCKING_REPORTS": root / "candidate_docking" / "reports",
        "VISUALIZATION": root / "visualization",
        "VIS_FIGURES": root / "visualization" / "figures",
        "VIS_REPORTS": root / "visualization" / "reports",
    }
    for directory in p.values():
        directory.mkdir(parents=True, exist_ok=True)
    return p


class ACEConfig:
    """Central ACE workflow state, model settings, and My Drive path registry.

    This mirrors the DDR1 workflow's shared configuration object so all
    notebooks resolve exactly the same persistent directories in VS Code and
    Colab.  It never changes raw data files.
    """

    def __init__(self):
        self.is_colab = is_colab()
        self.execution_env = "colab_cpu" if self.is_colab else "vscode_cpu"
        self.drive_root: Optional[Path] = None
        self.project_root: Optional[Path] = None
        self._paths: dict[str, Path] = {}

    def mount_drive(self) -> Path:
        self.drive_root = mount_google_drive()
        return self.drive_root

    def setup(self) -> dict[str, Path]:
        self.drive_root = self.mount_drive()
        self.project_root = workflow_root()
        self._paths = setup_paths()
        return self._paths

    def get_config_dict(self) -> dict[str, str]:
        if not self._paths:
            self.setup()
        return {key: str(value) for key, value in self._paths.items()}

    def runtime_metadata(self) -> dict[str, object]:
        if not self._paths:
            self.setup()
        return {
            "project_name": PROJECT_NAME,
            "execution_environment": self.execution_env,
            "project_root": str(self.project_root),
            "drive_root": str(self.drive_root),
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_workers": CPU_WORKERS,
            "random_state": RANDOM_STATE,
            "fingerprint_radius": FP_RADIUS,
            "fingerprint_bits": FP_SIZE,
            "ic50_cutoff_nM": IC50_CUTOFF_NM,
            "biochemical_only": BIOCHEMICAL_ONLY,
            "censored_inactive_min_nM": CENSORED_INACTIVE_MIN_NM,
            "exclude_prodrugs": EXCLUDE_PRODRUGS,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }


CONFIG = ACEConfig()
PATHS = CONFIG.setup()


def setup_workflow() -> dict[str, Path]:
    """Explicit notebook entry point; safe to call repeatedly."""
    paths = CONFIG.setup()
    print(f"Environment : {CONFIG.execution_env.upper()}")
    print(f"Project root: {CONFIG.project_root}")
    print(f"Directories : {len(paths)}")
    return paths


def verify_paths(paths: Optional[dict[str, Path]] = None, required_keys: Optional[Iterable[str]] = None) -> bool:
    """Report persistent workflow directory availability in a stable order."""
    paths = paths or PATHS
    keys = list(required_keys or paths.keys())
    ok = True
    for key in keys:
        value = paths.get(key)
        exists = value is not None and value.exists()
        print(f"{key:24} {'OK' if exists else 'MISSING':7} {value}")
        ok &= exists
    return ok


def write_run_manifest(path: Path, extra: Optional[dict[str, object]] = None) -> Path:
    """Persist a machine-readable configuration record alongside an analysis."""
    payload = CONFIG.runtime_metadata()
    payload["inputs"] = {key: str(value) for key, value in INPUTS.items()} if "INPUTS" in globals() else {}
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def input_files() -> dict[str, Path]:
    """Return input locations stored under Google My Drive/ACE_Workflow/data/raw."""
    raw = PATHS["DATA_RAW"]
    def file(name: str) -> Path:
        choices = [raw / f"{name}.csv", raw / f"{name}.xlsx", raw / f"{name}.tsv"]
        return next((path for path in choices if path.exists()), choices[0])
    return {
        "CHEMBL1808": file("CHEMBL1808"),
        "CHEMBL2096989": file("CHEMBL2096989"),
        "FDA_DRUGBANK": file("FDA_DrugBank_cleaned"),
        "EXPERIMENTAL_ZIAI": file("Experimental_ziai_cleaned"),
        "NPASS": file("NPASS_cleaned"),
        "PLANTS_CMAUP": file("Plants_CMAUP_cleaned"),
        "GLAXO_ALERTS": file("glaxo_structural_alerts"),
        "RECEPTOR_1O86": raw / "1o86.pdb",
    }


INPUTS = input_files()


def verify_inputs(keys: Optional[Iterable[str]] = None) -> bool:
    keys = list(keys or INPUTS)
    ok = True
    for key in keys:
        path = INPUTS[key]
        state = "OK" if path.exists() else "MISSING"
        print(f"{key:20} {state:7} {path}")
        ok &= path.exists()
    return ok


def read_table(path, **kwargs):
    """Read CSV, TSV, and Excel files safely."""
    from pathlib import Path
    import pandas as pd
    import csv

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, **kwargs)

    if suffix in {".tsv", ".tab"}:
        kwargs.setdefault("sep", "\t")
        kwargs.setdefault("low_memory", False)
        return pd.read_csv(path, **kwargs)

    # Respect an explicitly supplied delimiter.
    if "sep" not in kwargs:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(20000)

        try:
            delimiter = csv.Sniffer().sniff(
                sample,
                delimiters=";,\t"
            ).delimiter
        except csv.Error:
            delimiter = ";"

        kwargs["sep"] = delimiter

    kwargs.setdefault("low_memory", False)

    # Pass sep only through kwargs, never a second time.
    return pd.read_csv(path, **kwargs)


def normalise_column(column: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(column).strip().lower())


def locate_column(df: pd.DataFrame, aliases: Iterable[str], required: bool = True) -> Optional[str]:
    lookup = {normalise_column(c): c for c in df.columns}
    for alias in aliases:
        if normalise_column(alias) in lookup:
            return lookup[normalise_column(alias)]
    if required:
        raise KeyError(f"Expected one of {list(aliases)}; found {list(df.columns)}")
    return None


def canonical_smiles(smiles: object) -> Optional[str]:
    """Keep the largest neutral organic component; return None when invalid."""
    from rdkit import Chem
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError:
        rdMolStandardize = None
    mol = Chem.MolFromSmiles(str(smiles)) if pd.notna(smiles) else None
    if mol is None:
        return None
    if rdMolStandardize is not None:
        try:
            mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
            mol = rdMolStandardize.Uncharger().uncharge(mol)
        except Exception:
            pass
    return Chem.MolToSmiles(mol, canonical=True)


def _unit_to_nm_factor(unit: object) -> float:
    value = str(unit).strip().lower().replace("μ", "u").replace("µ", "u")
    return {"pm": 1e-3, "nm": 1.0, "um": 1e3, "mm": 1e6, "m": 1e9}.get(value, np.nan)


def _clean_text(series: pd.Series) -> pd.Series:
    """Strip whitespace and the wrapping quotes found in some ChEMBL exports ('IC50', '=')."""
    return series.astype("string").str.strip().str.strip("'\"").str.strip()


# SMARTS for ACE-inhibitor prodrug chemotypes (inactive until hydrolysed in vivo).
ACE_PRODRUG_SMARTS = {
    # enalapril, ramipril, perindopril, quinapril, benazepril, trandolapril, moexipril, ... (ethyl ester of the N-carboxyalkyl group)
    "carboxylate_ester_prodrug": "[CX4][OX2]C(=O)[CX4;H1]([#6])[NX3;H1][CX4]C(=O)[NX3]",
    # zofenopril, alacepril (S-acyl captopril-type thioesters)
    "thioester_prodrug": "O=C([#6])[SX2][CX4][CX4]C(=O)[NX3]",
    # fosinopril (acyloxyalkyl phosphinate ester)
    "acyloxyalkyl_phosphinate_prodrug": "O=C[OX2][CX4][OX2]P(=O)",
}
_PRODRUG_PATTERNS: dict = {}


def ace_prodrug_reason(smiles: object) -> str:
    """Return the matched prodrug chemotype(s) (';'-joined), or '' when none match."""
    from rdkit import Chem
    if not _PRODRUG_PATTERNS:
        _PRODRUG_PATTERNS.update({k: Chem.MolFromSmarts(v) for k, v in ACE_PRODRUG_SMARTS.items()})
    mol = Chem.MolFromSmiles(str(smiles)) if pd.notna(smiles) else None
    if mol is None:
        return ""
    return ";".join(k for k, patt in _PRODRUG_PATTERNS.items() if mol.HasSubstructMatch(patt))


def split_ace_prodrugs(df: pd.DataFrame, smiles_col: str = "canonical_smiles") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (kept, removed); `removed` carries an `ace_prodrug_reason` column for review."""
    reasons = df[smiles_col].map(ace_prodrug_reason)
    flagged = reasons.ne("")
    removed = df.loc[flagged].copy()
    removed["ace_prodrug_reason"] = reasons[flagged]
    return df.loc[~flagged].copy(), removed


# --- Zinc-binding-group (ZBG) annotation ------------------------------------------------------------------------------
# ACE is a zinc metallopeptidase and the main inhibitor classes carry a group that can coordinate the catalytic Zn.
# These flags are an interpretable annotation, NOT a filter or an activity prediction (some natural-product inhibitors
# may not bind through the zinc at all).
ZBG_SMARTS = {
    "carboxylic_acid": "[CX3](=O)[OX2H1,OX1-]",
    "thiol": "[#6][SX2H1,SX1-]",
    "phosphorus_acid": "[PX4](=O)[OX2H1,OX1-]",
    "hydroxamic_acid": "[CX3](=O)[NX3][OX2H1,OX1-]",
    "tetrazole": "[$(c1nn[nH]n1),$(c1nnn[nH]1),$(c1nnn[n-]1)]",
}
ZBG_WEAK_SMARTS = {"catechol": "c([OX2H1,OX1-])c[OX2H1,OX1-]"}   # possible chelator; weaker evidence
_ZBG_PATTERNS: dict = {}


def zinc_binding_groups(smiles: object) -> tuple[str, str]:
    """Return (strong ZBG classes, weak chelator classes), each ';'-joined ('' when none)."""
    from rdkit import Chem
    if not _ZBG_PATTERNS:
        _ZBG_PATTERNS["strong"] = {k: Chem.MolFromSmarts(v) for k, v in ZBG_SMARTS.items()}
        _ZBG_PATTERNS["weak"] = {k: Chem.MolFromSmarts(v) for k, v in ZBG_WEAK_SMARTS.items()}
    mol = Chem.MolFromSmiles(str(smiles)) if pd.notna(smiles) else None
    if mol is None:
        return "", ""
    strong = ";".join(k for k, p in _ZBG_PATTERNS["strong"].items() if mol.HasSubstructMatch(p))
    weak = ";".join(k for k, p in _ZBG_PATTERNS["weak"].items() if mol.HasSubstructMatch(p))
    return strong, weak


def annotate_zbg(df: pd.DataFrame, smiles_col: str = "canonical_smiles") -> pd.DataFrame:
    """Add zbg_classes, weak_chelator_classes, has_zbg columns (one RDKit parse per unique SMILES)."""
    out = df.copy()
    lookup = {smi: zinc_binding_groups(smi) for smi in out[smiles_col].dropna().unique()}
    out["zbg_classes"] = out[smiles_col].map(lambda x: lookup.get(x, ("", ""))[0])
    out["weak_chelator_classes"] = out[smiles_col].map(lambda x: lookup.get(x, ("", ""))[1])
    out["has_zbg"] = out["zbg_classes"].ne("")
    return out


def _biochemical_mask(work: pd.DataFrame) -> tuple[pd.Series, str, dict]:
    """Select biochemical / binding assays. Tries assay_type, then BAO format, then description keywords."""
    col = locate_column(work, ["assay_type"], required=False)
    if col is not None:
        values = _clean_text(work[col]).str.upper()
        return values.isin(["B", "BINDING"]), f"assay_type == B (Binding) [column '{col}']", values.value_counts(dropna=False).to_dict()
    col = locate_column(work, ["bao_label", "bao_format", "assay_format", "bao_endpoint"], required=False)
    if col is not None:
        values = _clean_text(work[col]).fillna("")
        return values.str.contains("single protein", case=False), f"BAO format contains 'single protein' [column '{col}']", values.value_counts(dropna=False).to_dict()
    col = locate_column(work, ["assay_description", "description"], required=False)
    if col is not None:
        text = _clean_text(work[col]).fillna("")
        non_biochem = text.str.contains(
            r"in vivo|ex vivo|plasma|serum|blood pressure|hypertensi|antihypertensive|\bcells?\b|tissue|aorta|isolated|ileum|urine|administ|\boral|\bi\.v\.|\bp\.o\.|pressor",
            case=False, regex=True)
        print("WARNING: no assay_type/BAO column found; biochemical filter falls back to assay-description keywords. "
              "Re-export from ChEMBL with 'assay_type' for a reliable filter.")
        return ~non_biochem, f"assay_description keyword heuristic [column '{col}']", {"kept": int((~non_biochem).sum()), "dropped": int(non_biochem.sum())}
    raise KeyError("BIOCHEMICAL_ONLY is on, but the ChEMBL table has no assay_type, BAO or assay_description column. "
                   f"Columns: {list(work.columns)}. Add one to the export, or call with biochemical_only=False.")


def prepare_chembl_activity(
    path: Path,
    cutoff_nm: float = IC50_CUTOFF_NM,
    dataset_name: str = "dataset",
    biochemical_only: bool = BIOCHEMICAL_ONLY,
    censored_inactive_min_nm: Optional[float] = CENSORED_INACTIVE_MIN_NM,
    exclude_prodrugs: bool = EXCLUDE_PRODRUGS,
    return_diagnostics: bool = False,
    verbose: bool = True,
):
    """Standardise a ChEMBL IC50 export into one row per compound.

    Rules (all documented in the returned diagnostics):
    * IC50 records only; text columns are de-quoted (ChEMBL exports often wrap values in quotes).
    * ``biochemical_only``: keep assay_type "B" (binding / biochemical); fall back to BAO format or
      description keywords only if that column is absent.
    * Exact ("=") records give the measured IC50 (pChEMBL when available, else standard value x units).
    * Censored ">" / ">=" records with value >= ``censored_inactive_min_nm`` are kept as *inactive*
      (``ic50_censored`` = True, ``pchembl`` = NaN so they never enter regression). "<" records and
      lower ">" bounds are dropped. If a compound also has exact records, the exact median is used.
    * ``exclude_prodrugs``: drop ester / thioester / phosphinate-ester ACE-inhibitor prodrugs.
    Output columns: canonical_smiles, ic50_nM, pchembl, n_measurements, ic50_censored, active_ic50, scaffold.
    """
    raw = read_table(path)
    raw.columns = raw.columns.astype(str).str.strip()
    smiles_col = locate_column(raw, ["canonical_smiles", "smiles", "cleaned_smiles"])
    type_col = locate_column(raw, ["standard_type", "type"], required=False)
    relation_col = locate_column(raw, ["standard_relation", "relation"], required=False)
    pchembl_col = locate_column(raw, ["pchembl_value", "pchembl", "pchemblvalue"], required=False)
    value_col = locate_column(raw, ["standard_value", "value", "ic50", "ic50_nm"], required=False)
    units_col = locate_column(raw, ["standard_units", "units", "unit"], required=False)
    if pchembl_col is None and value_col is None:
        raise KeyError("The ChEMBL table needs pChEMBL or a numeric IC50/standard value column.")

    diag: dict = {"dataset": dataset_name, "raw_rows": len(raw)}
    work = raw.copy()
    if type_col is not None:
        work = work[_clean_text(work[type_col]).str.upper().eq("IC50").fillna(False)].copy()
    diag["rows_after_ic50_filter"] = len(work)

    if biochemical_only:
        mask, method, breakdown = _biochemical_mask(work)
        diag["biochemical_filter"] = method
        diag["assay_breakdown_before_filter"] = breakdown
        work = work[mask.fillna(False)].copy()
    diag["rows_after_assay_filter"] = len(work)

    relation = _clean_text(work[relation_col]) if relation_col is not None else pd.Series("=", index=work.index, dtype="string")
    exact = relation.eq("=").fillna(False)
    censored_gt = relation.isin([">", ">="]).fillna(False) & (censored_inactive_min_nm is not None)

    if value_col is not None and units_col is not None:
        std_nm = pd.to_numeric(work[value_col], errors="coerce") * _clean_text(work[units_col]).map(_unit_to_nm_factor).astype(float)
    elif value_col is not None:
        std_nm = pd.to_numeric(work[value_col], errors="coerce")  # assume nM
    else:
        std_nm = pd.Series(np.nan, index=work.index)
    pchembl = pd.to_numeric(work[pchembl_col], errors="coerce") if pchembl_col is not None else pd.Series(np.nan, index=work.index)
    ic50 = pd.Series(np.nan, index=work.index, dtype=float)
    has_p = exact & pchembl.notna()
    ic50[has_p] = np.power(10.0, 9.0 - pchembl[has_p])
    use_std = (exact & ~has_p) | censored_gt
    ic50[use_std] = std_nm[use_std]

    work["ic50_nM"] = ic50
    work["ic50_censored"] = censored_gt.to_numpy()
    work["canonical_smiles"] = work[smiles_col].map(canonical_smiles)
    keep_censored = work["ic50_censored"] & (work["ic50_nM"] >= (censored_inactive_min_nm or np.inf))
    diag["exact_rows"] = int(exact.sum())
    diag["censored_gt_rows_total"] = int(censored_gt.sum())
    diag["censored_gt_rows_kept_as_inactive"] = int(keep_censored.sum())
    diag["rows_dropped_other_relations"] = int((~exact & ~censored_gt).sum())

    valid = (
        work["canonical_smiles"].notna() & work["ic50_nM"].notna() & np.isfinite(work["ic50_nM"])
        & work["ic50_nM"].gt(0) & work["ic50_nM"].le(1e15)
        & ((~work["ic50_censored"]) | keep_censored)
    )
    work = work.loc[valid].copy()
    diag["valid_measurement_rows"] = len(work)
    if work.empty:
        raise ValueError(f"{dataset_name}: no valid records remain after filtering "
                         f"(IC50 rows {diag['rows_after_ic50_filter']}, after assay filter {diag['rows_after_assay_filter']}). "
                         "Check the assay filter and column contents.")

    exact_rows = work[~work["ic50_censored"]]
    cens_rows = work[work["ic50_censored"]]
    out_exact = exact_rows.groupby("canonical_smiles", as_index=False).agg(
        ic50_nM=("ic50_nM", "median"), n_measurements=("ic50_nM", "size"))
    out_exact["ic50_censored"] = False
    out_cens = cens_rows.groupby("canonical_smiles", as_index=False).agg(
        ic50_nM=("ic50_nM", "max"), n_measurements=("ic50_nM", "size"))
    out_cens = out_cens[~out_cens["canonical_smiles"].isin(out_exact["canonical_smiles"])].copy()
    out_cens["ic50_censored"] = True
    out = pd.concat([out_exact, out_cens], ignore_index=True)
    out["pchembl"] = np.where(out["ic50_censored"], np.nan, 9.0 - np.log10(out["ic50_nM"]))
    out["active_ic50"] = (out["ic50_nM"] <= cutoff_nm).astype(int)
    diag["unique_compounds_before_prodrug_filter"] = len(out)

    removed = pd.DataFrame()
    if exclude_prodrugs:
        out, removed = split_ace_prodrugs(out)
    diag["prodrugs_removed"] = len(removed)
    diag["prodrugs_removed_table"] = removed.reset_index(drop=True)
    out["scaffold"] = out["canonical_smiles"].map(murcko_scaffold)
    out = out.reset_index(drop=True)[["canonical_smiles", "ic50_nM", "pchembl", "n_measurements", "ic50_censored", "active_ic50", "scaffold"]]

    diag["unique_compounds"] = len(out)
    diag["exact_compounds"] = int((~out["ic50_censored"]).sum())
    diag["censored_inactive_compounds"] = int(out["ic50_censored"].sum())
    diag["active_compounds"] = int(out["active_ic50"].sum())
    diag["inactive_compounds"] = int((out["active_ic50"] == 0).sum())
    if verbose:
        print(f"\n{dataset_name} preparation diagnostics")
        for key, value in diag.items():
            if key not in {"dataset", "prodrugs_removed_table"}:
                print(f"  {key}: {value}")
        print(f"  active prevalence: {out['active_ic50'].mean():.1%}")
    return (out, diag) if return_diagnostics else out


def murcko_scaffold(smiles: str) -> str:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol else ""
    return scaffold or smiles


def morgan_matrix(smiles: Iterable[str], radius: int = FP_RADIUS, n_bits: int = FP_SIZE) -> np.ndarray:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smi}")
        arr = np.zeros((n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), arr)
        rows.append(arr)
    return np.asarray(rows, dtype=np.uint8)


def tanimoto_kernel(X1, X2):
    X1, X2 = np.asarray(X1, dtype=float), np.asarray(X2, dtype=float)
    intersection = X1 @ X2.T
    union = X1.sum(axis=1)[:, None] + X2.sum(axis=1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


class TanimotoKNN(BaseEstimator, RegressorMixin):
    """Fingerprint k-NN regressor with a genuine Tanimoto similarity metric."""
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_ = (np.asarray(X) > 0).astype(float)
        self.y_ = np.asarray(y, dtype=float)
        return self

    def predict(self, X):
        similarities = tanimoto_kernel((np.asarray(X) > 0).astype(float), self.X_)
        n = min(self.n_neighbors, len(self.y_))
        nearest = np.argsort(-similarities, axis=1)[:, :n]
        return self.y_[nearest].mean(axis=1)


def max_tanimoto_similarity(query: np.ndarray, reference: np.ndarray, chunk: int = 2_000) -> np.ndarray:
    """Return each query's maximum training-set similarity without a giant matrix."""
    values = []
    for start in range(0, len(query), chunk):
        values.extend(tanimoto_kernel(query[start:start + chunk], reference).max(axis=1))
    return np.asarray(values)


def prepare_screening_library(path: Path, source_name: str) -> pd.DataFrame:
    raw = read_table(path)
    # DrugBank's cleaned export can include non-approved records.  Preserve only
    # the requested FDA-approved subset when the status column is available.
    if source_name == "FDA_DRUGBANK":
        approval_col = locate_column(raw, ["drug_groups", "drug_group", "groups"], required=False)
        if approval_col is not None:
            raw = raw[raw[approval_col].fillna("").astype(str).str.contains(r"\bapproved\b", case=False, regex=True)].copy()
    smiles_col = locate_column(raw, ["cleaned_smiles", "canonical_smiles", "smiles"])
    name_col = locate_column(raw, ["name", "compound", "pref_name", "drugbankid", "np_id", "id", "cid"], required=False)
    out = pd.DataFrame({
        "source": source_name,
        "source_id": raw[name_col].astype(str) if name_col else raw.index.astype(str),
        "input_smiles": raw[smiles_col].astype(str),
    })
    out["canonical_smiles"] = out["input_smiles"].map(canonical_smiles)
    out = out.dropna(subset=["canonical_smiles"]).drop_duplicates(["source", "canonical_smiles"]).reset_index(drop=True)
    return out


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def write_ligand_pdb(pdbqt_path: Path, out_pdb: Path, resname: str = "LIG") -> int:
    """Write heavy atoms from the first PDBQT model in standard PDB columns."""
    element_map = {"A": "C", "C": "C", "N": "N", "NA": "N", "OA": "O", "O": "O", "S": "S", "SA": "S", "P": "P", "F": "F", "CL": "Cl", "BR": "Br", "I": "I", "ZN": "Zn"}
    lines, counters, in_model = [], {}, False
    for line in Path(pdbqt_path).read_text(errors="replace").splitlines():
        if line.startswith("MODEL"):
            if in_model:
                break
            in_model = True
            continue
        if line.startswith("ENDMDL") and in_model:
            break
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            x, y, z = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        atom_type = line[77:79].strip().upper()
        if atom_type in {"H", "HD", "HS"}:
            continue
        element = element_map.get(atom_type, atom_type[:1] or "C")
        counters[element] = counters.get(element, 0) + 1
        atom_name = f"{element.upper()}{counters[element]}"[:4]
        lines.append(f"HETATM{len(lines)+1:5d} {atom_name:<4} {resname:>3} Z{1001:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2}")
    Path(out_pdb).write_text("\n".join(lines) + "\nEND\n")
    return len(lines)
