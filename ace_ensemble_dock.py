"""Conformer-ensemble rigid docking followed by flexible local refinement (the protocol validated in notebook 4).

Kept as an importable module so notebooks 4-style redocking, candidate docking (5b) and the screening-power test (7)
all run exactly the same code.
"""
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom
from scipy.spatial.distance import cdist

PROTOCOL_NAME = 'conformer_ensemble_rigid_docking_then_flexible_refinement'


def _lines(path, first_model_only=True):
    out, started = [], False
    for line in Path(path).read_text(errors='replace').splitlines():
        if line.startswith('MODEL'):
            started = True
            continue
        if started and line.startswith('ENDMDL') and first_model_only:
            break
        if line.startswith(('ATOM', 'HETATM')):
            out.append(line)
    return out


def _heavy(line):
    return line[77:79].strip().upper() not in {'H', 'HD', 'HS'}


def _xyz(line):
    return [float(line[30:38]), float(line[38:46]), float(line[46:54])]


def _to_template_order(input_lines, pose_lines, conf_heavy_xyz):
    """Map PDBQT heavy atoms (Meeko order) onto the SMILES atom order via the INPUT conformer coordinates."""
    inp = np.array([_xyz(l) for l in input_lines if _heavy(l)])
    pose = np.array([_xyz(l) for l in pose_lines if _heavy(l)])
    d = cdist(inp, conf_heavy_xyz)
    j = d.argmin(axis=1)
    if d.min(axis=1).max() > 0.05 or len(set(j)) != len(j):
        raise RuntimeError('Atom mapping between Meeko PDBQT and RDKit conformer failed.')
    out = np.empty_like(pose)
    out[j] = pose
    return out


class Reference:
    """Crystal reference for a symmetry-aware heavy-atom RMSD (no superposition)."""

    def __init__(self, template_mol, crystal_mol):
        self.ref_xyz = crystal_mol.GetConformer().GetPositions()
        self.matches = template_mol.GetSubstructMatches(crystal_mol, uniquify=False, useChirality=False, maxMatches=20000)
        if not self.matches:
            raise RuntimeError('Could not map the SMILES-based molecule onto the crystal ligand.')
        probe = np.zeros_like(self.ref_xyz)
        for i, j in enumerate(self.matches[0]):
            probe[j] = self.ref_xyz[i]
        assert self.rmsd(probe) < 1e-6, 'RMSD mapping self-check failed'

    def rmsd(self, pose_xyz_template_order):
        return min(float(np.sqrt(np.mean(np.sum((pose_xyz_template_order[list(m)] - self.ref_xyz) ** 2, axis=1)))) for m in self.matches)


class EnsembleDocker:
    def __init__(self, receptor_pdbqt, box, vina, meeko_ligand, workdir, cpu=2):
        self.receptor, self.box, self.vina, self.meeko = str(receptor_pdbqt), box, vina, meeko_ligand
        self.workdir, self.cpu = Path(workdir), cpu
        self.workdir.mkdir(parents=True, exist_ok=True)

    def _box_args(self, pad=0.0):
        b = self.box
        return ['--center_x', str(b['center_x']), '--center_y', str(b['center_y']), '--center_z', str(b['center_z']),
                '--size_x', str(b['size_x'] + pad), '--size_y', str(b['size_y'] + pad), '--size_z', str(b['size_z'] + pad)]

    def _vina(self, ligand, extra, pad=0.0):
        r = subprocess.run([self.vina, '--receptor', self.receptor, '--ligand', str(ligand), *self._box_args(pad), '--cpu', str(self.cpu), *extra],
                           text=True, capture_output=True)
        if r.returncode:
            raise RuntimeError((r.stderr or r.stdout)[-800:])
        return r.stdout

    def dock(self, smiles, tag, seed, n_conformers=100, rigid_exhaustiveness=4, refine_top_k=20, reference=None, keep_intermediate=False):
        """Return a dict with the best refined score/pose; a 'refined' DataFrame lists the refined conformers.

        Undefined stereocentres are embedded in both configurations (the ensemble contains both), so the reported score is the
        best over the possible stereoisomers.
        """
        template = Chem.MolFromSmiles(smiles)
        if template is None:
            return {'status': 'INVALID_SMILES', 'tag': tag, 'seed': seed}
        mol_h = Chem.AddHs(template)
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = int(seed)
        params.pruneRmsThresh = 0.5
        params.numThreads = self.cpu
        cids = list(rdDistGeom.EmbedMultipleConfs(mol_h, n_conformers, params))
        if not cids:
            params.useRandomCoords = True
            cids = list(rdDistGeom.EmbedMultipleConfs(mol_h, n_conformers, params))
        if not cids:
            return {'status': 'FAILED_EMBED', 'tag': tag, 'seed': seed}
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol_h, maxIters=1000, numThreads=self.cpu)
        except Exception:
            pass
        n_heavy = template.GetNumHeavyAtoms()
        d = self.workdir / f'{tag}_s{seed}'
        d.mkdir(parents=True, exist_ok=True)

        records, store = [], {}
        for cid in cids:
            name = f'c{cid:04d}'
            sdf, flex = d / f'{name}.sdf', d / f'{name}.pdbqt'
            sdf.write_text(Chem.MolToMolBlock(mol_h, confId=cid))
            r = subprocess.run([self.meeko, '-i', str(sdf), '-o', str(flex)], text=True, capture_output=True)
            if r.returncode or not flex.exists():
                continue
            flex_text = flex.read_text().splitlines()
            flex_atoms = _lines(flex, first_model_only=False)
            rigid, out = d / f'{name}_rigid.pdbqt', d / f'{name}_rigid_docked.pdbqt'
            rigid.write_text('\n'.join([l for l in flex_text if l.startswith('REMARK')] + ['ROOT', *flex_atoms, 'ENDROOT', 'TORSDOF 0']) + '\n')
            try:
                self._vina(rigid, ['--exhaustiveness', str(rigid_exhaustiveness), '--num_modes', '3', '--seed', str(seed), '--out', str(out)])
            except RuntimeError:
                continue
            pose_lines = _lines(out)
            score = next((float(x.split()[3]) for x in out.read_text().splitlines() if x.startswith('REMARK VINA RESULT')), np.nan)
            conf_heavy = mol_h.GetConformer(cid).GetPositions()[:n_heavy]
            rec = {'conformer': cid, 'rigid_score': score}
            if reference is not None:
                rec['rigid_rmsd_A'] = reference.rmsd(_to_template_order(flex_atoms, pose_lines, conf_heavy))
            records.append(rec)
            store[cid] = {'flex': flex, 'pose': pose_lines, 'atoms': flex_atoms, 'conf_heavy': conf_heavy}
        if not records:
            shutil.rmtree(d, ignore_errors=True)
            return {'status': 'FAILED_DOCKING', 'tag': tag, 'seed': seed}
        ens = pd.DataFrame(records).sort_values('rigid_score').reset_index(drop=True)

        refined_rows = []
        for row in ens.head(refine_top_k).itertuples():
            o = store[row.conformer]
            it, start_lines = iter(o['pose']), []
            for l in Path(o['flex']).read_text().splitlines():
                if l.startswith(('ATOM', 'HETATM')):
                    p = next(it)
                    l = l[:30] + p[30:54] + l[54:]          # same atoms, rigid-pose coordinates
                start_lines.append(l)
            start = d / f'c{row.conformer:04d}_start.pdbqt'
            start.write_text('\n'.join(start_lines) + '\n')
            refined = d / f'c{row.conformer:04d}_refined.pdbqt'
            try:
                txt = self._vina(start, ['--local_only', '--out', str(refined)], pad=12.0)   # larger box: edge poses may poke out of the docking box
            except RuntimeError:
                continue
            rr = {'conformer': row.conformer, 'rigid_score': row.rigid_score, 'refined_score': float(re.findall(r'Estimated Free Energy of Binding\s*:\s*(-?\d+\.?\d*)', txt)[-1]),
                  'refined_pdbqt': str(refined)}
            if reference is not None:
                rr['refined_rmsd_A'] = reference.rmsd(_to_template_order(o['atoms'], _lines(refined), o['conf_heavy']))
            refined_rows.append(rr)
        if not refined_rows:
            shutil.rmtree(d, ignore_errors=True)
            return {'status': 'FAILED_REFINEMENT', 'tag': tag, 'seed': seed}
        refined_df = pd.DataFrame(refined_rows).sort_values('refined_score').reset_index(drop=True)
        best = refined_df.iloc[0]
        best_pose = self.workdir / f'{tag}_s{seed}_best.pdbqt'
        shutil.copy2(best.refined_pdbqt, best_pose)
        result = {'status': 'DONE', 'tag': tag, 'seed': seed, 'n_conformers': len(cids), 'n_rigid_docked': len(ens), 'best_rigid_score': float(ens.rigid_score.min()),
                  'best_refined_score': float(best.refined_score), 'best_conformer': int(best.conformer), 'best_pose_pdbqt': str(best_pose),
                  'refined': refined_df.drop(columns='refined_pdbqt'), 'all_conformers': ens}
        if reference is not None:
            result['top_rmsd_A'] = float(best.refined_rmsd_A)
            result['best_rigid_rmsd_A'] = float(ens.rigid_rmsd_A.min())
        if not keep_intermediate:
            shutil.rmtree(d, ignore_errors=True)
        return result
