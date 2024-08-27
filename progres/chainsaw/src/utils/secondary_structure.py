"""
Created by Jude Wells 2023-04-20
Objective is to create a secondary structure matrix for each protein
1) renumbers the pdb file (1-indexed)
2) runs stride to get secondary structure file
3) parses the secondary structure file to create a matrix of secondary structure

# should take around 3 minutes for 1000 structures
"""

import os
import re
import subprocess
import sys

import numpy as np

from progres.chainsaw.src.constants import REPO_ROOT
import logging
LOG = logging.getLogger(__name__)


def calculate_ss(pdbfile, chain, stride_path, ssfile='pdb_ss'):
    assert os.path.exists(pdbfile)
    with open(ssfile, 'w') as ssout_file:
        chain_arg = '-r' + chain
        args = [stride_path, pdbfile, chain_arg.rstrip()]
        LOG.info(f"Running command: {' '.join(args)}")
        try:
            subprocess.run(args, 
                        stdout=ssout_file, 
                        stderr=subprocess.DEVNULL, 
                        check=True)
        except subprocess.CalledProcessError as e:
            LOG.warning(f"Stride failed on {pdbfile}, creating empty file")
            pass


def make_ss_matrix(ss_path, nres):
    # create matrices for helix and strad residues where entry ij = 1 if i and j are in the same helix or strand
    with open(ss_path) as f:
        lines = f.readlines()
    type_set = set()
    helix  = np.zeros([nres, nres], dtype=np.float32)
    strand = np.zeros([nres, nres], dtype=np.float32)
    for line in lines:
        if line.startswith('LOC'):
            start = int(re.sub('\D', '',line[22:28].strip()))
            end = int(re.sub('\D', '',line[40:46].strip()))
            type = line[5:17].strip()
            type_set.add(type)
            if type in ['AlphaHelix', '310Helix']:
                helix[start-1:end, start-1:end] = 1
            elif type == 'Strand':
                strand[start-1:end, start-1:end] = 1
        elif line.startswith('ASG'):
            break
    return helix, strand


def renum_pdb_file(pdb_path, output_pdb_path):
    pdb_reres_path = REPO_ROOT / 'src/utils/pdb_reres.py'
    with open(output_pdb_path, "w") as output_file:
        subprocess.run([sys.executable, str(pdb_reres_path), pdb_path],
                       stdout=output_file,
                       check=True,
                       text=True)
 

def main(chain_ids, pdb_dir, feature_dir, stride_path, reres_path, savedir, job_index=0):
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(os.path.join(savedir, '2d_features'), exist_ok=True)
    for chain_id in chain_ids:
        try:
            pdb_path = os.path.join(pdb_dir, chain_id + '.pdb')
            if os.path.exists(pdb_path):
                features = np.load(os.path.join(feature_dir, chain_id + '.npz'))['arr_0']
                nres = features.shape[-1]
                LOG.info("Processing", pdb_path)
                chain = chain_id[4]
                output_pdb_path = os.path.join(savedir, f"{job_index}.pdb") # this gets overwritten to save memory
                file_nres = renum_pdb_file(pdb_path, reres_path, output_pdb_path)
                if nres != file_nres:
                    with open(os.path.join(savedir, 'error.txt'), 'a') as f:
                        msg = f' residue number mismatch (from features) {nres}, (from pdb file) {file_nres}'
                        f.write(chain_id + msg + '\n')
                ss_filepath = os.path.join(savedir, f'pdb_ss{job_index}.txt') # this gets overwritten to save memory
                calculate_ss(output_pdb_path, chain, stride_path, ssfile=ss_filepath)
                helix, strand = make_ss_matrix(ss_filepath, nres=nres)
                np.savez_compressed(
                    os.path.join(*[savedir, '2d_features', chain_id + '.npz']), 
                    np.stack((features, helix, strand), axis=0))
        except Exception as e:
            with open(os.path.join(savedir, 'error.txt'), 'a') as f:
                f.write(chain_id + str(e) + '\n')
