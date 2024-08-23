import logging
import os
from typing import List
from tempfile import NamedTemporaryFile

import Bio.PDB
from Bio.PDB.mmtf import MMTFParser
import numpy as np
import torch
from scipy.spatial import distance_matrix

from progres.chainsaw.src.constants import _3to1, STRIDE_EXE
from progres.chainsaw.src.utils.cif2pdb import cif2pdb
from progres.chainsaw.src.utils.secondary_structure import calculate_ss, make_ss_matrix

LOG = logging.getLogger(__name__)


def get_model_structure(structure_path, fileformat="pdb") -> Bio.PDB.Structure:
    """
    Returns the Bio.PDB.Structure object for a given PDB, mmCIF or MMTF file
    """
    structure_id = os.path.split(structure_path)[-1].split('.')[0]
    if fileformat == "pdb":
        structure = Bio.PDB.PDBParser().get_structure(structure_id, structure_path)
    elif fileformat == "mmcif":
        structure = Bio.PDB.MMCIFParser().get_structure(structure_id, structure_path)
    elif fileformat == "mmtf":
        structure = MMTFParser.get_structure(structure_path)
    elif fileformat == "coords":
        raise ValueError("Coordinate file format not compatible with Chainsaw")
    else:
        raise ValueError(f"Unrecognized file format: {fileformat}")
    model = structure[0]
    return model


class Residue:
    def __init__(self, index: int, res_label: str, aa: str):
        self.index = int(index)
        self.res_label = str(res_label)
        self.aa = str(aa)

def get_model_structure_residues(structure_model: Bio.PDB.Structure, chain='A') -> List[Residue]:
    """
    Returns a list of residues from a given PDB or MMCIF structure
    """
    residues = []
    res_index = 1
    for biores in structure_model[chain].child_list:
        res_num = biores.id[1]
        res_ins = biores.id[2]
        res_label = str(res_num)
        if res_ins != ' ':
            res_label += str(res_ins)
        
        aa3 = biores.get_resname()
        if aa3 not in _3to1:
            continue

        aa = _3to1[aa3]
        res = Residue(res_index, res_label, aa)
        residues.append(res)
        
        # increment the residue index after we have filtered out non-standard amino acids
        res_index += 1
    
    return residues


def inference_time_create_features(file_path, feature_config, chain="A", *,
                                   model_structure: Bio.PDB.Structure=None,
                                   stride_path=STRIDE_EXE, fileformat="pdb",
                                   ):
    if fileformat == "pdb":
        pdb_path = file_path
    else:
        temp_pdb_file = NamedTemporaryFile()
        pdb_path = temp_pdb_file.name
        cif2pdb(file_path, pdb_path, fileformat)

    if not model_structure:
        model_structure = get_model_structure(pdb_path)

    dist_matrix = get_distance(model_structure, chain=chain)
    temp_ss_file = NamedTemporaryFile()
    ss_filepath = temp_ss_file.name
    calculate_ss(pdb_path, chain, stride_path, ssfile=ss_filepath)
    helix, strand = make_ss_matrix(ss_filepath, nres=dist_matrix.shape[-1])
    if feature_config['ss_bounds']:
        end_res_val = -1 if feature_config['negative_ss_end'] else 1
        helix_boundaries = make_boundary_matrix(helix, end_res_val=end_res_val)
        strand_boundaries = make_boundary_matrix(strand, end_res_val=end_res_val)
    temp_ss_file.close()
    if fileformat != "pdb":
        temp_pdb_file.close()
    LOG.info(f"Distance matrix shape: {dist_matrix.shape}, SS matrix shape: {helix.shape}")
    if feature_config['ss_bounds']:
        if feature_config['same_channel_boundaries_and_ss']:
            helix_boundaries[helix == 1] = 1
            strand_boundaries[strand == 1] = 1
            stacked_features = np.stack((dist_matrix, helix_boundaries, strand_boundaries), axis=0)
        else:
            stacked_features = np.stack((dist_matrix, helix, strand, helix_boundaries, strand_boundaries), axis=0)
    else:
        stacked_features = np.stack((dist_matrix, helix, strand), axis=0)
    stacked_features = stacked_features[None] # add batch dimension
    return torch.Tensor(stacked_features)


def distance_matrix(x):
    """Compute the distance matrix.

    Returns the matrix of all pair-wise distances.

    Parameters
    ----------
    x : (M, K) array_like
        Matrix of M vectors in K dimensions.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.
    Returns
    -------
    result : (M, M) ndarray
        Matrix containing the distance from every vector in `x` to every vector
        in `x`.
    """
    x1 = x[:, np.newaxis, :]  # Expand x to 3D for broadcasting, shape (M, 1, K)
    x2 = x[np.newaxis, :, :]  # Expand x to 3D for broadcasting, shape (1, M, K)

    distance_matrix = np.sum(np.abs(x2 - x1) ** 2, axis=-1) ** 0.5

    return distance_matrix.astype(np.float16)


def get_distance(structure_model: Bio.PDB.Structure, chain='A'):
    alpha_coords = np.array([residue['CA'].get_coord() for residue in \
                             structure_model[chain].get_residues() if Bio.PDB.is_aa(residue) and \
                             'CA' in residue and residue.get_resname() in _3to1], dtype=np.float16)
    x = distance_matrix(alpha_coords)
    return x


def make_boundary_matrix(ss, end_res_val=1):
    """
    makes a matrix where  the boundary residues
    of the sec struct component are 1
    """
    ss_lines = np.zeros_like(ss)
    diag = np.diag(ss)
    if max(diag) == 0:
        return ss_lines
    padded_diag = np.zeros(len(diag) + 2)
    padded_diag[1:-1] = diag
    diff_before = diag - padded_diag[:-2]
    diff_after = diag - padded_diag[2:]
    start_res = np.where(diff_before == 1)[0]
    end_res = np.where(diff_after == 1)[0]
    ss_lines[start_res, :] = 1
    ss_lines[:, start_res] = 1
    ss_lines[end_res, :] = end_res_val
    ss_lines[:, end_res] = end_res_val
    return ss_lines
