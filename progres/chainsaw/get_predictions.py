"""
Script for running Chainsaw

Created by: Jude Wells 2023-04-19

Modified for Progres
"""

import hashlib
import logging
import os
import sys
import time
from typing import List

from .src import constants, featurisers
from .src.domain_assignment.util import convert_domain_dict_strings
from .src.factories import pairwise_predictor
from .src.models.results import PredictionResult
from .src.utils import common as common_utils

LOG = logging.getLogger(__name__)

def setup_logging(loglevel):
    # log all messages to stderr so results can be sent to stdout
    logging.basicConfig(level=loglevel,
                    stream=sys.stderr,
                    format='%(asctime)s | %(levelname)s | %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

def load_model(*,
               model_dir: str,
               remove_disordered_domain_threshold: float = 0.35,
               min_ss_components: int = 2,
               min_domain_length: int = 30,
               post_process_domains: bool = True,
               device: str = "cpu"):
    config = common_utils.load_json(os.path.join(model_dir, "config.json"))
    feature_config = common_utils.load_json(os.path.join(model_dir, "feature_config.json"))
    config["learner"]["remove_disordered_domain_threshold"] = remove_disordered_domain_threshold
    config["learner"]["post_process_domains"] = post_process_domains
    config["learner"]["min_ss_components"] = min_ss_components
    config["learner"]["min_domain_length"] = min_domain_length
    config["learner"]["dist_transform_type"] = config["data"].get("dist_transform", 'min_replace_inverse')
    config["learner"]["distance_denominator"] = config["data"].get("distance_denominator", None)
    learner = pairwise_predictor(config["learner"], output_dir=model_dir, device=device)
    learner.feature_config = feature_config
    learner.load_checkpoints()
    learner.eval()
    return learner

def predict(model, pdb_path, renumber_pdbs=True, pdbchain=None,
            fileformat="pdb") -> List[PredictionResult]:
    """
    Makes the prediction and returns a list of PredictionResult objects
    """
    start = time.time()

    # get model structure metadata
    model_structure = featurisers.get_model_structure(pdb_path, fileformat)

    if pdbchain is None:
        LOG.warning(f"No chain specified for {pdb_path}, using first chain")
        # get all the chain ids from the model structure
        all_chain_ids = [c.id for c in model_structure.get_chains()]
        # take the first chain id
        pdbchain = all_chain_ids[0]

    model_residues = featurisers.get_model_structure_residues(model_structure, chain=pdbchain)
    model_res_label_by_index = { int(r.index): str(r.res_label) for r in model_residues}
    model_structure_seq = "".join([r.aa for r in model_residues])
    model_structure_md5 = hashlib.md5(model_structure_seq.encode('utf-8')).hexdigest()

    x = featurisers.inference_time_create_features(pdb_path,
                                                    feature_config=model.feature_config,
                                                    chain=pdbchain,
                                                    renumber_pdbs=renumber_pdbs,
                                                    model_structure=model_structure,
                                                    fileformat=fileformat,
                                                   )

    A_hat, domain_dict, confidence = model.predict(x)
    # Convert 0-indexed to 1-indexed to match AlphaFold indexing:
    domain_dict = [{k: [r + 1 for r in v] for k, v in d.items()} for d in domain_dict]
    names_str, bounds_str = convert_domain_dict_strings(domain_dict[0])
    confidence = confidence[0]

    if names_str == "":
        names = bounds = ()
    else:
        names = names_str.split('|')
        bounds = bounds_str.split('|')

    assert len(names) == len(bounds)

    class Seg:
        def __init__(self, domain_id: str, start_index: int, end_index: int):
            self.domain_id = domain_id
            self.start_index = int(start_index)
            self.end_index = int(end_index)
        
        def res_label_of_index(self, index: int):
            if index not in model_res_label_by_index:
                raise ValueError(f"Index {index} not in model_res_label_by_index ({model_res_label_by_index})")
            return model_res_label_by_index[int(index)]

        @property
        def start_label(self):
            return self.res_label_of_index(self.start_index)
        
        @property
        def end_label(self):
            return self.res_label_of_index(self.end_index)

    class Dom:
        def __init__(self, domain_id, segs: List[Seg] = None):
            self.domain_id = domain_id
            if segs is None:
                segs = []
            self.segs = segs

        def add_seg(self, seg: Seg):
            self.segs.append(seg)

    # gather choppings into segments in domains
    domains_by_domain_id = {}
    for domain_id, chopping_by_index in zip(names, bounds):
        if domain_id not in domains_by_domain_id:
            domains_by_domain_id[domain_id] = Dom(domain_id)
        start_index, end_index = chopping_by_index.split('-')
        seg = Seg(domain_id, start_index, end_index)
        domains_by_domain_id[domain_id].add_seg(seg)

    # sort domain choppings by the start residue in first segment
    domains = sorted(domains_by_domain_id.values(), key=lambda dom: dom.segs[0].start_index)

    # collect domain choppings as strings
    domain_choppings = []
    for dom in domains:
        # convert segments to strings
        segs_str = [f"{seg.start_label}-{seg.end_label}" for seg in dom.segs]
        segs_index_str = [f"{seg.start_index}-{seg.end_index}" for seg in dom.segs]
        LOG.info(f"Segments (index to label): {segs_index_str} -> {segs_str}")
        # join discontinuous segs with '_' 
        domain_choppings.append('_'.join(segs_str))

    # join domains with ','
    chopping_str = ','.join(domain_choppings)

    num_domains = len(domain_choppings)
    if num_domains == 0:
        chopping_str = None
    runtime = round(time.time() - start, 3)
    result = PredictionResult(
        pdb_path=pdb_path,
        chain_id="A", # Placeholder
        sequence_md5=model_structure_md5,
        nres=len(model_structure_seq),
        ndom=num_domains,
        chopping=chopping_str,
        confidence=confidence,
        time_sec=runtime,
    )

    LOG.info(f"Runtime: {round(runtime, 3)}s")
    return result

def predict_domains(structure_file, fileformat=None, device="cpu", pdbchain=None):
    loglevel = os.environ.get("LOGLEVEL", "ERROR").upper() # Change to "INFO" to see more
    setup_logging(loglevel)
    if fileformat is None:
        fileformat = "pdb"
        file_ext = os.path.splitext(structure_file)[1].lower()
        if file_ext == ".cif" or file_ext == ".mmcif":
            fileformat = "mmcif"
        elif file_ext == ".mmtf":
            fileformat = "mmtf"
    model = load_model(
        model_dir=os.path.join(constants.REPO_ROOT, "model_v3"),
        device=device,
    )
    result = predict(model, structure_file, renumber_pdbs=False,
                     pdbchain=pdbchain, fileformat=fileformat)
    return result.chopping