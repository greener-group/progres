"""There are three configurable things: predictors/models, data, training/evaluation.

Only first two require factories.
"""
import os

import pandas as pd
import torch

import logging

from .domain_chop import PairwiseDomainPredictor
from .models.rosetta import trRosettaNetwork
from .domain_assignment.assigners import SparseLowRank
from .utils import common as common_utils


LOG = logging.getLogger(__name__)


def get_assigner(config):
    assigner_type = config["type"]
    if assigner_type == "sparse_lowrank":
        assigner = SparseLowRank(**config["kwargs"])
    else:
        return ValueError()
    return assigner


def get_model(config):
    model_type = config["type"]
    if model_type == "trrosetta":
        model = trRosettaNetwork(**config["kwargs"])
    else:
        return ValueError()
    return model


def pairwise_predictor(learner_config, force_cpu=False, output_dir=None, device="cpu"):
    model = get_model(learner_config["model"])
    assigner = get_assigner(learner_config["assignment"])
    device = torch.device(device)
    model.to(device)
    kwargs = {k: v for k, v in learner_config.items() if k not in ["model",
                                                                   "assignment",
                                                                   "save_every_epoch",
                                                                   "uncertainty_model"]}
    LOG.info(f"Learner kwargs: {kwargs}")
    return PairwiseDomainPredictor(model, assigner, device, checkpoint_dir=output_dir, **kwargs)


def get_test_ids(label_path, feature_path, csv_path=None):
    ids = [id.split('.')[0] for id in set(os.listdir(label_path)).intersection(set(os.listdir(feature_path)))]
    if csv_path is not None:
        df = pd.read_csv(csv_path)
        ids = sorted(list(set(ids).intersection(set(df.chain_id))))
    return ids


def filter_plddt(df_path, ids, threshold=90):
    df = pd.read_csv(df_path)
    df = df[df.plddt > threshold]
    ids = [i for i in ids if i in df.casp_id.values]
    return ids
