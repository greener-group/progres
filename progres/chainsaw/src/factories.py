"""There are three configurable things: predictors/models, data, training/evaluation.

Only first two require factories.
"""
import os

import torch

import logging

from progres.chainsaw.src.domain_chop import PairwiseDomainPredictor
from progres.chainsaw.src.models.rosetta import trRosettaNetwork
from progres.chainsaw.src.domain_assignment.assigners import SparseLowRank
from progres.chainsaw.src.utils import common as common_utils


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
