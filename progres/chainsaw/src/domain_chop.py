"""Domain predictor classes.
"""
import os
import glob
import numpy as np
from pathlib import Path
import torch
from torch import nn
from progres.chainsaw.src.domain_assignment.util import make_pair_labels, make_domain_mapping_dict

import logging
LOG = logging.getLogger(__name__)


def get_checkpoint_epoch(checkpoint_file):
    return int(os.path.splitext(checkpoint_file)[0].split(".")[-1])


class PairwiseDomainPredictor(nn.Module):

    """Wrapper for a pairwise domain co-membership predictor, adding in domain prediction post-processing."""

    def __init__(
        self,
        model,
        domain_caller,
        device,
        loss="bce",
        x_has_padding_mask=True,
        mask_padding=True,
        n_checkpoints_to_average=1,
        checkpoint_dir=None,
        load_checkpoint_if_exists=False,
        save_val_best=True,
        max_recycles=0,
        post_process_domains=True,
        min_ss_components=2,
        min_domain_length=30,
        remove_disordered_domain_threshold=0,
        trim_each_domain=True,
        dist_transform_type="min_replace_inverse",
        distance_denominator=10,
    ):
        super().__init__()
        self._train_model = model  # we want to keep this hanging around so that optimizer references dont break
        self.model = self._train_model
        self.domain_caller = domain_caller
        self.device = device
        self.x_has_padding_mask = x_has_padding_mask
        self.mask_padding = mask_padding  # if True use padding mask to mask loss
        self.n_checkpoints_to_average = n_checkpoints_to_average
        self.checkpoint_dir = checkpoint_dir
        self._epoch = 0
        self.save_val_best = save_val_best
        self.best_val_metrics = {}
        self.max_recycles = max_recycles
        self.post_process_domains = post_process_domains
        self.remove_disordered_domain_threshold = remove_disordered_domain_threshold
        self.trim_each_domain = trim_each_domain
        self.min_domain_length = min_domain_length
        self.min_ss_components = min_ss_components
        self.dist_transform_type = dist_transform_type
        self.distance_denominator = distance_denominator
        if load_checkpoint_if_exists:
            checkpoint_files = sorted(
                glob.glob(os.path.join(self.checkpoint_dir, "weights*")),
                key=get_checkpoint_epoch,
                reverse=True,
            )
            if len(checkpoint_files) > 0:
                self._epoch = get_checkpoint_epoch(checkpoint_files[0])
                LOG.info(f"Loading saved checkpoint(s) ending at epoch {self._epoch}")
                self.load_checkpoints(average=True)
                self.load_checkpoints()
            else:
                LOG.info("No checkpoints found to load")

        if loss == "bce":
            self.loss_function = nn.BCELoss(reduction="none")
        elif loss == "mse":
            self.loss_function = nn.MSELoss(reduction="none")

    def load_checkpoints(self, average=False, old_style=False):
        if self.n_checkpoints_to_average == 1:
            data_dir = os.getenv(
                "PROGRES_DATA_DIR",
                default=Path(__file__).parent.parent.parent.resolve(),
            )
            weights_file = os.path.join(data_dir, "chainsaw", "model_v3", "weights.pt")
        else:
            # for e.g. resetting training weights for next training epoch after testing with avg
            LOG.info(f"Loading last checkpoint (epoch {self._epoch})")
            weights_file = os.path.join(self.checkpoint_dir, f"weights.{self._epoch}.pt")
        LOG.info(f"Loading weights from: {weights_file}")
        state_dict = torch.load(weights_file, map_location=self.device)
        if old_style:
            self.load_state_dict(state_dict, strict=False)
        else:
            self.model.load_state_dict(state_dict)

    def predict_pairwise(self, x):
        x = x.to(self.device)
        if np.isnan(x.cpu().numpy()).any():
            raise Exception('NAN values in data')
        y_pred = self.model(x).squeeze(1)  # b, L, L
        assert y_pred.ndim == 3
        return y_pred

    def get_mask(self, x):
        """Binary mask 1 for observed, 0 for padding."""
        x = x.to(self.device)
        if self.x_has_padding_mask:
            mask = 1 - x[:, -1]  # b, L, L
        else:
            mask = None
        return mask

    def epoch_start(self):
        self.model = self._train_model
        self.model.train()
        self._epoch += 1

    def test_begin(self):
        if self.n_checkpoints_to_average > 1:
            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, f'weights.{self._epoch}.pt'))
            start_idx = self._epoch - self.n_checkpoints_to_average
            if start_idx >= 2:
                os.remove(os.path.join(self.checkpoint_dir, f"weights.{start_idx-1}.pt"))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, 'weights.pt'))

        if self.n_checkpoints_to_average > 1:
            # self.model.to("cpu")  # free up gpu memory for average model
            self.load_checkpoints(average=True)

        self.model.eval()

    def forward(self, x, y, batch_average=True):
        """A training step."""
        x, y = x.to(self.device), y.to(self.device)
        y_pred = self.predict_pairwise(x)
        mask = self.get_mask(x)
        return self.compute_loss(y_pred, y, mask=mask)

    def compute_loss(self, y_pred, y, mask=None, batch_average=True):
        y_pred, y = y_pred.to(self.device), y.to(self.device)
        if mask is None or not self.mask_padding:
            mask = torch.ones_like(y)
        # mask is b, L, L. To normalise correctly, we need to divide by number of observations
        loss = (self.loss_function(y_pred, y)*mask).sum((-1,-2)) / mask.sum((-1,-2))

        # metrics characterising inputs: how many residues, how many with domain assignments.
        labelled_residues = ((y*mask).sum(-1) > 0).sum(-1)  # b
        non_padding_residues = (mask.sum(-1) > 0).sum(-1)  # b
        labelled_frac = labelled_residues / non_padding_residues
        metrics = {
            "labelled_residues": labelled_residues.detach().cpu().numpy(),
            "residues": non_padding_residues.detach().cpu().numpy(),
            "labelled_frac": labelled_frac.detach().cpu().numpy(),
            "loss": loss.detach().cpu().numpy(),
        }
        if batch_average:
            loss = loss.mean(0)
            metrics = {k: np.mean(v) for k, v in metrics.items()}

        return loss, metrics

    def domains_from_pairwise(self, y_pred):
        assert y_pred.ndim == 3
        domain_preds = []
        confidence_list = []
        for pred_single in y_pred.cpu().numpy():
            single_domains, confidence = self.domain_caller(pred_single)
            domain_preds.append(single_domains)
            confidence_list.append(confidence)
        return domain_preds, confidence_list

    def distance_transform(self, x):
        dist_chan = x[0, 0]
        # Find the minimum non-zero value in the channel
        min_nonzero = dist_chan[dist_chan > 0].min()
        # Replace zero values in the channel with the minimum non-zero value
        dist_chan[dist_chan == 0] = min_nonzero
        if self.dist_transform_type == "min_replace_inverse":
            # replace zero values and then invert.
            dist_chan = dist_chan ** (-1)
            x[0, 0] = dist_chan
            return x

        elif self.dist_transform_type == "unidoc_exponent": # replace zero values in pae / distance
            spread = self.distance_denominator
            dist_chan = (1 + np.exp((dist_chan - 8) / spread)) ** -1
            x[0,0] = dist_chan
            return x

    @torch.no_grad()
    def predict(self, x, return_pairwise=True):
        x = self.distance_transform(x)
        if self.max_recycles > 0:
            for i in range(self.max_recycles):
                # add recycling channels
                n_res = x.shape[-1]
                recycle_channels = torch.zeros(1, 2, n_res, n_res)
                # Concatenate the original tensor and the zeros tensor along the second dimension
                x = torch.cat((x, recycle_channels), dim=1)
                x = self.recycle_predict(x)
        y_pred = self.predict_pairwise(x)
        domain_dicts, confidence = self.domains_from_pairwise(y_pred)
        if self.post_process_domains:
            domain_dicts = self.post_process(domain_dicts, x) # todo move this to domains from pairwise function
        if return_pairwise:
            return y_pred, domain_dicts, confidence
        else:
            return domain_dicts, confidence

    @torch.no_grad()
    def recycle_predict(self, x):
        x = x.to(self.device)
        y_pred = self.predict_pairwise(x)
        domain_dicts, confidence = self.domains_from_pairwise(y_pred)
        y_pred_from_domains = np.array(
            [make_pair_labels(n_res=x.shape[-1], domain_dict=d_dict) for d_dict in domain_dicts])
        y_pred_from_domains = torch.tensor(y_pred_from_domains).to(self.device)
        if self.x_has_padding_mask:
            x[:, -2, :, :] = y_pred # assumes that last dimension is padding mask
            x[:, -3, :, :] = y_pred_from_domains
        else:
            x[:, -1, :, :] = y_pred
            x[:, -2, :, :] = y_pred_from_domains
        return x


    def post_process(self, domain_dicts, x_batch):
        new_domain_dicts = []
        for domain_dict, x in zip(domain_dicts, x_batch):
            x = x.cpu().numpy()
            domain_dict = {k: list(v) for k, v in domain_dict.items()}
            helix, sheet = x[1], x[2]
            diag_helix = np.diagonal(helix)
            diag_sheet = np.diagonal(sheet)
            ss_residues = list(np.where(diag_helix > 0)[0]) + list(np.where(diag_sheet > 0)[0])

            domain_dict = self.trim_disordered_boundaries(domain_dict, ss_residues)

            if self.remove_disordered_domain_threshold > 0:
                domain_dict = self.remove_disordered_domains(domain_dict, ss_residues)

            if self.min_ss_components > 0:
                domain_dict = self.remove_domains_with_few_ss_components(domain_dict, x)

            if self.min_domain_length > 0:
                domain_dict = self.remove_domains_with_short_length(domain_dict)
            new_domain_dicts.append(domain_dict)
        return new_domain_dicts

    def trim_disordered_boundaries(self, domain_dict, ss_residues):
        if not self.trim_each_domain:
            start = min(ss_residues)
            end = max(ss_residues)
        for dname, res in domain_dict.items():
            if dname == "linker":
                continue
            if self.trim_each_domain:
                domain_specific_ss = set(ss_residues).intersection(set(res))
                if len(domain_specific_ss) == 0:
                    continue
                start = min(domain_specific_ss)
                end = max(domain_specific_ss)
            domain_dict["linker"] += [r for r in res if r < start or r > end]
            domain_dict[dname] = [r for r in res if r >= start and r <= end]
        return domain_dict

    def remove_disordered_domains(self, domain_dict, ss_residues):
        new_domain_dict = {}
        for dname, res in domain_dict.items():
            if dname == "linker":
                continue
            if len(res) == 0:
                continue
            if len(set(res).intersection(set(ss_residues))) / len(res) < self.remove_disordered_domain_threshold:
                domain_dict["linker"] += res
            else:
                new_domain_dict[dname] = res
        new_domain_dict["linker"] = domain_dict["linker"]
        return new_domain_dict


    def remove_domains_with_few_ss_components(self, domain_dict, x):
        """
        Remove domains where number of ss components is less than minimum
        eg if self.min_ss_components=2 domains made of only a single helix or sheet are removed
        achieve this by counting the number of unique string hashes in domain rows of x
        """
        new_domain_dict = {}
        for dname, res in domain_dict.items():
            if dname == "linker":
                continue
            res = sorted(res)
            helix = x[1][res, :][:, res]
            strand = x[2][res, :][:, res]
            helix = helix[np.any(helix, axis=1)]
            strand = strand[np.any(strand, axis=1)]
            # residues in the same secondary structure component have the same representation in the helix or strand matrix
            n_helix = len(set(["".join([str(int(i)) for i in row]) for row in helix]))
            n_sheet = len(set(["".join([str(int(i)) for i in row]) for row in strand]))
            if len(res) == 0:
                continue
            if n_helix + n_sheet < self.min_ss_components:
                domain_dict["linker"] += res
            else:
                new_domain_dict[dname] = res
        new_domain_dict["linker"] = domain_dict["linker"]
        return new_domain_dict

    def remove_domains_with_short_length(self, domain_dict):
        """
        Remove domains where length is less than minimum
        """
        new_domain_dict = {}
        for dname, res in domain_dict.items():
            if dname == "linker":
                continue


            if len(res) < self.min_domain_length:
                domain_dict["linker"] += res
            else:
                new_domain_dict[dname] = res
        new_domain_dict["linker"] = domain_dict["linker"]
        return new_domain_dict
