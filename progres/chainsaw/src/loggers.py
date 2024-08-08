import glob
import os

import logging
LOG = logging.getLogger(__name__)

def get_versioned_dir(output_dir, version=None, resume=False):
    """version gets dir for specific version, resume gets dir for last version."""
    if version is None:
        current_versions = glob.glob(os.path.join(output_dir, "version*"))
        if current_versions:
            last_version = max([int(os.path.basename(v).split("_")[1]) for v in current_versions])
            version = last_version if resume else last_version + 1
        else:
            assert not resume, f"Passed resume True but no matching directories in {output_dir}"
            version = 1

    version_dir = os.path.join(output_dir, f"version_{version}")
    return version_dir, version


def log_epoch_metrics(
    epoch,
    metrics,
    output_file,
    extra_keys=None,
    start_epoch=0,
    new_file=False
):
    """
    New file gets created if epoch == 1

    We are going for a hierarchical structure /experiment_group/model_name/train_metrics.csv etc
    because this works best with tensorboard and avoids file clutter in a single 
    experiment_group directory

    tensorboard refs:
        https://pytorch.org/docs/stable/tensorboard.html
        https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
    """
    # output_filename = (model_name + f"_{msa_name}" + f"_vae" + 
                       # ("_posembed{args.pos_embed_dim}" if args.embed_pos else ""))
    
    metrics.pop("epoch", None)
    metric_names = list(metrics.keys())
    extra_keys = extra_keys or []
    assert all([m not in metric_names for m in extra_keys]), f"{metric_names} {extra_keys}"
    metric_names += list(extra_keys)

    if new_file:  # c.f. training/core epoch 0 is for validation.
        with open(output_file, "w") as csvf:
            csvf.write(",".join(["epoch"] + metric_names) + "\n")

    with open(output_file, "a") as csvf:
        csvf.write(",".join([str(epoch + start_epoch)] + [str(metrics.get(m, "")) for m in metric_names])+"\n")


class StdOutLogger:

    def __init__(self, log_freq, start_epoch=0):
        self.start_epoch = start_epoch
        self.log_freq = log_freq

    def log(self, epoch, metrics, batch=None):
        if self.log_freq is not None and epoch % self.log_freq == 0:
            if batch is None:
                header = f"Epoch {epoch + self.start_epoch}:   "
            else:
                header = f"[{epoch:d}, {batch:5d}]:   "

            train_metric_components = [f"{m}: {v:.3f} " for m, v in metrics.items() if not m.startswith("val_")]
            if train_metric_components:
                LOG.info(
                    header
                    + "  ".join(train_metric_components),
                )
            val_metric_components = [f"{m}: {v:.3f} " for m, v in metrics.items() if m.startswith("val_")]
            if val_metric_components:
                LOG.info(
                    "  ".join(val_metric_components),
                )
            if batch is None:
                LOG.info("--------------------------------------\n")


class CSVLogger:
    def __init__(self, output_dir, start_epoch=0):
        self.output_dir = output_dir
        self.start_epoch = start_epoch
        self.val_keys = None
        self.filename = f"train_log.{'' if start_epoch == 0 else (str(start_epoch) + '.')}csv"
        self.logged = 0

    @property
    def filepath(self):
        return str(os.path.join(self.output_dir, self.filename))

    def log(self, epoch, metrics, batch=None):
        metrics["batch"] = batch
        
        if epoch == 0:
            self.val_keys = metrics.keys()
        elif self.output_dir is not None and epoch > 0:
            # LOG.info([k for k in metrics.keys() if k not in self._prev_keys])
            extra_keys = [k for k in self.val_keys if k not in metrics and k != "epoch"]
            os.makedirs(self.output_dir, exist_ok=True)
            log_epoch_metrics(
                epoch,
                metrics,
                self.filepath,
                extra_keys=extra_keys,
                start_epoch=self.start_epoch,
                new_file=self.logged == 0
            )
            self.logged += 1
            # self._prev_keys = metrics.keys()


class LoggerContainer:

    def __init__(self, loggers, start_epoch=0):
        self.train_log = []
        self.loggers = loggers
        self.start_epoch = start_epoch

    def log(self, epoch, metrics, batch=None):
        for logger in self.loggers:
            logger.log(epoch, metrics, batch=batch)
        metrics["epoch"] = epoch + self.start_epoch
        self.train_log.append(metrics)
