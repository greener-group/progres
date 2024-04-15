# Train structural embedding with contrastive learning
# Unpack training_coords first with `tar -xvf training_coords.tgz`

import torch
from torch.nn import Dropout, Identity, Linear, Sequential, SiLU
from torch.nn.functional import normalize
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_scatter import scatter
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from functools import cache
from math import ceil, isnan
import os
import random
from random import choice, sample
from shutil import copyfile
from time import time

learning_rate = 5e-5
weight_decay = 1e-16
n_layers = 6
embedding_size = 128
hidden_dim = 128
hidden_egnn_dim = 64
hidden_edge_dim = 256
use_node_feats = True
use_tau_angle = True
pos_embed_dim = 64
pos_embed_freq_inv = 2000
contact_dist = 10.0 # Å
dropout = 0.0
dropout_final = 0.0
gaussian_noise = 1.0
loss_temp = 0.1
n_contrastive_groups = 6
n_contrastive_group_size = 6
n_contrastive = n_contrastive_groups * n_contrastive_group_size
batch_size = n_contrastive # Must be a multiple of n_contrastive
model_batch_train = 6 # Must be a factor of n_contrastive
model_batch_search = 2
n_epochs = 10000
holdout = "sid" # "sid" or "sfam"
out_dir = "training"
log_file = os.path.join(out_dir, "training.log")
dataset_dir = "dataset"
coord_dir = "training_coords"
device = torch.device("cuda:0")

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

n_features = pos_embed_dim
if use_node_feats:
    n_features += 3
if use_tau_angle:
    n_features += 1

def report(msg):
    print(msg)
    with open(log_file, "a") as of:
        of.write(msg + "\n")

def scop_fold(fam):
    return fam.rsplit(".", 2)[0]

def read_domids(fp):
    with open(fp) as f:
        lines = [l.split()[0] for l in f.readlines()]
    return lines

domids_train = []
fam_to_domids = defaultdict(list)
with open(os.path.join(dataset_dir, f"train_{holdout}.txt")) as f:
    for line in f.readlines():
        domid, fam = line.rstrip().split()
        domids_train.append(domid)
        fam_to_domids[fam].append(domid)

class SinusoidalPositionalEncoding(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        channels = int(ceil(channels / 2) * 2)
        inv_freq = 1.0 / (pos_embed_freq_inv ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        sin_inp_x = torch.einsum("...i,j->...ij", tensor, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        return emb_x

if pos_embed_dim > 0:
    pos_embedder = SinusoidalPositionalEncoding(pos_embed_dim)

@cache
def read_coords(domid):
    coords = []
    with open(os.path.join(coord_dir, domid)) as f:
        for line in f.readlines():
            cols = line.rstrip().split()
            coords.append([float(v) for v in cols])
    return coords

def read_graph(domid, gaussian_noise=None):
    coords = read_coords(domid)
    n_res = len(coords)
    coords = torch.tensor(coords)
    if gaussian_noise is not None and gaussian_noise > 0:
        coords = coords + torch.randn(coords.shape) * gaussian_noise
    dmap = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0),
                       compute_mode="donot_use_mm_for_euclid_dist")
    contacts = (dmap <= contact_dist).squeeze(0)
    edge_index = contacts.to_sparse().indices()

    if use_node_feats:
        degrees = contacts.sum(dim=0)
        norm_degrees = (degrees / degrees.max()).unsqueeze(1)
        termini_feats = torch.tensor([[1] + [0] * (n_res - 1),
                                      [0] * (n_res - 1) + [1]]).transpose(0, 1)
    else:
        norm_degrees, termini_feats = torch.zeros(n_res, 0), torch.zeros(n_res, 0)
    if use_tau_angle:
        # The tau torsion angle is between 4 consecutive Cα atoms, we assign it to the second Cα
        # This feature breaks mirror invariance
        vec_ab = coords[1:-2] - coords[ :-3]
        vec_bc = coords[2:-1] - coords[1:-2]
        vec_cd = coords[3:  ] - coords[2:-1]
        cross_ab_bc = torch.cross(vec_ab, vec_bc, dim=1)
        cross_bc_cd = torch.cross(vec_bc, vec_cd, dim=1)
        taus = torch.atan2(
            (torch.cross(cross_ab_bc, cross_bc_cd, dim=1) * normalize(vec_bc, dim=1)).sum(dim=1),
            (cross_ab_bc * cross_bc_cd).sum(dim=1),
        )
        taus_pad = torch.cat((
            torch.tensor([0.0]),
            taus / torch.pi, # Convert to range -1 -> 1
            torch.tensor([0.0, 0.0]),
        )).unsqueeze(1)
    else:
        taus_pad = torch.zeros(n_res, 0)
    if pos_embed_dim > 0:
        pos_embed = pos_embedder(torch.arange(1, n_res + 1))
    else:
        pos_embed = torch.zeros(n_res, 0)
    x = torch.cat((norm_degrees, termini_feats, taus_pad, pos_embed), dim=1)
    data = Data(x=x, edge_index=edge_index, coords=coords)
    return data

def read_fam_domids(fam, n_domains):
    n_fam_domains = len(fam_to_domids[fam])
    if n_domains > n_fam_domains:
        domids = []
        for di in range(n_domains):
            if di < n_fam_domains:
                domids.append(fam_to_domids[fam][di])
            else:
                domids.append(domids[0])
        return domids
    else:
        return sample(fam_to_domids[fam], n_domains)

class ContrastiveDataset(Dataset):
    def __init__(self, shuffle=True):
        fams = list(fam_to_domids.keys())
        if shuffle:
            random.shuffle(fams)
        else:
            fams = sorted(fams)

        self.domids = []
        for fam in fams:
            # Pick domains from this family
            ds = read_fam_domids(fam, n_contrastive_group_size)
            self.domids.extend(ds)
            existing_fams = [fam]

            # Pick domains from other families
            while len(existing_fams) < n_contrastive_groups:
                other_fam = fam
                while other_fam in existing_fams:
                    other_fam = choice(fams)
                existing_fams.append(other_fam)
                ds = read_fam_domids(other_fam, n_contrastive_group_size)
                self.domids.extend(ds)

    def __len__(self):
        return len(self.domids)

    def __getitem__(self, idx):
        return read_graph(self.domids[idx], gaussian_noise=gaussian_noise)

def batched_index_select(values, indices, dim=1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

# From https://github.com/lucidrains/egnn-pytorch
class EGNN(torch.nn.Module):
    def __init__(
        self,
        dim,
        m_dim=16,
        dropout=0.0,
        init_eps=1e-3,
    ):
        super().__init__()
        edge_input_dim = (dim * 2) + 1
        dropout = Dropout(dropout) if dropout > 0 else Identity()

        self.edge_mlp = Sequential(
            Linear(edge_input_dim, hidden_edge_dim),
            dropout,
            SiLU(),
            Linear(hidden_edge_dim, m_dim),
            SiLU(),
        )

        self.node_mlp = Sequential(
            Linear(dim + m_dim, dim * 2),
            dropout,
            SiLU(),
            Linear(dim * 2, dim),
        )

        self.init_eps = init_eps
        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {Linear}:
            # seems to be needed to keep the network from exploding to NaN with greater depths
            torch.nn.init.normal_(module.weight, std=self.init_eps)

    def forward(self, feats, coors, mask, adj_mat):
        b, n, d, device = *feats.shape, feats.device

        rel_coors = rearrange(coors, "b i d -> b i () d") - rearrange(coors, "b j d -> b () j d")
        rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=True)

        i = j = n
        ranking = rel_dist[..., 0].clone()
        rank_mask = mask[:, :, None] * mask[:, None, :]
        ranking.masked_fill_(~rank_mask, 1e5)

        num_nearest = int(adj_mat.float().sum(dim=-1).max().item())
        valid_radius = 0

        self_mask = rearrange(torch.eye(n, device=device, dtype=torch.bool), "i j -> () i j")

        adj_mat = adj_mat.masked_fill(self_mask, False)
        ranking.masked_fill_(self_mask, -1.)
        ranking.masked_fill_(adj_mat, 0.)

        nbhd_ranking, nbhd_indices = ranking.topk(num_nearest, dim=-1, largest=False)
        nbhd_mask = nbhd_ranking <= valid_radius

        rel_coors = batched_index_select(rel_coors, nbhd_indices, dim=2)
        rel_dist = batched_index_select(rel_dist, nbhd_indices, dim=2)

        j = num_nearest
        feats_j = batched_index_select(feats, nbhd_indices, dim=1)
        feats_i = rearrange(feats, "b i d -> b i () d")
        feats_i, feats_j = torch.broadcast_tensors(feats_i, feats_j)

        edge_input = torch.cat((feats_i, feats_j, rel_dist), dim=-1)
        m_ij = self.edge_mlp(edge_input)

        mask_i = rearrange(mask, "b i -> b i ()")
        mask_j = batched_index_select(mask, nbhd_indices, dim = 1)
        mask = (mask_i * mask_j) & nbhd_mask

        m_ij_mask = rearrange(mask, "... -> ... ()")
        m_ij = m_ij.masked_fill(~m_ij_mask, 0.)
        m_i = m_ij.sum(dim=-2)

        node_mlp_input = torch.cat((feats, m_i), dim = -1)
        node_out = self.node_mlp(node_mlp_input) + feats

        return node_out, coors

# From https://github.com/lucidrains/egnn-pytorch
# and https://github.com/vgsatorras/egnn/blob/main/qm9/models.py
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.node_enc = Linear(n_features, hidden_dim)
        self.layers = torch.nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(EGNN(
                dim=hidden_dim,
                m_dim=hidden_egnn_dim,
                dropout=dropout,
            ))
        self.node_dec = torch.nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            Dropout(dropout) if dropout > 0 else Identity(),
            SiLU(),
            Linear(hidden_dim, hidden_dim),
        )
        self.graph_dec = torch.nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            Dropout(dropout) if dropout > 0 else Identity(),
            SiLU(),
            Dropout(dropout_final) if dropout_final > 0 else Identity(),
            Linear(hidden_dim, embedding_size),
        )

    def forward(self, data):
        feats, coords = data.x.unsqueeze(0), data.coords.unsqueeze(0)
        adj_mat = torch.sparse_coo_tensor(
            indices=data.edge_index,
            values=torch.tensor([1] * data.edge_index.size(1), device=device),
            size=(data.num_nodes, data.num_nodes),
        ).to_dense().bool().unsqueeze(0)
        mask = torch.ones(1, data.num_nodes, dtype=torch.bool, device=device)
        feats = self.node_enc(feats)
        for layer in self.layers:
            feats, coords = layer(feats, coords, mask, adj_mat)
        feats = self.node_dec(feats)
        batch = torch.tensor([0] * data.num_nodes, device=device) if data.batch is None else data.batch
        graph_feats = scatter(feats.squeeze(0), batch, dim=0, reduce="sum")
        out = self.graph_dec(graph_feats)
        return normalize(out, dim=1)

def embedding_distance(emb_1, emb_2):
    cosine_dist = (emb_1 * emb_2).sum(dim=-1) # Normalised in the model
    return (1 - cosine_dist) / 2 # Runs 0 (close) to 1 (far)

# From https://github.com/HobbitLong/SupContrast
class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode="all",
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [bsz, n_views, ...],"
                             "at least 3 dimensions are required")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        bsz = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(bsz, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != bsz:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(bsz * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss_val = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss_val = loss_val.view(anchor_count, bsz).mean()

        return loss_val

def validate_searching(search_embeddings, search_fams, search_inds, n_samples=None):
    sensitivities, top1_accuracies = [], []
    if n_samples is None or n_samples >= len(search_inds):
        sampled_is = search_inds[:]
    else:
        sampled_is = sample(search_inds, n_samples)
    for i in sampled_is:
        fam = search_fams[i]
        total_pos = sum(1 if h == fam else 0 for h in search_fams)
        dists = embedding_distance(search_embeddings[i:(i + 1)], search_embeddings)
        count_tp = 0
        for ji, j in enumerate(dists.argsort()):
            matched_fam = search_fams[j]
            if ji == 1:
                top1_accuracies.append(1 if matched_fam == fam else 0)
            if matched_fam == fam:
                count_tp += 1
            elif scop_fold(matched_fam) != scop_fold(fam):
                break
        sensitivities.append(count_tp / total_pos)
    return sensitivities, top1_accuracies

class SearchDataset(Dataset):
    def __init__(self, fp):
        self.domids = []
        self.fams = []
        with open(fp) as f:
            for line in f.readlines():
                domid, fam = line.rstrip().split()
                self.domids.append(domid)
                self.fams.append(fam)

    def __len__(self):
        return len(self.domids)

    def __getitem__(self, idx):
        return read_graph(self.domids[idx])

search_set = SearchDataset(os.path.join(dataset_dir, "search_all.txt"))
search_loader = DataLoader(search_set, batch_size=model_batch_search, shuffle=False)

domids_val_seen   = read_domids(os.path.join(dataset_dir, "val_seen.txt"))
domids_val_unseen = read_domids(os.path.join(dataset_dir, f"val_unseen_{holdout}.txt"))
val_seen_inds   = [search_set.domids.index(d) for d in domids_val_seen  ]
val_unseen_inds = [search_set.domids.index(d) for d in domids_val_unseen]

labels_contrastive = torch.tensor(
    [[i] * n_contrastive_group_size for i in range(n_contrastive_groups)], device=device).view(-1)
ys = [i / 100 for i in range(101)]

model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_fn = SupConLoss(temperature=loss_temp)

train_losses = []
senses_seen, senses_unseen = [], []
if os.path.isfile(log_file):
    loaded_model = torch.load(os.path.join(out_dir, "model.pt"), map_location=device)
    model.load_state_dict(loaded_model["model"])
    optimizer.load_state_dict(loaded_model["optimizer"])
    with open(log_file) as f:
        for line in f.readlines():
            if line.startswith("Epoch"):
                cols = line.rstrip().split()
                train_losses.append( float(cols[4 ]))
                senses_seen.append(  float(cols[8 ]))
                senses_unseen.append(float(cols[14]))
    starting_epoch_n = len(train_losses) + 1
    best_sens_seen_mean = max(senses_seen)
    best_sens_unseen_mean = max(senses_unseen)
    report(f"Restarting training from epoch {starting_epoch_n}")
else:
    starting_epoch_n = 1
    best_sens_seen_mean, best_sens_unseen_mean = 0.0, 0.0
    report("Starting training")

for epoch_n in range(starting_epoch_n, n_epochs + 1):
    start_time = time()

    # Redraw random samples each epoch
    train_set_cont = ContrastiveDataset(shuffle=True)

    # shuffle must be False here, shuffling occurs in ContrastiveDataset
    train_loader_cont = DataLoader(
        train_set_cont,
        batch_size=model_batch_train,
        shuffle=False,
        num_workers=batch_size // 2,
        prefetch_factor=2,
        pin_memory=True,
    )

    # Train model contrastively
    model.train()
    optimizer.zero_grad()
    train_loss_sum = 0.0
    batch_embeddings = []
    for bi, batch in enumerate(train_loader_cont):
        if bi % 100 == 0:
            print(bi, "/", len(train_loader_cont))
        out = model(batch.to(device))
        batch_embeddings.append(out)
        if (bi + 1) % (n_contrastive // model_batch_train) == 0:
            feats = torch.cat(batch_embeddings, dim=0).unsqueeze(1)
            loss = loss_fn(feats, labels_contrastive)
            loss.backward()
            train_loss_sum += loss.item()
            batch_embeddings = []
        if ((bi + 1) * model_batch_train) % batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()

    model.eval()
    with torch.no_grad():
        # Validate model for family searching
        search_embeddings = torch.zeros(len(search_set), embedding_size, device=device)
        for bi, batch in enumerate(search_loader):
            out = model(batch.to(device))
            search_embeddings[(bi * model_batch_search):(bi * model_batch_search + out.size(0))] = out

        sens_seen  , top1_seen   = validate_searching(search_embeddings, search_set.fams, val_seen_inds  )
        sens_unseen, top1_unseen = validate_searching(search_embeddings, search_set.fams, val_unseen_inds)
        sens_seen_mean, sens_unseen_mean = np.mean(sens_seen), np.mean(sens_unseen)
        top1_seen_mean, top1_unseen_mean = np.mean(top1_seen), np.mean(top1_unseen)

    train_loss_mean = train_loss_sum / (len(train_set_cont) / n_contrastive)
    train_losses.append(train_loss_mean)
    plt.plot(range(1, epoch_n + 1), train_losses, label="Training")
    plt.ylim(bottom=0)
    plt.legend()
    plt.xlabel("Epoch number")
    plt.ylabel("Loss")
    plt.title("Training progress")
    plt.savefig(os.path.join(out_dir, "loss.pdf"))
    plt.clf()

    senses_seen.append(sens_seen_mean)
    senses_unseen.append(sens_unseen_mean)
    plt.plot(range(1, epoch_n + 1), senses_seen  , label="Seen"  )
    plt.plot(range(1, epoch_n + 1), senses_unseen, label="Unseen")
    plt.legend()
    plt.xlabel("Epoch number")
    plt.ylabel("Mean sensitivity for structure searching")
    plt.title("Training progress")
    plt.savefig(os.path.join(out_dir, "searching.pdf"))
    plt.clf()

    cum_sens_seen   = [sum(1 if s >= y else 0 for s in sens_seen  ) / len(sens_seen  ) for y in ys]
    cum_sens_unseen = [sum(1 if s >= y else 0 for s in sens_unseen) / len(sens_unseen) for y in ys]
    plt.plot(cum_sens_seen  , ys, label="Seen"  )
    plt.plot(cum_sens_unseen, ys, label="Unseen")
    plt.xlim(-0.05, 1.05)
    plt.legend()
    plt.xlabel("Fraction of queries")
    plt.ylabel("Sensitivity up to the first FP")
    plt.title(f"Epoch {epoch_n}")
    plt.savefig(os.path.join(out_dir, "sensitivity.pdf"))
    plt.clf()

    torch.save({
        "epoch"    : epoch_n,
        "model"    : model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, os.path.join(out_dir, "model.pt"))

    if sens_seen_mean > best_sens_seen_mean:
        best_sens_seen_mean = sens_seen_mean
        copyfile(os.path.join(out_dir, "sensitivity.pdf"),
                 os.path.join(out_dir, "best_seen_sensitivity.pdf"))
        copyfile(os.path.join(out_dir, "model.pt"), os.path.join(out_dir, "best_seen_model.pt"))
    if sens_unseen_mean > best_sens_unseen_mean:
        best_sens_unseen_mean = sens_unseen_mean
        copyfile(os.path.join(out_dir, "sensitivity.pdf"),
                 os.path.join(out_dir, "best_unseen_sensitivity.pdf"))
        copyfile(os.path.join(out_dir, "model.pt"), os.path.join(out_dir, "best_unseen_model.pt"))

    end_time = time()

    report("Epoch {:4} - loss {:6.3f} - seen sens {:5.3f} top1 {:5.3f} - unseen sens {:5.3f} top1 {:5.3f} - {:5.1f} s".format(
        epoch_n,
        train_loss_mean,
        sens_seen_mean,
        top1_seen_mean,
        sens_unseen_mean,
        top1_unseen_mean,
        end_time - start_time,
    ))

    if isnan(train_loss_mean):
        break
