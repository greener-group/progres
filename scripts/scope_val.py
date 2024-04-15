# Calculate structure searching performance
# Unpack training_coords first with `tar -xvf training_coords.tgz`

import progres as pg
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
import numpy as np
from collections import defaultdict

dataset_dir = "dataset"
coord_dir = "training_coords"
device = "cpu"
model_batch_search = 8
embedding_size = 128

domids_search, fams_search = [], []
with open(dataset_dir + "/search_all.txt") as f:
    for line in f.readlines():
        cols = line.rstrip().split()
        domid, fam = cols[0], cols[1]
        domids_search.append(domid)
        fams_search.append(fam)

domid_to_nres, domid_to_contact_order = {}, {}
with open("other_methods/contact_order.txt") as f:
    for line in f.readlines():
        cols = line.rstrip().split()
        domid, nres, contact_order = cols[0], cols[1], cols[2]
        domid_to_nres[domid] = int(nres)
        domid_to_contact_order[domid] = float(contact_order)

class CoordinateDataset(Dataset):
    def __init__(self, domids):
        self.domids = domids
        self.model = pg.load_trained_model(device)

    def __len__(self):
        return len(self.domids)

    def __getitem__(self, idx):
        fp = f"{coord_dir}/{self.domids[idx]}"
        return pg.embed_structure(fp, fileformat="coords", device=device, model=self.model)

query_set = CoordinateDataset(domids_search)
num_workers = torch.get_num_threads() if device == torch.device("cpu") else 0
query_loader = DataLoader(query_set, batch_size=model_batch_search, shuffle=False,
                          num_workers=num_workers)

with torch.no_grad():
    query_embeddings = torch.zeros(len(query_set), embedding_size, device=device)
    for bi, out in enumerate(query_loader):
        query_embeddings[(bi * model_batch_search):(bi * model_batch_search + out.size(0))] = out

    query_embeddings = query_embeddings.to(torch.float16)
    dists = torch.zeros(len(query_set), len(query_set), device=device)
    for bi in range(len(query_set)):
        dists[bi:(bi + 1)] = pg.embedding_distance(query_embeddings[bi:(bi + 1)], query_embeddings)

with open(f"{dataset_dir}/test_unseen_sid.txt") as f:
    domids_unseen = [l.split()[0] for l in f.readlines()]

def scop_fold(fam):
    return fam.rsplit(".", 2)[0]

def scop_sfam(fam):
    return fam.rsplit(".", 1)[0]

def searching_accuracy(di, dists):
    fam_query = fams_search[di]
    sfam_query = scop_sfam(fam_query)
    fold_query = scop_fold(fam_query)
    count_tp_fam, count_tp_sfam, count_tp_fold = 0, 0, 0
    total_fam = sum(1 if f == fam_query else 0 for f in fams_search)
    total_sfam = sum(1 if f != fam_query and scop_sfam(f) == sfam_query else 0 for f in fams_search)
    total_fold = sum(1 if scop_sfam(f) != sfam_query and scop_fold(f) == fold_query else 0 for f in fams_search)
    top1_fam, top1_sfam, top1_fold, top5_fam = 0, 0, 0, 0

    for ji, j in enumerate(dists[di].argsort()):
        fam_match = fams_search[j]
        sfam_match = scop_sfam(fam_match)
        fold_match = scop_fold(fam_match)
        # Don't count self match for top 1 accuracy
        if ji == 1:
            if fam_match == fam_query:
                top1_fam = 1
            if sfam_match == sfam_query:
                top1_sfam = 1
            if fold_match == fold_query:
                top1_fold = 1
        if 0 < ji < 6 and fam_match == fam_query:
            top5_fam = 1
        if fam_match == fam_query:
            count_tp_fam += 1
        elif sfam_match == sfam_query:
            count_tp_sfam += 1
        elif fold_match == fold_query:
            count_tp_fold += 1
        else:
            break

    sens_fam  = count_tp_fam  / total_fam
    sens_sfam = count_tp_sfam / total_sfam
    sens_fold = count_tp_fold / total_fold

    return sens_fam, sens_sfam, sens_fold, top1_fam, top1_sfam, top1_fold, top5_fam

senses_fam, senses_sfam, senses_fold, top1s_fam, top1s_sfam, top1s_fold, top5s_fam = [], [], [], [], [], [], []
senses_fam_class, senses_sfam_class, senses_fold_class = defaultdict(list), defaultdict(list), defaultdict(list)
senses_fam_nres , senses_sfam_nres , senses_fold_nres  = defaultdict(list), defaultdict(list), defaultdict(list)
senses_fam_co   , senses_sfam_co   , senses_fold_co    = defaultdict(list), defaultdict(list), defaultdict(list)

for dc, domid in enumerate(domids_unseen):
    di = domids_search.index(domid)
    sens_fam, sens_sfam, sens_fold, top1_fam, top1_sfam, top1_fold, top5_fam = searching_accuracy(di, dists)

    senses_fam.append(sens_fam)
    senses_sfam.append(sens_sfam)
    senses_fold.append(sens_fold)
    top1s_fam.append(top1_fam)
    top1s_sfam.append(top1_sfam)
    top1s_fold.append(top1_fold)
    top5s_fam.append(top5_fam)

    cl = fams_search[di][0]
    senses_fam_class[cl].append(sens_fam)
    senses_sfam_class[cl].append(sens_sfam)
    senses_fold_class[cl].append(sens_fold)

    nres = domid_to_nres[domid]
    if 0 <= nres < 100:
        nres_group = "0-99"
    elif 100 <= nres < 200:
        nres_group = "100-199"
    elif 200 <= nres < 300:
        nres_group = "200-299"
    else:
        nres_group = "300+"

    senses_fam_nres[nres_group].append(sens_fam)
    senses_sfam_nres[nres_group].append(sens_sfam)
    senses_fold_nres[nres_group].append(sens_fold)

    co = domid_to_contact_order[domid]
    if 0 <= co < 0.075:
        co_group = "0-0.075"
    elif 0.075 <= co < 0.125:
        co_group = "0.075-0.125"
    elif 0.125 <= co < 0.175:
        co_group = "0.125-0.175"
    else:
        co_group = "0.175+"

    senses_fam_co[co_group].append(sens_fam)
    senses_sfam_co[co_group].append(sens_sfam)
    senses_fold_co[co_group].append(sens_fold)

print(np.mean(senses_fam))
print(np.mean(senses_sfam))
print(np.mean(senses_fold))
print()

for cl in sorted(list(senses_fam_class.keys())):
    print(cl, "class")
    print(np.mean(senses_fam_class[cl]))
    print(np.mean(senses_sfam_class[cl]))
    print(np.mean(senses_fold_class[cl]))
    print()

for nres_group in sorted(list(senses_fam_nres.keys())):
    print(nres_group, "nres")
    print(np.mean(senses_fam_nres[nres_group]))
    print(np.mean(senses_sfam_nres[nres_group]))
    print(np.mean(senses_fold_nres[nres_group]))
    print()

for co_group in sorted(list(senses_fam_co.keys())):
    print(co_group, "contact_order")
    print(np.mean(senses_fam_co[co_group]))
    print(np.mean(senses_sfam_co[co_group]))
    print(np.mean(senses_fold_co[co_group]))
    print()
