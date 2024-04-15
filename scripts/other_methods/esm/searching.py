# Calculate structure searching performance

import torch
from torch.nn.functional import normalize
import numpy as np
import os
from random import sample

dataset_dir = "../../dataset"
esm_embedding_size = 2560

def scop_fold(fam):
    return fam.rsplit(".", 2)[0]

def scop_sfam(fam):
    return fam.rsplit(".", 1)[0]

def read_dataset(fp):
    domids, fams = [], []
    with open(fp) as f:
        for line in f.readlines():
            domid, fam = line.rstrip().split()
            domids.append(domid)
            fams.append(fam)
    return domids, fams

domids_all, fams_all = read_dataset(dataset_dir + "/search_all.txt")
domids_unseen, fams_unseen = read_dataset(dataset_dir + "/test_unseen_sid.txt")

inds_unseen = [domids_all.index(d) for d in domids_unseen]

search_embeddings = torch.zeros(len(domids_all), esm_embedding_size)
for di, domid in enumerate(domids_all):
    l = torch.load(os.path.join("embed", domid + ".pt"), map_location="cpu")
    emb = l["mean_representations"][36]
    search_embeddings[di] = normalize(emb, dim=0)

def embedding_distance(emb_1, emb_2):
    cosine_dist = (emb_1 * emb_2).sum(dim=-1) # Normalised in the model
    return (1 - cosine_dist) / 2 # Runs 0 (close) to 1 (far)

def validate_searching_fam(search_embeddings, search_fams, search_inds, n_samples=None, log=False):
    sensitivities, top1_accuracies, top5_accuracies = [], [], []
    if n_samples is None or n_samples >= len(search_inds):
        sampled_is = search_inds[:]
    else:
        sampled_is = sample(search_inds, n_samples)
    for i in sampled_is:
        fam = search_fams[i]
        total_pos = sum(1 if h == fam else 0 for h in search_fams)
        dists = embedding_distance(search_embeddings[i:(i + 1)], search_embeddings)
        count_tp = 0
        top5_fam = 0
        for ji, j in enumerate(dists.argsort()):
            matched_fam = search_fams[j]
            if ji == 1:
                top1_accuracies.append(1 if matched_fam == fam else 0)
            if 0 < ji < 6 and matched_fam == fam:
                top5_fam = 1
            if matched_fam == fam:
                count_tp += 1
            elif scop_fold(matched_fam) != scop_fold(fam):
                break
        sensitivities.append(count_tp / total_pos)
        top5_accuracies.append(top5_fam)
        if log:
            print(domids_all[i], fam, count_tp / total_pos)
    return sensitivities, top1_accuracies, top5_accuracies

def validate_searching_sfam(search_embeddings, search_fams, search_inds, n_samples=None, log=False):
    sensitivities, top1_accuracies = [], []
    if n_samples is None or n_samples >= len(search_inds):
        sampled_is = search_inds[:]
    else:
        sampled_is = sample(search_inds, n_samples)
    for i in sampled_is:
        fam = search_fams[i]
        sfam = scop_sfam(fam)
        fold = scop_fold(fam)
        total_pos = sum(1 if f != fam and scop_sfam(f) == sfam else 0 for f in search_fams)
        dists = embedding_distance(search_embeddings[i:(i + 1)], search_embeddings)
        count_tp = 0
        for ji, j in enumerate(dists.argsort()):
            matched_fam = search_fams[j]
            if ji == 1:
                top1_accuracies.append(1 if scop_sfam(matched_fam) == sfam else 0)
            if matched_fam != fam and scop_sfam(matched_fam) == sfam:
                count_tp += 1
            elif scop_fold(matched_fam) != fold:
                break
        sensitivities.append(count_tp / total_pos)
        if log:
            print(domids_all[i], fam, count_tp / total_pos)
    return sensitivities, top1_accuracies

def validate_searching_fold(search_embeddings, search_fams, search_inds, n_samples=None, log=False):
    sensitivities, top1_accuracies = [], []
    if n_samples is None or n_samples >= len(search_inds):
        sampled_is = search_inds[:]
    else:
        sampled_is = sample(search_inds, n_samples)
    for i in sampled_is:
        fam = search_fams[i]
        sfam = scop_sfam(fam)
        fold = scop_fold(fam)
        total_pos = sum(1 if scop_sfam(f) != sfam and scop_fold(f) == fold else 0 for f in search_fams)
        dists = embedding_distance(search_embeddings[i:(i + 1)], search_embeddings)
        count_tp = 0
        for ji, j in enumerate(dists.argsort()):
            matched_fam = search_fams[j]
            if ji == 1:
                top1_accuracies.append(1 if scop_fold(matched_fam) == fold else 0)
            if scop_sfam(matched_fam) != sfam and scop_fold(matched_fam) == fold:
                count_tp += 1
            elif scop_fold(matched_fam) != fold:
                break
        sensitivities.append(count_tp / total_pos)
        if log:
            print(domids_all[i], fam, count_tp / total_pos)
    return sensitivities, top1_accuracies

sens_unseen_fam, top1s_fam, top5s_fam = validate_searching_fam(search_embeddings, fams_all, inds_unseen, None, True)
sens_unseen_sfam, top1s_sfam = validate_searching_sfam(search_embeddings, fams_all, inds_unseen, None, True)
sens_unseen_fold, top1s_fold = validate_searching_fold(search_embeddings, fams_all, inds_unseen, None, True)

print(np.mean(sens_unseen_fam ))
print(np.mean(sens_unseen_sfam))
print(np.mean(sens_unseen_fold))
