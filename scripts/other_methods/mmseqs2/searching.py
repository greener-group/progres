# Calculate structure searching performance

import numpy as np

dataset_dir = "../../dataset"

with open("out.m8") as f:
    mmseqs_lines = [l.rstrip() for l in f.readlines()]

domid_to_fam = {}
fams_search = []
with open(dataset_dir + "/search_all.txt") as f:
    for line in f.readlines():
        cols = line.rstrip().split()
        fam = cols[1]
        domid_to_fam[cols[0]] = fam
        fams_search.append(fam)

with open(dataset_dir + "/test_unseen_sid.txt") as f:
    domids_unseen = [l.split()[0] for l in f.readlines()]

def scop_fold(fam):
    return fam.rsplit(".", 2)[0]

def scop_sfam(fam):
    return fam.rsplit(".", 1)[0]

def searching_accuracy(domid_query, mmseqs_lines):
    fam_matches = []
    for line in mmseqs_lines:
        cols = line.split()
        if cols[0] == domid_query:
            domid_match = cols[1]
            if domid_match in domid_to_fam:
                fam_matches.append(domid_to_fam[domid_match])

    fam_query = domid_to_fam[domid_query]
    sfam_query = scop_sfam(fam_query)
    fold_query = scop_fold(fam_query)
    count_tp_fam, count_tp_sfam, count_tp_fold = 0, 0, 0
    total_fam = sum(1 if f == fam_query else 0 for f in fams_search)
    total_sfam = sum(1 if f != fam_query and scop_sfam(f) == sfam_query else 0 for f in fams_search)
    total_fold = sum(1 if scop_sfam(f) != sfam_query and scop_fold(f) == fold_query else 0 for f in fams_search)
    top1_fam, top1_sfam, top1_fold, top5_fam = 0, 0, 0, 0

    for fi, fam_match in enumerate(fam_matches):
        sfam_match = scop_sfam(fam_match)
        fold_match = scop_fold(fam_match)
        # Don't count self match for top 1 accuracy
        if fi == 1:
            if fam_match == fam_query:
                top1_fam = 1
            if sfam_match == sfam_query:
                top1_sfam = 1
            if fold_match == fold_query:
                top1_fold = 1
        if 0 < fi < 6 and fam_match == fam_query:
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

for di, domid in enumerate(domids_unseen):
    sens_fam, sens_sfam, sens_fold, top1_fam, top1_sfam, top1_fold, top5_fam = searching_accuracy(domid, mmseqs_lines)
    senses_fam.append(sens_fam)
    senses_sfam.append(sens_sfam)
    senses_fold.append(sens_fold)
    top1s_fam.append(top1_fam)
    top1s_sfam.append(top1_sfam)
    top1s_fold.append(top1_fold)
    top5s_fam.append(top5_fam)

print(np.mean(senses_fam))
print(np.mean(senses_sfam))
print(np.mean(senses_fold))
