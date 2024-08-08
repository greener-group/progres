from collections import defaultdict
import copy
import numpy as np


def check_unique_assignments(assn):
    res_counts = defaultdict(int)  # check for repeated residues
    for res_ids in assn.values():
        for res_id in res_ids:
            res_counts[res_id] += 1
    assert all([v == 1 for v in res_counts.values()]), f"Non-unique domain assignment {assn}"


def make_domain_dict(domain_str, n_res):
    """
    Converts string representation of domain boundaries to a dictionary
    Delimit separate domains with commas , and discontinuous domains
    with underscores _. Residue ranges separated by hyphens -, e.g. 1-100,101-200_300-340
    :param domain_str: domain boundaries expressed zero-indexed sequential  e.g. 1-100,101-200_300-340
    :param n_res: number of residues in the sequence
    :return:
    """
    domain_dict = {}
    bounds = domain_str.split(',')
    assigned_res = set()
    for i, bound in enumerate(bounds):
        if len(bound):

            for segment in bound.split('_'):
                if '-' in segment:
                    start, end = segment.split('-')
                else:
                    start = end = segment
                segment_res = set(range(int(start), int(end) + 1))
                assert len(segment_res.intersection(assigned_res)) == 0, f"Overlapping domain assignments {domain_str}"
                assigned_res.update(segment_res)
                domain_dict[f"D{i + 1}"] = domain_dict.get(f"D{i + 1}", []) + list(segment_res)
    domain_dict['linker'] = list(set(range(n_res)).difference(assigned_res))
    return domain_dict


def ndo_score(true: dict, pred: dict):
    """
    Normalized Domain Overlap Score
    Approximately corresponds to the fraction of residues that are assigned to the correct domain
    (Tai, C.H., et al. Evaluation of domain prediction in CASP6. Proteins 2005;61:183-192.)
    Full description of algorithm at https://ccrod.cancer.gov/confluence/display/CCRLEE/NDO]
    https://ccrod.cancer.gov/confluence/display/CCRLEE/NDO
    :param true: dict of domain assignments, with keys "linker" and "dX" where X is the domain number
    :param pred: dict of domain assignments, with keys "linker" and "dX" where X is the domain number
    example domain dictionary: {'linker':[0,1], 'D1':[2,3,4,8,9], 'D2':[5,6,7]}
    """

    # domains definitions must be mutually exclusive.
    check_unique_assignments(true)
    check_unique_assignments(pred)
    # alternative data structure would be a list of res ids for linker, plus a list of lists of res ids for domains.
    true = copy.deepcopy(true)
    pred = copy.deepcopy(pred)

    n_dom_pred = len([k for k in pred.keys() if k != "linker"])
    n_dom_gt = len([k for k in true.keys() if k != "linker"])

    # linkers are treated specially, so put them at row/col 0 to make this easy

    pred_linker = pred.pop("linker", [])
    gt_linker = true.pop("linker", [])

    gt_res_ids = [gt_linker] + list(true.values())
    pred_res_ids = [pred_linker] + list(pred.values())

    overlap = np.zeros((n_dom_pred + 1, n_dom_gt + 1))
    assert len(gt_res_ids) == overlap.shape[1] and len(pred_res_ids) == overlap.shape[0]

    for i, p_res in enumerate(pred_res_ids):
        for j, gt_res in enumerate(gt_res_ids):
            overlap[i, j] = len(set(p_res).intersection(set(gt_res)))

    # modified from v0: max overlap for a domain cannot be with a linker (hence max slices start at 1)
    if overlap.shape[0] == 1 or overlap.shape[1] == 1:
        # either the predictions or the true labels (or both) are all linker
        print('NDO score undefined for case where all linker')
        return 0
    row_scores = overlap[1:, 1:].max(axis=1)
    row_scores -= (overlap[1:].sum(axis=1) - row_scores)

    col_scores = overlap[1:, 1:].max(axis=0)
    col_scores -= (overlap[:, 1:].sum(axis=0) - col_scores)

    total_score = (row_scores.sum() + col_scores.sum()) / 2
    # count number of domain (non-linker) residues to normalize
    max_score = sum([len(gt_res) for gt_res in gt_res_ids[1:]])

    return total_score / max_score
