"""
Domain boundary distance score, as defined in CASP 7 paper:
Assessment of predictions submitted for the CASP7 domain prediction category (2007)
https://onlinelibrary.wiley.com/doi/10.1002/prot.21675

Under our scoring scheme all predictions within 8 residues of the correct boundary
will score, but predictions that are closer to the correct domain boundary would score more.
All distances between the predicted and correct domain boundaries are calculated. If the
domain boundary has a linker, the whole linker is regarded as the domain boundary.
predictions are given one point for being within 1 residue of each correct boundary,
another point if they are within two residues, a further point if they are within three,
and so on up to eight residues. A prediction two residues away from the correct boundary
would therefore have 7 points.
The total score for each domain prediction is then calculated as the sum of all predicted
boundary scores divided by eight and the total number of domain boundaries. The number of
domain boundaries comes from either the target or the prediction, whichever is higher. In
this way over-prediction is penalized.

"""
import numpy as np


def pred_domains_to_bounds(pred_domains, optimize_for_linkers=False):
    """
    Converts domain dictionary to list of boundary residues
    """
    pred_bounds = np.array([])
    for name, res in pred_domains.items():
        if name=='linker' or len(res) ==0:
            continue
        res = np.array(sorted(set(res)))
        gaps  = res[1:] - res[:-1]
        if any(gaps > 1):
            gaps_start_indexes = np.where(gaps > 1)[0]
            pred_bounds = np.append(pred_bounds, res[gaps_start_indexes] + 1)
            pred_bounds = np.append(pred_bounds, res[gaps_start_indexes + 1])
        pred_bounds = np.append(pred_bounds, [res[0], res[-1] +1])
    if optimize_for_linkers:
        # given that linkers all count as boundaries probs possible
        # to score more points by predicting boundaries in the middle of linker
        # regions rather than the edges
        raise NotImplementedError # todo
    return np.array(sorted(list(set(pred_bounds.astype(int)))))


def get_true_boundary_res(domain_dict):
    """
    In the case where there are multiple non-domain
    residues between two domains, all of these NDRs
    are counted as domain boundaries. This adjustment
    is applied to the true boundaries but not the
    predicted boundaries.
    """
    bounds = pred_domains_to_bounds(domain_dict)
    # c.f. get_boundary_res below
    boundaries = {
        "boundary_res": list(bounds),
        "n_boundaries": len(list(bounds))
    }
    boundaries["boundary_res"] += list(domain_dict["linker"])
    return boundaries


def boundary_distance_score(domains, boundaries):
    pred_bounds = pred_domains_to_bounds(domains)
    # distance score as specified in CASP7 paper {distance_to_true_bound:score}
    dist_to_score = {0:8, 1:7, 2:6, 3:5, 4:4, 5:3, 6:2, 7:1} 
    score = 0
    # the final score is divided by the number of segments and the maximum score for each boundary
    normalizing_term = 8 * max(len(pred_bounds), boundaries['n_boundaries'])
    scores = []
    for b in pred_bounds:
        distance = min(abs(boundaries['boundary_res'] - b))
        if distance < 8:
            scores.append(dist_to_score[distance])
    # JW adjustment: additional boundaries that can't be mapped to a real bound 
    # should not be added to the un-normalized score
    score = sum(sorted(scores)[-boundaries['n_boundaries']:])
    return score / normalizing_term
