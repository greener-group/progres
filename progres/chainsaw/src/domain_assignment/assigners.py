import numpy as np
import torch
from progres.chainsaw.src.domain_assignment.sparse_lowrank import greedy_V


class BaseAssigner:

    def __call__(self, y_pred, uncertainty=False):
        assert y_pred.ndim == 2
        if torch.is_tensor(y_pred):
            y_pred = y_pred.detach().cpu().numpy()
        return self.assign_domains(y_pred)


class SparseLowRank(BaseAssigner):

    """Use Brooks' method to generate a matrix of cluster assignments,
    and post-process to remove any excessively small clusters.
    """

    def __init__(self, N_iters=3, K_init=4, linker_threshold=5, cost_type="mse"):
        self.N_iters = N_iters
        self.K_init = K_init
        self.linker_threshold = linker_threshold
        self.cost_type = cost_type

    def get_entropy(self, y_pred):
        """calculate the entropy of the upper triangle of the matrix
        used for uncertainty calculation but does not correlate with NDO as well
        as likelihood of assignment under y_pred
        """
        return np.triu(-1 * y_pred * np.log(y_pred) - (1 - y_pred) * np.log(1 - y_pred), 1).sum()


    def assign_domains(self, y_pred):
        # N x K, columns are then indicator vectors
        epsilon = 1e-6
        y_pred = np.clip(y_pred, epsilon, 1-epsilon)
        V, loss = greedy_V(y_pred, N_iters=self.N_iters, K_init=self.K_init, cost_type=self.cost_type)
        K = V.shape[-1]
        A = V@V.T
        average_likelihood = np.exp((A * np.log(y_pred) + (1-(A))*np.log(1-y_pred)).mean())
        # throw away small clusters
        V = np.where(
            V.sum(0, keepdims=True) < self.linker_threshold,  # 1, K
            np.zeros_like(V),
            V,
        )

        assignments = {
            "linker": np.argwhere((V == 0).sum(-1) == K).reshape(-1)
        }

        domain_ix = 1
        for col_ix in range(K):
            cluster_inds = np.argwhere(V[:, col_ix] == 1).reshape(-1)
            if cluster_inds.size > 0:
                assignments[f"domain_{domain_ix}"] = cluster_inds
                domain_ix += 1
        return assignments, round(average_likelihood, 4)


class SpectralClustering(BaseAssigner):
    """Challenge is how to determine n_clusters.
    
    We can look at eigengaps
    Or use MSE, running clustering multiple times.
    Or use Brooks' method and then refine with SC.
    """

    def __init__(self, n_cluster_method="mse", max_domains=10):
        self.n_cluster_method = n_cluster_method
        self.max_domains = max_domains  # TODO - make this flexible or length dependent.

    def assign_domains(self, y_pred):
        raise NotImplementedError()
