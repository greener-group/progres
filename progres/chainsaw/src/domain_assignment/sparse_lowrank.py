import numpy as np


def mse_loss_at_residue(V_hat, Y, residue_index):
    # loss contribution of residue given current assignment
    # assumes Y is symmetric
    d = residue_index
    # avoid double-counting the diagonal
    return 2*((V_hat[d]@V_hat.T - Y[d])**2).sum() - (V_hat[d]@V_hat[d] - Y[d,d])**2


def null_mse_loss_at_residue(Y, residue_index):
    # loss contribution of residue when unassigned
    d = residue_index
    return 2*(Y[d]**2).sum() - Y[d,d]**2


def mse_loss(V_hat, Y):
    # only use to compute initial loss
    return ((V_hat@V_hat.T - Y)**2).sum()


def nll_loss(V_hat, Y):
    """We now interpret ys as probabilities.

    We want to minimise the negative log probability.
    
    NB Y must be symmetric and [0-1]
    """
    # only use to compute initial loss
    log_pij = (V_hat@V_hat.T) * np.log(Y) + (1-(V_hat@V_hat.T))*np.log(1-Y)
    # simple sum double counts off diagonals but single counts on diagonals, so need
    # to add diagonal again before dividing by 2
    return -0.5*(log_pij.sum() + np.diag(log_pij).sum())


def nll_loss_at_residue(V_hat, Y, residue_index):
    # d is residue_index
    d = residue_index  # V_hat[d]@V_hat.T is just Adj without instantiating A
    # n.b. whereas in case of mse we double-count off-diagonals, here we don't, hence the difference
    log_pdj = (V_hat[d]@V_hat.T) * np.log(Y[d]) + (1-(V_hat[d]@V_hat.T))*np.log(1-Y[d])
    return -log_pdj.sum()


def null_nll_loss_at_residue(Y, residue_index):
    return - np.log(1-Y[residue_index]).sum()


def greedy_V(Y, N_iters=3, K_init=4, cost_type="mse"):
    """ 
    Learn a binary matrix V, with at most one nonzero entry per row, to minimize
    
     || VV' - Y ||_2
     
    This is done by initializing a V of all zeros, and then doing a greedy optimization.
    V is initially D x K_init, where D is the number of residues (i.e., where Y is DxD).
    K is learned automatically; different values of K_init will not change the result but
    might be more or less efficient in terms of memory usage. The code keeps track of the 
    current number of nonzero columns of V, and when all the columns are full it adds extra
    columns of zeros at the end.
    
    Each iteration sweeps through all residues (i.e. rows of V) once.

    The implementation relies on additivity of loss to reduce computational cost
    by only computing required increments to the loss at each iteration.

    
    INPUTS:
    
    Y: matrix of predictions with entries in [0, 1]
    N_iters: number of iterations
    K_init: initial number of columns of V, adjust this for tweaking performance
    """
    Y = (Y + Y.T) / 2 # required for consistency with model_v1
    V_hat = np.zeros((Y.shape[0], K_init), dtype=np.uint8)
    if cost_type == "mse":
        loss = mse_loss(V_hat, Y) # initial loss for the zero matrix V_hat = 0
    elif cost_type == "nll":
        loss = nll_loss(V_hat, Y)
    else:
        raise ValueError(cost_type)
    K_max = K_init # track number of columns in K_max

    for it in range(N_iters):
        for d in range(V_hat.shape[0]):
            # d is a residue index
            # compute the loss, excluding contribution of this residue
            # TODO check minus signs are consistent
            loss_minus_d = loss - mse_loss_at_residue(V_hat, Y, d)

            # sweep through all K+1 options and compute what the contribution to the loss would be
            V_hat[d] *= 0

            # loss with no assignment
            L0 = loss_minus_d + null_mse_loss_at_residue(Y, d)
            L_opt = np.zeros(K_max)
            for k in range(K_max):
                # note this could be vectorized with a bit of work
                V_hat[d,k] = 1
                L_opt[k] = loss_minus_d + mse_loss_at_residue(V_hat, Y, d)
                V_hat[d,k] = 0

            # select the option which minimizes the squared error
            z = np.argmin(L_opt)
            if L_opt[z] < L0:
                V_hat[d,z] = 1
            
            # update loss
            loss = loss_minus_d + mse_loss_at_residue(V_hat, Y, d)

            if z == K_max-1:
                # Expand V_hat to make room for extra potential clusters (no nonzero columns remain)
                # TODO: note that if a cluster is "removed" on a later iteration we will miss this.
                #       probably doesn't matter, just means it is occasionally possible that the V
                #       that is returned could sometimes have a few nonzero columns.
                V_hat = np.concatenate((V_hat, np.zeros_like(V_hat)), -1)
                K_max = V_hat.shape[1]

    # drop columns that are zeros (i.e. unused clusters) before returning
    empty = V_hat.sum(0) == 0
    return V_hat[:,~empty], loss
