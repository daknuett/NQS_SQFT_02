from model import get_M, reshape_params
from physics import H
from util import get_svd_inverse_cut
import torch


def F_imagt(t, y, model, sample_space, ainvsquared, E_subtract, regulator, svd_cut, elmr_store):
    for pi, yi in zip(model.parameters(), reshape_params(y, model)):
        pi.data = yi
        pi.grad = None

    M_R = get_M(sample_space, model)
    with torch.no_grad():
        E_L_R = H(sample_space, model, ainvsquared, E_subtract).flatten()

    S_R = M_R.adjoint() @ M_R

    S_R_regulated = S_R + torch.eye(S_R.shape[0]) * regulator

    S_R_inv = get_svd_inverse_cut(S_R_regulated, threshold=svd_cut)

    rhs_R = -M_R.adjoint() @ E_L_R
    sol_R = S_R_inv @ rhs_R
    elmr_store.M_R = M_R
    elmr_store.E_L_R = E_L_R

    return sol_R
