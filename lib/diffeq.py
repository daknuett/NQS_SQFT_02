from model import get_M, reshape_params
from physics import H
from util import get_svd_inverse_cut
import torch


def F_imagt(t, y, model, sample_space, ainvsquared, E_subtract, regulator, svd_cut, elmr_store):
    for pi, yi in zip(model.parameters(), reshape_params(y, model)):
        pi.data = yi
        pi.grad = None

    M_R = get_M(sample_space, model.real)
    with torch.no_grad():
        E_L_R = H(sample_space, model.real, ainvsquared, E_subtract).flatten()

    M_I = get_M(sample_space, model.imag)
    with torch.no_grad():
        E_L_I = H(sample_space, model.imag, ainvsquared, E_subtract).flatten()

    S_R = M_R.adjoint() @ M_R
    S_I = M_I.adjoint() @ M_I

    S_R_regulated = S_R + torch.eye(S_R.shape[0]) * regulator
    S_I_regulated = S_I + torch.eye(S_I.shape[0]) * regulator

    S_R_inv = get_svd_inverse_cut(S_R_regulated, threshold=svd_cut)
    S_I_inv = get_svd_inverse_cut(S_I_regulated, threshold=svd_cut)

    rhs_R = -M_R.adjoint() @ E_L_R
    sol_R = S_R_inv @ rhs_R
    rhs_I = -M_I.adjoint() @ E_L_I
    sol_I = S_I_inv @ rhs_I
    elmr_store.M_R = M_R
    elmr_store.E_L_R = E_L_R
    elmr_store.M_I = M_I
    elmr_store.E_L_I = E_L_I

    return torch.concat([sol_R, sol_I])
