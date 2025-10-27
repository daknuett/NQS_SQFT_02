import torch


def P2(sample_space, psi, dof, eps=1e-5):
    # -1 because of (-i)**2
    spp = sample_space.clone()
    spp[:, dof] += eps
    spm = sample_space.clone()
    spm[:, dof] -= eps
    P2_psi = -(psi(spp) + psi(spm) - 2*psi(sample_space)) / (eps**2)
    return P2_psi.squeeze()


def diagonal_term(sample_space, psi, ainvsquared, E_subtract):
     prefactor = (1 + 2*ainvsquared)
     return (prefactor * torch.sum(sample_space**2, dim=1) / 2 - E_subtract) * psi(sample_space).squeeze()


def interaction_term(sample_space, psi, ainvsquared):
    prefactor = - 1/2 * ainvsquared
    off_site = torch.roll(sample_space, 1, dims=1) + torch.roll(sample_space, -1, dims=1)
    return prefactor * torch.sum(sample_space * off_site, dim=1) * psi(sample_space).squeeze()


def H(sample_space, psi, ainvsquared, E_subtract=0.0):
    ndof = sample_space.shape[1]
    P2psi = sum(P2(sample_space, psi, dof) for dof in range(ndof))

    return P2psi / 2 + diagonal_term(sample_space, psi, ainvsquared, E_subtract) + interaction_term(sample_space, psi, ainvsquared)


def operator_expect(sample_space, psi, OP, p):
    psi_n = psi(sample_space)
    p_n = p(sample_space)
    Op_psi = OP(sample_space, psi)
    return (psi_n.conj().squeeze() * p_n * Op_psi).sum()

def operatorDG_expect(sample_space, psi, OP, p):
    psi_n = psi(sample_space)
    p_n = p(sample_space)
    Op_psi = OP(sample_space, psi)
    return (psi_n.squeeze() * p_n * Op_psi.conj()).sum()

def state_norm(sample_space, psi, p):
    psi_n = psi(sample_space)
    p_n = p(sample_space)
    return (psi_n.squeeze() * p_n * psi_n.conj().squeeze()).sum()
