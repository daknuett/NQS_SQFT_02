import torch


def get_bounded_importance_sample(psi, n_samples, ndof, boundary, proposal_sigma, ndecorr):
    def prob(x):
        with torch.no_grad():
            raw_prob = torch.abs(psi(x)**2).squeeze()
        raw_prob[torch.where( torch.sqrt((x**2).sum(axis=-1)) > boundary)] = 0
        return raw_prob

    if ndecorr < 1:
        raise ValueError("ndecorr must be at least 1 but is recommended to be considerably bigger to equilibrate")

    sample = torch.zeros(n_samples, ndof).to(torch.double)
    p_old = prob(sample)
    
    for k in range(ndecorr):
        update = torch.randn(n_samples, ndof).to(torch.double) * proposal_sigma
        updated_sample = sample + update

        p = prob(updated_sample)
        accept_p = torch.rand(n_samples).to(torch.double)
        update[torch.where(p / p_old <= accept_p)] = 0

        sample += update
        p_old = prob(sample)
    return sample, p_old


def continue_bounded_importance_sample(psi, sample, boundary, proposal_sigma, ndecorr):
    def prob(x):
        with torch.no_grad():
            raw_prob = torch.abs(psi(x)**2).squeeze()
        raw_prob[torch.where( torch.sqrt((x**2).sum(axis=-1)) > boundary)] = 0
        return raw_prob

    if ndecorr < 1:
        raise ValueError("ndecorr must be at least 1 but is recommended to be considerably bigger to equilibrate")

    p_old = prob(sample)
    n_samples = sample.shape[0]
    
    for k in range(ndecorr):
        update = torch.randn_like(sample) * proposal_sigma
        updated_sample = sample + update

        p = prob(updated_sample)
        accept_p = torch.rand(n_samples).to(torch.double)
        update[torch.where(p / p_old <= accept_p)] = 0

        sample += update
        p_old = prob(sample)
    return sample, p_old
