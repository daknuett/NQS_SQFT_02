import torch
import numpy as np
import copy

from types import SimpleNamespace

import sys
sys.path.insert(0, snakemake.params.module_path)

from model import get_grad, get_params, get_n_params, reshape_params, ModelConvL, ModelConvBL, ModelDense, get_M, decode_model_string
from integrator import Euler1, HeunsMethod, RK4, Midpoint, SSPRK3
from physics import H, operator_expect, operatorDG_expect, state_norm
from importance_sampling import get_bounded_importance_sample, continue_bounded_importance_sample
from diffeq import F_imagt
from util import TimingLogger

#torch.set_default_device('cuda')

ainvsquared = float(snakemake.wildcards.ainvsquared)
svd_cut = snakemake.params.svd_cut
regulator = snakemake.params.regulator
seed = snakemake.params.seed
ndof = int(snakemake.wildcards.ndof)

mode_class_dict = {
        ("dense", None): ModelDense
        , ("convolution", "L"): ModelConvL
        , ("convolution", "B"): ModelConvBL
}

model_desciption, layout = decode_model_string(snakemake.wildcards.model_layout)
mdl_str = snakemake.wildcards.model_layout
model_class = mode_class_dict[model_desciption]
model_layout = (ndof, *layout)

integrator_dict = {
    "Euler1": Euler1,
    "HeunsMethod": HeunsMethod,
    "RK4": RK4,
    "Midpoint": Midpoint,
    "SSPRK3": SSPRK3
}
integrator = integrator_dict[snakemake.wildcards.integrator]()
save_every_t = snakemake.params.save_every_t

torch.manual_seed(seed)


t_stop = float(snakemake.wildcards.t_stop)
steps = int(snakemake.wildcards.steps)
delta = t_stop / steps
save_every_n = int(save_every_t / delta)

n_samples = int(snakemake.wildcards.n_samples)
boundary = float(snakemake.wildcards.boundary)
sampling_sigma = snakemake.params.sampling_sigma
sampling_decorr = snakemake.params.sampling_decorr
sampling_equilibrate = snakemake.params.sampling_equilibrate
sampling_shape = (n_samples, ndof)

initial_undersampling = snakemake.params.initial_undersampling
initial_stepwidth_factor = snakemake.params.initial_stepwidth_factor
initial_time = snakemake.params.initial_time
initial_boundary_factor = snakemake.params.initial_boundary_factor

t_stop_initial = initial_time
delta_initial = delta * initial_stepwidth_factor
steps_initial = int(t_stop_initial / delta_initial)
steps_per_step_initial = int(1 / initial_stepwidth_factor)
steps_left = int((t_stop - t_stop_initial) / delta)

save_every_n_initial = int(save_every_t / delta_initial)
sampling_shape_initial = (int(n_samples / initial_undersampling), ndof)

torch.random.manual_seed(seed)
model = model_class(*model_layout)


updates = []
M_R_log = []
E_L_R_log = []
model_log = []
sample_spaces = []

energies = np.zeros((steps, 1))
energy_subtract = np.zeros(steps)
state_norm_log = np.zeros(steps)
ie_sample = np.zeros(steps)


tl = TimingLogger(f"ITE initial {mdl_str}")

# Coldstart
sample_space = torch.zeros(*sampling_shape_initial, dtype=torch.double)
sample_space, probability = continue_bounded_importance_sample(lambda x: H(x, model, ainvsquared), sample_space, boundary * initial_boundary_factor, sampling_sigma, sampling_equilibrate)

elmr_store = SimpleNamespace()

tl.start()
for i in range(steps_initial):
    with tl.time_section("importance sampling"):
        sample_space, probability = continue_bounded_importance_sample(lambda x: H(x, model, ainvsquared), sample_space, boundary * initial_boundary_factor, sampling_sigma, sampling_decorr)
    prob = lambda _: 1 / probability / sampling_shape[0]

    params_before = get_params(model)

    with tl.time_section("integration step"):
        params_after = integrator(params_before, i * delta_initial, delta_initial, F_imagt, (model, sample_space, ainvsquared, 0.0, regulator, svd_cut, elmr_store))

    M_R_log = [elmr_store.M_R]
    E_L_R_log = [elmr_store.E_L_R]

    
    for p, pdn in zip(model.parameters(), reshape_params(params_after, model)):
        p.data = pdn
        p.grad = None

    updates.append((params_after - params_before) / delta_initial)

    with torch.no_grad():
        energies[i // steps_per_step_initial, 0] = (operator_expect(sample_space, model, lambda sps, psi: H(sps, psi, ainvsquared), prob)
                               / state_norm(sample_space, model, prob)).detach().to("cpu").numpy()
        e_subtract = energies[i // steps_per_step_initial, 0]
                               
    sol = updates[-1]
    ie_sample[i // steps_per_step_initial] = torch.sum(torch.abs(M_R_log[-1] @ sol + E_L_R_log[-1])) / torch.sum(E_L_R_log[-1])

    print(f"{mdl_str}[I][{i*delta_initial:2.1f}] {(i+1) / steps_initial * 100:5.1f} % <E>_r = {energies[i // steps_per_step_initial, 0]:5.2f} (s)     {tl.get_formatted_eta(i+1, steps_initial)}   ", end="\n")
    if i % save_every_n_initial == 0:
        model_log.append(copy.deepcopy(model.state_dict()))

    if i in (2, steps // 4, steps // 2):
        tl.make_report()
tl.make_report()


tl = TimingLogger(f"ITE {mdl_str}")
# Coldstart
sample_space = torch.zeros(*sampling_shape, dtype=torch.double)
sample_space, probability = continue_bounded_importance_sample(lambda x: H(x, model, ainvsquared), sample_space, boundary, sampling_sigma, sampling_equilibrate)
sample_space_ref = torch.clone(sample_space)
probability_ref = torch.clone(probability)
prob_ref = lambda _: 1 / probability_ref / sampling_shape[0]
with torch.no_grad():
    norm_ref = state_norm(sample_space_ref, model, prob_ref)

tl.start()
for i in range(steps - steps_left, steps):
    with tl.time_section("importance sampling"):
        sample_space, probability = continue_bounded_importance_sample(lambda x: H(x, model, ainvsquared), sample_space, boundary, sampling_sigma, sampling_decorr)
    prob = lambda _: 1 / probability / sampling_shape[0]

    with torch.no_grad():
        norm_here = state_norm(sample_space_ref, model, prob_ref)

    e_subtract += snakemake.params.esubtract_dampening * torch.log(norm_ref / norm_here)
    state_norm_log[i] = norm_here
    energy_subtract[i] = e_subtract


    params_before = get_params(model)

    with tl.time_section("integration step"):
        params_after = integrator(params_before, i * delta, delta, F_imagt, (model, sample_space, ainvsquared, e_subtract, regulator, svd_cut, elmr_store))

    M_R_log = [elmr_store.M_R]
    E_L_R_log = [elmr_store.E_L_R]

    
    for p, pdn in zip(model.parameters(), reshape_params(params_after, model)):
        p.data = pdn
        p.grad = None

    updates.append((params_after - params_before) / delta)

    with torch.no_grad():
        energies[i, 0] = (operator_expect(sample_space, model, lambda sps, psi: H(sps, psi, ainvsquared), prob)
                               / state_norm(sample_space, model, prob)).detach().to("cpu").numpy()
        e_subtract = energies[i, 0]
                               
    sol = updates[-1]
    ie_sample[i] = torch.sum(torch.abs(M_R_log[-1] @ sol + E_L_R_log[-1])) / torch.sum(E_L_R_log[-1])

    print(f"{mdl_str}[{i*delta:2.1f}] {(i+1) / steps * 100:5.1f} % <E>_r = {energies[i, 0]:5.2f} (s)     {tl.get_formatted_eta(i+1 - (steps - steps_left), steps_left)}   ", end="\n")
    if i % save_every_n == 0:
        model_log.append(copy.deepcopy(model.state_dict()))

    if i in (2, steps // 4, steps // 2):
        tl.make_report()

model_log.append(copy.deepcopy(model.state_dict()))
tl.make_report()

torch.save(model_log, snakemake.output[0])
np.save(snakemake.output[1], energies)
np.save(snakemake.output[2], ie_sample)
np.save(snakemake.output[3], energy_subtract)
np.save(snakemake.output[4], state_norm_log)
