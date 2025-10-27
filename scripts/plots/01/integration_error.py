import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use(snakemake.params.mplstyle)


import sys
sys.path.insert(0, snakemake.params.module_path)
from model import Model, get_n_params

ndof = int(snakemake.wildcards.ndof)
model_layout = (ndof, snakemake.wildcards.model_layout)
t_stop = float(snakemake.wildcards.t_stop)
steps = int(snakemake.wildcards.steps)
delta = t_stop / steps
n_samples = int(snakemake.wildcards.n_samples)
dt = delta
model_activation_function = "tanh"

integration_error = np.load(snakemake.input[0])

plt.plot(np.arange(0, integration_error.shape[0], 1) * dt, integration_error)
plt.yscale("log")
plt.xlabel(r"$t$")
plt.ylabel(r"$|M\vec{a} - \vec{E}^{L}| / |\vec{E}^{L}| $" "\nwhere" r" $\vec{a} = S^{-1} M^T \vec{E}^{L}$")
plt.title(
    fr"Sites: {ndof}; $\Delta t = {delta:.1e}$; {snakemake.wildcards.integrator}; $N_S={n_samples}$; "
    fr"Model: Conv, {model_layout}; $\beta={snakemake.wildcards.ainvsquared}$"
    )

plt.savefig(snakemake.output[0])
