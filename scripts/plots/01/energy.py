import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use(snakemake.params.mplstyle)


ndof = int(snakemake.wildcards.ndof)
model_layout = (ndof, snakemake.wildcards.model_layout)

t_stop = float(snakemake.wildcards.t_stop)
steps = int(snakemake.wildcards.steps)
delta = t_stop / steps
n_samples = int(snakemake.wildcards.n_samples)
model_activation_function = "tanh"

energy = np.load(snakemake.input[0])

time = np.arange(0, energy.shape[0], 1) * delta

fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

omega_k = lambda beta, k, N: np.sqrt(1 - 2*beta*(np.cos(2*np.pi*k / N) - 1))
e_gs = lambda ainvsquared, ndof: (0.5 * omega_k(ainvsquared, np.arange(0, ndof, 1), ndof)).sum()


axs[0].plot(time, energy.real)
axs[0].set_ylabel(r"$<E>$")
axs[0].set_title(
    fr"Sts: {ndof}; $\Delta t = {delta:.1e}$; {snakemake.wildcards.integrator}; $N_S={n_samples}$; " "\n"
    fr"Model: {model_layout}; $\beta={snakemake.wildcards.ainvsquared}$, {snakemake.wildcards.E_subtract_or_descr}"
)
axs[0].set_yscale("log")


#n = time.shape[0] // 3
#axs[1].plot(time[n:], energy.real[n:])
#axs[1].set_ylabel(r"$<E>$")

e_expect = e_gs(float(snakemake.wildcards.ainvsquared), ndof)
error = np.sqrt( (energy.real - e_expect)**2 / e_expect**2)
axs[1].plot(time, error)
axs[1].set_ylabel(r"$\sqrt{(<E> - E_0)^2 / E_0^2}$")
axs[1].set_xlabel(r"$\tau$")
axs[1].set_yscale("log")


plt.tight_layout()
plt.savefig(snakemake.output[0])
