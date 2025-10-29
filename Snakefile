import os

wildcard_constraints:
    model_layout = r"(D|C)\d+(L|B)\d+",
    n_samples = r"\d+",
    steps = r"\d+",
    n_dof = r"\d+",
    ainvsquared = r"\d+\.?\d*",
    t_stop = r"\d+\.?\d*",
    integrator = r"Midpoint|HeunsMethod|RK4|Euler1|SSPRK3",
    boundary = r"\d+\.?\d*",
    E_subtract = r"\d+\.?\d*",


rule all:
    input:
        expand("data/{ndof}/{model_layout}_{integrator}_initial/{n_samples}_{boundary}_SCE/models/model_{t_stop}_{steps}_{ainvsquared}.pt"
               , ndof=[8]
               , model_layout=["C6L4","C6B4", "D6L8", "C3L4","C3B4", "D3L20"]
               , integrator="Euler1"
               , n_samples=[2000]
               , boundary=10
               , t_stop=1.4
               , steps=[700]
               , ainvsquared=[1.0]
               ),
        expand("data/plots/{ndof}/{model_layout}_{integrator}_initial/{n_samples}_{boundary}_SCE/energy/energy_{t_stop}_{steps}_{ainvsquared}.png"
               , ndof=[8]
               , model_layout=["C6L4","C6B4", "D6L8", "C3L4","C3B4", "D3L20"]
               , integrator="Euler1"
               , n_samples=[2000]
               , boundary=10
               , t_stop=1.4
               , steps=[700]
               , ainvsquared=[1.0]
               ),
        expand("data/plots/{ndof}/{model_layout}_{integrator}_initial/{n_samples}_{boundary}_SCE/manifold_eror/mfe_{t_stop}_{steps}_{ainvsquared}.png"
               , ndof=[8]
               , model_layout=["C6L4","C6B4", "D6L8", "C3L4","C3B4", "D3L20"]
               , integrator="Euler1"
               , n_samples=[2000]
               , boundary=10
               , t_stop=1.4
               , steps=[700]
               , ainvsquared=[1.0]
               ),
        expand("data/{ndof}/{model_layout}_{integrator}_initial/{n_samples}_{boundary}_{E_subtract}/models/model_{t_stop}_{steps}_{ainvsquared}.pt"
               , ndof=[8]
               , model_layout=["C6L4","C6B4", "D6L8", "C3L4","C3B4", "D3L20"]
               , integrator="Euler1"
               , n_samples=[2000]
               , boundary=10
               , t_stop=1.4
               , steps=[700]
               , ainvsquared=[1.0]
               , E_subtract=6.5
               ),
        expand("data/plots/{ndof}/{model_layout}_{integrator}_initial/{n_samples}_{boundary}_{E_subtract}/energy/energy_{t_stop}_{steps}_{ainvsquared}.png"
               , ndof=[8]
               , model_layout=["C6L4","C6B4", "D6L8", "C3L4","C3B4", "D3L20"]
               , integrator="Euler1"
               , n_samples=[2000]
               , boundary=10
               , t_stop=1.4
               , steps=[700]
               , ainvsquared=[1.0]
               , E_subtract=6.5
               ),
        expand("data/plots/{ndof}/{model_layout}_{integrator}_initial/{n_samples}_{boundary}_{E_subtract}/manifold_eror/mfe_{t_stop}_{steps}_{ainvsquared}.png"
               , ndof=[8]
               , model_layout=["C6L4","C6B4", "D6L8", "C3L4","C3B4", "D3L20"]
               , integrator="Euler1"
               , n_samples=[2000]
               , boundary=10
               , t_stop=1.4
               , steps=[700]
               , ainvsquared=[1.0]
               , E_subtract=6.5
               ),
        expand("data/{ndof}/{model_layout}_{integrator}_initial/{n_samples}_{boundary}_APN/models/model_{t_stop}_{steps}_{ainvsquared}.pt"
               , ndof=[8]
               , model_layout=["C6L4","C6B4", "D6L8", "C3L4","C3B4", "D3L20"]
               , integrator="Euler1"
               , n_samples=[2000]
               , boundary=10
               , t_stop=1.4
               , steps=[700]
               , ainvsquared=[1.0]
               ),
        expand("data/plots/{ndof}/{model_layout}_{integrator}_initial/{n_samples}_{boundary}_APN/energy/energy_{t_stop}_{steps}_{ainvsquared}.png"
               , ndof=[8]
               , model_layout=["C6L4","C6B4", "D6L8", "C3L4","C3B4", "D3L20"]
               , integrator="Euler1"
               , n_samples=[2000]
               , boundary=10
               , t_stop=1.4
               , steps=[700]
               , ainvsquared=[1.0]
               ),
        expand("data/plots/{ndof}/{model_layout}_{integrator}_initial/{n_samples}_{boundary}_APN/manifold_eror/mfe_{t_stop}_{steps}_{ainvsquared}.png"
               , ndof=[8]
               , model_layout=["C6L4","C6B4", "D6L8", "C3L4","C3B4", "D3L20"]
               , integrator="Euler1"
               , n_samples=[2000]
               , boundary=10
               , t_stop=1.4
               , steps=[700]
               , ainvsquared=[1.0]
               ),


        # Less models more couplings
        expand("data/{ndof}/{model_layout}_{integrator}_initial/{n_samples}_{boundary}_{E_subtract_or_descr}/models/model_{t_stop}_{steps}_{ainvsquared}.pt"
               , ndof=[8]
               , model_layout=["C6L4","D6L8", "C3L4"]
               , integrator="Euler1"
               , n_samples=[2000]
               , boundary=10
               , t_stop=1.4
               , steps=[700]
               , ainvsquared=[1.0, 2, 5, 10, 20, 40]
               , E_subtract_or_descr=["SCE", "APN", 6.5, 10.0]
               ),
        expand("data/plots/{ndof}/{model_layout}_{integrator}_initial/{n_samples}_{boundary}_{E_subtract_or_descr}/energy/energy_{t_stop}_{steps}_{ainvsquared}.png"
               , ndof=[8]
               , model_layout=["C6L4","D6L8", "C3L4"]
               , integrator="Euler1"
               , n_samples=[2000]
               , boundary=10
               , t_stop=1.4
               , steps=[700]
               , ainvsquared=[1.0, 2, 5, 10, 20, 40]
               , E_subtract_or_descr=["SCE", "APN", 6.5, 10.0]
               ),
        expand("data/plots/{ndof}/{model_layout}_{integrator}_initial/{n_samples}_{boundary}_{E_subtract_or_descr}/manifold_eror/mfe_{t_stop}_{steps}_{ainvsquared}.png"
               , ndof=[8]
               , model_layout=["C6L4","D6L8", "C3L4"]
               , integrator="Euler1"
               , n_samples=[2000]
               , boundary=10
               , t_stop=1.4
               , steps=[700]
               , ainvsquared=[1.0, 2, 5, 10, 20, 40]
               , E_subtract_or_descr=["SCE", "APN", 6.5, 10.0]
               ),


rule imaginary_time_evolution_Econstsubtract:
    resources:
        runtime=460,
        slurm_extra="'--gres=gpu:a40:1'",
    params:
        module_path = os.path.abspath("./lib"),
        svd_cut = 1e-4,
        regulator = 1e-3,
        seed = 0xdeadbeef,
        sampling_sigma = 1,
        sampling_decorr = 5,
        sampling_equilibrate = 50,
        save_every_t = 0.1,
        initial_time = 0.1,
        initial_undersampling = 2,
        initial_boundary_factor = 0.6,
        initial_stepwidth_factor = 0.5,
    output:
        "data/{ndof}/{model_layout}_{integrator}_initial/{n_samples}_{boundary}_{E_subtract}/models/model_{t_stop}_{steps}_{ainvsquared}.pt",
        "data/{ndof}/{model_layout}_{integrator}_initial/{n_samples}_{boundary}_{E_subtract}/energy/energy_{t_stop}_{steps}_{ainvsquared}.npy",
        "data/{ndof}/{model_layout}_{integrator}_initial/{n_samples}_{boundary}_{E_subtract}/manifold_eror/mfe_{t_stop}_{steps}_{ainvsquared}.npy",
    script:
        "scripts/01_A_ite_const_subtract.py"



rule ite_plot_ie_E_subtract:
    threads: 1
    localrule: True
    params:
        module_path = os.path.abspath("./lib"),
        mplstyle = os.path.abspath("./assets/notes.mplstyle"),
    resources:
        threads = 1,
    input:
        "data/{ndof}/{model_layout}_{integrator}_initial/{n_samples}_{boundary}_{E_subtract_or_descr}/manifold_eror/mfe_{t_stop}_{steps}_{ainvsquared}.npy",
    output:
        "data/plots/{ndof}/{model_layout}_{integrator}_initial/{n_samples}_{boundary}_{E_subtract_or_descr}/manifold_eror/mfe_{t_stop}_{steps}_{ainvsquared}.png",
    script:
        "scripts/plots/01/integration_error.py"

rule ite_plot_E_E_subtract:
    threads: 1
    localrule: True
    params:
        module_path = os.path.abspath("./lib"),
        mplstyle = os.path.abspath("./assets/notes.mplstyle"),
    resources:
        threads = 1,
    input:
        "data/{ndof}/{model_layout}_{integrator}_initial/{n_samples}_{boundary}_{E_subtract_or_descr}/energy/energy_{t_stop}_{steps}_{ainvsquared}.npy",
    output:
        "data/plots/{ndof}/{model_layout}_{integrator}_initial/{n_samples}_{boundary}_{E_subtract_or_descr}/energy/energy_{t_stop}_{steps}_{ainvsquared}.png",
    script:
        "scripts/plots/01/energy.py"



rule imaginary_time_evolution_Ecurrsubtract:
    resources:
        runtime=460,
        slurm_extra="'--gres=gpu:a40:1'",
    params:
        module_path = os.path.abspath("./lib"),
        svd_cut = 1e-4,
        regulator = 1e-3,
        seed = 0xdeadbeef,
        sampling_sigma = 1,
        sampling_decorr = 5,
        sampling_equilibrate = 50,
        save_every_t = 0.1,
        initial_time = 0.1,
        initial_undersampling = 2,
        initial_boundary_factor = 0.6,
        initial_stepwidth_factor = 0.5,
        e_subtract_damping=0.95,
    output:
        "data/{ndof}/{model_layout}_{integrator}_initial/{n_samples}_{boundary}_SCE/models/model_{t_stop}_{steps}_{ainvsquared}.pt",
        "data/{ndof}/{model_layout}_{integrator}_initial/{n_samples}_{boundary}_SCE/energy/energy_{t_stop}_{steps}_{ainvsquared}.npy",
        "data/{ndof}/{model_layout}_{integrator}_initial/{n_samples}_{boundary}_SCE/manifold_eror/mfe_{t_stop}_{steps}_{ainvsquared}.npy",
        "data/{ndof}/{model_layout}_{integrator}_initial/{n_samples}_{boundary}_SCE/energy/esubtract_{t_stop}_{steps}_{ainvsquared}.npy",
    script:
        "scripts/01_B_ite_ecurr_subtract.py"




rule imaginary_time_evolution_normpreserve:
    resources:
        runtime=460,
        slurm_extra="'--gres=gpu:a40:1'",
    params:
        module_path = os.path.abspath("./lib"),
        svd_cut = 1e-4,
        regulator = 1e-3,
        seed = 0xdeadbeef,
        sampling_sigma = 1,
        sampling_decorr = 5,
        sampling_equilibrate = 50,
        save_every_t = 0.1,
        initial_time = 0.1,
        initial_undersampling = 2,
        initial_boundary_factor = 0.6,
        initial_stepwidth_factor = 0.5,
        esubtract_dampening=0.01,
    output:
        "data/{ndof}/{model_layout}_{integrator}_initial/{n_samples}_{boundary}_APN/models/model_{t_stop}_{steps}_{ainvsquared}.pt",
        "data/{ndof}/{model_layout}_{integrator}_initial/{n_samples}_{boundary}_APN/energy/energy_{t_stop}_{steps}_{ainvsquared}.npy",
        "data/{ndof}/{model_layout}_{integrator}_initial/{n_samples}_{boundary}_APN/manifold_eror/mfe_{t_stop}_{steps}_{ainvsquared}.npy",
        "data/{ndof}/{model_layout}_{integrator}_initial/{n_samples}_{boundary}_APN/energy/esubtract_{t_stop}_{steps}_{ainvsquared}.npy",
        "data/{ndof}/{model_layout}_{integrator}_initial/{n_samples}_{boundary}_APN/norm/snref_{t_stop}_{steps}_{ainvsquared}.npy",
    script:
        "scripts/01_C_ite_subtract_normpreserve.py"
