import torch


def get_svd_inverse_cut(M, threshold=1e-4):
    U, S, Vh = torch.linalg.svd(M)

    S_inv = 1.0 / S
    S_inv[torch.where(S.abs() < threshold)] = 0
    S_inv = torch.diag(S_inv)

    Minv = Vh.conj().T @ S_inv @ U.conj().T
    return Minv

def get_importance_points(model, x0, npoints, ndecorr=1, metropolis_step_sigma=1.0):
    result = torch.zeros(npoints, *(x0.shape), dtype=x0.dtype)
    x = x0

    for i in range(npoints):
        for _ in range(ndecorr):
            xbar = x + torch.randn_like(x) * metropolis_step_sigma
            with torch.no_grad():
                pbar = torch.abs(model.forward(xbar))**2
                p = torch.abs(model.forward(x))**2
            alpha = pbar / p
            if torch.rand(1) < alpha:
                x = xbar
        result[i] = x
    return result

from collections import defaultdict
from time import perf_counter_ns
import datetime
from contextlib import contextmanager


def ns_to_time(ns):
    total_seconds = ns / 1e9  # convert nanoseconds to seconds
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    microseconds = int((ns % 1_000_000_000) / 1000)  # remaining microseconds

    return datetime.time(hour=hours % 24, minute=minutes, second=seconds, microsecond=microseconds)



class TimingLogger:
    def __init__(self, name):
        self.name = name
        self.individual_timings = defaultdict(int)
        self.t_start = None

    def start(self):
        self.t_start = perf_counter_ns()

    def get_elapsed_since_start(self):
        stop = perf_counter_ns()
        elapsed = stop - self.t_start
        return elapsed

    def get_eta(self, iteration_current, total_iterations):
        elapsed = self.get_elapsed_since_start()
        progress = iteration_current / total_iterations
        return elapsed / progress

    def get_formatted_eta(self, iteration_current, total_iterations):
        elapsed = ns_to_time(self.get_elapsed_since_start())
        eta = ns_to_time(self.get_eta(iteration_current, total_iterations))
        return f"elpsd: {elapsed.isoformat()} || eta: {eta.isoformat()}"

    @contextmanager
    def time_section(self, section_name):
        t_section_start = perf_counter_ns()
        try:
            yield
        finally:
            t_stop = perf_counter_ns()
            self.individual_timings[section_name] += t_stop - t_section_start

    def make_report(self):
        maxlen = max(len(k) for k in self.individual_timings.keys())
        
        for k,v in sorted(self.individual_timings.items(), key=lambda x: x[1]):
            print(self.name, "-", k.ljust(maxlen+1), ":", ns_to_time(v).isoformat())

