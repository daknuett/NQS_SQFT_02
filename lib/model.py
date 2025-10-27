import torch
import numpy as np


import torch.nn.functional as F


class ConvModelRe(torch.nn.Module):
    def __init__(self, ndof, nlayer_broad, broad_width,
                 nl=torch.tanh, residual=True, kernel_size=3):
        super().__init__()
        self.residual = residual
        self.nl = nl
        self.kernel_size = kernel_size

        n_deep = nlayer_broad - 2
        if n_deep < 0:
            raise ValueError("need at least 2 broad layers")

        # Note: set padding=0 because we’ll handle circular padding manually
        first_conv = torch.nn.Conv1d(1, broad_width, kernel_size=kernel_size, padding=0, dtype=torch.double)
        last_conv  = torch.nn.Conv1d(broad_width, 1, kernel_size=kernel_size, padding=0, dtype=torch.double)

        self.deep_layers = torch.nn.ModuleList(
            [first_conv]
            + [torch.nn.Conv1d(broad_width, broad_width, kernel_size=kernel_size, padding=0, dtype=torch.double)
               for _ in range(n_deep)]
            + [last_conv]
        )

    def periodic_pad(self, x):
        pad = self.kernel_size // 2
        return F.pad(x, (pad, pad), mode="circular")

    def forward(self, features):
        x = features.unsqueeze(1)

        x = self.nl(self.deep_layers[0](self.periodic_pad(x)))
        for dl in self.deep_layers[1:-1]:
            if self.residual:
                x = self.nl(dl(self.periodic_pad(x))) + x
            else:
                x = self.nl(dl(self.periodic_pad(x)))
        x = self.nl(self.deep_layers[-1](self.periodic_pad(x)))
        return x

class ModelConvBL(torch.nn.Module):
    def __init__(self, ndof, nlayer_broad, broad_width, nl=torch.tanh, residual=True, kernel_size=3):
        super().__init__()
        self.end_model_linear = torch.nn.Linear(ndof, 1, dtype=torch.double)
        self.end_model_bilinear = torch.nn.Bilinear(ndof, ndof, 1, dtype=torch.double)
        self.mr = ConvModelRe(ndof, nlayer_broad, broad_width, nl=nl, residual=residual, kernel_size=kernel_size)
    
    def forward(self, features):
        x = self.mr(features)
        x = self.end_model_linear(x) + self.end_model_bilinear(x, x)
        return torch.exp(x).squeeze()

class ModelConvL(torch.nn.Module):
    def __init__(self, ndof, nlayer_broad, broad_width, nl=torch.tanh, residual=True, kernel_size=3):
        super().__init__()
        self.end_model_linear = torch.nn.Linear(ndof, 1, dtype=torch.double)
        self.mr = ConvModelRe(ndof, nlayer_broad, broad_width, nl=nl, residual=residual, kernel_size=kernel_size)
    
    def forward(self, features):
        x = self.mr(features)
        x = self.end_model_linear(x)
        return torch.exp(x).squeeze()


class ModelRe(torch.nn.Module):
    def __init__(self, ndof, nlayer_broad, broad_width, nl=torch.tanh, residual=True):
        super().__init__()
        self.residual = residual
        self.nl = nl
        n_deep = nlayer_broad - 2
        if n_deep < 0:
            raise ValueError("need at least 2 broad layers")
        first_broad = torch.nn.Linear(ndof, broad_width, dtype=torch.double)
        last_broad = torch.nn.Linear(broad_width, 1, dtype=torch.double)
        self.deep_layers = torch.nn.ModuleList(
            [first_broad]
            + [torch.nn.Linear(broad_width, broad_width, dtype=torch.double) for _ in range(n_deep)]
            + [last_broad]
        )

    def forward(self, features):
        features = self.deep_layers[0].forward(features)
        features = self.nl(features)
        for dl in self.deep_layers[1:-1]:
            if self.residual:
                features = self.nl(dl.forward(features)) + features
            else:
                features = self.nl(dl.forward(features))
        features = self.deep_layers[-1].forward(features)
        return features

class ModelDense(torch.nn.Module):
    def __init__(self, ndof, nlayer_broad, broad_width, nl=torch.tanh, residual=True):
        super().__init__()
        self.mr = ModelRe(ndof, nlayer_broad, broad_width, nl=nl, residual=residual)
    def forward(self, features):
        features = self.mr.forward(features)
        return torch.exp(features)

def get_M(sample_space, model, proj=lambda x: x):
    M = torch.zeros(sample_space.shape[0], get_n_params(model, proj), dtype=sample_space.dtype)
    for i, ni in enumerate(sample_space):
        for p in model.parameters():
            p.grad = None
        npsi = proj(model(torch.stack([ni])))
        npsi.backward()
        M[i,:] = get_grad(model, proj)

    return M

def get_n_params(model, proj=lambda x:x):
    return sum(np.prod(pi.shape) for pi in proj(model).parameters())
    
def get_param_shapes(model, proj=lambda x:x):
    return [p.shape for p in proj(model).parameters()]

def get_param_offsets(model, proj=lambda x:x):
    return np.concatenate([np.array([0]), np.cumsum(np.array([np.prod(pi.shape) for pi in proj(model).parameters()]))])

def get_grad(model, proj=lambda x:x):
    return torch.concat([p.grad.flatten().detach().clone() for p in proj(model).parameters()])

def get_params(model, proj=lambda x:x):
    return torch.concat([p.data.flatten().detach().clone() for p in proj(model).parameters()])

def reshape_params(params_flat, model, proj=lambda x: x):
    param_shapes = get_param_shapes(model, proj=proj)
    param_offsets = get_param_offsets(model, proj=proj)
    res = []
    for i, si in enumerate(param_shapes):
        pslice = params_flat[param_offsets[i]: param_offsets[i+1]]
        res.append(pslice.reshape(si))
    return res


def update_params(model, thetadot, h, proj=lambda x:x):
    params_reshaped = reshape_params(thetadot, model, proj=proj)
    for pi, dpi in zip(proj(model).parameters(), params_reshaped):
        pi.data += h * dpi


def decode_model_string(model_descr: str):
    if model_descr.startswith("D"):
        model_type = "dense"
        model_end = None
    elif model_descr.startswith("C"):
        model_type = "convolution"
    else:
        raise ValueError("model type must be 'D' or 'C'")

    if model_type == "dense":
        deep, broad = model_descr[1:].split("L")
    elif model_type == "convolution":
        if "L" in model_descr:
            model_end = "L"
        elif "B" in model_descr:
            model_end = "B"
        else:
            raise ValueError("expected model description to contain 'L' or 'B'")
        deep, broad = model_descr[1:].split(model_end)

    deep = int(deep)
    broad = int(broad)
    return (model_type, model_end), (deep, broad)
