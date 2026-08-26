from __future__ import annotations

"""PCD-style residual scalar potential backend for discrete IPS.

This module adapts architectural ideas from `jatan12/PCD`'s
`ResidualMLPDenoiser` to the finite-state IPS setting:

* sinusoidal diffusion-time/noise embedding,
* condition projection,
* LayerNorm residual MLP blocks,
* SiLU activation.

The PCD repository does not ship a root LICENSE file in the snapshot inspected
for this project, so this file is an independent reimplementation of the
architecture pattern rather than a source copy.  The output is deliberately a
single scalar potential; IPS still forms edge scores externally as
g(y; context) - g(x; context).
"""

import math
import random
from typing import Any, List, Sequence, Tuple


class PCDResidualScalarNet:
    """Torch-backed PCD-style conditional residual MLP scalar oracle."""

    backend_name = "pcd_residual_scalar"

    def __init__(self, input_dim: int, hidden_units: int, rng: random.Random) -> None:
        torch, nn = _torch_modules()
        self.input_dim = int(input_dim)
        self.hidden_units = int(hidden_units) if hidden_units > 0 else 96
        self.rng = rng
        self.device = torch.device("cpu")
        seed = rng.randrange(1, 2_147_483_647)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.module = _PCDResidualScalarModule(
                input_dim=self.input_dim,
                dim_t=max(16, min(128, self.hidden_units)),
                width=max(16, self.hidden_units),
                depth=3,
            ).to(self.device)
        self.module.train()

    def to_dict(self) -> dict:
        torch, _ = _torch_modules()
        state = {
            key: value.detach().cpu().tolist()
            for key, value in self.module.state_dict().items()
        }
        return {
            "backend": self.backend_name,
            "input_dim": self.input_dim,
            "hidden_units": self.hidden_units,
            "state_dict": state,
        }

    @classmethod
    def from_dict(cls, payload: dict, rng: random.Random) -> "PCDResidualScalarNet":
        torch, _ = _torch_modules()
        net = cls(int(payload["input_dim"]), int(payload["hidden_units"]), rng)
        state = {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in payload["state_dict"].items()
        }
        net.module.load_state_dict(state)
        return net

    def predict(self, x: Sequence[float]) -> float:
        return float(self.predict_batch([x])[0])

    def predict_batch(self, inputs: Sequence[Sequence[float]]) -> List[float]:
        if not inputs:
            return []
        torch, _ = _torch_modules()
        self.module.eval()
        with torch.no_grad():
            x = torch.tensor(inputs, dtype=torch.float32, device=self.device)
            out = self.module(x)
        return out.detach().cpu().view(-1).tolist()

    def fit(
        self,
        inputs: Sequence[Sequence[float]],
        targets: Sequence[float],
        epochs: int,
        learning_rate: float,
    ) -> None:
        if not inputs or epochs <= 0 or learning_rate <= 0.0:
            return
        torch, _ = _torch_modules()
        self.module.train()
        x = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        y = torch.tensor(targets, dtype=torch.float32, device=self.device)
        optimizer = torch.optim.Adam(self.module.parameters(), lr=learning_rate)
        order = list(range(len(inputs)))
        batch_size = min(256, max(16, len(inputs)))
        for _ in range(epochs):
            self._shuffle_indices(order)
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                xb = x[idx]
                yb = y[idx]
                pred = self.module(xb).view(-1)
                loss = torch.mean(torch.clamp(pred - yb, -10.0, 10.0) ** 2)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.module.parameters(), 5.0)
                optimizer.step()

    def fit_mixed(
        self,
        inputs: Sequence[Sequence[float]],
        targets: Sequence[float],
        residual_pairs: Sequence[Tuple[Sequence[float], Sequence[float], float]],
        ranking_pairs: Sequence[Tuple[Sequence[float], Sequence[float], float]],
        hypercone_pairs: Sequence[Tuple[Sequence[float], Sequence[float], float]],
        epochs: int,
        learning_rate: float,
        flow_residual_weight: float = 1.0,
        ranking_weight: float = 0.0,
        hypercone_weight: float = 0.0,
        weight_norm_bound: float = 0.0,
        coverage_pairs: Sequence[Tuple[Sequence[float], Sequence[float], float]] = (),
        expert_pairs: Sequence[Tuple[Sequence[float], Sequence[float], float]] = (),
        coverage_weight: float = 0.0,
        expert_weight: float = 0.0,
    ) -> None:
        if epochs <= 0 or learning_rate <= 0.0:
            return
        torch, _ = _torch_modules()
        self.module.train()
        optimizer = torch.optim.Adam(self.module.parameters(), lr=learning_rate)
        state_x = torch.tensor(inputs, dtype=torch.float32, device=self.device) if inputs else None
        state_y = torch.tensor(targets, dtype=torch.float32, device=self.device) if targets else None
        pair_tensors = {
            "residual": _pair_tensors(residual_pairs, self.device),
            "ranking": _pair_tensors(ranking_pairs, self.device),
            "hypercone": _pair_tensors(hypercone_pairs, self.device),
            "coverage": _pair_tensors(coverage_pairs, self.device),
            "expert": _pair_tensors(expert_pairs, self.device),
        }
        for _ in range(epochs):
            loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            if state_x is not None and state_y is not None and len(state_x) > 0:
                pred = self.module(state_x).view(-1)
                loss = loss + torch.mean(torch.clamp(pred - state_y, -10.0, 10.0) ** 2)
            loss = loss + flow_residual_weight * self._residual_loss(pair_tensors["residual"])
            loss = loss + ranking_weight * self._margin_loss(pair_tensors["ranking"])
            loss = loss + hypercone_weight * self._margin_loss(pair_tensors["hypercone"])
            loss = loss + coverage_weight * self._margin_loss(pair_tensors["coverage"])
            loss = loss + expert_weight * self._margin_loss(pair_tensors["expert"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), 5.0)
            optimizer.step()
            self.clip_weight_norms(weight_norm_bound)

    def clip_weight_norms(self, bound: float) -> None:
        if bound <= 0.0:
            return
        torch, nn = _torch_modules()
        with torch.no_grad():
            for module in self.module.modules():
                if isinstance(module, nn.Linear):
                    sigma = torch.linalg.matrix_norm(module.weight, ord=2)
                    if float(sigma) > bound:
                        module.weight.mul_(bound / max(float(sigma), 1e-12))
                    if module.bias is not None:
                        module.bias.clamp_(-bound, bound)

    def spectral_diagnostics(self, bound: float = 0.0) -> dict:
        torch, nn = _torch_modules()
        norms = []
        max_bias = 0.0
        with torch.no_grad():
            for module in self.module.modules():
                if isinstance(module, nn.Linear):
                    norms.append(float(torch.linalg.matrix_norm(module.weight, ord=2)))
                    if module.bias is not None:
                        max_bias = max(max_bias, float(module.bias.abs().max()))
        lipschitz_proxy = 1.0
        for norm in norms:
            lipschitz_proxy *= norm
        active_bound = float(bound) if bound > 0.0 else 0.0
        max_excess = max((norm / active_bound for norm in norms), default=0.0) if active_bound > 0.0 else 0.0
        return {
            "backend": self.backend_name,
            "w1_spectral_norm": norms[0] if norms else 0.0,
            "w2_euclidean_norm": norms[-1] if norms else 0.0,
            "hidden_spectral_norms": norms[1:-1],
            "max_spectral_norm": max(norms, default=0.0),
            "lipschitz_proxy": lipschitz_proxy,
            "max_abs_bias": max_bias,
            "clip_bound": active_bound,
            "max_excess_ratio": max_excess,
            "clip_active": bool(active_bound > 0.0 and max_excess > 1.000001),
        }

    def _residual_loss(self, tensors: Tuple[Any, Any, Any]) -> Any:
        torch, _ = _torch_modules()
        first, second, value = tensors
        if first is None:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)
        delta = self.module(second).view(-1) - self.module(first).view(-1)
        return torch.mean(torch.clamp(delta - value, -10.0, 10.0) ** 2)

    def _margin_loss(self, tensors: Tuple[Any, Any, Any]) -> Any:
        torch, _ = _torch_modules()
        first, second, margin = tensors
        if first is None:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)
        violation = margin + self.module(first).view(-1) - self.module(second).view(-1)
        return torch.mean(torch.relu(violation) ** 2)

    def _shuffle_indices(self, indices: List[int]) -> None:
        for idx in range(len(indices) - 1, 0, -1):
            swap = self.rng.randrange(idx + 1)
            indices[idx], indices[swap] = indices[swap], indices[idx]


class _PCDResidualScalarModule:
    pass


def _build_module_classes() -> None:
    global _PCDResidualScalarModule
    import torch
    import torch.nn as nn

    class SinusoidalPosEmb(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.dim = dim

        def forward(self, x):
            half = max(1, self.dim // 2)
            if half == 1:
                freqs = torch.ones(1, dtype=x.dtype, device=x.device)
            else:
                scale = math.log(10000.0) / max(1, half - 1)
                freqs = torch.exp(torch.arange(half, dtype=x.dtype, device=x.device) * -scale)
            emb = x.view(-1, 1) * freqs.view(1, -1)
            out = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
            if out.shape[-1] < self.dim:
                out = torch.nn.functional.pad(out, (0, self.dim - out.shape[-1]))
            return out[..., : self.dim]

    class ResidualBlock(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.ln = nn.LayerNorm(dim)
            self.linear = nn.Linear(dim, dim)

        def forward(self, x):
            return x + self.linear(torch.nn.functional.silu(self.ln(x)))

    class PCDResidualScalarModule(nn.Module):
        def __init__(self, input_dim: int, dim_t: int, width: int, depth: int) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.proj = nn.Linear(input_dim, dim_t)
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim_t),
                nn.Linear(dim_t, dim_t),
                nn.SiLU(),
                nn.Linear(dim_t, dim_t),
            )
            self.input_up = nn.Linear(dim_t, width) if dim_t != width else nn.Identity()
            self.blocks = nn.ModuleList([ResidualBlock(width) for _ in range(depth)])
            self.final_ln = nn.LayerNorm(width)
            self.final = nn.Linear(width, 1)

        def forward(self, x):
            time_value = self._time_condition(x)
            h = self.proj(x) + self.time_mlp(time_value)
            h = self.input_up(h)
            for block in self.blocks:
                h = block(h)
            return self.final(torch.nn.functional.silu(self.final_ln(h))).view(-1)

        def _time_condition(self, x):
            if self.input_dim >= 18:
                w0 = x[:, 4]
                w1 = x[:, 5]
                gap = torch.clamp(x[:, 8], 0.0, 1.0)
                extreme = torch.clamp(x[:, 10], 0.0, 1.0)
                return torch.clamp(0.15 + 0.45 * torch.abs(w0 - w1) + 0.25 * gap + 0.15 * extreme, 0.0, 1.0)
            if self.input_dim >= 6:
                w0 = x[:, 2]
                w1 = x[:, 3]
                hv = torch.clamp(x[:, 4], 0.0, 1.0)
                return torch.clamp(0.20 + 0.55 * torch.abs(w0 - w1) + 0.25 * hv, 0.0, 1.0)
            return torch.full((x.shape[0],), 0.5, dtype=x.dtype, device=x.device)

    _PCDResidualScalarModule = PCDResidualScalarModule


def _torch_modules() -> Tuple[Any, Any]:
    # The Windows ssm_env contains both Conda's and PyTorch's Intel OpenMP
    # runtimes.  Loading NumPy first selects the Conda runtime consistently;
    # the reverse order aborts when NumPy is imported later.
    import numpy as _numpy
    import torch
    import torch.nn as nn

    _ = _numpy
    global _PCDResidualScalarModule
    if _PCDResidualScalarModule is _PCDResidualScalarNetPlaceholder:
        _build_module_classes()
    return torch, nn


class _PCDResidualScalarNetPlaceholder:
    pass


_PCDResidualScalarModule = _PCDResidualScalarNetPlaceholder


def _pair_tensors(
    pairs: Sequence[Tuple[Sequence[float], Sequence[float], float]],
    device: Any,
) -> Tuple[Any, Any, Any]:
    if not pairs:
        return None, None, None
    torch, _ = _torch_modules()
    first = torch.tensor([pair[0] for pair in pairs], dtype=torch.float32, device=device)
    second = torch.tensor([pair[1] for pair in pairs], dtype=torch.float32, device=device)
    value = torch.tensor([pair[2] for pair in pairs], dtype=torch.float32, device=device)
    return first, second, value
