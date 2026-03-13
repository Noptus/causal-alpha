"""Allocation helpers and backtest mechanics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


EPS = 1e-8


def _cap_mix(weights: np.ndarray, max_weight: float) -> np.ndarray:
    if weights.size == 0:
        return weights
    uniform = np.full_like(weights, 1.0 / len(weights))
    weight_max = float(weights.max())
    if weight_max <= max_weight + EPS:
        return weights
    denominator = max(weight_max - (1.0 / len(weights)), EPS)
    lam = np.clip((weight_max - max_weight) / denominator, 0.0, 1.0)
    mixed = (1.0 - lam) * weights + lam * uniform
    mixed /= mixed.sum()
    return mixed


def weights_from_scores(
    scores: np.ndarray,
    available: np.ndarray,
    *,
    top_k: int,
    max_weight: float,
    temperature: float = 1.0,
) -> np.ndarray:
    mask = available.astype(bool)
    output = np.zeros_like(scores, dtype=float)
    if mask.sum() == 0:
        return output
    k = min(top_k, int(mask.sum()))
    masked_scores = np.where(mask, scores, -np.inf)
    selected = np.argpartition(masked_scores, -k)[-k:]
    selected_scores = masked_scores[selected]
    selected_scores = selected_scores - np.max(selected_scores)
    softmax = np.exp(selected_scores / max(temperature, EPS))
    softmax /= softmax.sum()
    output[selected] = _cap_mix(softmax, max_weight)
    return output


def weights_from_scores_torch(
    scores: torch.Tensor,
    available: torch.Tensor,
    *,
    top_k: int,
    max_weight: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    output = torch.zeros_like(scores)
    valid_idx = torch.nonzero(available > 0.5, as_tuple=False).flatten()
    if valid_idx.numel() == 0:
        return output
    k = min(top_k, int(valid_idx.numel()))
    masked_scores = torch.where(available > 0.5, scores, torch.full_like(scores, -1e9))
    top_values, top_positions = torch.topk(masked_scores, k)
    softmax = torch.softmax(top_values / max(temperature, EPS), dim=0)
    uniform = torch.full_like(softmax, 1.0 / k)
    weight_max = torch.max(softmax)
    denominator = torch.clamp(weight_max - (1.0 / k), min=EPS)
    lam = torch.clamp((weight_max - max_weight) / denominator, min=0.0, max=1.0)
    mixed = (1.0 - lam) * softmax + lam * uniform
    mixed = mixed / torch.clamp(mixed.sum(), min=EPS)
    output[top_positions] = mixed
    return output


def smooth_weights_from_scores_torch(
    scores: torch.Tensor,
    available: torch.Tensor,
    *,
    max_weight: float,
    temperature: float = 0.5,
) -> torch.Tensor:
    valid_idx = torch.nonzero(available > 0.5, as_tuple=False).flatten()
    output = torch.zeros_like(scores)
    if valid_idx.numel() == 0:
        return output
    masked_scores = torch.where(available > 0.5, scores, torch.full_like(scores, -1e9))
    softmax = torch.softmax(masked_scores / max(temperature, EPS), dim=0)
    k = int(valid_idx.numel())
    uniform = torch.full_like(softmax, 1.0 / k)
    weight_max = torch.max(softmax[valid_idx])
    denominator = torch.clamp(weight_max - (1.0 / k), min=EPS)
    lam = torch.clamp((weight_max - max_weight) / denominator, min=0.0, max=1.0)
    mixed = (1.0 - lam) * softmax + lam * uniform
    mixed = mixed * (available > 0.5)
    mixed = mixed / torch.clamp(mixed.sum(), min=EPS)
    output = mixed
    return output


@dataclass(slots=True)
class BacktestResult:
    net_returns: np.ndarray
    gross_returns: np.ndarray
    turnovers: np.ndarray
    weights: np.ndarray


def evaluate_weight_path(
    weights: np.ndarray,
    future_returns: np.ndarray,
    *,
    transaction_cost: float,
) -> BacktestResult:
    previous = np.zeros(weights.shape[1], dtype=float)
    net_returns = np.zeros(weights.shape[0], dtype=float)
    gross_returns = np.zeros_like(net_returns)
    turnovers = np.zeros_like(net_returns)
    for t in range(weights.shape[0]):
        gross = float(np.dot(weights[t], future_returns[t]))
        turnover = float(np.abs(weights[t] - previous).sum())
        net = gross - transaction_cost * turnover
        gross_returns[t] = gross
        turnovers[t] = turnover
        net_returns[t] = net
        previous = weights[t]
    return BacktestResult(
        net_returns=net_returns,
        gross_returns=gross_returns,
        turnovers=turnovers,
        weights=weights,
    )
