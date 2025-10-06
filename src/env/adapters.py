"""Adapters that feed market data into the trading environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from src.data.loader import SlidingWindowDataset, WindowSample


def _flatten(sample: WindowSample) -> np.ndarray:
    return sample.features.reshape(-1).astype(np.float32, copy=False)

@dataclass(slots=True)
class WindowedDataAdapter:
    """Adapter that streams preprocessed sliding-window samples into the environment."""

    dataset: SlidingWindowDataset
    split: str = "train"
    flatten: bool = True
    samples: List[WindowSample] = field(init=False)
    observation_shape: Tuple[int, ...] = field(init=False)
    asset_dim: int = field(init=False)
    _cursor: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.samples = list(self.dataset.iter_split(self.split))
        if not self.samples:
            raise ValueError(f"No samples available for split '{self.split}'")
        first = self.samples[0]
        if self.flatten:
            self.observation_shape = (first.features.size,)
        else:
            self.observation_shape = first.features.shape
        self.asset_dim = 1  # each sample represents a single symbol trajectory
        self._cursor = 0

    def reset(self, *, seed: int | None = None):  # pylint: disable=unused-argument
        self._cursor = 0
        sample = self.samples[self._cursor]
        observation = self._format_features(sample)
        info: Dict[str, np.ndarray] = {
            "returns": np.zeros(self.asset_dim, dtype=np.float64),
            "mask": sample.mask.astype(np.float32, copy=False),
        }
        return observation, info

    def step(self, action: np.ndarray):  # pylint: disable=unused-argument
        current = self.samples[self._cursor]
        realised_return = np.array([float(current.target[0])], dtype=np.float64)

        self._cursor += 1
        terminated = self._cursor >= len(self.samples)

        if terminated:
            observation = np.zeros(self.observation_shape, dtype=np.float32)
            next_mask = np.ones(current.mask.shape, dtype=np.float32)
        else:
            next_sample = self.samples[self._cursor]
            observation = self._format_features(next_sample)
            next_mask = next_sample.mask.astype(np.float32, copy=False)

        info = {
            "returns": realised_return,
            "mask": current.mask.astype(np.float32, copy=False),
            "next_mask": next_mask,
        }
        return observation, info, terminated

    def _format_features(self, sample: WindowSample) -> np.ndarray:
        if self.flatten:
            return _flatten(sample)
        return sample.features.astype(np.float32, copy=False)


__all__ = ["WindowedDataAdapter"]
