#!/usr/bin/env python
"""CLI entrypoint for training PPO baseline."""

import sys

import gymnasium as gym

sys.modules.setdefault("gym", gym)

from src.pipelines.training import main


if __name__ == "__main__":
    main()
