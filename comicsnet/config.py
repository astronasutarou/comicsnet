#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FitConfig:
    '''Configuration for the background/sparse decomposition loop.'''

    outer_steps: int = 5
    inner_steps: int = 100
    learning_rate: float = 1.0e-3
    beta: float = 1.0e-4
    threshold_sigma: float = 5.0
    min_scale: float = 1.0e-6
    hidden_channels: int = 8
    latent_channels: int = 2
    seed: int = 0
    standardize: bool = True
    update_mask_each_outer_step: bool = True
