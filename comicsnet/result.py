#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .config import FitConfig


@dataclass(frozen=True)
class FitResult:
    '''Result returned by :func:`comicsnet.fit`.'''

    background: jax.Array
    uncertainty: jax.Array
    mask: jax.Array
    sparse: jax.Array
    model: Any
    config: FitConfig
    losses: tuple[float, ...]


def make_sparse(
    cube: jax.Array,
    background: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    '''Return residual signal only where the sparse mask is active.'''

    return jnp.where(mask, cube - background, 0.0)
