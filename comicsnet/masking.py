#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import jax
import jax.numpy as jnp


def robust_scale(x: jax.Array, min_scale: float) -> jax.Array:
    '''Estimate a robust residual scale with the median absolute deviation.'''

    median = jnp.median(x)
    mad = jnp.median(jnp.abs(x - median))
    return jnp.maximum(1.4826 * mad, min_scale)


def update_sparse_mask(
    cube: jax.Array,
    background: jax.Array,
    *,
    threshold_sigma: float,
    min_scale: float,
) -> jax.Array:
    '''Update the sparse-signal mask from standardized residuals.'''

    residual = cube - background
    scale = robust_scale(residual, min_scale)
    return jnp.abs(residual) > threshold_sigma * scale


def observed_weight(mask: jax.Array) -> jax.Array:
    '''Return weights for pixels used to fit the background.'''

    return 1.0 - mask.astype(jnp.float32)
