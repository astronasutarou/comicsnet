#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import jax
import jax.numpy as jnp


def gaussian_nll(
    x: jax.Array,
    mean: jax.Array,
    logvar: jax.Array,
    weight: jax.Array,
) -> jax.Array:
    '''Masked Gaussian negative log likelihood.'''

    logvar = jnp.clip(logvar, -12.0, 8.0)
    nll = 0.5 * ((x - mean) ** 2 * jnp.exp(-logvar) + logvar)
    return jnp.sum(weight * nll) / jnp.maximum(jnp.sum(weight), 1.0)


def kl_normal(mean: jax.Array, logvar: jax.Array) -> jax.Array:
    '''KL divergence from diagonal normal posterior to a unit normal prior.'''

    variance = jnp.exp(logvar)
    kl = 0.5 * (variance + mean ** 2 - 1.0 - logvar)
    return jnp.mean(kl)
