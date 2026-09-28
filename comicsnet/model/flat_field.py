#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Autoencoder for a fixed flat field and variable uniform illumination."""

from __future__ import annotations

import math
import operator

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp


class FlatFieldAE(eqx.Module):
    """Reconstruct frames as a shared bias plus flux times a shared flat.

    Bias, flat perturbations, and output log variance are learned arrays
    shared across frames. The encoder receives the weighted image,
    observation weights, the full bias image, and the normalized flat as
    four flattened channels. Fractional weights are retained as supplied.
    Its scalar output is unrestricted in sign to support standardized data.
    The latent flux is also the sole coefficient of the flat image.

    Parameters:
        frame_shape (tuple[int, int]):
            Detector shape as (height, width).
        hidden_dim (int):
            Width of each encoder hidden layer. Defaults to 8. Unused when
            depth is zero.
        depth (int):
            Number of encoder hidden layers. Zero uses one linear layer.
            Defaults to 2. Hidden activations are GELU; the output is linear.
        flat_variation_scale (float):
            Scale of the mean-centered tanh flat perturbation. Defaults to
            0.05. The flat has mean one and deviations bounded by twice this
            scale, not by the scale itself. Must be finite and in [0, 0.5)
            to guarantee positive sensitivity. Zero fixes the flat to one.
        key (jax.Array):
            Random key for encoder initialization. Bias, dflat, and output
            log variance are initialized to zero.
    """

    encoder: eqx.nn.MLP
    bias: jax.Array
    dflat: jax.Array
    out_logvar: jax.Array
    frame_shape: tuple[int, int]
    hidden_dim: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    flat_variation_scale: float = eqx.field(static=True)
    use_kl: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        frame_shape: tuple[int, int],
        hidden_dim: int = 8,
        depth: int = 2,
        flat_variation_scale: float = 0.05,
        key: jax.Array,
    ) -> None:
        height, width = (operator.index(size) for size in frame_shape)
        hidden_dim = operator.index(hidden_dim)
        depth = operator.index(depth)
        if min(height, width, hidden_dim) < 1:
            raise ValueError('frame and hidden dimensions must be positive')
        if depth < 0:
            raise ValueError('depth must be non-negative')
        if (not math.isfinite(flat_variation_scale)
                or not 0 <= flat_variation_scale < 0.5):
            raise ValueError('flat_variation_scale must be in [0, 0.5)')

        self.frame_shape = (height, width)
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.flat_variation_scale = float(flat_variation_scale)
        self.use_kl = False
        self.encoder = eqx.nn.MLP(
            in_size=4 * height * width,
            out_size=1,
            width_size=hidden_dim,
            depth=depth,
            activation=jnn.gelu,
            key=key,
        )
        self.bias = jnp.zeros(self.frame_shape)
        self.dflat = jnp.zeros_like(self.bias)
        self.out_logvar = jnp.zeros_like(self.bias)

    @property
    def flat(self) -> jax.Array:
        """Return the positive, mean-one flat image with shape (y, x)."""
        variation = jnp.tanh(self.dflat)
        return 1.0 + self.flat_variation_scale * (
            variation - jnp.mean(variation)
        )

    def encode(
        self,
        x: jax.Array,
        weight: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array]:
        if weight is None:
            weight = jnp.ones_like(x)
        observed = jnp.where(weight == 0, 0.0, x) * weight
        augmented = jnp.concatenate(
            [
                observed,
                weight,
                self.bias[jnp.newaxis, ...],
                self.flat[jnp.newaxis, ...],
            ],
            axis=0,
        )
        flux = self.encoder(jnp.ravel(augmented))
        return flux, jnp.zeros_like(flux)

    def decode(self, flux: jax.Array) -> tuple[jax.Array, jax.Array]:
        frame = self.bias + flux[0] * self.flat
        return frame[jnp.newaxis, ...], self.out_logvar[jnp.newaxis, ...]

    def __call__(
        self,
        x: jax.Array,
        key: jax.Array,
        weight: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        del key
        flux, flux_logvar = self.encode(x, weight)
        mean, logvar = self.decode(flux)
        return mean, logvar, flux, flux_logvar

    def predict(
        self,
        x: jax.Array,
        weight: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array]:
        flux, _ = self.encode(x, weight)
        return self.decode(flux)
