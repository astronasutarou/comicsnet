#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp

from .basis import BasisAE


class AdaptiveBasisAE(BasisAE):
    """BasisAE with state-dependent, bounded basis perturbations.

    The BasisAE encoder supplies both coefficient and modulation branches.
    For each basis, a latent-dependent query attends to variation_dim keys
    and mixes the corresponding detector-fixed perturbation images.
    Attention is single-head scaled dot product, normalized over variants.
    Values are perturbation images themselves, without a dense image-sized
    value/output projection. Coefficients remain unrestricted in sign.

    basis_delta contains unconstrained parameters. tanh bounds each variant,
    then modulation_scale scales it by the current RMS of its base image
    (floored at 1e-6). This bounds the absolute pixel correction, not its
    fraction of the local pixel value. Zero modulation_scale disables the
    correction. Frame-independent log variance is inherited from BasisAE.
    """

    query_layer: eqx.nn.Linear
    modulation_keys: jax.Array
    basis_delta: jax.Array
    variation_dim: int = eqx.field(static=True)
    modulation_scale: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        frame_shape: tuple[int, int],
        latent_dim: int,
        basis_dim: int,
        hidden_dim: int,
        key: jax.Array,
        variation_dim: int = 4,
        modulation_scale: float = 0.1,
        init_scale: float = 1.0e-3,
    ) -> None:
        height, width = frame_shape
        if min(height, width, latent_dim, basis_dim, hidden_dim) < 1:
            raise ValueError('frame and model dimensions must be positive')
        if variation_dim < 1:
            raise ValueError('variation_dim must be positive')
        if not math.isfinite(modulation_scale) or modulation_scale < 0:
            raise ValueError(
                'modulation_scale must be finite and non-negative',
            )
        keys = jax.random.split(key, 4)
        super().__init__(
            frame_shape=frame_shape,
            latent_dim=latent_dim,
            basis_dim=basis_dim,
            hidden_dim=hidden_dim,
            key=keys[0],
            init_scale=init_scale,
        )
        self.query_layer = eqx.nn.Linear(
            latent_dim, basis_dim * hidden_dim, key=keys[1],
        )
        self.modulation_keys = jax.random.normal(
            keys[2], (basis_dim, variation_dim, hidden_dim),
        )
        self.basis_delta = jax.random.normal(
            keys[3], (basis_dim, variation_dim, height, width),
        )
        self.variation_dim = variation_dim
        self.modulation_scale = modulation_scale

    def modulation_weights(self, latent: jax.Array) -> jax.Array:
        """Return basis-by-variant attention probabilities."""
        query = self.query_layer(jnn.gelu(latent)).reshape(self.basis_dim, -1)
        logits = jnp.einsum('kd,krd->kr', query, self.modulation_keys)
        return jnn.softmax(logits / math.sqrt(query.shape[-1]), axis=-1)

    def modulated_basis(self, latent: jax.Array) -> jax.Array:
        """Return the state-dependent basis with shape (basis_dim, y, x)."""
        weights = self.modulation_weights(latent)
        delta = jnp.einsum('kr,kryx->kyx', weights, jnp.tanh(self.basis_delta))
        mean_square = jnp.mean(self.basis ** 2, axis=(-2, -1), keepdims=True)
        rms = jnp.sqrt(jnp.maximum(mean_square, 1.0e-12))
        return self.basis + self.modulation_scale * rms * delta

    def decode(self, latent: jax.Array) -> tuple[jax.Array, jax.Array]:
        coeff = self.coeff_layer(jnn.gelu(latent))
        basis = self.modulated_basis(latent)
        frame = self.bias + jnp.einsum('k,kyx->yx', coeff, basis)
        return frame[jnp.newaxis, ...], self.out_logvar[jnp.newaxis, ...]
