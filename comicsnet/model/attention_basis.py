#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp

from .basis import BasisAE


class AttentionBasisAE(BasisAE):
    """Cross-attention encoder with detector-fixed basis images.

    Each pixel supplies [weighted value, weight, x coordinate, y coordinate].
    Coordinates span [0, 1]. Fixed identity queries are projected inside
    MultiheadAttention, so no separate trainable query array is needed.
    num_queries controls the number of pooled tokens; hidden_dim is their
    feature width and must be divisible by num_heads. The pooled tokens
    map to a latent_dim vector, then to unrestricted basis coefficients.

    Zero-weight pixels are excluded from attention. Positive fractional
    weights scale the input value and are also supplied as features; they
    do not directly multiply the attention probabilities. With no observed
    pixels, pooled features are zero and prediction uses learned biases.
    The basis decoder and frame-independent log variance match BasisAE.
    """

    attention: eqx.nn.MultiheadAttention
    num_queries: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        frame_shape: tuple[int, int],
        latent_dim: int,
        basis_dim: int,
        hidden_dim: int,
        key: jax.Array,
        num_queries: int = 4,
        num_heads: int = 1,
        init_scale: float = 1.0e-3,
    ) -> None:
        height, width = frame_shape
        if min(height, width, latent_dim, basis_dim, hidden_dim) < 1:
            raise ValueError('frame and model dimensions must be positive')
        if num_queries < 1 or num_heads < 1:
            raise ValueError('num_queries and num_heads must be positive')
        if hidden_dim % num_heads:
            raise ValueError('hidden_dim must be divisible by num_heads')
        keys = jax.random.split(key, 5)
        self.encode_layer0 = eqx.nn.Linear(4, hidden_dim, key=keys[0])
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=num_queries,
            key_size=hidden_dim,
            value_size=hidden_dim,
            output_size=hidden_dim,
            qk_size=hidden_dim // num_heads,
            vo_size=hidden_dim // num_heads,
            key=keys[1],
        )
        self.encode_layer1 = eqx.nn.Linear(
            num_queries * hidden_dim, latent_dim, key=keys[2],
        )
        self.coeff_layer = eqx.nn.Linear(
            latent_dim, basis_dim, key=keys[3],
        )
        self.bias = jnp.zeros(frame_shape, dtype=jnp.float32)
        self.basis = init_scale * jax.random.normal(
            keys[4], (basis_dim, height, width),
        )
        self.out_logvar = jnp.zeros(frame_shape, dtype=jnp.float32)
        self.frame_shape = frame_shape
        self.latent_dim = latent_dim
        self.basis_dim = basis_dim
        self.num_queries = num_queries
        self.use_kl = False

    def encode(
        self,
        x: jax.Array,
        weight: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array]:
        if x.shape != (1, *self.frame_shape):
            raise ValueError('x must have shape (1, *frame_shape)')
        if weight is None:
            weight = jnp.ones_like(x)
        if weight.shape != x.shape:
            raise ValueError('weight must have the same shape as x')
        height, width = self.frame_shape
        y_coord, x_coord = jnp.meshgrid(
            jnp.linspace(0.0, 1.0, height, dtype=x.dtype),
            jnp.linspace(0.0, 1.0, width, dtype=x.dtype),
            indexing='ij',
        )
        observed = weight.ravel() > 0
        values = jnp.where(weight > 0, x, 0.0) * weight
        features = jnp.stack(
            (values.ravel(), weight.ravel(), x_coord.ravel(), y_coord.ravel()),
            axis=-1,
        )
        features = jnn.gelu(jax.vmap(self.encode_layer0)(features))
        queries = jnp.eye(self.num_queries, dtype=features.dtype)

        # Keep softmax defined for empty frames, then discard its output.
        has_observed = jnp.any(observed)
        safe_observed = observed.at[0].set(observed[0] | ~has_observed)
        mask = jnp.broadcast_to(
            safe_observed, (self.num_queries, observed.size),
        )
        pooled = self.attention(queries, features, features, mask=mask)
        pooled = jnp.where(has_observed, pooled, 0.0)
        latent = self.encode_layer1(pooled.ravel())
        return latent, jnp.zeros_like(latent)
