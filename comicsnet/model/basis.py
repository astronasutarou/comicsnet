#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp


def _ones_weight(x: jax.Array) -> jax.Array:
    return jnp.ones_like(x)


def _mask_augmented_input(
    x: jax.Array,
    weight: jax.Array | None,
) -> jax.Array:
    if weight is None:
        weight = _ones_weight(x)

    return jnp.concatenate([x * weight, weight], axis=0)


class BasisAE(eqx.Module):
    '''Basis autoencoder for detector-fixed backgrounds.

    The encoder maps a full frame to a compact latent vector.  A small MLP
    then maps the latent vector to coefficients of detector-fixed basis
    images.
    '''

    encode_layer0: eqx.nn.Linear
    encode_layer1: eqx.nn.Linear
    coeff_layer: eqx.nn.Linear
    bias: jax.Array
    basis: jax.Array
    out_logvar: jax.Array
    frame_shape: tuple[int, int]
    latent_dim: int
    basis_dim: int
    use_kl: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        frame_shape: tuple[int, int],
        latent_dim: int,
        basis_dim: int,
        hidden_dim: int,
        key: jax.Array,
        init_scale: float = 1.0e-3,
    ) -> None:
        height, width = frame_shape
        n_pixels = height * width
        keys = jax.random.split(key, 4)

        self.encode_layer0 = eqx.nn.Linear(
            2 * n_pixels,
            hidden_dim,
            key=keys[0],
        )
        self.encode_layer1 = eqx.nn.Linear(
            hidden_dim,
            latent_dim,
            key=keys[1],
        )
        self.coeff_layer = eqx.nn.Linear(
            latent_dim,
            basis_dim,
            key=keys[2],
        )
        self.bias = jnp.zeros(frame_shape, dtype=jnp.float32)
        self.basis = init_scale * jax.random.normal(
            keys[3],
            (basis_dim, height, width),
        )
        self.out_logvar = jnp.zeros(frame_shape, dtype=jnp.float32)
        self.frame_shape = frame_shape
        self.latent_dim = latent_dim
        self.basis_dim = basis_dim
        self.use_kl = False

    def encode(
        self,
        x: jax.Array,
        weight: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        model_input = _mask_augmented_input(x, weight)
        h = jnn.gelu(self.encode_layer0(jnp.ravel(model_input)))
        latent = self.encode_layer1(h)
        latent_logvar = jnp.zeros_like(latent)
        return latent, latent_logvar

    def decode(self, latent: jax.Array) -> tuple[jax.Array, jax.Array]:
        coeff = self.coeff_layer(jnn.gelu(latent))
        frame = self.bias + jnp.einsum('k,kyx->yx', coeff, self.basis)
        return frame[jnp.newaxis, ...], self.out_logvar[jnp.newaxis, ...]

    def __call__(
        self,
        x: jax.Array,
        key: jax.Array,
        weight: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        del key
        latent, latent_logvar = self.encode(x, weight)
        mean, logvar = self.decode(latent)
        return mean, logvar, latent, latent_logvar

    def predict(
        self,
        x: jax.Array,
        weight: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        latent, _ = self.encode(x, weight)
        return self.decode(latent)


class BasisVAE(eqx.Module):
    '''Basis VAE for detector-fixed backgrounds.

    The encoder maps a full frame to a Gaussian latent distribution.  A small
    MLP maps sampled latent vectors to coefficients of detector-fixed basis
    images.
    '''

    encode_layer0: eqx.nn.Linear
    z_mean: eqx.nn.Linear
    z_logvar: eqx.nn.Linear
    coeff_layer: eqx.nn.Linear
    bias: jax.Array
    basis: jax.Array
    out_logvar: jax.Array
    frame_shape: tuple[int, int]
    latent_dim: int
    basis_dim: int
    use_kl: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        frame_shape: tuple[int, int],
        latent_dim: int,
        basis_dim: int,
        hidden_dim: int,
        key: jax.Array,
        init_scale: float = 1.0e-3,
    ) -> None:
        height, width = frame_shape
        n_pixels = height * width
        keys = jax.random.split(key, 5)

        self.encode_layer0 = eqx.nn.Linear(
            2 * n_pixels,
            hidden_dim,
            key=keys[0],
        )
        self.z_mean = eqx.nn.Linear(
            hidden_dim,
            latent_dim,
            key=keys[1],
        )
        self.z_logvar = eqx.nn.Linear(
            hidden_dim,
            latent_dim,
            key=keys[2],
        )
        self.coeff_layer = eqx.nn.Linear(
            latent_dim,
            basis_dim,
            key=keys[3],
        )
        self.bias = jnp.zeros(frame_shape, dtype=jnp.float32)
        self.basis = init_scale * jax.random.normal(
            keys[4],
            (basis_dim, height, width),
        )
        self.out_logvar = jnp.zeros(frame_shape, dtype=jnp.float32)
        self.frame_shape = frame_shape
        self.latent_dim = latent_dim
        self.basis_dim = basis_dim
        self.use_kl = True

    def encode(
        self,
        x: jax.Array,
        weight: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        model_input = _mask_augmented_input(x, weight)
        h = jnn.gelu(self.encode_layer0(jnp.ravel(model_input)))
        return self.z_mean(h), self.z_logvar(h)

    def decode(self, z: jax.Array) -> tuple[jax.Array, jax.Array]:
        coeff = self.coeff_layer(jnn.gelu(z))
        frame = self.bias + jnp.einsum('k,kyx->yx', coeff, self.basis)
        return frame[jnp.newaxis, ...], self.out_logvar[jnp.newaxis, ...]

    def __call__(
        self,
        x: jax.Array,
        key: jax.Array,
        weight: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        z_mean, z_logvar = self.encode(x, weight)
        eps = jax.random.normal(key, z_mean.shape)
        z = z_mean + jnp.exp(0.5 * z_logvar) * eps
        mean, logvar = self.decode(z)
        return mean, logvar, z_mean, z_logvar

    def predict(
        self,
        x: jax.Array,
        weight: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        z_mean, _ = self.encode(x, weight)
        return self.decode(z_mean)
