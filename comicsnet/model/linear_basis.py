#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


def _ones_weight(x: jax.Array) -> jax.Array:
    return jnp.ones_like(x)


def _fraction_normalized_input(
    x: jax.Array,
    weight: jax.Array | None,
) -> jax.Array:
    if weight is None:
        weight = _ones_weight(x)

    fraction = jnp.mean(weight)
    fraction = jnp.maximum(fraction, 1.0e-6)
    return x * weight / fraction


class LinearBasisAE(eqx.Module):
    '''Linear basis autoencoder for detector-fixed backgrounds.

    A frame is represented as ``bias + sum_k coeff[k] * basis[k]``.  This is
    close to a low-rank frame model, but the coefficients are inferred by a
    learnable linear encoder.
    '''

    encoder: eqx.nn.Linear
    bias: jax.Array
    basis: jax.Array
    out_logvar: jax.Array
    frame_shape: tuple[int, int]
    latent_dim: int
    use_kl: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        frame_shape: tuple[int, int],
        latent_dim: int,
        key: jax.Array,
        init_scale: float = 1.0e-3,
    ) -> None:
        height, width = frame_shape
        n_pixels = height * width
        encoder_key, basis_key = jax.random.split(key)

        self.encoder = eqx.nn.Linear(n_pixels, latent_dim, key=encoder_key)
        self.bias = jnp.zeros(frame_shape, dtype=jnp.float32)
        self.basis = init_scale * jax.random.normal(
            basis_key,
            (latent_dim, height, width),
        )
        self.out_logvar = jnp.zeros(frame_shape, dtype=jnp.float32)
        self.frame_shape = frame_shape
        self.latent_dim = latent_dim
        self.use_kl = False

    def encode(
        self,
        x: jax.Array,
        weight: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        model_input = _fraction_normalized_input(x, weight)
        coeff = self.encoder(jnp.ravel(model_input[0]))
        coeff_logvar = jnp.zeros_like(coeff)
        return coeff, coeff_logvar

    def decode(self, coeff: jax.Array) -> tuple[jax.Array, jax.Array]:
        frame = self.bias + jnp.einsum('k,kyx->yx', coeff, self.basis)
        return frame[jnp.newaxis, ...], self.out_logvar[jnp.newaxis, ...]

    def __call__(
        self,
        x: jax.Array,
        key: jax.Array,
        weight: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        del key
        coeff, coeff_logvar = self.encode(x, weight)
        mean, logvar = self.decode(coeff)
        return mean, logvar, coeff, coeff_logvar

    def predict(
        self,
        x: jax.Array,
        weight: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        coeff, _ = self.encode(x, weight)
        return self.decode(coeff)


class LinearBasisVAE(eqx.Module):
    '''Linear basis VAE for detector-fixed backgrounds.

    The encoder predicts a Gaussian distribution over basis coefficients.
    The decoder is the same detector-fixed linear basis model as
    :class:`LinearBasisAE`.
    '''

    z_mean: eqx.nn.Linear
    z_logvar: eqx.nn.Linear
    bias: jax.Array
    basis: jax.Array
    out_logvar: jax.Array
    frame_shape: tuple[int, int]
    latent_dim: int
    use_kl: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        frame_shape: tuple[int, int],
        latent_dim: int,
        key: jax.Array,
        init_scale: float = 1.0e-3,
    ) -> None:
        height, width = frame_shape
        n_pixels = height * width
        mean_key, logvar_key, basis_key = jax.random.split(key, 3)

        self.z_mean = eqx.nn.Linear(n_pixels, latent_dim, key=mean_key)
        self.z_logvar = eqx.nn.Linear(n_pixels, latent_dim, key=logvar_key)
        self.bias = jnp.zeros(frame_shape, dtype=jnp.float32)
        self.basis = init_scale * jax.random.normal(
            basis_key,
            (latent_dim, height, width),
        )
        self.out_logvar = jnp.zeros(frame_shape, dtype=jnp.float32)
        self.frame_shape = frame_shape
        self.latent_dim = latent_dim
        self.use_kl = True

    def encode(
        self,
        x: jax.Array,
        weight: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        model_input = _fraction_normalized_input(x, weight)
        flat = jnp.ravel(model_input[0])
        return self.z_mean(flat), self.z_logvar(flat)

    def decode(self, z: jax.Array) -> tuple[jax.Array, jax.Array]:
        frame = self.bias + jnp.einsum('k,kyx->yx', z, self.basis)
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
