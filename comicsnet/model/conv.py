#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp


class ConvAE(eqx.Module):
    '''Small fully convolutional AE for full-frame background modelling.'''

    encode_layer0: eqx.nn.Conv
    encode_layer1: eqx.nn.Conv
    z_layer: eqx.nn.Conv
    decode_layer0: eqx.nn.Conv
    decode_layer1: eqx.nn.Conv
    out_mean: eqx.nn.Conv
    out_logvar: eqx.nn.Conv
    use_kl: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        hidden_channels: int,
        latent_channels: int,
        key: jax.Array,
    ) -> None:
        keys = jax.random.split(key, 7)
        self.encode_layer0 = eqx.nn.Conv(
            2,
            1,
            hidden_channels,
            3,
            padding='SAME',
            key=keys[0],
        )
        self.encode_layer1 = eqx.nn.Conv(
            2,
            hidden_channels,
            hidden_channels,
            3,
            padding='SAME',
            key=keys[1],
        )
        self.z_layer = eqx.nn.Conv(
            2,
            hidden_channels,
            latent_channels,
            1,
            padding='SAME',
            key=keys[2],
        )
        self.decode_layer0 = eqx.nn.Conv(
            2,
            latent_channels,
            hidden_channels,
            3,
            padding='SAME',
            key=keys[3],
        )
        self.decode_layer1 = eqx.nn.Conv(
            2,
            hidden_channels,
            hidden_channels,
            3,
            padding='SAME',
            key=keys[4],
        )
        self.out_mean = eqx.nn.Conv(
            2,
            hidden_channels,
            1,
            1,
            padding='SAME',
            key=keys[5],
        )
        self.out_logvar = eqx.nn.Conv(
            2,
            hidden_channels,
            1,
            1,
            padding='SAME',
            key=keys[6],
        )
        self.use_kl = False

    def encode(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        h = jnn.gelu(self.encode_layer0(x))
        h = jnn.gelu(self.encode_layer1(h))
        z = self.z_layer(h)
        z_logvar = jnp.zeros_like(z)
        return z, z_logvar

    def decode(self, z: jax.Array) -> tuple[jax.Array, jax.Array]:
        h = jnn.gelu(self.decode_layer0(z))
        h = jnn.gelu(self.decode_layer1(h))
        return self.out_mean(h), self.out_logvar(h)

    def __call__(
        self,
        x: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        del key
        z, z_logvar = self.encode(x)
        x_mean, x_logvar = self.decode(z)
        return x_mean, x_logvar, z, z_logvar

    def predict(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        z, _ = self.encode(x)
        x_mean, x_logvar = self.decode(z)
        return x_mean, x_logvar


class ConvVAE(eqx.Module):
    '''Small fully convolutional VAE for full-frame background modelling.

    Inputs and outputs use channel-first shape ``(1, y, x)``.  Each training
    step sees one complete detector frame, not a spatial patch.
    '''

    encode_layer0: eqx.nn.Conv
    encode_layer1: eqx.nn.Conv
    z_mean: eqx.nn.Conv
    z_logvar: eqx.nn.Conv
    decode_layer0: eqx.nn.Conv
    decode_layer1: eqx.nn.Conv
    out_mean: eqx.nn.Conv
    out_logvar: eqx.nn.Conv
    use_kl: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        hidden_channels: int,
        latent_channels: int,
        key: jax.Array,
    ) -> None:
        keys = jax.random.split(key, 8)
        self.encode_layer0 = eqx.nn.Conv(
            2,
            1,
            hidden_channels,
            3,
            padding='SAME',
            key=keys[0],
        )
        self.encode_layer1 = eqx.nn.Conv(
            2,
            hidden_channels,
            hidden_channels,
            3,
            padding='SAME',
            key=keys[1],
        )
        self.z_mean = eqx.nn.Conv(
            2,
            hidden_channels,
            latent_channels,
            1,
            padding='SAME',
            key=keys[2],
        )
        self.z_logvar = eqx.nn.Conv(
            2,
            hidden_channels,
            latent_channels,
            1,
            padding='SAME',
            key=keys[3],
        )
        self.decode_layer0 = eqx.nn.Conv(
            2,
            latent_channels,
            hidden_channels,
            3,
            padding='SAME',
            key=keys[4],
        )
        self.decode_layer1 = eqx.nn.Conv(
            2,
            hidden_channels,
            hidden_channels,
            3,
            padding='SAME',
            key=keys[5],
        )
        self.out_mean = eqx.nn.Conv(
            2,
            hidden_channels,
            1,
            1,
            padding='SAME',
            key=keys[6],
        )
        self.out_logvar = eqx.nn.Conv(
            2,
            hidden_channels,
            1,
            1,
            padding='SAME',
            key=keys[7],
        )
        self.use_kl = True

    def encode(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        h = jnn.gelu(self.encode_layer0(x))
        h = jnn.gelu(self.encode_layer1(h))
        return self.z_mean(h), self.z_logvar(h)

    def decode(self, z: jax.Array) -> tuple[jax.Array, jax.Array]:
        h = jnn.gelu(self.decode_layer0(z))
        h = jnn.gelu(self.decode_layer1(h))
        return self.out_mean(h), self.out_logvar(h)

    def __call__(
        self,
        x: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        z_mean, z_logvar = self.encode(x)
        eps = jax.random.normal(key, z_mean.shape)
        z = z_mean + jnp.exp(0.5 * z_logvar) * eps
        x_mean, x_logvar = self.decode(z)
        return x_mean, x_logvar, z_mean, z_logvar

    def predict(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        z_mean, _ = self.encode(x)
        x_mean, x_logvar = self.decode(z_mean)
        return x_mean, x_logvar
