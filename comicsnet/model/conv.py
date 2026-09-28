#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from operator import index

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp


def _ones_weight(x: jax.Array) -> jax.Array:
    return jnp.ones_like(x)


def _mask_augmented_input(
    x: jax.Array,
    weight: jax.Array | None,
    coordinate_max_frequency: int = 4,
) -> jax.Array:
    """Stack weighted image, weight, x/y planes, and cosine coordinates.

    Coordinates run from 0 to 1 along each spatial axis and remain visible
    at masked pixels. A singleton axis has coordinate 0. Append x/y cosine
    pairs for frequencies 1 through coordinate_max_frequency, inclusive.
    """

    if weight is None:
        weight = _ones_weight(x)

    _, height, width = x.shape
    x_coord = jnp.linspace(0.0, 1.0, width, dtype=x.dtype)
    y_coord = jnp.linspace(0.0, 1.0, height, dtype=x.dtype)
    x_plane = jnp.broadcast_to(x_coord[None, None, :], (1, height, width))
    y_plane = jnp.broadcast_to(y_coord[None, :, None], (1, height, width))
    planes = [x * weight, weight, x_plane, y_plane]
    for n in range(1, coordinate_max_frequency + 1):
        planes.extend([
            jnp.cos(2 * jnp.pi * n * x_plane),
            jnp.cos(2 * jnp.pi * n * y_plane),
        ])
    return jnp.concatenate(planes, axis=0)


def _mean_pooling_2x2(x: jax.Array) -> jax.Array:
    channels, height, width = x.shape
    return jax.image.resize(
        x,
        (channels, height // 2, width // 2),
        method='area',
    )


def _bilinear_upsample_2x2(x: jax.Array) -> jax.Array:
    channels, height, width = x.shape
    return jax.image.resize(
        x,
        (channels, 2 * height, 2 * width),
        method='bilinear',
    )


class ConvAE(eqx.Module):
    """Pooling convolutional AE for full-frame background modelling.

    coordinate_max_frequency sets the highest x/y cosine frequency.
    The default is 4; 0 retains only the image, weight, and x/y planes.
    """

    encode_layer0: eqx.nn.Conv
    encode_layer1: eqx.nn.Conv
    z_layer: eqx.nn.Conv
    decode_layer0: eqx.nn.Conv
    decode_layer1: eqx.nn.Conv
    out_mean: eqx.nn.Conv
    out_logvar: eqx.nn.Conv
    use_kl: bool = eqx.field(static=True)
    coordinate_max_frequency: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        hidden_channels: int,
        latent_channels: int,
        key: jax.Array,
        coordinate_max_frequency: int = 4,
    ) -> None:
        coordinate_max_frequency = index(coordinate_max_frequency)
        if coordinate_max_frequency < 0:
            raise ValueError('coordinate_max_frequency must be non-negative')
        self.coordinate_max_frequency = coordinate_max_frequency
        keys = jax.random.split(key, 7)
        self.encode_layer0 = eqx.nn.Conv(
            2,
            4 + 2 * coordinate_max_frequency,
            hidden_channels,
            3,
            padding='SAME',
            padding_mode='REPLICATE',
            key=keys[0],
        )
        self.encode_layer1 = eqx.nn.Conv(
            2,
            hidden_channels,
            hidden_channels,
            3,
            padding='SAME',
            padding_mode='REPLICATE',
            key=keys[1],
        )
        self.z_layer = eqx.nn.Conv(
            2,
            hidden_channels,
            latent_channels,
            1,
            padding='SAME',
            padding_mode='REPLICATE',
            key=keys[2],
        )
        self.decode_layer0 = eqx.nn.Conv(
            2,
            latent_channels,
            hidden_channels,
            3,
            padding='SAME',
            padding_mode='REPLICATE',
            key=keys[3],
        )
        self.decode_layer1 = eqx.nn.Conv(
            2,
            hidden_channels,
            hidden_channels,
            3,
            padding='SAME',
            padding_mode='REPLICATE',
            key=keys[4],
        )
        self.out_mean = eqx.nn.Conv(
            2,
            hidden_channels,
            1,
            1,
            padding='SAME',
            padding_mode='REPLICATE',
            key=keys[5],
        )
        self.out_logvar = eqx.nn.Conv(
            2,
            hidden_channels,
            1,
            1,
            padding='SAME',
            padding_mode='REPLICATE',
            key=keys[6],
        )
        self.use_kl = False

    def encode(
        self,
        x: jax.Array,
        weight: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array]:
        model_input = _mask_augmented_input(
            x, weight, self.coordinate_max_frequency,
        )
        h = jnn.gelu(self.encode_layer0(model_input))
        h = _mean_pooling_2x2(h)
        h = jnn.gelu(self.encode_layer1(h))
        h = _mean_pooling_2x2(h)
        z = self.z_layer(h)
        z_logvar = jnp.zeros_like(z)
        return z, z_logvar

    def decode(self, z: jax.Array) -> tuple[jax.Array, jax.Array]:
        h = _bilinear_upsample_2x2(z)
        h = jnn.gelu(self.decode_layer0(h))
        h = _bilinear_upsample_2x2(h)
        h = jnn.gelu(self.decode_layer1(h))
        return self.out_mean(h), self.out_logvar(h)

    def __call__(
        self,
        x: jax.Array,
        key: jax.Array,
        weight: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        del key
        z, z_logvar = self.encode(x, weight)
        x_mean, x_logvar = self.decode(z)
        return x_mean, x_logvar, z, z_logvar

    def predict(
        self,
        x: jax.Array,
        weight: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array]:
        z, _ = self.encode(x, weight)
        x_mean, x_logvar = self.decode(z)
        return x_mean, x_logvar


class ConvVAE(eqx.Module):
    """Pooling convolutional VAE for full-frame background modelling.

    Inputs and outputs use channel-first shape ``(1, y, x)``.  Each training
    step sees one complete detector frame, not a spatial patch.
    coordinate_max_frequency sets the highest x/y cosine frequency.
    The default is 4; 0 retains only the image, weight, and x/y planes.
    """

    encode_layer0: eqx.nn.Conv
    encode_layer1: eqx.nn.Conv
    z_mean: eqx.nn.Conv
    z_logvar: eqx.nn.Conv
    decode_layer0: eqx.nn.Conv
    decode_layer1: eqx.nn.Conv
    out_mean: eqx.nn.Conv
    out_logvar: eqx.nn.Conv
    use_kl: bool = eqx.field(static=True)
    coordinate_max_frequency: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        hidden_channels: int,
        latent_channels: int,
        key: jax.Array,
        coordinate_max_frequency: int = 4,
    ) -> None:
        coordinate_max_frequency = index(coordinate_max_frequency)
        if coordinate_max_frequency < 0:
            raise ValueError('coordinate_max_frequency must be non-negative')
        self.coordinate_max_frequency = coordinate_max_frequency
        keys = jax.random.split(key, 8)
        self.encode_layer0 = eqx.nn.Conv(
            2,
            4 + 2 * coordinate_max_frequency,
            hidden_channels,
            3,
            padding='SAME',
            padding_mode='REPLICATE',
            key=keys[0],
        )
        self.encode_layer1 = eqx.nn.Conv(
            2,
            hidden_channels,
            hidden_channels,
            3,
            padding='SAME',
            padding_mode='REPLICATE',
            key=keys[1],
        )
        self.z_mean = eqx.nn.Conv(
            2,
            hidden_channels,
            latent_channels,
            1,
            padding='SAME',
            padding_mode='REPLICATE',
            key=keys[2],
        )
        self.z_logvar = eqx.nn.Conv(
            2,
            hidden_channels,
            latent_channels,
            1,
            padding='SAME',
            padding_mode='REPLICATE',
            key=keys[3],
        )
        self.decode_layer0 = eqx.nn.Conv(
            2,
            latent_channels,
            hidden_channels,
            3,
            padding='SAME',
            padding_mode='REPLICATE',
            key=keys[4],
        )
        self.decode_layer1 = eqx.nn.Conv(
            2,
            hidden_channels,
            hidden_channels,
            3,
            padding='SAME',
            padding_mode='REPLICATE',
            key=keys[5],
        )
        self.out_mean = eqx.nn.Conv(
            2,
            hidden_channels,
            1,
            1,
            padding='SAME',
            padding_mode='REPLICATE',
            key=keys[6],
        )
        self.out_logvar = eqx.nn.Conv(
            2,
            hidden_channels,
            1,
            1,
            padding='SAME',
            padding_mode='REPLICATE',
            key=keys[7],
        )
        self.use_kl = True

    def encode(
        self,
        x: jax.Array,
        weight: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array]:
        model_input = _mask_augmented_input(
            x, weight, self.coordinate_max_frequency,
        )
        h = jnn.gelu(self.encode_layer0(model_input))
        h = _mean_pooling_2x2(h)
        h = jnn.gelu(self.encode_layer1(h))
        h = _mean_pooling_2x2(h)
        return self.z_mean(h), self.z_logvar(h)

    def decode(self, z: jax.Array) -> tuple[jax.Array, jax.Array]:
        h = _bilinear_upsample_2x2(z)
        h = jnn.gelu(self.decode_layer0(h))
        h = _bilinear_upsample_2x2(h)
        h = jnn.gelu(self.decode_layer1(h))
        return self.out_mean(h), self.out_logvar(h)

    def __call__(
        self,
        x: jax.Array,
        key: jax.Array,
        weight: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        z_mean, z_logvar = self.encode(x, weight)
        eps = jax.random.normal(key, z_mean.shape)
        z = z_mean + jnp.exp(0.5 * z_logvar) * eps
        x_mean, x_logvar = self.decode(z)
        return x_mean, x_logvar, z_mean, z_logvar

    def predict(
        self,
        x: jax.Array,
        weight: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array]:
        z_mean, _ = self.encode(x, weight)
        x_mean, x_logvar = self.decode(z_mean)
        return x_mean, x_logvar
