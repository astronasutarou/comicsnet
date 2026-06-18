#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from comicsnet import BasisAE, BasisVAE, ConvAE, ConvVAE
from comicsnet.model.basis import _mask_augmented_input
from comicsnet.model.linear_basis import _fraction_normalized_input


def test_linear_basis_fraction_normalized_input() -> None:
    x = jnp.ones((1, 2, 2), dtype=jnp.float32)
    weight = jnp.asarray([[[1.0, 0.0], [1.0, 0.0]]])

    model_input = _fraction_normalized_input(x, weight)

    np.testing.assert_array_equal(
        np.asarray(model_input),
        np.asarray([[[2.0, 0.0], [2.0, 0.0]]], dtype=np.float32),
    )


def test_mask_augmented_input_appends_weight_channel() -> None:
    x = jnp.asarray([[[1.0, 2.0], [3.0, 4.0]]])
    weight = jnp.asarray([[[1.0, 0.0], [0.0, 1.0]]])

    model_input = _mask_augmented_input(x, weight)

    np.testing.assert_array_equal(
        np.asarray(model_input),
        np.asarray(
            [
                [[1.0, 0.0], [0.0, 4.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ],
            dtype=np.float32,
        ),
    )


def test_basis_models_accept_mask_augmented_input() -> None:
    key = jax.random.PRNGKey(0)
    x = jnp.ones((1, 4, 4), dtype=jnp.float32)
    weight = jnp.ones_like(x).at[:, :2, :2].set(0.0)

    ae = BasisAE(
        frame_shape=(4, 4),
        latent_dim=2,
        basis_dim=3,
        hidden_dim=4,
        key=key,
    )
    vae = BasisVAE(
        frame_shape=(4, 4),
        latent_dim=2,
        basis_dim=3,
        hidden_dim=4,
        key=key,
    )

    assert ae.encode_layer0.weight.shape == (4, 32)
    assert vae.encode_layer0.weight.shape == (4, 32)
    assert ae.predict(x, weight)[0].shape == (1, 4, 4)
    assert vae.predict(x, weight)[0].shape == (1, 4, 4)


def test_conv_models_accept_mask_augmented_input() -> None:
    key = jax.random.PRNGKey(0)
    x = jnp.ones((1, 4, 4), dtype=jnp.float32)
    weight = jnp.ones_like(x).at[:, :2, :2].set(0.0)

    ae = ConvAE(
        hidden_channels=2,
        latent_channels=1,
        key=key,
    )
    vae = ConvVAE(
        hidden_channels=2,
        latent_channels=1,
        key=key,
    )

    assert ae.encode_layer0.in_channels == 2
    assert vae.encode_layer0.in_channels == 2
    assert ae.predict(x, weight)[0].shape == (1, 4, 4)
    assert vae.predict(x, weight)[0].shape == (1, 4, 4)

