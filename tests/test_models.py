#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from comicsnet import (
    BasisAE,
    BasisVAE,
    ConvAE,
    ConvVAE,
    FitConfig,
    LinearBasisAE,
    LinearBasisVAE,
    fit,
)
from comicsnet.model.basis import _mask_augmented_input
from comicsnet.model.linear_basis import _fraction_normalized_input


FRAME_SHAPE = (4, 4)
FRAME = jnp.arange(16, dtype=jnp.float32).reshape(1, *FRAME_SHAPE)
WEIGHT = jnp.ones_like(FRAME).at[:, :2, :2].set(0.0)
CUBE = jnp.stack([FRAME[0], FRAME[0] + 1.0], axis=0)


def _conv_ae():
    return ConvAE(
        hidden_channels=2,
        latent_channels=1,
        key=jax.random.PRNGKey(0),
    )


def _conv_vae():
    return ConvVAE(
        hidden_channels=2,
        latent_channels=1,
        key=jax.random.PRNGKey(1),
    )


def _linear_basis_ae():
    return LinearBasisAE(
        frame_shape=FRAME_SHAPE,
        latent_dim=3,
        key=jax.random.PRNGKey(2),
    )


def _linear_basis_vae():
    return LinearBasisVAE(
        frame_shape=FRAME_SHAPE,
        latent_dim=3,
        key=jax.random.PRNGKey(3),
    )


def _basis_ae():
    return BasisAE(
        frame_shape=FRAME_SHAPE,
        latent_dim=2,
        basis_dim=3,
        hidden_dim=4,
        key=jax.random.PRNGKey(4),
    )


def _basis_vae():
    return BasisVAE(
        frame_shape=FRAME_SHAPE,
        latent_dim=2,
        basis_dim=3,
        hidden_dim=4,
        key=jax.random.PRNGKey(5),
    )


MODEL_CASES = [
    pytest.param(_conv_ae, (1, *FRAME_SHAPE), False, id='conv_ae'),
    pytest.param(_conv_vae, (1, *FRAME_SHAPE), True, id='conv_vae'),
    pytest.param(_linear_basis_ae, (3,), False, id='linear_basis_ae'),
    pytest.param(_linear_basis_vae, (3,), True, id='linear_basis_vae'),
    pytest.param(_basis_ae, (2,), False, id='basis_ae'),
    pytest.param(_basis_vae, (2,), True, id='basis_vae'),
]


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

    assert ae.encode_layer0.weight.shape == (4, 33)
    assert vae.encode_layer0.weight.shape == (4, 33)
    assert ae.predict(x, weight, jnp.asarray(0.5))[0].shape == (1, 4, 4)
    assert vae.predict(x, weight, jnp.asarray(0.5))[0].shape == (1, 4, 4)


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

    assert ae.encode_layer0.in_channels == 3
    assert vae.encode_layer0.in_channels == 3
    assert ae.predict(x, weight, jnp.asarray(0.5))[0].shape == (1, 4, 4)
    assert vae.predict(x, weight, jnp.asarray(0.5))[0].shape == (1, 4, 4)


@pytest.mark.parametrize(
    'factory, latent_shape, use_kl',
    MODEL_CASES,
)
def test_model_predict_minimal(factory, latent_shape, use_kl) -> None:
    del latent_shape
    model = factory()

    mean, logvar = model.predict(FRAME, WEIGHT, jnp.asarray(0.5))

    assert model.use_kl is use_kl
    assert mean.shape == FRAME.shape
    assert logvar.shape == FRAME.shape
    assert bool(jnp.isfinite(mean).all())
    assert bool(jnp.isfinite(logvar).all())


@pytest.mark.parametrize(
    'factory, latent_shape, use_kl',
    MODEL_CASES,
)
def test_model_call_minimal(factory, latent_shape, use_kl) -> None:
    del use_kl
    model = factory()

    mean, logvar, z_mean, z_logvar = model(
        FRAME,
        jax.random.PRNGKey(10),
        WEIGHT,
        jnp.asarray(0.5),
    )

    assert mean.shape == FRAME.shape
    assert logvar.shape == FRAME.shape
    assert z_mean.shape == latent_shape
    assert z_logvar.shape == latent_shape
    assert bool(jnp.isfinite(mean).all())
    assert bool(jnp.isfinite(logvar).all())
    assert bool(jnp.isfinite(z_mean).all())
    assert bool(jnp.isfinite(z_logvar).all())


@pytest.mark.parametrize(
    'factory, latent_shape, use_kl',
    MODEL_CASES,
)
def test_model_encode_requires_explicit_weight(
    factory,
    latent_shape,
    use_kl,
) -> None:
    del use_kl
    model = factory()

    with pytest.raises(TypeError):
        model.encode(FRAME)

    with pytest.raises(TypeError):
        model.encode(FRAME, None)

    z_mean, z_logvar = model.encode(FRAME, None, jnp.asarray(0.0))

    assert z_mean.shape == latent_shape
    assert z_logvar.shape == latent_shape


@pytest.mark.parametrize(
    'factory, latent_shape, use_kl',
    MODEL_CASES,
)
def test_model_predict_requires_frame_coord(
    factory,
    latent_shape,
    use_kl,
) -> None:
    del latent_shape, use_kl
    model = factory()

    with pytest.raises(TypeError):
        model.predict(FRAME, WEIGHT)


@pytest.mark.parametrize(
    'factory, latent_shape, use_kl',
    MODEL_CASES,
)
def test_model_fit_minimal(factory, latent_shape, use_kl) -> None:
    del latent_shape, use_kl
    model = factory()
    config = FitConfig(
        outer_steps=1,
        inner_steps=1,
        erosion_size=1,
        dilation_size=1,
    )

    result = fit(model, CUBE, config=config)

    assert result.data.shape == CUBE.shape
    assert result.background.shape == CUBE.shape
    assert result.uncertainty.shape == CUBE.shape
    assert result.mask.shape == CUBE.shape
    assert len(result.losses) == 1
    assert bool(jnp.isfinite(jnp.asarray(result.losses)).all())
