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
    Config,
    LinearBasisAE,
    LinearBasisVAE,
    fit,
)
from comicsnet.fit import predict_background
from comicsnet.model.basis import _mask_augmented_input
from comicsnet.model.conv import (
    _mask_augmented_input as _conv_augmented_input,
)
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
    pytest.param(_conv_ae, (1, 1, 1), False, id='conv_ae'),
    pytest.param(_conv_vae, (1, 1, 1), True, id='conv_vae'),
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


@pytest.mark.parametrize('use_weight', [True, False])
@pytest.mark.parametrize('max_frequency', [0, 1, 3])
def test_conv_augmented_input_appends_spatial_coordinates(
    use_weight, max_frequency,
) -> None:
    x = jnp.arange(15, dtype=jnp.float32).reshape(1, 3, 5)
    weight = jnp.ones_like(x).at[:, 1, 2].set(0.0)
    weight = weight.at[:, 2, 4].set(0.5) if use_weight else None

    actual = jax.jit(_conv_augmented_input, static_argnums=2)(
        x, weight, max_frequency,
    )

    expected_weight = np.ones_like(x) if weight is None else np.asarray(weight)
    assert actual.shape == (4 + 2 * max_frequency, 3, 5)
    assert actual.dtype == x.dtype
    np.testing.assert_array_equal(actual[0], (x * expected_weight)[0])
    np.testing.assert_array_equal(actual[1], expected_weight[0])
    np.testing.assert_allclose(
        actual[2], np.tile([0.0, 0.25, 0.5, 0.75, 1.0], (3, 1)),
    )
    np.testing.assert_allclose(
        actual[3], np.tile([[0.0], [0.5], [1.0]], (1, 5)),
    )
    for n in range(1, max_frequency + 1):
        expected_x = np.cos(2 * np.pi * n * np.linspace(0, 1, 5))
        expected_y = np.cos(2 * np.pi * n * np.linspace(0, 1, 3))
        np.testing.assert_allclose(
            actual[2 + 2 * n], np.tile(expected_x, (3, 1)), atol=1.0e-6,
        )
        np.testing.assert_allclose(
            actual[3 + 2 * n], np.tile(expected_y[:, None], (1, 5)),
            atol=1.0e-6,
        )


@pytest.mark.parametrize('shape', [(1, 1), (1, 3), (3, 1)])
def test_conv_augmented_input_singleton_axis(shape) -> None:
    x = jnp.ones((1, *shape), dtype=jnp.float32)

    actual = _conv_augmented_input(x, None)

    assert actual.shape == (12, *shape)
    assert bool(jnp.isfinite(actual).all())
    if shape[1] == 1:
        np.testing.assert_array_equal(actual[2], np.zeros(shape))
        np.testing.assert_array_equal(actual[4::2], np.ones((4, *shape)))
    if shape[0] == 1:
        np.testing.assert_array_equal(actual[3], np.zeros(shape))
        np.testing.assert_array_equal(actual[5::2], np.ones((4, *shape)))


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

    assert ae.coordinate_max_frequency == 4
    assert vae.coordinate_max_frequency == 4
    assert ae.encode_layer0.in_channels == 12
    assert vae.encode_layer0.in_channels == 12
    assert ae.predict(x, weight)[0].shape == (1, 4, 4)
    assert vae.predict(x, weight)[0].shape == (1, 4, 4)


@pytest.mark.parametrize('model_type', [ConvAE, ConvVAE])
@pytest.mark.parametrize('max_frequency', [0, 2])
def test_conv_models_custom_coordinate_frequency(
    model_type, max_frequency,
) -> None:
    model = model_type(
        hidden_channels=2,
        latent_channels=1,
        coordinate_max_frequency=max_frequency,
        key=jax.random.PRNGKey(0),
    )

    mean, logvar = model.predict(FRAME, WEIGHT)

    assert model.encode_layer0.in_channels == 4 + 2 * max_frequency
    assert mean.shape == FRAME.shape
    assert logvar.shape == FRAME.shape
    assert bool(jnp.isfinite(mean).all())
    assert bool(jnp.isfinite(logvar).all())


@pytest.mark.parametrize('model_type', [ConvAE, ConvVAE])
@pytest.mark.parametrize(
    'frequency, error', [(-1, ValueError), (1.5, TypeError)],
)
def test_conv_models_reject_invalid_coordinate_frequency(
    model_type, frequency, error,
) -> None:
    with pytest.raises(error):
        model_type(
            hidden_channels=2,
            latent_channels=1,
            coordinate_max_frequency=frequency,
            key=jax.random.PRNGKey(0),
        )


@pytest.mark.parametrize(
    'factory, latent_shape, use_kl',
    MODEL_CASES,
)
def test_model_predict_minimal(factory, latent_shape, use_kl) -> None:
    del latent_shape
    model = factory()

    mean, logvar = model.predict(FRAME, WEIGHT)

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

    z_mean, z_logvar = model.encode(FRAME, None)

    assert z_mean.shape == latent_shape
    assert z_logvar.shape == latent_shape


@pytest.mark.parametrize(
    'factory, latent_shape, use_kl',
    MODEL_CASES,
)
def test_model_prediction_is_independent_of_frame_order(
    factory,
    latent_shape,
    use_kl,
) -> None:
    del latent_shape, use_kl
    model = factory()

    mask = jnp.stack([WEIGHT[0] == 0, WEIGHT[0] != 0])
    config = Config()

    original = predict_background(model, CUBE, config, mask=mask)
    reversed_frames = predict_background(
        model, CUBE[::-1], config, mask=mask[::-1],
    )

    for expected, actual in zip(original, reversed_frames):
        np.testing.assert_allclose(actual[::-1], expected)


@pytest.mark.parametrize(
    'factory, latent_shape, use_kl',
    MODEL_CASES,
)
def test_model_fit_minimal(factory, latent_shape, use_kl) -> None:
    del latent_shape, use_kl
    model = factory()
    config = Config(
        outer_steps=1,
        inner_steps=1,
        global_norm=1.0,
        adam_b1=0.95,
        adam_b2=0.99,
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
