#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from comicsnet import Config, FlatFieldAE, fit
from comicsnet.model import FlatFieldAE as ExportedFlatFieldAE


FRAME = jnp.arange(12, dtype=jnp.float32).reshape(1, 3, 4) / 12
WEIGHT = jnp.ones_like(FRAME).at[:, 0, :2].set(0.0)


def _model(**kwargs):
    return FlatFieldAE(
        frame_shape=(3, 4), key=jax.random.PRNGKey(0), **kwargs,
    )


def test_defaults_and_exports() -> None:
    model = _model()

    assert FlatFieldAE is ExportedFlatFieldAE
    assert model.frame_shape == (3, 4)
    assert model.hidden_dim == model.encoder.width_size == 8
    assert model.depth == model.encoder.depth == 2
    assert model.flat_variation_scale == 0.05
    assert not model.use_kl
    assert model.encoder.in_size == 48
    assert model.encoder.out_size == 1
    for array in (model.bias, model.dflat, model.out_logvar):
        assert array.shape == (3, 4)
        np.testing.assert_array_equal(array, 0.0)
    np.testing.assert_array_equal(model.flat, 1.0)


@pytest.mark.parametrize('depth', [0, 1, 2, 3])
def test_encoder_depth_and_augmented_channels(depth) -> None:
    model = _model(depth=depth, hidden_dim=5)
    bias = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 20
    dflat = FRAME[0] - 0.5
    model = eqx.tree_at(
        lambda m: (m.bias, m.dflat), model, (bias, dflat),
    )
    weight = WEIGHT * 0.5
    augmented = jnp.concatenate(
        [
            FRAME * weight,
            weight,
            bias[jnp.newaxis],
            model.flat[jnp.newaxis],
        ],
        axis=0,
    ).ravel()

    expected = augmented
    for layer in model.encoder.layers[:-1]:
        expected = jax.nn.gelu(layer(expected))
    expected = model.encoder.layers[-1](expected)
    actual, logvar = model.encode(FRAME, weight)

    assert len(model.encoder.layers) == depth + 1
    np.testing.assert_allclose(actual, expected, atol=1.0e-6)
    np.testing.assert_array_equal(logvar, jnp.zeros(1))
    assert model.encoder.layers[0].in_features == 48
    assert model.encoder.layers[-1].out_features == 1
    for layer in model.encoder.layers[:-1]:
        assert layer.out_features == 5


def test_encoder_flat_channel_is_differentiable() -> None:
    model = _model(depth=0)
    n_pixels = model.bias.size
    weights = jnp.zeros_like(model.encoder.layers[0].weight)
    weights = weights.at[0, 3 * n_pixels].set(1.0)
    model = eqx.tree_at(
        lambda m: (m.encoder.layers[0].weight, m.encoder.layers[0].bias),
        model, (weights, jnp.zeros(1)),
    )

    def flux(m):
        return m.encode(FRAME, WEIGHT)[0][0]

    np.testing.assert_allclose(flux(model), model.flat[0, 0])
    grads = eqx.filter_grad(flux)(model)
    expected = jnp.full_like(model.dflat, -0.05 / n_pixels)
    expected = expected.at[0, 0].add(0.05)
    np.testing.assert_allclose(grads.dflat, expected, atol=1.0e-7)


@pytest.mark.parametrize('scale', [0.0, 0.05, 0.49])
def test_flat_is_centered_bounded_and_positive(scale) -> None:
    model = _model(flat_variation_scale=scale)
    dflat = jnp.array([[-100.0, 100.0, 0.3, 1.0]] * 3)
    model = eqx.tree_at(lambda m: m.dflat, model, dflat)

    variation = jnp.tanh(dflat)
    expected = 1 + scale * (variation - variation.mean())
    np.testing.assert_allclose(model.flat, expected, atol=1.0e-7)
    np.testing.assert_allclose(model.flat.mean(), 1.0, atol=1.0e-6)
    assert bool(jnp.all(jnp.abs(model.flat - 1) <= 2 * scale + 1.0e-7))
    assert bool(jnp.all(model.flat > 0))


def test_decoder_and_differential_response() -> None:
    model = _model()
    model = eqx.tree_at(
        lambda m: (m.bias, m.dflat, m.out_logvar), model,
        (FRAME[0], FRAME[0] - 0.5, jnp.full((3, 4), -2.0)),
    )
    flux = jnp.array([-1.5])

    mean, logvar = model.decode(flux)
    np.testing.assert_allclose(mean[0], model.bias + flux[0] * model.flat)
    np.testing.assert_array_equal(logvar[0], model.out_logvar)
    derivative = jax.jvp(
        lambda f: model.decode(f)[0], (flux,), (jnp.ones_like(flux),),
    )[1]
    np.testing.assert_allclose(derivative[0], model.flat)
    np.testing.assert_array_equal(model.decode(-flux)[1], logvar)


def test_encoder_allows_negative_flux() -> None:
    model = _model()
    model = eqx.tree_at(
        lambda m: (m.encoder.layers[-1].weight, m.encoder.layers[-1].bias),
        model, (jnp.zeros_like(model.encoder.layers[-1].weight),
                jnp.array([-2.0])),
    )
    flux, _ = model.encode(FRAME, WEIGHT)
    np.testing.assert_array_equal(flux, [-2.0])
    np.testing.assert_array_equal(model.predict(FRAME, WEIGHT)[0], -2.0)


def test_masked_pixels_do_not_affect_prediction() -> None:
    model = _model()
    predict = eqx.filter_jit(lambda m, x, w: m.predict(x, w))
    expected = predict(model, FRAME, WEIGHT)
    for value in (1000.0, jnp.nan):
        changed = jnp.where(WEIGHT == 0, value, FRAME)
        actual = predict(model, changed, WEIGHT)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)

    grad = jax.grad(lambda x: model.predict(x, WEIGHT)[0].sum())(FRAME)
    np.testing.assert_array_equal(grad[WEIGHT == 0], 0.0)


def test_none_weight_matches_all_observed() -> None:
    model = _model()
    for a, b in zip(
        model.predict(FRAME, None),
        model.predict(FRAME, jnp.ones_like(FRAME)),
    ):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('fraction', [0.0, 0.5, 1.0])
def test_weights_have_finite_gradients(fraction) -> None:
    model = _model()
    weight = WEIGHT * fraction

    def loss(m):
        return jnp.sum(m.predict(FRAME, weight)[0] ** 2)

    value, grads = eqx.filter_jit(eqx.filter_value_and_grad(loss))(model)
    assert bool(jnp.isfinite(value))
    for grad in jax.tree_util.tree_leaves(grads):
        assert bool(jnp.isfinite(grad).all())


def test_fit_updates_shared_arrays_and_encoder() -> None:
    model = _model()
    cube = jnp.concatenate([FRAME, FRAME + 0.5])
    mask = jnp.repeat(WEIGHT == 0, 2, axis=0)
    result = fit(
        model, cube, mask=mask,
        config=Config(
            outer_steps=1, inner_steps=3, update_mask=False,
            standardize=False,
        ),
    )

    np.testing.assert_array_equal(result.mask, mask)
    assert result.background.shape == result.uncertainty.shape == cube.shape
    assert bool(jnp.isfinite(jnp.asarray(result.losses)).all())
    assert bool(jnp.all(result.uncertainty > 0))
    np.testing.assert_allclose(result.model.flat.mean(), 1.0, atol=1.0e-6)
    for name in ('bias', 'dflat', 'out_logvar'):
        before = getattr(model, name)
        after = getattr(result.model, name)
        assert bool(jnp.any(before != after))
    before = model.encoder.layers[0].weight
    after = result.model.encoder.layers[0].weight
    assert bool(jnp.any(before != after))


def test_fit_recovers_flat_from_uniform_flux_variations() -> None:
    true_flat = 1 + jnp.linspace(-0.03, 0.03, 12).reshape(3, 4)
    bias = jnp.linspace(-0.2, 0.2, 12).reshape(3, 4)[:, ::-1]
    flux = jnp.linspace(-2, 2, 16)
    cube = bias + flux[:, None, None] * true_flat

    result = fit(
        _model(depth=0), cube,
        config=Config(
            outer_steps=1, inner_steps=600, learning_rate=0.01,
            update_mask=False, standardize=False, global_norm=1.0,
        ),
    )

    np.testing.assert_allclose(result.model.flat, true_flat, atol=0.01)


@pytest.mark.parametrize(
    'kwargs, error',
    [
        ({'frame_shape': (0, 4)}, ValueError),
        ({'frame_shape': (3,)}, ValueError),
        ({'frame_shape': (3, 4, 5)}, ValueError),
        ({'frame_shape': (3.5, 4)}, TypeError),
        ({'hidden_dim': 0}, ValueError),
        ({'hidden_dim': 2.5}, TypeError),
        ({'depth': -1}, ValueError),
        ({'depth': 1.5}, TypeError),
        ({'flat_variation_scale': -0.1}, ValueError),
        ({'flat_variation_scale': 0.5}, ValueError),
        ({'flat_variation_scale': float('nan')}, ValueError),
        ({'flat_variation_scale': float('inf')}, ValueError),
    ],
)
def test_invalid_options(kwargs, error) -> None:
    options = {'frame_shape': (3, 4), **kwargs}
    with pytest.raises(error):
        FlatFieldAE(key=jax.random.PRNGKey(0), **options)
