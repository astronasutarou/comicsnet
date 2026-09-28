#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from comicsnet import AdaptiveBasisAE, AttentionBasisAE, Config, fit


MODEL_TYPES = [AttentionBasisAE, AdaptiveBasisAE]
FRAME = jnp.arange(12, dtype=jnp.float32).reshape(1, 3, 4) / 12
WEIGHT = jnp.ones_like(FRAME).at[:, 0, :2].set(0.0)


def _model(model_type, **kwargs):
    return model_type(
        frame_shape=(3, 4),
        latent_dim=2,
        basis_dim=3,
        hidden_dim=4,
        key=jax.random.PRNGKey(0),
        **kwargs,
    )


@pytest.mark.parametrize('model_type', MODEL_TYPES)
def test_masked_pixels_do_not_affect_prediction(model_type) -> None:
    model = _model(model_type)
    predict = eqx.filter_jit(lambda m, x, w: m.predict(x, w))
    changed = jnp.where(WEIGHT == 0, 1000.0, FRAME)

    expected = predict(model, FRAME, WEIGHT)
    actual = predict(model, changed, WEIGHT)

    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)

    grad = jax.grad(lambda x: jnp.sum(model.predict(x, WEIGHT)[0]))(FRAME)
    np.testing.assert_array_equal(grad[WEIGHT == 0], 0.0)
    assert bool(jnp.isfinite(grad).all())


@pytest.mark.parametrize('model_type', MODEL_TYPES)
def test_none_weight_matches_all_observed(model_type) -> None:
    model = _model(model_type)
    for a, b in zip(
        model.predict(FRAME, None),
        model.predict(FRAME, jnp.ones_like(FRAME)),
    ):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('model_type', MODEL_TYPES)
def test_empty_and_fractional_weight_are_finite(model_type) -> None:
    model = _model(model_type)
    for weight in (jnp.zeros_like(FRAME), WEIGHT * 0.5):
        mean, logvar, latent, latent_logvar = eqx.filter_jit(model)(
            FRAME, jax.random.PRNGKey(1), weight,
        )
        assert mean.shape == logvar.shape == FRAME.shape
        assert latent.shape == latent_logvar.shape == (2,)
        assert bool(jnp.isfinite(mean).all())
        assert bool(jnp.isfinite(latent).all())
        np.testing.assert_array_equal(latent_logvar, 0.0)

        def loss(m):
            return jnp.sum(m.predict(FRAME, weight)[0] ** 2)

        grads = eqx.filter_grad(loss)(model)
        for grad in jax.tree_util.tree_leaves(grads):
            assert bool(jnp.isfinite(grad).all())


@pytest.mark.parametrize('model_type', MODEL_TYPES)
def test_fit_updates_attention_parameters(model_type) -> None:
    model = _model(model_type)
    cube = jnp.concatenate([FRAME, FRAME + 0.1])
    mask = jnp.repeat(WEIGHT == 0, 2, axis=0)
    result = fit(
        model, cube, mask=mask,
        config=Config(
            outer_steps=1, inner_steps=2, update_mask=False,
            standardize=False,
        ),
    )
    assert bool(jnp.isfinite(jnp.asarray(result.losses)).all())
    np.testing.assert_array_equal(result.mask, mask)
    assert result.background.shape == result.uncertainty.shape == cube.shape
    assert bool(jnp.isfinite(result.background).all())
    assert bool(jnp.isfinite(result.uncertainty).all())
    if model_type is AttentionBasisAE:
        before = model.attention.query_proj.weight
        after = result.model.attention.query_proj.weight
        assert bool(jnp.any(before != after))
        before = model.attention.key_proj.weight
        after = result.model.attention.key_proj.weight
        assert bool(jnp.any(before != after))
    else:
        for name in ('modulation_keys', 'basis_delta'):
            assert bool(jnp.any(
                getattr(model, name) != getattr(result.model, name),
            ))
        assert bool(jnp.any(
            model.query_layer.weight != result.model.query_layer.weight,
        ))


def test_attention_encoder_uses_only_observed_tokens() -> None:
    model = _model(AttentionBasisAE, num_queries=2, num_heads=2)
    y, x = jnp.meshgrid(
        jnp.linspace(0, 1, 3), jnp.linspace(0, 1, 4), indexing='ij',
    )
    tokens = jnp.stack(
        ((FRAME * WEIGHT).ravel(), WEIGHT.ravel(), x.ravel(), y.ravel()),
        axis=-1,
    )
    tokens = tokens[WEIGHT.ravel() > 0]
    features = jax.nn.gelu(jax.vmap(model.encode_layer0)(tokens))
    pooled = model.attention(jnp.eye(2), features, features)
    expected = model.encode_layer1(pooled.ravel())
    actual, _ = model.encode(FRAME, WEIGHT)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-6)


def test_attention_empty_frame_has_zero_pooled_features() -> None:
    model = _model(AttentionBasisAE)
    actual, _ = model.encode(jnp.full_like(FRAME, jnp.nan), 0 * WEIGHT)
    np.testing.assert_array_equal(actual, model.encode_layer1.bias)


@pytest.mark.parametrize('shape', [(1, 4), (3, 1)])
def test_attention_singleton_axis(shape) -> None:
    model = AttentionBasisAE(
        frame_shape=shape, latent_dim=2, basis_dim=2, hidden_dim=4,
        key=jax.random.PRNGKey(0),
    )
    mean, _ = model.predict(jnp.ones((1, *shape)), None)
    assert mean.shape == (1, *shape)
    assert bool(jnp.isfinite(mean).all())


@pytest.mark.parametrize('model_type', MODEL_TYPES)
def test_coefficients_allow_negative_values(model_type) -> None:
    model = _model(model_type)
    model = eqx.tree_at(
        lambda m: (m.coeff_layer.weight, m.coeff_layer.bias), model,
        (jnp.zeros_like(model.coeff_layer.weight), jnp.array([-1., 0., 0.])),
    )
    model = eqx.tree_at(lambda m: m.basis, model, jnp.ones_like(model.basis))
    mean, _ = model.decode(jnp.zeros(2))
    assert bool(jnp.all(mean < 0))


def test_adaptive_basis_depends_on_state_and_is_bounded() -> None:
    model = _model(AdaptiveBasisAE, modulation_scale=0.1)
    z0 = jnp.zeros(2)
    z1 = jnp.array([1.0, -1.0])
    weights = model.modulation_weights(z1)
    assert weights.shape == (3, 4)
    assert bool(jnp.all(weights >= 0))
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1.0e-6)
    assert bool(jnp.any(
        model.modulated_basis(z0) != model.modulated_basis(z1),
    ))

    for z in (z0, z1, z1 * 100):
        basis = model.modulated_basis(z)
        assert basis.shape == model.basis.shape
        mean_square = jnp.mean(model.basis ** 2, axis=(-2, -1), keepdims=True)
        rms = jnp.sqrt(mean_square)
        bound = model.modulation_scale * jnp.maximum(rms, 1.0e-6)
        assert bool(jnp.all(jnp.abs(basis - model.basis) <= bound + 1.0e-9))
        coeff = model.coeff_layer(jax.nn.gelu(z))
        expected = model.bias + jnp.einsum('k,kyx->yx', coeff, basis)
        np.testing.assert_allclose(model.decode(z)[0][0], expected)


def test_zero_modulation_matches_fixed_basis_decoder() -> None:
    model = _model(AdaptiveBasisAE, modulation_scale=0.0)
    z = jnp.array([1.0, -1.0])
    np.testing.assert_array_equal(model.modulated_basis(z), model.basis)
    coeff = model.coeff_layer(jax.nn.gelu(z))
    expected = model.bias + jnp.einsum('k,kyx->yx', coeff, model.basis)
    np.testing.assert_allclose(model.decode(z)[0][0], expected)


def test_zero_base_has_finite_modulation_gradients() -> None:
    model = _model(AdaptiveBasisAE, init_scale=0.0)
    grads = eqx.filter_grad(
        lambda m: jnp.sum(m.modulated_basis(jnp.ones(2))),
    )(model)
    for grad in jax.tree_util.tree_leaves(grads):
        assert bool(jnp.isfinite(grad).all())


@pytest.mark.parametrize(
    'model_type, kwargs',
    [
        (AttentionBasisAE, {'num_queries': 0}),
        (AttentionBasisAE, {'num_heads': 0}),
        (AttentionBasisAE, {'num_heads': 3}),
        (AdaptiveBasisAE, {'variation_dim': 0}),
        (AdaptiveBasisAE, {'modulation_scale': -0.1}),
        (AdaptiveBasisAE, {'modulation_scale': float('nan')}),
        (AdaptiveBasisAE, {'modulation_scale': float('inf')}),
    ],
)
def test_invalid_attention_options(model_type, kwargs) -> None:
    with pytest.raises(ValueError):
        _model(model_type, **kwargs)
