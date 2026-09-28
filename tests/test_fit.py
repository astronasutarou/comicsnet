#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from comicsnet import BasisAE, BasisVAE, Config, LinearBasisAE, fit
from comicsnet.fit import (
    _make_optimizer,
    _train_inner_loop,
    predict_background,
)


class ConstantLogvarModel:
    def predict(
        self,
        x: jax.Array,
        weight: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        del weight
        mean = jnp.zeros_like(x)
        logvar = jnp.full_like(x, jnp.log(4.0))
        return mean, logvar


class WeightEchoModel:
    def predict(
        self,
        x: jax.Array,
        weight: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        del x
        return weight, jnp.zeros_like(weight)


class FrameEchoModel:
    def predict(
        self,
        x: jax.Array,
        weight: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        del weight
        return x, jnp.zeros_like(x)


def test_predict_background_returns_stddev_uncertainty() -> None:
    cube = jnp.ones((2, 3, 4), dtype=jnp.float32)

    background, uncertainty = predict_background(
        ConstantLogvarModel(),
        cube,
        Config(),
    )

    np.testing.assert_array_equal(
        np.asarray(background),
        np.zeros((2, 3, 4), dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(uncertainty),
        np.full((2, 3, 4), 2.0, dtype=np.float32),
    )


def test_predict_background_converts_mask_to_weight() -> None:
    cube = jnp.ones((2, 2, 2), dtype=jnp.float32)
    mask = jnp.zeros_like(cube, dtype=bool)
    mask = mask.at[0, 0, 1].set(True)

    background, uncertainty = predict_background(
        WeightEchoModel(),
        cube,
        Config(),
        mask=mask,
    )

    expected = jnp.ones_like(cube)
    expected = expected.at[0, 0, 1].set(0.0)
    np.testing.assert_array_equal(
        np.asarray(background),
        np.asarray(expected),
    )
    np.testing.assert_array_equal(
        np.asarray(uncertainty),
        np.ones((2, 2, 2), dtype=np.float32),
    )


def test_predict_background_rejects_mask_shape_mismatch() -> None:
    cube = jnp.ones((2, 2, 2), dtype=jnp.float32)
    mask = jnp.zeros((2, 2), dtype=bool)

    try:
        predict_background(
            WeightEchoModel(),
            cube,
            Config(),
            mask=mask,
        )
    except ValueError as error:
        assert str(error) == 'mask must have shape (time, y, x)'
    else:
        raise AssertionError('ValueError was not raised')


def test_predict_background_passes_full_frame() -> None:
    cube = jnp.arange(12, dtype=jnp.float32).reshape(3, 2, 2)

    background, uncertainty = predict_background(
        FrameEchoModel(),
        cube,
        Config(),
    )

    np.testing.assert_array_equal(
        np.asarray(background),
        np.asarray(cube),
    )
    np.testing.assert_array_equal(
        np.asarray(uncertainty),
        np.ones((3, 2, 2), dtype=np.float32),
    )


def test_fit_uses_initial_mask_without_forced_mask_update() -> None:
    cube = jnp.ones((2, 2, 2), dtype=jnp.float32)
    mask = jnp.zeros_like(cube, dtype=bool)
    mask = mask.at[1, 0, 0].set(True)
    model = LinearBasisAE(
        frame_shape=(2, 2),
        basis_dim=1,
        key=jax.random.PRNGKey(0),
    )
    config = Config(
        outer_steps=1,
        inner_steps=1,
        erosion_size=1,
        dilation_size=1,
        standardize=False,
        update_mask=False,
    )

    result = fit(model, cube, config=config, mask=mask)

    np.testing.assert_array_equal(
        np.asarray(result.mask),
        np.asarray(mask),
    )


@pytest.mark.parametrize('model_type', [BasisAE, BasisVAE])
def test_inner_loop_preserves_state_across_blocks(model_type) -> None:
    model = model_type(
        frame_shape=(2, 2),
        hidden_dim=4,
        latent_dim=2,
        basis_dim=2,
        key=jax.random.PRNGKey(0),
    )
    data = jnp.arange(12, dtype=jnp.float32).reshape(3, 2, 2) / 10
    weight = jnp.ones_like(data).at[:, 0, 1].set(0.0)
    config = Config(
        inner_steps=3,
        global_norm=1.0,
        beta=0.1,
    )
    optimizer = _make_optimizer(config)
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    key = jax.random.PRNGKey(1)
    expected_model, expected_state, expected_key, expected_losses = (
        _train_inner_loop(
            model, state, optimizer, data, weight, key,
            replace(config, inner_steps=2 * config.inner_steps),
        )
    )
    losses = ()
    for _ in range(2):
        previous_key = key
        model, state, key, block_losses = _train_inner_loop(
            model, state, optimizer, data, weight, key, config,
        )
        assert not np.array_equal(key, previous_key)
        assert isinstance(block_losses, tuple)
        assert len(block_losses) == config.inner_steps
        assert all(isinstance(loss, float) for loss in block_losses)
        assert np.isfinite(block_losses).all()
        losses += block_losses

    np.testing.assert_allclose(losses, expected_losses, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(key, expected_key)
    assert eqx.tree_equal(
        (model, state), (expected_model, expected_state),
        rtol=1e-5, atol=1e-6,
    )


def test_inner_loop_uses_updated_weight() -> None:
    model = LinearBasisAE(
        frame_shape=(2, 2),
        basis_dim=1,
        key=jax.random.PRNGKey(0),
    )
    data = jnp.ones((2, 2, 2))
    config = Config(inner_steps=3)
    optimizer = _make_optimizer(config)
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    key = jax.random.PRNGKey(1)

    model, state, key, observed_losses = _train_inner_loop(
        model, state, optimizer, data, jnp.ones_like(data), key, config,
    )
    assert np.all(np.asarray(observed_losses) > 0.0)

    _, _, _, masked_losses = _train_inner_loop(
        model, state, optimizer, data, jnp.zeros_like(data), key, config,
    )
    assert len(masked_losses) == config.inner_steps
    np.testing.assert_array_equal(masked_losses, np.zeros(config.inner_steps))
