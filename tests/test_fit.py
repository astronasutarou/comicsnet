#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from comicsnet import BasisAE, BasisVAE, Config, LinearBasisAE, fit
from comicsnet.fit import (
    _make_optimizer,
    _train_inner_loop,
    _train_step,
    predict_background,
)
from comicsnet.frames import channel_first, sample_frame_index


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
@pytest.mark.parametrize('global_norm', [None, 1.0])
def test_train_scan_matches_python_loop(model_type, global_norm) -> None:
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
        global_norm=global_norm,
        beta=0.1,
    )
    optimizer = _make_optimizer(config)
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    key = jax.random.PRNGKey(1)
    reference_model, reference_state, reference_key = model, state, key

    # Continue with new weights, as happens across outer mask updates.
    for current_weight in (weight, 1.0 - weight):
        reference_losses = []
        for _ in range(config.inner_steps):
            reference_key, frame_key, vae_key = jax.random.split(
                reference_key, 3,
            )
            index = sample_frame_index(frame_key, data.shape[0])
            reference_model, reference_state, loss = _train_step(
                reference_model,
                reference_state,
                optimizer,
                channel_first(data[index]),
                channel_first(current_weight[index]),
                vae_key,
                config.beta,
            )
            reference_losses.append(float(loss))

        model, state, key, losses = _train_inner_loop(
            model, state, optimizer, data, current_weight, key, config,
        )

        assert isinstance(losses, tuple)
        assert all(isinstance(loss, float) for loss in losses)
        np.testing.assert_allclose(
            losses, reference_losses, rtol=1e-5, atol=1e-6,
        )
        np.testing.assert_array_equal(key, reference_key)
        actual = eqx.filter((model, state), eqx.is_array)
        expected = eqx.filter(
            (reference_model, reference_state), eqx.is_array,
        )
        assert (
            jax.tree_util.tree_structure(actual)
            == jax.tree_util.tree_structure(expected)
        )
        for value, reference in zip(
            jax.tree_util.tree_leaves(actual),
            jax.tree_util.tree_leaves(expected),
            strict=True,
        ):
            np.testing.assert_allclose(
                value, reference, rtol=1e-5, atol=1e-6,
            )
