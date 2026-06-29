#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from comicsnet import Config, LinearBasisAE, fit
from comicsnet.fit import normalized_frame_coord, predict_background


class ConstantLogvarModel:
    def predict(
        self,
        x: jax.Array,
        weight: jax.Array,
        frame_coord: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        del weight, frame_coord
        mean = jnp.zeros_like(x)
        logvar = jnp.full_like(x, jnp.log(4.0))
        return mean, logvar


class WeightEchoModel:
    def predict(
        self,
        x: jax.Array,
        weight: jax.Array,
        frame_coord: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        del x, frame_coord
        return weight, jnp.zeros_like(weight)


class FrameCoordEchoModel:
    def predict(
        self,
        x: jax.Array,
        weight: jax.Array,
        frame_coord: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        del weight
        mean = jnp.ones_like(x) * frame_coord
        return mean, jnp.zeros_like(x)


def test_normalized_frame_coord() -> None:
    assert float(normalized_frame_coord(0, 3)) == 0.0
    assert float(normalized_frame_coord(1, 3)) == 0.5
    assert float(normalized_frame_coord(2, 3)) == 1.0
    assert float(normalized_frame_coord(0, 1)) == 0.0


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


def test_predict_background_passes_frame_coord() -> None:
    cube = jnp.ones((3, 2, 2), dtype=jnp.float32)

    background, uncertainty = predict_background(
        FrameCoordEchoModel(),
        cube,
        Config(),
    )

    expected = jnp.zeros_like(cube)
    expected = expected.at[1].set(0.5)
    expected = expected.at[2].set(1.0)
    np.testing.assert_array_equal(
        np.asarray(background),
        np.asarray(expected),
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
        latent_dim=1,
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
