#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from comicsnet import FitConfig
from comicsnet.fit import predict_background


class ConstantLogvarModel:
    def predict(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
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


def test_predict_background_returns_stddev_uncertainty() -> None:
    cube = jnp.ones((2, 3, 4), dtype=jnp.float32)

    background, uncertainty = predict_background(
        ConstantLogvarModel(),
        cube,
        FitConfig(),
    )

    np.testing.assert_array_equal(
        np.asarray(background),
        np.zeros((2, 3, 4), dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(uncertainty),
        np.full((2, 3, 4), 2.0, dtype=np.float32),
    )


def test_predict_background_passes_mask_weight() -> None:
    cube = jnp.ones((2, 2, 2), dtype=jnp.float32)
    mask = jnp.zeros_like(cube, dtype=bool)
    mask = mask.at[0, 0, 1].set(True)

    background, uncertainty = predict_background(
        WeightEchoModel(),
        cube,
        FitConfig(),
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
