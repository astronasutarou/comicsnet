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
