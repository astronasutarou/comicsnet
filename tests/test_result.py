#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from comicsnet.config import Config
from comicsnet.result import FitResult, make_sparse


def test_make_sparse_keeps_masked_residuals_only() -> None:
    cube = jnp.asarray([1.0, 2.0, 4.0])
    background = jnp.asarray([0.5, 1.5, 3.0])
    mask = jnp.asarray([True, False, True])

    sparse = make_sparse(cube, background, mask)

    np.testing.assert_array_equal(
        np.asarray(sparse),
        np.asarray([0.5, 0.0, 1.0], dtype=np.float32),
    )


def test_fit_result_stores_outputs() -> None:
    data = jnp.ones((2, 3, 4))
    background = jnp.zeros((2, 3, 4))
    uncertainty = jnp.ones((2, 3, 4))
    mask = jnp.zeros((2, 3, 4), dtype=bool)
    config = Config(outer_steps=1)

    result = FitResult(
        data=data,
        background=background,
        uncertainty=uncertainty,
        mask=mask,
        model='model',
        config=config,
        losses=(1.0, 0.5),
    )

    assert result.data.shape == (2, 3, 4)
    assert result.background.shape == (2, 3, 4)
    assert result.uncertainty.shape == (2, 3, 4)
    np.testing.assert_array_equal(
        np.asarray(result.residual),
        np.asarray(data - background),
    )
    np.testing.assert_array_equal(
        np.asarray(result.sparse),
        np.zeros((2, 3, 4), dtype=np.float32),
    )
    assert result.model == 'model'
    assert result.config.outer_steps == 1
    assert result.losses == (1.0, 0.5)
