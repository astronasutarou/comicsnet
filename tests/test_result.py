#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from comicsnet.config import FitConfig
from comicsnet.result import FitResult, make_sparse


class ResultTest(unittest.TestCase):
    def test_make_sparse_keeps_masked_residuals_only(self) -> None:
        cube = jnp.asarray([1.0, 2.0, 4.0])
        background = jnp.asarray([0.5, 1.5, 3.0])
        mask = jnp.asarray([True, False, True])

        sparse = make_sparse(cube, background, mask)

        np.testing.assert_array_equal(
            np.asarray(sparse),
            np.asarray([0.5, 0.0, 1.0], dtype=np.float32),
        )

    def test_fit_result_stores_outputs(self) -> None:
        background = jnp.zeros((2, 3, 4))
        uncertainty = jnp.ones((2, 3, 4))
        mask = jnp.zeros((2, 3, 4), dtype=bool)
        sparse = jnp.zeros((2, 3, 4))
        config = FitConfig(outer_steps=1)

        result = FitResult(
            background=background,
            uncertainty=uncertainty,
            mask=mask,
            sparse=sparse,
            model='model',
            config=config,
            losses=(1.0, 0.5),
        )

        self.assertEqual(result.background.shape, (2, 3, 4))
        self.assertEqual(result.uncertainty.shape, (2, 3, 4))
        self.assertEqual(result.model, 'model')
        self.assertEqual(result.config.outer_steps, 1)
        self.assertEqual(result.losses, (1.0, 0.5))


if __name__ == '__main__':
    unittest.main()

