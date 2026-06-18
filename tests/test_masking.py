#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from comicsnet.masking import (
    observed_weight,
    robust_scale,
    update_sparse_mask,
)


class MaskingTest(unittest.TestCase):
    def test_robust_scale_uses_mad(self) -> None:
        values = jnp.asarray([0.0, 0.0, 1.0, 2.0, 100.0])

        scale = robust_scale(values, min_scale=1.0e-6)

        self.assertAlmostEqual(float(scale), 1.4826, places=4)

    def test_robust_scale_respects_min_scale(self) -> None:
        values = jnp.ones((4,))

        scale = robust_scale(values, min_scale=2.0)

        self.assertAlmostEqual(float(scale), 2.0)

    def test_update_sparse_mask_thresholds_residuals(self) -> None:
        cube = jnp.asarray([0.0, 0.0, 0.0, 10.0])
        background = jnp.zeros_like(cube)

        mask = update_sparse_mask(
            cube,
            background,
            threshold_sigma=5.0,
            min_scale=1.0,
        )

        np.testing.assert_array_equal(
            np.asarray(mask),
            np.asarray([False, False, False, True]),
        )

    def test_observed_weight_is_inverse_of_mask(self) -> None:
        mask = jnp.asarray([True, False, True])

        weight = observed_weight(mask)

        np.testing.assert_array_equal(
            np.asarray(weight),
            np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        )


if __name__ == '__main__':
    unittest.main()

