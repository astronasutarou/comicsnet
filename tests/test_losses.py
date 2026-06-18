#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest

import jax.numpy as jnp

from comicsnet.losses import gaussian_nll, kl_normal


class LossesTest(unittest.TestCase):
    def test_gaussian_nll_uses_masked_average(self) -> None:
        x = jnp.asarray([1.0, 3.0])
        mean = jnp.asarray([0.0, 1.0])
        logvar = jnp.asarray([0.0, 0.0])
        weight = jnp.asarray([1.0, 0.0])

        loss = gaussian_nll(x, mean, logvar, weight)

        self.assertAlmostEqual(float(loss), 0.5)

    def test_gaussian_nll_returns_zero_for_empty_weight(self) -> None:
        x = jnp.asarray([1.0, 3.0])
        mean = jnp.asarray([0.0, 1.0])
        logvar = jnp.asarray([0.0, 0.0])
        weight = jnp.asarray([0.0, 0.0])

        loss = gaussian_nll(x, mean, logvar, weight)

        self.assertAlmostEqual(float(loss), 0.0)

    def test_kl_normal_is_zero_for_unit_normal(self) -> None:
        mean = jnp.zeros((2, 3))
        logvar = jnp.zeros((2, 3))

        kl = kl_normal(mean, logvar)

        self.assertAlmostEqual(float(kl), 0.0)

    def test_kl_normal_penalizes_nonzero_mean(self) -> None:
        mean = jnp.ones((2, 3))
        logvar = jnp.zeros((2, 3))

        kl = kl_normal(mean, logvar)

        self.assertAlmostEqual(float(kl), 0.5)


if __name__ == '__main__':
    unittest.main()

