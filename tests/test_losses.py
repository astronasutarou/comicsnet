#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from comicsnet.losses import gaussian_nll, kl_normal


def test_gaussian_nll_uses_masked_average() -> None:
    x = jnp.asarray([1.0, 3.0])
    mean = jnp.asarray([0.0, 1.0])
    logvar = jnp.asarray([0.0, 0.0])
    weight = jnp.asarray([1.0, 0.0])

    loss = gaussian_nll(x, mean, logvar, weight)

    assert float(loss) == pytest.approx(0.5)


def test_gaussian_nll_returns_zero_for_empty_weight() -> None:
    x = jnp.asarray([1.0, 3.0])
    mean = jnp.asarray([0.0, 1.0])
    logvar = jnp.asarray([0.0, 0.0])
    weight = jnp.asarray([0.0, 0.0])

    loss = gaussian_nll(x, mean, logvar, weight)

    assert float(loss) == pytest.approx(0.0)


@pytest.mark.parametrize('logvar', [-14.0, 10.0])
def test_gaussian_nll_does_not_clip_logvar(logvar) -> None:
    x = jnp.zeros(1)

    def loss(value):
        return gaussian_nll(x, x, value, jnp.ones_like(x))

    value, grad = jax.value_and_grad(loss)(jnp.asarray(logvar))

    assert float(value) == pytest.approx(0.5 * logvar)
    assert float(grad) == pytest.approx(0.5)


def test_kl_normal_is_zero_for_unit_normal() -> None:
    mean = jnp.zeros((2, 3))
    logvar = jnp.zeros((2, 3))

    kl = kl_normal(mean, logvar)

    assert float(kl) == pytest.approx(0.0)


def test_kl_normal_penalizes_nonzero_mean() -> None:
    mean = jnp.ones((2, 3))
    logvar = jnp.zeros((2, 3))

    kl = kl_normal(mean, logvar)

    assert float(kl) == pytest.approx(0.5)
