#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from comicsnet.masking import (
    binary_opening,
    circular_kernel,
    observed_weight,
    robust_scale,
    update_sparse_mask,
)


def test_robust_scale_uses_mad() -> None:
    values = jnp.asarray([0.0, 0.0, 1.0, 2.0, 100.0])

    scale = robust_scale(values, min_scale=1.0e-6)

    assert float(scale) == pytest.approx(1.4826)


def test_robust_scale_respects_min_scale() -> None:
    values = jnp.ones((4,))

    scale = robust_scale(values, min_scale=2.0)

    assert float(scale) == pytest.approx(2.0)


def test_update_sparse_mask_thresholds_residuals() -> None:
    cube = jnp.asarray([0.0, 0.0, 0.0, 10.0])
    background = jnp.zeros_like(cube)

    mask = update_sparse_mask(
        cube,
        background,
        threshold_sigma=5.0,
        min_scale=1.0,
        erosion_size=1,
        dilation_size=1,
    )

    np.testing.assert_array_equal(
        np.asarray(mask),
        np.asarray([False, False, False, True]),
    )


def test_observed_weight_is_inverse_of_mask() -> None:
    mask = jnp.asarray([True, False, True])

    weight = observed_weight(mask)

    np.testing.assert_array_equal(
        np.asarray(weight),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    )


def test_binary_opening_removes_isolated_pixels() -> None:
    mask = jnp.zeros((5, 5), dtype=bool)
    mask = mask.at[2, 2].set(True)

    opened = binary_opening(mask, erosion_size=3, dilation_size=3)

    assert not bool(jnp.any(opened))


def test_binary_opening_keeps_large_components() -> None:
    mask = jnp.zeros((5, 5), dtype=bool)
    mask = mask.at[1:4, 1:4].set(True)

    opened = binary_opening(mask, erosion_size=3, dilation_size=3)

    expected = jnp.zeros((5, 5), dtype=bool)
    expected = expected.at[1, 2].set(True)
    expected = expected.at[2, 1:4].set(True)
    expected = expected.at[3, 2].set(True)

    np.testing.assert_array_equal(np.asarray(opened), np.asarray(expected))


def test_circular_kernel_for_size_three() -> None:
    kernel = circular_kernel(3)

    np.testing.assert_array_equal(
        np.asarray(kernel),
        np.asarray(
            [
                [False, True, False],
                [True, True, True],
                [False, True, False],
            ],
        ),
    )


def test_binary_opening_does_not_connect_time_axis() -> None:
    mask = jnp.zeros((3, 5, 5), dtype=bool)
    mask = mask.at[:, 2, 2].set(True)

    opened = binary_opening(mask, erosion_size=3, dilation_size=3)

    assert not bool(jnp.any(opened))


def test_binary_opening_can_use_larger_dilation() -> None:
    mask = jnp.zeros((7, 7), dtype=bool)
    mask = mask.at[2:5, 2:5].set(True)

    opened = binary_opening(mask, erosion_size=3, dilation_size=5)

    assert bool(opened[3, 3])
    assert bool(opened[1, 3])
    assert bool(opened[5, 3])
