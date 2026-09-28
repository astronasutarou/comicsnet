#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from comicsnet.frames import (
    channel_first,
    prepare_cube,
    sample_frame_index,
    strip_channel,
)


@pytest.mark.parametrize('enable_x64', [False, True])
@pytest.mark.parametrize(
    'input_dtype', [np.int16, np.int32, np.int64, np.float32, np.float64],
)
@pytest.mark.filterwarnings('error')
def test_prepare_cube_uses_default_float_dtype(
    enable_x64, input_dtype,
) -> None:
    data = np.arange(24, dtype=input_dtype).reshape(2, 3, 4)
    with jax.enable_x64(enable_x64):
        cube = prepare_cube(data)

        assert cube.shape == data.shape
        assert cube.dtype == (jnp.float64 if enable_x64 else jnp.float32)
        np.testing.assert_array_equal(cube, data)


def test_prepare_cube_rejects_non_3d_input() -> None:
    with pytest.raises(ValueError, match='cube must have shape'):
        prepare_cube(jnp.zeros((3, 4)))


def test_channel_first_and_strip_channel() -> None:
    frame = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)

    with_channel = channel_first(frame)
    restored = strip_channel(with_channel)

    assert with_channel.shape == (1, 2, 3)
    np.testing.assert_array_equal(np.asarray(restored), np.asarray(frame))


def test_channel_first_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match='frame must have shape'):
        channel_first(jnp.zeros((1, 2, 3)))


def test_sample_frame_index_is_in_range() -> None:
    key = jax.random.PRNGKey(0)

    index = sample_frame_index(key, 5)

    assert index >= 0
    assert index < 5
