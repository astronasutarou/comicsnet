#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from comicsnet.frames import (
    channel_first,
    normalize_cube,
    sample_frame_index,
    strip_channel,
)


def test_normalize_cube_returns_float32_3d_array() -> None:
    cube = normalize_cube(np.zeros((2, 3, 4), dtype=np.int16))

    assert cube.shape == (2, 3, 4)
    assert cube.dtype == jnp.float32


def test_normalize_cube_rejects_non_3d_input() -> None:
    with pytest.raises(ValueError, match='cube must have shape'):
        normalize_cube(jnp.zeros((3, 4)))


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

