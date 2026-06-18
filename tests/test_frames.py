#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from comicsnet.frames import (
    channel_first,
    normalize_cube,
    sample_frame_index,
    strip_channel,
)


class FramesTest(unittest.TestCase):
    def test_normalize_cube_returns_float32_3d_array(self) -> None:
        cube = normalize_cube(np.zeros((2, 3, 4), dtype=np.int16))

        self.assertEqual(cube.shape, (2, 3, 4))
        self.assertEqual(cube.dtype, jnp.float32)

    def test_normalize_cube_rejects_non_3d_input(self) -> None:
        with self.assertRaisesRegex(ValueError, 'cube must have shape'):
            normalize_cube(jnp.zeros((3, 4)))

    def test_channel_first_and_strip_channel(self) -> None:
        frame = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)

        with_channel = channel_first(frame)
        restored = strip_channel(with_channel)

        self.assertEqual(with_channel.shape, (1, 2, 3))
        np.testing.assert_array_equal(np.asarray(restored), np.asarray(frame))

    def test_channel_first_rejects_non_2d_input(self) -> None:
        with self.assertRaisesRegex(ValueError, 'frame must have shape'):
            channel_first(jnp.zeros((1, 2, 3)))

    def test_sample_frame_index_is_in_range(self) -> None:
        key = jax.random.PRNGKey(0)

        index = sample_frame_index(key, 5)

        self.assertGreaterEqual(index, 0)
        self.assertLess(index, 5)


if __name__ == '__main__':
    unittest.main()

