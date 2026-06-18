#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import jax
import jax.numpy as jnp


def normalize_cube(cube: jax.Array) -> jax.Array:
    '''Convert input data to a floating point ``(time, y, x)`` array.'''

    array = jnp.asarray(cube, dtype=jnp.float32)
    if array.ndim != 3:
        raise ValueError('cube must have shape (time, y, x)')
    return array


def channel_first(frame: jax.Array) -> jax.Array:
    '''Add a singleton channel axis for a full detector frame.'''

    if frame.ndim != 2:
        raise ValueError('frame must have shape (y, x)')
    return frame[jnp.newaxis, ...]


def strip_channel(frame: jax.Array) -> jax.Array:
    '''Remove the singleton channel axis used by the network.'''

    return frame[0]


def sample_frame_index(key: jax.Array, n_frames: int) -> int:
    '''Sample one frame index for stochastic full-frame optimization.'''

    return int(jax.random.randint(key, (), 0, n_frames))
