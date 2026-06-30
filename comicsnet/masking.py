#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


def robust_scale(
    x: jax.Array,
    min_scale: float,
    axis: int | tuple[int, ...] | None = None
) -> jax.Array:
    '''Estimate a robust residual scale with the median absolute deviation.'''

    median = jnp.median(x, axis=axis, keepdims=True)
    mad = jnp.median(jnp.abs(x - median), axis=axis, keepdims=True)
    return jnp.maximum(1.4826 * mad, min_scale)


def update_sparse_mask(
    cube: jax.Array,
    background: jax.Array,
    *,
    threshold_sigma: float,
    min_scale: float,
    erosion_size: int,
    dilation_size: int,
    mask_fraction_limit: float,
    modifier: Callable = lambda x: x,
) -> jax.Array:
    '''Update the sparse-signal mask from standardized residuals.'''

    residual = cube - background
    residual -= jnp.mean(residual, axis=(1, 2), keepdims=True)
    scale = robust_scale(residual, min_scale, axis=(1, 2))
    mask = modifier(residual) > threshold_sigma * scale
    mask = binary_opening(mask, erosion_size, dilation_size)
    return limit_mask_fraction(mask, mask_fraction_limit)


def limit_mask_fraction(
    mask: jax.Array,
    mask_fraction_limit: float,
) -> jax.Array:
    '''Clear frames whose mask fraction exceeds a limit.'''

    if mask_fraction_limit >= 1.0:
        return mask

    values = mask.astype(jnp.float32)
    if mask.ndim < 2:
        fraction = jnp.mean(values)
        return jnp.where(fraction > mask_fraction_limit, False, mask)

    fraction = jnp.mean(values, axis=(-2, -1), keepdims=True)
    return jnp.where(fraction > mask_fraction_limit, False, mask)


def binary_opening(
    mask: jax.Array,
    erosion_size: int,
    dilation_size: int,
) -> jax.Array:
    '''Apply spatial circular binary opening to a mask.

    For movie masks, opening is applied only over the final two spatial axes.
    The time axis is not connected by the morphology operation.
    '''

    if mask.ndim == 0:
        return mask

    values = mask.astype(jnp.float32)

    if erosion_size > 1:
        erosion_kernel = circular_kernel(erosion_size)
        values = _binary_erosion(values, erosion_kernel)

    if dilation_size > 1:
        dilation_kernel = circular_kernel(dilation_size)
        values = _binary_dilation(values, dilation_kernel)

    return values.astype(bool)


def circular_kernel(opening_size: int) -> jax.Array:
    '''Return a discretized circular 2D kernel.'''

    if opening_size <= 1:
        return jnp.ones((1, 1), dtype=bool)

    center = (opening_size - 1) / 2.0
    radius = (opening_size - 1) / 2.0
    y, x = jnp.ogrid[:opening_size, :opening_size]
    distance2 = (y - center) ** 2 + (x - center) ** 2
    return distance2 <= radius ** 2


def _binary_erosion(values: jax.Array, kernel: jax.Array) -> jax.Array:
    counts = _spatial_convolve(values, kernel)
    return (counts == jnp.sum(kernel)).astype(jnp.float32)


def _binary_dilation(values: jax.Array, kernel: jax.Array) -> jax.Array:
    counts = _spatial_convolve(values, kernel)
    return (counts > 0).astype(jnp.float32)


def _spatial_convolve(values: jax.Array, kernel: jax.Array) -> jax.Array:
    if values.ndim == 1:
        values = values[jnp.newaxis, :]
        return _spatial_convolve_2d(values, kernel)[0]

    leading_shape = values.shape[:-2]
    spatial_shape = values.shape[-2:]
    frames = jnp.reshape(values, (-1, *spatial_shape))
    convolved = jax.vmap(lambda frame: _spatial_convolve_2d(frame, kernel))(
        frames,
    )
    return jnp.reshape(convolved, (*leading_shape, *spatial_shape))


def _spatial_convolve_2d(values: jax.Array, kernel: jax.Array) -> jax.Array:
    x = values[jnp.newaxis, jnp.newaxis, :, :]
    k = kernel.astype(values.dtype)[jnp.newaxis, jnp.newaxis, :, :]
    y = jax.lax.conv_general_dilated(
        x,
        k,
        window_strides=(1, 1),
        padding='SAME',
        dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
    )
    return y[0, 0]


def observed_weight(mask: jax.Array) -> jax.Array:
    '''Return weights for pixels used to fit the background.'''

    return 1.0 - mask.astype(jnp.float32)
