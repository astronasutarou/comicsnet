#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .config import FitConfig
from .losses import gaussian_nll, kl_normal
from .masking import observed_weight, robust_scale, update_sparse_mask
from .result import FitResult
from .frames import channel_first, normalize_cube, sample_frame_index
from .frames import strip_channel


def fit(
    model: Any,
    cube: jax.Array,
    *,
    config: FitConfig | None = None,
    mask: jax.Array | None = None,
) -> FitResult:
    '''Fit a background model and sparse residual mask.'''

    if config is None:
        config = FitConfig()

    raw_data = normalize_cube(cube)
    data, data_offset, data_scale = _standardize(raw_data, config)
    mask = _normalize_mask(mask, data)
    weight = observed_weight(mask)
    key = jax.random.PRNGKey(config.seed)

    optimizer = _make_optimizer(config)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    losses: list[float] = []

    for outer_step in range(config.outer_steps):
        model, opt_state, key, step_losses = _train_inner_loop(
            model,
            opt_state,
            optimizer,
            data,
            weight,
            key,
            config,
        )
        losses.extend(step_losses)
        background, _ = predict_background(
            model,
            data,
            config,
            mask=mask,
        )
        if config.update_mask:
            mask = update_sparse_mask(
                data,
                background,
                threshold_sigma=config.threshold_sigma,
                min_scale=config.min_scale,
                erosion_size=config.erosion_size,
                dilation_size=config.dilation_size,
                mask_fraction_limit=config.mask_fraction_limit,
            )
            weight = observed_weight(mask)

    background, uncertainty = predict_background(
        model,
        data,
        config,
        mask=mask,
    )
    background = background * data_scale + data_offset
    uncertainty = uncertainty * data_scale
    return FitResult(
        data=raw_data,
        background=background,
        uncertainty=uncertainty,
        mask=mask,
        model=model,
        config=config,
        losses=tuple(losses),
    )


def predict_background(
    model: Any,
    cube: jax.Array,
    config: FitConfig,
    mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    '''Predict background mean and uncertainty frame-by-frame.'''

    data = normalize_cube(cube)
    mask = _normalize_mask(mask, data)
    weight = observed_weight(mask)

    mean = jnp.zeros_like(data)
    uncertainty = jnp.zeros_like(data)

    for frame_index in range(data.shape[0]):
        x = channel_first(data[frame_index])
        w = channel_first(weight[frame_index])
        frame_coord = normalized_frame_coord(frame_index, data.shape[0])
        frame_mean, frame_logvar = model.predict(x, w, frame_coord)
        mean = mean.at[frame_index].set(strip_channel(frame_mean))
        frame_logvar = jnp.clip(strip_channel(frame_logvar), -12.0, 8.0)
        frame_uncertainty = jnp.exp(0.5 * frame_logvar)
        uncertainty = uncertainty.at[frame_index].set(frame_uncertainty)

    return mean, uncertainty


def _standardize(
    data: jax.Array,
    config: FitConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    if not config.standardize:
        return data, jnp.asarray(0.0), jnp.asarray(1.0)

    offset = jnp.median(data)
    scale = robust_scale(data - offset, config.min_scale)
    return (data - offset) / scale, offset, scale


def _normalize_mask(
    mask: jax.Array | None,
    data: jax.Array,
) -> jax.Array:
    if mask is None:
        return jnp.zeros_like(data, dtype=bool)

    mask = jnp.asarray(mask, dtype=bool)
    if mask.shape != data.shape:
        raise ValueError('mask must have shape (time, y, x)')

    return mask


def _make_optimizer(config: FitConfig) -> optax.GradientTransformation:
    optimizer = optax.adam(
        config.learning_rate,
        b1=config.adam_b1,
        b2=config.adam_b2,
    )
    if config.global_norm is None:
        return optimizer

    return optax.chain(
        optax.clip_by_global_norm(config.global_norm),
        optimizer,
    )


def _train_inner_loop(
    model: Any,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    data: jax.Array,
    weight: jax.Array,
    key: jax.Array,
    config: FitConfig,
) -> tuple[Any, optax.OptState, jax.Array, tuple[float, ...]]:
    losses: list[float] = []

    for _ in range(config.inner_steps):
        key, frame_key, vae_key = jax.random.split(key, 3)
        frame_index = sample_frame_index(frame_key, data.shape[0])
        x = channel_first(data[frame_index])
        w = channel_first(weight[frame_index])
        frame_coord = normalized_frame_coord(frame_index, data.shape[0])
        model, opt_state, loss = _train_step(
            model,
            opt_state,
            optimizer,
            x,
            w,
            frame_coord,
            vae_key,
            config.beta,
        )
        losses.append(float(loss))

    return model, opt_state, key, tuple(losses)


@eqx.filter_jit
def _train_step(
    model: Any,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    x: jax.Array,
    weight: jax.Array,
    frame_coord: jax.Array,
    key: jax.Array,
    beta: float,
) -> tuple[Any, optax.OptState, jax.Array]:
    loss, grads = eqx.filter_value_and_grad(_loss)(
        model,
        x,
        weight,
        frame_coord,
        key,
        beta,
    )
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def _loss(
    model: Any,
    x: jax.Array,
    weight: jax.Array,
    frame_coord: jax.Array,
    key: jax.Array,
    beta: float,
) -> jax.Array:
    mean, logvar, z_mean, z_logvar = model(x, key, weight, frame_coord)
    regularization = jnp.asarray(0.0)
    if getattr(model, 'use_kl', True):
        regularization = kl_normal(z_mean, z_logvar)
    return gaussian_nll(x, mean, logvar, weight) + beta * regularization


def normalized_frame_coord(frame_index: int, n_frames: int) -> jax.Array:
    '''Return a normalized frame coordinate in the range [0, 1].'''

    denominator = max(n_frames - 1, 1)
    value = frame_index / denominator
    return jnp.asarray(value, dtype=jnp.float32)
