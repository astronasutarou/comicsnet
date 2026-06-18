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
from .model import ConvVAE
from .result import FitResult, make_sparse
from .frames import (
    channel_first,
    normalize_cube,
    sample_frame_index,
    strip_channel,
)


def fit(
    cube: jax.Array,
    *,
    config: FitConfig | None = None,
    model: Any | None = None,
) -> FitResult:
    '''Fit a background model and sparse residual mask.'''

    if config is None:
        config = FitConfig()

    raw_data = normalize_cube(cube)
    data, data_offset, data_scale = _standardize(raw_data, config)
    key = jax.random.PRNGKey(config.seed)
    if model is None:
        key, model_key = jax.random.split(key)
        model = ConvVAE(
            hidden_channels=config.hidden_channels,
            latent_channels=config.latent_channels,
            key=model_key,
        )

    mask = jnp.zeros_like(data, dtype=bool)
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    losses: list[float] = []

    for outer_step in range(config.outer_steps):
        model, opt_state, key, step_losses = _train_inner_loop(
            model,
            opt_state,
            optimizer,
            data,
            mask,
            key,
            config,
        )
        losses.extend(step_losses)
        background, _ = predict_background(model, data, config)
        should_update = config.update_mask_each_outer_step
        should_update = should_update or outer_step == config.outer_steps - 1
        if should_update:
            mask = update_sparse_mask(
                data,
                background,
                threshold_sigma=config.threshold_sigma,
                min_scale=config.min_scale,
            )

    background, uncertainty = predict_background(model, data, config)
    background = background * data_scale + data_offset
    uncertainty = uncertainty * data_scale
    sparse = make_sparse(raw_data, background, mask)
    return FitResult(
        background=background,
        uncertainty=uncertainty,
        mask=mask,
        sparse=sparse,
        model=model,
        config=config,
        losses=tuple(losses),
    )


def predict_background(
    model: Any,
    cube: jax.Array,
    config: FitConfig,
) -> tuple[jax.Array, jax.Array]:
    '''Predict background mean and uncertainty frame-by-frame.'''

    data = normalize_cube(cube)
    mean = jnp.zeros_like(data)
    uncertainty = jnp.zeros_like(data)

    for frame_index in range(data.shape[0]):
        x = channel_first(data[frame_index])
        frame_mean, frame_logvar = model.predict(x)
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


def _train_inner_loop(
    model: Any,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    data: jax.Array,
    mask: jax.Array,
    key: jax.Array,
    config: FitConfig,
) -> tuple[Any, optax.OptState, jax.Array, tuple[float, ...]]:
    losses: list[float] = []

    for _ in range(config.inner_steps):
        key, frame_key, vae_key = jax.random.split(key, 3)
        frame_index = sample_frame_index(frame_key, data.shape[0])
        x = channel_first(data[frame_index])
        m = channel_first(mask[frame_index])
        w = observed_weight(m)
        model, opt_state, loss = _train_step(
            model,
            opt_state,
            optimizer,
            x,
            w,
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
    key: jax.Array,
    beta: float,
) -> tuple[Any, optax.OptState, jax.Array]:
    loss, grads = eqx.filter_value_and_grad(_loss)(model, x, weight, key, beta)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def _loss(
    model: Any,
    x: jax.Array,
    weight: jax.Array,
    key: jax.Array,
    beta: float,
) -> jax.Array:
    model_input = x * weight
    mean, logvar, z_mean, z_logvar = model(model_input, key)
    regularization = jnp.asarray(0.0)
    if getattr(model, 'use_kl', True):
        regularization = kl_normal(z_mean, z_logvar)
    return gaussian_nll(x, mean, logvar, weight) + beta * regularization
