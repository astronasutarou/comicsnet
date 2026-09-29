#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import FrozenInstanceError
from dataclasses import fields

import jax.numpy as jnp
import numpy as np
import pytest

from comicsnet.config import Config


@pytest.mark.parametrize(
    'name, expected_types',
    [
        ('outer_steps', (int,)),
        ('inner_steps', (int,)),
        ('learning_rate', (float,)),
        ('global_norm', (float, type(None))),
        ('adam_b1', (float,)),
        ('adam_b2', (float,)),
        ('adam_epsilon', (float,)),
        ('beta', (float,)),
        ('threshold_sigma', (float,)),
        ('min_scale', (float,)),
        ('erosion_size', (int,)),
        ('dilation_size', (int,)),
        ('mask_fraction_limit', (float,)),
        ('seed', (int,)),
        ('standardize', (bool,)),
        ('update_mask', (bool,)),
    ],
)
def test_default_field_exists_and_has_expected_type(
    name, expected_types,
) -> None:
    config = Config()

    assert name in {field.name for field in fields(Config)}
    # Exact types distinguish integer options from bool, an int subclass.
    assert type(getattr(config, name)) in expected_types


def test_overrides() -> None:
    def logvar_clip(x):
        return x

    config = Config(
        outer_steps=2,
        inner_steps=3,
        learning_rate=2.0e-3,
        global_norm=1.0,
        adam_b1=0.9,
        adam_b2=0.99,
        adam_epsilon=1.0e-6,
        beta=2.0e-4,
        logvar_clip=logvar_clip,
        threshold_sigma=4.0,
        min_scale=1.0e-5,
        seed=42,
        erosion_size=5,
        dilation_size=9,
        mask_fraction_limit=0.4,
        standardize=False,
        update_mask=False,
    )

    assert config.outer_steps == 2
    assert config.inner_steps == 3
    assert config.learning_rate == 2.0e-3
    assert config.global_norm == 1.0
    assert config.adam_b1 == 0.9
    assert config.adam_b2 == 0.99
    assert config.adam_epsilon == 1.0e-6
    assert config.beta == 2.0e-4
    assert config.logvar_clip is logvar_clip
    assert config.threshold_sigma == 4.0
    assert config.min_scale == 1.0e-5
    assert config.seed == 42
    assert config.erosion_size == 5
    assert config.dilation_size == 9
    assert config.mask_fraction_limit == 0.4
    assert not config.standardize
    assert not config.update_mask


def test_default_logvar_clip() -> None:
    config = Config()

    assert 'logvar_clip' in {field.name for field in fields(Config)}
    assert callable(config.logvar_clip)
    logvar = jnp.asarray([-20.0, -12.0, 0.0, 8.0, 20.0])

    np.testing.assert_array_equal(
        config.logvar_clip(logvar), [-12.0, -12.0, 0.0, 8.0, 8.0],
    )


def test_frozen() -> None:
    config = Config()

    with pytest.raises(FrozenInstanceError):
        config.outer_steps = 10


def test_model_parameters_are_not_fit_config_fields() -> None:
    names = {field.name for field in fields(Config)}

    assert 'hidden_channels' not in names
    assert 'latent_channels' not in names
    assert 'update_mask_each_outer_step' not in names


def test_model_parameters_are_rejected() -> None:
    with pytest.raises(TypeError):
        Config(hidden_channels=8)


def test_old_update_mask_parameter_is_rejected() -> None:
    with pytest.raises(TypeError):
        Config(update_mask_each_outer_step=False)
