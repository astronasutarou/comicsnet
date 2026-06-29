#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import FrozenInstanceError
from dataclasses import fields

import pytest

from comicsnet.config import Config


def test_defaults() -> None:
    config = Config()

    assert config.outer_steps == 10
    assert config.inner_steps == 100
    assert config.erosion_size == 3
    assert config.dilation_size == 7
    assert config.mask_fraction_limit == 0.2
    assert config.learning_rate == 1.0e-4
    assert config.global_norm is None
    assert config.adam_b1 == 0.9
    assert config.adam_b2 == 0.999
    assert config.standardize
    assert config.update_mask


def test_overrides() -> None:
    config = Config(
        outer_steps=2,
        inner_steps=3,
        learning_rate=2.0e-3,
        global_norm=1.0,
        adam_b1=0.95,
        adam_b2=0.99,
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
    assert config.adam_b1 == 0.95
    assert config.adam_b2 == 0.99
    assert config.erosion_size == 5
    assert config.dilation_size == 9
    assert config.mask_fraction_limit == 0.4
    assert not config.standardize
    assert not config.update_mask


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
