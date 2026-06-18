#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import FrozenInstanceError
from dataclasses import fields

import pytest

from comicsnet.config import FitConfig


def test_defaults() -> None:
    config = FitConfig()

    assert config.outer_steps == 10
    assert config.inner_steps == 100
    assert config.erosion_size == 3
    assert config.dilation_size == 7
    assert config.learning_rate == 1.0e-4
    assert config.standardize
    assert config.update_mask_each_outer_step


def test_overrides() -> None:
    config = FitConfig(
        outer_steps=2,
        inner_steps=3,
        learning_rate=2.0e-3,
        erosion_size=5,
        dilation_size=9,
        standardize=False,
    )

    assert config.outer_steps == 2
    assert config.inner_steps == 3
    assert config.learning_rate == 2.0e-3
    assert config.erosion_size == 5
    assert config.dilation_size == 9
    assert not config.standardize


def test_frozen() -> None:
    config = FitConfig()

    with pytest.raises(FrozenInstanceError):
        config.outer_steps = 10


def test_model_parameters_are_not_fit_config_fields() -> None:
    names = {field.name for field in fields(FitConfig)}

    assert 'hidden_channels' not in names
    assert 'latent_channels' not in names


def test_model_parameters_are_rejected() -> None:
    with pytest.raises(TypeError):
        FitConfig(hidden_channels=8)
