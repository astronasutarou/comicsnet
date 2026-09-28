#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""COMICSNET: A library for background models in image sequences."""

from .config import Config
from .fit import fit, predict_background
from .model import (
    AdaptiveBasisAE,
    AttentionBasisAE,
    BasisAE,
    BasisVAE,
    ConvAE,
    ConvVAE,
    FlatFieldAE,
    FlatFieldVAE,
    LinearBasisAE,
    LinearBasisVAE,
)
from .result import FitResult


__version__ = '0.0.10'


__all__ = [
    'AdaptiveBasisAE',
    'AttentionBasisAE',
    'BasisAE',
    'BasisVAE',
    'ConvAE',
    'ConvVAE',
    'FlatFieldAE',
    'FlatFieldVAE',
    'LinearBasisAE',
    'LinearBasisVAE',
    'Config',
    'FitResult',
    '__version__',
    'fit',
    'predict_background',
]
