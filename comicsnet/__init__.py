#!/usr/bin/env python
# -*- coding: utf-8 -*-


from .config import FitConfig
from .fit import fit, predict_background
from .model import (
    BasisAE,
    BasisVAE,
    ConvAE,
    ConvVAE,
    LinearBasisAE,
    LinearBasisVAE,
)
from .result import FitResult


__version__ = '0.0.1'


__all__ = [
    'BasisAE',
    'BasisVAE',
    'ConvAE',
    'ConvVAE',
    'FitConfig',
    'FitResult',
    'LinearBasisAE',
    'LinearBasisVAE',
    '__version__',
    'fit',
    'predict_background',
]
