#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Models of comicsnet."""

from __future__ import annotations

from .adaptive_basis import AdaptiveBasisAE
from .attention_basis import AttentionBasisAE
from .basis import BasisAE, BasisVAE
from .conv import ConvAE, ConvVAE
from .linear_basis import LinearBasisAE, LinearBasisVAE


__all__ = [
    'AdaptiveBasisAE',
    'AttentionBasisAE',
    'BasisAE',
    'BasisVAE',
    'ConvAE',
    'ConvVAE',
    'LinearBasisAE',
    'LinearBasisVAE',
]
