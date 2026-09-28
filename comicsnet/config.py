#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Configurations for comicsnet fitting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Configuration for the background/sparse decomposition loop.

    Parameters:
        outer_steps (int):
            Number of training blocks. Each block runs
            ``inner_steps`` optimizer updates, then updates the mask if
            ``update_mask`` is True. Optimizer state is retained across blocks.
            Defaults to 5.
        inner_steps (int):
            Optimizer updates per block. Each update uses one
            randomly selected complete frame, not a full pass through the data
            cube. The total number of updates is ``outer_steps * inner_steps``.
            Defaults to 1000.
        learning_rate (float):
            Learning rate for Adam. Defaults to 1.0e-4.
        global_norm (float or None):
            Maximum global gradient L2 norm before
            Adam transforms the gradients. None disables gradient clipping.
            Defaults to None.
        adam_b1 (float):
            Exponential decay rate for Adam's first gradient
            moment estimate. Defaults to 0.95.
        adam_b2 (float):
            Exponential decay rate for Adam's second gradient
            moment estimate. Defaults to 0.999.
        adam_epsilon (float):
            Constant added outside the square root in Adam's denominator
            for numerical stability. Passed as Optax's ``eps`` parameter.
            Defaults to 1.0e-8.
        beta (float):
            Weight of the KL regularization term for VAE models. The
            loss is the weighted mean Gaussian negative log likelihood plus
            ``beta`` times the KL divergence to a standard normal prior,
            averaged over latent elements. Ignored by models with
            ``use_kl=False``. Defaults to 1.0e-4.
        threshold_sigma (float):
            Threshold for positive residuals after
            subtracting each frame's mean residual, in units of the frame's
            robust residual scale. Used only when updating the mask. Defaults
            to 5.0.
        min_scale (float):
            Lower bound on robust scales used for data
            standardization and mask thresholds. Scales are estimated as 1.4826
            times the median absolute deviation, in the units of the data at
            each processing stage. Defaults to 1.0e-6.
        erosion_size (int):
            Side length in pixels of the circular erosion
            kernel used in mask updates. Values <= 1 skip erosion. Only spatial
            axes are processed. Defaults to 3.
        dilation_size (int):
            Side length in pixels of the circular dilation
            kernel applied after erosion in mask updates. Values <= 1 skip
            dilation. The time axis is not connected by either morphology
            operation. Defaults to 3.
        mask_fraction_limit (float):
            Maximum masked fraction per frame after
            morphology. Frames exceeding this limit have their updated mask
            cleared. Values >= 1 disable the limit. This does not modify an
            externally supplied fixed mask. Defaults to 0.2.
        seed (int):
            Random seed for frame sampling and VAE latent sampling
            during fit. Model initialization uses its separately supplied
            random key. Defaults to 0.
        standardize (bool):
            Subtract the global cube median and divide by its
            robust scale before training. Masked pixels are included in these
            statistics. Returned background and uncertainty are rescaled to
            input units; model parameters remain in standardized units.
            Defaults to True.
        update_mask (bool):
            Recompute the source mask after each training
            block. The supplied mask is an initial value and is replaced, not
            combined, with updates. If False, retain the supplied mask (or an
            all-False mask if omitted) throughout training and final
            prediction. Defaults to True.
    """

    outer_steps: int = 5
    inner_steps: int = 1000
    learning_rate: float = 1.0e-4
    global_norm: float | None = None
    adam_b1: float = 0.95
    adam_b2: float = 0.999
    adam_epsilon: float = 1.0e-8
    beta: float = 1.0e-4
    threshold_sigma: float = 5.0
    min_scale: float = 1.0e-6
    erosion_size: int = 3
    dilation_size: int = 3
    mask_fraction_limit: float = 0.2
    seed: int = 0
    standardize: bool = True
    update_mask: bool = True
