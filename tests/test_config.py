#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from comicsnet.config import FitConfig


class FitConfigTest(unittest.TestCase):
    def test_defaults(self) -> None:
        config = FitConfig()

        self.assertEqual(config.outer_steps, 5)
        self.assertEqual(config.inner_steps, 100)
        self.assertEqual(config.hidden_channels, 8)
        self.assertEqual(config.latent_channels, 2)
        self.assertTrue(config.standardize)
        self.assertTrue(config.update_mask_each_outer_step)

    def test_overrides(self) -> None:
        config = FitConfig(
            outer_steps=2,
            inner_steps=3,
            learning_rate=2.0e-3,
            standardize=False,
        )

        self.assertEqual(config.outer_steps, 2)
        self.assertEqual(config.inner_steps, 3)
        self.assertEqual(config.learning_rate, 2.0e-3)
        self.assertFalse(config.standardize)

    def test_frozen(self) -> None:
        config = FitConfig()

        with self.assertRaises(FrozenInstanceError):
            config.outer_steps = 10


if __name__ == '__main__':
    unittest.main()

