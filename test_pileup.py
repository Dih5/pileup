#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_pileup.py: Tests for `pileup.py` module.
"""

import unittest

import numpy as np
from scipy import stats

from pileup import *


def l1(y1, y2):
    """Distance in the sense of the 1-norm"""
    return sum(abs(y2 - y1))


class PileupTest(unittest.TestCase):
    def test_pile_depile(self):
        """Test the depile reverts a piling"""
        for bin_size in [0.1, 0.02, 1]:
            for l in [0.01, 0.2, 1, 2]:
                x = np.arange(0, 10, bin_size)
                y = stats.gamma.pdf(x, 1.5)
                piled = poisson_pile(y, l)
                depiled = poisson_depile(piled, l)

                self.assertAlmostEqual(l1(y, depiled), 0)

    def test_pile_methods(self):
        """Test the pile methods provide similar results, excluding the mathematically unnacurate series method"""
        for bin_size in [0.1, 0.02, 0.005]:
            for l in [0.01, 0.2, 1]:
                x = np.arange(0, 10, bin_size)
                y = stats.gamma.pdf(x, 1.5)

                piled_fourier = poisson_pile(y, l, method="Fourier")
                piled_fourier_c = poisson_pile(y, l, method="Fourier-C")
                piled_fourier_series = poisson_pile(y, l, method="Fourier_Series", series_order=30)

                self.assertAlmostEqual(l1(piled_fourier, piled_fourier_c), 0)
                self.assertAlmostEqual(l1(piled_fourier, piled_fourier_series), 0)

    def test_depile_methods(self):
        """Test the depile methods provide similar results, excluding the mathematically unnacurate series method"""
        for bin_size in [0.1, 0.02, 0.005]:
            for l in [0.01, 0.2, 1]:
                x = np.arange(0, 10, bin_size)
                y = stats.gamma.pdf(x, 1.5)

                depiled_fourier = poisson_depile(y, l, method="Fourier")
                depiled_fourier_c = poisson_depile(y, l, method="Fourier-C")

                self.assertAlmostEqual(l1(depiled_fourier, depiled_fourier_c), 0)

                # TODO: Add a known pile up check (?)


if __name__ == "__main__":
    unittest.main()
