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


# Usa the following models:
distributions = [lambda xx: stats.gamma.pdf(xx, 1.5),  # Maxwellian
                 lambda xx: stats.gamma.pdf(xx, 1.5) + 0.3 * stats.norm.pdf(xx, loc=2, scale=0.1),  # Maxwellian + peak
                 lambda xx: stats.norm.pdf(xx, loc=2, scale=0.1)]  # Peak


class PileupTest(unittest.TestCase):
    def test_pile_depile(self):
        """Test the depile reverts a piling"""
        for bin_size in [0.1, 0.02, 1]:
            xx = np.arange(0, 10, bin_size)
            for l in [0.01, 0.2, 1, 2]:
                for distribution in distributions:
                    yy = distribution(xx)
                    piled = pile(yy, l)
                    depiled = depile(piled, l)

                    self.assertAlmostEqual(l1(yy, depiled), 0)

    def test_pile_methods(self):
        """Test the pile methods provide similar results, excluding the mathematically unnacurate series method"""
        for bin_size in [0.1, 0.02, 0.005]:
            xx = np.arange(0, 10, bin_size)
            for l in [0.01, 0.2, 1, 2]:
                for distribution in distributions:
                    yy = distribution(xx)

                    piled_fourier = pile(yy, l, method="Fourier")
                    piled_fourier_c = pile(yy, l, method="Fourier-C")
                    piled_fourier_series = pile(yy, l, method="Fourier_Series", series_order=30)

                    self.assertAlmostEqual(l1(piled_fourier, piled_fourier_c), 0)
                    self.assertAlmostEqual(l1(piled_fourier, piled_fourier_series), 0)

    def test_depile_methods(self):
        """Test the depile methods provide similar results, excluding the mathematically inaccurate series method"""
        for bin_size in [0.1, 0.02, 0.005]:
            for l in [0.01, 0.2, 1, 2]:
                xx = np.arange(0, 10, bin_size)
                for distribution in distributions:
                    yy = distribution(xx)

                    depiled_fourier = depile(yy, l, method="Fourier")
                    depiled_fourier_c = depile(yy, l, method="Fourier-C")

                    self.assertAlmostEqual(l1(depiled_fourier, depiled_fourier_c), 0)


if __name__ == "__main__":
    unittest.main()
