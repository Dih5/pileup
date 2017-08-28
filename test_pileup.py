#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_pileup.py: Tests for `pileup.py` module.
"""

import unittest

import numpy as np
from scipy import stats

from pileup import *

def L1(y1,y2):
    """Distance in the sense of the 1-norm"""
    return sum(abs(y2-y1))
        
class pileup_test(unittest.TestCase):
    
    
    def testPileDepile(self):
        """Test the depile reverts a pile"""
        for bin_size in [0.1,0.02]:
            for l in [0.01,0.2,1]:
                x=np.arange(0,10,bin_size)
                y=stats.gamma.pdf(x, 1.5)
                piled=poisson_pile(y,l)
                depiled=poisson_depile(piled,l)
                
                self.assertAlmostEqual(L1(y,depiled),0)
                
    def testPileMethods(self):
        """Test the pile methods provide similar results, excluding the mathematically unnacurate series method"""
        for bin_size in [0.1,0.02,0.005]:
            for l in [0.01,0.2,1]:
                x=np.arange(0,10,bin_size)
                y=stats.gamma.pdf(x, 1.5)
                
                piled_fourier=poisson_pile(y,l,method="Fourier")
                piled_fourier_c=poisson_pile(y,l,method="Fourier-C")
                piled_fourier_series=poisson_pile(y,l,method="Fourier_Series",series_order=30)
                
                self.assertAlmostEqual(L1(piled_fourier,piled_fourier_c),0)
                self.assertAlmostEqual(L1(piled_fourier,piled_fourier_series),0)
                
    def testDepileMethods(self):
        """Test the depile methods provide similar results, excluding the mathematically unnacurate series method"""
        for bin_size in [0.1,0.02,0.005]:
            for l in [0.01,0.2,1]:
                x=np.arange(0,10,bin_size)
                y=stats.gamma.pdf(x, 1.5)
                
                depiled_fourier=poisson_depile(y,l,method="Fourier")
                depiled_fourier_c=poisson_depile(y,l,method="Fourier-C")
                
                self.assertAlmostEqual(L1(depiled_fourier,depiled_fourier_c),0)
    
    #TODO: Add a known pile up check (?)
                
        
    
    
if __name__ == "__main__":
    unittest.main()
