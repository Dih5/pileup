#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""A module to study the pile-up of distributions"""

from __future__ import print_function

import math

import sys
import numpy as np
from scipy.optimize import fmin_l_bfgs_b, curve_fit
from scipy.spatial.distance import sqeuclidean

__author__ = 'Dih5'
__version__ = "0.1.0"


def poisson_lambda(rate):
    """Return the poisson characteristic parameter, given a rate of counting/pulse frequency"""
    return -np.log(1 - rate)


def count_to_particle_ratio(l):
    """
    Return the expected value of the ratio of the number of particles arriving in a detector with the number of counts.

    This number is calculated assuming the number of counts per pulse in a detection is a random process conditioned by
    (number of counts > 0). Thus, the formula is $\frac{\lambda}{1-e^{-\lambda}}$.
    """
    return l/(1-math.exp(-l))


def _exp_n(x, n):
    """Maclaurin series of order n of the exponential"""
    return np.sum([x ** j / np.math.factorial(j) for j in range(0, n + 1)])


def _mercator(x, n):
    """Mercator series of order n of the logarithm"""
    return np.sum([(-1) ** (j + 1) * x ** j / j for j in range(1, n + 1)])


def _pile_series(yy, l, n, bin_size=1.0):
    f = np.array(l) * yy  # the piled-up function
    f_i = np.copy(f)  # i-th convolution power of f
    n_fact = l  # Hold lambda^i/i!
    # The first one is already added
    for i in range(2, n + 1):
        n_fact *= l / i
        f_i = np.convolve(f_i, yy)  # Longer first to avoid swapping
        f_i /= sum(f_i) * bin_size
        f.resize(f_i.shape)
        f += f_i * np.array(n_fact)

    return f[:len(yy)] * np.array(1 / (np.exp(l) - 1))  # Make sure to cut added tails


def _pile_fourier(yy, l, bin_size=1.0):
    four = bin_size * l * np.fft.fft(yy)  # Discrete Fourier Transform of lambda*yy
    pile_factor = np.exp(l) - 1  # exp(lambda)-1
    piled_four = list(map(lambda t: (np.exp(t) - 1) / pile_factor, four))  # Piled-up function in Fourier Space
    return np.real(np.fft.ifft(piled_four)) / bin_size


def _pile_fourier_series(yy, l, n, bin_size=1.0):
    four = bin_size * l * np.fft.fft(yy)  # Discrete Fourier Transform of lambda*yy
    pile_factor = np.exp(l) - 1  # exp(lambda)-1
    piled_four = list(map(lambda t: (_exp_n(t, n) - 1) / pile_factor, four))  # Piled-up function in Fourier Space
    return np.real(np.fft.ifft(piled_four)) / bin_size


def _pile_fourier_r(yy, l, bin_size=1.0):
    four = bin_size * l * np.fft.rfft(yy)  # Discrete Fourier Transform of lambda*yy
    pile_factor = np.exp(l) - 1  # exp(lambda)-1
    piled_four = list(map(lambda t: (np.exp(t) - 1) / pile_factor, four))  # Piled-up function in Fourier Space
    return np.fft.irfft(piled_four, len(yy)) / bin_size


def pile(yy, l, method='Fourier', series_order=10, bin_size=None, zero_pad=None):
    """
    Calculate a piled spectrum.

    Args:
        yy (list of float): vector defining the original spectrum.
        l (float): poisson piling parameter.
        method (str): the method used to calculate the piling. Available methods include:

            * "series": A series expansion of the convolution exponential.
            * "fourier": "Exact" solution in the fourier domain.
            * "fourier_c": Same as fourier, using the fft instead of rfft.
            * "fourier_series": A series expansion in the fourier domain.


        series_order (int): the number of terms used in a series expansion method.
        bin_size(float): dE in the spectrum. If None, automatically chosen so it is normalized.
        zero_pad (int): Number of zeros to pad to the end of yy.

    Returns:
        (`numpy.ndarray`): The piled spectrum.
        
    """
    if l == 0.0:  # If pile-up is null, just copy the original spectrum
        return yy[:]

    if not bin_size:
        bin_size = 1 / sum(yy)

    if zero_pad:
        if isinstance(zero_pad, int):
            yy = np.pad(yy, (0, zero_pad), 'constant')
        else:
            raise TypeError("zero_pad must be an integer.")

    if not isinstance(method, str):
        print("Selected method is not a string. Using Fourier.", file=sys.stderr)
        r = _pile_fourier(yy, l, bin_size=bin_size)
    else:
        m = method.lower()
        if m == 'series':
            r = _pile_series(yy, l, series_order, bin_size=bin_size)
        elif m == 'fourier':
            r = _pile_fourier_r(yy, l, bin_size=bin_size)
        elif m == "fourierc" or m == "fourier_c" or m == "fourier-c":
            r = _pile_fourier(yy, l, bin_size=bin_size)
        elif m == 'fourier_series':
            r = _pile_fourier_series(yy, l, series_order, bin_size=bin_size)
        else:
            print("Unknown method selected. Using Fourier.", file=sys.stderr)
            r = _pile_fourier(yy, l, bin_size=bin_size)

    if zero_pad:
        return r[:-zero_pad]
    else:
        return r


def _depile_series(yy, l, n, bin_size=1.0):
    # BEWARE: This method fails to depile spectra... but the failure is mathematical!
    # Note this is analogous to the Mercator series, which only converges in (-1,1]
    f = np.array(np.exp(l) - 1) * yy  # the depiled-up function
    f_i = np.copy(yy)  # i-th convolution power of yy
    n_fact = 1  # (-1)^(i+1)*(exp(lambda)-1)^i
    # The first one is already added
    for i in range(2, n + 1):
        n_fact *= -1 * (np.exp(l) - 1)
        f_i = np.convolve(f_i, yy)  # Longer first to avoid swapping
        f_i /= sum(f_i) * bin_size
        f.resize(f_i.shape)
        f += f_i * np.array(n_fact / i)

    return f / np.array(l)


def _depile_fourier(yy, l, bin_size=1.0):
    four = bin_size * np.fft.fft(yy)  # Discrete Fourier Transform of yy
    depile_factor = np.exp(l) - 1  # exp(lambda)-1
    depiled_four = list(map(lambda t: np.log(1 + depile_factor * t) / l, four))  # Depiled function in Fourier Space
    return np.real(np.fft.ifft(depiled_four)) / bin_size


def _depile_fourier_r(yy, l, bin_size=1.0):
    four = bin_size * np.fft.rfft(yy)  # Discrete Fourier Transform of yy
    depile_factor = np.exp(l) - 1  # exp(lambda)-1
    depiled_four = list(map(lambda t: np.log(1 + depile_factor * t) / l, four))  # Depiled function in Fourier Space
    return np.fft.irfft(depiled_four, len(yy)) / bin_size


def _depile_fourier_series(yy, l, n, bin_size=1.0):
    four = bin_size * np.fft.fft(yy)  # Discrete Fourier Transform of yy
    depile_factor = np.exp(l) - 1  # exp(lambda)-1
    depiled_four = list(
        map(lambda t: (_mercator(depile_factor * t, n)) / l, four))  # Piled-up function in Fourier Space
    return np.real(np.fft.ifft(depiled_four)) / bin_size


def _depile_nonparametric_fit(yy, l, depiled_0=None, bin_size=1.0):
    def distance_function(yy_tentative):
        return sqeuclidean(pile(yy_tentative, l, bin_size=bin_size), yy)

    if depiled_0 is None:
        depiled_0 = np.repeat(1 / len(yy), len(yy))

    x, y, d = fmin_l_bfgs_b(distance_function, depiled_0, bounds=[(0, np.inf) for _ in range(len(yy))],
                            approx_grad=True)
    return x


def _depile_parametric_fit(yy, l, f, par_0=None, bin_size=1.0, fit_pars=None):
    xx = np.linspace(0, len(yy) * bin_size, len(yy))

    def _piled_function(*args):
        return pile(f(*args), l)

    popt, pcov = curve_fit(_piled_function, np.linspace(0, len(yy) * bin_size, len(yy)), yy, p0=par_0)
    if fit_pars is not None:
        fit_pars.append(popt)
        fit_pars.append(pcov)
    return f(xx, *popt)


def depile(yy, l, method='Fourier', series_order=20, bin_size=None, zero_pad=None, f=None,
           par_0=None, fit_pars=None, depiled_0=None):
    """
        Calculate a depiled spectrum.

        Args:
            yy (list of float): vector defining the piled spectrum.
            l (float): poisson piling parameter.
            method (str): the method used to calculate the depiling. Available methods include:

                * "series": A series expansion analogous to the Mercator series. **Might not converge**.
                * "fourier": "Exact" solution in the fourier domain.
                * "fourier_c": Same as fourier, using the fft instead of rfft.
                * "fourier_series": A series expansion in the fourier domain.
                * "parametric": Non-linear least squares fit a function using `scipy.optimize.curve_fit`.
                * "nonparametric": Non-linear least squares vector minimization using `scipy.optimize.fmin_l_bfgs_b`.


            series_order (int): the number of terms used in a series expansion method.
            f (callable): if "parametric" is used, the model function, which takes the independent variable as as the first argument and the parameters to fit as separate remaining arguments.
            par_0 (list of float): if "parametric" is used, the initial guess for the parameters. If None, then the initial values will all be 1 (if the number of parameters for the function can be determined using introspection, otherwise a ValueError is raised).
            fit_pars (list): If "parametric" is used, provide an empty list to recover the best-fit parameters and the covariance estimation. See `scipy.optimize.curve_fit`. 
            depiled_0 (list of float): if "nonparametric" is used, the initial guess for the distribution. If None, the uniform distribution will be the starting distribution.
            bin_size (float): dE in the spectrum. If None, chosen so it is normalized.
            zero_pad (int): Number of zeros to pad to the end of yy.

        Returns:
            (`numpy.ndarray`): The depiled spectrum.
            
        Note:
            "parametric" and "nonparametric" methods might return non-global best-fits. Also, the time taken might be long, specially in "nonparametric".
            
    """

    if l == 0.0:  # If pile-up is null, just copy the original spectrum
        return yy[:]

    if not bin_size:
        bin_size = 1 / sum(yy)

    if zero_pad:
        if isinstance(zero_pad, int):
            yy = np.pad(yy, (0, zero_pad), 'constant')
        else:
            raise TypeError("zero_pad must be an integer.")

    if not isinstance(method, str):
        print("Selected method is not a string. Using Fourier.", file=sys.stderr)
        return _depile_fourier(yy, l)

    m = method.lower()
    if m == 'series':
        r = _depile_series(yy, l, series_order, bin_size=bin_size)
    elif m == 'fourier':
        r = _depile_fourier_r(yy, l, bin_size=bin_size)
    elif m == "fourierc" or m == "fourier_c" or m == "fourier-c":
        r = _depile_fourier(yy, l, bin_size=bin_size)
    elif m == 'fourier_series':
        r = _depile_fourier_series(yy, l, series_order, bin_size=bin_size)
    elif m == 'nonparametric':
        r = _depile_nonparametric_fit(yy, l, bin_size=bin_size, depiled_0=depiled_0)
    elif m == 'parametric':
        if f is None:
            raise TypeError("A function must be supplied to make a parametric fit.")
        r = _depile_parametric_fit(yy, l, f, par_0=par_0, bin_size=bin_size, fit_pars=fit_pars)
    else:
        print("Unknown method selected. Using Fourier.", file=sys.stderr)
        r = _depile_fourier(yy, l, bin_size=bin_size)

    if zero_pad:
        return r[:-zero_pad]
    else:
        return r


def piled_sample(l, f_sample, size=None):
    """ Sample from a distributions with an added poissonian pile-up"""
    # Sample once and add latter is faster than sampling each time
    event_counts = np.random.poisson(l, size)
    all_samples = f_sample(sum(event_counts))
    cumulative = 0
    for i in event_counts:
        if i == 0:
            continue
        yield sum(all_samples[cumulative:cumulative + i])
        cumulative += i
