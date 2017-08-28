#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""A module to study the pile-up of distributions"""

from __future__ import print_function

import sys
import numpy as np

__author__ = 'Dih5'
__version__ = "0.1.0"


def poisson_lambda(rate):
    """Return the poisson characteristic parameter, given a rate of counting/pulse frequency"""
    return -np.log(1 - rate)


def exp_n(x, n):
    """Maclaurin series of order n of the exponential"""
    return np.sum([x ** j / np.math.factorial(j) for j in range(0, n + 1)])


def mercator(x, n):
    """Mercator series of order n of the exponential"""
    return np.sum([(-1) ** (j + 1) * x ** j / j for j in range(1, n + 1)])


def _poisson_pile_series(y, l, n, bin_size=1):
    f = np.array(l) * y  # the piled-up function
    f_i = np.copy(f)  # i-th convolution power of f
    n_fact = l  # Hold lambda^i/i!
    # The first one is already added
    for i in range(2, n + 1):
        n_fact *= l / i
        f_i = np.convolve(f_i, y)  # Longer first to avoid swapping
        f_i /= sum(f_i) * bin_size
        f.resize(f_i.shape)
        f += f_i * np.array(n_fact)

    return f[:len(y)] * np.array(1 / (np.exp(l) - 1))  # Make sure to cut added tails


def _poisson_pile_fourier(y, l, bin_size=1):
    four = bin_size * l * np.fft.fft(y)  # Discrete Fourier Transform of lambda*y
    pile_factor = np.exp(l) - 1  # exp(lambda)-1
    piled_four = list(map(lambda t: (np.exp(t) - 1) / pile_factor, four))  # Piled-up function in Fourier Space
    return np.real(np.fft.ifft(piled_four)) / bin_size


def _poisson_pile_fourier_series(y, l, n, bin_size=1):
    four = bin_size * l * np.fft.fft(y)  # Discrete Fourier Transform of lambda*y
    pile_factor = np.exp(l) - 1  # exp(lambda)-1
    piled_four = list(map(lambda t: (exp_n(t, n) - 1) / pile_factor, four))  # Piled-up function in Fourier Space
    return np.real(np.fft.ifft(piled_four)) / bin_size


def _poisson_pile_fourier_r(y, l, bin_size=1):
    four = bin_size * l * np.fft.rfft(y)  # Discrete Fourier Transform of lambda*y
    pile_factor = np.exp(l) - 1  # exp(lambda)-1
    piled_four = list(map(lambda t: (np.exp(t) - 1) / pile_factor, four))  # Piled-up function in Fourier Space
    return np.fft.irfft(piled_four, len(y)) / bin_size


def poisson_pile(y, l, method='Fourier', series_order=10, bin_size=None, zero_pad=None):
    """
    Calculate a piled spectrum.

    Args:
        y: vector defining the original spectrum.
        l: poisson piling parameter.
        method (str): the method used to calculate the piling. Available methods include:

            * "series": A series expansion of the convolution exponential.
            * "fourier": "Exact" solution in the fourier domain.
            * "fourier_c": Same as fourier, using the fft instead of rfft.
            * fourier_series: A series expansion in the fourier domain.


        series_order: the number of terms used in a series expansion method.
        bin_size: dE in the spectrum. If None, chosen so it is normalized.

    Returns:
        The piled spectrum
    """
    if l == 0.0:  # If pile-up is null, just copy the original spectrum
        return y[:]

    if not bin_size:
        bin_size = 1 / sum(y)

    if zero_pad:
        if isinstance(zero_pad, int):
            y = np.pad(y, (0, zero_pad), 'constant')
        else:
            raise TypeError("zero_pad must be an integer.")

    if not isinstance(method, str):
        print("Selected method is not a string. Using Fourier.", file=sys.stderr)
        r = _poisson_pile_fourier(y, l, bin_size=bin_size)
    else:
        m = method.lower()
        if m == 'series':
            r = _poisson_pile_series(y, l, series_order, bin_size=bin_size)
        elif m == 'fourier':
            r = _poisson_pile_fourier_r(y, l, bin_size=bin_size)
        elif m == "fourierc" or m == "fourier_c" or m == "fourier-c":
            r = _poisson_pile_fourier(y, l, bin_size=bin_size)
        elif m == 'fourier_series':
            r = _poisson_pile_fourier_series(y, l, series_order, bin_size=bin_size)
        else:
            print("Unknown method selected. Using Fourier.", file=sys.stderr)
            r = _poisson_pile_fourier(y, l, bin_size=bin_size)

    if zero_pad:
        return r[:-zero_pad]
    else:
        return r


def _poisson_depile_series(y, l, n, bin_size=1):
    # BEWARE: This method fails to depile spectra... but the failure is mathematical!
    # Note this is analogous to the Mercator series, which only converges in (-1,1]
    f = np.array(np.exp(l) - 1) * y  # the depiled-up function
    f_i = np.copy(y)  # i-th convolution power of y
    n_fact = 1  # (-1)^(i+1)*(exp(lambda)-1)^i
    # The first one is already added
    for i in range(2, n + 1):
        n_fact *= -1 * (np.exp(l) - 1)
        f_i = np.convolve(f_i, y)  # Longer first to avoid swapping
        f_i /= sum(f_i) * bin_size
        f.resize(f_i.shape)
        f += f_i * np.array(n_fact / i)

    return f / np.array(l)


def _poisson_depile_fourier(y, l, bin_size=1):
    four = bin_size * np.fft.fft(y)  # Discrete Fourier Transform of y
    depile_factor = np.exp(l) - 1  # exp(lambda)-1
    depiled_four = list(map(lambda t: np.log(1 + depile_factor * t) / l, four))  # Depiled function in Fourier Space
    return np.real(np.fft.ifft(depiled_four)) / bin_size


def _poisson_depile_fourier_r(y, l, bin_size=1):
    four = bin_size * np.fft.rfft(y)  # Discrete Fourier Transform of y
    depile_factor = np.exp(l) - 1  # exp(lambda)-1
    depiled_four = list(map(lambda t: np.log(1 + depile_factor * t) / l, four))  # Depiled function in Fourier Space
    return np.fft.irfft(depiled_four, len(y)) / bin_size


def _poisson_depile_fourier_series(y, l, n, bin_size=1):
    four = bin_size * np.fft.fft(y)  # Discrete Fourier Transform of y
    depile_factor = np.exp(l) - 1  # exp(lambda)-1
    depiled_four = list(map(lambda t: (mercator(depile_factor * t, n)) / l, four))  # Piled-up function in Fourier Space
    return np.real(np.fft.ifft(depiled_four)) / bin_size


def poisson_depile(y, l, method='Fourier', series_order=20, bin_size=None, zero_pad=None):
    """
        Calculate a piled spectrum.

        Args:
            y: vector defining the piled spectrum.
            l: poisson piling parameter.
            method (str): the method used to calculate the depiling. Available methods include:

                * "series": A series expansion of the convolution exponential.
                * "fourier": "Exact" solution in the fourier domain.
                * "fourier_c": Same as fourier, using the fft instead of rfft.
                * fourier_series: A series expansion in the fourier domain.


            series_order: the number of terms used in a series expansion method.
            bin_size: dE in the spectrum. If None, chosen so it is normalized.

        Returns:
            The depiled spectrum
    """

    if l == 0.0:  # If pile-up is null, just copy the original spectrum
        return y[:]

    if not bin_size:
        bin_size = 1 / sum(y)

    if zero_pad:
        if isinstance(zero_pad, int):
            y = np.pad(y, (0, zero_pad), 'constant')
        else:
            raise TypeError("zero_pad must be an integer.")

    if not isinstance(method, str):
        print("Selected method is not a string. Using Fourier.", file=sys.stderr)
        return _poisson_depile_fourier(y, l)

    m = method.lower()
    if m == 'series':
        r = _poisson_depile_series(y, l, series_order, bin_size=bin_size)
    elif m == 'fourier':
        r = _poisson_depile_fourier_r(y, l, bin_size=bin_size)
    elif m == "fourierc" or m == "fourier_c" or m == "fourier-c":
        r = _poisson_depile_fourier(y, l, bin_size=bin_size)
    elif m == 'fourier_series':
        r = _poisson_depile_fourier_series(y, l, series_order, bin_size=bin_size)
    else:
        print("Unknown method selected. Using Fourier.", file=sys.stderr)
        r = _poisson_depile_fourier(y, l, bin_size=bin_size)

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
