#!/usr/bin/env python

"""
Utility functions related to data/signal processing

"""

import numpy as np
import logging
from itertools import product
import scipy.sparse.linalg
from scipy.special import gamma
import scipy.signal as signal
from numpy.polynomial.polynomial import polyfit as np_polyfit

LOGGER = logging.getLogger()

def cosine_similarity(x, y):
    return (np.dot(x,y)/(np.sum(x**2)**.5 * np.sum(y**2)**.5))[0]

def smooth(x,k):
    """
    Convolve `x` with a Hamming window of length `k`.

    :param x: signal to be convolved
    :param k: size of Hamming window in samples

    :returns: convolved signal with same length as x
    """
    
    w = signal.hamming(k)
    w /= np.sum(w)
    return np.convolve(x,w,'same')

def smooth_hanning(x,k):
    """
    Convolve `x` with a Hanning window of length `k`.

    :param x: signal to be convolved
    :param k: size of Hanning window in samples

    :returns: convolved signal with same length as x
    """
    return np.convolve(x,signal.hanning(k)*2/(k-1),'same')

def find_peaks(x):
    return np.where(np.diff(np.sign(np.diff(x)))<0)[0]+1

def find_valleys(x):
    return np.where(np.diff(np.sign(np.diff(x)))>0)[0]+1

def inverse_gamma_pdf(x,a=1,b=2): 
    #return ((b**a)/gamma(a))*x**(-a-1)*np.exp(-b/x)
    return (((b**int(np.floor(a/2.)))/gamma(a))*(x**(-a-1)*np.exp(-b/x)))*(b**int(np.ceil(a/2.)))

def sigmoid(V):
    return 1/(1+np.exp(-V))

def polyfit_normal(x, y, order=2):
    """
    Returns a polynomial function that is the 
    least squares approximation of the input data
    """
    powers = np.arange(order+1)
    m, residuals = np.linalg.lstsq(np.array([[d**k for k in powers] for d in x]),y)[:2]
    def poly(xin,coeffs=m):
        return np.dot(np.vstack([np.power(xin,powers[i]) for i in powers]).T,m)
    poly.coeffs = m
    poly.residuals = residuals
    return poly

def polyfit_weighted(x, y, order=2, w = None):
    """
    Returns a polynomial function that is the 
    least squares approximation of the input data
    """

    powers = np.arange(order+1)
    m, (residuals, _rank, _sing_vals, _rcond) = np_polyfit(x, y, order, full = True, w = w)

    def poly(xin,coeffs=m):
        return np.dot(np.vstack([np.power(xin,powers[i]) for i in powers]).T,m)

    poly.coeffs = m
    poly.residuals = residuals

    return poly

def polyfit_regularized(x, y, order=2):
    """
    Returns a polynomial function that is the 
    least squares approximation of the input data
    """
    powers = np.arange(order+1)
    factor = 10**12
    y = y*factor
    damping = 100000000.
    m = np.array([scipy.sparse.linalg.lsqr(np.array([[d**k for k in powers] for d in x]),y[:,i], damp = damping)[0]
                  for i in range(y.shape[1])])
    print(m)
    def poly(xin,coeffs=m):
        return np.dot(np.vstack([np.power(xin,powers[i]) for i in powers]).T,m)/factor
    poly.coeffs = m
    return poly

polyfit = polyfit_weighted

def normalize(v):
    vmin = np.min(v,0)
    vmax = np.max(v,0)
    if np.isscalar(vmin):
        if vmax > vmin:
            v = (v-vmin)/(vmax-vmin)
    else:
        for i in range(len(vmin)):
            if vmax[i] > vmin[i]:
                v[:,i] = (v[:,i]-vmin[i])/(vmax[i]-vmin[i])
    return v

def rescale(v, vmin=0, vmax=1, in_place = False):
    """
    Rescale a vector to the range [vmin,vmax]
    """
    
    if in_place:
        w = v
    else:
        w = v.copy()

    w -= np.min(w)

    max_w = np.max(w)

    if max_w <= 0.0:
        LOGGER.warning('Refusing to rescale vector with zero standard deviation')
        return v

    w /= max_w

    if (vmax-vmin) != 1:
        w *= (vmax-vmin)
    if vmin != 0:
        w += vmin

    return w

def make_colors(n,noGrey=True):
    """
    Return a list of at least n colors (i.e. triples <i,j,k>,
    where 0 <= i,j,k <=255), ordered from saturated to non-saturated.
    If noGrey = True, no grey tones are returned
    """
    k = int(np.ceil(n**(1.0/3)))
    if noGrey and n > (k**3-k):
        k += 1
    assert k > 1
    basis = np.arange(k)/(k-1)
    combinations = np.array([np.array(i) for i in product(basis,basis,basis)])
    stddevs = np.array([np.std(i) for i in combinations])
    return [(255.0*i).astype(int) for i in combinations[stddevs.argsort()][::-1]]

if __name__ == '__main__':
    pass
