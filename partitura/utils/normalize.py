#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains normalization utilities
"""
import numpy as np

DEFAULT_NORM_FUNCS = {
    "FEATURE": {"func": lambda x: x / 127, # some normalization function
                "kvargs": {}}, # some keyword arguments
    # fill up with all note and performance features
}

EPSILON=0.0001


def range_normalize(array, 
                    min_value = None, 
                    max_value = None,
                    log = False,
                    exp = False,
                    hard_clip = True):
    
    """
    Linear mapping a vector from range [min_value, max_value] to [0, 1].
    Preprocessing possible with log and exp.
    Values exceeding the range [0, 1] are clipped to 0 or 1 if 
    clip is True, otherwise they are extrapolated.
    """
    if min_value is None:
        min_value = array.min()
    if max_value is None:
        max_value = array.max()
    if log:
        array = np.log(np.abs(array) + EPSILON)
    if exp:
        array = np.exp(array)
    array = (array - min_value) / (max_value - min_value)
    if hard_clip:
        return np.clip(array, 0, 1)
    else:
        return array
    

def range_normalize(array, 
                    min_value = -3.0, 
                    max_value = 3.0,
                    log = False,
                    exp = False,
                    clip = True):
    
    """
    Compute zero mean and unit variance of a vector.
    Preprocessing possible with log and exp.
    Values exceeding the range [-min_value, max_value] 
    are clipped if clip is True.
    """

    if log:
        array = np.log(np.abs(array) + EPSILON)
    if exp:
        array = np.exp(array)

    array = (array - array.mean()) / array.std()
    if clip:
        return np.clip(array, min_value, max_value)
    else:
        return array


def minmaxrange_normalize(array):
    """
    Linear mapping a vector from range [min_value, max_value] to [0, 1].
    """
    return (array - array.min()) / (array.max() - array.min())


def normalize(array, 
              norm_funcs = DEFAULT_NORM_FUNCS,
              norm_func_fallback = minmaxrange_normalize,
              default_value = -1.0):
    
    """
    Normalize a note array.
    May include note features as well as performance features.
    
    Parameters
    ----------
    array : np.ndarray
        The performance array to be normalized.
    norm_funcs : dict
        A dictionary of normalization functions for each feature.
    
    Returns
    -------
    array : np.ndarray
        The normalized performance array.
    """

    for feature in array.dtype.names:

        # use mask for non-default values and don't change default values
        non_default_mask = array[feature] != default_value

        # check whether the feature has non-uniform values
        if len(np.unique(array[feature][non_default_mask])) == 1:
            array[feature][non_default_mask] = 0.0
        else:
            # check whether a normalization function is defined for the feature
            if feature not in norm_funcs:
                array[feature][non_default_mask] = norm_func_fallback(array[feature][non_default_mask])
            else:
                array[feature][non_default_mask] = norm_funcs[feature]["func"](array[feature][non_default_mask], 
                                                                               **norm_funcs[feature]["kvargs"])     
    
    return array
