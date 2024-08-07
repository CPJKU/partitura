#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module implements Krumhansl and Schmuckler key estimation method.

References
----------
.. [2] Krumhansl, Carol L. (1990) "Cognitive foundations of musical pitch",
       Oxford University Press, New York.
"""
import numpy as np
from scipy.linalg import circulant
from partitura.utils.music import ensure_notearray
from partitura.utils.globals import (
    KEYS,
    key_prof_maj_kk,
    key_prof_min_kk,
    key_prof_maj_cbms,
    key_prof_min_cbms,
    key_prof_maj_kp,
    key_prof_min_kp,
    VALID_KEY_PROFILES,
)

__all__ = ["estimate_key"]

# List of labels for each key (Use enharmonics as needed).
# Each tuple is (key root name, mode, fifths)
# The key root name is equal to that with the smallest fifths in
# the circle of fifths.


def build_key_profile_matrix(key_prof_maj, key_prof_min):
    """
    Generate Matrix of key profiles
    """
    # Normalize Key profiles
    key_prof_maj /= np.sum(key_prof_maj)
    key_prof_min /= np.sum(key_prof_min)

    # Create matrix of key profiles
    Key_prof_mat = np.vstack(
        (circulant(key_prof_maj).transpose(), circulant(key_prof_min).transpose())
    )

    return Key_prof_mat


# Key profile matrices
KRUMHANSL_KESSLER = build_key_profile_matrix(key_prof_maj_kk, key_prof_min_kk)
CMBS = build_key_profile_matrix(key_prof_maj_cbms, key_prof_min_cbms)
KOSTKA_PAYNE = build_key_profile_matrix(key_prof_maj_kp, key_prof_min_kp)


def estimate_key(note_info, method="krumhansl", *args, **kwargs):
    """
    Estimate key of a piece by comparing the pitch statistics of the
    note array to key profiles [2]_, [3]_.

    Parameters
    ----------
    note_info : structured array, `Part` or `PerformedPart`
        Note information as a `Part` or `PerformedPart` instances or
        as a structured array. If it is a structured array, it has to
        contain the fields generated by the `note_array` properties
        of `Part` or `PerformedPart` objects. If the array contains
        onset and duration information of both score and performance,
        (e.g., containing both `onset_beat` and `onset_sec`), the score
        information will be preferred.
    method : {'krumhansl'}
        Method for estimating the key. For now 'krumhansl' is the only
        supported method.
    args, kwargs
        Positional and Keyword arguments for the key estimation method

    Returns
    -------
    str
       String representing the key name (i.e., Root(alteration)(m if minor)).
       See `partitura.utils.key_name_to_fifths_mode` and
       `partitura.utils.fifths_mode_to_key_name`.

    References
    ----------
    .. [2] Krumhansl, Carol L. (1990) "Cognitive foundations of musical pitch",
           Oxford University Press, New York.
    .. [3] Temperley, D. (1999) "What's key for key? The Krumhansl-Schmuckler
           key-finding algorithm reconsidered". Music Perception. 17(1),
           pp. 65--100.

    """

    if method not in ("krumhansl",):
        raise ValueError('For now the only valid method is "krumhansl"')

    if method == "krumhansl":
        kid = ks_kid

        if "key_profiles" not in kwargs:
            kwargs["key_profiles"] = "krumhansl_kessler"
        else:
            if kwargs["key_profiles"] not in VALID_KEY_PROFILES:
                raise ValueError(
                    "Invalid key_profiles. " 'Valid options are "ks", "cmbs" or "kp"'
                )

    note_array = ensure_notearray(note_info)

    return kid(note_array, *args, **kwargs)


def format_key(root, mode, fifths):
    return "{}{}".format(root, "m" if mode == "minor" else "")


def ks_kid(note_array, key_profiles=KRUMHANSL_KESSLER, return_sorted_keys=False):
    """
    Estimate key of a piece using the Krumhansl-Schmuckler
    algorithm.
    """
    if isinstance(key_profiles, str):
        if key_profiles in ("ks", "krumhansl_kessler"):
            key_profiles = KRUMHANSL_KESSLER
        elif key_profiles in ("temperley", "cmbs"):
            key_profiles = CMBS
        elif key_profiles in ("kp", "kostka_payne"):
            key_profiles = KOSTKA_PAYNE

        else:
            raise ValueError(
                "Invalid key_profiles. " 'Valid options are "ks", "cmbs" or "kp"'
            )

    corrs = _similarity_with_pitch_profile(
        note_array=note_array, key_profiles=key_profiles, similarity_func=corr
    )

    if return_sorted_keys:
        return [format_key(*KEYS[i]) for i in np.argsort(corrs)[::-1]]
    else:
        return format_key(*KEYS[corrs.argmax()])


def corr(x, y):
    return np.corrcoef(x, y)[0, 1]


def _similarity_with_pitch_profile(
    note_array,
    key_profiles=KRUMHANSL_KESSLER,
    similarity_func=corr,
    normalize_distribution=False,
):
    from partitura.utils.music import get_time_units_from_note_array

    _, duration_unit = get_time_units_from_note_array(note_array)
    # Get pitch classes
    pitch_classes = np.mod(note_array["pitch"], 12)

    # Compute weighted key distribution
    pitch_distribution = np.array(
        [
            note_array[duration_unit][np.where(pitch_classes == pc)[0]].sum()
            for pc in range(12)
        ]
    )

    if normalize_distribution:
        # normalizing is unnecessary for computing the correlation, but might
        # be necessary for other similarity metrics
        pitch_distribution = pitch_distribution / float(pitch_distribution.sum())

    # Compute correlation with key profiles
    similarity = np.array(
        [similarity_func(pitch_distribution, kp) for kp in key_profiles]
    )

    return similarity
