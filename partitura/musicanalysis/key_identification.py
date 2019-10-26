# -*- coding: utf-8 -*-
"""
Krumhansl and Shepard key estimation

"""
import numpy as np
from scipy.linalg import circulant

__all__ = ['estimate_key']

# List of labels for each key (Use enharmonics as needed).
# Each tuple is (key root name, mode, fifths)
# The key root name is equal to that with the smallest fifths in
# the circle of fifths.
KEYS = [('C', 'major', 0),
        ('Db', 'major', -5),
        ('D', 'major', 2),
        ('Eb', 'major', -3),
        ('E', 'major', 4),
        ('F', 'major', -1),
        ('F#', 'major', 6),
        ('G', 'major', 1),
        ('Ab', 'major', -4),
        ('A', 'major', 3),
        ('Bb', 'major', -2),
        ('B', 'major', 5),
        ('C', 'minor', -3),
        ('C#', 'minor', 4),
        ('D', 'minor', -1),
        ('D#', 'minor', 6),
        ('E', 'minor', 1),
        ('F', 'minor', -4),
        ('F#', 'minor', 3),
        ('G', 'minor', -2),
        ('G#', 'minor', 5),
        ('A', 'minor', 0),
        ('Bb', 'minor', -5),
        ('B', 'minor', 2)]


################ Krumhansl--Kessler Key Profiles ########################

# From Krumhansl's "Cognitive Foundations of Musical Pitch" pp.30
key_prof_maj_kk = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])

key_prof_min_kk = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

############### Temperley Key Profiles ###########################

# CBMS (from "Music and Probability" Table 6.1, pp. 86)
key_prof_maj_cbms = np.array(
    [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0])

key_prof_min_cbms = np.array(
    [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0])

# Kostka-Payne (from "Music and Probability" Table 6.1, pp. 86)
key_prof_maj_kp = np.array(
    [0.748, 0.060, 0.488, 0.082, 0.670, 0.460, 0.096, 0.715, 0.104, 0.366, 0.057, 0.400])

key_prof_min_kp = np.array(
    [0.712, 0.048, 0.474, 0.618, 0.049, 0.460, 0.105, 0.747, 0.404, 0.067, 0.133, 0.330])


def build_key_profile_matrix(key_prof_maj, key_prof_min):
    """
    Generate Matrix of key profiles
    """
    # Normalize Key profiles
    key_prof_maj /= np.sum(key_prof_maj)
    key_prof_min /= np.sum(key_prof_min)

    # Create matrix of key profiles
    Key_prof_mat = np.vstack(
        (circulant(key_prof_maj).transpose(),
         circulant(key_prof_min).transpose()))

    return Key_prof_mat


# Key profile matrices
KRUMHANSL_KESSLER = build_key_profile_matrix(key_prof_maj_kk, key_prof_min_kk)
CMBS = build_key_profile_matrix(key_prof_maj_cbms, key_prof_min_cbms)
KOSTKA_PAYNE = build_key_profile_matrix(key_prof_maj_kp, key_prof_min_kp)


def estimate_key(note_array, method='krumhansl', *args, **kwargs):
    """
    Estimate key of a piece by comparing the pitch statistics of the
    note array to key profiles [2]_, [3]_.

    Parameters
    ----------
    note_array : structured array
        Array containing the score
    method : {'krumhansl', 'temperley'}
        Method for estimating the key. Default is 'krumhansl'.
    args, kwargs
        Positional and Keyword arguments for the key estimation method

    Returns
    -------
    root : str
        Root of the key (key name)
    mode : str
        Mode of the key ('major' or 'minor')
    fifths : int
        Position in the circle of fifths
    
    References
    ----------
    .. [2] Krumhansl, Carol L. (1990) "Cognitive foundations of musical pitch",
           Oxford University Press, New York.
    .. [3] Temperley, D. (1999) "What's key for key? The Krumhansl-Schmuckler
           key-finding algorithm reconsidered". Music Perception. 17(1), 
           pp. 65--100.

    """
    if method not in ('krumhansl', ):
        raise ValueError('For now the only valid method is "krumhansl"')

    if method == 'krumhansl':
        kid = ks_kid
    if method == 'temperley':
        kid = ks_kid
        if 'key_profiles' not in kwargs:
            kwargs['key_profiles'] = 'temperley'

    return kid(note_array, *args, **kwargs)


def format_key(root, mode, fifths):
    return '{}{}'.format(root, 'm' if mode == 'minor' else '')

def ks_kid(note_array, key_profiles=KRUMHANSL_KESSLER, return_sorted_keys=False):
    """Estimate key of a piece
    
    """
    if isinstance(key_profiles, str):
        if key_profiles in ('ks', 'krumhansl_kessler'):
            key_profiles = KRUMHANSL_KESSLER
        elif key_profiles in ('temperley', 'cmbs'):
            key_profiles = CMBS
        elif key_profiles in ('kp', 'kostka_payne'):
            key_profiles = KOSTKA_PAYNE

        else:
            raise ValueError('Invalid key_profiles. '
                             'Valid options are "ks", "cmbs" or "kp"')

    # Get pitch classes
    pitch_classes = np.mod(note_array['pitch'], 12)

    # Compute weighted key distribution
    pitch_distribution = np.array([note_array['duration'][np.where(pitch_classes == pc)[0]].sum()
                                   for pc in range(12)])

    # normalizing is unnecessary for computing the corrcoef with the profiles:
    # pitch_distribution = pitch_distribution/float(pitch_distribution.sum())

    # Compute correlation with key profiles
    corrs = np.array([np.corrcoef(pitch_distribution, kp)[0, 1]
                      for kp in key_profiles])

    if return_sorted_keys:
        return [format_key(*KEYS[i]) for i in np.argsort(corrs)[::-1]]
    else:
        return format_key(*KEYS[corrs.argmax()])
