"""
Key Profiles and utilities for key identification algorithms
"""
import numpy as np
from scipy.linalg import circulant

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
        ('B' 'minor', 2)]


# Krumhansl Kessler Key Profiles

# From Krumhansl's "Cognitive Foundations of Musical Pitch" pp.30
key_prof_maj_kk = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])

key_prof_min_kk = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Temperley Key Profiles

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
