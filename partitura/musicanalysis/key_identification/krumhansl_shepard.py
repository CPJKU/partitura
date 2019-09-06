"""
Krumhansl and Shepard key estimation

TODO
----
* Documentation
"""
import numpy as np

from ._kid_utils import KEYS, KRUMHANSL_KESSLER, CMBS, KOSTKA_PAYNE


def estimate_key(note_array, key_profiles=KRUMHANSL_KESSLER):
    """
    Estimate key of a piece
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

    pitch_distribution /= pitch_distribution.sum()

    # Compute correlation with key profiles
    corrs = np.array([np.corrcoef(pitch_distribution, kp)[0, 1]
                      for kp in key_profiles])

    return KEYS[corrs.argmax()]
