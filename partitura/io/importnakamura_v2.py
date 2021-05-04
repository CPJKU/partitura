"""
This module contains methods for parsing score-to-performance alignments
in Nakamura et al.'s [1]_ format.

References
----------
.. [1] Nakamura, E., Yoshii, K. and Katayose, H. (2017) "Performance Error
       Detection and Post-Processing for Fast and Accurate Symbolic Music
       Alignment"
"""

import numpy as np


def load_nakamuracorresp_v2(fn):
    note_array_dtype = [("onset", "f4"), ("pitch", "i4"), ("id", "U256")]
    dtype = [
        ("alignID", "U256"),
        ("alignOntime", "f"),
        ("alignSitch", "U256"),
        ("alignPitch", "i"),
        ("alignOnvel", "i"),
        ("refID", "U256"),
        ("refOntime", "f"),
        ("refSitch", "U256"),
        ("refPitch", "i"),
        ("refOnvel", "i"),
    ]
    result = np.loadtxt(fn, dtype=dtype, comments="//")

    align_valid = result["alignID"] != "*"
    n_align = sum(align_valid)
    align = np.empty((n_align,), dtype=note_array_dtype)
    align[:] = result[["alignOntime", "alignPitch", "alignID"]][align_valid]

    ref_valid = result["refID"] != "*"
    n_ref = sum(ref_valid)
    ref = np.empty((n_ref,), dtype=note_array_dtype)
    ref[:] = result[["refOntime", "refPitch", "refID"]][ref_valid]

    alignment = []
    for alignID, refID in result[["alignID", "refID"]]:
        if alignID == "*":
            label = "deletion"
        elif refID == "*":
            label = "insertion"
        else:
            label = "match"
        alignment.append(dict(label=label, performance_id=alignID, score_id=refID))

    return align, ref, alignment
