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
import re
from partitura.utils import (pitch_spelling_to_midi_pitch,
                             ensure_pitch_spelling_format)
from partitura.utils.music import SIGN_TO_ALTER

NAME_PATT = re.compile(r"([A-G]{1})([xb\#]*)(\d+)")


def load_nakamuracorresp_v2(fn):
    note_array_dtype = [("onset_sec", "f4"), ("pitch", "i4"), ("id", "U256")]
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
            alnote = dict(label="deletion",
                          score_id=refID)
        elif refID == "*":
            alnote = dict(label="insertion",
                          performance_id=alignID)
        else:
            alnote = dict(label="match",
                          score_id=refID,
                          performance_id=alignID)
        alignment.append(alnote)

    return align, ref, alignment


def load_nakamuramatch(fn, ppqn=1000):
    # ID (onset time) (offset time) (spelled pitch) (onset velocity) (offset velocity)
    # channel (match status) (score time) (note ID) (error index) (skip index)
    perf_dtype = [("onset_sec", "f4"),
                  ("duration_sec", "f4"),
                  ("pitch", "i4"),
                  ("velocity", "i4"),
                  ("channel", "i4"),
                  ("id", "U256"),
                  ]
    score_dtype = [("onset_div", "i4"),
                   # ("duration_div", "i4"),
                   ("onset_quarter", "f4"),
                   # ("duration_quarter", "f4"),
                   ("pitch", "i4"),
                   ("step", "U256"),
                   ("alter", "i4"),
                   ("octave", "i4"),
                   ("id", "U256")
                   ]
    dtype = [
        ("alignID", "U256"),
        ("alignOntime", "f"),
        ("alignOfftime", "f"),
        ("alignSitch", "U256"),
        ("alignOnvel", "i"),
        ("alignOffvel", "i"),
        ("alignChannel", "i"),
        ("matchstatus", "i"),
        ("refOntime", "f"),
        ("refID", "U256"),
        ("errorindex", "i"),
        ("skipindex", "U256"),
    ]
    dtype_missing = [("refOntime", "f"),
                     ("refID", "U256")]
    pattern = r"//Missing\s(\d+)\t(.+)"
    # load alignment notes
    result = np.loadtxt(fn, dtype=dtype, comments="//")
    # load missing notes
    missing = np.fromregex(fn, pattern, dtype=dtype_missing)

    midi_pitch = np.array(
        [note_name_to_midi_pitch_(n.replace('#', r'\#'))
         for n in result["alignSitch"]])

    align_valid = result["alignID"] != "*"
    n_align = sum(align_valid)
    align = np.empty((n_align,), dtype=perf_dtype)
    align["id"] = result["alignID"][align_valid]
    align["onset_sec"] = result["alignOntime"]
    align["duration_sec"] = (result["alignOfftime"][align_valid] -
                             result["alignOntime"][align_valid])
    align["pitch"] = midi_pitch[align_valid]
    align["velocity"] = result["alignOnvel"][align_valid]
    align["channel"] = result["alignChannel"][align_valid]

    ref_valid = result["refID"] != "*"
    n_valid = sum(ref_valid)
    n_ref = n_valid + len(missing)
    ref = np.empty((n_ref,), dtype=score_dtype)

    ref["id"][:n_valid] = result["refID"][ref_valid]
    ref["id"][n_valid:] = missing["refID"]
    ref["onset_div"][:n_valid] = result["refOntime"][ref_valid]
    ref["onset_div"][n_valid:] = missing["refOntime"]
    ref["onset_quarter"][:n_valid] = result["refOntime"][ref_valid] / ppqn
    ref["onset_quarter"][n_valid:] = missing["refOntime"] / ppqn
    ref["pitch"][:n_valid] = midi_pitch[ref_valid]
    ref["pitch"][n_valid:] = -1
    pitch_spelling = [NAME_PATT.search(nn).groups()
                      for nn in result["alignSitch"]]

    pitch_spelling = np.array(
        [(ps[0], SIGN_TO_ALTER[ps[1] if ps[1] != "" else "n"], int(ps[2]))
         for ps in pitch_spelling])
    # add pitch spelling information
    ref["step"][:n_valid] = pitch_spelling[ref_valid][:, 0]
    ref["alter"][:n_valid] = pitch_spelling[ref_valid][:, 1]
    ref["octave"][:n_valid] = pitch_spelling[ref_valid][:, 2]

    alignment = []
    for alignID, refID in result[["alignID", "refID"]]:
        if alignID == "*":
            alnote = dict(label="deletion",
                          score_id=refID)
        elif refID == "*":
            alnote = dict(label="insertion",
                          performance_id=alignID)
        else:
            alnote = dict(label="match",
                          score_id=refID,
                          performance_id=alignID)
        alignment.append(alnote)

    for refID in missing["refID"]:
        alignment.append(dict(label="deletion",
                              score_id=refID))

    return align, ref, alignment


def load_nakamuraspr(fn):
    """
    TODO
    ----
    * Import pedal information
    """
    # ID (onset time) (offset time) (spelled pitch) (onset velocity)
    # (offset velocity) channel

    note_array_dtype = [("onset_sec", "f4"),
                        ("duration_sec", "f4"),
                        ("pitch", "i4"),
                        ("velocity", "i4"),
                        ("channel", "i4"),
                        ("id", "U256")]
    dtype = [
        ("ID", "U256"),
        ("Ontime", "f"),
        ("Offtime", "f"),
        ("Sitch", "U256"),
        ("Onvel", "i"),
        ("Offvel", "i"),
        ("Channel", "i")
    ]

    pattern = r"(\d+)\t(.+)\t(.+)\t(.+)\t(.+)\t(.+)\t(.+)"

    result = np.fromregex(fn, pattern, dtype=dtype)
    note_array = np.empty(len(result), dtype=note_array_dtype)

    note_array["id"] = result["ID"]
    note_array["onset_sec"] = result["Ontime"]
    note_array["duration_sec"] = result["Offtime"] - result["Ontime"]
    note_array["pitch"] = np.array(
        [note_name_to_midi_pitch_(n.replace('#', r'\#'))
         for n in result["Sitch"]])
    note_array["velocity"] = result["Onvel"]
    note_array["channel"] = result["Channel"]

    return note_array


def note_name_to_midi_pitch_(note_name):

    note_info = NAME_PATT.search(note_name)

    if note_info is None:
        raise ValueError("Incorrect note name")

    step, alter, octave = note_info.groups()
    step, alter, octave = ensure_pitch_spelling_format(
        step=step,
        alter=alter if alter != "" else "n",
        octave=int(octave))

    return pitch_spelling_to_midi_pitch(step, alter, octave)
