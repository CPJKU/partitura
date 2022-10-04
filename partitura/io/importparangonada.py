#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains methods for parsing Parangonada csv files
"""
import os
import numpy as np

from typing import List

from partitura.performance import PerformedPart, Performance
from partitura.utils.misc import PathLike, deprecated_alias

NOTE_ARRAY_DTYPES = dict(
    onset_sec=("onset_sec", "f4"),
    duration_sec=("duration_sec", "f4"),
    onset_beat=("onset_beat", "f4"),
    duration_beat=("duration_beat", "f4"),
    onset_quarter=("onset_quarter", "f4"),
    duration_quarter=("duration_quarter", "f4"),
    onset_div=("onset_div", "i4"),
    duration_div=("duration_div", "i4"),
    pitch=("pitch", "i4"),
    voice=("voice", "i4"),
    velocity=("velocity", "i4"),
    track=("track", "i4"),
    channel=("channel", "i4"),
    id=("id", "U256"),
)

__all__ = [
    "load_parangonada_alignment",
    "load_parangonada_csv",
]


def _load_csv(filename: PathLike) -> np.ndarray:
    """
    Load a CSV file where the headers are one of the note array columns
    and return a structured array

    Parameters
    ----------
    filename: PathLike
        Path of the CSV file


    Returns
    -------
    struct_array: np.ndarray
        Structured array
    """
    raw_array = np.loadtxt(
        fname=filename,
        delimiter=",",
        comments=None,
        dtype=str,
    )

    columns = raw_array[0]
    dtypes = [NOTE_ARRAY_DTYPES.get(c, (c, "U256")) for c in columns]

    struct_array = np.empty(len(raw_array) - 1, dtype=dtypes)
    for i, (c, dt) in enumerate(zip(columns, dtypes)):

        if dt[-1] == "i4":
            # Weird behavior trying to cast 0.0 as an integer
            struct_array[c] = raw_array[1:, i].astype(float).astype(int)
        else:
            struct_array[c] = raw_array[1:, i].astype(dt[-1])

    return struct_array


@deprecated_alias(outfile="filename")
def load_parangonada_alignment(filename) -> List[dict]:
    """
    load an alignment exported from parangonda.

    Parameters
    ----------
    filename : str
        A path to the alignment csv file

    Returns
    -------
    alignment : list
        A list of note alignment dictionaries.
    """
    array = np.loadtxt(filename, dtype=str, delimiter=",")
    alignment = list()
    # match = 0, deletion  = 1, insertion = 2
    for k in range(1, array.shape[0]):
        if int(array[k, 1]) == 0:
            alignment.append(
                {
                    "label": "match",
                    "score_id": array[k, 2],
                    "performance_id": array[k, 3],
                }
            )

        elif int(array[k, 1]) == 2:
            alignment.append({"label": "insertion", "performance_id": array[k, 3]})

        elif int(array[k, 1]) == 1:
            alignment.append({"label": "deletion", "score_id": array[k, 2]})
    return alignment


def load_parangonada_csv(dirname: PathLike, create_score: bool = False) -> np.ndarray:
    """
    Load Parangonada Project alignment files

    Parameters
    ----------
    dirname : PathLike
        Directory with the CSV files in Parangonada
    create_score: bool
        Create a score. For now it just creats a note array, but the argument
        name was chosen to be consistent with `load_match`.

    Returns
    -------
    performance : partitura.performance.Performance
        The performance in the alignment
    alignment : List of dict
        The main alignment
    zalignment : List of dict
        The secondary alignment (for comparing the first one)
    feature : np.ndarray
        A structured array with note-level feature information
    score_note_array
        A note array containing note information in the score. Will change to a
        score object in a future release!
    """
    # Will the names change in the future?
    perf_note_array_fn = os.path.join(dirname, "ppart.csv")
    score_note_array_fn = os.path.join(dirname, "part.csv")
    alignment_fn = os.path.join(dirname, "align.csv")
    feature_fn = os.path.join(dirname, "feature.csv")
    zalign_fn = os.path.join(dirname, "zalign.csv")

    perf_note_array = _load_csv(perf_note_array_fn)

    performed_part = PerformedPart.from_note_array(perf_note_array)
    performance = Performance(
        performedparts=performed_part,
        id=dirname,
    )

    feature = _load_csv(feature_fn)
    alignment = load_parangonada_alignment(alignment_fn)
    zalignment = load_parangonada_alignment(zalign_fn)

    if create_score:
        # TODO: Generate a Score
        score_note_array = _load_csv(score_note_array_fn)

        return (
            performance,
            alignment,
            zalignment,
            feature,
            score_note_array,
        )

    else:
        return (
            performance,
            alignment,
            zalignment,
            feature,
        )


@deprecated_alias(outfile="filename")
def load_alignment_from_ASAP(filename: PathLike) -> List[dict]:
    """
    load a note alignment of the ASAP dataset.

    Parameters
    ----------
    filename : str
        A path to the alignment tsv file

    Returns
    -------
    alignment : list
        A list of note alignment dictionaries.
    """
    alignment = list()
    with open(filename, "r") as f:
        for line in f.readlines():
            fields = line.split("\t")
            if fields[0][0] == "n" and "deletion" not in fields[1]:
                alignment.append(
                    {
                        "label": "match",
                        "score_id": fields[0],
                        "performance_id": fields[1],
                    }
                )
            elif fields[0] == "insertion":
                alignment.append({"label": "insertion", "performance_id": fields[1]})
            elif fields[0][0] == "n" and "deletion" in fields[1]:
                alignment.append({"label": "deletion", "score_id": fields[0]})

    return alignment
