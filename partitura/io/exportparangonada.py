#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains methods for saving Parangonada csv files
"""

import os

import numpy as np

from typing import Union, List, Iterable, Tuple, Optional

from partitura.score import ScoreLike
from partitura.performance import PerformanceLike, Performance, PerformedPart

from partitura.utils import ensure_notearray

from partitura.utils.misc import PathLike, deprecated_alias

__all__ = [
    "save_parangonada_csv",
    "save_parangonada_alignment",
]


def alignment_dicts_to_array(alignment: List[dict]) -> np.ndarray:
    """
    create structured array from list of dicts type alignment.

    Parameters
    ----------
    alignment : list
        A list of note alignment dictionaries.

    Returns
    -------
    alignarray : structured ndarray
        Structured array containing note alignment.
    """
    fields = [
        ("idx", "i4"),
        ("matchtype", "U256"),
        ("partid", "U256"),
        ("ppartid", "U256"),
    ]

    array = []
    # for all dicts create an appropriate entry in an array:
    # match = 0, deletion  = 1, insertion = 2
    for no, i in enumerate(alignment):

        if i["label"] == "match":
            array.append((no, "0", i["score_id"], str(i["performance_id"])))
        elif i["label"] == "insertion":
            array.append((no, "2", "undefined", str(i["performance_id"])))
        elif i["label"] == "deletion":
            array.append((no, "1", i["score_id"], "undefined"))

    alignarray = np.array(array, dtype=fields)

    return alignarray


@deprecated_alias(
    spart="score_data",
    ppart="performance_data",
    align="alignment",
)
def save_parangonada_csv(
    alignment: List[dict],
    performance_data: Union[PerformanceLike, np.ndarray],
    score_data: Union[ScoreLike, np.ndarray],
    outdir: Optional[PathLike] = None,
    zalign: Optional[List[dict]] = None,
    feature: Optional[List[dict]] = None,
) -> Optional[Tuple[np.ndarray]]:
    """
    Save an alignment for visualization with parangonda.

    Parameters
    ----------
    alignment : list
        A list of note alignment dictionaries.
    performance_data : Performance, PerformedPart, structured ndarray
        The performance information
    score_data : ScoreLike
        The musical score. A :class:`partitura.score.Score` object,
        a :class:`partitura.score.Part`, a :class:`partitura.score.PartGroup` or
        a list of these.
    outdir : PathLike
        A directory to save the files into.
    ppart : PerformedPart, structured ndarray
        A PerformedPart or its note_array.
    zalign : list, optional
        A second list of note alignment dictionaries.
    feature : list, optional
        A list of expressive feature dictionaries.

    Returns
    -------
    perf_note_array : np.ndarray
        The performance note array. Only returned if `outdir` is None.
    score_note_array: np.ndarray
        The note array from the score. Only returned if `outdir` is None.
    alignarray: np.ndarray
    zalignarray: np.ndarray
    featurearray: np.ndarray
    """

    score_note_array = ensure_notearray(score_data)

    perf_note_array = ensure_notearray(performance_data)

    ffields = [
        ("velocity", "<f4"),
        ("timing", "<f4"),
        ("articulation", "<f4"),
        ("id", "U256"),
    ]

    farray = []
    notes = list(score_note_array["id"])
    if feature is not None:
        # veloctiy, timing, articulation, note
        for no, i in enumerate(list(feature["id"])):
            farray.append(
                (
                    feature["velocity"][no],
                    feature["timing"][no],
                    feature["articulation"][no],
                    i,
                )
            )
    else:
        for no, i in enumerate(notes):
            farray.append((0, 0, 0, i))

    featurearray = np.array(farray, dtype=ffields)
    alignarray = alignment_dicts_to_array(alignment)

    if zalign is not None:
        zalignarray = alignment_dicts_to_array(zalign)
    else:  # if no zalign is available, save the same alignment twice
        zalignarray = alignment_dicts_to_array(alignment)

    if outdir is not None:
        np.savetxt(
            os.path.join(outdir, "ppart.csv"),
            # outdir + os.path.sep + "perf_note_array.csv",
            perf_note_array,
            fmt="%.20s",
            delimiter=",",
            header=",".join(perf_note_array.dtype.names),
            comments="",
        )
        np.savetxt(
            os.path.join(outdir, "part.csv"),
            # outdir + os.path.sep + "score_note_array.csv",
            score_note_array,
            fmt="%.20s",
            delimiter=",",
            header=",".join(score_note_array.dtype.names),
            comments="",
        )
        np.savetxt(
            os.path.join(outdir, "align.csv"),
            # outdir + os.path.sep + "align.csv",
            alignarray,
            fmt="%.20s",
            delimiter=",",
            header=",".join(alignarray.dtype.names),
            comments="",
        )
        np.savetxt(
            os.path.join(outdir, "zalign.csv"),
            # outdir + os.path.sep + "zalign.csv",
            zalignarray,
            fmt="%.20s",
            delimiter=",",
            header=",".join(zalignarray.dtype.names),
            comments="",
        )
        np.savetxt(
            os.path.join(outdir, "feature.csv"),
            # outdir + os.path.sep + "feature.csv",
            featurearray,
            fmt="%.20s",
            delimiter=",",
            header=",".join(featurearray.dtype.names),
            comments="",
        )
    else:
        return (
            perf_note_array,
            score_note_array,
            alignarray,
            zalignarray,
            featurearray,
        )


# alias
save_csv_for_parangonada = save_parangonada_csv


@deprecated_alias(align="alignment", outfile="out")
def save_parangonada_alignment(
    alignment: List[dict],
    out: Optional[PathLike] = None,
):
    """
    Save only an alignment csv for visualization with parangonda.
    For score, performance, and expressive features use
    save_csv_for_parangonada()

    Parameters
    ----------
    align : list
        A list of note alignment dictionaries.

    outdir : str
        A directory to save the files into.


    Returns
    -------
    alignarray : np.ndarray
        Array containing the alignment. This array will only be returned if `out`
        is not None
    """
    alignarray = alignment_dicts_to_array(alignment)

    if out is not None:
        np.savetxt(
            out,
            alignarray,
            fmt="%.20s",
            delimiter=",",
            header=",".join(alignarray.dtype.names),
            comments="",
        )
    else:
        return alignarray


# alias
save_alignment_for_parangonada = save_parangonada_alignment


@deprecated_alias(outfile="out", ppart="performance_data")
def save_alignment_for_ASAP(
    alignment: List[dict],
    performance_data: PerformanceLike,
    out: PathLike,
) -> None:
    """
    load an alignment exported from parangonda.

    Parameters
    ----------
    alignment : list
        A list of note alignment dictionaries.
    performance_data : PerformanceLike
        A performance.
    out : str
        A path for the alignment tsv file.
    """
    if isinstance(performance_data, (Performance, Iterable)):
        ppart = performance_data[0]
    elif isinstance(performance_data, PerformedPart):
        ppart = performance_data
    notes_indexed_by_id = {
        str(n["id"]): [
            str(n["id"]),
            str(n["track"]),
            str(n["channel"]),
            str(n["midi_pitch"]),
            str(n["note_on"]),
        ]
        for n in ppart.notes
    }
    with open(out, "w") as f:
        f.write("xml_id\tmidi_id\ttrack\tchannel\tpitch\tonset\n")
        for line in alignment:
            if line["label"] == "match":
                outline_score = [str(line["score_id"])]
                outline_perf = notes_indexed_by_id[str(line["performance_id"])]
                f.write("\t".join(outline_score + outline_perf) + "\n")
            elif line["label"] == "deletion":
                outline_score = str(line["score_id"])
                f.write(outline_score + "\tdeletion\n")
            elif line["label"] == "insertion":
                outline_score = ["insertion"]
                outline_perf = notes_indexed_by_id[str(line["performance_id"])]
                f.write("\t".join(outline_score + outline_perf) + "\n")
