import os

import numpy as np

from typing import Union, List, Iterable, Tuple, Optional

from partitura.score import ScoreLike, Score
from partitura.performance import PerformanceLike, Performance

from partitura.utils import ensure_notearray

from partitura.utils.misc import PathLike, deprecated_alias


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
def save_csv_for_parangonada(
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
    """

    if isinstance(score_data, (Score, Iterable)):
        # Only use the first score_note_array if the score
        # has more than one score_note_array
        score_note_array = ensure_notearray(score_data[0])
    else:
        score_note_array = ensure_notearray(score_note_array)

    if isinstance(performance_data, (Performance, Iterable)):
        # Only use the first performed score_note_array if
        # the performance has more than one score_note_array
        perf_note_array = ensure_notearray(performance_data[0])
    else:
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
            os.path.join(outdir, "perf_note_array.csv"),
            # outdir + os.path.sep + "perf_note_array.csv",
            perf_note_array,
            fmt="%.20s",
            delimiter=",",
            header=",".join(perf_note_array.dtype.names),
            comments="",
        )
        np.savetxt(
            os.path.join(outdir, "score_note_array.csv"),
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


def save_alignment_for_parangonada(outfile, align):
    """
    Save only an alignment csv for visualization with parangonda.
    For score, performance, and expressive features use
    save_csv_for_parangonada()

    Parameters
    ----------
    outdir : str
        A directory to save the files into.
    align : list
        A list of note alignment dictionaries.

    """
    alignarray = alignment_dicts_to_array(align)

    np.savetxt(
        outfile,
        alignarray,
        fmt="%.20s",
        delimiter=",",
        header=",".join(alignarray.dtype.names),
        comments="",
    )


def load_alignment_from_parangonada(outfile):
    """
    load an alignment exported from parangonda.

    Parameters
    ----------
    outfile : str
        A path to the alignment csv file

    Returns
    -------
    alignlist : list
        A list of note alignment dictionaries.
    """
    array = np.loadtxt(outfile, dtype=str, delimiter=",")
    alignlist = list()
    # match = 0, deletion  = 1, insertion = 2
    for k in range(1, array.shape[0]):
        if int(array[k, 1]) == 0:
            alignlist.append(
                {
                    "label": "match",
                    "score_id": array[k, 2],
                    "performance_id": array[k, 3],
                }
            )

        elif int(array[k, 1]) == 2:
            alignlist.append({"label": "insertion", "performance_id": array[k, 3]})

        elif int(array[k, 1]) == 1:
            alignlist.append({"label": "deletion", "score_id": array[k, 2]})
    return alignlist


def save_alignment_for_ASAP(outfile, ppart, alignment):
    """
    load an alignment exported from parangonda.

    Parameters
    ----------
    outfile : str
        A path for the alignment tsv file.
    ppart : PerformedPart, structured ndarray
        A PerformedPart or its note_array.
    align : list
        A list of note alignment dictionaries.

    """
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
    with open(outfile, "w") as f:
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
