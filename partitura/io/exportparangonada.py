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


@deprecated_alias(spart="score_data", ppart="performance_data")
def save_csv_for_parangonada(
    outdir: PathLike,
    score_data: Union[ScoreLike, np.ndarray],
    performance_data: Union[PerformanceLike, np.ndarray],
    align: List[dict],
    zalign: Optional[List[dict]] = None,
    feature: Optional[List[dict]] = None,
) -> Optional[Tuple[np.ndarray]]:
    """
    Save an alignment for visualization with parangonda.

    Parameters
    ----------
    outdir : str
        A directory to save the files into.
    part : Part, structured ndarray
        A score part or its note_array.
    ppart : PerformedPart, structured ndarray
        A PerformedPart or its note_array.
    align : list
        A list of note alignment dictionaries.
    zalign : list, optional
        A second list of note alignment dictionaries.
    feature : list, optional
        A list of expressive feature dictionaries.

    """

    if isinstance(score_data, (Score, Iterable)):
        # Only use the first part if the score has more than one part
        part = ensure_notearray(score_data[0])
    else:
        part = ensure_notearray(part)

    if isinstance(performance_data, (Performance, Iterable)):
        # Only use the first performed part if the performance has more
        # than one part
        ppart = ensure_notearray(performance_data[0])
    else:
        ppart = ensure_notearray(performance_data)

    ffields = [
        ("velocity", "<f4"),
        ("timing", "<f4"),
        ("articulation", "<f4"),
        ("id", "U256"),
    ]

    farray = []
    notes = list(part["id"])
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
    alignarray = alignment_dicts_to_array(align)

    if zalign is not None:
        zalignarray = alignment_dicts_to_array(zalign)
    else:  # if no zalign is available, save the same alignment twice
        zalignarray = alignment_dicts_to_array(align)

    if outdir is not None:
        np.savetxt(
            os.path.join(outdir, "ppart.csv"),
            # outdir + os.path.sep + "ppart.csv",
            ppart,
            fmt="%.20s",
            delimiter=",",
            header=",".join(ppart.dtype.names),
            comments="",
        )
        np.savetxt(
            os.path.join(outdir, "part.csv"),
            # outdir + os.path.sep + "part.csv",
            part,
            fmt="%.20s",
            delimiter=",",
            header=",".join(part.dtype.names),
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
        return (ppart, part, alignarray, zalignarray, featurearray)


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


def load_alignment_from_ASAP(outfile):
    """
    load a note alignment of the ASAP dataset.

    Parameters
    ----------
    outfile : str
        A path to the alignment tsv file

    Returns
    -------
    alignlist : list
        A list of note alignment dictionaries.
    """
    alignlist = list()
    with open(outfile, "r") as f:
        for line in f.readlines():
            fields = line.split("\t")
            if fields[0][0] == "n" and "deletion" not in fields[1]:
                alignlist.append(
                    {
                        "label": "match",
                        "score_id": fields[0],
                        "performance_id": fields[1],
                    }
                )
            elif fields[0] == "insertion":
                alignlist.append({"label": "insertion", "performance_id": fields[1]})
            elif fields[0][0] == "n" and "deletion" in fields[1]:
                alignlist.append({"label": "deletion", "score_id": fields[0]})

    return alignlist
