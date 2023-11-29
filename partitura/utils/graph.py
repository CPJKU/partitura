import numpy as np
from partitura.score import ScoreLike


def compute_edge_list_from_note_array(note_array, heterogeneous=True):
    """
    Compute edge list from note array.

    Parameters:
    -----------
    note_array: structured array
        A structured array with fields 'onset_div', 'duration_div', 'pitch'
    heterogeneous: bool
        If True, it returns a dict with the edge types. If False, it returns a 2xN array with the edge list.
    Returns:
    --------
    edges: np.ndarray
        A 2xN array with the edge list. When heterogeneous is True, it returns a dict with the edge types.
    """

    types_to_num = {"onset": 0, "consecutive": 1, "during": 2, "silence": 3}
    num_to_types = {v: k for k, v in types_to_num.items()}
    edg_src = list()
    edg_dst = list()
    edg_type = list()
    for i, x in enumerate(note_array):
        for j in np.where((note_array["onset_div"] == x["onset_div"]) & (note_array["id"] != x["id"]))[0]:
            edg_src.append(i)
            edg_dst.append(j)
            edg_type.append(0)

        for j in np.where(note_array["onset_div"] == x["onset_div"] + x["duration_div"])[0]:
            edg_src.append(i)
            edg_dst.append(j)
            edg_type.append(1)

        for j in np.where((x["onset_div"] < note_array["onset_div"]) & (
                x["onset_div"] + x["duration_div"] > note_array["onset_div"]))[0]:
            edg_src.append(i)
            edg_dst.append(j)
            edg_type.append(2)

    end_times = note_array["onset_div"] + note_array["duration_div"]
    for et in np.sort(np.unique(end_times))[:-1]:
        if et not in note_array["onset_div"]:
            scr = np.where(end_times == et)[0]
            diffs = note_array["onset_div"] - et
            tmp = np.where(diffs > 0, diffs, np.inf)
            dst = np.where(tmp == tmp.min())[0]
            for i in scr:
                for j in dst:
                    edg_src.append(i)
                    edg_dst.append(j)
                    edg_type.append(3)

    e = np.array([edg_src, edg_dst, edg_type])
    if heterogeneous:
        edges = {num_to_types[i]: e[:, e[2] == i][:2] for i in np.unique(e[2])}
    else:
        edges = e[:2]

    return edges


def compute_edge_list(score: ScoreLike, heterogeneous=True):
    """
    Compute edge list from score.
    Please note that indices of the nodes correspond to the indices of the note array of the score.

    Parameters
    ----------
    score: ScoreLike
        A ScoreLike object (e.g. Score, Part, PartGroup, etc.)
    heterogeneous: bool
        If True, it returns a dict with the edge types. If False, it returns a 2xN array with the edge list.

    Returns
    -------

    """
    edges = compute_edge_list_from_note_array(score.note_array(), heterogeneous=heterogeneous)
    return edges