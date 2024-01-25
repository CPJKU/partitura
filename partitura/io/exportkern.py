#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains methods for exporting Kern files.
"""
import math
from collections import defaultdict
import partitura.score as score
from operator import itemgetter
from typing import Optional
import numpy as np
import warnings
from partitura.utils import partition, iter_current_next, to_quarter_tempo
from partitura.utils.misc import deprecated_alias, PathLike

__all__ = ["save_kern"]


def sym_dur_to_kern(symbolic_duration: dict) -> str:
    return ""

def pitch_to_kern(element: score.GenericNote) -> str:
    step, alter, octave = element.step, element.alter, element.octave
    return ""

def markings_to_kern(element: score.GenericNote) -> str:
     return ""

def save_kern(
    score_data: score.ScoreLike,
    out: Optional[PathLike] = None,
    ) -> Optional[str]:
    # Header extracts meta information about the score
    header = score_data.composer
    # Kern can output only from part so first let's merge parts (we need a timewise representation)
    if isinstance(score_data, score.Score):
        # TODO check that divisions are the same
        part = score.merge_parts(score.partlist)
    else:
        part = score


    note_array = part.note_array(include_staff=True)
    unique_voc_staff = np.unique(note_array[["voice", "staff"]], axis=0)
    vocstaff_map_dict = {f"{unique_voc_staff[i][0]}-{unique_voc_staff[i][1]}": i for i in range(unique_voc_staff.shape[0])}
    part_elements = list(part.iter_all())
    out_data = np.empty((len(part_elements) + 1, len(unique_voc_staff)), dtype=object)

    # Fill all values with the "." character to filter afterwards
    out_data.fill(".")
    out_data[0] = "**kern"
    prev_note_time = None
    prev_note_col_idx = None
    prev_note_row_idx = None
    for row_idx, el in enumerate(part_elements, start=1):
        if isinstance(el, score.GenericNote):
            voice = el.voice
            staff = el.staff
            duration = sym_dur_to_kern(el.symbolic_duration)
            pitch = pitch_to_kern(el)
            col_idx = vocstaff_map_dict[f"{voice}-{staff}"]
            markings = markings_to_kern(el)
            kern_el = duration + pitch + markings
            if prev_note_time == el.start.t:
                if prev_note_col_idx == col_idx:
                    # Chords in Kern
                    out_data[prev_note_row_idx, prev_note_col_idx] = out_data[prev_note_row_idx, prev_note_col_idx] + " " + kern_el
                else:
                    # Same row (start.t) other spline
                    out_data[prev_note_row_idx, col_idx] = kern_el
            else:
                # New line
                out_data[row_idx, col_idx] = kern_el
        elif isinstance(el, score.Measure):
            # Apply measure to all splines
            kern_el = f"={el.number}"
            out_data[row_idx] = kern_el
        elif isinstance(el, score.TimeSignature):
            # Apply element to all splines
            kern_el = f"*M{el.beats}/{el.beat_type}"
            out_data[row_idx] = kern_el
        elif isinstance(el, score.KeySignature):
            # Apply element to all splines
            alters = ""
            kern_el = f"*{alters}"
            out_data[row_idx] = kern_el
        else:
            warnings.warn(f"Element {el} is supported for kern export yet.")

    # if an entire row is filled with "." elements remove it.
    out_data = out_data[~np.all(out_data == ".", axis=1)]

    # Use numpy savetxt to save the file
    footer = "Encoded using the Partitura Python package, version 1.5.0"
    np.savetxt(out, out_data, fmt="utf-8", delimiter="\t", newline="\n", header=header, footer=footer, comments="!!!")


