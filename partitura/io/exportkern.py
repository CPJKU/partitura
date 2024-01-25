#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains methods for exporting Kern files.
"""
import math
from collections import defaultdict
import partitura.score as spt
from operator import itemgetter
from typing import Optional
import numpy as np
import warnings
from partitura.utils import partition, iter_current_next, to_quarter_tempo
from partitura.utils.misc import deprecated_alias, PathLike

__all__ = ["save_kern"]


ACC_TO_SIGN = {
    0: "n",
    -1: "-",
    1: "#",
    -2: "--",
    2: "##",
}

KERN_NOTES = {
    ('C', 3): 'C',
    ('D', 3): 'D',
    ('E', 3): 'E',
    ('F', 3): 'F',
    ('G', 3): 'G',
    ('A', 3): 'A',
    ('B', 3): 'B',
    ('C', 4): 'c',
    ('D', 4): 'd',
    ('E', 4): 'e',
    ('F', 4): 'f',
    ('G', 4): 'g',
    ('A', 4): 'a',
    ('B', 4): 'b'
}

KERN_DURS = {
    'maxima': '000',
    'long': '00',
    'breve': '0',
    'whole': '1',
    'half': '2',
    'quarter': '4',
    'eighth': '8',
    '16th': '16',
    '32nd': '32',
    '64th': '64',
    '128th': '128',
    '256th': '256'
     }


def sym_dur_to_kern(symbolic_duration: dict) -> str:
    kern_base = KERN_DURS[symbolic_duration["type"]]
    dots = "."*symbolic_duration["dots"] if "dots" in symbolic_duration.keys() else ""
    if "actual_notes" in symbolic_duration.keys() and "normal_notes":
        kern_base = int(kern_base) * symbolic_duration["actual_notes"] / symbolic_duration["normal_notes"]
        kern_base = str(kern_base)
    return kern_base + dots

def duration_to_kern(element: spt.GenericNote) -> str:
    if isinstance(element, spt.GraceNote):
        if element.grace_type == "acciaccatura":
            return "p"
        else:
            return "q"
    else:
        return sym_dur_to_kern(element.symbolic_duration)

def pitch_to_kern(element: spt.GenericNote) -> str:
    if isinstance(element, spt.Rest):
        return "r"
    step, alter, octave = element.step, element.alter, element.octave
    if octave > 4:
        octave = 4
        multiply_character = octave - 3
    elif octave < 3:
        octave = 3
        multiply_character = 4 - octave
    else:
        multiply_character = 1
    kern_step = KERN_NOTES[(step, octave)] * multiply_character
    kern_alter = ACC_TO_SIGN[alter] if alter is not None else ""
    return kern_step + kern_alter


def markings_to_kern(element: spt.GenericNote) -> str:
    symbols = ""
    if not isinstance(element, spt.Rest):
        if element.tie_next and element.tie_prev:
            symbols += "-"
        elif element.tie_next:
            symbols += "["
        elif element.tie_prev:
            symbols += "]"
        if element.slur_starts:
            symbols += "("
        if element.slur_stops:
            symbols += ")"
    return symbols


def save_kern(
    score_data: spt.ScoreLike,
    out: Optional[PathLike] = None,
    ) -> Optional[str]:
    # Header extracts meta information about the score
    header = "Here is some random piece"
    # Kern can output only from part so first let's merge parts (we need a timewise representation)
    if isinstance(score_data, spt.Score):
        # TODO check that divisions are the same
        part = spt.merge_parts(score_data.parts)
    else:
        part = score_data

    note_array = part.note_array(include_staff=True)
    unique_voc_staff = np.unique(note_array[["voice", "staff"]], axis=0)
    vocstaff_map_dict = {f"{unique_voc_staff[i][0]}-{unique_voc_staff[i][1]}": i for i in range(unique_voc_staff.shape[0])}
    part_elements = np.array(list(part.iter_all()))
    # Part elements is really the maximum number of lines we could have in the kern file
    # we add two to account for the **kern and the *- encoding at beginning and end of file
    out_data = np.empty((len(part_elements) + 2, len(unique_voc_staff)), dtype=object)
    part_el_start_times = np.array([el.start.t for el in part_elements])
    unique_start_times, indices = np.unique(part_el_start_times, return_inverse=True)
    # Fill all values with the "." character to filter afterwards
    out_data.fill(".")
    out_data[0] = "**kern"
    out_data[-1] = "*-"
    prev_note_time = None
    prev_note_col_idx = None
    prev_note_row_idx = None
    row_idx = 1
    for unique_start_time_idx in range(len(unique_start_times)):
        # Get all elements starting at this time
        elements_starting = part_elements[indices == unique_start_time_idx]
        # Find notes
        note_mask = np.array([isinstance(el, spt.GenericNote) for el in elements_starting])
        if np.any(~note_mask):
            bar_mask = np.array([isinstance(el, spt.Measure) for el in elements_starting[~note_mask]])
            tandem_mask = ~bar_mask
            structural_elements = elements_starting[~note_mask]
            structural_elements = np.hstack((structural_elements[tandem_mask], structural_elements[bar_mask]))
        else:
            structural_elements = elements_starting[~note_mask]
        # Put structural elements first (start with tandem elements, then measure elements, then notes and rests)
        elements_starting = np.hstack((structural_elements, elements_starting[note_mask]))
        for el in elements_starting:
            if isinstance(el, spt.GenericNote):
                voice = el.voice
                staff = el.staff
                duration = duration_to_kern(el)
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
                    prev_note_row_idx = row_idx
                prev_note_col_idx = col_idx
                prev_note_time = el.start.t
            elif isinstance(el, spt.Measure):
                # Apply measure to all splines
                kern_el = f"={el.number}"
                out_data[row_idx] = kern_el
            elif isinstance(el, spt.TimeSignature):
                # Apply element to all splines
                kern_el = f"*M{el.beats}/{el.beat_type}"
                out_data[row_idx] = kern_el
            elif isinstance(el, spt.KeySignature):
                # Apply element to all splines
                alters = ""
                kern_el = f"*{alters}"
                out_data[row_idx] = kern_el
            else:
                warnings.warn(f"Element {el} is supported for kern export yet.")
            row_idx += 1
    # if an entire row is filled with "." elements remove it.
    out_data = out_data[~np.all(out_data == ".", axis=1)]
    # Use numpy savetxt to save the file
    footer = "Encoded using the Partitura Python package, version 1.5.0"
    np.savetxt(fname=out, X=out_data, fmt="%1.26s",
               delimiter="\t", newline="\n",
               header=header, footer=footer,
               comments="!!!", encoding="utf-8")


if __name__ == "__main__":
    import partitura as pt
    score = pt.load_score(pt.EXAMPLE_MUSICXML)
    save_kern(score, "C:/Users/melki/Desktop/test.krn")