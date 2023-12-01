#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains methods for importing Humdrum Kern files.
"""
import re
import warnings

from typing import Union, Optional

import numpy as np

import partitura.score as spt
from partitura.utils import PathLike


SIGN_TO_ACC = {
    "n": 0,
    "#": 1,
    "s": 1,
    "ss": 2,
    "x": 2,
    "##": 2,
    "###": 3,
    "b": -1,
    "f": -1,
    "bb": -2,
    "ff": -2,
    "bbb": -3,
    "-": None,
}

KERN_NOTES = {
    "C": ("C", 3),
    "D": ("D", 3),
    "E": ("E", 3),
    "F": ("F", 3),
    "G": ("G", 3),
    "A": ("A", 3),
    "B": ("B", 3),
    "c": ("C", 4),
    "d": ("D", 4),
    "e": ("E", 4),
    "f": ("F", 4),
    "g": ("G", 4),
    "a": ("A", 4),
    "b": ("B", 4),
}

KERN_DURS = {
    "000": "maxima",
    "00": "long",
    "0": "breve",
    "1": "whole",
    "2": "half",
    "4": "quarter",
    "8": "eighth",
    "16": "16th",
    "32": "32nd",
    "64": "64th",
    "128": "128th",
    "256": "256th",
}


def _handle_kern_with_spine_splitting(kern_path):
    file = np.loadtxt(kern_path, dtype=str, delimiter="\n", comments="!", encoding="utf-8")
    # Get Main Number of parts and Spline Types
    spline_types = file[0].split("\t")
    # Decide Parts

    # Find all expansions points
    expansion_indices = np.where(np.char.find(file, "*^") != -1)[0]
    # For all expansion points find which stream is being expanded
    expansion_streams_per_index = [np.argwhere(np.array(line.split("\t")) == "*^")[0] for line in
                                   file[expansion_indices]]

    # Find all Spline Reduction points
    reduction_indices = np.where(np.char.find(file, "*v\t*v") != -1)[0]
    # For all reduction points find which stream is being reduced
    reduction_streams_per_index = [
        np.argwhere(np.char.add(np.array(line.split("\t")[:-1]), np.array(line.split("\t")[1:])) == "*v*v")[0] for line
        in file[reduction_indices]]

    # Find all pairs of expansion and reduction points
    expansion_reduction_pairs = []
    last_exhaustive_reduction = 0
    for expansion_index in expansion_indices:
        for expansion_stream in expansion_index:
            # Find the first reduction index that is after the expansion index and has the same index.
            for i, reduction_index in enumerate(reduction_indices[last_exhaustive_reduction:]):
                for reduction_stream in reduction_streams_per_index[i]:
                    if expansion_stream == reduction_stream:
                        expansion_reduction_pairs.append((expansion_index, reduction_index))
                        last_exhaustive_reduction = i if i == last_exhaustive_reduction + 1 else last_exhaustive_reduction
                        break


# functions to initialize the kern parser
def parse_kern(kern_path: PathLike, num_workers=0) -> np.ndarray:
    """
    Parses an KERN file from path to Part.

    Parameters
    ----------
    kern_path : PathLike
        The path of the KERN document.
    Returns
    -------
    continuous_parts : numpy character array
    non_continuous_parts : list
    """
    try:
        # This version of the parser is faster but does not support spine splitting.
        file = np.loadtxt(kern_path, dtype=str, delimiter="\t", comments="!", encoding="utf-8")
    except ValueError:
        # This version of the parser supports spine splitting but is slower.
        # It adds the splines to with a special character.
        file = _handle_kern_with_spine_splitting(kern_path)

    # Get Main Number of parts and Spline Types
    spline_types = file[0]
    # Decide Parts
    part_number, staff, voice = 0, 0, 0
    # Get Splines
    splines = file.T
    for spline in splines:
       parser = SplineParser(spline, part_number, staff, voice)
       elements = parser.parse()


class SplineParser(object):
    def init(self, spline, part_number, staff, voice):
        self.spline = spline
        self.part_number = part_number
        self.staff = staff
        self.voice = voice

    def parse(self):
        spline = self.spline
        # Remove "-" lines
        spline = spline[spline != "-"]
        # Remove "." lines
        spline = spline[spline != "."]
        # Find Global indices, i.e. where spline cells start with "*"
        global_mask = np.char.find(spline, "*") != -1
        # Remove the global indices from the spline
        spline = spline[~global_mask]

        # Empty Numpy array with objects
        elements = np.empty(len(spline), dtype=object)

        # Find Barline indices, i.e. where spline cells start with "="
        bar_mask = np.char.find(spline, "=") != -1
        # Find Chord indices, i.e. where spline cells contain " "
        chord_mask = np.char.find(spline, " ") != -1

        # All the rest are note indices
        note_mask = np.logical_and(~bar_mask, ~chord_mask)

        elements[note_mask] = np.vectorize(self.meta_note_line, otypes=[object])(spline[note_mask])
        elements[chord_mask] = np.vectorize(self.meta_chord_line, otypes=[object])(spline[chord_mask])
        elements[bar_mask] = np.vectorize(self.meta_barline_line, otypes=[object])(spline[bar_mask])

        return elements

    def _process_kern_pitch(self, pitch):
        # find accidentals
        alter = re.search(r"([n#\-]+)", pitch)
        # remove alter from pitch
        pitch = pitch.replace(alter.group(0), "") if alter else pitch
        step, octave = KERN_NOTES[pitch[0]]
        if octave == 4:
            octave = octave + pitch.count(pitch[0]) - 1
        elif octave == 3:
            octave = octave - pitch.count(pitch[0]) + 1
        alter = SIGN_TO_ACC[alter.group(0)] if alter else None
        return step, octave, alter

    def _process_kern_duration(self, duration):
        dur = duration.replace(".", "")
        if dur in KERN_DURS.keys():
            symbolic_duration = {"type": KERN_DURS[dur]}
        else:
            diff = dict(
                (
                    map(
                        lambda x: (dur - x, x) if dur > x else (dur + x, x),
                        KERN_DURS.keys(),
                    )
                )
            )
            symbolic_duration = {
                "type": KERN_DURS[diff[min(list(diff.keys()))]],
                "actual_notes": dur / 4,
                "normal_notes": diff[min(list(diff.keys()))] / 4,
            }
        symbolic_duration["dots"] = duration.count(".")
        return symbolic_duration

    def meta_note_line(self, line):
        """
        Grammar Defining a note line.

        A note line is specified by the following grammar:
        note_line = symbol | duration | pitch | symbol

        Parameters
        ----------
        line

        Returns
        -------

        """
        # extract first occurence of one of the following: a-g A-G r # - n
        pitch = re.search(r"([a-gA-Gr\-n#]+)", line).group(0)
        # extract duration can be any of the following: 0-9 .
        duration = re.search(r"([0-9]+|\.)", line).group(0)
        # extract symbol can be any of the following: _()[]{}<>|:
        symbol = re.findall(r"([_()[]{}<>|:])", line)
        symbolic_duration = self._process_kern_duration(duration)
        if pitch == "r":
            return spt.Rest(symbolic_duration=symbolic_duration)
        step, octave, alter = self._process_kern_pitch(pitch)
        return spt.Note(step, octave, alter, symbolic_duration=symbolic_duration)

    def meta_barline_line(self, line):
        """
        Grammar Defining a barline line.

        A barline line is specified by the following grammar:
        barline_line = repeat | barline | number | repeat

        Parameters
        ----------
        line

        Returns
        -------

        """
        # find number and keep its index.
        number = re.findall(r"([0-9]+)", line)
        number_index = line.index(number[0]) if number else line.index("=")
        closing_repeat = re.findall(r"[:|]", line[:number_index])
        opening_repeat = re.findall(r"[:|]", line[number_index:])
        return spt.BarLine()

    def meta_chord_line(self, line):
        """
        Grammar Defining a chord line.

        A chord line is specified by the following grammar:
        chord_line = note | chord

        Parameters
        ----------
        line

        Returns
        -------

        """
        return ("c", [self.meta_note_line(n) for n in line.split(" ")])


if __name__ == "__main__":
    kern_path = "/home/manos/Desktop/JKU/data/wtc-fugues/wtc1f02.krn"
    x = parse_kern(kern_path)