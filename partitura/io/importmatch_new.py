#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains methods for parsing matchfiles
"""
import re
import os
from collections import namedtuple
from typing import Union, Tuple, Optional, Callable, List

import numpy as np

from partitura.score import Part, Score, ScoreLike
from partitura.performance import PerformedPart, Performance, PerformanceLike

from partitura.io.matchlines_v0 import (
    FROM_MATCHLINE_METHODS as FROM_MATCHLINE_METHODSV0,
    parse_matchline as parse_matchlinev0,
    MatchInfo as MatchInfoV0,
    MatchMeta as MatchMetaV0,
    MatchSnote as MatchSnoteV0,
    MatchNote as MatchNoteV0,
    MatchSnoteNote as MatchSnoteNoteV0,
    MatchSnoteDeletion as MatchSnoteDeletionV0,
    MatchSnoteTrailingScore as MatchSnoteTrailingScoreV0,
    MatchInsertionNote as MatchInsertionNoteV0,
    MatchHammerBounceNote as MatchHammerBounceNoteV0,
    MatchTrailingPlayedNote as MatchTrailingPlayedNoteV0,
    MatchSustainPedal as MatchSustainPedalV0,
    MatchSoftPedal as MatchSoftPedalV0,
    MatchTrillNote as MatchTrillNoteV0,
)

from partitura.io.matchlines_v1 import (
    FROM_MATCHLINE_METHODS as FROM_MATCHLINE_METHODSV1,
    MatchInfo as MatchInfoV1,
    MatchScoreProp as MatchScorePropV1,
    MatchSection as MatchSectionV1,
    MatchStime as MatchStimeV1,
    MatchPtime as MatchPtimeV1,
    MatchStimePtime as MatchStimePtimeV1,
    MatchSnote as MatchSnoteV1,
    MatchNote as MatchNoteV1,
    MatchSnoteNote as MatchSnoteNoteV1,
    MatchSnoteDeletion as MatchSnoteDeletionV1,
    MatchInsertionNote as MatchInsertionNoteV1,
    MatchSustainPedal as MatchSustainPedalV1,
    MatchSoftPedal as MatchSoftPedalV1,
    MatchOrnamentNote as MatchOrnamentNoteV1,
)

from partitura.io.matchfile_base import (
    MatchError,
    MatchFile,
    MatchLine,
    BaseSnoteNoteLine,
    BaseStimePtimeLine,
    BaseDeletionLine,
    BaseInsertionLine,
    BaseOrnamentLine,
    BaseSustainPedalLine,
    BaseSoftPedalLine,
)

from partitura.io.matchfile_utils import Version

from partitura.utils.misc import deprecated_alias, deprecated_parameter, PathLike

__all__ = ["load_matchfile"]


def get_version(line: str) -> Version:
    """
    Get version from the first line. Since the
    first version of the format did not include this line,
    we assume that the version is 0.1.0 if version is found.

    Parameters
    ----------
    line: str
        The first line of the match file.

    Returns
    -------
    version : Version
        The version of the match file
    """
    version = Version(0, 1, 0)

    for parser in (MatchInfoV1, MatchInfoV0):
        try:
            ml = parser.from_matchline(line)
            if isinstance(getattr(ml, "Value", None), Version):
                version = ml.Value
                return version

        except MatchError:

            pass

    return version


def parse_matchline(
    line: str,
    from_matchline_methods: List[Callable[[str], MatchLine]],
    version: Version,
    debug: bool = False,
) -> Optional[MatchLine]:
    """
    Return objects representing the line as one of:

    * hammer_bounce-PlayedNote.
    * info(Attribute, Value).
    * insertion-PlayedNote.
    * ornament(Anchor)-PlayedNote.
    * ScoreNote-deletion.
    * ScoreNote-PlayedNote.
    * ScoreNote-trailing_score_note.
    * trailing_played_note-PlayedNote.
    * trill(Anchor)-PlayedNote.
    * meta(Attribute, Value, Bar, Beat).
    * sustain(Time, Value)
    * soft(Time, Value)

    or False if none can be matched

    Parameters
    ----------
    line : str
        Line of the match file
    from_matchline_methods : List[Callable[[str], MatchLine]]

    Returns
    -------
    matchline : subclass of `MatchLine`
       Object representing the line.
    """

    matchline = None
    for from_matchline in from_matchline_methods:
        try:
            matchline = from_matchline(line, version=version)
            break
        except Exception as e:
            if not isinstance(e, MatchError):
                print(line, e, version)
            continue

    return matchline


@deprecated_alias(fn="filename", create_part="create_score")
def load_matchfile(
    filename: PathLike,
    create_score: bool = False,
    pedal_threshold: int = 64,
    first_note_at_zero: bool = False,
    debug: bool = True,
    n_processes: Optional[int] = 4,
) -> MatchFile:

    if not os.path.exists(filename):
        raise ValueError("Filename does not exist")

    with open(filename) as f:
        raw_lines = f.read().splitlines()

    version = get_version(raw_lines[0])

    from_matchline_methods = FROM_MATCHLINE_METHODSV1
    if version < Version(1, 0, 0):
        from_matchline_methods = FROM_MATCHLINE_METHODSV0

    parsed_lines = [
        parse_matchline(line, from_matchline_methods, version, debug)
        for line in raw_lines
    ]

    parsed_lines = [pl for pl in parsed_lines if pl is not None]

    if debug:
        print(filename, len(parsed_lines) == len(raw_lines))

    mf = MatchFile(lines=parsed_lines)

    return mf


def note_alignment_from_matchfile(mf: MatchFile) -> List[dict]:
    result = []

    for line in mf.lines:
        if isinstance(line, BaseSnoteNoteLine):
            result.append(
                dict(
                    label="match",
                    score_id=str(line.snote.Anchor),
                    performance_id=str(line.note.Id),
                )
            )

        elif isinstance(
            line,
            BaseDeletionLine,
        ):
            if "leftOutTied" in line.snote.ScoreAttributesList:
                continue
            else:
                result.append(dict(label="deletion", score_id=str(line.snote.Anchor)))
        elif isinstance(
            line,
            BaseInsertionLine,
        ):
            result.append(dict(label="insertion", performance_id=str(line.note.Id)))
        elif isinstance(line, BaseOrnamentLine):
            if isinstance(line, MatchTrillNoteV0):
                ornament_type = "trill"
            elif isinstance(line, MatchOrnamentNoteV1):
                ornament_type = line.OrnamentType
            else:
                ornament_type = "generic_ornament"
            result.append(
                dict(
                    label="ornament",
                    score_id=str(line.Anchor),
                    performance_id=str(line.note.Id),
                    type=ornament_type,
                )
            )

    return result


# alias
alignment_from_matchfile = note_alignment_from_matchfile


def time_alignment_from_matchfile(mf: MatchFile) -> List[dict]:

    for line in mf.lines:

        if isinstance(line, BaseStimePtimeLine):

            pass


def performed_part_from_match(
    mf: MatchFile,
    pedal_threshold: int = 64,
    first_note_at_zero: bool = False,
) -> PerformedPart:
    """
    Make PerformedPart from performance info in a MatchFile

    Parameters
    ----------
    mf : MatchFile
        A MatchFile instance
    pedal_threshold : int, optional
        Threshold for adjusting sound off of the performed notes using
        pedal information. Defaults to 64.
    first_note_at_zero : bool, optional
        When True the note_on and note_off times in the performance
        are shifted to make the first note_on time equal zero.

    Returns
    -------
    ppart : PerformedPart
        A performed part

    """
    # Get midi time units
    mpq = mf.info("midiClockRate")  # 500000 -> microseconds per quarter
    ppq = mf.info("midiClockUnits")  # 500 -> parts per quarter

    # PerformedNote instances for all MatchNotes
    notes = []

    first_note = next(mf.iter_notes(), None)
    if first_note and first_note_at_zero:
        offset = first_note.Onset * mpq / (10**6 * ppq)
    else:
        offset = 0

    notes = [
        dict(
            id=note.Id,
            midi_pitch=note.MidiPitch,
            note_on=note.Onset * mpq / (10**6 * ppq) - offset,
            note_off=note.Offset * mpq / (10**6 * ppq) - offset,
            sound_off=note.Offset * mpq / (10**6 * ppq) - offset,
            velocity=note.Velocity,
        )
        for note in mf.notes
    ]

    # SustainPedal instances for sustain pedal lines
    sustain_pedal = [
        dict(
            number=64,
            time=ped.Time * mpq / (10**6 * ppq),
            value=ped.Value,
        )
        for ped in mf.soft_pedal
    ]

    # SoftPedal instances for soft pedal lines
    soft_pedal = [
        dict(
            number=67,
            time=ped.Time * mpq / (10**6 * ppq),
            value=ped.Value,
        )
        for ped in mf.soft_pedal
    ]

    # Make performed part
    ppart = PerformedPart(
        id="P1",
        part_name=mf.info("piece"),
        notes=notes,
        controls=sustain_pedal + soft_pedal,
        sustain_pedal_threshold=pedal_threshold,
    )
    return ppart


if __name__ == "__main__":

    pass
