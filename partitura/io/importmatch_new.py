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

from partitura.io.matchlines_v0 import (
    FROM_MATCHLINE_METHODS as FROM_MATCHLINE_METHODSV0,
    MatchInfo as MatchInfoV0,
)
from partitura.io.matchlines_v1 import (
    FROM_MATCHLINE_METHODS as FROM_MATCHLINE_METHODSV1,
    MatchInfo as MatchInfoV1,
)
from partitura.io.matchfile_base import MatchError, MatchFile, MatchLine
from partitura.io.matchfile_utils import Version

from partitura.utils.misc import deprecated_alias, deprecated_parameter, PathLike

__all__ = ["load_match"]


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
):

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

    mf = MatchFile(lines=parsed_lines)
    print(filename, len(parsed_lines) == len(raw_lines))
    pass


if __name__ == "__main__":

    pass
