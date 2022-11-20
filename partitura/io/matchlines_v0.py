#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains definitions for Matchfile lines for version <1.0.0
"""
from __future__ import annotations

from collections import defaultdict

import re

from typing import Any, Callable, Tuple, Union, List

from partitura.io.matchfile_base import (
    MatchLine,
    BaseInfoLine,
    BaseSnoteLine,
    MatchError,
)

from partitura.io.matchfile_utils import (
    Version,
    interpret_version,
    format_version,
    interpret_as_string,
    format_string,
    interpret_as_float,
    format_float,
    format_float_unconstrained,
    interpret_as_int,
    format_int,
    FractionalSymbolicDuration,
    format_fractional,
    interpret_as_fractional,
    interpret_as_list,
    format_list,
)

# Define last supported version of the match file format in this module
# other modules might include different versions.
LAST_MAJOR_VERSION = 0
LAST_MINOR_VERSION = 5
LAST_PATCH_VERSION = 0

LAST_VERSION = Version(
    LAST_MAJOR_VERSION,
    LAST_MINOR_VERSION,
    LAST_PATCH_VERSION,
)


# Dictionary of interpreter, formatters and datatypes for info lines
# each entry in the dictionary is a tuple with
# an intepreter (to parse the input), a formatter (for the output matchline)
# and type


default_infoline_attributes = {
    "matchFileVersion": (interpret_version, format_version, Version),
    "piece": (interpret_as_string, format_string, str),
    "scoreFileName": (interpret_as_string, format_string, str),
    "scoreFilePath": (interpret_as_string, format_string, str),
    "midiFileName": (interpret_as_string, format_string, str),
    "midiFilePath": (interpret_as_string, format_string, str),
    "audioFileName": (interpret_as_string, format_string, str),
    "audioFilePath": (interpret_as_string, format_string, str),
    "audioFirstNote": (interpret_as_float, format_float_unconstrained, float),
    "audioLastNote": (interpret_as_float, format_float_unconstrained, float),
    "performer": (interpret_as_string, format_string, str),
    "composer": (interpret_as_string, format_string, str),
    "midiClockUnits": (interpret_as_int, format_int, int),
    "midiClockRate": (interpret_as_int, format_int, int),
    "approximateTempo": (interpret_as_float, format_float_unconstrained, float),
    "subtitle": (interpret_as_string, format_string, str),
    "keySignature": (interpret_as_list, format_list, list),
    "timeSignature": (
        interpret_as_fractional,
        format_fractional,
        FractionalSymbolicDuration,
    ),
    "tempoIndication": (interpret_as_list, format_list, list),
    "beatSubDivision": (interpret_as_list, format_list, list),
}

INFO_LINE = defaultdict(lambda: default_infoline_attributes.copy())


class MatchInfo(BaseInfoLine):
    """
    Main class specifying global information lines.

    For version 0.x.0, these lines have the general structure:

    `info(attribute,value).`

    Parameters
    ----------
    version : Version
        The version of the info line.
    kwargs : keyword arguments
        Keyword arguments specifying the type of line and its value.
    """

    def __init__(
        self,
        version: Version,
        attribute: str,
        value: Any,
        value_type: type,
        format_fun: Callable[Any, str],
    ) -> None:

        if version >= Version(1, 0, 0):
            raise MatchError("The version must be < 1.0.0")

        super().__init__(
            version=version,
            attribute=attribute,
            value=value,
            value_type=value_type,
            format_fun=format_fun,
        )

    @classmethod
    def from_matchline(
        cls,
        matchline: str,
        pos: int = 0,
        version: Version = LAST_VERSION,
    ) -> MatchInfo:
        """
        Create a new MatchLine object from a string

        Parameters
        ----------
        matchline : str
            String with a matchline
        pos : int (optional)
            Position of the matchline in the input string. By default it is
            assumed that the matchline starts at the beginning of the input
            string.
        version : Version (optional)
            Version of the matchline. By default it is the latest version.

        Returns
        -------
        a MatchInfo instance
        """

        if version >= Version(1, 0, 0):
            raise MatchError("The version must be < 1.0.0")

        match_pattern = cls.pattern.search(matchline, pos=pos)

        class_dict = INFO_LINE[version]

        if match_pattern is not None:
            attribute, value_str = match_pattern.groups()
            if attribute not in class_dict:
                raise ValueError(f"Attribute {attribute} is not specified in {version}")

            interpret_fun, format_fun, value_type = class_dict[attribute]

            value = interpret_fun(value_str)

            return cls(
                version=version,
                attribute=attribute,
                value=value,
                value_type=value_type,
                format_fun=format_fun,
            )

        else:
            raise MatchError("Input match line does not fit the expected pattern.")


class MatchSnote(BaseSnoteLine):
    def __init__(
        self,
        version: Version,
        anchor: str,
        note_name: str,
        modifier: str,
        octave: Union[int, str],
        measure: int,
        beat: int,
        offset: FractionalSymbolicDuration,
        duration: FractionalSymbolicDuration,
        onset_in_beats: float,
        offset_in_beats: float,
        score_attributes_list: List[str],
    ):
        super().__init__(
            version=version,
            anchor=anchor,
            note_name=note_name,
            modifier=modifier,
            octave=octave,
            measure=measure,
            beat=beat,
            offset=offset,
            duration=duration,
            onset_in_beats=onset_in_beats,
            offset_in_beats=offset_in_beats,
            score_attributes_list=score_attributes_list,
        )

    @classmethod
    def from_matchline(
        cls,
        matchline: str,
        pos: int = 0,
        version: Version = LAST_VERSION,
    ) -> MatchSnote:
        """
        Create a new MatchLine object from a string

        Parameters
        ----------
        matchline : str
            String with a matchline
        pos : int (optional)
            Position of the matchline in the input string. By default it is
            assumed that the matchline starts at the beginning of the input
            string.
        version : Version (optional)
            Version of the matchline. By default it is the latest version.

        Returns
        -------
        a MatchSnote object
        """

        if version >= Version(1, 0, 0):
            raise ValueError(f"{version} > Version(1, 0, 0)")

        kwargs = cls.prepare_kwargs_from_matchline(
            matchline=matchline,
            pos=pos,
        )

        return cls(version=version, **kwargs)
