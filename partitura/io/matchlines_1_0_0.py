#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains definitions for Matchfile lines for version >1.0.0
"""
from __future__ import annotations

from typing import Any, Callable

from partitura.io.matchfile_base import (
    MatchLine,
    Version,
    BaseInfoLine,
    MatchError,
    interpret_version,
    format_version,
    interpret_as_string,
    format_string,
    interpret_as_float,
    format_float,
    interpret_as_int,
    format_int,
)

# Define current version of the match file format
CURRENT_MAJOR_VERSION = 1
CURRENT_MINOR_VERSION = 0
CURRENT_PATCH_VERSION = 0

CURRENT_VERSION = Version(
    CURRENT_MAJOR_VERSION,
    CURRENT_MINOR_VERSION,
    CURRENT_PATCH_VERSION,
)


# Dictionary of interpreter, formatters and datatypes for info lines
# each entry in the dictionary is a tuple with
# an intepreter (to parse the input), a formatter (for the output matchline)
# and type

INFO_LINE = {
    Version(1, 0, 0): {
        "matchFileVersion": (interpret_version, format_version, Version),
        "piece": (interpret_as_string, format_string, str),
        "scoreFileName": (interpret_as_string, format_string, str),
        "scoreFilePath": (interpret_as_string, format_string, str),
        "midiFileName": (interpret_as_string, format_string, str),
        "midiFilePath": (interpret_as_string, format_string, str),
        "audioFileName": (interpret_as_string, format_string, str),
        "audioFilePath": (interpret_as_string, format_string, str),
        "audioFirstNote": (interpret_as_float, format_float, float),
        "audioLastNote": (interpret_as_float, format_float, float),
        "performer": (interpret_as_string, format_string, str),
        "composer": (interpret_as_string, format_string, str),
        "midiClockUnits": (interpret_as_int, format_int, int),
        "midiClockRate": (interpret_as_int, format_int, int),
        "approximateTempo": (interpret_as_float, format_float, float),
        "subtitle": (interpret_as_string, format_string, str),
    }
}


class MatchInfo(BaseInfoLine):
    """
    Main class specifying global information lines.

    For version 1.0.0, these lines have the general structure:

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

        if version < Version(1, 0, 0):
            raise MatchError("The version must be >= 1.0.0")

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
        version: Version = CURRENT_VERSION,
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
        a Mat instance
        """

        match_pattern = cls.pattern.search(matchline, pos=pos)

        if version not in INFO_LINE:
            raise MatchError(f"{version} is not specified for this class.")
        class_dict = INFO_LINE[version]

        if match_pattern is not None:
            attribute, value_str = match_pattern.groups()
            if attribute not in class_dict:
                raise ValueError(
                    f"Attribute {attribute} is not specified in {version}"
                )

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
