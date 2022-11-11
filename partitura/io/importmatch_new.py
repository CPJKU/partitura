#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains methods for parsing matchfiles
"""
import re

from collections import namedtuple
from typing import Union, Tuple

# from packaging import version

import numpy as np
from partitura.io.matchfile_fields import FractionalSymbolicDuration

# Define current version of the match file format
CURRENT_MAJOR_VERSION = 1
CURRENT_MINOR_VERSION = 0
CURRENT_PATCH_VERSION = 0

Version = namedtuple("Version", ["major", "minor", "patch"])
VersionOld = namedtuple("Version", ["major", "minor"])


CURRENT_VERSION = Version(
    CURRENT_MAJOR_VERSION,
    CURRENT_MINOR_VERSION,
    CURRENT_PATCH_VERSION,
)

# General patterns
rational_pattern = re.compile(r"^([0-9]+)/([0-9]+)$")
double_rational_pattern = re.compile(r"^([0-9]+)/([0-9]+)/([0-9]+)$")
version_pattern = re.compile(r"^([0-9]+)\.([0-9]+)\.([0-9]+)")


class MatchError(Exception):
    pass


def interpret_version(version_string: str) -> Version:
    """
    Parse matchfile format version from a string. This method
    parses a string like "1.0.0" and returns a Version instance.

    Parameters
    ----------
    version_string : str
        The string containg the version. The version string should b
        in the form "{major}.{minor}.{patch}". Incorrectly formatted strings
        will result in an error.

    Returns
    -------
    version : Version
        A named tuple specifying the version
    """
    version_info = version_pattern.search(version_string)

    if version_info is not None:
        ma, mi, pa = version_info.groups()
        version = Version(int(ma), int(mi), int(pa))

        return version
    else:
        raise ValueError(f"The version '{version_string}' is incorrectly formatted!")


def format_version(version: Version) -> str:
    """
    Format version as a string.

    Parameters
    ----------
    version : Version
        A Version instance.

    Returns
    -------
    version_str : str
        A string representation of the version.
    """
    ma, mi, pa = version

    version_str = f"{ma}.{mi}.{pa}"
    return version_str


def interpret_as_int(value: str) -> int:
    """
    Interpret value as an integer

    Parameters
    ----------
    value : str
       The value to interpret as integer.

    Returns
    -------
    int
        The value cast as an integer.
    """
    return int(value)


def format_int(value: int) -> str:
    return f"{value}"


def interpret_as_float(value: str) -> float:
    return float(value)


def format_float(value: float) -> str:
    return f"{value:.4f}"


def interpret_as_string(value: str) -> str:
    return value


def format_string(value: str) -> str:
    """
    For completeness
    """
    return value.strip()


class MatchLine(object):
    """
    Main class representing a match line.

    This class should be subclassed for the different match lines.
    """

    version: Version
    field_names: tuple
    pattern: re.Pattern
    out_pattern: str
    line_dict: dict

    def __init__(self, version: Version, **kwargs) -> None:
        # set version
        self.version = version
        # Get pattern
        self.pattern = self.line_dict[self.version]["pattern"]
        # Get field names
        self.field_names = self.line_dict[self.version]["field_names"]
        # Get out pattern
        self.out_pattern = self.line_dict[self.version]["matchline"]

        # set field names
        for field in self.field_names:
            if field.lower() not in kwargs:
                raise ValueError(f"{field.lower()} is not given in keyword arguments")
            setattr(self, field, kwargs[field.lower()])

    def __str__(self) -> str:
        """
        Prints the printing the match line
        """
        r = [self.__class__.__name__]
        for fn in self.field_names:
            r.append(" {0}: {1}".format(fn, self.__dict__[fn.lower()]))
        return "\n".join(r) + "\n"

    @property
    def matchline(self) -> str:
        """
        Generate matchline as a string.
        """
        matchline = self.out_pattern.format(
            **dict(
                [
                    (field, self.format_fun[field](getattr(self, field)))
                    for field in self.field_names
                ]
            )
        )

        return matchline

    @classmethod
    def from_matchline(
        cls,
        matchline: str,
        version: Version = CURRENT_MAJOR_VERSION,
    ):
        """
        Create a new MatchLine object from a string

        Parameters
        ----------
        matchline : str
            String with a matchline
        version : Version
            Version of the matchline

        Returns
        -------
        a MatchLine instance
        """
        raise NotImplementedError

    def check_types(self) -> bool:
        """
        Check whether the values of the attributes are of the correct type.
        """
        raise NotImplementedError


# Dictionary of interpreter, formatters and datatypes for version 1.0.0
# each entry in the dictionary is a tuple with
# an intepreter (to parse the input), a formatter (for the output matchline)
# and type
INFO_LINE_INTERPRETERS_V_1_0_0 = {
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

# Dictionary containing the definition of all versions of the MatchInfo line
# starting from version 1.0.0
INFO_LINE = {
    Version(1, 0, 0): {
        "pattern": re.compile(
            # r"info\(\s*(?P<Attribute>[^,]+)\s*,\s*(?P<Value>.+)\s*\)\."
            r"info\((?P<attribute>[^,]+),(?P<value>.+)\)\."
        ),
        "field_names": ("attribute", "value"),
        "matchline": "info({attribute},{value}).",
        "value": INFO_LINE_INTERPRETERS_V_1_0_0,
    }
}


class MatchInfo(MatchLine):
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

    line_dict = INFO_LINE

    def __init__(self, version: Version, **kwargs) -> None:
        super().__init__(version, **kwargs)

        self.interpret_fun = self.line_dict[self.version]["value"][self.attribute][0]
        self.value_type = self.line_dict[self.version]["value"][self.attribute][2]
        self.format_fun = {
            "attribute": format_string,
            "value": self.line_dict[self.version]["value"][self.attribute][1],
        }

    @property
    def matchline(self) -> str:
        matchline = self.out_pattern.format(
            **dict(
                [
                    (field, self.format_fun[field](getattr(self, field)))
                    for field in self.field_names
                ]
            )
        )

        return matchline

    @classmethod
    def from_matchline(
        cls,
        matchline: str,
        pos: int = 0,
        version=CURRENT_VERSION,
    ) -> MatchLine:
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
        a MatchLine instance
        """
        class_dict = INFO_LINE[version]

        match_pattern = class_dict["pattern"].search(matchline, pos=pos)

        if match_pattern is not None:
            attribute, value_str = match_pattern.groups()
            if attribute not in class_dict["value"].keys():
                raise ValueError(
                    f"Attribute {attribute} is not specified in version {version}"
                )

            value = class_dict["value"][attribute][0](value_str)

            return cls(version=version, attribute=attribute, value=value)

        else:
            raise MatchError("Input match line does not fit the expected pattern.")


SCOREPROP_LINE_INTERPRETERS_V_1_0_0 = {
    "keySignature": (interpret_as_string, format_string, str),
    "timeSignature": (interpret_as_string, format_string, str),
}

SCOREPROP_LINE = {
    Version(1, 0, 0): {
        "pattern": re.compile(
            r"scoreProp\((?P<attribute>[^,]+),(?P<value>[^,]+),(?P<measure>\d+):(?P<beat>[\d\/]*)\)\."
        ),
        "field_names": (
            "attribute",
            "value",
            "measure",
            "beat",
            "offset",
            "onset_in_beats",
        ),
        "matchline": "scoreProp({attribute},{value},{measure}:{beat},{offset},{onset_in_beats}).",
        "value": SCOREPROP_LINE_INTERPRETERS_V_1_0_0,
    }
}


class MatchScoreProp(MatchLine):

    line_dict = SCOREPROP_LINE

    def __init__(self, version: Version, **kwargs) -> None:
        super().__init__(version, **kwargs)

        self.interpret_fun = self.line_dict[self.version]["value"]


class KeySignatureLine(MatchScoreProp):
    def __init__(
        self,
        version: Version,
        key_signature: str,
        measure: int,
        beat: int,
        offset: Union[int, FractionalSymbolicDuration],
        onset_in_beats: float,
    ) -> None:
        super().__init__(
            version=version,
            attribute="keySignature",
            value=key_signature,
            measure=measure,
            beat=beat,
            offset=offset,
            onset_in_beats=onset_in_beats,
        )


def load_match(fn, create_part=False, pedal_threshold=64, first_note_at_zero=False):
    pass


if __name__ == "__main__":

    matchfile_version_line_str = "info(matchFileVersion,1.0.0)."

    matchfile_version_line = MatchInfo.from_matchline(matchfile_version_line_str)

    print(matchfile_version_line)
    print(matchfile_version_line.matchline)

    assert matchfile_version_line.matchline == matchfile_version_line_str
