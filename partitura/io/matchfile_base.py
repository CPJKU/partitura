#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains base classes for Match lines and utilities for
parsing and formatting match lines.
"""
from __future__ import annotations

from typing import Callable, Tuple, Any, Optional, Union, Dict, List
import re

import numpy as np

from partitura.utils.music import (
    pitch_spelling_to_midi_pitch,
    ensure_pitch_spelling_format,
    ALTER_SIGNS,
)

from partitura.io.matchfile_utils import (
    Version,
    interpret_version,
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


class MatchError(Exception):
    """
    Base exception for parsing match files.
    """

    pass


class MatchLine(object):
    """
    Base class for representing match lines.

    This class should be subclassed for the different match lines.

    Parameters
    ----------
    version : Version
        Indicate the version of the match line.

    Attributes
    ----------
    version: Version
        The version of the match line.
    field_names: Tuple[str]
        The names of the different fields with information in the match line.
    field_types : Tuple[type]
        The data type of the different fields.
    out_pattern : str
        The output pattern for the match line (i.e., how the match line looks like
        in a match file).
    pattern : re.Pattern
        Regular expression to parse information from a string.
    format_fun: Dict[str, Callable]
        A dictionary of methods for formatting the values of each field name.
    """

    # Version of the match line
    version: Version

    # Field names that appear in the match line
    # A match line will generally have these
    # field names as attributes.
    # Following the original Prolog-based specification
    # the names of the attributes start with upper-case letters
    field_names: Tuple[str]

    # type of the information in the fields
    field_types: Tuple[Union[type, Tuple[type]]]

    # Output pattern
    out_pattern: str

    # A dictionary of callables for each field name
    # the callables should get the value of the input
    # and return a string formatted for the matchfile.
    format_fun: Dict[str, Callable[Any, str]]

    # Regular expression to parse
    # information from a string.
    pattern = re.Pattern

    def __init__(self, version: Version) -> None:
        # Subclasses need to initialize the other
        # default field names
        self.version = version

    def __str__(self) -> str:
        """
        Prints the printing the match line
        """
        r = [self.__class__.__name__]
        for fn in self.field_names:
            r.append(" {0}: {1}".format(fn, self.__dict__[fn]))
        return "\n".join(r) + "\n"

    @property
    def matchline(self) -> str:
        """
        Generate matchline as a string.

        This method can be adapted as needed by subclasses.
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
        version: Version,
    ) -> MatchLine:
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

    def check_types(self, verbose: bool = False) -> bool:
        """
        Check whether the values of the fields are of the correct type.

        Parameters
        ----------
        verbose : bool
            Prints whether each of the attributes in field_names has the correct dtype.
            values are

        Returns
        -------
        types_are_correct : bool
            True if the values of all fields in the match line have the
            correct type.
        """
        types_are_correct_list = [
            isinstance(getattr(self, field), field_type)
            for field, field_type in zip(self.field_names, self.field_types)
        ]
        if verbose:
            print(list(zip(self.field_names, types_are_correct_list)))

        types_are_correct = all(types_are_correct_list)
        return types_are_correct


## The following classes define match lines that appear in all matchfile versions
## These classes need to be subclassed in the corresponding module for each version.


class BaseInfoLine(MatchLine):
    """
    Base class specifying global information lines.

    These lines have the general structure "info(<Attribute>,<Value>)."
    Which attributes are valid depending on the version of the match line.

    Parameters
    ----------
    version : Version
        The version of the info line.
    kwargs : keyword arguments
        Keyword arguments specifying the type of line and its value.
    """

    # Base field names (can be updated in subclasses).
    # "attribute" will have type str, but the type of value needs to be specified
    # during initialization.
    field_names = ("Attribute", "Value")

    out_pattern = "info({Attribute},{Value})."

    pattern = re.compile(r"info\((?P<Attribute>[^,]+),(?P<Value>.+)\)\.")

    def __init__(
        self,
        version: Version,
        attribute: str,
        value: Any,
        value_type: type,
        format_fun: Callable[Any, str],
    ) -> None:
        super().__init__(version)

        self.field_types = (str, value_type)
        self.format_fun = dict(Attribute=format_string, Value=format_fun)
        self.Attribute = attribute
        self.Value = value


# deprecate bar for measure
class BaseSnoteLine(MatchLine):
    """
    Base class to represent score notes.

    Parameters
    ----------
    version: Version
    anchor: str
    note_name: str
    modifier: str
    octave: Optional[int]
    measure: int
    beat: int
    offset: FractionalSymbolicDuration
    duration: FractionalSymbolicDuration
    onset_in_beats: float
    offset_in_beats: float
    score_attributes_list: List[str]

    Attributes
    ----------
    DurationInBeats : float
    DurationSymbolic : float
    MidiPitch : float

    Notes
    -----
    * The snote line has not changed much since the first version of
      the Match file format. New versions are just more explicit in the
      the formatting of the attributes (field names), e.g., NoteName
      should always be uppercase starting from version 1.0.0, etc.
    """

    # All derived classes should include
    # at least these field names
    field_names = (
        "Anchor",
        "NoteName",
        "Modifier",
        "Octave",
        "Measure",
        "Beat",
        "Offset",
        "Duration",
        "OnsetInBeats",
        "OffsetInBeats",
        "ScoreAttributesList",
    )

    field_types = (
        str,
        str,
        (int, type(None)),
        int,
        int,
        int,
        FractionalSymbolicDuration,
        FractionalSymbolicDuration,
        float,
        float,
        list,
    )

    out_pattern = (
        "snote({Anchor},[{NoteName},{Modifier}],{Octave},"
        "{Measure}:{Beat},{Offset},{Duration},{OnsetInBeats},"
        "{OffsetInBeats},{ScoreAttributesList})"
    )

    pattern = re.compile(
        r"snote\("
        r"(?P<Anchor>[^,]+),"
        r"\[(?P<NoteName>[^,]+),(?P<Modifier>[^,]+)\],"
        r"(?P<Octave>[^,]+),"
        r"(?P<Measure>[^,]+):(?P<Beat>[^,]+),"
        r"(?P<Offset>[^,]+),"
        r"(?P<Duration>[^,]+),"
        r"(?P<OnsetInBeats>[^,]+),"
        r"(?P<OffsetInBeats>[^,]+),"
        r"\[(?P<ScoreAttributesList>.*)\]\)"
    )

    format_fun = dict(
        Anchor=format_string,
        NoteName=lambda x: str(x.upper()),
        Modifier=lambda x: "n" if x == 0 else ALTER_SIGNS[x],
        Octave=format_int,
        Measure=format_int,
        Beat=format_int,
        Offset=format_fractional,
        Duration=format_fractional,
        OnsetInBeats=format_float_unconstrained,
        OffsetInBeats=format_float_unconstrained,
        ScoreAttributesList=format_list,
    )

    def __init__(
        self,
        version: Version,
        anchor: str,
        note_name: str,
        modifier: str,
        octave: Optional[int],
        measure: int,
        beat: int,
        offset: FractionalSymbolicDuration,
        duration: FractionalSymbolicDuration,
        onset_in_beats: float,
        offset_in_beats: float,
        score_attributes_list: List[str],
    ) -> None:
        super().__init__(version)

        # All of these attributes should have the
        # correct dtype (otherwise we need to be constantly
        # checking the types).
        self.Anchor = anchor
        self.NoteName = note_name
        self.Modifier = modifier
        self.Octave = octave
        self.Measure = measure
        self.Beat = beat
        self.Offset = offset
        self.Duration = duration
        self.OnsetInBeats = onset_in_beats
        self.OffsetInBeats = offset_in_beats
        self.ScoreAttributesList = score_attributes_list

    @property
    def DurationInBeats(self) -> float:
        return self.OffsetInBeats - self.OnsetInBeats

    @property
    def DurationSymbolic(self) -> str:
        # Duration should always be a FractionalSymbolicDuration
        return str(self.Duration)

    @property
    def MidiPitch(self) -> Optional[int]:
        if isinstance(self.Octave, int):
            return pitch_spelling_to_midi_pitch(
                step=self.NoteName, octave=self.Octave, alter=self.Modifier
            )
        else:
            return None

    @classmethod
    def prepare_kwargs_from_matchline(
        cls,
        matchline: str,
        pos: int = 0,
    ) -> Dict:

        match_pattern = cls.pattern.search(matchline, pos)

        if match_pattern is not None:

            (
                anchor_str,
                note_name_str,
                modifier_str,
                octave_str,
                measure_str,
                beat_str,
                offset_str,
                duration_str,
                onset_in_beats_str,
                offset_in_beats_str,
                score_attributes_list_str,
            ) = match_pattern.groups()

            anchor = interpret_as_string(anchor_str)
            note_name, modifier, octave = ensure_pitch_spelling_format(
                step=note_name_str,
                alter=modifier_str,
                octave=octave_str,
            )

            return dict(
                anchor=interpret_as_string(anchor),
                note_name=note_name,
                modifier=modifier,
                octave=octave,
                measure=interpret_as_int(measure_str),
                beat=interpret_as_int(beat_str),
                offset=interpret_as_fractional(offset_str),
                duration=interpret_as_fractional(duration_str),
                onset_in_beats=interpret_as_float(onset_in_beats_str),
                offset_in_beats=interpret_as_float(offset_in_beats_str),
                score_attributes_list=interpret_as_list(score_attributes_list_str),
            )

        else:
            raise MatchError("Input match line does not fit the expected pattern.")


class BaseNoteLine(MatchLine):

    # All derived classes should include at least
    # these field names
    field_names = (
        "Id",
        "MidiPitch",
        "Onset",
        "Offset",
        "Velocity",
    )

    field_types = (
        str,
        int,
        float,
        float,
        int,
    )

    def __init__(
        self,
        version: Version,
        id: str,
        midi_pitch: int,
        onset: float,
        offset: float,
        velocity: int,
    ) -> None:
        super().__init__(version)
        self.Id = id
        self.MidiPitch = midi_pitch
        self.Onset = onset
        self.Offset = offset
        self.Velocity = velocity

    @property
    def Duration(self):
        return self.Offset - self.Onset
