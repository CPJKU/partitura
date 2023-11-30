#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains base classes for Match lines and utilities for
parsing and formatting match lines.
"""
from __future__ import annotations

from typing import Callable, Tuple, Any, Optional, Union, Dict, List, Iterable
import re

import numpy as np

from partitura.utils.music import (
    pitch_spelling_to_midi_pitch,
    ensure_pitch_spelling_format,
    ALTER_SIGNS,
)

from partitura.io.matchfile_utils import (
    Version,
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
    interpret_as_list_int,
    format_list,
    MatchKeySignature,
    MatchTimeSignature,
)

from partitura.utils.misc import (
    PathLike,
    deprecated_alias,
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
    format_fun: Union[
        Dict[str, Callable[Any, str]],
        Tuple[Dict[str, Callable[Any, str]]],
    ]

    # Regular expression to parse
    # information from a string.
    pattern: Union[re.Pattern, Tuple[re.Pattern]]

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
        raise NotImplementedError  # pragma: no cover

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


## The following classes define base information for match lines
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
    attribute: str
        Name of the attribute
    value: Any
        Value of the attribute
    value_type: type
        Type of the value
    format_fun: callable
        A function for maping values to the attribute to strings (for formatting
        the output matchline)
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


class BaseStimeLine(MatchLine):
    """
    Base class for specifying stime lines. These lines should looke like

    stime(<Measure>:<Beat>,<Offset>,<OnsetInBeats>,<AnnotationType>)

    Parameters
    ----------
    version: Version
        The version of the matchline
    measure: int
        The measure number
    beat: int
        The beat within the measure (first beat starts at 1)
    offset: FractionalSymbolicDuration
        The offset of the event with respect to the current beat.
    onset_in_beats: float
        Onset in beats
    annotation_type: List[str]
        List of annotation types for the score time.
    """

    field_names = ("Measure", "Beat", "Offset", "OnsetInBeats", "AnnotationType")

    field_types = (int, int, FractionalSymbolicDuration, float, list)

    format_fun = dict(
        Measure=format_int,
        Beat=format_int,
        Offset=format_fractional,
        OnsetInBeats=format_float,
        AnnotationType=format_list,
    )

    out_pattern = "stime({Measure}:{Beat},{Offset},{OnsetInBeats},{AnnotationType})"

    pattern = re.compile(
        r"stime\("
        r"(?P<Measure>[^,]+):(?P<Beat>[^,]+),"
        r"(?P<Offset>[^,]+),"
        r"(?P<OnsetInBeats>[^,]+),"
        r"\[(?P<AnnotationType>[a-z,]*)\]\)"
    )

    def __init__(
        self,
        version: Version,
        measure: int,
        beat: int,
        offset: FractionalSymbolicDuration,
        onset_in_beats: float,
        annotation_type: List[str],
    ) -> None:
        super().__init__(version)

        self.Measure = measure
        self.Beat = beat
        self.Offset = offset
        self.OnsetInBeats = onset_in_beats
        self.AnnotationType = annotation_type


class BasePtimeLine(MatchLine):
    """
    Base class for specifying performance time. Tese lines have the form

    ptime([<Onsets>])

    Parameters
    ----------
    version: Version
        The version of the matchline
    onsets: List[int]
        The list of onsets.
    """

    field_names = ("Onsets",)
    field_types = (list,)

    out_pattern = "ptime({Onsets})."

    pattern = re.compile(r"ptime\(\[(?P<Onsets>[0-9,]+)\]\)\.")

    format_fun = dict(Onsets=format_list)

    def __init__(self, version: Version, onsets: List[int]) -> None:
        super().__init__(version)
        self.Onsets = onsets

    @property
    def Onset(self) -> float:
        """
        Average onset time
        """
        return np.mean(self.Onsets)


class BaseStimePtimeLine(MatchLine):
    """
    Base class for represeting score-to-performance time alignments

    Parameters
    ----------
    version: Version
        The version of the matchline
    stime: BaseStimeLine
        Score time as a BaseStimeLine instance.
    ptime: BasePtimeLine
        Performance time as a BasePtimeLine instance.
    """

    out_pattern = "{StimeLine}-{PtimeLine}"

    def __init__(
        self,
        version: Version,
        stime: BaseStimeLine,
        ptime: BasePtimeLine,
    ) -> None:
        super().__init__(version)

        self.stime = stime
        self.ptime = ptime

        self.field_names = self.stime.field_names + self.ptime.field_names
        self.field_types = self.stime.field_types + self.ptime.field_types

        self.pattern = (self.stime.pattern, self.ptime.pattern)

        self.format_fun = (self.stime.format_fun, self.ptime.format_fun)

    @property
    def matchline(self) -> str:
        return self.out_pattern.format(
            StimeLine=self.stime.matchline,
            PtimeLine=self.ptime.matchline,
        )

    def __str__(self) -> str:
        """
        String magic method
        """
        r = [self.__class__.__name__]
        r += [" Stime"] + [
            "   {0}: {1}".format(fn, getattr(self.stime, fn, None))
            for fn in self.stime.field_names
        ]

        r += [" Ptime"] + [
            "   {0}: {1}".format(fn, getattr(self.ptime, fn, None))
            for fn in self.ptime.field_names
        ]

        return "\n".join(r) + "\n"

    def check_types(self, verbose: bool) -> bool:
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

        stime_types_are_correct = self.stime.check_types(verbose)
        ptime_types_are_correct = self.ptime.check_types(verbose)

        types_are_correct = stime_types_are_correct and ptime_types_are_correct

        return types_are_correct

    @classmethod
    def prepare_kwargs_from_matchline(
        cls,
        matchline: str,
        stime_class: BaseStimeLine,
        ptime_class: BaseNoteLine,
        version: Version,
    ) -> Dict:
        stime = stime_class.from_matchline(matchline, version=version)
        ptime = ptime_class.from_matchline(matchline, version=version)

        kwargs = dict(
            version=version,
            stime=stime,
            ptime=ptime,
        )

        return kwargs


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
        (int, type(None)),
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
        NoteName=lambda x: str(x).upper(),
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
    def Bar(self) -> int:
        # deprecatd property measure
        return self.Measure

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
        # The MIDI pitch is not a part of all
        # note versions. For versions < 1.0.0
        # it needs to be inferred from pitch spelling.
        self.MidiPitch = midi_pitch
        self.Onset = onset
        self.Offset = offset
        self.Velocity = velocity

    @property
    def Duration(self):
        return self.Offset - self.Onset


class BaseSnoteNoteLine(MatchLine):
    out_pattern = "{SnoteLine}-{NoteLine}"

    def __init__(
        self,
        version: Version,
        snote: BaseSnoteLine,
        note: BaseNoteLine,
    ) -> None:
        super().__init__(version)

        self.snote = snote
        self.note = note

        self.field_names = self.snote.field_names + self.note.field_names

        self.field_types = self.snote.field_types + self.note.field_types

        self.pattern = (self.snote.pattern, self.note.pattern)

        self.format_fun = (self.snote.format_fun, self.note.format_fun)

    @property
    def matchline(self) -> str:
        return self.out_pattern.format(
            SnoteLine=self.snote.matchline,
            NoteLine=self.note.matchline,
        )

    def __str__(self) -> str:
        """
        Prints the printing the match line
        """
        r = [self.__class__.__name__]
        r += [" Snote"] + [
            "   {0}: {1}".format(fn, getattr(self.snote, fn, None))
            for fn in self.snote.field_names
        ]

        r += [" Note"] + [
            "   {0}: {1}".format(fn, getattr(self.note, fn, None))
            for fn in self.note.field_names
        ]

        return "\n".join(r) + "\n"

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
        snote_types_are_correct = self.snote.check_types(verbose)
        note_types_are_correct = self.note.check_types(verbose)

        types_are_correct = snote_types_are_correct and note_types_are_correct

        return types_are_correct

    @classmethod
    def prepare_kwargs_from_matchline(
        cls,
        matchline: str,
        snote_class: BaseSnoteLine,
        note_class: BaseNoteLine,
        version: Version,
    ) -> Dict:
        snote = snote_class.from_matchline(matchline, version=version)
        note = note_class.from_matchline(matchline, version=version)

        kwargs = dict(
            version=version,
            snote=snote,
            note=note,
        )

        return kwargs


class BaseDeletionLine(MatchLine):
    out_pattern = "{SnoteLine}-deletion."
    identifier_pattern = re.compile(r"-deletion\.")

    def __init__(self, version: Version, snote: BaseSnoteLine) -> None:
        super().__init__(version)

        self.snote = snote

        self.field_names = self.snote.field_names

        self.field_types = self.snote.field_types

        self.pattern = re.compile(rf"{self.snote.pattern.pattern}-deletion\.")

        self.format_fun = self.snote.format_fun

        for fn in self.field_names:
            setattr(self, fn, getattr(self.snote, fn))

    @property
    def matchline(self) -> str:
        return self.out_pattern.format(
            SnoteLine=self.snote.matchline,
        )

    @classmethod
    def prepare_kwargs_from_matchline(
        cls,
        matchline: str,
        snote_class: BaseSnoteLine,
        version: Version,
        pos: int = 0,
    ) -> Dict:
        match_pattern = cls.identifier_pattern.search(matchline, pos=pos)

        if match_pattern is None:
            raise MatchError("Input match line does not fit the expected pattern.")
        snote = snote_class.from_matchline(matchline, version=version)

        kwargs = dict(
            version=version,
            snote=snote,
        )

        return kwargs


class BaseInsertionLine(MatchLine):
    out_pattern = "insertion-{NoteLine}"
    identifier_pattern = re.compile(r"insertion-")

    def __init__(self, version: Version, note: BaseNoteLine) -> None:
        super().__init__(version)

        self.note = note

        self.field_names = self.note.field_names

        self.field_types = self.note.field_types

        self.pattern = re.compile(f"insertion-{self.note.pattern.pattern}")

        self.format_fun = self.note.format_fun

        for fn in self.field_names:
            setattr(self, fn, getattr(self.note, fn))

    @property
    def matchline(self) -> str:
        return self.out_pattern.format(
            NoteLine=self.note.matchline,
        )

    @classmethod
    def prepare_kwargs_from_matchline(
        cls,
        matchline: str,
        note_class: BaseNoteLine,
        version: Version,
        pos: int = 0,
    ) -> Dict:
        match_pattern = cls.identifier_pattern.search(matchline, pos=pos)

        if match_pattern is None:
            raise MatchError("Input match line does not fit the expected pattern.")

        note = note_class.from_matchline(matchline, version=version)

        kwargs = dict(
            version=version,
            note=note,
        )

        return kwargs


class BaseOrnamentLine(MatchLine):
    # These field names and types need to be expanded
    # with the attributes of the note
    field_names = ("Anchor",)
    field_types = (str,)
    format_fun = dict(Anchor=format_string)
    out_pattern = "ornament({Anchor})-{NoteLine}"
    ornament_pattern: re.Pattern = re.compile(r"ornament\((?P<Anchor>[^\)]*)\)-")

    def __init__(self, version: Version, anchor: str, note: BaseNoteLine) -> None:
        super().__init__(version)

        self.note = note

        self.field_names = self.field_names + self.note.field_names

        self.field_types = self.field_types + self.note.field_types

        self.pattern = (self.ornament_pattern, self.note.pattern)

        self.format_fun = (self.format_fun, self.note.format_fun)

        for fn in self.note.field_names:
            setattr(self, fn, getattr(self.note, fn))

        self.Anchor = anchor

    @property
    def matchline(self) -> str:
        return self.out_pattern.format(
            Anchor=self.Anchor,
            NoteLine=self.note.matchline,
        )

    @classmethod
    def prepare_kwargs_from_matchline(
        cls,
        matchline: str,
        note_class: BaseNoteLine,
        version: Version,
    ) -> Dict:
        anchor_pattern = cls.ornament_pattern.search(matchline)

        if anchor_pattern is None:
            raise MatchError("Input match line does not fit the expected pattern.")

        anchor = interpret_as_string(anchor_pattern.group("Anchor"))
        note = note_class.from_matchline(matchline, version=version)

        kwargs = dict(
            version=version,
            anchor=anchor,
            note=note,
        )

        return kwargs


class BasePedalLine(MatchLine):
    """
    Class for representing a sustain pedal line
    """

    field_names = ("Time", "Value")
    field_types = (int, int)
    base_pattern: str = r"pedal\((?P<Time>[^,]+),(?P<Value>[^,]+)\)\."
    out_pattern: str = "pedal({Time},{Value})."

    format_fun = dict(
        Time=format_int,
        Value=format_int,
    )

    def __init__(
        self,
        version: Version,
        time: int,
        value: int,
    ):
        super().__init__(version)
        self.Time = time
        self.Value = value

    @classmethod
    def prepare_kwargs_from_matchline(
        cls,
        matchline: str,
        version: Version,
        pos: int = 0,
    ) -> Dict:
        kwargs = None
        # pattern = re.compile(cls.base_pattern.format(pedal_type=pedal_type))

        match_pattern = cls.pattern.search(matchline, pos=pos)

        if match_pattern is not None:
            time_str, value_str = match_pattern.groups()

            kwargs = dict(
                version=version,
                time=interpret_as_int(time_str),
                value=interpret_as_int(value_str),
            )

        return kwargs


class BaseSustainPedalLine(BasePedalLine):
    pattern = re.compile(r"sustain\((?P<Time>[^,]+),(?P<Value>[^,]+)\)\.")
    out_pattern: str = "sustain({Time},{Value})."

    def __init__(
        self,
        version: Version,
        time: int,
        value: int,
    ):
        super().__init__(version=version, time=time, value=value)


class BaseSoftPedalLine(BasePedalLine):
    pattern = re.compile(r"soft\((?P<Time>[^,]+),(?P<Value>[^,]+)\)\.")
    out_pattern: str = "soft({Time},{Value})."

    def __init__(
        self,
        version: Version,
        time: int,
        value: int,
    ):
        super().__init__(version=version, time=time, value=value)


## MatchFile

# classes that contain score notes
snote_classes = (BaseSnoteLine, BaseSnoteNoteLine, BaseDeletionLine)

# classes that contain performed notes.
note_classes = (BaseNoteLine, BaseSnoteNoteLine, BaseInsertionLine)


class MatchFile(object):
    """
    Class for representing MatchFiles
    """

    version: Version
    lines: np.ndarray

    def __init__(self, lines: Iterable[MatchLine]) -> None:
        # check that all lines have the same version
        same_version = all([line.version == lines[0].version for line in lines])

        if not same_version:
            raise ValueError("All lines should have the same version")

        self.lines = np.array(lines)

    @property
    def note_pairs(self) -> List[Tuple[BaseSnoteLine, BaseNoteLine]]:
        """
        Return all(snote, note) tuples

        """
        return [
            (x.snote, x.note) for x in self.lines if isinstance(x, BaseSnoteNoteLine)
        ]

    @property
    def notes(self) -> List[BaseNoteLine]:
        """
        Return all performed notes (as MatchNote objects)
        """
        return [x.note for x in self.lines if isinstance(x, note_classes)]

    def iter_notes(self) -> BaseNoteLine:
        """
        Iterate over all performed notes (as MatchNote objects)
        """
        for x in self.lines:
            if isinstance(x, note_classes):
                yield x.note

    @property
    def snotes(self) -> List[BaseSnoteLine]:
        """
        Return all score notes (as MatchSnote objects)
        """
        return [x.snote for x in self.lines if isinstance(x, snote_classes)]

    def iter_snotes(self) -> BaseSnoteLine:
        """
        Iterate over all performed notes (as MatchNote objects)
        """
        for x in self.lines:
            if hasattr(x, "snote"):
                yield x.snote

    @property
    def sustain_pedal(self) -> List[BaseSustainPedalLine]:
        return [line for line in self.lines if isinstance(line, BaseSustainPedalLine)]

    @property
    def soft_pedal(self) -> List[BasePedalLine]:
        return [line for line in self.lines if isinstance(line, BaseSoftPedalLine)]

    @property
    def insertions(self) -> List[BaseNoteLine]:
        return [x.note for x in self.lines if isinstance(x, BaseInsertionLine)]

    @property
    def deletions(self) -> List[BaseSnoteLine]:
        return [x.snote for x in self.lines if isinstance(x, BaseDeletionLine)]

    @property
    def _info(self) -> List[BaseInfoLine]:
        """
        Return all InfoLine objects

        """
        return [i for i in self.lines if isinstance(i, BaseInfoLine)]

    def info(
        self, attribute: Optional[str] = None
    ) -> Union[BaseInfoLine, List[BaseInfoLine]]:
        """
        Return the value of the MatchInfo object corresponding to
        attribute, or None if there is no such object

        : param attribute: the name of the attribute to return the value for

        """
        if attribute:
            try:
                idx = [i.Attribute for i in self._info].index(attribute)
                return self._info[idx].Value
            except ValueError:
                return None
        else:
            return self._info

    @property
    def first_onset(self) -> float:
        return min([n.OnsetInBeats for n in self.snotes])

    @property
    def first_measure(self) -> float:
        return min([n.Measure for n in self.snotes])

    @property
    def time_signatures(self):
        """
        A list of tuples(t, b, (n, d)), indicating a time signature of
        n over v, starting at t in bar b

        """
        _tsigs = [
            (
                getattr(tsl, "TimeInBeats", self.first_onset),
                getattr(tsl, "Measure", self.first_measure),
                tsl.Value,
            )
            for tsl in self.time_sig_lines
        ]

        _tsigs.sort(key=lambda x: x[0])

        tsigs = []
        if len(_tsigs) > 0:
            tsigs.append(_tsigs[0])

            for k in _tsigs:
                if k[2] != tsigs[-1][2]:
                    tsigs.append(k)

        return tsigs

    @property
    def time_sig_lines(self):
        ml = [
            line
            for line in self.lines
            if getattr(line, "Attribute", None) == "timeSignature"
        ]
        return ml

    @property
    def key_signatures(self):
        """
        A list of tuples (t, b, (ks,)) or (t, b, (ks1, ks2))
        """
        _keysigs = [
            (
                getattr(ksl, "TimeInBeats", self.first_onset),
                getattr(ksl, "Measure", self.first_measure),
                ksl.Value,
            )
            for ksl in self.key_sig_lines
        ]

        _keysigs.sort(key=lambda x: x[0])

        keysigs = []
        if len(_keysigs) > 0:
            keysigs.append(_keysigs[0])

            for k in _keysigs:
                if k[2] != keysigs[-1][2]:
                    keysigs.append(k)

        return keysigs

    @property
    def key_sig_lines(self):
        ml = [
            line
            for line in self.lines
            if getattr(line, "Attribute", None) == "keySignature"
        ]

        return ml

    def write(self, filename: PathLike) -> None:
        with open(filename, "w") as f:
            for line in self.lines:
                f.write(line.matchline + "\n")
