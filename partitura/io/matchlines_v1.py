#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains definitions for Matchfile lines for version >1.0.0
"""
from __future__ import annotations

import re

from typing import Any, Callable, Tuple, Union, List

from partitura.utils.music import (
    ALTER_SIGNS,
    ensure_pitch_spelling_format,
)

from partitura.io.matchfile_base import (
    MatchLine,
    MatchError,
    Version,
    BaseInfoLine,
    BaseSnoteLine,
    BaseStimeLine,
    BasePtimeLine,
    BaseStimePtimeLine,
    BaseNoteLine,
    BaseSnoteNoteLine,
    BaseDeletionLine,
    BaseInsertionLine,
    BaseSustainPedalLine,
    BaseSoftPedalLine,
)

from partitura.io.matchfile_utils import (
    interpret_version,
    format_version,
    interpret_as_string,
    format_string,
    interpret_as_float,
    format_float,
    interpret_as_int,
    format_int,
    FractionalSymbolicDuration,
    format_fractional,
    interpret_as_fractional,
    interpret_as_list,
    interpret_as_list_int,
    format_list,
    to_camel_case,
    get_kwargs_from_matchline,
)

# Define current version of the match file format
LATEST_MAJOR_VERSION = 1
LATEST_MINOR_VERSION = 0
LATEST_PATCH_VERSION = 0

LATEST_VERSION = Version(
    LATEST_MAJOR_VERSION,
    LATEST_MINOR_VERSION,
    LATEST_PATCH_VERSION,
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
            raise ValueError("The version must be >= 1.0.0")

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
        version: Version = LATEST_VERSION,
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
        if version not in INFO_LINE:
            raise ValueError(f"{version} is not specified for this class.")

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


SCOREPROP_LINE = {
    Version(1, 0, 0): {
        "timeSignature": (
            interpret_as_fractional,
            format_fractional,
            FractionalSymbolicDuration,
        ),
        "keySignature": (interpret_as_string, format_string, str),
        "beatSubDivision": (interpret_as_int, format_int, int),
        "directions": (interpret_as_list, format_list, list),
    }
}


class MatchScoreProp(MatchLine):

    field_names = (
        "Attribute",
        "Value",
        "Measure",
        "Beat",
        "Offset",
        "TimeInBeats",
    )

    out_pattern = (
        "scoreprop({Attribute},{Value},{Measure}:{Beat},{Offset},{TimeInBeats})."
    )

    pattern = re.compile(
        r"scoreprop\("
        r"(?P<Attribute>[^,]+),"
        r"(?P<Value>[^,]+),"
        r"(?P<Measure>[^,]+):(?P<Beat>[^,]+),"
        r"(?P<Offset>[^,]+),"
        r"(?P<TimeInBeats>[^,]+)\)\."
    )

    def __init__(
        self,
        version: Version,
        attribute: str,
        value: Any,
        value_type: type,
        format_fun: Callable[Any, str],
        measure: int,
        beat: int,
        offset: FractionalSymbolicDuration,
        time_in_beats: float,
    ) -> None:

        if version < Version(1, 0, 0):
            raise ValueError("The version must be >= 1.0.0")

        super().__init__(version)

        self.field_types = (
            str,
            value_type,
            int,
            int,
            FractionalSymbolicDuration,
            float,
        )

        self.format_fun = dict(
            Attribute=format_string,
            Value=format_fun,
            Measure=format_int,
            Beat=format_int,
            Offset=format_fractional,
            TimeInBeats=format_float,
        )

        # set class attributes
        self.Attribute = attribute
        self.Value = value
        self.Measure = measure
        self.Beat = beat
        self.Offset = offset
        self.TimeInBeats = time_in_beats

    @classmethod
    def from_matchline(
        cls,
        matchline: str,
        pos: int = 0,
        version: Version = LATEST_VERSION,
    ) -> MatchInfo:
        """
        Create a new MatchScoreProp object from a string

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
        a MatchScoreProp object
        """

        if version not in SCOREPROP_LINE:
            raise ValueError(f"{version} is not specified for this class.")

        match_pattern = cls.pattern.search(matchline, pos=pos)

        class_dict = SCOREPROP_LINE[version]

        if match_pattern is not None:

            (
                attribute,
                value_str,
                measure_str,
                beat_str,
                offset_str,
                time_in_beats_str,
            ) = match_pattern.groups()

            if attribute not in class_dict:
                raise ValueError(f"Attribute {attribute} is not specified in {version}")

            interpret_fun, format_fun, value_type = class_dict[attribute]

            value = interpret_fun(value_str)

            measure = interpret_as_int(measure_str)

            beat = interpret_as_int(beat_str)

            offset = interpret_as_fractional(offset_str)

            time_in_beats = interpret_as_float(time_in_beats_str)

            return cls(
                version=version,
                attribute=attribute,
                value=value,
                value_type=value_type,
                format_fun=format_fun,
                measure=measure,
                beat=beat,
                offset=offset,
                time_in_beats=time_in_beats,
            )

        else:
            raise MatchError("Input match line does not fit the expected pattern.")


SECTION_LINE = {
    Version(1, 0, 0): {
        "StartInBeatsUnfolded": (interpret_as_float, format_float, float),
        "EndInBeatsUnfolded": (interpret_as_float, format_float, float),
        "StartInBeatsOriginal": (interpret_as_float, format_float, float),
        "EndInBeatsOriginal": (interpret_as_float, format_float, float),
        "RepeatEndType": (interpret_as_list, format_list, list),
    }
}


class MatchSection(MatchLine):
    """
    Class for specifiying structural information (i.e., sections).

    section(StartInBeatsUnfolded,EndInBeatsUnfolded,StartInBeatsOriginal,EndInBeatsOriginal,RepeatEndType).

    Parameters
    ----------
    version: Version,
    start_in_beats_unfolded: float,
    end_in_beats_unfolded: float,
    start_in_beats_original: float,
    end_in_beats_original: float,
    repeat_end_type: List[str]
    """

    field_names = (
        "StartInBeatsUnfolded",
        "EndInBeatsUnfolded",
        "StartInBeatsOriginal",
        "EndInBeatsOriginal",
        "RepeatEndType",
    )

    out_pattern = (
        "section({StartInBeatsUnfolded},"
        "{EndInBeatsUnfolded},{StartInBeatsOriginal},"
        "{EndInBeatsOriginal},{RepeatEndType})."
    )
    pattern = re.compile(
        r"section\("
        r"(?P<StartInBeatsUnfolded>[^,]+),"
        r"(?P<EndInBeatsUnfolded>[^,]+),"
        r"(?P<StartInBeatsOriginal>[^,]+),"
        r"(?P<EndInBeatsOriginal>[^,]+),"
        r"\[(?P<RepeatEndType>.*)\]\)."
    )

    def __init__(
        self,
        version: Version,
        start_in_beats_unfolded: float,
        end_in_beats_unfolded: float,
        start_in_beats_original: float,
        end_in_beats_original: float,
        repeat_end_type: List[str],
    ) -> None:

        if version not in SECTION_LINE:
            raise ValueError(
                f"Unknown version {version}!. "
                f"Supported versions are {list(SECTION_LINE.keys())}"
            )
        super().__init__(version)

        self.field_types = tuple(
            SECTION_LINE[version][fn][2] for fn in self.field_names
        )
        self.format_fun = dict(
            [(fn, ft[1]) for fn, ft in SECTION_LINE[version].items()]
        )

        self.StartInBeatsUnfolded = start_in_beats_unfolded
        self.EndInBeatsUnfolded = end_in_beats_unfolded
        self.StartInBeatsOriginal = start_in_beats_original
        self.EndInBeatsOriginal = end_in_beats_original
        self.RepeatEndType = repeat_end_type

    @classmethod
    def from_matchline(
        cls,
        matchline: str,
        pos: int = 0,
        version: Version = LATEST_VERSION,
    ) -> MatchSection:
        if version not in SECTION_LINE:
            raise ValueError(
                f"Unknown version {version}!. "
                f"Supported versions are {list(SECTION_LINE.keys())}"
            )

        match_pattern = cls.pattern.search(matchline, pos=pos)
        class_dict = SECTION_LINE[version]

        if match_pattern is not None:

            kwargs = dict(
                [
                    (to_camel_case(fn), class_dict[fn][0](match_pattern.group(fn)))
                    for fn in cls.field_names
                ]
            )

            return cls(version=version, **kwargs)

        else:
            raise MatchError("Input match line does not fit the expected pattern.")


STIME_LINE = {
    Version(1, 0, 0): {
        "Measure": (interpret_as_int, format_int, int),
        "Beat": (interpret_as_int, format_int, int),
        "Offset": (
            interpret_as_fractional,
            format_fractional,
            FractionalSymbolicDuration,
        ),
        "OnsetInBeats": (interpret_as_float, format_float, float),
        "AnnotationType": (interpret_as_list, format_list, list),
    }
}


class MatchStime(BaseStimeLine):
    def __init__(
        self,
        version: Version,
        measure: int,
        beat: int,
        offset: FractionalSymbolicDuration,
        onset_in_beats: float,
        annotation_type: List[str],
    ) -> None:

        if version < Version(1, 0, 0):
            raise ValueError(f"{version} < Version(1, 0, 0)")

        super().__init__(
            version=version,
            measure=measure,
            beat=beat,
            offset=offset,
            onset_in_beats=onset_in_beats,
            annotation_type=annotation_type,
        )

        self.field_types = tuple(STIME_LINE[version][fn][2] for fn in self.field_names)
        self.format_fun = dict(
            [(fn, STIME_LINE[version][fn][1]) for fn in self.field_names]
        )

    @classmethod
    def from_matchline(
        cls,
        matchline: str,
        pos: int = 0,
        version: Version = LATEST_VERSION,
    ) -> MatchStime:

        if version not in STIME_LINE:
            raise ValueError(
                f"Unknown version {version}!. "
                f"Supported versions are {list(STIME_LINE.keys())}"
            )

        kwargs = get_kwargs_from_matchline(
            matchline=matchline,
            pattern=cls.pattern,
            field_names=cls.field_names,
            class_dict=STIME_LINE[version],
            pos=pos,
        )

        if kwargs is None:
            raise MatchError("Input match line does not fit the expected pattern.")

        return cls(version=version, **kwargs)


PTIME_LINE = {
    Version(1, 0, 0): {
        "Onsets": (interpret_as_list_int, format_list, list),
    }
}


class MatchPtime(BasePtimeLine):
    def __init__(
        self,
        version: Version,
        onsets: List[int],
    ) -> None:

        if version < Version(1, 0, 0):
            raise ValueError(f"{version} < Version(1, 0, 0)")

        super().__init__(
            version=version,
            onsets=onsets,
        )

        self.field_types = tuple(PTIME_LINE[version][fn][2] for fn in self.field_names)
        self.format_fun = dict(
            [(fn, PTIME_LINE[version][fn][1]) for fn in self.field_names]
        )

    @classmethod
    def from_matchline(
        cls,
        matchline: str,
        pos: int = 0,
        version: Version = LATEST_VERSION,
    ) -> MatchStime:

        if version not in PTIME_LINE:
            raise ValueError(
                f"Unknown version {version}!. "
                f"Supported versions are {list(STIME_LINE.keys())}"
            )

        kwargs = get_kwargs_from_matchline(
            matchline=matchline,
            pattern=cls.pattern,
            field_names=cls.field_names,
            class_dict=PTIME_LINE[version],
            pos=pos,
        )

        if kwargs is None:
            raise MatchError("Input match line does not fit the expected pattern.")

        return cls(version=version, **kwargs)


class MatchSnote(BaseSnoteLine):

    format_fun = dict(
        Anchor=format_string,
        NoteName=lambda x: str(x.upper()),
        Modifier=lambda x: "n" if x == 0 else ALTER_SIGNS[x],
        Octave=format_int,
        Measure=format_int,
        Beat=format_int,
        Offset=format_fractional,
        Duration=format_fractional,
        OnsetInBeats=format_float,
        OffsetInBeats=format_float,
        ScoreAttributesList=format_list,
    )

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
    ) -> None:

        if version < Version(1, 0, 0):
            raise ValueError(f"{version} < Version(1, 0, 0)")
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
        version: Version = LATEST_VERSION,
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

        if version < Version(1, 0, 0):
            raise ValueError(f"{version} < Version(1, 0, 0)")

        kwargs = cls.prepare_kwargs_from_matchline(
            matchline=matchline,
            pos=pos,
        )

        return cls(version=version, **kwargs)


NOTE_LINE = {
    Version(1, 0, 0): {
        "Id": (interpret_as_string, format_string, str),
        "MidiPitch": (interpret_as_int, format_int, int),
        "Onset": (interpret_as_int, format_int, int),
        "Offset": (interpret_as_int, format_int, int),
        "Velocity": (interpret_as_int, format_int, int),
        "Channel": (interpret_as_int, format_int, int),
        "Track": (interpret_as_int, format_int, int),
    }
}


class MatchNote(BaseNoteLine):

    field_names = (
        "Id",
        "MidiPitch",
        "Onset",
        "Offset",
        "Velocity",
        "Channel",
        "Track",
    )

    out_pattern = (
        "note({Id},{MidiPitch},{Onset},{Offset},{Velocity},{Channel},{Track})."
    )

    pattern = re.compile(
        # r"note\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)\)"
        r"note\((?P<Id>[^,]+),"
        r"(?P<MidiPitch>[^,]+),"
        r"(?P<Onset>[^,]+),"
        r"(?P<Offset>[^,]+),"
        r"(?P<Velocity>[^,]+),"
        r"(?P<Channel>[^,]+),"
        r"(?P<Track>[^,]+)\)"
    )

    def __init__(
        self,
        version: Version,
        id: str,
        midi_pitch: int,
        onset: int,
        offset: int,
        velocity: int,
        channel: int,
        track: int,
    ) -> None:

        if version not in NOTE_LINE:
            raise ValueError(
                f"Unknown version {version}!. "
                f"Supported versions are {list(NOTE_LINE.keys())}"
            )

        super().__init__(
            version=version,
            id=id,
            midi_pitch=midi_pitch,
            onset=onset,
            offset=offset,
            velocity=velocity,
        )

        self.Channel = channel
        self.Track = track

        self.field_types = tuple(NOTE_LINE[version][fn][2] for fn in self.field_names)
        self.format_fun = dict(
            [(fn, NOTE_LINE[version][fn][1]) for fn in self.field_names]
        )

    @classmethod
    def from_matchline(
        cls,
        matchline: str,
        pos: int = 0,
        version: Version = LATEST_VERSION,
    ) -> MatchNote:

        if version < Version(1, 0, 0):
            raise ValueError(f"{version} < Version(1, 0, 0)")

        kwargs = get_kwargs_from_matchline(
            matchline=matchline,
            pattern=cls.pattern,
            field_names=cls.field_names,
            class_dict=NOTE_LINE[version],
            pos=pos,
        )

        if kwargs is not None:
            return cls(version=version, **kwargs)

        else:
            raise MatchError("Input match line does not fit the expected pattern.")


class MatchStimePtime(BaseStimePtimeLine):
    def __init__(
        self,
        version: Version,
        stime: MatchStime,
        ptime: MatchPtime,
    ) -> None:
        super().__init__(
            version=version,
            stime=stime,
            ptime=ptime,
        )

    @classmethod
    def from_matchline(
        cls,
        matchline: str,
        version: Version = LATEST_VERSION,
    ) -> MatchSnoteNote:

        if version < Version(1, 0, 0):
            raise ValueError(f"{version} < Version(1, 0, 0)")

        kwargs = cls.prepare_kwargs_from_matchline(
            matchline=matchline,
            stime_class=MatchStime,
            ptime_class=MatchPtime,
            version=version,
        )

        return cls(**kwargs)


class MatchSnoteNote(BaseSnoteNoteLine):
    def __init__(
        self,
        version: Version,
        snote: BaseSnoteLine,
        note: BaseNoteLine,
    ) -> None:

        super().__init__(
            version=version,
            snote=snote,
            note=note,
        )

    @classmethod
    def from_matchline(
        cls,
        matchline: str,
        version: Version = LATEST_VERSION,
    ) -> MatchSnoteNote:

        if version < Version(1, 0, 0):
            raise ValueError(f"{version} < Version(1, 0, 0)")

        kwargs = cls.prepare_kwargs_from_matchline(
            matchline=matchline,
            snote_class=MatchSnote,
            note_class=MatchNote,
            version=version,
        )

        return cls(**kwargs)


class MatchSnoteDeletion(BaseDeletionLine):
    def __init__(self, version: Version, snote: MatchSnote) -> None:

        super().__init__(
            version=version,
            snote=snote,
        )

    @classmethod
    def from_matchline(
        cls,
        matchline: str,
        version: Version = LATEST_VERSION,
    ) -> MatchSnoteDeletion:

        if version < Version(1, 0, 0):
            raise ValueError(f"{version} < Version(1, 0, 0)")

        kwargs = cls.prepare_kwargs_from_matchline(
            matchline=matchline,
            snote_class=MatchSnote,
            version=version,
        )

        return cls(**kwargs)


class MatchInsertionNote(BaseInsertionLine):
    def __init__(self, version: Version, note: MatchNote) -> None:

        super().__init__(
            version=version,
            note=note,
        )

    @classmethod
    def from_matchline(
        cls,
        matchline: str,
        version: Version = LATEST_VERSION,
    ) -> MatchInsertionNote:

        if version < Version(1, 0, 0):
            raise ValueError(f"{version} < Version(1, 0, 0)")

        kwargs = cls.prepare_kwargs_from_matchline(
            matchline=matchline,
            note_class=MatchNote,
            version=version,
        )

        return cls(**kwargs)


class MatchSustainPedal(BaseSustainPedalLine):
    def __init__(
        self,
        version: Version,
        time: int,
        value: int,
    ) -> None:

        super().__init__(
            version=version,
            time=time,
            value=value,
        )

    @classmethod
    def from_matchline(
        cls,
        matchline: str,
        version: Version = LATEST_VERSION,
        pos: int = 0,
    ) -> MatchSustainPedal:

        if version < Version(1, 0, 0):
            raise ValueError(f"{version} < Version(1, 0, 0)")

        kwargs = cls.prepare_kwargs_from_matchline(
            matchline=matchline,
            version=version,
            pos=pos,
        )

        if kwargs is None:
            raise MatchError("Input match line does not fit the expected pattern.")

        return cls(**kwargs)


class MatchSoftPedal(BaseSoftPedalLine):
    def __init__(
        self,
        version: Version,
        time: int,
        value: int,
    ) -> None:

        super().__init__(
            version=version,
            time=time,
            value=value,
        )

    @classmethod
    def from_matchline(
        cls,
        matchline: str,
        version: Version = LATEST_VERSION,
        pos: int = 0,
    ) -> MatchSoftPedal:

        if version < Version(1, 0, 0):
            raise ValueError(f"{version} < Version(1, 0, 0)")

        kwargs = cls.prepare_kwargs_from_matchline(
            matchline=matchline,
            version=version,
            pos=pos,
        )

        if kwargs is None:
            raise MatchError("Input match line does not fit the expected pattern.")

        return cls(**kwargs)
