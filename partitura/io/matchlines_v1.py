#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains definitions for Matchfile lines for version >1.0.0
"""

from __future__ import annotations

import re

import numpy as np

from typing import Any, Callable, Tuple, Union, List, Optional

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
    VirtualSnoteLine,
    VirtualNoteLine,
    VirtualSnoteNoteLine,
    VirtualSnoteVirtualNoteLine,
    BaseSnoteNoteLine,
    BaseSnoteVirtualNoteLine,
    BaseDeletionLine,
    BaseInsertionLine,
    BaseOrnamentLine,
    BaseSustainPedalLine,
    BaseSoftPedalLine,
    BaseSectionLine,
    BaseOmittedSectionLine,
)

from partitura.io.matchfile_utils import (
    interpret_version,
    format_version,
    interpret_as_string,
    format_string,
    format_accidental_old,
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
    MatchTimeSignature,
    interpret_as_time_signature,
    format_time_signature,
    MatchKeySignature,
    interpret_as_key_signature,
    format_key_signature_v1_0_0,
    to_snake_case,
    get_kwargs_from_matchline,
    MatchTempoIndication,
    interpret_as_tempo_indication,
    format_tempo_indication,
)

# Define current version of the match file format
LATEST_MAJOR_VERSION = 1
LATEST_MINOR_VERSION = 1
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

INFO_LINE[Version(1, 1, 0)] = INFO_LINE[Version(1, 0, 0)]

INFO_ATTRIBUTE_EQUIVALENCES = dict(
    midiFilename="midiFileName",
)


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

    @classmethod
    def from_instance(
        cls, instance: BaseInfoLine, version: Version = LATEST_VERSION
    ) -> MatchInfo:
        if version < Version(1, 0, 0):
            raise ValueError(f"{version} < Version(1, 0, 0)")

        if not isinstance(instance, BaseInfoLine):
            raise ValueError("`instance` needs to be a subclass of `BaseInfoLine`")

        class_dict = INFO_LINE[version]

        if instance.Attribute in INFO_ATTRIBUTE_EQUIVALENCES:
            attr = INFO_ATTRIBUTE_EQUIVALENCES[instance.Attribute]
        else:
            attr = instance.Attribute

        if attr not in class_dict:
            raise ValueError(f"Attribute {attr} is not specified in {version}")

        interpret_fun, format_fun, value_type = class_dict[attr]

        value = instance.Value

        if attr == "subtitle" and isinstance(value, list):
            value = "" if len(value) == 0 else str(value)
        return cls(
            version=version,
            attribute=attr,
            value=value,
            value_type=value_type,
            format_fun=format_fun,
        )


SCOREPROP_LINE = {
    Version(1, 0, 0): {
        "timeSignature": (
            interpret_as_time_signature,
            format_time_signature,
            MatchTimeSignature,
        ),
        "keySignature": (
            interpret_as_key_signature,
            format_key_signature_v1_0_0,
            MatchKeySignature,
        ),
        "tempoIndication": (
            interpret_as_tempo_indication,
            format_tempo_indication,
            MatchTempoIndication,
        ),
        "beatSubDivision": (interpret_as_list_int, format_list, list),
        "directions": (interpret_as_list, format_list, list),
    }
}

SCOREPROP_LINE[Version(1, 1, 0)] = SCOREPROP_LINE[Version(1, 0, 0)]

SCOREPROP_ATTRIBUTE_EQUIVALENCES = dict(
    beatSubdivision="beatSubDivision",
)


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
        r"(?P<Value>.+),"
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
    ) -> MatchScoreProp:
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

    @classmethod
    def from_instance(
        cls,
        instance: MatchLine,
        version: Version = LATEST_VERSION,
        measure: Optional[int] = None,
        beat: Optional[int] = None,
        offset: Optional[FractionalSymbolicDuration] = None,
        time_in_beats: Optional[float] = None,
    ) -> MatchScoreProp:
        if version < Version(1, 0, 0):
            raise ValueError(f"{version} < Version(1, 0, 0)")

        if not isinstance(instance, MatchLine):
            raise ValueError("`instance` needs to be a subclass of `MatchLine`")

        # ensure that at least the basic attributes are in the field names of the match line
        if not (
            "Attribute" in instance.field_names and "Value" in instance.field_names
        ):
            raise ValueError(
                "`instance` must contain at least 'Attribute', 'Value' and 'TimeInBeats'"
            )

        class_dict = SCOREPROP_LINE[version]

        if not (
            instance.Attribute in class_dict
            or instance.Attribute in SCOREPROP_ATTRIBUTE_EQUIVALENCES
        ):
            raise ValueError(
                f"Attribute {instance.Attribute} is not specified in {version}"
            )

        if instance.Attribute in SCOREPROP_ATTRIBUTE_EQUIVALENCES:
            attr = SCOREPROP_ATTRIBUTE_EQUIVALENCES[instance.Attribute]
        else:
            attr = instance.Attribute
        interpret_fun, format_fun, value_type = class_dict[attr]

        return cls(
            version=version,
            attribute=attr,
            value=instance.Value,
            value_type=value_type,
            format_fun=format_fun,
            measure=getattr(instance, "Measure", measure if measure is not None else 1),
            beat=getattr(instance, "Beat", beat if beat is not None else 1),
            offset=getattr(
                instance,
                "Offset",
                offset if offset is not None else FractionalSymbolicDuration(0),
            ),
            time_in_beats=getattr(
                instance,
                "TimeInBeats",
                time_in_beats if time_in_beats is not None else 0.0,
            ),
        )


SECTION_LINE = {
    Version(1, 0, 0): {
        "StartInBeatsUnfolded": (interpret_as_float, format_float, float),
        "EndInBeatsUnfolded": (interpret_as_float, format_float, float),
        "StartInBeatsOriginal": (interpret_as_float, format_float, float),
        "EndInBeatsOriginal": (interpret_as_float, format_float, float),
        "RepeatEndType": (interpret_as_list, format_list, list),
    },
    Version(1, 1, 0): {
        "id": (interpret_as_string, format_string, str),
        "StartInBeatsUnfolded": (interpret_as_float, format_float, float),
        "EndInBeatsUnfolded": (interpret_as_float, format_float, float),
        "StartInBeatsOriginal": (interpret_as_float, format_float, float),
        "EndInBeatsOriginal": (interpret_as_float, format_float, float),
        "StartInPerfTime": (interpret_as_int, format_int, int),
        "EndInPerfTime": (interpret_as_int, format_int, int),
        "SectionAttrList": (interpret_as_list, format_list, list),
    },
}


class MatchSection(BaseSectionLine):
    """
    Class for specifiying structural information (i.e., sections).

    For version 1.0.0:
    section(StartInBeatsUnfolded,EndInBeatsUnfolded,StartInBeatsOriginal,EndInBeatsOriginal,RepeatEndType).

    For version 1.1.0:
    section(id, StartInBeatsUnfolded,EndInBeatsUnfolded,StartInBeatsOriginal,EndInBeatsOriginal,StartInPerfTime,EndInPerfTime,SectionAttrList).

    Parameters
    ----------
    version: Version,
    start_in_beats_unfolded: float,
    end_in_beats_unfolded: float,
    start_in_beats_original: float,
    end_in_beats_original: float,
    repeat_end_type: List[str] (for version 1.0.0, optional)
    id: str (for version 1.1.0, optional)
    start_in_perf_time: int (for version 1.1.0, optional)
    end_in_perf_time: int (for version 1.1.0, optional)
    section_attr_list: List[str] (for version 1.1.0, optional)
    """

    pattern_v1_0_0 = re.compile(
        r"section\("
        r"(?P<StartInBeatsUnfolded>[^,]+),"
        r"(?P<EndInBeatsUnfolded>[^,]+),"
        r"(?P<StartInBeatsOriginal>[^,]+),"
        r"(?P<EndInBeatsOriginal>[^,]+),"
        r"\[(?P<RepeatEndType>.*)\]\)."
    )

    pattern_v1_1_0 = re.compile(
        r"section\("
        r"(?P<id>[^,]+),"
        r"(?P<StartInBeatsUnfolded>[^,]+),"
        r"(?P<EndInBeatsUnfolded>[^,]+),"
        r"(?P<StartInBeatsOriginal>[^,]+),"
        r"(?P<EndInBeatsOriginal>[^,]+),"
        r"(?P<StartInPerfTime>[^,]+),"
        r"(?P<EndInPerfTime>[^,]+),"
        r"\[(?P<SectionAttrList>.*)\]\)."
    )

    def __init__(
        self,
        version: Version,
        start_in_beats_unfolded: float,
        end_in_beats_unfolded: float,
        start_in_beats_original: float,
        end_in_beats_original: float,
        repeat_end_type: Optional[List[str]] = None,
        id: Optional[str] = None,
        start_in_perf_time: Optional[int] = None,
        end_in_perf_time: Optional[int] = None,
        section_attr_list: Optional[List[str]] = None,
    ) -> None:
        if version not in SECTION_LINE:
            raise ValueError(
                f"Unknown version {version}!. "
                f"Supported versions are {list(SECTION_LINE.keys())}"
            )
        super().__init__(version)

        self.field_names = tuple(SECTION_LINE[version].keys())
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

        if version == Version(1, 0, 0):
            self.RepeatEndType = repeat_end_type
            self.out_pattern = (
                "section({StartInBeatsUnfolded},"
                "{EndInBeatsUnfolded},{StartInBeatsOriginal},"
                "{EndInBeatsOriginal},{RepeatEndType})."
            )
            self.pattern = self.pattern_v1_0_0
        elif version >= Version(1, 1, 0):
            self.id = id
            self.StartInPerfTime = start_in_perf_time
            self.EndInPerfTime = end_in_perf_time
            self.SectionAttrList = section_attr_list
            self.out_pattern = (
                "section({id},{StartInBeatsUnfolded},"
                "{EndInBeatsUnfolded},{StartInBeatsOriginal},"
                "{EndInBeatsOriginal},{StartInPerfTime},"
                "{EndInPerfTime},{SectionAttrList})."
            )
            self.pattern = self.pattern_v1_1_0

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

        class_dict = SECTION_LINE[version]

        # Select the appropriate pattern based on version
        if version > Version(1, 0, 0):
            pattern = cls.pattern_v1_1_0
        else:
            pattern = cls.pattern_v1_0_0

        match_pattern = pattern.search(matchline, pos=pos)

        if match_pattern is not None:
            kwargs = dict(
                [
                    (to_snake_case(fn), class_dict[fn][0](match_pattern.group(fn)))
                    for fn in class_dict.keys()
                ]
            )

            return cls(version=version, **kwargs)

        else:
            raise MatchError("Input match line does not fit the expected pattern.")


class MatchOmittedSection(BaseOmittedSectionLine):
    """
    Class for specifiying sections of the score that were not performed. The
    score notes in these sections are not listed in the match file.

    omittedSection(id,StartInBeatsUnfolded,EndInBeatsUnfolded,StartInBeatsOriginal,EndInBeatsOriginal,SectionAttrList).

    Parameters
    ----------
    version: Version,
    id: str,
    start_in_beats_unfolded: float,
    end_in_beats_unfolded: float,
    start_in_beats_original: float,
    end_in_beats_original: float,
    section_attr_list: List[str]
    """

    field_names = (
        "id",
        "StartInBeatsUnfolded",
        "EndInBeatsUnfolded",
        "StartInBeatsOriginal",
        "EndInBeatsOriginal",
        "SectionAttrList",
    )

    out_pattern = (
        "omittedSection({id},{StartInBeatsUnfolded},"
        "{EndInBeatsUnfolded},{StartInBeatsOriginal},"
        "{EndInBeatsOriginal},{SectionAttrList})."
    )
    pattern = re.compile(
        r"omittedSection\("
        r"(?P<id>[^,]+),"
        r"(?P<StartInBeatsUnfolded>[^,]+),"
        r"(?P<EndInBeatsUnfolded>[^,]+),"
        r"(?P<StartInBeatsOriginal>[^,]+),"
        r"(?P<EndInBeatsOriginal>[^,]+),"
        r"\[(?P<SectionAttrList>.*)\]\)."
    )

    def __init__(
        self,
        version: Version,
        id: str,
        start_in_beats_unfolded: float,
        end_in_beats_unfolded: float,
        start_in_beats_original: float,
        end_in_beats_original: float,
        section_attr_list: List[str],
    ) -> None:
        if version < Version(1, 1, 0):
            raise ValueError(f"{version} < Version(1, 1, 0)")
        super().__init__(version)

        self.field_types = tuple(
            SECTION_LINE[version][fn][2] for fn in self.field_names
        )
        self.format_fun = dict(
            [(fn, ft[1]) for fn, ft in SECTION_LINE[version].items()]
        )

        self.id = id
        self.StartInBeatsUnfolded = start_in_beats_unfolded
        self.EndInBeatsUnfolded = end_in_beats_unfolded
        self.StartInBeatsOriginal = start_in_beats_original
        self.EndInBeatsOriginal = end_in_beats_original
        self.SectionAttrList = section_attr_list

    @classmethod
    def from_matchline(
        cls,
        matchline: str,
        pos: int = 0,
        version: Version = LATEST_VERSION,
    ) -> MatchOmittedSection:
        match_pattern = cls.pattern.search(matchline, pos=pos)

        if match_pattern is None:
            raise MatchError("Input match line does not fit the expected pattern.")

        if version < Version(1, 1, 0):
            raise ValueError(f"{version} < Version(1, 1, 0)")

        class_dict = SECTION_LINE[version]

        kwargs = dict(
            [
                (to_snake_case(fn), class_dict[fn][0](match_pattern.group(fn)))
                for fn in cls.field_names
            ]
        )

        return cls(version=version, **kwargs)


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

STIME_LINE[Version(1, 1, 0)] = STIME_LINE[Version(1, 0, 0)]


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
        Modifier=format_accidental_old,
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

    @classmethod
    def from_instance(
        cls,
        instance: BaseSnoteLine,
        version: Version = LATEST_VERSION,
    ) -> MatchSnote:
        if version < Version(1, 0, 0):
            raise ValueError(f"{version} < Version(1, 0, 0)")

        if not isinstance(instance, BaseSnoteLine):
            raise ValueError("`instance` needs to be a subclass of `BaseSnoteLine`")

        return cls(
            version=version,
            anchor=instance.Anchor,
            note_name=instance.NoteName,
            modifier=instance.Modifier,
            octave=instance.Octave,
            measure=instance.Measure,
            beat=instance.Beat,
            offset=instance.Offset,
            duration=instance.Duration,
            onset_in_beats=instance.OnsetInBeats,
            offset_in_beats=instance.OffsetInBeats,
            score_attributes_list=instance.ScoreAttributesList,
        )


class MatchVirtualSnote(VirtualSnoteLine):
    format_fun = dict(
        Anchor=format_string,
        AttributesList=format_list,
    )

    def __init__(
        self,
        version: Version,
        anchor: str,
        attributes_list: List[str],
    ) -> None:
        if version < Version(1, 1, 0):
            raise MatchError(
                f"Version {version} is not supported by this 1.1.0 parser."
            )
        super().__init__(
            version=version,
            anchor=anchor,
            attributes_list=attributes_list,
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

        if version < Version(1, 1, 0):
            raise MatchError(
                f"Version {version} is not supported by this 1.1.0 parser."
            )

        kwargs = cls.prepare_kwargs_from_matchline(
            matchline=matchline,
            pos=pos,
        )

        return cls(version=version, **kwargs)

    @classmethod
    def from_instance(
        cls,
        instance: VirtualSnoteLine,
        version: Version = LATEST_VERSION,
    ) -> MatchSnote:
        if version < Version(1, 1, 0):
            raise MatchError(
                f"Version {version} is not supported by this 1.1.0 parser."
            )

        if not isinstance(instance, VirtualSnoteLine):
            raise ValueError("`instance` needs to be a subclass of `VirtualSnoteLine`")

        return cls(
            version=version,
            anchor=instance.Anchor,
            attributes_list=instance.AttributesList,
        )


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

NOTE_LINE[Version(1, 1, 0)] = NOTE_LINE[Version(1, 0, 0)]


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

    # the word boundary avoids matching the "note(" in "virtualSnote("
    pattern = re.compile(
        r"\bnote\((?P<Id>[^,]+),"
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

    @classmethod
    def from_instance(
        cls,
        instance: BaseNoteLine,
        version: Version = LATEST_VERSION,
    ) -> MatchNote:
        if version < Version(1, 0, 0):
            raise ValueError(f"{version} < Version(1, 0, 0)")

        if not isinstance(instance, BaseNoteLine):
            raise ValueError("`instance` needs to be a subclass of `BaseNoteLine`")

        if instance.version < Version(1, 0, 0):
            return cls(
                version=version,
                id=instance.Id,
                midi_pitch=instance.MidiPitch,
                onset=int(np.round(instance.Onset)),
                offset=int(np.round(instance.Offset)),
                velocity=instance.Velocity,
                channel=1,
                track=0,
            )


class MatchVirtualPNote(VirtualNoteLine):
    field_names = ("Id",)

    out_pattern = "virtualPnote({Id})."

    # files written with partitura 1.9.0 have a trailing comma after the Id
    pattern = re.compile(r"virtualPnote\((?P<Id>[^,)]+),?\)")

    def __init__(
        self,
        version: Version,
        id: str,
    ) -> None:
        if version not in NOTE_LINE:
            raise ValueError(
                f"Unknown version {version}!. "
                f"Supported versions are {list(NOTE_LINE.keys())}"
            )

        super().__init__(
            version=version,
            id=id,
        )

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
        if version < Version(1, 1, 0):
            raise ValueError(f"{version} < Version(1, 1, 0)")

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

    @classmethod
    def from_instance(
        cls,
        instance: BaseNoteLine,
        version: Version = LATEST_VERSION,
    ) -> MatchNote:
        if version < Version(1, 0, 0):
            raise ValueError(f"{version} < Version(1, 0, 0)")

        if not isinstance(instance, BaseNoteLine):
            raise ValueError("`instance` needs to be a subclass of `BaseNoteLine`")

        if instance.version < Version(1, 0, 0):
            return cls(
                version=version,
                id=instance.Id,
                midi_pitch=instance.MidiPitch,
                onset=int(np.round(instance.Onset)),
                offset=int(np.round(instance.Offset)),
                velocity=instance.Velocity,
                channel=1,
                track=0,
            )


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

    @classmethod
    def from_instance(
        cls, instance: BaseSnoteNoteLine, version: Version
    ) -> MatchSnoteNote:
        if version < Version(1, 0, 0):
            raise ValueError(f"{version} < Version(1, 0, 0)")

        if not isinstance(instance, BaseSnoteNoteLine):
            raise ValueError("`instance` needs to be a subclass of `BaseSnoteNoteLine`")

        return cls(
            version=version,
            snote=MatchSnote.from_instance(instance.snote, version=version),
            note=MatchNote.from_instance(instance.note, version=version),
        )


class MatchSnoteVirtualNote(BaseSnoteVirtualNoteLine):
    def __init__(
        self,
        version: Version,
        snote: BaseSnoteLine,
        note: VirtualNoteLine,
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
    ) -> MatchSnoteVirtualNote:
        if version < Version(1, 1, 0):
            raise MatchError(
                f"Version {version} is not supported by this 1.1.0 parser."
            )

        kwargs = cls.prepare_kwargs_from_matchline(
            matchline=matchline,
            snote_class=MatchSnote,
            note_class=MatchVirtualPNote,
            version=version,
        )

        return cls(**kwargs)

    @classmethod
    def from_instance(
        cls, instance: BaseSnoteVirtualNoteLine, version: Version
    ) -> MatchSnoteVirtualNote:
        if version < Version(1, 1, 0):
            raise MatchError(
                f"Version {version} is not supported by this 1.1.0 parser."
            )

        if not isinstance(instance, BaseSnoteVirtualNoteLine):
            raise ValueError(
                "`instance` needs to be a subclass of `BaseSnoteVirtualNoteLine`"
            )

        return cls(
            version=version,
            snote=MatchSnote.from_instance(instance.snote, version=version),
            note=MatchVirtualPNote.from_instance(instance.note, version=version),
        )


class MatchVirtualSnoteNote(VirtualSnoteNoteLine):
    def __init__(
        self,
        version: Version,
        snote: VirtualSnoteLine,
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
    ) -> MatchVirtualSnoteNote:
        if version < Version(1, 1, 0):
            raise MatchError(
                f"Version {version} is not supported by this 1.1.0 parser."
            )

        kwargs = cls.prepare_kwargs_from_matchline(
            matchline=matchline,
            snote_class=MatchVirtualSnote,
            note_class=MatchNote,
            version=version,
        )

        return cls(**kwargs)

    @classmethod
    def from_instance(
        cls, instance: VirtualSnoteNoteLine, version: Version
    ) -> MatchVirtualSnoteNote:
        if version < Version(1, 1, 0):
            raise MatchError(
                f"Version {version} is not supported by this 1.1.0 parser."
            )

        if not isinstance(instance, VirtualSnoteNoteLine):
            raise ValueError(
                "`instance` needs to be a subclass of `VirtualSnoteNoteLine`"
            )

        return cls(
            version=version,
            snote=MatchVirtualSnote.from_instance(instance.snote, version=version),
            note=MatchNote.from_instance(instance.note, version=version),
        )


class MatchVirtualSnoteVirtualNote(VirtualSnoteVirtualNoteLine):
    def __init__(
        self,
        version: Version,
        snote: VirtualSnoteLine,
        note: VirtualNoteLine,
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
    ) -> MatchVirtualSnoteVirtualNote:
        if version < Version(1, 1, 0):
            raise MatchError(
                f"Version {version} is not supported by this 1.1.0 parser."
            )

        kwargs = cls.prepare_kwargs_from_matchline(
            matchline=matchline,
            snote_class=MatchVirtualSnote,
            note_class=MatchVirtualPNote,
            version=version,
        )

        return cls(**kwargs)

    @classmethod
    def from_instance(
        cls, instance: VirtualSnoteVirtualNoteLine, version: Version
    ) -> MatchVirtualSnoteVirtualNote:
        if version < Version(1, 1, 0):
            raise MatchError(
                f"Version {version} is not supported by this 1.1.0 parser."
            )

        if not isinstance(instance, VirtualSnoteVirtualNoteLine):
            raise ValueError(
                "`instance` needs to be a subclass of `VirtualSnoteVirtualNoteLine`"
            )

        return cls(
            version=version,
            snote=MatchVirtualSnote.from_instance(instance.snote, version=version),
            note=MatchVirtualPNote.from_instance(instance.note, version=version),
        )


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

    @classmethod
    def from_instance(
        cls, instance: BaseDeletionLine, version: Version
    ) -> MatchSnoteNote:
        if version < Version(1, 0, 0):
            raise ValueError(f"{version} < Version(1, 0, 0)")

        if not isinstance(instance, BaseDeletionLine):
            raise ValueError("`instance` needs to be a subclass of `BaseDeletionLine`")

        return cls(
            version=version,
            snote=MatchSnote.from_instance(instance.snote, version=version),
        )


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

    @classmethod
    def from_instance(
        cls, instance: BaseInsertionLine, version: Version
    ) -> MatchInsertionNote:
        if version < Version(1, 0, 0):
            raise ValueError(f"{version} < Version(1, 0, 0)")

        if not isinstance(instance, BaseInsertionLine):
            raise ValueError("`instance` needs to be a subclass of `BaseInsertionLine`")

        return cls(
            version=version,
            note=MatchNote.from_instance(instance.note, version=version),
        )


class MatchOrnamentNote(BaseOrnamentLine):
    field_names = (
        "Anchor",
        "OrnamentType",
    )
    field_types = (
        str,
        list,
    )
    format_fun = dict(Anchor=format_string, OrnamentType=format_list)
    out_pattern = "ornament({Anchor},{OrnamentType})-{NoteLine}"
    ornament_pattern: re.Pattern = re.compile(
        r"ornament\((?P<Anchor>[^\)]*),\[(?P<OrnamentType>.*)\]\)-"
    )

    def __init__(
        self,
        version: Version,
        anchor: str,
        ornament_type: List[str],
        note: BaseNoteLine,
    ) -> None:
        super().__init__(
            version=version,
            anchor=anchor,
            note=note,
        )
        self.OrnamentType = ornament_type

    @property
    def matchline(self) -> str:
        return self.out_pattern.format(
            Anchor=self.format_fun[0]["Anchor"](self.Anchor),
            OrnamentType=self.format_fun[0]["OrnamentType"](self.OrnamentType),
            NoteLine=self.note.matchline,
        )

    @classmethod
    def from_matchline(
        cls,
        matchline: str,
        version: Version = LATEST_VERSION,
    ) -> MatchOrnamentNote:
        if version < Version(1, 0, 0):
            raise ValueError(f"{version} < Version(1, 0, 0)")

        anchor_pattern = cls.ornament_pattern.search(matchline)

        if anchor_pattern is None:
            raise MatchError("Input match line does not fit the expected pattern.")
        note = MatchNote.from_matchline(matchline, version=version)

        return cls(
            version=version,
            note=note,
            anchor=interpret_as_string(anchor_pattern.group("Anchor")),
            ornament_type=interpret_as_list(anchor_pattern.group("OrnamentType")),
        )

    @classmethod
    def from_instance(
        cls, instance: BaseOrnamentLine, version: Version
    ) -> MatchOrnamentNote:
        if version < Version(1, 0, 0):
            raise ValueError(f"{version} < Version(1, 0, 0)")

        if not isinstance(instance, BaseOrnamentLine):
            raise ValueError("`instance` needs to be a subclass of `BaseOrnamentLine`")

        return cls(
            version=version,
            anchor=instance.Anchor,
            note=MatchNote.from_instance(instance.note, version=version),
            ornament_type=(
                ["trill"]
                if instance.version < Version(1, 0, 0)
                else instance.OrnamentType
            ),
        )


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

    @classmethod
    def from_instance(
        cls, instance: BaseSustainPedalLine, version: Version
    ) -> MatchOrnamentNote:
        if version < Version(1, 0, 0):
            raise ValueError(f"{version} < Version(1, 0, 0)")

        if not isinstance(instance, BaseSustainPedalLine):
            raise ValueError(
                "`instance` needs to be a subclass of `BaseSustainPedalLine`"
            )

        return cls(
            version=version,
            time=int(instance.Time),
            value=int(instance.Value),
        )


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

    @classmethod
    def from_instance(
        cls,
        instance: BaseSoftPedalLine,
        version: Version,
    ) -> MatchOrnamentNote:
        if version < Version(1, 0, 0):
            raise ValueError(f"{version} < Version(1, 0, 0)")

        if not isinstance(instance, BaseSoftPedalLine):
            raise ValueError("`instance` needs to be a subclass of `BaseSoftPedalLine`")

        return cls(
            version=version,
            time=int(instance.Time),
            value=int(instance.Value),
        )


FROM_MATCHLINE_METHODS = [
    MatchSnoteNote.from_matchline,
    MatchSnoteVirtualNote.from_matchline,
    MatchVirtualSnoteNote.from_matchline,
    MatchVirtualSnoteVirtualNote.from_matchline,
    MatchSnoteDeletion.from_matchline,
    MatchInsertionNote.from_matchline,
    MatchOrnamentNote.from_matchline,
    MatchSustainPedal.from_matchline,
    MatchSoftPedal.from_matchline,
    MatchInfo.from_matchline,
    MatchScoreProp.from_matchline,
    MatchSection.from_matchline,
    MatchOmittedSection.from_matchline,
    MatchStimePtime.from_matchline,
]


## Helper methods to build the corresponding line for each parameter


def make_info(version: Version, attribute: str, value: Any) -> MatchInfo:
    """
    Get version line from attributes
    """

    if attribute == "matchFileVersion":
        if version != value:
            raise ValueError(
                f"The specified version ({version}) should be the same as "
                f"`value` ({value})"
            )
    class_dict = INFO_LINE[version]

    _, format_fun, dtype = class_dict[attribute]

    ml = MatchInfo(
        version=version,
        attribute=attribute,
        value=value,
        value_type=dtype,
        format_fun=format_fun,
    )

    return ml


def make_scoreprop(
    version: Version,
    attribute: str,
    value: Any,
    measure: int,
    beat: int,
    offset: FractionalSymbolicDuration,
    time_in_beats: float,
) -> MatchScoreProp:
    class_dict = SCOREPROP_LINE[version]

    _, format_fun, dtype = class_dict[attribute]

    ml = MatchScoreProp(
        version=version,
        attribute=attribute,
        value=value,
        value_type=dtype,
        format_fun=format_fun,
        measure=measure,
        beat=beat,
        offset=offset,
        time_in_beats=time_in_beats,
    )

    return ml


def make_section(
    version: Version,
    start_in_beats_unfolded: float,
    end_in_beats_unfolded: float,
    start_in_beats_original: float,
    end_in_beats_original: float,
    repeat_end_type: Union[str, List[str]],
) -> MatchSection:
    ml = MatchSection(
        version=version,
        start_in_beats_unfolded=start_in_beats_unfolded,
        start_in_beats_original=start_in_beats_original,
        end_in_beats_unfolded=end_in_beats_unfolded,
        end_in_beats_original=end_in_beats_original,
        repeat_end_type=(
            [repeat_end_type] if isinstance(repeat_end_type, str) else repeat_end_type
        ),
    )
    return ml


def to_v1(matchline: MatchLine, version: Version = LATEST_VERSION) -> MatchLine:
    """
    Convert matchline to version 1_x_x

    Parameters
    ----------
    matchline : MatchLine
        Matchline to be converted
    version: Version
        Target version. Default is `LATEST_VERSION`

    Returns
    -------
    MatchLine
         A new matchline with the equivalent of the input matchline in
         the specified version.
    """
    from partitura.io.matchlines_v0 import MatchMeta

    if isinstance(matchline, BaseInfoLine):
        if (
            matchline.Attribute in INFO_LINE[version]
            or matchline.Attribute in INFO_ATTRIBUTE_EQUIVALENCES
        ):
            return MatchInfo.from_instance(instance=matchline, version=version)

        if (
            matchline.Attribute in SCOREPROP_LINE[version]
            or matchline.Attribute in SCOREPROP_ATTRIBUTE_EQUIVALENCES
        ):
            return MatchScoreProp.from_instance(instance=matchline, version=version)

    if isinstance(matchline, MatchMeta):
        return MatchScoreProp.from_instance(instance=matchline, version=version)

    if isinstance(matchline, BaseSnoteNoteLine):
        return MatchSnoteNote.from_instance(instance=matchline, version=version)

    if isinstance(matchline, BaseInsertionLine):
        return MatchInsertionNote.from_instance(instance=matchline, version=version)

    if isinstance(matchline, BaseDeletionLine):
        return MatchSnoteDeletion.from_instance(instance=matchline, version=version)

    if isinstance(matchline, BaseOrnamentLine):
        return MatchOrnamentNote.from_instance(instance=matchline, version=version)

    if isinstance(matchline, BaseSustainPedalLine):
        return MatchSustainPedal.from_instance(instance=matchline, version=version)

    if isinstance(matchline, BaseSoftPedalLine):
        return MatchSoftPedal.from_instance(instance=matchline, version=version)

    else:
        print(matchline.matchline)
        raise MatchError(f"No equivalent line in version {version}")
