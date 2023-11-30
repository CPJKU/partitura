#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains definitions for Matchfile lines for version <1.0.0
"""
from __future__ import annotations

from collections import defaultdict

import re

from typing import Any, Callable, Tuple, Union, List, Dict, Optional

from partitura.io.matchfile_base import (
    MatchLine,
    MatchError,
    BaseInfoLine,
    BaseSnoteLine,
    BaseNoteLine,
    BaseSnoteNoteLine,
    BaseDeletionLine,
    BaseInsertionLine,
    BaseSustainPedalLine,
    BaseSoftPedalLine,
    BaseOrnamentLine,
)

from partitura.io.matchfile_utils import (
    Version,
    interpret_version,
    format_version,
    interpret_as_string,
    interpret_as_string_old,
    format_string,
    format_string_old,
    interpret_as_float,
    format_float,
    format_float_unconstrained,
    interpret_as_int,
    format_int,
    FractionalSymbolicDuration,
    format_fractional,
    format_fractional_rational,
    interpret_as_fractional,
    interpret_as_list,
    format_list,
    format_accidental_old,
    MatchTimeSignature,
    interpret_as_time_signature,
    format_time_signature,
    format_time_signature_list,
    MatchKeySignature,
    interpret_as_key_signature,
    format_key_signature_v0_1_0,
    format_key_signature_v0_3_0,
    format_key_signature_v0_3_0_list,
    get_kwargs_from_matchline,
)

from partitura.utils.music import (
    ALTER_SIGNS,
    pitch_spelling_to_midi_pitch,
    ensure_pitch_spelling_format,
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
    "piece": (interpret_as_string_old, format_string_old, str),
    "scoreFileName": (interpret_as_string_old, format_string_old, str),
    "scoreFilePath": (interpret_as_string_old, format_string_old, str),
    "midiFileName": (interpret_as_string_old, format_string_old, str),
    "midiFilename": (interpret_as_string_old, format_string_old, str),
    "midiFilePath": (interpret_as_string_old, format_string_old, str),
    "audioFileName": (interpret_as_string_old, format_string_old, str),
    "audioFilePath": (interpret_as_string_old, format_string_old, str),
    "audioFirstNote": (interpret_as_float, format_float_unconstrained, float),
    "audioLastNote": (interpret_as_float, format_float_unconstrained, float),
    "performer": (interpret_as_string_old, format_string_old, str),
    "composer": (interpret_as_string_old, format_string_old, str),
    "midiClockUnits": (interpret_as_int, format_int, int),
    "midiClockRate": (interpret_as_int, format_int, int),
    "approximateTempo": (interpret_as_float, format_float_unconstrained, float),
    "subtitle": (interpret_as_list, format_list, list),
    # "keySignature": (interpret_as_list, format_list, list),
    # "timeSignature": (
    #     interpret_as_fractional,
    #     format_fractional,
    #     (FractionalSymbolicDuration, list),
    # ),
    "tempoIndication": (interpret_as_list, format_list, list),
    "beatSubDivision": (interpret_as_list, format_list, list),
    "beatSubdivision": (interpret_as_list, format_list, list),
    "partSequence": (interpret_as_string, format_string, str),
    "mergedFrom": (interpret_as_list, format_list, list),
}

# INFO_LINE = defaultdict(lambda: default_infoline_attributes.copy())

INFO_LINE = {
    Version(0, 1, 0): {
        "keySignature": (
            interpret_as_key_signature,
            format_key_signature_v0_1_0,
            MatchKeySignature,
        ),
        "timeSignature": (
            interpret_as_time_signature,
            format_time_signature,
            MatchTimeSignature,
        ),
        **default_infoline_attributes,
    },
    Version(0, 2, 0): {
        "keySignature": (
            interpret_as_key_signature,
            format_key_signature_v0_1_0,
            MatchKeySignature,
        ),
        "timeSignature": (
            interpret_as_time_signature,
            format_time_signature,
            MatchTimeSignature,
        ),
        **default_infoline_attributes,
    },
    Version(0, 3, 0): {
        "keySignature": (
            interpret_as_key_signature,
            format_key_signature_v0_3_0_list,
            MatchKeySignature,
        ),
        "timeSignature": (
            interpret_as_time_signature,
            format_time_signature,
            MatchTimeSignature,
        ),
        **default_infoline_attributes,
    },
    Version(0, 4, 0): {
        "keySignature": (
            interpret_as_key_signature,
            format_key_signature_v0_3_0_list,
            MatchKeySignature,
        ),
        "timeSignature": (
            interpret_as_time_signature,
            format_time_signature_list,
            MatchTimeSignature,
        ),
        **default_infoline_attributes,
    },
    Version(0, 5, 0): {
        "keySignature": (
            interpret_as_key_signature,
            format_key_signature_v0_3_0_list,
            MatchKeySignature,
        ),
        "timeSignature": (
            interpret_as_time_signature,
            format_time_signature_list,
            MatchTimeSignature,
        ),
        **default_infoline_attributes,
    },
}


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
            raise ValueError("The version must be < 1.0.0")

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
            raise ValueError("The version must be < 1.0.0")

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


default_meta_attributes = {
    "timeSignature": (
        interpret_as_time_signature,
        format_time_signature,
        MatchTimeSignature,
    ),
    "keySignature": (
        interpret_as_key_signature,
        format_key_signature_v0_3_0,
        MatchKeySignature,
    ),
}
META_LINE = {
    Version(0, 3, 0): default_meta_attributes,
    Version(0, 4, 0): default_meta_attributes,
    Version(0, 5, 0): default_meta_attributes,
}


class MatchMeta(MatchLine):
    field_names = (
        "Attribute",
        "Value",
        "Measure",
        "TimeInBeats",
    )

    out_pattern = "meta({Attribute},{Value},{Measure},{TimeInBeats})."

    pattern = re.compile(
        r"meta\("
        r"(?P<Attribute>[^,]+),"
        r"(?P<Value>[^,]+),"
        r"(?P<Measure>[^,]+),"
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
        time_in_beats: float,
    ) -> None:
        if version >= Version(1, 0, 0):
            raise ValueError("The version must be < 1.0.0")

        super().__init__(version)

        self.field_types = (
            str,
            value_type,
            int,
            float,
        )

        self.format_fun = dict(
            Attribute=format_string,
            Value=format_fun,
            Measure=format_int,
            TimeInBeats=format_float_unconstrained,
        )

        # set class attributes
        self.Attribute = attribute
        self.Value = value
        self.Measure = measure
        self.TimeInBeats = time_in_beats

    @classmethod
    def from_matchline(
        cls,
        matchline: str,
        pos: int = 0,
        version: Version = LAST_VERSION,
    ) -> MatchMeta:
        """
        Create a new MatchMeta object from a string

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

        if version not in META_LINE:
            raise ValueError(f"{version} is not specified for this class.")

        match_pattern = cls.pattern.search(matchline, pos=pos)

        class_dict = META_LINE[version]

        if match_pattern is not None:
            (
                attribute,
                value_str,
                measure_str,
                time_in_beats_str,
            ) = match_pattern.groups()

            if attribute not in class_dict:
                raise ValueError(f"Attribute {attribute} is not specified in {version}")

            interpret_fun, format_fun, value_type = class_dict[attribute]

            value = interpret_fun(value_str)

            measure = interpret_as_int(measure_str)

            time_in_beats = interpret_as_float(time_in_beats_str)

            return cls(
                version=version,
                attribute=attribute,
                value=value,
                value_type=value_type,
                format_fun=format_fun,
                measure=measure,
                time_in_beats=time_in_beats,
            )

        else:
            raise MatchError("Input match line does not fit the expected pattern.")


SNOTE_LINE_Vgeq0_4_0 = dict(
    Anchor=format_string,
    NoteName=lambda x: str(x).upper(),
    Modifier=format_accidental_old,
    Octave=format_int,
    Measure=format_int,
    Beat=format_int,
    Offset=format_fractional,
    Duration=format_fractional,
    OnsetInBeats=format_float_unconstrained,
    OffsetInBeats=format_float_unconstrained,
    ScoreAttributesList=format_list,
)

SNOTE_LINE_Vlt0_3_0 = dict(
    Anchor=format_string,
    NoteName=lambda x: str(x).lower(),
    Modifier=format_accidental_old,
    Octave=format_int,
    Measure=format_int,
    Beat=format_int,
    Offset=format_fractional_rational,
    Duration=format_fractional_rational,
    OnsetInBeats=lambda x: f"{x:.5f}",
    OffsetInBeats=lambda x: f"{x:.5f}",
    ScoreAttributesList=format_list,
)

SNOTE_LINE = {
    Version(0, 5, 0): SNOTE_LINE_Vgeq0_4_0,
    Version(0, 4, 0): SNOTE_LINE_Vgeq0_4_0,
    Version(0, 3, 0): dict(
        Anchor=format_string,
        NoteName=lambda x: str(x).lower(),
        Modifier=format_accidental_old,
        Octave=format_int,
        Measure=format_int,
        Beat=format_int,
        Offset=format_fractional,
        Duration=format_fractional,
        OnsetInBeats=format_float_unconstrained,
        OffsetInBeats=format_float_unconstrained,
        ScoreAttributesList=format_list,
    ),
    Version(0, 2, 0): SNOTE_LINE_Vlt0_3_0,
    Version(0, 1, 0): SNOTE_LINE_Vlt0_3_0,
}


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
    ) -> None:
        if version not in SNOTE_LINE:
            raise ValueError(
                f"Unknown version {version}!. "
                f"Supported versions are {list(SNOTE_LINE.keys())}"
            )
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

        self.format_fun = SNOTE_LINE[version]

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


# Note lines for versions larger than 3.0
NOTE_LINE_Vge0_3_0 = {
    "field_names": (
        "Id",
        "NoteName",
        "Modifier",
        "Octave",
        "Onset",
        "Offset",
        "AdjOffset",
        "Velocity",
    ),
    "out_pattern": (
        "note({Id},[{NoteName},{Modifier}],{Octave},{Onset},{Offset},"
        "{AdjOffset},{Velocity})."
    ),
    "pattern": re.compile(
        r"note\((?P<Id>[^,]+),"
        r"\[(?P<NoteName>[^,]+),(?P<Modifier>[^,]+)\],"
        r"(?P<Octave>[^,]+),"
        r"(?P<Onset>[^,]+),"
        r"(?P<Offset>[^,]+),"
        r"(?P<AdjOffset>[^,]+),"
        r"(?P<Velocity>[^,]+)\)"
    ),
    "field_interpreters": {
        "Id": (interpret_as_string, format_string, str),
        "NoteName": (interpret_as_string, lambda x: str(x).upper(), str),
        "Modifier": (
            interpret_as_string,
            format_accidental_old,
            (int, type(None)),
        ),
        "Octave": (interpret_as_int, format_int, (int, type(None))),
        "Onset": (interpret_as_int, format_int, int),
        "Offset": (interpret_as_int, format_int, int),
        "AdjOffset": (interpret_as_int, format_int, int),
        "Velocity": (interpret_as_int, format_int, int),
    },
}

NOTE_LINE_Vlt0_3_0 = {
    "field_names": (
        "Id",
        "NoteName",
        "Modifier",
        "Octave",
        "Onset",
        "Offset",
        "Velocity",
    ),
    "out_pattern": (
        "note({Id},[{NoteName},{Modifier}],{Octave},{Onset},{Offset},{Velocity})."
    ),
    "pattern": re.compile(
        r"note\((?P<Id>[^,]+),"
        r"\[(?P<NoteName>[^,]+),(?P<Modifier>[^,]+)\],"
        r"(?P<Octave>[^,]+),"
        r"(?P<Onset>[^,]+),"
        r"(?P<Offset>[^,]+),"
        r"(?P<Velocity>[^,]+)\)"
    ),
    "field_interpreters": {
        "Id": (interpret_as_string, format_string, str),
        "NoteName": (interpret_as_string, lambda x: str(x).lower(), str),
        "Modifier": (
            interpret_as_string,
            format_accidental_old,
            (int, type(None)),
        ),
        "Octave": (interpret_as_int, format_int, (int, type(None))),
        "Onset": (interpret_as_float, lambda x: f"{x:.2f}", float),
        "Offset": (interpret_as_float, lambda x: f"{x:.2f}", float),
        "Velocity": (interpret_as_int, format_int, int),
    },
}


NOTE_LINE = {
    Version(0, 5, 0): NOTE_LINE_Vge0_3_0,
    Version(0, 4, 0): NOTE_LINE_Vge0_3_0,
    Version(0, 3, 0): {
        "field_names": (
            "Id",
            "NoteName",
            "Modifier",
            "Octave",
            "Onset",
            "Offset",
            "AdjOffset",
            "Velocity",
        ),
        "out_pattern": (
            "note({Id},[{NoteName},{Modifier}],{Octave},{Onset},{Offset},"
            "{AdjOffset},{Velocity})."
        ),
        "pattern": re.compile(
            r"note\((?P<Id>[^,]+),"
            r"\[(?P<NoteName>[^,]+),(?P<Modifier>[^,]+)\],"
            r"(?P<Octave>[^,]+),"
            r"(?P<Onset>[^,]+),"
            r"(?P<Offset>[^,]+),"
            r"(?P<AdjOffset>[^,]+),"
            r"(?P<Velocity>[^,]+)\)"
        ),
        "field_interpreters": {
            "Id": (interpret_as_string, format_string, str),
            "NoteName": (interpret_as_string, lambda x: str(x).lower(), str),
            "Modifier": (
                interpret_as_string,
                format_accidental_old,
                (int, type(None)),
            ),
            "Octave": (
                interpret_as_int,
                format_int,
                (int, type(None)),
            ),
            "Onset": (interpret_as_int, format_int, int),
            "Offset": (interpret_as_int, format_int, int),
            "AdjOffset": (interpret_as_int, format_int, int),
            "Velocity": (interpret_as_int, format_int, int),
        },
    },
    Version(0, 2, 0): NOTE_LINE_Vlt0_3_0,
    Version(0, 1, 0): NOTE_LINE_Vlt0_3_0,
}


class MatchNote(BaseNoteLine):
    def __init__(
        self,
        version: Version,
        id: str,
        note_name: str,
        modifier: int,
        octave: int,
        onset: int,
        offset: int,
        velocity: int,
        **kwargs,
    ) -> None:
        if version not in NOTE_LINE:
            raise ValueError(
                f"Unknown version {version}!. "
                f"Supported versions are {list(NOTE_LINE.keys())}"
            )

        step, alter, octave = ensure_pitch_spelling_format(note_name, modifier, octave)
        midi_pitch = pitch_spelling_to_midi_pitch(step, alter, octave)

        super().__init__(
            version=version,
            id=id,
            midi_pitch=midi_pitch,
            onset=onset,
            offset=offset,
            velocity=velocity,
        )

        self.field_names = NOTE_LINE[version]["field_names"]
        self.field_types = tuple(
            NOTE_LINE[version]["field_interpreters"][fn][2] for fn in self.field_names
        )
        self.format_fun = dict(
            [
                (fn, NOTE_LINE[version]["field_interpreters"][fn][1])
                for fn in self.field_names
            ]
        )

        self.pattern = NOTE_LINE[version]["pattern"]
        self.out_pattern = NOTE_LINE[version]["out_pattern"]

        self.NoteName = step
        self.Modifier = alter
        self.Octave = octave
        self.AdjOffset = offset

        if "adj_offset" in kwargs:
            self.AdjOffset = kwargs["adj_offset"]

    @property
    def AdjDuration(self) -> float:
        return self.AdjOffset - self.Onset

    @classmethod
    def from_matchline(
        cls,
        matchline: str,
        pos: int = 0,
        version: Version = LAST_VERSION,
    ) -> MatchNote:
        if version >= Version(1, 0, 0):
            raise ValueError(f"{version} >= Version(1, 0, 0)")

        kwargs = get_kwargs_from_matchline(
            matchline=matchline,
            pattern=NOTE_LINE[version]["pattern"],
            field_names=NOTE_LINE[version]["field_names"],
            class_dict=NOTE_LINE[version]["field_interpreters"],
            pos=pos,
        )

        if kwargs is not None:
            return cls(version=version, **kwargs)

        else:
            raise MatchError("Input match line does not fit the expected pattern.")


class MatchSnoteNote(BaseSnoteNoteLine):
    def __init__(
        self,
        version: Version,
        snote: MatchSnote,
        note: MatchNote,
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
        version: Version = LAST_VERSION,
    ) -> MatchSnoteNote:
        if version >= Version(1, 0, 0):
            raise ValueError(f"{version} >= Version(1, 0, 0)")

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
        version: Version = LAST_VERSION,
    ) -> MatchSnoteDeletion:
        if version >= Version(1, 0, 0):
            raise ValueError(f"{version} >= Version(1, 0, 0)")

        kwargs = cls.prepare_kwargs_from_matchline(
            matchline=matchline,
            snote_class=MatchSnote,
            version=version,
        )

        return cls(**kwargs)


class MatchSnoteTrailingScore(MatchSnoteDeletion):
    out_pattern = "{SnoteLine}-trailing_score_note."
    identifier_pattern = re.compile(r"-trailing_score_note\.")

    def __init__(self, version: Version, snote: MatchSnote) -> None:
        super().__init__(version=version, snote=snote)
        self.pattern = re.compile(
            rf"{self.snote.pattern.pattern}-trailing_score_note\."
        )


class MatchSnoteNoPlayedNote(MatchSnoteDeletion):
    out_pattern = "{SnoteLine}-no_played_note."
    identifier_pattern = re.compile(r"-no_played_note\.")

    def __init__(self, version: Version, snote: MatchSnote) -> None:
        super().__init__(version=version, snote=snote)
        self.pattern = re.compile(rf"{self.snote.pattern.pattern}-no_played_note\.")


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
        version: Version = LAST_VERSION,
    ) -> MatchInsertionNote:
        if version >= Version(1, 0, 0):
            raise ValueError(f"{version} >= Version(1, 0, 0)")

        kwargs = cls.prepare_kwargs_from_matchline(
            matchline=matchline,
            note_class=MatchNote,
            version=version,
        )

        return cls(**kwargs)


class MatchHammerBounceNote(MatchInsertionNote):
    out_pattern = "hammer_bounce-{NoteLine}"
    identifier_pattern = re.compile(r"hammer_bounce-")

    def __init__(self, version: Version, note: MatchNote) -> None:
        super().__init__(version=version, note=note)
        self.pattern = re.compile(f"hammer_bounce-{self.note.pattern.pattern}")


class MatchTrailingPlayedNote(MatchInsertionNote):
    out_pattern = "trailing_played_note-{NoteLine}"
    identifier_pattern = re.compile(r"trailing_played_note-")

    def __init__(self, version: Version, note: MatchNote) -> None:
        super().__init__(version=version, note=note)
        self.pattern = re.compile(f"trailing_played_note-{self.note.pattern.pattern}")


class MatchTrillNote(BaseOrnamentLine):
    out_pattern = "trill({Anchor})-{NoteLine}"
    ornament_pattern: re.Pattern = re.compile(r"trill\((?P<Anchor>[^\)]*)\)-")

    def __init__(
        self,
        version: Version,
        anchor: str,
        note: BaseNoteLine,
    ) -> None:
        super().__init__(
            version=version,
            anchor=anchor,
            note=note,
        )

    @classmethod
    def from_matchline(
        cls,
        matchline: str,
        version: Version = LAST_VERSION,
    ) -> MatchTrillNote:
        if version >= Version(1, 0, 0):
            raise ValueError(f"{version} >= Version(1, 0, 0)")

        anchor_pattern = cls.ornament_pattern.search(matchline)

        if anchor_pattern is None:
            raise MatchError("Input match line does not fit the expected pattern.")
        note = MatchNote.from_matchline(matchline, version=version)

        return cls(
            version=version,
            note=note,
            anchor=interpret_as_string(anchor_pattern.group("Anchor")),
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
        version: Version = LAST_VERSION,
        pos: int = 0,
    ) -> MatchSustainPedal:
        if version >= Version(1, 0, 0):
            raise ValueError(f"{version} less than 1.0.0")

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
        version: Version = LAST_VERSION,
        pos: int = 0,
    ) -> MatchSoftPedal:
        if version >= Version(1, 0, 0):
            raise ValueError(f"{version} should be less than 1.0.0")

        kwargs = cls.prepare_kwargs_from_matchline(
            matchline=matchline,
            version=version,
            pos=pos,
        )

        if kwargs is None:
            raise MatchError("Input match line does not fit the expected pattern.")

        return cls(**kwargs)


FROM_MATCHLINE_METHODS = [
    MatchSnoteNote.from_matchline,
    MatchSnoteDeletion.from_matchline,
    MatchSnoteTrailingScore.from_matchline,
    MatchSnoteNoPlayedNote.from_matchline,
    MatchInsertionNote.from_matchline,
    MatchHammerBounceNote.from_matchline,
    MatchTrailingPlayedNote.from_matchline,
    MatchTrillNote.from_matchline,
    MatchSustainPedal.from_matchline,
    MatchSoftPedal.from_matchline,
    MatchInfo.from_matchline,
    MatchMeta.from_matchline,
]


def parse_matchline(line: str, version: Version) -> Optional[MatchLine]:
    def parse(mlt: MatchLine) -> Optional[MatchLine]:
        matchline = None
        try:
            matchline = mlt.from_matchline(line, version)
        except MatchError:
            pass

        return matchline

    matchline = None
    if matchline.startswith("info"):
        return parse(MatchInfo)

    if line.startswith("meta"):
        return parse(MatchMeta)

    if line.startswith("snote"):
        for mlt in [
            MatchSnoteNote,
            MatchSnoteDeletion,
            MatchSnoteTrailingScore,
            MatchSnoteNoPlayedNote,
        ]:
            matchline = parse(mlt)
            if matchline is not None:
                return matchline

    if line.startswith("insertion"):
        return parse(MatchInsertionNote)

    if line.startswith("trailing_played"):
        return parse(MatchTrailingPlayedNote)

    if line.startswith("hammer_bounce"):
        return parse(MatchHammerBounceNote)

    if line.startswith("sustain"):
        return parse(MatchSustainPedal)

    if line.startswith("soft"):
        return parse(MatchSoftPedal)

    if line.startswith("trill"):
        return parse(MatchTrillNote)

    return matchline
