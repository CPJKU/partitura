#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains base classes for Match lines and utilities for
parsing and formatting match lines.
"""
from __future__ import annotations

from typing import Callable, Tuple, Any, Optional, Union, List, Dict
import re

import numpy as np

from partitura.utils.music import (
    pitch_spelling_to_midi_pitch,
)

from collections import namedtuple

Version = namedtuple("Version", ["major", "minor", "patch"])

# General patterns
rational_pattern = re.compile(r"^(?P<numerator>[0-9]+)/(?P<denominator>[0-9]+)$")
double_rational_pattern = re.compile(
    r"^(?P<numerator>[0-9]+)/(?P<denominator>[0-9]+)/(?P<tuple_div>[0-9]+)$"
)
integer_pattern = re.compile(r"^(?P<integer>[0-9]+)$")
version_pattern = re.compile(
    r"^(?P<major>[0-9]+)\.(?P<minor>[0-9]+)\.(?P<patch>[0-9]+)"
)
attribute_list_pattern = re.compile(r"^\[(?P<attributes>.*)\]")

# For matchfiles before 1.0.0.
old_version_pattern = re.compile(r"^(?P<minor>[0-9]+)\.(?P<patch>[0-9]+)")


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
            r.append(" {0}: {1}".format(fn, self.__dict__[fn.lower()]))
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
    ):
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

    @property
    def DurationInBeats(self):
        return self.OffsetInBeats - self.OnsetInBeats

    @property
    def DurationSymbolic(self):
        # Duration should always be a FractionalSymbolicDuration
        return str(self.Duration)

    @property
    def MidiPitch(self):
        if isinstance(self.Octave, int):
            return pitch_spelling_to_midi_pitch(
                step=self.NoteName, octave=self.Octave, alter=self.Modifier
            )
        else:
            return None


## The following methods are helpers for interpretting and formatting
## information from match lines.


def interpret_version(version_string: str) -> Version:
    """
    Parse matchfile format version from a string. This method
    parses a string like "1.0.0" and returns a Version instance.

    Parameters
    ----------
    version_string : str
        The string containg the version. The version string should be
        in the form "{major}.{minor}.{patch}" or "{minor}.{patch}" for versions
        previous to 1.0.0. Incorrectly formatted strings
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

    # If using the first pattern fails, try with old version
    version_info = old_version_pattern.search(version_string)

    if version_info is not None:
        mi, pa = version_info.groups()
        version = Version(0, int(mi), int(pa))
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
    """
    Format a string from an integer

    Parameters
    ----------
    value : int
        The value to be converted to format as a string.

    Returns
    -------
    str
        The value formatted as a string.
    """
    return f"{value}"


def interpret_as_float(value: str) -> float:
    """
    Interpret value as a float

    Parameters
    ----------
    value : str
       The string to interpret as float.

    Returns
    -------
    int
        The value cast as an float.
    """
    return float(value)


def format_float(value: float) -> str:
    """
    Format a float as a string (with 4 digits of precision).

    Parameters
    ----------
    value : float
        The value to be converted to format as a string.

    Returns
    -------
    str
        The value formatted as a string.
    """
    return f"{value:.4f}"


def format_float_unconstrained(value: float) -> str:
    """
    Format a float as a string.

    Parameters
    ----------
    value : float
        The value to be converted to format as a string.

    Returns
    -------
    str
        The value formatted as a string.
    """
    return str(value)


def interpret_as_string(value: Any) -> str:
    """
    Interpret value as a string. This method is for completeness.

    Parameters
    ----------
    value : Any
       The value to be interpreted as a string.

    Returns
    -------
    int
        The string representation of the value.
    """
    return str(value)


def format_string(value: str) -> str:
    """
    Format a string as a string (for completeness ;).

    Parameters
    ----------
    value : int
        The value to be converted to format as a string.

    Returns
    -------
    str
        The value formatted as a string.
    """
    return value.strip()


class FractionalSymbolicDuration(object):
    """
    A class to represent symbolic duration information.

    Parameters
    ----------
    numerator : int
        The value of the numerator.
    denominator: int
        The denominator of the duration (whole notes = 1, half notes = 2, etc.)
    tuple_div : int (optional)
        Tuple divisor (for triplets, etc.). For example a single note in a quintuplet
        with a total duration of one quarter could be specified as
        `duration = FractionalSymbolicDuration(1, 4, 5)`.
    add_components : List[Tuple[int, int, Optional[int]]] (optional)
        additive components (to express durations like 1/4+1/16+1/32). The components
        are a list of tuples, each of which contains its own numerator, denominator
        and tuple_div (or None). To represent the components 1/16+1/32
        in the example above, this variable would look like
        `add_components = [(1, 16, None), (1, 32, None)]`.
    """

    def __init__(
        self,
        numerator: int,
        denominator: int = 1,
        tuple_div: Optional[int] = None,
        add_components: Optional[List[Tuple[int, int, Optional[int]]]] = None,
    ) -> None:

        self.numerator = numerator
        self.denominator = denominator
        self.tuple_div = tuple_div
        self.add_components = add_components
        self.bound_integers(1024)

    def _str(
        self,
        numerator: int,
        denominator: int,
        tuple_div: Optional[int],
    ) -> str:
        """
        Helper for representing an instance as a string.
        """
        if denominator == 1 and tuple_div is None:
            return str(numerator)
        else:
            if tuple_div is None:
                return "{0}/{1}".format(numerator, denominator)
            else:
                return "{0}/{1}/{2}".format(numerator, denominator, tuple_div)

    def bound_integers(self, bound: int) -> None:
        """
        Bound numerator and denominator
        """
        denominators = [
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            28,
            32,
            48,
            64,
            96,
            128,
        ]
        sign = np.sign(self.numerator) * np.sign(self.denominator)
        self.numerator = np.abs(self.numerator)
        self.denominator = np.abs(self.denominator)

        if self.numerator > bound or self.denominator > bound:
            val = float(self.numerator / self.denominator)
            dif = []
            for den in denominators:
                if np.round(val * den) > 0.9:
                    dif.append(np.abs(np.round(val * den) - val * den))
                else:
                    dif.append(np.abs(1 - val * den))

            difn = np.array(dif)
            min_idx = int(np.argmin(difn))

            self.denominator = denominators[min_idx]
            if int(np.round(val * self.denominator)) < 1:
                self.numerator = sign * 1
            else:
                self.numerator = sign * int(np.round(val * self.denominator))

    def __str__(self) -> str:
        """
        Represent an instance as a string.
        """
        if self.add_components is None:
            return self._str(self.numerator, self.denominator, self.tuple_div)
        else:
            r = [self._str(*i) for i in self.add_components]
            return "+".join(r)

    def __add__(
        self, sd: Union[FractionalSymbolicDuration, int]
    ) -> FractionalSymbolicDuration:
        """
        Define addition between FractionalSymbolicDuration instances.

        Parameters
        ----------
        sd : Union[FractionalSymbolicDuration, int]
            A FractionalSymbolicDuration instance or an integer to add
            to the current instance (self).

        Returns
        -------
        FractionalSymbolicDuration
            A new instance with the value equal to the sum
            of `sd` + `self`.
        """
        if isinstance(sd, int):
            sd = FractionalSymbolicDuration(sd, 1)

        dens = np.array([self.denominator, sd.denominator], dtype=int)
        new_den = np.lcm(dens[0], dens[1])
        a_mult = new_den // dens
        new_num = np.dot(a_mult, [self.numerator, sd.numerator])

        if self.add_components is None and sd.add_components is None:
            add_components = [
                (self.numerator, self.denominator, self.tuple_div),
                (sd.numerator, sd.denominator, sd.tuple_div),
            ]

        elif self.add_components is not None and sd.add_components is None:
            add_components = self.add_components + [
                (sd.numerator, sd.denominator, sd.tuple_div)
            ]
        elif self.add_components is None and sd.add_components is not None:
            add_components = [
                (self.numerator, self.denominator, self.tuple_div)
            ] + sd.add_components
        else:
            add_components = self.add_components + sd.add_components

        # Remove spurious components with 0 in the numerator
        add_components = [c for c in add_components if c[0] != 0]

        return FractionalSymbolicDuration(
            numerator=new_num,
            denominator=new_den,
            add_components=add_components,
        )

    def __radd__(
        self, sd: Union[FractionalSymbolicDuration, int]
    ) -> FractionalSymbolicDuration:
        return self.__add__(sd)

    def __float__(self) -> float:
        # Cast as float since the ability to return an instance of a strict
        # subclass of float is deprecated, and may be removed in a future
        # version of Python. (following a deprecation warning)
        return float(self.numerator / (self.denominator * (self.tuple_div or 1)))

    @classmethod
    def from_string(cls, string: str, allow_additions: bool = True):

        m = rational_pattern.match(string)
        m2 = double_rational_pattern.match(string)
        m3 = integer_pattern.match(string)
        if m:
            groups = m.groups()
            return cls(*[int(g) for g in groups])
        elif m2:
            groups = m2.groups()
            return cls(*[int(g) for g in groups])
        elif m3:
            return cls(numerator=int(m3.group("integer")))

        else:
            if allow_additions:
                parts = string.split("+")

                if len(parts) > 1:
                    iparts = [
                        cls.from_string(
                            i,
                            allow_additions=False,
                        )
                        for i in parts
                    ]

                    # to be replaced with isinstance(i,numbers.Number)
                    if all(type(i) in (int, float, cls) for i in iparts):
                        if any([isinstance(i, cls) for i in iparts]):
                            iparts = [
                                cls(i) if not isinstance(i, cls) else i for i in iparts
                            ]
                        return sum(iparts)

        raise ValueError(
            f"{string} cannot be interpreted as FractionalSymbolicDuration"
        )


def interpret_as_fractional(value: str) -> FractionalSymbolicDuration:
    return FractionalSymbolicDuration.from_string(value, allow_additions=True)


def format_fractional(value: FractionalSymbolicDuration) -> str:
    return str(value)


def interpret_as_list(value: str) -> List[str]:
    """
    Interpret string as list of values.

    Parameters
    ----------
    value: str

    Returns
    -------
    content_list : List[str]
    """
    content = attribute_list_pattern.search(value)

    if content is not None:
        # string includes square brackets
        vals_string = content.group("attributes")
        content_list = [v.strip() for v in vals_string.split(",")]

    else:
        # value is not inside square brackets
        content_list = [v.strip() for v in value.split(",")]

    return content_list


def format_list(value: List[Any]) -> str:
    formatted_string = f"[{','.join([str(v) for v in value])}]"
    return formatted_string
