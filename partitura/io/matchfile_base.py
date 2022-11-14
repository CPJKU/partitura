#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains base classes for Match lines and utilities for
parsing and formatting match lines.
"""
from __future__ import annotations

from typing import Callable, Tuple, Any
import re

from collections import namedtuple

Version = namedtuple("Version", ["major", "minor", "patch"])

# General patterns
rational_pattern = re.compile(r"^([0-9]+)/([0-9]+)$")
double_rational_pattern = re.compile(r"^([0-9]+)/([0-9]+)/([0-9]+)$")
version_pattern = re.compile(r"^([0-9]+)\.([0-9]+)\.([0-9]+)")

# For matchfiles before 1.0.0.
old_version_pattern = re.compile(r"^([0-9]+)\.([0-9]+)")


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
    field_names: Tuple[str]

    # type of the information in the fields
    field_types: Tuple[type]

    # Output pattern
    out_pattern: str

    # Regular expression to parse
    # information from a string.
    pattern = re.Pattern

    def __init__(self, version: Version) -> None:
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
        Generate matchline as a string
        """
        raise NotImplementedError

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

    def check_types(self) -> bool:
        """
        Check whether the values of the fields are of the correct type.

        Returns
        -------
        types_are_correct : bool
            True if the values of all fields in the match line have the
            correct type.
        """
        types_are_correct = all(
            [
                isinstance(getattr(self, field), field_type)
                for field, field_type in zip(self.field_names, self.field_types)
            ]
        )

        return types_are_correct


## The following classes define match lines that appear in all matchfile versions
## These classes need to be subclassed in the corresponding module for each version.


class BaseInfoLine(MatchLine):
    """
    Base class specifying global information lines.

    These lines have the general structure "info(<attribute>,<value>)."
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
    field_names: Tuple[str] = ("attribute", "value")

    out_pattern: str = "info({attribute},{value})."

    pattern: re.Pattern = re.compile(r"info\((?P<attribute>[^,]+),(?P<value>.+)\)\.")

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
        self.format_fun = dict(attribute=format_string, value=format_fun)
        self.attribute = attribute
        self.value = value

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
