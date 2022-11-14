#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains definitions for Matchfile lines for version >1.0.0
"""

from partitura.io.matchfile_base import MatchLine, Version

# Define current version of the match file format
CURRENT_MAJOR_VERSION = 1
CURRENT_MINOR_VERSION = 0
CURRENT_PATCH_VERSION = 0

CURRENT_VERSION = Version(
    CURRENT_MAJOR_VERSION,
    CURRENT_MINOR_VERSION,
    CURRENT_PATCH_VERSION,
)


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

    def __init__(self, version: Version = CURRENT_VERSION, **kwargs) -> None:
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
