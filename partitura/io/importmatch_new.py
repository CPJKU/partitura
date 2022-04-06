#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains methods for parsing matchfiles
"""
import re

from typing import Union

import numpy as np

MAX_VERSION = "1.0.0"

rational_pattern = re.compile(r"^([0-9]+)/([0-9]+)$")
double_rational_pattern = re.compile(r"^([0-9]+)/([0-9]+)/([0-9]+)$")


class MatchError(Exception):
    pass


class MatchLine(object):

    field_names = tuple()
    pattern = None

    def __str__(self):
        r = [self.__class__.__name__]
        for fn in self.field_names:
            r.append(" {0}: {1}".format(fn, self.__dict__[fn]))
        return "\n".join(r) + "\n"

    @property
    def matchline(self) -> str:
        raise NotImplementedError

    @classmethod
    def from_matchline(cls, matchline: str, *args, **kwargs):
        raise NotImplementedError

    def check_types(self):
        raise NotImplementedError


class MatchInfo(MatchLine):

    field_names = ('Attribute', 'Value')

    pattern = re.compile(r"info\(\s*([^,]+)\s*,\s*(.+)\s*\)\.")

    def __init__(self, attribute: str, value: Union[str, int]):
        self.attribute = attribute
        self.value = value

    @property
    def matchline(self):
        matchline = f"info({self.attribute},{self.value})."
        return matchline

    @classmethod
    def from_matchline(cls, matchline: str):
        re_info = cls.pattern.search(matchline)

        if re_info is not None:
            attribute, value_str = re_info.groups()

            if attribute in ('matchFileVersion', 'composer', 'piece'):
                value = value_str
            elif attribute in ('midiClockRate', 'midiClockUnites'):
                value = int(value_str)
            else:
                raise ValueError('Invalid attribute name!')

            return cls(attribute, value)


class MatchScoreProp(MatchLine):

    field_names = ('Attribute', 'Value', 'Bar', '')

    pattern = re.compile(r"info\(\s*([^,]+)\s*,\s*(.+)\s*\)\.")

    def __init__(self, attribute: str, value: str, bar: int, beat: float):
        self.attribute = attribute
        self.value = value
        self.bar = bar
        self.beat = beat

    @property
    def matchline(self):
        matchline = f"scoreprop({self.attribute},{self.value},{self.bar},{self.beat})."
        return matchline

    @classmethod
    def from_matchline(cls, matchline: str):
        pass



                 

    


def load_match(fn, create_part=False, pedal_threshold=64, first_note_at_zero=False):
    pass
