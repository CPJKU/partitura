#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tools for music analysis.

"""

from .voice_separation import estimate_voices
from .key_identification import estimate_key
from .pitch_spelling import estimate_spelling
from .tonal_tension import estimate_tonaltension
from .note_features import make_note_feats, compute_note_array, full_note_array, make_rest_feats, make_rest_features
from .performance_codec import encode_performance, decode_performance
from .note_array_to_part import note_array_to_part

__all__ = [
    "estimate_voices",
    "estimate_key",
    "estimate_spelling",
    "estimate_tonaltension",
]
