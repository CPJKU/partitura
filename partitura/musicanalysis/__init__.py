#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tools for music analysis.

"""

from .voice_separation import estimate_voices
from .key_identification import estimate_key
from .pitch_spelling import estimate_spelling
from .tonal_tension import estimate_tonaltension
from .meter import estimate_time
from .note_features import (
    list_note_feats_functions,
    make_note_features,
    make_note_feats,
    compute_note_array,
    full_note_array,
    make_rest_feats,
    make_rest_features,
)
from .performance_codec import encode_performance, decode_performance


__all__ = [
    "estimate_voices",
    "estimate_key",
    "estimate_spelling",
    "estimate_tonaltension",
    "estimate_time", 
    "list_note_feats_functions",
    "make_note_features",
    "make_rest_features",
    "encode_performance",
    "decode_performance",
    "compute_note_array",
    "full_note_array",
]
