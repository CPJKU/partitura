#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tools for estimating key signature, time signature,
pitch spelling, voice information, tonal tension, as well as methods for
deriving note-level features and performance encodings.
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
from .performance_features import make_performance_features
from .note_array_to_score import note_array_to_score


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
    "make_performance_features",
    "note_array_to_score",
]
