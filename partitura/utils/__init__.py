#!/usr/bin/env python

from partitura.utils.generic import (
    ComparableMixin,
    ReplaceRefMixin,
    partition,
    iter_subclasses,
    iter_current_next,
    sorted_dict_items,
    PrettyPrintTree,
    find_nearest,
    add_field,
    show_diff,
    search,
    _OrderedSet,
)
from partitura.utils.music import (
    MIDI_BASE_CLASS,
    ALTER_SIGNS,
    find_tie_split,
    fifths_mode_to_key_name,
    key_name_to_fifths_mode,
    estimate_symbolic_duration,
    format_symbolic_duration,
    symbolic_to_numeric_duration,
    to_quarter_tempo,
    pitch_spelling_to_midi_pitch,
    estimate_clef_properties,
    ensure_pitch_spelling_format,
    ensure_notearray,
    compute_pianoroll,
    pianoroll_to_notearray,
    match_note_arrays,
    key_mode_to_int,
    remove_silence_from_performed_part,
    note_array_from_part_list,
    slice_notearray_by_time,
    note_array_from_part,
    note_array_from_note_list,
    get_time_units_from_note_array,
    update_note_ids_after_unfolding,
    note_name_to_pitch_spelling,
    note_name_to_midi_pitch,
    pitch_spelling_to_note_name
)

__all__ = [
    "key_name_to_fifths_mode",
    "fifths_mode_to_key_name",
    "key_mode_to_int",
    "pitch_spelling_to_midi_pitch",
    "compute_pianoroll",
    "pianoroll_to_notearray",
]
