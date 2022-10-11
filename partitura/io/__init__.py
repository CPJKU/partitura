from typing import Union

from .importmusicxml import load_musicxml
from .importmidi import load_score_midi, load_performance_midi
from .musescore import load_via_musescore
from .importmatch import load_match
from .importmei import load_mei
from .importkern import load_kern
from .importparangonada import load_parangonada_csv

from partitura.utils.misc import (
    deprecated_alias,
    deprecated_parameter,
    PathLike,
)

from partitura.score import Score, Part, merge_parts
from partitura.performance import Performance


class NotSupportedFormatError(Exception):
    pass


@deprecated_alias(score_fn="filename")
@deprecated_parameter("ensure_list")
def load_score(filename: PathLike, force_note_ids="keep") -> Score:
    """
    Load a score format supported by partitura. Currently the accepted formats
    are MusicXML, MIDI, Kern and MEI, plus all formats for which
    MuseScore has support import-support (requires MuseScore 3).

    Parameters
    ----------
    filename : str or file-like  object
        Filename of the score to parse, or a file-like object
    force_note_ids : (None, bool or "keep")
        When True each Note in the returned Part(s) will have a newly
        assigned unique id attribute. Existing note id attributes in
        the input file will be discarded. If 'keep', only notes without
        a note id will be assigned one. If None or False, the notes in the
        resulting Part(s) will have an id only if the input file has ids for
        the notes.

    Returns
    -------
    scr: :class:`partitura.score.Score`
        A score instance.
    """
    part = None

    # Catch exceptions
    exception_dictionary = dict()
    # Load MusicXML
    try:
        return load_musicxml(
            filename=filename,
            force_note_ids=force_note_ids,
        )
    except Exception as e:
        exception_dictionary["MusicXML"] = e
    # Load MIDI
    try:
        if (force_note_ids is None) or (not force_note_ids):
            assign_note_ids = False
        else:
            assign_note_ids = True
        return load_score_midi(
            filename=filename,
            assign_note_ids=assign_note_ids,
        )
    except Exception as e:
        exception_dictionary["MIDI"] = e
    # Load MEI
    try:
        return load_mei(filename=filename)
    except Exception as e:
        exception_dictionary["MEI"] = e
    # Load Kern
    try:
        return load_kern(
            filename=filename,
            force_note_ids=force_note_ids,
        )
    except Exception as e:
        exception_dictionary["Kern"] = e
    # Load MuseScore
    try:
        return load_via_musescore(
            filename=filename,
            force_note_ids=force_note_ids,
        )
    except Exception as e:
        exception_dictionary["MuseScore"] = e
    try:
        # Load the score information from a Matchfile
        _, _, part = load_match(
            filename=filename,
            create_score=True,
        )

    except Exception as e:
        exception_dictionary["matchfile"] = e
    if part is None:
        for score_format, exception in exception_dictionary.items():
            print(f"Error loading score as {score_format}:")
            print(exception)

        raise NotSupportedFormatError


def load_score_as_part(filename: PathLike) -> Part:
    """
    load part helper function:
    Load a score format supported by partitura and
    merge the result in a single part

    Parameters
    ----------
    filename : str or file-like  object
        Filename of the score to parse, or a file-like object

    Returns
    -------
    part: :class:`partitura.score.Part`
        A part instance.
    """
    scr = load_score(filename)
    part = merge_parts(scr.parts)
    return part


# alias
lp = load_score_as_part


@deprecated_alias(performance_fn="filename")
def load_performance(
    filename: PathLike,
    default_bpm: Union[float, int] = 120,
    merge_tracks: bool = False,
    first_note_at_zero: bool = False,
    pedal_threshold: int = 64,
) -> Performance:
    """
    Load a performance format supported by partitura. Currently the accepted formats
    are MIDI and matchfiles.

    Parameters
    ----------
    filename: str or file-like  object
        Filename of the score to parse, or a file-like object
    default_bpm : number, optional
        Tempo to use wherever the MIDI does not specify a tempo.
        Defaults to 120.
    merge_tracks: bool, optional
        For MIDI files, merges all tracks into a single track.
    first_note_at_zero: bool, optional
        Remove silence at the beginning, so that the first note (or
        first MIDI message, e.g., pedal) starts at time 0.
    pedal_threshold: int
        Threshold for the sustain pedal.

    Returns
    -------
    performance: :class:`partitura.performance.Performance`
        A `Performance` instance.

    TODO
    ----
    * Force loading scores as PerformedParts?
    """
    from partitura.utils.music import remove_silence_from_performed_part

    performance = None

    # Catch exceptions
    exception_dictionary = dict()
    try:
        performance = load_performance_midi(
            filename=filename,
            default_bpm=default_bpm,
            merge_tracks=merge_tracks,
        )

        # set threshold for sustain pedal
        performance[0].sustain_pedal_threshold = pedal_threshold

        if first_note_at_zero:
            remove_silence_from_performed_part(performance[0])

    except Exception as e:
        exception_dictionary["midi"] = e

    try:
        performance, _ = load_match(
            filename=filename,
            first_note_at_zero=first_note_at_zero,
            pedal_threshold=pedal_threshold,
        )
    except Exception as e:
        exception_dictionary["match"] = e

    if performance is None:
        for file_format, exception in exception_dictionary.items():
            print(f"Error loading score as {file_format}:")
            print(exception)
        raise NotSupportedFormatError

    return performance
