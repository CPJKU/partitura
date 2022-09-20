from .importmusicxml import load_musicxml
from .importmidi import load_score_midi, load_performance_midi
from .musescore import load_via_musescore
from .importmatch import load_match
from .importmei import load_mei
from .importkern import load_kern


class NotSupportedFormatError(Exception):
    pass


def load_score(score_fn, ensure_list=False, force_note_ids="keep"):
    """
    Load a score format supported by partitura. Currently the accepted formats
    are MusicXML, MIDI, Kern and MEI, plus all formats for which
    MuseScore has support import-support (requires MuseScore 3).

    Parameters
    ----------
    score_fn : str or file-like  object
        Filename of the score to parse, or a file-like object
    ensure_list : bool
        When True, return a list independent of how many part or
        group elements where created.
    force_note_ids : (None, bool or "keep")
        When True each Note in the returned Part(s) will have a newly
        assigned unique id attribute. Existing note id attributes in
        the input file will be discarded. If 'keep', only notes without
        a note id will be assigned one. If None or False, the notes in the
        resulting Part(s) will have an id only if the input file has ids for
        the notes.

    Returns
    -------
    part: list or Part
        A score part. If `ensure_list` the output will be a list.
    """
    part = None

    # Catch exceptions
    exception_dictionary = dict()
    # Load MusicXML
    try:
        return load_musicxml(
            xml=score_fn,
            ensure_list=ensure_list,
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
            fn=score_fn,
            assign_note_ids=assign_note_ids,
            ensure_list=ensure_list,
        )
    except Exception as e:
        exception_dictionary["MIDI"] = e
    # Load MEI
    try:
        return load_mei(mei_path=score_fn)
    except Exception as e:
        exception_dictionary["MEI"] = e
    # Load Kern
    try:
        return load_kern(
            kern_path=score_fn,
            ensure_list=ensure_list,
            force_note_ids=force_note_ids,
        )
    except Exception as e:
        exception_dictionary["Kern"] = e
    # Load MuseScore
    try:
        return load_via_musescore(
            fn=score_fn,
            force_note_ids=force_note_ids,
            ensure_list=ensure_list,
        )
    except Exception as e:
        exception_dictionary["MuseScore"] = e
    try:
        # Load the score information from a Matchfile
        _, _, part = load_match(score_fn, create_part=True)

        if ensure_list:
            return [part]
        else:
            return part
    except Exception as e:
        exception_dictionary["matchfile"] = e
    if part is None:
        for score_format, exception in exception_dictionary.items():
            print(f"Error loading score as {score_format}:")
            print(exception)

        raise NotSupportedFormatError


def load_performance(
    performance_fn,
    default_bpm=120,
    merge_tracks=False,
    first_note_at_zero=False,
    pedal_threshold=64,
):
    """
    Load a performance format supported by partitura. Currently the accepted formats
    are MIDI and matchfiles.

    Parameters
    ----------
    performance_fn: str or file-like  object
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
    performed_part: :class:`partitura.performance.PerformedPart`
        A PerformedPart instance.

    TODO
    ----
    * Force loading scores as PerformedParts?
    """
    from partitura.utils.music import remove_silence_from_performed_part

    performed_part = None

    # Catch exceptions
    exception_dictionary = dict()
    try:
        performance = load_performance_midi(
            performance_fn,
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
            fn=performance_fn,
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

    return performed_part
