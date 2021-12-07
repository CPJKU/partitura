from .importmusicxml import load_musicxml
from .importmidi import load_score_midi
from .musescore import load_via_musescore
from .importmatch import load_match


def load_score(score_fn, ensure_list=False, force_note_ids='keep'):
    """
    Load a score format supported by partitura. Currently the accepted formats
    are MusicXML and MIDI (native Python support), plus all formats for which
    MuseScore has support import-support (requires MuseScore 3)

    Parameters
    ----------
    score_fn : str
        Filename of the score to load.
    ensure_list : bool
        When True, return a list independent of how many part or
        group ementes where created.
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
    # Load MusicXML
    try:
        return load_musicxml(
            xml=score_fn,
            ensure_list=ensure_list,
            force_note_ids=force_note_ids
        )
    except:
        pass
    # Load MIDI
    try:
        if (force_note_ids is None) or (not force_note_ids):
            assign_note_ids = False
        else:
            assign_note_ids = True
        return load_score_midi(
            fn=score_fn,
            assign_note_ids=assign_note_ids,
            ensure_list=ensure_list
        )
    except:
        pass
    # Load MuseScore
    try:
        return load_via_musescore(
            fn=score_fn,
            force_note_ids=force_note_ids,
            ensure_list=ensure_list
        )
    except:
        pass
    try:
        # Load the score information from a Matchfile
        _, _, part = load_match(score_fn, create_part=True)

        if ensure_list:
            return [part]
        else:
            return part
    except:
        pass
    if part is None:
        raise ValueError('The score is not in one of the supported formats')
