#!/usr/bin/env python

import argparse
import numpy as np
import re

import partitura.match as match

key_patt = re.compile('\[([a-g])(.*),(.*)\]')


def summarize_match_file(m):
    """
    Display some information from a Match file

    :param m: a MatchFile object

    """

    # print header info
    for line in m.info():
        print(('  {0}\t{1}'.format(line.Attribute, line.Value).expandtabs(20)))

    # print time sig info
    print('Time signatures:')
    for t, (n, d) in m.time_signatures:
        print((' {0}/{1} at beat {2}'.format(n, d, t)))


def get_note_info(snote, note, div=480.0, rate=50000.0):
    """
    Get note information from a `Snote` instance

    Parameters
    ----------
    snote : `match.Snote`
        Instance of `Snote` from the MatchFile` containing
        the score note information
    note : `match.Note`
        Performed note information
    div = float
        Time division

    Returns
    -------
    note_name : str
        Name of the note
    octave : int
        Octave of the note
    pitch : int
        MIDI pitch
    onset_b : float
        Onset in beats
    duration_b : float
        Duration in beats
    velocity : int
        Performed MIDI velocity
    onset_s : float
        Performed onset in seconds
    duration_s : float
        Performed duration in seconds
    soprano : bool
        If the note is part of the melody
    """
    step = str(snote.NoteName).upper()
    modifier = str(snote.Modifier)
    onset_b = snote.OnsetInBeats
    offset_b = snote.OffsetInBeats
    octave = snote.Octave

    pitch = match.pitch_name_2_midi_PC(modifier, step, octave)[0]

    # The soprano check (for database differrences):
    soprano = 0
    if 's' in snote.ScoreAttributesList:
        soprano = 1

    # TODO:
    # * Check if this case is correct for bb and x
    if modifier != 'n':
        step += modifier

    velocity = note.Velocity

    onset_s = float(note.Onset) * rate / (div * 1e6)
    offset_s = float(note.Offset) * rate / (div * 1e6)
    return (step, octave, pitch,
            onset_b, offset_b - onset_b, velocity,
            onset_s, offset_s - onset_s, soprano)


def get_score_from_match(fn, version='auto'):
    """
    Get

    Parameters
    ----------
    fn : string
        Path to Match File.
    """
    mf = match.MatchFile(fn, version)

    div = float(mf.info('midiClockUnits'))
    rate = float(mf.info('midiClockRate'))
    key = str(mf.info('keySignature'))

    try:
        # Searching for the
        root = str(key_patt.search(key).group(1)).upper()
        modifier = str(key_patt.search(key).group(2))
        if modifier != 'n':
            root += modifier

        mode = str(key_patt.search(key).group(3))

    except:
        # TODO: Handle keys correctly
        print('Key in wrong format... Using C major')
        root = 'C'
        mode = 'major'

    timesig = str(mf.info('timeSignature'))

    note_info = []
    for snote, note in mf.note_pairs:
        note_info.append(get_note_info(snote, note, div, rate))

    note_info = np.array(note_info,
                         dtype=[('notes', 'S8'),
                                ('octave', 'i4'),
                                ('pitch', 'i4'),
                                ('onset', 'f4'),
                                ('duration', 'f4'),
                                ('velocity', 'f4'),
                                ('p_onset', 'f4'),
                                ('p_duration', 'f4'),
                                ('soprano', 'i4')])

    return note_info, [root, mode], mf, timesig  # , tempo


def main():
    """
    Illustrate some functionality of the match module

    """
    parser = argparse.ArgumentParser("Get information from a Matchfile file")
    parser.add_argument("file", help="Match file")
    args = parser.parse_args()

    m = match.MatchFile(args.file)

    summarize_match_file(m)
    # get_notes_from_match(m)


def test_dir():
    import os
    import glob
    parser = argparse.ArgumentParser("Get information from a Matchfile file")
    parser.add_argument("dir", help="Match file")
    parser.add_argument('--version', default='auto')
    args = parser.parse_args()

    matchfiles = glob.glob(os.path.join(args.dir, '*.match'))

    for fn in matchfiles:
        print(os.path.basename(fn))
        note_info, key, mf, timesig = get_score_from_match(
            fn, version=args.version)

        print(key)


if __name__ == '__main__':
    main()
