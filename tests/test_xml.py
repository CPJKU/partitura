#!/usr/bin/env python

import argparse
import sys
import os
import numpy as np

import ofaiutils.data_handling.musicxml as mxml
import ofaiutils.data_handling.scoreontology as ont


def get_notes_from_part(part):
    """
    print the notes of a score part

    :param part: a ScorePart object

    """

    tl = part.timeline
    assert len(tl.points) > 0

    tl.lock()

    div = 4.0
    print(('part', part))
    f = open('/tmp/out.txt', 'w')
    for tp in tl.points:
        notes = tp.starting_objects.get(ont.Note, [])
        tp_divs = tp.starting_objects.get(ont.Divisions, [])

        if len(tp_divs) > 0:
            div = float(tp_divs[0].divs)

        for n in notes:
            print(('{0} {1} {2}'.format(n.start.t/div, n.end.t/div, n.midi_pitch)))
            f.write('{0} {1} {2}\n'.format(
                n.start.t/div, n.end.t/div, n.midi_pitch))
    f.close()


def main():
    parser = argparse.ArgumentParser("Get information from a MusicXML file")
    parser.add_argument("file", help="MusicXML file")
    args = parser.parse_args()

    # the musicxml returns a list of score parts
    structure = mxml.parse_music_xml(args.file)

    if isinstance(structure, list):

        for g in structure:
            print((g.pprint()))

    else:
        structure.pprint()


def test_load_file():
    parser = argparse.ArgumentParser("Get information from a MusicXML file")
    parser.add_argument("file", help="MusicXML file")
    parser.add_argument("--get-parts", help="Output a list of parts",
                        action="store_true", default=False)
    args = parser.parse_args()

    # filename
    fn = args.file

    flatten = not args.get_parts

    # the musicxml returns a list of score parts
    structure = mxml.parse_music_xml(fn)

    parts = []
    for part in structure.score_parts:
        part.expand_grace_notes()
        part.piece_name = os.path.splitext(os.path.basename(fn))[0]
        parts.append(part)

    score = []
    for part in parts:
        bm = part.beat_map
        _score = np.array(
            [(n.midi_pitch, bm(n.start.t), bm(n.end.t) - bm(n.start.t))
             for n in part.notes],
            dtype=[('pitch', 'i4'), ('onset', 'f4'), ('duration', 'f4')])

        score += [_score]

    if flatten:
        if len(score) > 1:
            score = np.vstack(score)
        else:
            score = score[0]

    print(score)


if __name__ == '__main__':
    # main()
    parser = argparse.ArgumentParser("Get information from a MusicXML file")
    parser.add_argument("file", help="MusicXML file")
    parser.add_argument("--get-parts", help="Output a list of parts",
                        action="store_true", default=False)
    args = parser.parse_args()

    # filename
    fn = args.file

    flatten = not args.get_parts

    # the musicxml returns a list of score parts
    structure = mxml.parse_music_xml(fn)

    parts = []
    for part in structure.score_parts:
        part.expand_grace_notes()
        part.unfold_timeline()
        part.piece_name = os.path.splitext(os.path.basename(fn))[0]
        parts.append(part)

    score = []
    for part in parts:
        bm = part.beat_map
        _score = np.array(
            [(n.midi_pitch, bm(n.start.t), bm(n.end.t) - bm(n.start.t))
             for n in part.notes],
            dtype=[('pitch', 'i4'), ('onset', 'f4'), ('duration', 'f4')])

        score += [_score]

    if flatten:
        if len(score) > 1:
            score = np.vstack(score)
        else:
            score = score[0]

    print(score)
