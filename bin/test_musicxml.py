#!/usr/bin/env python

import argparse

from partitura import load_musicxml, to_musicxml
import partitura.score as score
from partitura.musicxml import xml_to_notearray

def test_musicxml(fn, validate=False):
    parts = load_musicxml(fn, validate)
    # part = parts[0]
    part = next(score.iter_parts(parts))
    # print(part.pprint())
    notes = part.notes
    print(len(notes))
    tnotes = [n for n in notes if n.tie_prev is None]
    for note in tnotes:
        print(note.id, note.duration, note.duration_tied)

        
    # measures = part.list_all(score.Measure)
    # for measure in measures:
    #     print( measure.number, [measure, measure.start, measure.end])

    # repeats = part.list_all(score.Repeat)
    # for repeat in repeats:
    #     print('repeat', repeat.start.t, repeat.end.t)
    # endings = part.list_all(score.Ending)
    # for ending in endings:
    #     print('ending', ending.start.t, ending.end.t)

    # svs = part.make_score_variants()
    # for sv in svs:
    #     print(sv.segment_times)
    #     tl = sv.create_variant_timeline()
    # #     print(sv)
    # #     tl.test()
    # #     break
    # return
    # part.timeline.test()
    
    timeline = part.unfold_timeline_maximal()
    print(timeline.last_point)

    try:
        part.timeline = part.unfold_timeline_maximal()
    except NotImplementedError:
        raise
        pass
    
    # print(part.pprint())
    # part.timeline.test()

    # for measure in measures:
    #     print( measure.number, [measure, measure.start, measure.end])

    # ending no repeat:
    # ending 1 start
    # 0-96 0-12 96-180 

    # notes = p.list_all(score.Note)
    # bm = part.quarter_map

    out_fn = '/tmp/out.xml'
    to_musicxml(part, out_fn)

def main():
    parser = argparse.ArgumentParser(description="Load and export a MusicXML file")
    parser.add_argument("xml", help="a MusicXML file")
    # parser.add_argument("--schema", help="an XSD file specifying the MusicXML syntax")
    parser.add_argument("--validate", action='store_true', help="validate MusicXML against the 3.1 XSD specification", default=False)
    args = parser.parse_args()
    test_musicxml(args.xml, args.validate)
    # print(xml_to_notearray(args.xml, validate=args.validate))

if __name__ == '__main__':
    main()
