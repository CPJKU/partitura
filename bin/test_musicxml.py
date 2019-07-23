#!/usr/bin/env python

import argparse

from partitura import load_musicxml, to_musicxml
import partitura.score as score

def test_musicxml(fn, schema=None):
    parts = load_musicxml(fn, schema)
    print(parts[0].pprint())
    p = parts[0]
    notes = p.list_all(score.Note)
    bm = p.quarter_map
    #print(notes)
    n0 = notes[0]
    # print(n0.duration)
    # print(bm([n0.start.t, n0.end.t]))
    # print('export:')
    out_fn = '/tmp/out.xml'
    to_musicxml(parts[0], out_fn)

def main():
    parser = argparse.ArgumentParser(description="Load and export a MusicXML file")
    parser.add_argument("xml", help="a MusicXML file")
    parser.add_argument("--schema", help="an XSD file specifying the MusicXML syntax")
    args = parser.parse_args()
    test_musicxml(args.xml, args.schema)
    

if __name__ == '__main__':
    main()
