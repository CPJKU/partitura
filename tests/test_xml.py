"""

This file contains test functions for the partitura.musicxml module.

"""

import unittest
import os

from . import MUSICXML_PATH
#import partitura.musicxml as mxml
from partitura import load_musicxml
import partitura.score as score

class TestMusicXML(unittest.TestCase):

    def setUp(self):
        files = os.listdir(MUSICXML_PATH)
        self.partlists = [load_musicxml(os.path.join(MUSICXML_PATH, fn)) 
                         for fn in files]
        
    def test_partlist(self):
        for partlist in self.partlists:
            for part in partlist:
                self.assertTrue(isinstance(part, (score.ScorePart, score.PartGroup)),
                                'partlists should be either ScorePart or PartGroup')
                # self.assertTrue(hasattr(pg, 'scoreparts'), 'PartGroup should have an property score_parts')
                # self.assertTrue(isinstance(pg.scoreparts, list), 'PartGroup property score_parts should be a list')
                # self.assertGreater(len(pg.scoreparts), 0, 'Should have one or more ScoreParts')

            # scorepart = partgroup.score_parts[0]
        #     # print(score.pprint())
        #     measures = scorepart.timeline.get_all_of_type(score.Measure)
        #     for measure in measures:
        #         print(measure.end.t - measure.start.t)
        # self.assertEqual(1, 1, "Should be equal")

    # def test_sum_tuple(self):
    #     self.assertEqual(sum((1, 2, 2)), 6, "Should be 6")

if __name__ == '__main__':
    unittest.main()
    
    
# def get_notes_from_part(part):
#     """
#     print the notes of a score part

#     :param part: a ScorePart object

#     """

#     tl = part.timeline
#     assert len(tl.points) > 0

#     tl.lock()

#     div = 4.0
#     print(('part', part))
#     f = open('/tmp/out.txt', 'w')
#     for tp in tl.points:
#         notes = tp.starting_objects.get(score.Note, [])
#         tp_divs = tp.starting_objects.get(score.Divisions, [])

#         if len(tp_divs) > 0:
#             div = float(tp_divs[0].divs)

#         for n in notes:
#             print(('{0} {1} {2}'.format(n.start.t/div, n.end.t/div, n.midi_pitch)))
#             f.write('{0} {1} {2}\n'.format(
#                 n.start.t/div, n.end.t/div, n.midi_pitch))
#     f.close()


# def main():
#     parser = argparse.ArgumentParser("Get information from a MusicXML file")
#     parser.add_argument("file", help="MusicXML file")
#     args = parser.parse_args()

#     # the musicxml returns a list of score parts
#     structure = mxml.parse_musicxml(args.file)

#     if isinstance(structure, list):

#         for g in structure:
#             print((g.pprint()))

#     else:
#         structure.pprint()


# def test_load_file():
#     parser = argparse.ArgumentParser("Get information from a MusicXML file")
#     parser.add_argument("file", help="MusicXML file")
#     parser.add_argument("--get-parts", help="Output a list of parts",
#                         action="store_true", default=False)
#     args = parser.parse_args()

#     # filename
#     fn = args.file

#     flatten = not args.get_parts

#     # the musicxml returns a list of score parts
#     structure = mxml.parse_musicxml(fn)

#     parts = []
#     for part in structure.score_parts:
#         part.expand_grace_notes()
#         part.piece_name = os.path.splitext(os.path.basename(fn))[0]
#         parts.append(part)

#     score = []
#     for part in parts:
#         bm = part.beat_map
#         _score = np.array(
#             [(n.midi_pitch, bm(n.start.t), bm(n.end.t) - bm(n.start.t))
#              for n in part.notes],
#             dtype=[('pitch', 'i4'), ('onset', 'f4'), ('duration', 'f4')])

#         score += [_score]

#     if flatten:
#         if len(score) > 1:
#             score = np.vstack(score)
#         else:
#             score = score[0]

#     print(score)


# if __name__ == '__main__':
#     # main()
#     parser = argparse.ArgumentParser("Get information from a MusicXML file")
#     parser.add_argument("file", help="MusicXML file")
#     parser.add_argument("--get-parts", help="Output a list of parts",
#                         action="store_true", default=False)
#     args = parser.parse_args()

#     # filename
#     fn = args.file

#     flatten = not args.get_parts

#     # the musicxml returns a list of score parts
#     structure = mxml.parse_musicxml(fn)

#     parts = []
#     for part in structure.score_parts:
#         part.expand_grace_notes()
#         part.unfold_timeline()
#         part.piece_name = os.path.splitext(os.path.basename(fn))[0]
#         parts.append(part)

#     score = []
#     for part in parts:
#         bm = part.beat_map
#         _score = np.array(
#             [(n.midi_pitch, bm(n.start.t), bm(n.end.t) - bm(n.start.t))
#              for n in part.notes],
#             dtype=[('pitch', 'i4'), ('onset', 'f4'), ('duration', 'f4')])

#         score += [_score]

#     if flatten:
#         if len(score) > 1:
#             score = np.vstack(score)
#         else:
#             score = score[0]

#     print(score)
