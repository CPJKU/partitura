import os.path
import re
import partitura.score as score
from joblib import Parallel, delayed


from partitura.utils.music import LABEL_DURS
import numpy as np


class KernGlobalPart(score.Part):
    def __init__(self, doc_name, part_id, qdivs):
        super(KernGlobalPart, self).__init__(doc_name, part_id, quarter_duration=qdivs)
        self.SIGN_TO_ACC = {
            "n": 0,
            "#": 1,
            "s": 1,
            "ss": 2,
            "x": 2,
            "##": 2,
            "###": 3,
            "b": -1,
            "f": -1,
            "bb": -2,
            "ff": -2,
            "bbb": -3,
            "-": None,
        }

        self.KERN_NOTES = {
            'C': ("C", 3),
            'D': ('D', 3),
            'E': ('E', 3),
            'F': ('F', 3),
            'G': ('G', 3),
            'A': ('A', 3),
            'B': ('B', 3),
            'c': ("C", 4),
            'd': ('D', 4),
            'e': ('E', 4),
            'f': ('F', 4),
            'g': ('G', 4),
            'a': ('A', 4),
            'b': ('B', 4)
        }

        self.KERN_DURS = {
            # "long": "long",
            # "breve": "breve",
            1: "whole",
            2: "half",
            4: "quarter",
            8: "eighth",
            16: "16th",
            32: "32nd",
            64: "64th",
            128: "128th",
            256: "256th",
        }

class KernParserPart(KernGlobalPart):
    def __init__(self, stream, init_pos, doc_name, part_id, qdivs):
        super(KernParserPart, self).__init__(doc_name, part_id, qdivs)
        self.position = init_pos
        self.process(stream)
        self.slur_dict = dict()
        self.tie_dict = dict()

    def process(self, line):
        staff = None
        for index, el in enumerate(line):
            if el.startswith("*staff"):
                self.staff = eval(el[len("*staff"):])
            elif el.startswith("*"):
                if self.staff == None:
                    self.staff = 1
                self._handle_glob_attr(el)
            elif el == ".":
                pass
            elif el.startswith("="):
                self._handle_bar(el)
            elif " " in el:
                self._handle_chord(el, index)
            elif "r" in el:
                self._handle_rest(el, "r-" + str(index))
            else:
                self._handle_note(el, "n-" + str(index))
        self._handle_slurs()
        self._handle_ties()

    # TODO handle ties
    def _handle_ties(self):
        pass

    # TODO handle slurs
    def _handle_slurs(self):
        pass

    def _handle_metersig(self, metersig):
        m = metersig[2:]
        numerator, denominator = map(eval, m.split("/"))
        new_time_signature = score.TimeSignature(numerator, denominator)
        self.add(new_time_signature, self.position)

    def _handle_keysig(self, element):
        keysig_el = element[2:]
        fifths = 0
        for c in keysig_el:
            if c == "#":
                fifths += 1
            if c == "b":
                fifths -= 1
        # TODO retrieve the key mode
        mode = "major"
        new_key_signature = score.KeySignature(fifths, mode)
        self.add(new_key_signature, self.position)

    def _compute_clef_octave(self, dis, dis_place):
        if dis is not None:
            sign = -1 if dis_place == "below" else 1
            octave = sign * int(int(dis) / 8)
        else:
            octave = 0
        return octave

    def _handle_clef(self, element):
        # handle the case where we have clef information
        # TODO Compute Clef Octave
        new_clef = score.Clef(self.staff, element[5], int(element[6]), 0)
        self.add(new_clef, self.position)

    def _handle_rest(self, el, rest_id):
        # find duration info
        _, duration, ntype = re.split('(\d+)', el)
        symbolic_duration = self.KERN_DURS[eval(duration)]
        duration = self.inv_quarter_map([LABEL_DURS[symbolic_duration]])
        # create rest
        rest = score.Rest(
            id=rest_id,
            voice=1,
            staff=1,
            symbolic_duration=symbolic_duration,
            articulations=None,
        )
        # add rest to the part
        self.add(rest, self.position, self.position + duration)
        # return duration to update the position in the layer
        self.position += duration

    def _handle_note(self, note, note_id):
        # TODO handle brackets
        if note.startswith("[") or note.startswith("{"):
            note = note[1:]
        _, duration, ntype = re.split('(\d+)', note)
        is_dotted = ntype.startswith(".")
        d = eval(duration)
        if is_dotted:
            symbolic_duration = self.KERN_DURS[d]
            ntype = ntype[1:]
        else:
            symbolic_duration = self.KERN_DURS[d]
        step, octave = self.KERN_NOTES[ntype[0]]
        if octave == 4:
            octave += ntype.count(step) - 1
        elif octave == 3:
            octave -= ntype.count(step) - 1
        alter = ntype.count('#') - ntype.count("-")
        # find if it's grace
        grace_attr = "q" in ntype or "Q" in ntype
        if not grace_attr:
            # create normal note
            note = score.Note(
                step=step,
                octave=octave,
                alter=alter,
                id=note_id,
                voice=1,
                staff=self.staff,
                symbolic_duration=symbolic_duration,
                articulations=None,  # TODO : add articulation
            )
        else:
            # create grace note
            if grace_attr == "unacc":
                grace_type = "acciaccatura"
            elif grace_attr == "acc":
                grace_type = "appoggiatura"
            else:  # unknow type
                grace_type = "grace"
            note = score.GraceNote(
                grace_type=grace_type,
                step=step,
                octave=octave,
                alter=alter,
                id=note_id,
                voice=1,
                staff=self.staff,
                symbolic_duration=symbolic_duration,
                articulations=None,  # TODO : add articulation
            )
        qdivs = self._quarter_durations[0]
        duration = LABEL_DURS[symbolic_duration] * qdivs
        self.add(note, self.position, self.position + duration)
        self.position += duration

    def _handle_chord(self, chord, id):
        notes = chord.split()
        pos = self.position
        for i, note_el in enumerate(notes):
            id = "c-" + str(i) + "-" + str(id)
            self.position = pos
            self._handle_note(note_el, id)

    def _handle_bar(self, el):
        pass

    def _handle_glob_attr(self, el):
        if el.startswith("*clef"):
            self._handle_clef(el)
        elif el.startswith("*k"):
            self._handle_keysig(el)
        elif el.startswith("*M"):
            self._handle_metersig(el)
        elif el == "*-":
            print("Reached the end of the stream.")


class KernParser():
    def __init__(self, document, doc_name, parallel=True, n_jobs=2):
        self.document = document
        self.doc_name = doc_name
        self.parallel = parallel
        self.n_jobs = n_jobs
        self.DIVS2Q = {
            1: 0.25,
            2: 0.5,
            4: 1,
            8: 2,
            16: 4,
            64: 8,
            128: 16,
            256: 32
        }
        self.parts = self.process()

    def __getitem__(self, item):
        return self.parts[item]

    def process(self):
        qdivs = self.initialize_part_with_div()
        has_pickup = not np.all(np.char.startswith(self.document, "=1-") == False)
        if not has_pickup:
            position = 0
        else:
            position = self._handle_pickup_position()

        if self.parallel:
            parts = list(zip(*Parallel(n_jobs=self.n_jobs)(delayed(KernParserPart)(self.document[i], position, self.doc_name, str(i), qdivs) for i in range(self.document.shape[0]))))
        else:
            parts = list(zip(*[KernParserPart(self.document[i], position, self.doc_name, str(i), qdivs) for i in range(self.document.shape[0])]))
        return parts

    # TODO pick-up measure
    def _handle_pickup_position(self):
        return 0

    # ------------- Functions to parse staves info --------------------------

    def initialize_part_with_div(self):
        def try_to_eval2(x):
            try:
                return eval(x[:2])
            except SyntaxError:
                return 0

        def try_to_eval3(x):
            try:
                return eval(x[:3])
            except SyntaxError:
                return 0

        nlist = list()
        for n in range(1, 10):
            idx = np.where(np.char.startswith(self.document, str(n)))
            nlist += [self.document[idx]]
        notes = np.concatenate(nlist)
        y = np.nonzero(np.vectorize(try_to_eval2)(notes))
        if y[0].size != 0:
            n = 2
            t = np.nonzero(np.vectorize(try_to_eval3)(notes))
            if t[0].size != 0:
                y = t
                n = 3
        else:
            y = np.nonzero(np.vectorize(lambda x: isinstance(eval(x[0]), int))(notes))
            n = 1
        min_value = np.max(np.vectorize(lambda x: eval(x[:n]))(notes[y]))
        qdivs = self.DIVS2Q[min_value]
        return qdivs

# functions to initialize the kern parser

def parse_kern(kern_path):
    """
    Parses an KERN file from path to an regular expression.

    Parameters
    ----------
    kern_path : str
        The path of the KERN document.
    Returns
    -------
    document : lxml tree
        An lxml tree of the MEI score.
    ns : str
        The namespace tag of the document.
    """
    with open(kern_path) as file:
        lines = file.read().splitlines()
    d = [line.split("\t") for line in lines]
    document = np.array(list(zip(d))).squeeze(1).T
    return document


def load_kern(kern_path: str):
    """
    Parameters
    ----------
    kern_path : str
        The path to an KERN score.

    Returns
    -------
    part_list : list
        A list of Partitura Part Objects.
    """
    # parse xml file
    document = parse_kern(kern_path)
    doc_name = os.path.basename(kern_path[:-4])
    parts = KernParser(document, doc_name).parts
    part = parts[0]
    print(part.notes())


