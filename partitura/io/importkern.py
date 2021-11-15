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
    def __init__(self, stream, init_pos, doc_name, part_id, qdivs, barline_dict=None):
        super(KernParserPart, self).__init__(doc_name, part_id, qdivs)
        self.position = init_pos
        self.mode = None
        self.barline_dict = dict() if not barline_dict else barline_dict
        self.slur_dict = {"open": [], "close": []}
        self.tie_dict = {"open": [], "close": []}
        self.process(stream)

    def process(self, line):
        self.staff = None
        for index, el in enumerate(line):
            if el.startswith("*staff"):
                self.staff = eval(el[len("*staff"):])
            # elif el.startswith("!!!"):
            #     self._handle_fileinfo(el)
            elif el.startswith("*"):
                if self.staff == None:
                    self.staff = 1
                self._handle_glob_attr(el)
            elif el == ".":
                pass
            elif el.startswith("="):
                self._handle_barline(el)
            elif " " in el:
                self._handle_chord(el, index)
            elif "r" in el:
                self._handle_rest(el, "r-" + str(index))
            else:
                self._handle_note(el, "n-" + str(index))
        self.nid_dict = {note.id : note for note in self.iter_all(cls=score.Note)}
        self._handle_slurs()
        self._handle_ties()

    # TODO handle !!!info
    def _handle_fileinfo(self, el):
        pass

    def _handle_ties(self):
        if len(self.tie_dict["open"]) != len(self.tie_dict["close"]):
            raise ValueError("Tie Mismatch! Uneven amount of closing to open tie brackets.")
        else:
            for (oid,cid) in list(zip(self.tie_dict["open"], self.tie_dict["close"])):
                self.nid_dict[oid].tie_next = self.nid_dict[cid]
                self.nid_dict[cid].tie_prev = self.nid_dict[oid]

    def _handle_slurs(self):
        if len(self.slur_dict["open"]) != len(self.slur_dict["close"]):
            raise ValueError("Slur Mismatch! Uneven amount of closing to open slur brackets.")
        else:
            for (oid, cid) in list(zip(self.slur_dict["open"], self.slur_dict["close"])):
                self.add(score.Slur(self.nid_dict[oid], self.nid_dict[cid]))

    def _handle_metersig(self, metersig):
        m = metersig[2:]
        numerator, denominator = map(eval, m.split("/"))
        new_time_signature = score.TimeSignature(numerator, denominator)
        self.add(new_time_signature, self.position)

    def _handle_barline(self, element):
        if element.endswith("!"):
            barline = score.Fine()
        elif element.endswith(":|"):
            barline = score.Repeat()
        # TODO repeat bars front back and double line bars.
        elif "!" in element:
            barline = score.Barline(style="think")
        else:
            bartype, barnum, _ = re.split('(\d+)', element)
            if eval(barnum) not in self.barline_dict.keys():
                self.barline_dict[eval(barnum)] = self.position
            else :
                self.position = self.barline_dict[eval(barnum)]
            if bartype == "=":
                barline = score.Barline(style="normal")
            else:
                barline = score.Barline(style="special")
        self.add(barline, self.position)

    # TODO maybe also append position for verification.
    def _handle_mode(self, element):
        if element[1].isupper():
            self.mode = "major"
        else:
            self.mode = "minor"

    def _handle_keysig(self, element):
        keysig_el = element[2:]
        fifths = 0
        for c in keysig_el:
            if c == "#":
                fifths += 1
            if c == "b":
                fifths -= 1
        # TODO retrieve the key mode
        mode = self.mode if self.mode else "major"
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
        line = int(element[6]) if element[6] != "v" else int(element[7])
        new_clef = score.Clef(self.staff, element[5], line, 1)
        self.add(new_clef, self.position)

    def _handle_rest(self, el, rest_id):
        # find duration info
        duration, symbolic_duration, rtype = self._handle_duration(el)
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

    def _handle_fermata(self, note_instance):
        self.add(note_instance, self.position)

    def _search_slurs_and_ties(self, note, note_id):
        if note.startswith("("):
            # acount for multiple opening brackets
            n = note.count("(")
            for _ in range(n):
                self.slur_dict["open"].append(note_id)
            # Re-order for correct parsing
            if len(self.slur_dict["open"]) > len(self.slur_dict["close"])+1:
                x = note_id
                lenc = len(self.slur_dict["open"]) - len(self.slur_dict["close"])
                self.slur_dict["open"][:lenc - 1] = self.slur_dict["open"][1:lenc]
                self.slur_dict["open"][lenc] = x
            note = note[n:]
        if ")" in note:
            x = note.count(")")
            if len(self.slur_dict["open"]) == len(self.slur_dict["close"])+x:
                for _ in range(x):
                    self.slur_dict["close"].append(note_id)
        if note.startswith("["):
            self.tie_dict["open"].append(note_id)
            note = note[1:]
        if "]" in note:
            self.tie_dict["close"].append(note_id)
        return note

    # TODO tuplet durations.
    def _handle_duration(self, note):
        # TODO deal with grace notes
        if "q" in note:
            _ , dur, ntype = re.split('(\d+)', note)
            ntype = _ + ntype
        else:
            _, dur, ntype = re.split('(\d+)', note)
        dur = eval(dur)
        if dur in self.KERN_DURS.keys():
            symbolic_duration = {"type": self.KERN_DURS[dur]}
        else:
            symbolic_duration = {
                "type" : self.KERN_DURS[4] if dur not in [13, 11, 9, 7, 5, 3] else self.KERN_DURS[4],
                "actual_notes" : dur/4,
                "normal_notes" : 1
            }
        # calculate duration to divs.
        qdivs = self._quarter_durations[0]
        duration = LABEL_DURS[symbolic_duration["type"]] * qdivs
        if "." in note:
            symbolic_duration["dots"] = note.count(".")
            ntype = ntype[note.count("."):]
            for i in range(symbolic_duration["dots"]):
                duration += duration / 2
        else:
            symbolic_duration["dots"] = 0
        return duration, symbolic_duration, ntype

    def _handle_note(self, note, note_id):
        has_fermata = ";" in note
        note = self._search_slurs_and_ties(note, note_id)
        duration, symbolic_duration, ntype = self._handle_duration(note)
        grace_attr = "q" in ntype or "Q" in ntype
        step, octave = self.KERN_NOTES[ntype[0]]
        if octave == 4:
            octave += ntype.count(step)
        elif octave == 3:
            octave -= ntype.count(step)
        alter = ntype.count('#') - ntype.count("-")
        # find if it's grace
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
            if has_fermata:
                self._handle_fermata(note)
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

        self.add(note, self.position, self.position + duration)
        self.position += duration

    def _handle_chord(self, chord, id):
        notes = chord.split()
        pos = self.position
        for i, note_el in enumerate(notes):
            id = "c-" + str(i) + "-" + str(id)
            self.position = pos
            self._handle_note(note_el, id)

    def _handle_glob_attr(self, el):
        if el.startswith("*clef"):
            self._handle_clef(el)
        elif el.startswith("*k"):
            self._handle_keysig(el)
        elif el.startswith("*MM"):
            pass
        elif el.startswith("*M"):
            self._handle_metersig(el)
        elif el.endswith(":"):
            self._handle_mode(el)
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
        self.qdivs = self.initialize_part_with_div()
        has_pickup = not np.all(np.char.startswith(self.document, "=1-") == False)
        if not has_pickup:
            position = 0
        else:
            position = self._handle_pickup_position()


        if self.parallel:
            parts = Parallel(n_jobs=self.n_jobs)(delayed(self.collect)(self.document[i], position, self.doc_name, str(i), self.qdivs) for i in range(self.document.shape[0]))
        else:
            parts = [self.collect(self.document[i], position, self.doc_name, str(i), qdivs) for i in range(self.document.shape[0])]

        return parts

    def add_part(self, unprocessed):
        flatten = [item for sublist in unprocessed for item in sublist]
        parts = KernParserPart(flatten, 0, self.doc_name, "x", self.qdivs, barline_dict=self.parts[0].barline_dict)


    def collect(self, doc, pos, doc_name, id, qdivs):
        if doc[0] != "**silbe":
            x = KernParserPart(doc, pos, doc_name, id, qdivs)
            return x

    # TODO handle position of pick-up measure
    def _handle_pickup_position(self):
        return 0

    # TODO fix divs calculation with LCM
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
        qdivs = self.DIVS2Q[min_value] if min_value in self.DIVS2Q.keys() else int(min_value/4)
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
    d = [line.split("\t") for line in lines if not line.startswith("!")]
    init_parts = len(d[0])
    striped_parts = list()
    non_continuous = list()
    for x in d:
        striped_parts.append(x[:init_parts])
        if len((x)) > init_parts:
            non_continuous.append(x[init_parts:])
    continuous_parts = np.array(list(zip(striped_parts))).squeeze(1).T
    return continuous_parts, non_continuous


def load_kern(kern_path: str, parallel=False):
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
    # parse kern file
    continuous_parts, non_continuous = parse_kern(kern_path)
    doc_name = os.path.basename(kern_path[:-4])
    parser = KernParser(continuous_parts, doc_name, parallel=parallel)
    parser.add_part(non_continuous)
    parts = parser.parts
    return parts


