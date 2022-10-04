import re
import warnings

from typing import Union, Optional

import numpy as np

import partitura.score as score
from partitura.utils import PathLike, get_document_name
from partitura.utils.misc import deprecated_alias, deprecated_parameter


__all__ = ["load_kern"]


class KernGlobalPart(object):
    def __init__(self, doc_name, part_id, qdivs):
        qdivs = int(1) if int(qdivs) == 0 else int(qdivs)
        # super(KernGlobalPart, self).__init__()
        self.part = score.Part(doc_name, part_id, quarter_duration=qdivs)
        self.default_clef_lines = {"G": 2, "F": 4, "C": 3}
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
            "C": ("C", 3),
            "D": ("D", 3),
            "E": ("E", 3),
            "F": ("F", 3),
            "G": ("G", 3),
            "A": ("A", 3),
            "B": ("B", 3),
            "c": ("C", 4),
            "d": ("D", 4),
            "e": ("E", 4),
            "f": ("F", 4),
            "g": ("G", 4),
            "a": ("A", 4),
            "b": ("B", 4),
        }

        self.KERN_DURS = {
            # "long": "long",
            # "breve": "breve",
            0: "breve",
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
    """
    Class for parsing kern file syntax.
    """

    def __init__(self, stream, init_pos, doc_name, part_id, qdivs, barline_dict=None):
        super(KernParserPart, self).__init__(doc_name, part_id, qdivs)
        self.position = int(init_pos)
        self.parsing = "full"
        self.stream = stream
        self.prev_measure_pos = init_pos
        # Check if part has pickup measure.
        self.measure_count = (
            0 if np.all(np.char.startswith(stream, "=1-") == False) else 1
        )
        self.last_repeat_pos = None
        self.mode = None
        self.barline_dict = dict() if not barline_dict else barline_dict
        self.slur_dict = {"open": [], "close": []}
        self.tie_dict = {"open": [], "close": []}
        self.process()

    def process(self):
        self.staff = None
        for index, el in enumerate(self.stream):
            self.current_index = index
            if el.startswith("*staff"):
                self.staff = eval(el[len("*staff") :])
            # elif el.startswith("!!!"):
            #     self._handle_fileinfo(el)
            elif el.startswith("*"):
                if self.staff == None:
                    self.staff = 1
                self._handle_glob_attr(el)
            elif el.startswith("="):
                self.select_parsing(el)
                self._handle_barline(el)
            elif " " in el:
                self._handle_chord(el, index)
            elif "r" in el:
                self._handle_rest(el, "r-" + str(index))
            else:
                self._handle_note(el, "n-" + str(index))
        self.nid_dict = dict(
            [(n.id, n) for n in self.part.iter_all(cls=score.Note)]
            + [(n.id, n) for n in self.part.iter_all(cls=score.GraceNote)]
        )
        self._handle_slurs()
        self._handle_ties()

    # Account for parsing priorities.
    def select_parsing(self, el):
        if self.parsing == "full":
            return el
        elif self.parsing == "right":
            return el.split()[-1]
        else:
            return el.split()[0]

    # TODO handle !!!info
    def _handle_fileinfo(self, el):
        pass

    def _handle_ties(self):
        try:
            if len(self.tie_dict["open"]) < len(self.tie_dict["close"]):
                for index, oid in enumerate(self.tie_dict["open"]):
                    if (
                        self.nid_dict[oid].midi_pitch
                        != self.nid_dict[self.tie_dict["close"][index]].midi_pitch
                    ):
                        dnote = self.nid_dict[self.tie_dict["close"][index]]
                        m_num = [
                            m
                            for m in self.part.iter_all(score.Measure)
                            if m.start.t == self.part.measure_map(dnote.start.t)[0]
                        ][0].number
                        warnings.warn(
                            "Dropping Closing Tie of note {} at position {} measure {}".format(
                                dnote.midi_pitch, dnote.start.t, m_num
                            )
                        )
                        self.tie_dict["close"].pop(index)
                        self._handle_ties()
            elif len(self.tie_dict["open"]) > len(self.tie_dict["close"]):
                for index, cid in enumerate(self.tie_dict["close"]):
                    if (
                        self.nid_dict[cid].midi_pitch
                        != self.nid_dict[self.tie_dict["open"][index]].midi_pitch
                    ):
                        dnote = self.nid_dict[self.tie_dict["open"][index]]
                        m_num = [
                            m
                            for m in self.part.iter_all(score.Measure)
                            if m.start.t == self.part.measure_map(dnote.start.t)[0]
                        ][0].number
                        warnings.warn(
                            "Dropping Opening Tie of note {} at position {} measure {}".format(
                                dnote.midi_pitch, dnote.start.t, m_num
                            )
                        )
                        self.tie_dict["open"].pop(index)
                        self._handle_ties()
            else:
                for (oid, cid) in list(
                    zip(self.tie_dict["open"], self.tie_dict["close"])
                ):
                    self.nid_dict[oid].tie_next = self.nid_dict[cid]
                    self.nid_dict[cid].tie_prev = self.nid_dict[oid]
        except Exception:
            raise ValueError(
                "Tie Mismatch! Uneven amount of closing to open tie brackets."
            )

    def _handle_slurs(self):
        if len(self.slur_dict["open"]) != len(self.slur_dict["close"]):
            raise ValueError(
                "Slur Mismatch! Uneven amount of closing to open slur brackets."
            )
        else:
            for (oid, cid) in list(
                zip(self.slur_dict["open"], self.slur_dict["close"])
            ):
                self.part.add(score.Slur(self.nid_dict[oid], self.nid_dict[cid]))

    def _handle_metersig(self, metersig):
        m = metersig[2:]
        numerator, denominator = map(eval, m.split("/"))
        new_time_signature = score.TimeSignature(numerator, denominator)
        self.part.add(new_time_signature, self.position)

    def _handle_barline(self, element):
        if self.position > self.prev_measure_pos:
            indicated_measure = re.findall("=([0-9]+)", element)
            if indicated_measure != []:
                m = eval(indicated_measure[0]) - 1
                barline = score.Barline(style="normal")
                self.part.add(barline, self.position)
                self.measure_count = m
                self.barline_dict[m] = self.position
            else:
                m = self.measure_count - 1
            self.part.add(score.Measure(m), self.prev_measure_pos, self.position)
            self.prev_measure_pos = self.position
            self.measure_count += 1
        if len(element.split()) > 1:
            element = element.split()[0]
        if element.endswith("!") or element == "==":
            barline = score.Fine()
            self.part.add(barline, self.position)
        if ":|" in element:
            barline = score.Repeat()
            self.part.add(
                barline,
                self.position,
                self.last_repeat_pos if self.last_repeat_pos else None,
            )
        # update position for backward repeat signs
        if "|:" in element:
            self.last_repeat_pos = self.position

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
        self.part.add(new_key_signature, self.position)

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
        if element[5] not in ["G", "F", "C"]:
            raise ValueError("Unknown Clef", element[5])
        if len(element) < 7:
            line = self.default_clef_lines[element[5]]
        else:
            line = int(element[6]) if element[6] != "v" else int(element[7])
        new_clef = score.Clef(
            staff=self.staff, sign=element[5], line=line, octave_change=0
        )
        self.part.add(new_clef, self.position)

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
        self.part.add(rest, self.position, self.position + duration)
        # return duration to update the position in the layer
        self.position += duration

    def _handle_fermata(self, note_instance):
        self.part.add(note_instance, self.position)

    def _search_slurs_and_ties(self, note, note_id):
        if ")" in note:
            x = note.count(")")
            if len(self.slur_dict["open"]) == len(self.slur_dict["close"]) + x:
                # for _ in range(x):
                self.slur_dict["close"].append(note_id)
        if note.startswith("("):
            # acount for multiple opening brackets
            n = note.count("(")
            # for _ in range(n):
            self.slur_dict["open"].append(note_id)
            # Re-order for correct parsing
            if len(self.slur_dict["open"]) > len(self.slur_dict["close"]) + 1:
                warnings.warn(
                    "Cannot deal with nested slurs. Dropping Opening slur for note id {}".format(
                        self.slur_dict["open"][len(self.slur_dict["open"]) - 2]
                    )
                )
                self.slur_dict["open"].pop(len(self.slur_dict["open"]) - 2)
                # x = note_id
                # lenc = len(self.slur_dict["open"]) - len(self.slur_dict["close"])
                # self.slur_dict["open"][:lenc - 1] = self.slur_dict["open"][1:lenc]
                # self.slur_dict["open"][lenc] = x
            note = note[n:]
        if "]" in note:
            self.tie_dict["close"].append(note_id)
        if note.startswith("["):
            self.tie_dict["open"].append(note_id)
            note = note[1:]
        return note

    def _handle_duration(self, note, isgrace=False):
        if isgrace:
            _, dur, ntype = re.split("(\d+)", note)
            ntype = _ + ntype
        else:
            _, dur, ntype = re.split("(\d+)", note)
        dur = eval(dur)
        if dur in self.KERN_DURS.keys():
            symbolic_duration = {"type": self.KERN_DURS[dur]}
        else:
            diff = dict(
                (
                    map(
                        lambda x: (dur - x, x) if dur > x else (dur + x, x),
                        self.KERN_DURS.keys(),
                    )
                )
            )
            symbolic_duration = {
                "type": self.KERN_DURS[diff[min(list(diff.keys()))]],
                "actual_notes": dur / 4,
                "normal_notes": diff[min(list(diff.keys()))] / 4,
            }

        # calculate duration to divs.
        qdivs = self.part._quarter_durations[0]
        duration = qdivs * 4 / dur if dur != 0 else qdivs * 8
        if "." in note:
            symbolic_duration["dots"] = note.count(".")
            ntype = ntype[note.count(".") :]
            d = duration
            for i in range(symbolic_duration["dots"]):
                d = d / 2
                duration += d
        else:
            symbolic_duration["dots"] = 0
        if isinstance(duration, float):
            if not duration.is_integer():
                raise ValueError("Duration divs is not an integer, {}".format(duration))
        # Check that duration is same as int
        assert int(duration) == duration
        return int(duration), symbolic_duration, ntype

    # TODO Handle beams and tuplets.

    def _handle_note(self, note, note_id, voice=1):
        if note == ".":
            return
        has_fermata = ";" in note
        note = self._search_slurs_and_ties(note, note_id)
        grace_attr = "q" in note  # or "p" in note # for appoggiatura not sure yet.
        duration, symbolic_duration, ntype = self._handle_duration(note, grace_attr)
        # Remove editorial symbols from string, i.e. "x"
        ntype = ntype.replace("x", "")
        step, octave = self.KERN_NOTES[ntype[0]]
        if octave == 4:
            octave = octave + ntype.count(ntype[0]) - 1
        elif octave == 3:
            octave = octave - ntype.count(ntype[0]) + 1
        alter = ntype.count("#") - ntype.count("-")
        # find if it's grace
        if not grace_attr:
            # create normal note
            note = score.Note(
                step=step,
                octave=octave,
                alter=alter,
                id=note_id,
                voice=int(voice),
                staff=self.staff,
                symbolic_duration=symbolic_duration,
                articulations=None,  # TODO : add articulation
            )
            if has_fermata:
                self._handle_fermata(note)
        else:
            # create grace note
            if "p" in ntype:
                grace_type = "acciaccatura"
            elif "q" in ntype:
                grace_type = "appoggiatura"
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
            duration = 0

        self.part.add(note, self.position, self.position + duration)
        self.position += duration

    def _handle_chord(self, chord, id):
        notes = chord.split()
        position_history = list()
        pos = self.position
        for i, note_el in enumerate(notes):
            id_new = "c-" + str(i) + "-" + str(id)
            self.position = pos
            if "r" in note_el:
                self._handle_rest(note_el, id_new)
            else:
                self._handle_note(note_el, id_new, voice=int(i))
            if note_el != ".":
                position_history.append(self.position)
        # To account for Voice changes and alternate voice order.
        self.position = min(position_history) if position_history else self.position

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
        elif el.startswith("*S/sic"):
            self.parsing = "left"
        elif el.startswith("*S/ossia"):
            self.parsing = "right"
        elif el.startswith("Xstrophe"):
            self.parsing = "full"


class KernParser:
    def __init__(self, document, doc_name):
        self.document = document
        self.doc_name = doc_name
        # TODO review this code
        self.DIVS2Q = {
            1: 0.25,
            2: 0.5,
            4: 1,
            6: 1.5,
            8: 2,
            16: 4,
            24: 6,
            32: 8,
            48: 12,
            64: 16,
            128: 32,
            256: 64,
        }
        # self.qdivs =
        self.parts = self.process()

    def __getitem__(self, item):
        return self.parts[item]

    def process(self):
        # TODO handle pickup
        # has_pickup = not np.all(np.char.startswith(self.document, "=1-") == False)
        # if not has_pickup:
        #     position = 0
        # else:
        #     position = self._handle_pickup_position()
        position = 0
        # Add for parallel processing
        parts = [
            self.collect(self.document[i], position, str(i), self.doc_name)
            for i in reversed(range(self.document.shape[0]))
        ]
        return [p for p in parts if p]

    def add2part(self, part, unprocessed):
        flatten = [item for sublist in unprocessed for item in sublist]
        if unprocessed:
            new_part = KernParserPart(
                flatten, 0, self.doc_name, "x", self.qdivs, part.barline_dict
            )
            self.parts.append(new_part)

    def collect(self, doc, pos, id, doc_name):
        if doc[0] == "**kern":
            qdivs = self.find_lcm(doc)
            x = KernParserPart(doc, pos, id, doc_name, qdivs).part
            return x

    # TODO handle position of pick-up measure?
    def _handle_pickup_position(self):
        return 0

    def find_lcm(self, doc):
        kern_string = "-".join([row for row in doc])
        match = re.findall(r"([0-9]+)([a-g]|[A-G]|r|\.)", kern_string)
        durs, _ = zip(*match)
        x = np.array(list(map(lambda x: int(x), durs)))
        divs = np.lcm.reduce(np.unique(x))
        return float(divs) / 4.00


# functions to initialize the kern parser
def parse_kern(kern_path: PathLike) -> np.ndarray:
    """
    Parses an KERN file from path to an regular expression.

    Parameters
    ----------
    kern_path : PathLike
        The path of the KERN document.
    Returns
    -------
    continuous_parts : numpy character array
    non_continuous_parts : list
    """
    with open(kern_path, encoding="cp437") as file:
        lines = file.read().splitlines()
    d = [line.split("\t") for line in lines if not line.startswith("!")]
    striped_parts = list()
    merge_index = []
    for x in d:
        if merge_index:
            for midx in merge_index:
                x[midx] = x[midx] + " " + x[midx + 1]
            y = [el for i, el in enumerate(x) if i - 1 not in merge_index]
            striped_parts.append(y)
        else:
            striped_parts.append(x)
        if "*^" in x or "*+":
            # Accounting for multiple voice ups at the same time.
            for i, el in enumerate(x):
                # Some faulty kerns create an extra part half way through the score.
                # We choose for the moment to add it to the closest column part.
                if el == "*^" or el == "*+":
                    k = i
                    if merge_index:
                        if k < min(merge_index):
                            merge_index = [midx + 1 for midx in merge_index]
                    merge_index.append(k)
        if "*v *v" in x:
            k = x.index("*v *v")
            temp = list()
            for i in merge_index:
                if i > k:
                    temp.append(i - 1)
                elif i < k:
                    temp.append(i)
            merge_index = temp
    # Final filter for mistabs and inconsistent tabs that would create
    # extra empty voice and would mess the parsing.
    striped_parts = [[el for el in part if el != ""] for part in striped_parts]
    numpy_parts = np.array(list(zip(striped_parts))).squeeze(1).T
    return numpy_parts


@deprecated_alias(kern_path="filename")
@deprecated_parameter("ensure_list")
def load_kern(
    filename: PathLike,
    force_note_ids: Optional[Union[bool, str]] = None,
    parallel: bool = False,
) -> score.Score:
    """Parse a Kern file and build a composite score ontology
    structure from it (see also scoreontology.py).

    Parameters
    ----------
    filename : PathLike
        Path to the Kern file to be parsed
    force_note_ids : (bool, 'keep') optional.
        When True each Note in the returned Part(s) will have a newly
        assigned unique id attribute. Existing note id attributes in
        the Kern will be discarded. If 'keep', only notes without
        a note id will be assigned one.

    Returns
    -------
    scr: :class:`partitura.score.Score`
        A `Score` object
    """
    # parse kern file
    numpy_parts = parse_kern(filename)
    # doc_name = os.path.basename(filename[:-4])
    doc_name = get_document_name(filename)
    parser = KernParser(numpy_parts, doc_name)
    partlist = parser.parts

    score.assign_note_ids(
        partlist, keep=(force_note_ids is True or force_note_ids == "keep")
    )

    # TODO: Parse score info (composer, lyricist, etc.)
    scr = score.Score(id=doc_name, partlist=partlist)

    return scr
