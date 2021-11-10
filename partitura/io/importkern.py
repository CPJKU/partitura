import os.path
import re
from xmlschema.names import XML_NAMESPACE
import partitura.score as score
from joblib import Parallel, delayed

import partitura.utils
from partitura.utils.music import LABEL_DURS


import numpy as np

SIGN_TO_ACC = {
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

KERN_NOTES ={
    'C' : ("C", 3),
    'D' : ('D', 3),
    'E' : ('E', 3),
    'F' : ('F', 3),
    'G' : ('G', 3),
    'A' : ('A', 3),
    'B' : ('B', 3),
    'c' : ("C", 4),
    'd' : ('D', 4),
    'e' : ('E', 4),
    'f' : ('F', 4),
    'g' : ('G', 4),
    'a' : ('A', 4),
    'b' : ('B', 4)
}

DIVS2Q = {
    1 : 0.25,
    2 : 0.5,
    4 : 1,
    8 : 2,
    16 : 4,
    64 : 8,
    128 : 16,
    256 : 32
}

KERN_DURS = {
    # "long": "long",
    # "breve": "breve",
    1 : "whole",
    2 : "half",
    4 : "quarter",
    8 : "eighth",
    16 : "16th",
    32 : "32nd",
    64 : "64th",
    128 : "128th",
    256 : "256th",
}


# functions to initialize the xml tree


def _parse_kern(kern_path):
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
    d = [line.split("\t")for line in lines]
    document = np.array(list(zip(d))).squeeze(1).T
    return document


# ------------- Functions to parse staves info --------------------------


def _handle_metersig(metersig, part, position):
    """
    Handles meter signature and adds to part.

    Parameters
    ----------
    staffdef_el : lxml tree
        A lxml substree of a staff's mei score.
    position : int
        The End of a section.
    part : particular.Part
        The created Partitura Part object.
    """
    m = metersig[2:]
    numerator, denominator = map(eval, m.split("/"))
    new_time_signature = score.TimeSignature(numerator, denominator)
    part.add(new_time_signature, position)


def _handle_keysig(element, part, position):
    """
    Handles key signature and adds to part.

    Parameters
    ----------
    element : str
        A keysig element.
    position : int
        The End of a section.
    part : particular.Part
        The created Partitura Part object.

    """
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
    part.add(new_key_signature, position)


def _compute_clef_octave(dis, dis_place):
    if dis is not None:
        sign = -1 if dis_place == "below" else 1
        octave = sign * int(int(dis) / 8)
    else:
        octave = 0
    return octave


def _handle_clef(element, part, position, staff):
    """Inserts a clef to part

    Parameters
    ----------
    element : str
        A element containing a cleff.
    part : particular.Part
        The created Partitura Part object.
    position : int
        The End of a section.
    staff : int
        The number of the staff.

    Returns
    -------
    position : int
    """
    # handle the case where we have clef information
    new_clef = score.Clef(staff, element[5], int(element[6]), 0)
    part.add(new_clef, position)


def divs_to_quarter(min_value):
    return DIVS2Q[min_value]

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

def _handle_rest(el, part, position, rest_id):
    # find duration info
    _, duration, ntype = re.split('(\d+)', el)
    symbolic_duration = KERN_DURS[eval(duration)]
    duration = part.inv_quarter_map([LABEL_DURS[symbolic_duration]])
    # create rest
    rest = score.Rest(
        id=rest_id,
        voice=1,
        staff=1,
        symbolic_duration=symbolic_duration,
        articulations=None,
    )
    # add rest to the part
    part.add(rest, position, position + duration)
    # return duration to update the position in the layer
    return position + duration

def _handle_note(note, part, position, note_id):
    # TODO handle brackets
    if note.startswith("[") or note.startswith("{"):
        note = note[1:]
    _ , duration, ntype = re.split('(\d+)',note)
    is_dotted = ntype.startswith(".")
    d = eval(duration)
    if is_dotted:
        symbolic_duration = KERN_DURS[d]
        ntype = ntype[1:]
    else :
        symbolic_duration = KERN_DURS[d]
    step, octave = KERN_NOTES[ntype[0]]
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
            staff=1,
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
            voice=voice,
            staff=staff,
            symbolic_duration=symbolic_duration,
            articulations=None,  # TODO : add articulation
        )
    qdivs = part._quarter_durations[0]
    duration = LABEL_DURS[symbolic_duration]*qdivs
    part.add(note, position, position+duration)
    return position + duration


def _handle_chord(chord, part, position, id):
    notes = chord.split()
    for i, note_el in enumerate(notes):
        id = "c-" + str(i) + "-" + str(id)
        new_pos = _handle_note(note_el, part, position, id)
    return new_pos

def _handle_bar(el, part, position):
    pass


def initialize_part_with_div(document, doc_name):
    nlist = list()
    for n in range(1, 10):
        idx = np.where(np.char.startswith(document, str(n)))
        nlist += [document[idx]]
    notes = np.concatenate(nlist)
    y = np.nonzero(np.vectorize(try_to_eval2)(notes))
    if y[0].size != 0:
        n = 2
        t = np.nonzero(np.vectorize(try_to_eval3)(notes))
        if t[0].size != 0 :
            y = t
            n = 3
    else :
        y = np.nonzero(np.vectorize(lambda x: isinstance(eval(x[0]), int))(notes))
        n = 1
    min_value = np.max(np.vectorize(lambda x : eval(x[:n]))(notes[y]))
    qdivs = divs_to_quarter(min_value)
    # init part
    parts = [score.Part(doc_name, str(i), quarter_duration=qdivs) for i in range(document.shape[0])]
    return parts


def _handle_glob_attr(el, part, position, staff=None):
    if el.startswith("*clef"):
        _handle_clef(el, part, position, staff)
    elif el.startswith("*k"):
        _handle_keysig(el, part, position)
    elif el.startswith("*M"):
        _handle_metersig(el, part, position)
    elif el == "*-":
        print("Reached the end of the stream.")


def _handle_part_elements(line, part, position):
    staff = None
    for index, el in enumerate(line):
        if el.startswith("*staff"):
            staff = eval(el[len("*staff"):])
        elif el.startswith("*"):
            if staff == None:
                staff = 1
            _handle_glob_attr(el, part, position, staff)
        elif el == ".":
            pass
        elif el.startswith("="):
            _handle_bar(el, part, position)
        elif " " in el:
            position = _handle_chord(el, part, position, index)
        elif "r" in el:
            position = _handle_rest(el, part, position, "r-"+str(index))
        else:
            position = _handle_note(el, part, position, "n-"+str(index))
    return position, part


def _handle_pickup_position(kern_doc):
    return 0


def _parse_elements(kern_doc, parts):
    has_pickup = not np.all(np.char.startswith(kern_doc, "=1-") == False)
    if not has_pickup:
        position = 0
    else:
        position = _handle_pickup_position(kern_doc)

    # positions, parts = zip(*Parallel(n_jobs=2)(delayed(_handle_part_elements)(kern_doc[i], parts[i], position) for i in range(kern_doc.shape[0])))
    positions, parts = zip(*[_handle_part_elements(kern_doc[i], parts[i], position) for i in range(kern_doc.shape[0])])
    return positions, parts


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
    document = _parse_kern(kern_path)
    doc_name = os.path.basename(kern_path[:-4])
    parts = initialize_part_with_div(document, doc_name)
    positions, parts = _parse_elements(document, parts)
    part = parts[0]
    print(part.notes())

    # # handle staff and staff groups info
    # main_partgroup_el = document.find(_ns_name("staffGrp", ns, True))
    # part_list = _handle_main_staff_group(main_partgroup_el, ns)
    #
    # # fill the content of the score
    # sections_el = document.findall(_ns_name("section", ns, True))
    # if len(sections_el) != 1:
    #     raise Exception("Only MEI with a single section are supported")
    # _handle_section(list(score.iter_parts(part_list)), sections_el[0], ns)
    # # handles ties
    # _tie_notes(sections_el[0], part_list, ns)
    # return part_list

