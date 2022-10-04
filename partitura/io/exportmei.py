# import partitura
# import partitura.score as score
# from lxml import etree
# from partitura.utils.generic import partition
# from partitura.utils.music import estimate_symbolic_duration
# from copy import copy


# name_space = "http://www.music-encoding.org/ns/mei"

# xml_id_string = "{http://www.w3.org/XML/1998/namespace}id"


# def extend_key(dict_of_lists, key, value):
#     """extend or create a list at the given key in the given dictionary

#     Parameters
#     ----------
#     dict_of_lists:    dictionary
#         where all values are lists
#     key:            self explanatory
#     value:          self explanatory

#     """

#     if key in dict_of_lists.keys():
#         if isinstance(value, list):
#             dict_of_lists[key].extend(value)
#         else:
#             dict_of_lists[key].append(value)
#     else:
#         dict_of_lists[key] = value if isinstance(value, list) else [value]


# def calc_dur_dots_split_notes_first_temp_dur(note, measure, num_to_numbase_ratio=1):
#     """
#     Notes have to be represented as a string of elemental notes (there is no notation for arbitrary durations)
#     This function calculates this string (the durations of the elemental notes and their dot counts),
#     whether the note crosses the measure and the temporal duration of the first elemental note

#     Parameters
#     ----------
#     note:               score.GenericNote
#         The note whose representation as a string of elemental notes is calculated
#     measure:            score.Measure
#         The measure which contains note
#     num_to_numbase_ratio: float, optional
#         scales the duration of note according to whether or not it belongs to a tuplet and which one


#     Returns
#     -------
#     dur_dots:       list of int pairs
#         this describes the string of elemental notes that represent the note notationally
#         every pair in the list contains the duration and the dot count of an elemental note and
#         the list is ordered by duration in decreasing order
#     split_notes:     list or None
#         an empty list if note crosses measure
#         None if it doesn't
#     first_temp_dur:   int or None
#         duration of first elemental note in partitura time
#     """

#     if measure == "pad":
#         return [], None, None

#     if isinstance(note, score.GraceNote):
#         main_note = note.main_note
#         # HACK: main note should actually be always not None for a proper GraceNote
#         if main_note != None:
#             dur_dots, _, _ = calc_dur_dots_split_notes_first_temp_dur(
#                 main_note, measure
#             )
#             dur_dots = [(2 * dur_dots[0][0], dur_dots[0][1])]
#         else:
#             dur_dots = [(8, 0)]
#             note.id += "_missing_main_note"
#         return dur_dots, None, None

#     note_duration = note.duration

#     split_notes = None

#     if note.start.t + note.duration > measure.end.t:
#         note_duration = measure.end.t - note.start.t
#         split_notes = []

#     quarter_dur = measure.start.quarter
#     fraction = num_to_numbase_ratio * note_duration / quarter_dur

#     int_part = int(fraction)
#     frac_part = fraction - int_part

#     # calc digits of fraction in base2
#     untied_durations = []
#     pow_of_2 = 1

#     while int_part > 0:
#         bit = int_part % 2
#         untied_durations.insert(0, bit * pow_of_2)
#         int_part = int_part // 2
#         pow_of_2 *= 2

#     pow_of_2 = 1 / 2

#     while frac_part > 0:
#         frac_part *= 2
#         bit = int(frac_part)
#         frac_part -= bit
#         untied_durations.append(bit * pow_of_2)
#         pow_of_2 /= 2

#     dur_dots = []

#     curr_dur = 0
#     curr_dots = 0

#     def add_dd(dur_dots, dur, dots):
#         dur_dots.append((int(4 / dur), dots))

#     for untied_dur in untied_durations:
#         if curr_dur != 0:
#             if untied_dur == 0:
#                 add_dd(dur_dots, curr_dur, curr_dots)
#                 curr_dots = 0
#                 curr_dur = 0
#             else:
#                 curr_dots += 1
#         else:
#             curr_dur = untied_dur

#     if curr_dur != 0:
#         add_dd(dur_dots, curr_dur, curr_dots)

#     first_temp_dur = int(untied_durations[0] * quarter_dur)

#     return dur_dots, split_notes, first_temp_dur


# def insert_elem_check(t, inbetween_notes_elems):
#     """Check if something like a clef etc appears before time t

#     Parameters
#     ----------
#     t:                      int
#         time from a Timepoint
#     inbetween_notes_elems:    list of InbetweenNotesElements
#         a list of objects describing things like clefs etc

#     Returns
#     -------
#     True if something like a clef etc appears before time t
#     """

#     for ine in inbetween_notes_elems:
#         if ine.elem != None and ine.elem.start.t <= t:
#             return True

#     return False


# def partition_handle_none(func, iter, partition_attrib):
#     p = partition(func, iter)
#     newKey = None

#     if None in p.keys():
#         raise KeyError(
#             'PARTITION ERROR: some elements of set do not have partition attribute "'
#             + partition_attrib
#             + '"'
#         )

#     return p


# def add_child(parent, child_name):
#     return etree.SubElement(parent, child_name)


# def set_attributes(elem, *list_attrib_val):
#     for attrib_val in list_attrib_val:
#         elem.set(attrib_val[0], str(attrib_val[1]))


# def attribs_of_key_sig(ks):
#     """
#     Returns values of a score.KeySignature object necessary for a MEI document

#     Parameters
#     ----------
#     ks: score.KeySignature

#     Returns
#     -------
#     fifths: string
#         describes the circle of fifths
#     mode:   string
#         "major" or "minor"
#     pname:  string
#         pitch letter
#     """

#     key = ks.name
#     pname = key[0].lower()
#     mode = "major"

#     if len(key) == 2:
#         mode = "minor"

#     fifths = str(abs(ks.fifths))

#     if ks.fifths < 0:
#         fifths += "f"
#     elif ks.fifths > 0:
#         fifths += "s"

#     return fifths, mode, pname


# def first_instances_per_part(
#     cls, parts, start=score.TimePoint(0), end=score.TimePoint(1)
# ):
#     """
#     Returns the first instances of a class (multiple objects with same start time are possible) in each part

#     Parameters
#     ----------
#     cls:    class
#     parts:  list of score.Part
#     start:  score.TimePoint, optional
#         start of the range to search in
#     end:    score.TimePoint, optional
#         end of the range to search in

#     Returns
#     -------
#     instances_per_part: list of list of instances of cls
#         sublists might be empty
#         if all sublists are empty, instances_per_part is empty
#     """
#     if not isinstance(start, list):
#         start = [start] * len(parts)
#     elif not len(parts) == len(start):
#         raise ValueError(
#             "ERROR at first_instances_per_part: start times are given as list with different size to parts list"
#         )

#     if not isinstance(end, list):
#         end = [end] * len(parts)
#     elif not len(parts) == len(end):
#         raise ValueError(
#             "ERROR at first_instances_per_part: end times are given as list with different size to parts list"
#         )

#     for i in range(len(parts)):
#         if start[i] == None and end[i] != None or start[i] != None and end[i] == None:
#             raise ValueError(
#                 "ERROR at first_instances_per_part: (start==None) != (end==None) (None elements in start have to be at same position as in end and vice versa)"
#             )

#     instances_per_part = []

#     non_empty = False

#     for i, p in enumerate(parts):
#         s = start[i]
#         e = end[i]

#         if s == None:
#             instances_per_part.append([])
#             continue

#         instances = list(p.iter_all(cls, s, e))

#         if len(instances) == 0:
#             instances_per_part.append([])
#             continue

#         non_empty = True
#         t = min(instances, key=lambda i: i.start.t).start.t
#         instances_per_part.append([i for i in instances if t == i.start.t])

#     if non_empty:
#         return instances_per_part

#     return []


# def first_instance_per_part(
#     cls, parts, start=score.TimePoint(0), end=score.TimePoint(1)
# ):
#     """
#     Reduce the result of first_instances_per_part, a 2D list, to a 1D list
#     If there are multiple first instances then program aborts with error message

#     Parameters
#     ----------
#     cls:    class
#     parts:  list of score.Part
#     start:  score.TimePoint, optional
#         start of the range to search in
#     end:    score.TimePoint, optional
#         end of the range to search in

#     Returns
#     -------
#     fipp: list of instances of cls
#         elements might be None
#     """
#     fispp = first_instances_per_part(cls, parts, start, end)

#     fipp = []

#     for i, fis in enumerate(fispp):
#         if len(fis) == 0:
#             fipp.append(None)
#         elif len(fis) == 1:
#             fipp.append(fis[0])
#         else:
#             raise ValueError(
#                 "Part " + parts[i].name,
#                 "ID " + parts[i].id,
#                 "has more than one instance of "
#                 + str(cls)
#                 + " at beginning t=0, but there should only be a single one",
#             )

#     return fipp


# def first_instances(cls, part, start=score.TimePoint(0), end=score.TimePoint(1)):
#     """
#     Returns the first instances of a class (multiple objects with same start time are possible) in the part

#     Parameters
#     ----------
#     cls:    class
#     part:   score.Part
#     start:  score.TimePoint, optional
#         start of the range to search in
#     end:    score.TimePoint, optional
#         end of the range to search in

#     Returns
#     -------
#     fis: list of instances of cls
#         might be empty
#     """
#     fis = first_instances_per_part(cls, [part], start, end)

#     if len(fis) == 0:
#         return []

#     return fis[0]


# def first_instance(cls, part, start=score.TimePoint(0), end=score.TimePoint(1)):
#     """
#     Reduce the result of first_instance_per_part, a 1D list, to an element
#     If there are multiple first instances then program aborts with error message

#     Parameters
#     ----------
#     cls:    class
#     part:   score.Part
#     start:  score.TimePoint, optional
#         start of the range to search in
#     end:    score.TimePoint, optional
#         end of the range to search in

#     Returns
#     -------
#     fi: instance of cls or None
#     """
#     fi = first_instance_per_part(cls, [part], start, end)

#     if len(fi) == 0:
#         return None

#     return fi[0]


# def common_signature(cls, sig_eql, parts, current_measures=None):
#     """
#     Calculate whether a list of parts has a common signature (as in key or time signature)

#     Parameters
#     ----------
#     cls:                score.KeySignature or score.TimeSignature
#     sig_eql:            function
#         takes 2 signature objects as input and returns whether they are equivalent (in some sense)
#     parts:              list of score.Part
#     current_measures:    list of score.Measure, optional
#         current as in the measures of the parts that are played at the same time and are processed

#     Returns
#     -------
#     common_sig:  instance of cls
#         might be None if there is no commonality between parts
#     """
#     sigs = None
#     if current_measures != None:
#         # HACK:  measures should probably not contain "pad" at this point, but an actual dummy measure with start and end times?
#         sigs = first_instance_per_part(
#             cls,
#             parts,
#             start=[cm.start if cm != "pad" else None for cm in current_measures],
#             end=[cm.end if cm != "pad" else None for cm in current_measures],
#         )
#     else:
#         sigs = first_instance_per_part(cls, parts)

#     if sigs == None or len(sigs) == 0 or None in sigs:
#         return None

#     common_sig = sigs.pop()

#     for sig in sigs:
#         if sig.start.t != common_sig.start.t or not sig_eql(sig, common_sig):
#             return None

#     return common_sig


# def vertical_slice(list_2d, index):
#     """
#     Returns elements of the sublists at index in a 1D list
#     all sublists of list_2d have to have len > index
#     """
#     vslice = []

#     for list_1d in list_2d:
#         vslice.append(list_1d[index])

#     return vslice


# def time_sig_eql(ts1, ts2):
#     """
#     equivalence function for score.TimeSignature objects
#     """
#     return ts1.beats == ts2.beats and ts1.beat_type == ts2.beat_type


# def key_sig_eql(ks1, ks2):
#     """
#     equivalence function for score.KeySignature objects
#     """
#     return ks1.name == ks2.name and ks1.fifths == ks2.fifths


# def idx(len_obj):
#     return range(len(len_obj))


# def attribs_of_clef(clef):
#     """
#     Returns values of a score.Clef object necessary for a MEI document

#     Parameters
#     ----------
#     clef: score.Clef

#     Returns
#     -------
#     sign: string
#         shape of clef (F,G, etc)
#     line:
#         which line to place clef on
#     """
#     sign = clef.sign

#     if sign == "percussion":
#         sign = "perc"

#     if clef.octave_change != None and clef.octave_change != 0:
#         place = "above"

#         if clef.octave_change < 0:
#             place = "below"

#         return sign, clef.line, 1 + 7 * abs(clef.octave_change), place

#     return sign, clef.line


# def create_staff_def(staff_grp, clef):
#     """

#     Parameters
#     ----------
#     staff_grp:   etree.SubElement
#     clef:       score.Clef
#     """
#     staff_def = add_child(staff_grp, "staffDef")

#     attribs = attribs_of_clef(clef)
#     set_attributes(
#         staff_def,
#         ("n", clef.number),
#         ("lines", 5),
#         ("clef.shape", attribs[0]),
#         ("clef.line", attribs[1]),
#     )
#     if len(attribs) == 4:
#         set_attributes(
#             staff_def, ("clef.dis", attribs[2]), ("clef.dis.place", attribs[3])
#         )


# def pad_measure(s, measure_per_staff, notes_within_measure_per_staff, auto_rest_count):
#     """
#     Adds a fake measure ("pad") to the measures of the staff s and a score.Rest object to the notes

#     Parameters
#     ----------
#     s:                              int
#         staff number
#     measure_per_staff:               dict of score.Measure objects
#     notes_within_measure_per_staff:   dict of lists of score.GenericNote objects
#     auto_rest_count:                  int
#         a counter for all the score.Rest objects that are created automatically

#     Returns
#     -------
#     incremented auto rest counter
#     """

#     measure_per_staff[s] = "pad"
#     r = score.Rest(id="pR" + str(auto_rest_count), voice=1)
#     r.start = score.TimePoint(0)
#     r.end = r.start

#     extend_key(notes_within_measure_per_staff, s, r)
#     return auto_rest_count + 1


# class InbetweenNotesElement:
#     """
#     InbetweenNotesElements contain information on objects like clefs, keysignatures, etc
#     within the score and how to process them

#     Parameters
#     ----------
#     name:           string
#         name of the element used in MEI
#     attrib_names:    list of strings
#         names of the attributes of the MEI element
#     attrib_vals_of:   function
#         a function that returns the attribute values of elem
#     container_dict: dict of lists of partitura objects
#         the container containing the required elements is at staff
#     staff:          int
#         staff number
#     skip_index:      int
#         init value for the cursor i (might skip 0)

#     Attributes
#     ----------
#     name:           string
#         name of the element used in MEI
#     attrib_names:    list of strings
#         names of the attributes of the MEI element
#     elem:           instance of partitura object
#     attrib_vals_of:   function
#         a function that returns the attribute values of elem
#     container:      list of partitura objects
#         the container where elem gets its values from
#     i:              int
#         cursor that keeps track of position in container
#     """

#     __slots__ = ["name", "attrib_names", "attrib_vals_of", "container", "i", "elem"]

#     def __init__(
#         self, name, attrib_names, attrib_vals_of, container_dict, staff, skip_index
#     ):
#         self.name = name
#         self.attrib_names = attrib_names
#         self.attrib_vals_of = attrib_vals_of

#         self.i = 0
#         self.elem = None

#         if staff in container_dict.keys():
#             self.container = container_dict[staff]
#             if len(self.container) > skip_index:
#                 self.elem = self.container[skip_index]
#                 self.i = skip_index
#         else:
#             self.container = []


# def chord_rep(chords, chord_i):
#     return chords[chord_i][0]


# def handle_beam(open_up, parents):
#     """
#     Using a stack of MEI elements, opens and closes beams

#     Parameters
#     ----------
#     open_up:     boolean
#         flag that indicates whether to open or close recent beam
#     parents:    list of etree.SubElement
#         stack of MEI elements that contain the beam element

#     Returns
#     -------
#     unchanged open_up value
#     """
#     if open_up:
#         parents.append(add_child(parents[-1], "beam"))
#     else:
#         parents.pop()

#     return open_up


# def is_chord_in_tuplet(chord_i, tuplet_indices):
#     """
#     check if chord falls in the range of a tuplet

#     Parameters
#     ----------
#     chord_i:        int
#         index of chord within chords array
#     tuplet_indices:  list of int pairs
#         contains the index ranges of all the tuplets in a measure of a staff

#     Returns
#     -------
#     whether chord falls in the range of a tuplet
#     """
#     for start, stop in tuplet_indices:
#         if start <= chord_i and chord_i <= stop:
#             return True

#     return False


# def calc_num_to_numbase_ratio(chord_i, chords, tuplet_indices):
#     """
#     calculates how to scale a notes duration with regard to the tuplet it is in

#     Parameters
#     ----------
#     chord_i:        int
#         index of chord within chords array
#     chords:         list of list of score.GenericNote
#         array of chords (which are lists of notes)
#     tuplet_indices:  list of int pairs
#         contains the index ranges of all the tuplets in a measure of a staff

#     Returns
#     -------
#     the num to numbase ratio of a tuplet (eg. 3 in 2 tuplet is 1.5)
#     """
#     rep = chords[chord_i][0]
#     if not isinstance(rep, score.GraceNote) and is_chord_in_tuplet(
#         chord_i, tuplet_indices
#     ):
#         return (
#             rep.symbolic_duration["actual_notes"]
#             / rep.symbolic_duration["normal_notes"]
#         )
#     return 1


# def process_chord(
#     chord_i,
#     chords,
#     inbetween_notes_elements,
#     open_beam,
#     auto_beaming,
#     parents,
#     dur_dots,
#     split_notes,
#     first_temp_dur,
#     tuplet_indices,
#     ties,
#     measure,
#     layer,
#     tuplet_id_counter,
#     open_tuplet,
#     last_key_sig,
#     note_alterations,
#     notes_next_measure_per_staff,
#     next_dur_dots=None,
# ):
#     """
#     creates <note>, <chord>, <rest>, etc elements from chords
#     also creates <beam>, <tuplet>, etc elements if necessary for chords objects
#     also creates <clef>, <keySig>, etc elements before chord objects from inbetween_notes_elements

#     Parameters
#     ----------
#     chord_i:                    int
#         index of chord within chords array
#     chords:                     list of list of score.GenericNote
#         chord array
#     inbetween_notes_elements:     list of InbetweenNotesElements
#         check this to see if something like clef needs to get inserted before chord
#     open_beam:                   boolean
#         flag that indicates whether a beam is currently open
#     auto_beaming:                boolean
#         flag that determines if automatic beams should be created or if it is kept manual
#     parents:                    list of etree.SubElement
#         stack of MEI elements that contain the most recent beam element
#     dur_dots:                   list of int pairs
#         describes how the chord actually gets notated via tied notes, each pair contains the duration of the notated note and its dot count
#     split_notes:                 list
#         this is either empty or None
#         if None, nothing is done with this
#         if an empty list, that means this chord crosses into the next measure and a chord is created for the next measure which is tied to this one
#     first_temp_dur:               int
#         amount of ticks (as in partitura) of the first notated note
#     tuplet_indices:              list of int pairs
#         the ranges of tuplets within the chords array
#     ties:                       dict
#         out parameter, contains pairs of IDs which need to be connected via ties
#         this function also adds to that
#     measure:                    score.Measure

#     layer:                      etree.SubElement
#         the parent element of the elements created here
#     tuplet_id_counter:           int

#     open_tuplet:                 boolean
#         describes if a tuplet is open or not
#     last_key_sig:                 score.KeySignature
#         the key signature this chord should be interpeted in
#     note_alterations:            dict
#         contains the alterations of staff positions (notes) that are relevant for this chord
#     notes_next_measure_per_staff: dict of lists of score.GenericNote
#         out parameter, add the result of split_notes into this
#     next_dur_dots:              list of int pairs, optional
#         needed for proper beaming

#     Returns
#     -------
#     tuplet_id_counter:    int
#         incremented if tuplet created
#     open_beam:           boolean
#         eventually modified if beam opened or closed
#     open_tuplet:         boolean
#         eventually modified if tuplet opened or closed
#     """

#     chord_notes = chords[chord_i]
#     rep = chord_notes[0]

#     for ine in inbetween_notes_elements:
#         if insert_elem_check(rep.start.t, [ine]):
#             # note should maybe be split according to keysig or clef etc insertion time, right now only beaming is disrupted
#             if open_beam and auto_beaming:
#                 open_beam = handle_beam(False, parents)

#             xml_elem = add_child(parents[-1], ine.name)
#             attrib_vals = ine.attrib_vals_of(ine.elem)

#             if ine.name == "keySig":
#                 last_key_sig = ine.elem

#             if len(ine.attrib_names) < len(attrib_vals):
#                 raise ValueError(
#                     "ERROR at insertion of inbetween_notes_elements: there are more attribute values than there are attribute names for xml element "
#                     + ine.name
#                 )

#             for nv in zip(ine.attrib_names[: len(attrib_vals)], attrib_vals):
#                 set_attributes(xml_elem, nv)

#             if ine.i + 1 >= len(ine.container):
#                 ine.elem = None
#             else:
#                 ine.i += 1
#                 ine.elem = ine.container[ine.i]

#     if is_chord_in_tuplet(chord_i, tuplet_indices):
#         if not open_tuplet:
#             parents.append(add_child(parents[-1], "tuplet"))
#             num = rep.symbolic_duration["actual_notes"]
#             numbase = rep.symbolic_duration["normal_notes"]
#             set_attributes(
#                 parents[-1],
#                 (xml_id_string, "t" + str(tuplet_id_counter)),
#                 ("num", num),
#                 ("numbase", numbase),
#             )
#             tuplet_id_counter += 1
#             open_tuplet = True
#     elif open_tuplet:
#         parents.pop()
#         open_tuplet = False

#     def set_dur_dots(elem, dur_dots):
#         dur, dots = dur_dots
#         set_attributes(elem, ("dur", dur))

#         if dots > 0:
#             set_attributes(elem, ("dots", dots))

#     if isinstance(rep, score.Note):
#         if auto_beaming:
#             # for now all notes are beamed, however some rules should be obeyed there, see Note Beaming and Grouping

#             # check to close beam
#             if open_beam and (
#                 dur_dots[0][0] < 8
#                 or chord_i - 1 >= 0
#                 and type(rep) != type(chord_rep(chords, chord_i - 1))
#             ):
#                 open_beam = handle_beam(False, parents)

#             # check to open beam (maybe again)
#             if not open_beam and dur_dots[0][0] >= 8:
#                 # open beam if there are multiple "consecutive notes" which don't get interrupted by some element
#                 if len(dur_dots) > 1 and not insert_elem_check(
#                     rep.start.t + first_temp_dur, inbetween_notes_elements
#                 ):
#                     open_beam = handle_beam(True, parents)

#                 # open beam if there is just a single note that is not the last one in measure and next note in measure is of same type and fits in beam as well, without getting interrupted by some element
#                 elif (
#                     len(dur_dots) <= 1
#                     and chord_i + 1 < len(chords)
#                     and next_dur_dots[0][0] >= 8
#                     and type(rep) == type(chord_rep(chords, chord_i + 1))
#                     and not insert_elem_check(
#                         chord_rep(chords, chord_i + 1).start.t, inbetween_notes_elements
#                     )
#                 ):
#                     open_beam = handle_beam(True, parents)
#         elif (
#             open_beam
#             and chord_i > 0
#             and rep.beam != chord_rep(chords, chord_i - 1).beam
#         ):
#             open_beam = handle_beam(False, parents)

#         if not auto_beaming and not open_beam and rep.beam != None:
#             open_beam = handle_beam(True, parents)

#         def conditional_gracify(elem, rep, chord_i, chords):
#             if isinstance(rep, score.GraceNote):
#                 grace = "unacc"

#                 if rep.grace_type == "appoggiatura":
#                     grace = "acc"

#                 set_attributes(elem, ("grace", grace))

#                 if rep.steal_proportion != None:
#                     set_attributes(
#                         elem, ("grace.time", str(rep.steal_proportion * 100) + "%")
#                     )

#                 if chord_i == 0 or not isinstance(
#                     chord_rep(chords, chord_i - 1), score.GraceNote
#                 ):
#                     chords[chord_i] = [copy(n) for n in chords[chord_i]]

#                     for n in chords[chord_i]:
#                         n.tie_next = n.main_note

#         def create_note(parent, n, id, last_key_sig, note_alterations):
#             note = add_child(parent, "note")

#             step = n.step.lower()
#             set_attributes(
#                 note, (xml_id_string, id), ("pname", step), ("oct", n.octave)
#             )

#             if n.articulations != None and len(n.articulations) > 0:
#                 artics = []

#                 translation = {
#                     "accent": "acc",
#                     "staccato": "stacc",
#                     "tenuto": "ten",
#                     "staccatissimo": "stacciss",
#                     "spiccato": "spicc",
#                     "scoop": "scoop",
#                     "plop": "plop",
#                     "doit": "doit",
#                 }

#                 for a in n.articulations:
#                     if a in translation.keys():
#                         artics.append(translation[a])
#                 set_attributes(note, ("artic", " ".join(artics)))

#             sharps = ["f", "c", "g", "d", "a", "e", "b"]
#             flats = list(reversed(sharps))

#             staff_pos = step + str(n.octave)

#             alter = n.alter or 0

#             def set_accid(note, acc, note_alterations, staff_pos, alter):
#                 if (
#                     staff_pos in note_alterations.keys()
#                     and alter == note_alterations[staff_pos]
#                 ):
#                     return
#                 set_attributes(note, ("accid", acc))
#                 note_alterations[staff_pos] = alter

#             # sharpen note if: is sharp, is not sharpened by key or prev alt
#             # flatten note if: is flat, is not flattened by key or prev alt
#             # neutralize note if: is neutral, is sharpened/flattened by key or prev alt

#             # check if note is sharpened/flattened by prev alt or key
#             if (
#                 staff_pos in note_alterations.keys()
#                 and note_alterations[staff_pos] != 0
#                 or last_key_sig.fifths > 0
#                 and step in sharps[: last_key_sig.fifths]
#                 or last_key_sig.fifths < 0
#                 and step in flats[: -last_key_sig.fifths]
#             ):
#                 if alter == 0:
#                     set_accid(note, "n", note_alterations, staff_pos, alter)
#             elif alter > 0:
#                 set_accid(note, "s", note_alterations, staff_pos, alter)
#             elif alter < 0:
#                 set_accid(note, "f", note_alterations, staff_pos, alter)

#             return note

#         if len(chord_notes) > 1:
#             chord = add_child(parents[-1], "chord")

#             set_dur_dots(chord, dur_dots[0])

#             conditional_gracify(chord, rep, chord_i, chords)

#             for n in chord_notes:
#                 create_note(chord, n, n.id, last_key_sig, note_alterations)

#         else:
#             note = create_note(parents[-1], rep, rep.id, last_key_sig, note_alterations)
#             set_dur_dots(note, dur_dots[0])

#             conditional_gracify(note, rep, chord_i, chords)

#         if len(dur_dots) > 1:
#             for n in chord_notes:
#                 ties[n.id] = [n.id]

#             def create_split_up_notes(chord_notes, i, parents, dur_dots, ties, rep):
#                 if len(chord_notes) > 1:
#                     chord = add_child(parents[-1], "chord")
#                     set_dur_dots(chord, dur_dots[i])

#                     for n in chord_notes:
#                         id = n.id + "-" + str(i)

#                         ties[n.id].append(id)
#                         create_note(chord, n, id, last_key_sig, note_alterations)
#                 else:
#                     id = rep.id + "-" + str(i)

#                     ties[rep.id].append(id)

#                     note = create_note(
#                         parents[-1], rep, id, last_key_sig, note_alterations
#                     )

#                     set_dur_dots(note, dur_dots[i])

#             for i in range(1, len(dur_dots) - 1):
#                 if not open_beam and dur_dots[i][0] >= 8:
#                     open_beam = handle_beam(True, parents)

#                 create_split_up_notes(chord_notes, i, parents, dur_dots, ties, rep)

#             create_split_up_notes(
#                 chord_notes, len(dur_dots) - 1, parents, dur_dots, ties, rep
#             )

#         if split_notes != None:

#             for n in chord_notes:
#                 split_notes.append(score.Note(n.step, n.octave, id=n.id + "s"))

#             if len(dur_dots) > 1:
#                 for n in chord_notes:
#                     ties[n.id].append(n.id + "s")
#             else:
#                 for n in chord_notes:
#                     ties[n.id] = [n.id, n.id + "s"]

#         for n in chord_notes:
#             if n.tie_next != None:
#                 if n.id in ties.keys():
#                     ties[n.id].append(n.tie_next.id)
#                 else:
#                     ties[n.id] = [n.id, n.tie_next.id]

#     elif isinstance(rep, score.Rest):
#         if split_notes != None:
#             split_notes.append(score.Rest(id=rep.id + "s"))

#         if (
#             measure == "pad"
#             or measure.start.t == rep.start.t
#             and measure.end.t == rep.end.t
#         ):
#             rest = add_child(layer, "mRest")

#             set_attributes(rest, (xml_id_string, rep.id))
#         else:
#             rest = add_child(layer, "rest")

#             set_attributes(rest, (xml_id_string, rep.id))

#             set_dur_dots(rest, dur_dots[0])

#             for i in range(1, len(dur_dots)):
#                 rest = add_child(layer, "rest")

#                 id = rep.id + str(i)

#                 set_attributes(rest, (xml_id_string, id))
#                 set_dur_dots(rest, dur_dots[i])

#     if split_notes != None:
#         for sn in split_notes:
#             sn.voice = rep.voice
#             sn.start = measure.end
#             sn.end = score.TimePoint(rep.start.t + rep.duration)

#             extend_key(notes_next_measure_per_staff, s, sn)

#     return tuplet_id_counter, open_beam, open_tuplet


# def create_score_def(measures, measure_i, parts, parent):
#     """
#     creates <score_def>

#     Parameters
#     ----------
#     measures:   list of score.Measure
#     measure_i:  int
#         index of measure currently processed within measures
#     parts:      list of score.Part
#     parent:     etree.SubElement
#         parent of <score_def>
#     """
#     reference_measures = vertical_slice(measures, measure_i)

#     common_key_sig = common_signature(
#         score.KeySignature, key_sig_eql, parts, reference_measures
#     )
#     common_time_sig = common_signature(
#         score.TimeSignature, time_sig_eql, parts, reference_measures
#     )

#     score_def = None

#     if common_key_sig != None or common_time_sig != None:
#         score_def = add_child(parent, "scoreDef")

#     if common_key_sig != None:
#         fifths, mode, pname = attribs_of_key_sig(common_key_sig)

#         set_attributes(
#             score_def, ("key.sig", fifths), ("key.mode", mode), ("key.pname", pname)
#         )

#     if common_time_sig != None:
#         set_attributes(
#             score_def,
#             ("meter.count", common_time_sig.beats),
#             ("meter.unit", common_time_sig.beat_type),
#         )

#     return score_def


# class MeasureContent:
#     """
#     Simply a bundle for all the data of a measure that needs to be processed for a MEI document

#     Attributes
#     ----------
#     ties_per_staff:      dict of lists
#     clefs_per_staff:     dict of lists
#     key_sigs_per_staff:   dict of lists
#     time_sigs_per_staff:  dict of lists
#     measure_per_staff:   dict of lists
#     tuplets_per_staff:   dict of lists
#     slurs:              list
#     dirs:               list
#     dynams:             list
#     tempii:             list
#     fermatas:           list
#     """

#     __slots__ = [
#         "ties_per_staff",
#         "clefs_per_staff",
#         "key_sigs_per_staff",
#         "time_sigs_per_staff",
#         "measure_per_staff",
#         "tuplets_per_staff",
#         "slurs",
#         "dirs",
#         "dynams",
#         "tempii",
#         "fermatas",
#     ]

#     def __init__(self):
#         self.ties_per_staff = {}
#         self.clefs_per_staff = {}
#         self.key_sigs_per_staff = {}
#         self.time_sigs_per_staff = {}
#         self.measure_per_staff = {}
#         self.tuplets_per_staff = {}

#         self.slurs = []
#         self.dirs = []
#         self.dynams = []
#         self.tempii = []
#         self.fermatas = []


# def extract_from_measures(
#     parts,
#     measures,
#     measure_i,
#     staves_per_part,
#     auto_rest_count,
#     notes_within_measure_per_staff,
# ):
#     """
#     Returns a bundle of data regarding the measure currently processed, things like notes, key signatures, etc
#     Also creates padding measures, necessary for example, for staves of instruments which do not play in the current measure

#     Parameters
#     ----------
#     parts:                          list of score.Part
#     measures:                       list of score.Measure
#     measure_i:                      int
#         index of current measure within measures
#     staves_per_part:                 dict of list of ints
#         staff enumeration partitioned by part
#     auto_rest_count:                  int
#         counter for the IDs of automatically generated rests
#     notes_within_measure_per_staff:   dict of lists of score.GenericNote
#         in and out parameter, might contain note objects that have crossed from previous measure into current one

#     Returns
#     -------
#     auto_rest_count:                  int
#         incremented if score.Rest created
#     current_measure_content:          MeasureContent
#         bundle for all the data that is extracted from the currently processed measure
#     """
#     current_measure_content = MeasureContent()

#     for part_i, part in enumerate(parts):
#         m = measures[part_i][measure_i]

#         if m == "pad":
#             for s in staves_per_part[part_i]:
#                 auto_rest_count = pad_measure(
#                     s,
#                     current_measure_content.measure_per_staff,
#                     notes_within_measure_per_staff,
#                     auto_rest_count,
#                 )

#             continue

#         def cls_within_measure(part, cls, measure, incl_subcls=False):
#             return part.iter_all(
#                 cls, measure.start, measure.end, include_subclasses=incl_subcls
#             )

#         def cls_within_measure_list(part, cls, measure, incl_subcls=False):
#             return list(cls_within_measure(part, cls, measure, incl_subcls))

#         clefs_within_measure_per_staff_per_part = partition_handle_none(
#             lambda c: c.number, cls_within_measure(part, score.Clef, m), "number"
#         )
#         key_sigs_within_measure = cls_within_measure_list(part, score.KeySignature, m)
#         time_sigs_within_measure = cls_within_measure_list(part, score.TimeSignature, m)
#         current_measure_content.slurs.extend(cls_within_measure(part, score.Slur, m))
#         tuplets_within_measure = cls_within_measure_list(part, score.Tuplet, m)

#         beat_map = part.beat_map

#         def calc_tstamp(beat_map, t, measure):
#             return beat_map(t) - beat_map(measure.start.t) + 1

#         for w in cls_within_measure(part, score.Words, m):
#             tstamp = calc_tstamp(beat_map, w.start.t, m)
#             current_measure_content.dirs.append((tstamp, w))

#         for tempo in cls_within_measure(part, score.Tempo, m):
#             tstamp = calc_tstamp(beat_map, tempo.start.t, m)
#             current_measure_content.tempii.append(
#                 (tstamp, staves_per_part[part_i][0], tempo)
#             )

#         for fermata in cls_within_measure(part, score.Fermata, m):
#             tstamp = calc_tstamp(beat_map, fermata.start.t, m)
#             current_measure_content.fermatas.append((tstamp, fermata.ref.staff))

#         for dynam in cls_within_measure(part, score.Direction, m, True):
#             tstamp = calc_tstamp(beat_map, dynam.start.t, m)
#             tstamp2 = None

#             if dynam.end != None:
#                 measure_counter = measure_i
#                 while True:
#                     if dynam.end.t <= measures[part_i][measure_counter].end.t:
#                         tstamp2 = calc_tstamp(
#                             beat_map, dynam.end.t, measures[part_i][measure_counter]
#                         )

#                         tstamp2 = str(measure_counter - measure_i) + "m+" + str(tstamp2)

#                         break
#                     elif (
#                         measure_counter + 1 >= len(measures[part_i])
#                         or measures[part_i][measure_counter + 1] == "pad"
#                     ):
#                         raise ValueError(
#                             "A score.Direction instance has an end time that exceeds actual non-padded measures"
#                         )
#                     else:
#                         measure_counter += 1

#             current_measure_content.dynams.append((tstamp, tstamp2, dynam))

#         notes_within_measure_per_staff_per_part = partition_handle_none(
#             lambda n: n.staff,
#             cls_within_measure(part, score.GenericNote, m, True),
#             "staff",
#         )

#         for s in staves_per_part[part_i]:
#             current_measure_content.key_sigs_per_staff[s] = key_sigs_within_measure
#             current_measure_content.time_sigs_per_staff[s] = time_sigs_within_measure
#             current_measure_content.tuplets_per_staff[s] = tuplets_within_measure

#             if s not in notes_within_measure_per_staff_per_part.keys():
#                 auto_rest_count = pad_measure(
#                     s,
#                     current_measure_content.measure_per_staff,
#                     notes_within_measure_per_staff,
#                     auto_rest_count,
#                 )

#         for s, nwp in notes_within_measure_per_staff_per_part.items():
#             extend_key(notes_within_measure_per_staff, s, nwp)
#             current_measure_content.measure_per_staff[s] = m

#         for s, cwp in clefs_within_measure_per_staff_per_part.items():
#             current_measure_content.clefs_per_staff[s] = cwp

#     return auto_rest_count, current_measure_content


# def create_measure(
#     section,
#     measure_i,
#     staves_sorted,
#     notes_within_measure_per_staff,
#     score_def,
#     tuplet_id_counter,
#     auto_beaming,
#     last_key_sig_per_staff,
#     current_measure_content,
# ):
#     """
#     creates a <measure> element within <section>
#     also returns an updated id counter for tuplets and a dictionary of notes that cross into the next measure

#     Parameters
#     ----------
#     section:                        etree.SubElement
#     measure_i:                      int
#         index of the measure created
#     staves_sorted:                  list of ints
#         a sorted list of the proper staff enumeration of the score
#     notes_within_measure_per_staff:   dict of lists of score.GenericNote
#         contains score.Note, score.Rest, etc objects of the current measure, partitioned by staff enumeration
#         will be further partitioned and sorted by voice, time and type (score.GraceNote) and eventually gathered into
#         a list of equivalence classes called chords
#     score_def:                       etree.SubElement
#     tuplet_id_counter:               int
#         tuplets usually don't come with IDs, so an automatic counter takes care of that
#     auto_beaming:                    boolean
#         enables automatic beaming
#     last_key_sig_per_staff:            dict of score.KeySignature
#         keeps track of the keysignature each staff is currently in
#     current_measure_content:          MeasureContent
#         contains all sorts of data for the measure like tuplets, slurs, etc

#     Returns
#     -------
#     tuplet_id_counter:               int
#         incremented if tuplet created
#     notes_next_measure_per_staff:     dict of lists of score.GenericNote
#         score.GenericNote objects that cross into the next measure
#     """
#     measure = add_child(section, "measure")
#     set_attributes(measure, ("n", measure_i + 1))

#     ties_per_staff = {}

#     for s in staves_sorted:
#         note_alterations = {}

#         staff = add_child(measure, "staff")

#         set_attributes(staff, ("n", s))

#         notes_within_measure_per_staff_per_voice = partition_handle_none(
#             lambda n: n.voice, notes_within_measure_per_staff[s], "voice"
#         )

#         ties_per_staff_per_voice = {}

#         m = current_measure_content.measure_per_staff[s]

#         tuplets = []
#         if s in current_measure_content.tuplets_per_staff.keys():
#             tuplets = current_measure_content.tuplets_per_staff[s]

#         last_key_sig = last_key_sig_per_staff[s]

#         for voice, notes in notes_within_measure_per_staff_per_voice.items():
#             layer = add_child(staff, "layer")

#             set_attributes(layer, ("n", voice))

#             ties = {}

#             notes_partition = partition_handle_none(
#                 lambda n: n.start.t, notes, "start.t"
#             )

#             chords = []

#             for t in sorted(notes_partition.keys()):
#                 ns = notes_partition[t]

#                 if len(ns) > 1:
#                     type_partition = partition_handle_none(
#                         lambda n: isinstance(n, score.GraceNote), ns, "isGraceNote"
#                     )

#                     if True in type_partition.keys():
#                         gns = type_partition[True]

#                         gn_chords = []

#                         def scan_backwards(gns):
#                             start = gns[0]

#                             while isinstance(start.grace_prev, score.GraceNote):
#                                 start = start.grace_prev

#                             return start

#                         start = scan_backwards(gns)

#                         def process_grace_note(n, gns):
#                             if not n in gns:
#                                 raise ValueError(
#                                     "Error at forward scan of GraceNotes: a grace_next has either different staff, voice or starting time than GraceNote chain"
#                                 )
#                             gns.remove(n)
#                             return n.grace_next

#                         while isinstance(start, score.GraceNote):
#                             gn_chords.append([start])
#                             start = process_grace_note(start, gns)

#                         while len(gns) > 0:
#                             start = scan_backwards(gns)

#                             i = 0
#                             while isinstance(start, score.GraceNote):
#                                 if i >= len(gn_chords):
#                                     raise IndexError(
#                                         "ERROR at GraceNote-forward scanning: Difference in lengths of grace note sequences for different chord notes"
#                                     )
#                                 gn_chords[i].append(start)
#                                 start = process_grace_note(start, gns)
#                                 i += 1

#                             if not i == len(gn_chords):
#                                 raise IndexError(
#                                     "ERROR at GraceNote-forward scanning: Difference in lengths of grace note sequences for different chord notes"
#                                 )

#                         for gnc in gn_chords:
#                             chords.append(gnc)

#                     if not False in type_partition.keys():
#                         raise KeyError(
#                             "ERROR at ChordNotes-grouping: GraceNotes detected without additional regular Notes at same time; staff "
#                             + str(s)
#                         )

#                     reg_notes = type_partition[False]

#                     rep = reg_notes[0]

#                     for i in range(1, len(reg_notes)):
#                         n = reg_notes[i]

#                         if n.duration != rep.duration:
#                             raise ValueError(
#                                 "In staff " + str(s) + ",",
#                                 "in measure " + str(m.number) + ",",
#                                 "for voice " + str(voice) + ",",
#                                 "2 notes start at time " + str(n.start.t) + ",",
#                                 "but have different durations, namely "
#                                 + n.id
#                                 + " has duration "
#                                 + str(n.duration)
#                                 + " and "
#                                 + rep.id
#                                 + " has duration "
#                                 + str(rep.duration),
#                                 "change to same duration for a chord or change voice of one of the notes for something else",
#                             )
#                         # HACK: unpitched notes are treated as Rests right now
#                         elif not isinstance(rep, score.Rest) and not isinstance(
#                             n, score.Rest
#                         ):
#                             if rep.beam != n.beam:
#                                 print(
#                                     "WARNING: notes within chords don't share the same beam",
#                                     "specifically note "
#                                     + str(rep)
#                                     + " has beam "
#                                     + str(rep.beam),
#                                     "and note " + str(n) + " has beam " + str(n.beam),
#                                     "export still continues though",
#                                 )
#                             elif set(rep.tuplet_starts) != set(n.tuplet_starts) and set(
#                                 rep.tuplet_stops
#                             ) != set(n.tuplet_stops):
#                                 print(
#                                     "WARNING: notes within chords don't share same tuplets, export still continues though"
#                                 )
#                     chords.append(reg_notes)
#                 else:
#                     chords.append(ns)

#             tuplet_indices = []
#             for tuplet in tuplets:
#                 ci = 0
#                 start = -1
#                 stop = -1
#                 while ci < len(chords):
#                     for n in chords[ci]:
#                         if tuplet in n.tuplet_starts:
#                             start = ci
#                             break
#                     for n in chords[ci]:
#                         if tuplet in n.tuplet_stops:
#                             stop = ci
#                             break

#                     if start >= 0 and stop >= 0:
#                         if not start <= stop:
#                             raise ValueError(
#                                 "In measure " + str(measure_i + 1) + ",",
#                                 "in staff " + str(s) + ",",
#                                 "[" + str(tuplet) + "] stops before it starts?",
#                                 "start=" + str(start + 1) + "; stop=" + str(stop + 1),
#                             )
#                         tuplet_indices.append((start, stop))
#                         break

#                     ci += 1

#             parents = [layer]
#             open_beam = False

#             (
#                 next_dur_dots,
#                 next_split_notes,
#                 next_first_temp_dur,
#             ) = calc_dur_dots_split_notes_first_temp_dur(
#                 chords[0][0], m, calc_num_to_numbase_ratio(0, chords, tuplet_indices)
#             )

#             inbetween_notes_elements = [
#                 InbetweenNotesElement(
#                     "clef",
#                     ["shape", "line", "dis", "dis.place"],
#                     attribs_of_clef,
#                     current_measure_content.clefs_per_staff,
#                     s,
#                     int(measure_i == 0),
#                 ),
#                 InbetweenNotesElement(
#                     "keySig",
#                     ["sig", "mode", "pname", "sig.showchange"],
#                     (lambda ks: attribs_of_key_sig(ks) + ("true",)),
#                     current_measure_content.key_sigs_per_staff,
#                     s,
#                     int(score_def != None),
#                 ),
#                 InbetweenNotesElement(
#                     "meterSig",
#                     ["count", "unit"],
#                     lambda ts: (ts.beats, ts.beat_type),
#                     current_measure_content.time_sigs_per_staff,
#                     s,
#                     int(score_def != None),
#                 ),
#             ]

#             open_tuplet = False

#             notes_next_measure_per_staff = {}

#             for chord_i in range(len(chords) - 1):
#                 dur_dots, split_notes, first_temp_dur = (
#                     next_dur_dots,
#                     next_split_notes,
#                     next_first_temp_dur,
#                 )
#                 (
#                     next_dur_dots,
#                     next_split_notes,
#                     next_first_temp_dur,
#                 ) = calc_dur_dots_split_notes_first_temp_dur(
#                     chord_rep(chords, chord_i + 1),
#                     m,
#                     calc_num_to_numbase_ratio(chord_i + 1, chords, tuplet_indices),
#                 )
#                 tuplet_id_counter, open_beam, open_tuplet = process_chord(
#                     chord_i,
#                     chords,
#                     inbetween_notes_elements,
#                     open_beam,
#                     auto_beaming,
#                     parents,
#                     dur_dots,
#                     split_notes,
#                     first_temp_dur,
#                     tuplet_indices,
#                     ties,
#                     m,
#                     layer,
#                     tuplet_id_counter,
#                     open_tuplet,
#                     last_key_sig,
#                     note_alterations,
#                     notes_next_measure_per_staff,
#                     next_dur_dots,
#                 )

#             tuplet_id_counter, _, _ = process_chord(
#                 len(chords) - 1,
#                 chords,
#                 inbetween_notes_elements,
#                 open_beam,
#                 auto_beaming,
#                 parents,
#                 next_dur_dots,
#                 next_split_notes,
#                 next_first_temp_dur,
#                 tuplet_indices,
#                 ties,
#                 m,
#                 layer,
#                 tuplet_id_counter,
#                 open_tuplet,
#                 last_key_sig,
#                 note_alterations,
#                 notes_next_measure_per_staff,
#             )

#             ties_per_staff_per_voice[voice] = ties

#         ties_per_staff[s] = ties_per_staff_per_voice

#     for fermata in current_measure_content.fermatas:
#         tstamp = fermata[0]
#         fermata_staff = fermata[1]

#         f = add_child(measure, "fermata")
#         set_attributes(f, ("staff", fermata_staff), ("tstamp", tstamp))

#     for slur in current_measure_content.slurs:
#         s = add_child(measure, "slur")
#         if slur.start_note == None or slur.end_note == None:
#             raise ValueError("Slur is missing start or end")
#         set_attributes(
#             s,
#             ("staff", slur.start_note.staff),
#             ("startid", "#" + slur.start_note.id),
#             ("endid", "#" + slur.end_note.id),
#         )

#     for tstamp, word in current_measure_content.dirs:
#         d = add_child(measure, "dir")
#         set_attributes(d, ("staff", word.staff), ("tstamp", tstamp))
#         d.text = word.text

#     # smufl individual notes start with E1
#     # these are the last 2 digits of the codes
#     metronome_codes = {
#         "breve": "D0",
#         "whole": "D2",
#         "half": "D3",
#         "h": "D3",
#         "quarter": "D5",
#         "q": "D5",
#         "eighth": "D7",
#         "e": "D5",
#         "16th": "D9",
#         "32nd": "DB",
#         "64th": "DD",
#         "128th": "DF",
#         "256th": "E1",
#     }

#     for tstamp, staff, tempo in current_measure_content.tempii:
#         t = add_child(measure, "tempo")
#         set_attributes(t, ("staff", staff), ("tstamp", tstamp))

#         unit = str(tempo.unit)

#         dots = unit.count(".")

#         unit = unit[:-dots]

#         string_to_build = [
#             ' <rend fontname="VerovioText">&#xE1',
#             metronome_codes[unit or "q"],
#             ";",
#         ]

#         for i in range(dots):
#             string_to_build.append("&#xE1E7;")

#         string_to_build.append("</rend> = ")
#         string_to_build.append(str(tempo.bpm))

#         t.text = "".join(string_to_build)

#     for tstamp, tstamp2, dynam in current_measure_content.dynams:
#         if isinstance(dynam, score.DynamicLoudnessDirection):
#             d = add_child(measure, "hairpin")
#             form = (
#                 "cres"
#                 if isinstance(dynam, score.IncreasingLoudnessDirection)
#                 else "dim"
#             )
#             set_attributes(d, ("form", form))

#             # duration can also matter for other dynamics, might want to move this out of branch
#             if tstamp2 != None:
#                 set_attributes(d, ("tstamp2", tstamp2))
#         else:
#             d = add_child(measure, "dynam")
#             d.text = dynam.text

#         set_attributes(d, ("staff", dynam.staff), ("tstamp", tstamp))

#     for s, tps in ties_per_staff.items():

#         for v, tpspv in tps.items():

#             for ties in tpspv.values():

#                 for i in range(len(ties) - 1):
#                     tie = add_child(measure, "tie")

#                     set_attributes(
#                         tie,
#                         ("staff", s),
#                         ("startid", "#" + ties[i]),
#                         ("endid", "#" + ties[i + 1]),
#                     )

#     for s, k in current_measure_content.key_sigs_per_staff.items():
#         if len(k) > 0:
#             last_key_sig_per_staff[s] = max(k, key=lambda k: k.start.t)

#     return tuplet_id_counter, notes_next_measure_per_staff


# def unpack_part_group(part_grp, parts=[]):
#     """
#     Recursively gather individual parts into a list, flattening the tree of parts so to say

#     Parameters
#     ----------
#     part_grp:    score.PartGroup
#     parts:      list of score.Part, optional

#     Returns
#     -------
#     parts:      list of score.Part
#     """
#     for c in part_grp.children:
#         if isinstance(c, score.PartGroup):
#             unpack_part_group(c, parts)
#         else:
#             parts.append(c)

#     return parts


# def save_mei(
#     parts,
#     auto_beaming=True,
#     file_name="testResult",
#     title_text=None,
#     proper_staff_grp=False,
# ):
#     """
#     creates an MEI document based on the parts provided
#     So far only <score> is used and not <part> which means all the parts are gathered in one whole score and
#     no individual scores are defined for individual parts

#     Parameters
#     ----------
#     parts:              score.Part, score.PartGroup or list of score.Part
#     auto_beaming:       boolean, optional
#         if all beaming has been done manually then set to False
#         otherwise this flag can be used to enable automatic beaming (beaming rules are still in progess)
#     file_name:          string, optional
#         should not contain file extension, .mei will be added automatically
#     title_text:         string, optional
#         name of the piece, e.g. "Klaviersonate Nr. 14" or "WAP"
#         if not provided, a title will be derived from file_name
#     proper_staff_grp:   boolean, optional
#         if true,    group staves per part
#         else        group all staves together
#         default is false because Verovio doesn't seem to render multiple staff groups correctly (but that just might be because multiple staff groups are not generated correctly in this function)
#     """

#     if isinstance(parts, score.PartGroup):
#         parts = unpack_part_group(parts)
#     elif isinstance(parts, score.Part):
#         parts = [parts]

#     for p in parts:
#         score.sanitize_part(p)

#     mei = etree.Element("mei")

#     mei_head = add_child(mei, "meiHead")
#     music = add_child(mei, "music")

#     mei_head.set("xmlns", name_space)
#     file_desc = add_child(mei_head, "fileDesc")
#     title_stmt = add_child(file_desc, "titleStmt")
#     pub_stmt = add_child(file_desc, "pubStmt")
#     title = add_child(title_stmt, "title")
#     title.set("type", "main")

#     # derive a title for the piece from the file_name
#     if title_text == None:
#         cursor = len(file_name) - 1
#         while cursor >= 0 and file_name[cursor] != "/":
#             cursor -= 1

#         tmp = file_name[cursor + 1 :].split("_")
#         tmp = [s[:1].upper() + s[1:] for s in tmp]
#         title.text = " ".join(tmp)
#     else:
#         title.text = title_text

#     body = add_child(music, "body")
#     mdiv = add_child(body, "mdiv")
#     mei_score = add_child(mdiv, "score")

#     classes_with_staff = [score.GenericNote, score.Words, score.Direction]

#     staves_per_part = []

#     staves_are_valid = True

#     for p in parts:
#         tmp = {
#             staffed_obj.staff
#             for cls in classes_with_staff
#             for staffed_obj in p.iter_all(cls, include_subclasses=True)
#         }
#         tmp = tmp.union({clef.number for clef in p.iter_all(score.Clef)})
#         staves_per_part.append(list(tmp))

#         if None in staves_per_part[-1]:
#             staves_are_valid = False
#             staves_per_part[-1].remove(None)

#             staves_per_part[-1].append(
#                 (max(staves_per_part[-1]) if len(staves_per_part[-1]) > 0 else 0) + 1
#             )

#         staves_per_part[-1].sort()

#     if staves_are_valid:
#         staves_sorted = sorted([s for staves in staves_per_part for s in staves])

#         i = 0

#         while i + 1 < len(staves_sorted):
#             if staves_sorted[i] == staves_sorted[i + 1]:
#                 staves_are_valid = False
#                 break

#             i += 1

#     if not staves_are_valid:
#         staves_per_part_backup = staves_per_part

#         staves_sorted = []
#         staves_per_part = []

#         # ASSUMPTION: staves are >0
#         max_staff = 0
#         for staves in staves_per_part_backup:
#             if len(staves) == 0:
#                 staves_per_part.append([])
#             else:
#                 shift = [s + max_staff for s in staves]

#                 max_staff += max(staves)

#                 staves_sorted.extend(shift)
#                 staves_per_part.append(shift)

#         # staves_sorted.sort()

#         max_staff = 0
#         for i, p in enumerate(parts):
#             for cls in classes_with_staff:
#                 for staff_obj in p.iter_all(cls, include_subclasses=True):
#                     staff_obj.staff = max_staff + (
#                         staff_obj.staff
#                         if staff_obj.staff != None
#                         else max(staves_per_part_backup[i])
#                     )

#             for clef in p.iter_all(score.Clef):
#                 clef.number = max_staff + (
#                     clef.number
#                     if clef.number != None
#                     else max(staves_per_part_backup[i])
#                 )

#             max_staff += (
#                 max(staves_per_part_backup[i])
#                 if len(staves_per_part_backup[i]) > 0
#                 else 0
#             )

#     measures = [list(parts[0].iter_all(score.Measure))]
#     padding_required = False
#     max_length = len(measures[0])
#     for i in range(1, len(parts)):
#         m = list(parts[i].iter_all(score.Measure))

#         if len(m) > max_length:
#             max_length = len(m)

#         if not padding_required:
#             padding_required = len(m) != len(measures[0])

#         measures.append(m)

#     score_def = create_score_def(measures, 0, parts, mei_score)

#     score_def_setup = score_def

#     if score_def == None:
#         score_def_setup = add_child(mei_score, "scoreDef")

#     clefs_per_part = first_instances_per_part(score.Clef, parts)

#     for i in idx(clefs_per_part):
#         clefs_per_part[i] = partition_handle_none(
#             lambda c: c.number, clefs_per_part[i], "number"
#         )

#     if len(clefs_per_part) == 0:
#         create_staff_def(
#             staff_grp, score.Clef(sign="G", line=2, number=1, octave_change=0)
#         )
#     else:
#         staff_grp = add_child(score_def_setup, "staffGrp")
#         for staves in staves_per_part:
#             if proper_staff_grp:
#                 staff_grp = add_child(score_def_setup, "staffGrp")

#             for s in staves:
#                 clefs = None

#                 for clefs_per_staff in clefs_per_part:
#                     if s in clefs_per_staff.keys():
#                         clefs = clefs_per_staff[s]
#                         break

#                 if clefs != None:
#                     clef = clefs[0]
#                     if len(clefs) != 1:
#                         raise ValueError(
#                             "ERROR at staff_def creation: Staff "
#                             + str(clef.number)
#                             + " starts with more than 1 clef at t=0"
#                         )
#                     create_staff_def(staff_grp, clef)
#                 else:
#                     create_staff_def(
#                         staff_grp,
#                         score.Clef(sign="G", line=2, number=s, octave_change=0),
#                     )

#     section = add_child(mei_score, "section")

#     measures_are_aligned = True
#     if padding_required:
#         cursors = [0] * len(measures)
#         tempii = [None] * len(measures)

#         while measures_are_aligned:
#             compare_measures = {}
#             for i, m in enumerate(measures):
#                 if cursors[i] < len(m):
#                     compare_measures[i] = m[cursors[i]]
#                     cursors[i] += 1

#             if len(compare_measures) == 0:
#                 break

#             compm_keys = list(compare_measures.keys())

#             new_tempii = first_instance_per_part(
#                 score.Tempo,
#                 [p for i, p in enumerate(parts) if i in compm_keys],
#                 start=[cm.start for cm in compare_measures.values()],
#                 end=[cm.end for cm in compare_measures.values()],
#             )

#             if len(new_tempii) == 0:
#                 for k in compm_keys:
#                     new_tempii.append(tempii[k])
#             else:
#                 for i, nt in enumerate(new_tempii):
#                     if nt == None:
#                         new_tempii[i] = tempii[compm_keys[i]]
#                     else:
#                         tempii[compm_keys[i]] = nt

#             def norm_dur(m):
#                 return (m.end.t - m.start.t) // m.start.quarter

#             rep_i = 0
#             while rep_i < len(new_tempii) and new_tempii[rep_i] == None:
#                 rep_i += 1

#             if rep_i == len(new_tempii):
#                 continue

#             rep_dur = (
#                 norm_dur(compare_measures[compm_keys[rep_i]]) * new_tempii[rep_i].bpm
#             )

#             for i in range(rep_i + 1, len(compm_keys)):
#                 nt = new_tempii[i]

#                 if nt == None:
#                     continue

#                 m = compare_measures[compm_keys[i]]
#                 dur = norm_dur(m) * new_tempii[i].bpm

#                 if dur != rep_dur:
#                     measures_are_aligned = False
#                     break

#     tuplet_id_counter = 0

#     if measures_are_aligned:
#         time_offset = [0] * len(measures)

#         if padding_required:
#             for i, mp in enumerate(measures):
#                 ii = len(mp)
#                 time_offset[i] = mp[ii - 1].end.t
#                 while ii < max_length:
#                     mp.append("pad")
#                     ii += 1

#         notes_last_measure_per_staff = {}
#         auto_rest_count = 0

#         notes_within_measure_per_staff = notes_last_measure_per_staff

#         auto_rest_count, current_measure_content = extract_from_measures(
#             parts,
#             measures,
#             0,
#             staves_per_part,
#             auto_rest_count,
#             notes_within_measure_per_staff,
#         )

#         last_key_sig_per_staff = {}

#         for s, k in current_measure_content.key_sigs_per_staff.items():
#             last_key_sig_per_staff[s] = (
#                 min(k, key=lambda k: k.start.t) if len(k) > 0 else None
#             )

#         tuplet_id_counter, notes_last_measure_per_staff = create_measure(
#             section,
#             0,
#             staves_sorted,
#             notes_within_measure_per_staff,
#             score_def,
#             tuplet_id_counter,
#             auto_beaming,
#             last_key_sig_per_staff,
#             current_measure_content,
#         )

#         for measure_i in range(1, len(measures[0])):
#             notes_within_measure_per_staff = notes_last_measure_per_staff

#             auto_rest_count, current_measure_content = extract_from_measures(
#                 parts,
#                 measures,
#                 measure_i,
#                 staves_per_part,
#                 auto_rest_count,
#                 notes_within_measure_per_staff,
#             )

#             score_def = create_score_def(measures, measure_i, parts, section)

#             tuplet_id_counter, notes_last_measure_per_staff = create_measure(
#                 section,
#                 measure_i,
#                 staves_sorted,
#                 notes_within_measure_per_staff,
#                 score_def,
#                 tuplet_id_counter,
#                 auto_beaming,
#                 last_key_sig_per_staff,
#                 current_measure_content,
#             )

#     (etree.ElementTree(mei)).write(file_name + ".mei", pretty_print=True)

#     # post processing step necessary
#     # etree won't write <,> and & into an element's text
#     with open(file_name + ".mei") as result:
#         text = list(result.read())
#         new_text = []

#         i = 0
#         while i < len(text):
#             ch = text[i]
#             if ch == "&":
#                 if text[i + 1 : i + 4] == ["l", "t", ";"]:
#                     ch = "<"
#                     i += 4
#                 elif text[i + 1 : i + 4] == ["g", "t", ";"]:
#                     ch = ">"
#                     i += 4
#                 elif text[i + 1 : i + 5] == ["a", "m", "p", ";"]:
#                     i += 5
#                 else:
#                     i += 1
#             else:
#                 i += 1

#             new_text.append(ch)

#         new_text = "".join(new_text)

#     with open(file_name + ".mei", "w") as result:
#         result.write(new_text)
