from lxml import etree
from xmlschema.names import XML_NAMESPACE
import partitura.score as score
from partitura.utils.music import (
    MEI_DURS_TO_SYMBOLIC,
    SYMBOLIC_TO_INT_DURS,
    SIGN_TO_ALTER,
    estimate_symbolic_duration,
)

import numpy as np


def load_mei(mei_path: str) -> list:
    """
    Loads a Mei score from path and returns a list of Partitura.Part

    Parameters
    ----------
    mei_path : str
        The path to an MEI score.

    Returns
    -------
    part_list : list
        A list of Partitura Part or GroupPart Objects.
    """
    parser = MeiParser(mei_path)
    # create parts from the specifications in the mei
    parser.create_parts()
    # fill parts with the content from the mei
    parser.fill_parts()
    return parser.parts


class MeiParser:
    def __init__(self, mei_path):
        document, ns = self._parse_mei(mei_path)
        self.document = document
        self.ns = ns  # the namespace in the MEI file
        self.parts = (
            None  # parts get initialized in create_parts() and filled in fill_parts()
        )

    def create_parts(self):
        # handle main scoreDef info: create the part list
        main_partgroup_el = self.document.find(self._ns_name("staffGrp", all=True))
        self.parts = self._handle_main_staff_group(main_partgroup_el)

    def fill_parts(self):
        # fill parts with the content of the score
        scores_el = self.document.findall(self._ns_name("score", all=True))
        if len(scores_el) != 1:
            raise Exception("Only MEI with a single score element are supported")
        sections_el = scores_el[0].findall(self._ns_name("section"))
        position = 0
        for section_el in sections_el:
            # insert in parts all elements except ties
            position = self._handle_section(
                section_el, list(score.iter_parts(self.parts)), position
            )

        # handles ties
        self._tie_notes(scores_el[0], self.parts)

    # -------------- Functions to initialize the xml tree -----------------

    def _ns_name(self, name, ns=None, all=False):
        """
        Combines document namespace tag with element to fetch object from MEI lxml trees.

        Parameters
        ----------
        name : str
            Name of MEI element.
        ns : str or None
            The namespace tag of the document. Default to None.
        all : bool
            if True, search the entire subtree, otherwise only the first level.
        """
        if ns is None:
            ns = self.ns

        if not all:
            return "{" + ns + "}" + name
        else:
            return ".//{" + ns + "}" + name

    def _parse_mei(self, mei_path):
        """
        Parses an MEI file from path to an lxml tree.

        Parameters
        ----------
        mei_path : str
            The path of the MEI document.
        Returns
        -------
        document : lxml tree
            An lxml tree of the MEI score.
        """
        parser = etree.XMLParser(
            resolve_entities=False,
            huge_tree=False,
            remove_comments=True,
            remove_blank_text=True,
        )
        document = etree.parse(mei_path, parser)
        # find the namespace
        ns = document.getroot().nsmap[None]
        # --> nsmap fetches a dict of the namespace Map, generally for root the key `None` fetches the namespace of the document.
        return document, ns

    # functions to parse staves info

    def _handle_metersig(self, staffdef_el, position, part):
        """
        Handles meter signature and adds to part.

        Parameters
        ----------
        staffdef_el : lxml etree
            A lxml substree of a staff's mei score.
        position : int
            Is the current position of the note on the timeline.
        part : particular.Part
            The created Partitura Part object.
        """
        metersig_el = staffdef_el.find(self._ns_name("meterSig"))
        if metersig_el is not None:  # new element inside
            numerator = int(metersig_el.attrib["count"])
            denominator = int(metersig_el.attrib["unit"])
        elif (
            staffdef_el.get("meter.count") is not None
        ):  # all encoded as attributes in staffdef
            numerator = int(staffdef_el.attrib["meter.count"])
            denominator = int(staffdef_el.attrib["meter.unit"])
        else:  # the informatio is encoded in a parent scoredef
            found_ancestor_with_metrical_info = False
            for anc in staffdef_el.iterancestors(tag=self._ns_name("scoreDef")):
                if anc.get("meter.count") is not None:
                    found_ancestor_with_metrical_info = True
                    break
            if found_ancestor_with_metrical_info:
                numerator = int(anc.attrib["meter.count"])
                denominator = int(anc.attrib["meter.unit"])
            else:
                raise Exception(
                    f"The time signature is not encoded in {staffdef_el.get(self._ns_name(id))} or in any ancestor scoreDef"
                )
        new_time_signature = score.TimeSignature(numerator, denominator)
        part.add(new_time_signature, position)

    def _handle_keysig(self, staffdef_el, position, part):
        """
        Handles key signature and adds to part.

        Parameters
        ----------
        staffdef_el : lxml tree
            A lxml substree of a staff's mei score.
        position : int
            Is the current position of the note on the timeline.
        part : particular.Part
            The created Partitura Part object.
        """
        keysig_el = staffdef_el.find(self._ns_name("keySig"))
        if keysig_el is not None:  # new element inside
            sig = keysig_el.attrib["sig"]
            # now extract partitura keysig parameters
            fifths = self._mei_sig_to_fifths(sig)
            mode = keysig_el.get("mode")
        elif (
            staffdef_el.get("key.sig") is not None
        ):  # all encoded as attributes in staffdef
            sig = staffdef_el.attrib["key.sig"]
            # now extract partitura keysig parameters
            fifths = self._mei_sig_to_fifths(sig)
            mode = staffdef_el.get("key.mode")
        else:  # the information is encoded in a parent scoredef
            found_ancestor_with_key_info = False
            for anc in staffdef_el.iterancestors(tag=self._ns_name("scoreDef")):
                if anc.get("key.sig") is not None:
                    found_ancestor_with_key_info = True
                    break
            if found_ancestor_with_key_info:
                sig = anc.attrib["key.sig"]
                # now extract partitura keysig parameters
                fifths = self._mei_sig_to_fifths(sig)
                mode = anc.get("key.mode")
            else:
                raise Exception(
                    f"The key signature is not encoded in {staffdef_el.get(self._ns_name(id))} or in any ancestor scoreDef"
                )

        new_key_signature = score.KeySignature(fifths, mode)
        part.add(new_key_signature, position)

    def _compute_clef_octave(self, dis, dis_place):
        if dis is not None:
            sign = -1 if dis_place == "below" else 1
            octave = sign * int(int(dis) / 8)
        else:
            octave = 0
        return octave

    def _mei_sig_to_fifths(self, sig):
        """Produces partitura KeySignature.fifths parameter from the MEI sig attribute."""
        if sig[0] == "0":
            fifths = 0
        else:
            sign = 1 if sig[-1] == "s" else -1
            fifths = sign * int(sig[:-1])
        return fifths

    def _handle_clef(self, element, position, part):
        """Inserts a clef. Element can be either a cleff element or staffdef element.

        Parameters
        ----------
        staffdef_el : lxml tree
            A lxml substree of a mei score.
        position : int
            Is the current position of the note on the timeline.
        part : particular.Part
            The created Partitura Part object.

        Returns
        -------
        position : int
            The current position of the note on the timeline.
        """
        # handle the case where we have clef informations inside staffdef el
        if element.tag == self._ns_name("staffDef"):
            clef_el = element.find(self._ns_name("clef"))
            if clef_el is not None:  # if there is a clef element inside
                return self._handle_clef(clef_el, position, part)
            else:  # if all info are in the staffdef element
                number = element.attrib["n"]
                sign = element.attrib["clef.shape"]
                line = element.attrib["clef.line"]
                octave = self._compute_clef_octave(
                    element.get("dis"), element.get("dis.place")
                )
        elif element.tag == self._ns_name("clef"):
            if element.get("sameas") is not None:  # this is a copy of another clef
                # it seems this is used in different layers for the same staff
                # we don't handle it to avoid clef duplications
                return position
            else:
                # find the staff number
                parent = element.getparent()
                if parent.tag == self._ns_name("staffDef"):
                    number = parent.attrib["n"]
                else:  # go back another level to staff element
                    number = parent.getparent().attrib["n"]
                sign = element.attrib["shape"]
                line = element.attrib["line"]
                octave = self._compute_clef_octave(
                    element.get("dis"), element.get("dis.place")
                )
        else:
            raise Exception("_handle_clef only accepts staffDef or clef elements")
        new_clef = score.Clef(int(number), sign, int(line), octave)
        part.add(new_clef, position)
        return position

    def _handle_staffdef(self, staffdef_el, position, part):
        """
        Derives meter, key and clef from lxml substree and pass them to part.

        Parameters
        ----------
        staffdef_el : lxml tree
            A lxml substree of a mei score.
        position : int
            Is the current position of the note on the timeline.
        part : particular.Part
            The created Partitura Part object.
        """
        # fill with time signature info
        self._handle_metersig(staffdef_el, position, part)
        # fill with key signature info
        self._handle_keysig(staffdef_el, position, part)
        # fill with clef info
        self._handle_clef(staffdef_el, position, part)

    def _intsymdur_from_symbolic(self, symbolic_dur):
        """Produce a int symbolic dur (e.g. 12 is a eight note triplet) and a dot number by looking at the symbolic dur dictionary:
        i.e., symbol, eventual tuplet ancestors."""
        intsymdur = SYMBOLIC_TO_INT_DURS[symbolic_dur["type"]]
        # deals with tuplets
        if symbolic_dur.get("actual_notes") is not None:
            assert symbolic_dur.get("normal_notes") is not None
            intsymdur = (
                intsymdur * symbolic_dur["actual_notes"] / symbolic_dur["normal_notes"]
            )
        # deals with dots
        dots = symbolic_dur.get("dots") if symbolic_dur.get("dots") is not None else 0
        return intsymdur, dots

    def _find_ppq(self):
        """Finds the ppq for MEI filed that do not explicitely encode this information"""
        els_with_dur = self.document.xpath(".//*[@dur]")
        durs = []
        for el in els_with_dur:
            symbolic_duration = self._get_symbolic_duration(el)
            intsymdur, dots = self._intsymdur_from_symbolic(symbolic_duration)
            # double the value if we have dots, to be sure be able to encode that with integers in partitura
            durs.append(intsymdur * (2 ** dots))

        # add 4 to be sure to not go under 1 ppq
        durs.append(4)

        # TODO : check if this can create problems with rounding of float durations
        least_common_multiple = np.lcm.reduce(np.array(durs, dtype=int))

        return least_common_multiple / 4

    def _handle_initial_staffdef(self, staffdef_el):
        """
        Handles the definition of a single staff.

        Parameters
        ----------
        staffdef_el : Element tree
            A subtree of a particular Staff from a score.

        Returns
        -------
        part : partitura.Part
            Returns a partitura part filled with meter, time signature, key signature information.
        """
        # Fetch the namespace of the staff.
        id = staffdef_el.attrib[self._ns_name("id", XML_NAMESPACE)]
        label_el = staffdef_el.find(self._ns_name("label"))
        name = label_el.text if label_el is not None else ""
        ppq_attrib = staffdef_el.get("ppq")
        if ppq_attrib is not None:
            ppq = int(ppq_attrib)
        else:
            ppq = self._find_ppq()
        # generate the part
        part = score.Part(id, name, quarter_duration=ppq)
        # fill it with other info, e.g. meter, time signature, key signature
        self._handle_staffdef(staffdef_el, 0, part)
        return part

    def _handle_staffgroup(self, staffgroup_el):
        """
        Handles a staffGrp. WARNING: in MEI piano staves are a staffGrp

        Parameters
        ----------
        staffgroup_el : element tree
            A subtree of Staff Group from a score.

        Returns
        -------
        staff_group : Partitura.PartGroup
            A partitura PartGroup object made by calling and appending as children ever staff separately.
        """
        group_symbol_el = staffgroup_el.find(self._ns_name("grpSym"))
        if group_symbol_el is None:
            group_symbol = staffgroup_el.attrib["symbol"]
        else:
            group_symbol = group_symbol_el.attrib["symbol"]
        label_el = staffgroup_el.find(self._ns_name("label"))
        name = label_el.text if label_el is not None else None
        id = staffgroup_el.attrib[self._ns_name("id", XML_NAMESPACE)]
        staff_group = score.PartGroup(group_symbol, group_name=name, id=id)
        staves_el = staffgroup_el.findall(self._ns_name("staffDef"))
        for s_el in staves_el:
            new_part = self._handle_initial_staffdef(s_el)
            staff_group.children.append(new_part)
        return staff_group

    def _handle_main_staff_group(self, main_staffgrp_el):
        """
        Handles the main staffGrp that contains all other staves or staff groups.

        Parameters
        ----------
        main_staffgrp_el : element_tree

        Returns
        -------
        part_list : list
            Created list of parts filled with key and time signature information.
        """
        staves_el = main_staffgrp_el.findall(self._ns_name("staffDef"))
        staff_groups_el = main_staffgrp_el.findall(self._ns_name("staffGrp"))
        # the list of parts or part groups
        part_list = []
        # process the parts
        # TODO add Parallelization to handle part parsing in parallel
        for s_el in staves_el:
            new_part = self._handle_initial_staffdef(s_el)
            part_list.append(new_part)
        # process the part groups
        for sg_el in staff_groups_el:
            new_staffgroup = self._handle_staffgroup(sg_el)
            part_list.append(new_staffgroup)
        return part_list

    # functions to parse the content of parts

    def _accidstring_to_int(self, accid_string: str) -> int:
        """Accidental string to intiger pitch manipulation."""
        if accid_string is None:
            return None
        else:
            return SIGN_TO_ALTER[accid_string]

    def _pitch_info(self, note_el):
        """
        Given a note element fetches PitchClassName, octave and accidental.

        Parameters
        ----------
        note_el

        Returns
        -------
        step : str
            The note Pitch class name.
        octave : int
            The number of octave
        alter : int
            Accidental string transformed to number.
        """
        step = note_el.attrib["pname"]
        octave = int(note_el.attrib["oct"])
        alter = self._accidstring_to_int(note_el.get("accid"))
        return step, octave, alter

    def _get_symbolic_duration(self, el):
        symbolic_duration = {}
        symbolic_duration["type"] = MEI_DURS_TO_SYMBOLIC[el.attrib["dur"]]
        if not el.get("dots") is None:
            symbolic_duration["dots"] = int(el.get("dots"))
        # find eventual time modifications
        tuplet_ancestors = list(el.iterancestors(tag=self._ns_name("tuplet")))
        if len(tuplet_ancestors) == 0:
            pass
        elif len(tuplet_ancestors) == 1:
            symbolic_duration["actual_notes"] = int(tuplet_ancestors[0].attrib["num"])
            symbolic_duration["normal_notes"] = int(
                tuplet_ancestors[0].attrib["numbase"]
            )
        else:
            raise Exception("Nested tuplets are not yet supported.")
        return symbolic_duration

    def _duration_info(self, el, part):
        """
        Extract duration info from a xml element.

        It works for example with note_el, chord_el

        Parameters
        ----------
        el : lxml tree
            the xml element to analyze
        part : partitura.Part
            The created partitura part object.

        Returns
        -------
        id :
        duration :
        symbolic_duration :
        """
        # symbolic duration
        symbolic_duration = self._get_symbolic_duration(el)

        # duration in ppq
        if el.get("dur.ppq") is not None or el.get("grace") is not None:
            # find duration in ppq. For grace notes is 0
            duration = 0 if el.get("grace") is not None else int(el.get("dur.ppq"))
        else:
            # compute the duration from the symbolic duration
            intsymdur, dots = self._intsymdur_from_symbolic(symbolic_duration)
            divs = part._quarter_durations[0]  # divs is the same as ppq
            duration = divs * 4 / intsymdur
            for d in range(dots):
                duration = duration + 0.5 * duration
            # sanity check to verify the divs are correctly set
            assert duration == int(duration)

        # find id
        id = el.attrib[self._ns_name("id", XML_NAMESPACE)]
        return id, int(duration), symbolic_duration

    def _handle_note(self, note_el, position, voice, staff, part) -> int:
        """
        Handles note elements and imports the to part.

        Parameters
        ----------
        note_el : lxml substree
            The lxml substree of a note element.
        position : int
            The current position on the timeline.
        voice : int
            The currect voice index.
        staff : int
            The current staff index.
        part : partitura.Part
            The created partitura part object.

        Returns
        -------
        position + duration : into
            The updated position on the timeline.
        """
        # find pitch info
        step, octave, alter = self._pitch_info(note_el)
        # find duration info
        note_id, duration, symbolic_duration = self._duration_info(note_el, part)
        # find if it's grace
        grace_attr = note_el.get("grace")
        if grace_attr is None:
            # create normal note
            note = score.Note(
                step=step,
                octave=octave,
                alter=alter,
                id=note_id,
                voice=voice,
                staff=staff,
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
        # add note to the part
        part.add(note, position, position + duration)
        # return duration to update the position in the layer
        return position + duration

    def _handle_rest(self, rest_el, position, voice, staff, part):
        """
        Handles the rest element updates part and position.

        Parameters
        ----------
        rest_el : lxml tree
            A rest element in the lxml tree.
        position : int
            The current position on the timeline.
        voice : int
            The voice of the section.
        staff : int
            The current staff also refers to a Part.
        part : Partitura.Part
            The created part to add elements to.

        Returns
        -------
        position + duration : int
            Next position on the timeline.
        Also adds the rest to the partitura part object.
        """
        # find duration info
        rest_id, duration, symbolic_duration = self._duration_info(rest_el, part)
        # create rest
        rest = score.Rest(
            id=rest_id,
            voice=voice,
            staff=staff,
            symbolic_duration=symbolic_duration,
            articulations=None,
        )
        # add rest to the part
        part.add(rest, position, position + duration)
        # return duration to update the position in the layer
        return position + duration

    def _handle_mrest(self, mrest_el, position, voice, staff, part):
        """
        Handles a rest that spawn the entire measure

        Parameters
        ----------
        mrest_el : lxml tree
            A mrest element in the lxml tree.
        position : int
            The current position on the timeline.
        voice : int
            The voice of the section.
        staff : int
            The current staff also refers to a Part.
        part : Partitura.Part
            The created part to add elements to.

        Returns
        -------
        position + duration : int
            Next position on the timeline.
        """
        # find id
        mrest_id = mrest_el.attrib[self._ns_name("id", XML_NAMESPACE)]
        # find closest time signature
        last_ts = list(part.iter_all(cls=score.TimeSignature))[-1]
        # find divs per measure
        ppq = part.quarter_duration_map(position)
        parts_per_measure = int(ppq * 4 * last_ts.beats / last_ts.beat_type)

        # create dummy rest to insert in the timeline
        rest = score.Rest(
            id=mrest_id,
            voice=voice,
            staff=staff,
            symbolic_duration=estimate_symbolic_duration(parts_per_measure, ppq),
            articulations=None,
        )
        # add mrest to the part
        part.add(rest, position, position + parts_per_measure)
        # now iterate
        # return duration to update the position in the layer
        return position + parts_per_measure

    def _handle_chord(self, chord_el, position, voice, staff, part):
        """
        Handles a rest that spawn the entire measure

        Parameters
        ----------
        chord_el : lxml tree
            A chord element in the lxml tree.
        position : int
            The current position on the timeline.
        voice : int
            The voice of the section.
        staff : int
            The current staff also refers to a Part.
        part : Partitura.Part
            The created part to add elements to.

        Returns
        -------
        position + duration : int
            Next position on the timeline.
        """
        # find duration info
        chord_id, duration, symbolic_duration = self._duration_info(chord_el, part)
        # find notes info
        notes_el = chord_el.findall(self._ns_name("note"))
        for note_el in notes_el:
            note_id = note_el.attrib[self._ns_name("id", XML_NAMESPACE)]
            # find pitch info
            step, octave, alter = self._pitch_info(note_el)
            # create note
            note = score.Note(
                step=step,
                octave=octave,
                alter=alter,
                id=note_id,
                voice=voice,
                staff=staff,
                symbolic_duration=symbolic_duration,
                articulations=None,  # TODO : add articulation
            )
            # add note to the part
            part.add(note, position, position + duration)
            # return duration to update the position in the layer
        return position + duration

    def _handle_space(self, e, position, part):
        """Moves current position."""
        space_id, duration, symbolic_duration = self._duration_info(e, part)
        return position + duration

    def _handle_layer_in_staff_in_measure(
        self, layer_el, ind_layer: int, ind_staff: int, position: int, part
    ) -> int:
        for i, e in enumerate(layer_el):
            if e.tag == self._ns_name("note"):
                new_position = self._handle_note(
                    e, position, ind_layer, ind_staff, part
                )
            elif e.tag == self._ns_name("chord"):
                new_position = self._handle_chord(
                    e, position, ind_layer, ind_staff, part
                )
            elif e.tag == self._ns_name("rest"):
                new_position = self._handle_rest(
                    e, position, ind_layer, ind_staff, part
                )
            elif e.tag == self._ns_name("mRest"):  # rest that spawn the entire measure
                new_position = self._handle_mrest(
                    e, position, ind_layer, ind_staff, part
                )
            elif e.tag == self._ns_name("beam"):
                # TODO : add Beam element
                # recursive call to the elements inside beam
                new_position = self._handle_layer_in_staff_in_measure(
                    e, ind_layer, ind_staff, position, part
                )
            elif e.tag == self._ns_name("tuplet"):
                # TODO : add Tuplet element
                # recursive call to the elements inside Tuplet
                new_position = self._handle_layer_in_staff_in_measure(
                    e, ind_layer, ind_staff, position, part
                )
            elif e.tag == self._ns_name("clef"):
                new_position = self._handle_clef(e, position, part)
            elif e.tag == self._ns_name("space"):
                new_position = self._handle_space(e, position, part)
            else:
                raise Exception("Tag " + e.tag + " not supported")

            # update the current position
            position = new_position
        return position

    def _handle_staff_in_measure(self, staff_el, staff_ind, position: int, part):
        """
        Handles staffs inside a measure element.

        Parameters
        ----------
        staff_el : lxml etree
            The lxml subtree for a staff element.
        staff_ind : int
            The Staff index.
        position : int
            The current position on the timeline.
        part : Partitura.Part
            The created partitura part object.

        Returns
        -------
        end_positions[0] : int
            The final position on the timeline.
        """
        # add measure
        measure = score.Measure(number=staff_el.getparent().get("n"))
        part.add(measure, position)

        layers_el = staff_el.findall(self._ns_name("layer"))
        end_positions = []
        for i_layer, layer_el in enumerate(layers_el):
            end_positions.append(
                self._handle_layer_in_staff_in_measure(
                    layer_el, i_layer, staff_ind, position, part
                )
            )
        # check if layers have equal duration (bad encoding, but it often happens)
        if not all([e == end_positions[0] for e in end_positions]):
            print(
                f"Warning: voices have different durations in staff {staff_el.attrib[self._ns_name('id',XML_NAMESPACE)]}"
            )

        # add end time of measure
        part.add(measure, None, max(end_positions))
        return max(end_positions)

    def _handle_section(self, section_el, parts, position: int):
        """
        Returns position and fills parts with elements.

        Parameters
        ----------
        section_el : lxml tree
            An lxml substree of a MEI score reffering to a section.
        parts : list()
            A list of partitura Parts.
        position : int
            The current position on the timeline.

        Returns
        -------
        position : int
            The end position of the section.
        """
        for i_el, element in enumerate(section_el):
            # handle measures
            if element.tag == self._ns_name("measure"):
                staves_el = element.findall(self._ns_name("staff"))
                if len(list(staves_el)) != len(list(parts)):
                    raise Exception("Not all parts are specified in measure" + i_el)
                end_positions = []
                for i_s, (part, staff_el) in enumerate(zip(parts, staves_el)):
                    end_positions.append(
                        self._handle_staff_in_measure(staff_el, i_s, position, part)
                    )
                # sanity check that all layers have equal duration
                if not all([e == end_positions[0] for e in end_positions]):
                    print(
                        f"Warning : parts have measures of different duration in measure {element.attrib[self._ns_name('id',XML_NAMESPACE)]}"
                    )
                position = max(end_positions)
            # handle staffDef elements
            elif element.tag == self._ns_name("scoreDef"):
                # meter modifications
                metersig_el = element.find(self._ns_name("meterSig"))
                if (metersig_el is not None) or (
                    element.get("meter.count") is not None
                ):
                    for part in parts:
                        self._handle_metersig(element, position, part)
                # key signature modifications
                keysig_el = element.find(self._ns_name("keySig"))
                if (keysig_el is not None) or (element.get("key.sig") is not None):
                    for part in parts:
                        self._handle_keysig(element, position, part)
            # handle nested section
            elif element.tag == self._ns_name("section"):
                position = self._handle_section(element, parts, position)
            elif element.tag == self._ns_name("ending"):
                position = self._handle_section(element, parts, position)
                # TODO : add the volta symbol and measure separator
            # explicit repetition expansions
            elif element.tag == self._ns_name("expansion"):
                pass
            # system break
            elif element.tag == self._ns_name("sb"):
                pass
            # page break
            elif element.tag == self._ns_name("pb"):
                pass
            else:
                raise Exception(f"element {element.tag} is not yet supported")

        return position

    def _tie_notes(self, section_el, part_list):
        """Ties all notes in a part.
        This function must be run after the parts are completely created."""
        # TODO : support ties written as attributes with @tie sintax
        ties_el = section_el.findall(self._ns_name("tie", all=True))
        # create a dict of id : note, to speed up search
        all_notes = [
            note
            for part in score.iter_parts(part_list)
            for note in part.iter_all(cls=score.Note)
        ]
        all_notes_dict = {note.id: note for note in all_notes}
        for tie_el in ties_el:
            start_id = tie_el.get("startid")
            end_id = tie_el.get("endid")
            if start_id is None or end_id is None:
                print(
                    f"Warning: tie {tie_el.attrib[self._ns_name('id',XML_NAMESPACE)]} is missing the a startid or endid"
                )
            else:
                # remove the # in first position
                start_id = start_id[1:]
                end_id = end_id[1:]
                # set tie prev and tie next in partira note objects
                all_notes_dict[start_id].tie_next = all_notes_dict[end_id]
                all_notes_dict[end_id].tie_prev = all_notes_dict[start_id]

