from lxml import etree
from xmlschema.names import XML_NAMESPACE
import partitura.score as score
from partitura.utils.music import MEI_DURS, SIGN_TO_ALTER


# functions to initialize the xml tree


def _parse_mei(xml_path):
    parser = etree.XMLParser(
        resolve_entities=False,
        huge_tree=False,
        remove_comments=True,
        remove_blank_text=True,
    )
    document = etree.parse(xml_path, parser)
    # find the namespace
    ns = document.getroot().nsmap[None]

    return document, ns


def _ns_name(name, ns, all=False):
    if not all:
        return "{" + ns + "}" + name
    else:
        return ".//{" + ns + "}" + name


# functions to parse staves info


def _handle_metersig(staffdef_el, position, part, ns):
    metersig_el = staffdef_el.find(_ns_name("meterSig", ns))
    if metersig_el is not None:  # new element inside
        numerator = int(metersig_el.attrib["count"])
        denominator = int(metersig_el.attrib["unit"])
    else:  # all encoded as attributes in staffdef
        numerator = int(staffdef_el.attrib["meter.count"])
        denominator = int(staffdef_el.attrib["meter.unit"])
    new_time_signature = score.TimeSignature(numerator, denominator)
    part.add(new_time_signature, position)


def _mei_sig_to_fifths(sig):
    """Produces partitura KeySignature.fifths parameter from the MEI sig attribute."""
    if sig[0] == "0":
        fifths = 0
    else:
        sign = 1 if sig[-1] == "s" else -1
        fifths = sign * int(sig[:-1])
    return fifths


def _handle_keysig(staffdef_el, position, part, ns):
    keysig_el = staffdef_el.find(_ns_name("keySig", ns))
    if keysig_el is not None:  # new element inside
        sig = keysig_el.attrib["sig"]
        # now extract partitura keysig parameters
        fifths = _mei_sig_to_fifths(sig)
        mode = keysig_el.get("mode")
    else:  # all encoded as attributes in staffdef
        sig = staffdef_el.attrib["key.sig"]
        # now extract partitura keysig parameters
        fifths = _mei_sig_to_fifths(sig)
        mode = staffdef_el.get("key.mode")
    new_key_signature = score.KeySignature(fifths, mode)
    part.add(new_key_signature, position)


def _compute_clef_octave(dis, dis_place):
    if dis is not None:
        sign = -1 if dis_place == "below" else 1
        octave = sign * int(int(dis) / 8)
    else:
        octave = 0
    return octave


def _handle_clef(element, position, part, ns):
    """Inserts a clef. Element can be either a cleff element or staffdef element."""
    # handle the case where we have clef informations inside staffdef el
    if element.tag == _ns_name("staffDef", ns):
        clef_el = element.find(_ns_name("clef", ns))
        if clef_el is not None:  # if there is a clef element inside
            return _handle_clef(clef_el, position, part, ns)
        else:  # if all info are in the staffdef element
            number = element.attrib["n"]
            sign = element.attrib["clef.shape"]
            line = element.attrib["clef.line"]
            octave = _compute_clef_octave(element.get("dis"), element.get("dis.place"))
    elif element.tag == _ns_name("clef", ns):
        if element.get("sameas") is not None:  # this is a copy of another clef
            # it seems this is used in different layers for the same staff
            # we don't handle it to avoid clef duplications
            return position
        else:
            # find the staff number
            parent = element.getparent()
            if parent.tag == _ns_name("staffDef", ns):
                number = parent.attrib["n"]
            else:  # go back another level to staff element
                number = parent.getparent().attrib["n"]
            sign = element.attrib["shape"]
            line = element.attrib["line"]
            octave = _compute_clef_octave(element.get("dis"), element.get("dis.place"))
    else:
        raise Exception("_handle_clef only accepts staffDef or clef elements")
    new_clef = score.Clef(int(number), sign, int(line), octave)
    part.add(new_clef, position)
    return position


def _handle_staffdef(staffdef_el, position, part, ns):
    # fill with time signature info
    _handle_metersig(staffdef_el, position, part, ns)
    # fill with key signature info
    _handle_keysig(staffdef_el, position, part, ns)
    # fill with clef info
    _handle_clef(staffdef_el, position, part, ns)


def _handle_initial_staffdef(staffdef_el, ns):
    """Handles the definition of a single staff"""
    id = staffdef_el.attrib[_ns_name("id", XML_NAMESPACE)]
    label_el = staffdef_el.find(_ns_name("label", ns))
    name = label_el.text if label_el is not None else ""
    ppq = int(staffdef_el.attrib["ppq"])
    # generate the part
    part = score.Part(id, name, quarter_duration=ppq)
    # fill it with other info, e.g. meter, time signature, key signature
    _handle_staffdef(staffdef_el, 0, part, ns)
    return part


def _handle_staffgroup(staffgroup_el, ns):
    """Handles a staffGrp. WARNING: in MEI piano staves are a staffGrp"""
    group_symbol_el = staffgroup_el.find(_ns_name("grpSym", ns))
    if group_symbol_el is None:
        group_symbol = staffgroup_el.attrib["symbol"]
    else:
        group_symbol = group_symbol_el.attrib["symbol"]
    label_el = staffgroup_el.find(_ns_name("label", ns))
    name = label_el.text if label_el is not None else None
    id = staffgroup_el.attrib[_ns_name("id", XML_NAMESPACE)]
    staff_group = score.PartGroup(group_symbol, group_name=name, id=id)
    staves_el = staffgroup_el.findall(_ns_name("staffDef", ns))
    for s_el in staves_el:
        new_part = _handle_initial_staffdef(s_el, ns)
        staff_group.children.append(new_part)
    return staff_group


def _handle_main_staff_group(main_staffgrp_el, ns):
    """Handles the main staffGrp that contains all other staves or staff groups"""
    staves_el = main_staffgrp_el.findall(_ns_name("staffDef", ns))
    staff_groups_el = main_staffgrp_el.findall(_ns_name("staffGrp", ns))
    # the list of parts or part groups
    part_list = []
    # process the parts
    for s_el in staves_el:
        new_part = _handle_initial_staffdef(s_el, ns)
        part_list.append(new_part)
    # process the part groups
    for sg_el in staff_groups_el:
        new_staffgroup = _handle_staffgroup(sg_el, ns)
        part_list.append(new_staffgroup)
    return part_list


# functions to parse the content of parts


def _accidstring_to_int(accid_string: str) -> int:
    if accid_string is None:
        return None
    else:
        return SIGN_TO_ALTER[accid_string]


def _pitch_info(note_el):
    step = note_el.attrib["pname"]
    octave = int(note_el.attrib["oct"])
    alter = _accidstring_to_int(note_el.get("accid"))
    return step, octave, alter


def _duration_info(el, ns):
    """Extract duration info from a xml element.

    It works for example with note_el, chord_el

    Args:
        el (lxml tree): the xml element to analyze

    Returns:
        id, duration and symbolic duration of the element
    """
    # find duration in ppq. For grace notes is 0
    duration = int(el.get("dur.ppq")) if el.get("dur.ppq") is not None else 0
    # find symbolic duration
    symbolic_duration = {}
    symbolic_duration["type"] = el.attrib["dur"]
    if not el.get("dots") is None:
        symbolic_duration["dots"] = int(el.get("dots"))
    # find eventual time modifications
    parent = el.getparent()
    if parent.tag == _ns_name("tuplet", ns):
        symbolic_duration["actual_notes"] = parent.attrib["num"]
        symbolic_duration["normal_notes"] = parent.attrib["numbase"]
    # find id
    id = el.attrib[_ns_name("id", XML_NAMESPACE)]
    return id, duration, symbolic_duration


def _handle_note(note_el, position, voice, staff, part, ns):
    # find pitch info
    step, octave, alter = _pitch_info(note_el)
    # find duration info
    note_id, duration, symbolic_duration = _duration_info(note_el, ns)
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


def _handle_rest(rest_el, position, voice, staff, part, ns):
    # find duration info
    rest_id, duration, symbolic_duration = _duration_info(rest_el, ns)
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


def _handle_mrest(mrest_el, position, voice, staff, part, ns):
    """Handles a rest that spawn the entire measure"""
    # find id
    mrest_id = mrest_el.attrib[_ns_name("id", XML_NAMESPACE)]
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
        symbolic_duration=None,
        articulations=None,
    )
    # add mrest to the part
    part.add(rest, position, position + 1)
    # now iterate
    # return duration to update the position in the layer
    return position + parts_per_measure


def _handle_chord(chord_el, position, voice, staff, part, ns):
    # find duration info
    chord_id, duration, symbolic_duration = _duration_info(chord_el, ns)
    # find notes info
    notes_el = chord_el.findall(_ns_name("note", ns))
    for note_el in notes_el:
        note_id = note_el.attrib[_ns_name("id", XML_NAMESPACE)]
        # find pitch info
        step, octave, alter = _pitch_info(note_el)
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


def _handle_layer_in_staff_in_measure(
    layer_el, ind_layer: int, ind_staff: int, position: int, part, ns
) -> int:
    for i, e in enumerate(layer_el):
        if e.tag == _ns_name("note", ns):
            new_position = _handle_note(e, position, ind_layer, ind_staff, part, ns)
        elif e.tag == _ns_name("chord", ns):
            new_position = _handle_chord(e, position, ind_layer, ind_staff, part, ns)
        elif e.tag == _ns_name("rest", ns):
            new_position = _handle_rest(e, position, ind_layer, ind_staff, part, ns)
        elif e.tag == _ns_name("mRest", ns):  # rest that spawn the entire measure
            new_position = _handle_mrest(e, position, ind_layer, ind_staff, part, ns)
        elif e.tag == _ns_name("beam", ns):
            # TODO : add Beam element
            # recursive call to the elements inside beam
            new_position = _handle_layer_in_staff_in_measure(
                e, ind_layer, ind_staff, position, part, ns
            )
        elif e.tag == _ns_name("tuplet", ns):
            # TODO : add Tuplet element
            # recursive call to the elements inside Tuplet
            new_position = _handle_layer_in_staff_in_measure(
                e, ind_layer, ind_staff, position, part, ns
            )
        elif e.tag == _ns_name("clef", ns):
            new_position = _handle_clef(e, position, part, ns)
        else:
            raise Exception("Tag " + e.tag + " not supported")

        # update the current position
        position = new_position
    return position


def _handle_staff_in_measure(staff_el, staff_ind, position: int, part, ns):
    # add measure
    measure = score.Measure(number=staff_el.getparent().get("n"))
    part.add(measure, position)

    layers_el = staff_el.findall(_ns_name("layer", ns))
    end_positions = []
    for i_layer, layer_el in enumerate(layers_el):
        end_positions.append(
            _handle_layer_in_staff_in_measure(
                layer_el, i_layer, staff_ind, position, part, ns
            )
        )
    # sanity check that all layers have equal duration
    if not all([e == end_positions[0] for e in end_positions]):
        raise Exception("Different voices have different durations")
    return end_positions[0]


def _handle_section(parts, section_el, ns):
    position = 0
    for i_el, element in enumerate(section_el):
        # handle measures
        if element.tag == _ns_name("measure", ns):
            staves_el = element.findall(_ns_name("staff", ns))
            if len(list(staves_el)) != len(list(parts)):
                raise Exception("Not all parts are specified in measure" + i_el)
            end_positions = []
            for i_s, (part, staff_el) in enumerate(zip(parts, staves_el)):
                end_positions.append(
                    _handle_staff_in_measure(staff_el, i_s, position, part, ns)
                )
            # sanity check that all layers have equal duration
            if not all([e == end_positions[0] for e in end_positions]):
                raise Exception("Different parts have measures of different duration")
            position = end_positions[0]
        # handle staffDef elements
        elif element.tag == _ns_name("scoreDef", ns):
            # meter modifications
            metersig_el = element.find(_ns_name("meterSig", ns))
            if (metersig_el is not None) or (element.get("meter.count") is not None):
                for part in parts:
                    _handle_metersig(element, position, part, ns)
            # key signature modifications
            keysig_el = element.find(_ns_name("keySig", ns))
            if (keysig_el is not None) or (element.get("key.sig") is not None):
                for part in parts:
                    _handle_keysig(element, position, part, ns)

    return position


def _tie_notes(section_el, part_list, ns):
    """Ties all notes in a part.
    This function must be run after the parts are completely created."""
    ties_el = section_el.findall(_ns_name("tie", ns, True))
    # create a dict of id : note, to speed up search
    all_notes = [
        note
        for part in score.iter_parts(part_list)
        for note in part.iter_all(cls=score.Note)
    ]
    all_notes_dict = {note.id: note for note in all_notes}
    for tie_el in ties_el:
        start_id = tie_el.attrib["startid"][1:]  # remove the # in first position
        end_id = tie_el.attrib["endid"][1:]  # remove the # in first position
        # set tie prev and tie next in partira note objects
        all_notes_dict[start_id].tie_next = all_notes_dict[end_id]
        all_notes_dict[end_id].tie_prev = all_notes_dict[start_id]


def load_mei(xml_path: str):
    # parse xml file
    document, ns = _parse_mei(xml_path)

    # handle staff and staff groups info
    main_partgroup_el = document.find(_ns_name("staffGrp", ns, True))
    part_list = _handle_main_staff_group(main_partgroup_el, ns)

    # fill the content of the score
    sections_el = document.findall(_ns_name("section", ns, True))
    if len(sections_el) != 1:
        raise Exception("Only MEI with a single section are supported")
    _handle_section(list(score.iter_parts(part_list)), sections_el[0], ns)
    # handle ties
    _tie_notes(sections_el[0], part_list, ns)

    return part_list

