from lxml import etree
from xmlschema.names import XML_NAMESPACE
import partitura.score as score

XML_NS = "http://www.w3.org/XML/1998/namespace"


def _ns_name(name, ns, all=False):
    if not all:
        return "{" + ns + "}" + name
    else:
        return ".//{" + ns + "}" + name


def _handle_staffdef(staffdef_el, ns):
    id = staffdef_el.attrib[_ns_name("id", XML_NAMESPACE)]
    label_el = staffdef_el.find(_ns_name("label", ns))
    name = label_el.text if label_el is not None else ""
    ppq = int(staffdef_el.attrib["ppq"])
    return score.Part(id, name, quarter_duration=ppq)


def _handle_staffgroup(staffgroup_el, ns):
    group_symbol_el = staffgroup_el.find(_ns_name("grpSym", ns))
    group_symbol = group_symbol_el.attrib["symbol"]
    label_el = staffgroup_el.find(_ns_name("label", ns))
    name = label_el.text if label_el is not None else None
    id = staffgroup_el.attrib[_ns_name("id", XML_NAMESPACE)]
    staff_group = score.PartGroup(group_symbol, group_name=name, id=id)
    staves_el = staffgroup_el.findall(_ns_name("staffDef", ns))
    for s_el in staves_el:
        new_part = _handle_staffdef(s_el, ns)
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
        new_part = _handle_staffdef(s_el, ns)
        part_list.append(new_part)
    # process the part groups
    for sg_el in staff_groups_el:
        new_staffgroup = _handle_staffgroup(sg_el, ns)
        part_list.append(new_staffgroup)
    return part_list


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


def load_mei(xml):

    # define parts
    parts = []
    parts_el = document.findall(_ns_name("staffDef", ns, True))
    for part_el in parts_el:
        id = part_el.attrib[_ns_name("id", XML_NAMESPACE)]
        label_el = part_el.find(_ns_name("label", ns))
        name = label_el.text if label_el is not None else ""
        ppq = part_el.attrib["ppq"]
        new_part = score.Part(id, name, quarter_duration=ppq)
        parts.append(new_part)

    # find the shortest element in the score
    el_with_durs = document.xpath(".//*[@dur]")
    durs = [e.attrib["dur"]]
    # find the dots
    dots = []

    sections = document.findall(_ns_name("section", ns, True))
    if len(sections) != 1:
        raise Exception("Only MEI with a single section are supported")
    _handle_section(sections[0])
    # TODO : handle section

    return


def _handle_section(section_el, ns):
    position = 0

    for measure_el in section_el.findall(_ns_name("measure", ns)):
        for layer_el in measure_el.findall(_ns_name("layer", ns)):
            position, doc_order = _handle_measure(measure_el, position)

    return


def _handle_measure(measure_el, position, part):
    """
    Parse a <measure>...</measure> element, adding it and its contents to the
    part.
    """
    # make a measure object
    measure = score.Measure()
    measure.number = None

    # add the start of the measure to the time line
    part.add(measure, position)

    # TODO: maybe look first for staff and layers?

    # keep track of the position within the measure
    measure_start = position
    # keep track of the previous note (in case of <chord>)
    prev_note = None
    # used to keep track of the duration of the measure
    measure_maxtime = measure_start
    trailing_children = []
    for i, e in enumerate(measure_el):
        if e.tag == "note":
            pass
        elif e.tag == "rest":
            pass

