from lxml import etree
from xmlschema.names import XML_NAMESPACE
import partitura.score as score

XML_NS = "http://www.w3.org/XML/1998/namespace"


def ns_name(name, ns, all=False):
    if not all:
        return "{" + ns + "}" + name
    else:
        return ".//{" + ns + "}" + name


def load_mei(xml):
    parser = etree.XMLParser(
        resolve_entities=False,
        huge_tree=False,
        remove_comments=True,
        remove_blank_text=True,
    )
    document = etree.parse(xml, parser)
    # find the namespace
    ns = document.getroot().nsmap[None]

    # define parts
    parts = []
    parts_el = document.findall(ns_name("staffDef", ns, True))
    for part_el in parts_el:
        id = part_el.attrib[ns_name("id", XML_NAMESPACE)]
        label_el = part_el.find(ns_name("label", ns))
        name = label_el.text() if label_el is not None else ""
        ppq = part_el.attrib[ns_name("ppq", XML_NAMESPACE)]
        new_part = score.Part(id, name, quarter_duration=ppq)
        parts.append(new_part)

    # find the shortest element in the score
    el_with_durs = document.xpath(".//*[@dur]")
    durs = [e.attrib["dur"]]
    # find the dots
    dots = []

    sections = document.findall(ns_name("section", ns, True))
    if len(sections) != 1:
        raise Exception("Only MEI with a single section are supported")
    _handle_section(sections[0])
    # TODO : handle section

    return


def _handle_section(section_el, ns):
    position = 0

    for measure_el in section_el.findall(ns_name("measure", ns)):
        for layer_el in measure_el.findall(ns_name("layer", ns)):
            position, doc_order = _handle_measure(measure_el, position)

    return


def _handle_(measure_el, position, part):
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

