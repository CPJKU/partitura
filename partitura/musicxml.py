#!/usr/bin/env python

# -*- coding: utf-8 -*-

from collections import defaultdict
import logging

import numpy as np
from lxml import etree
# lxml does XSD validation too but has problems with the MusicXML 3.1 XSD, so we use
# the xmlschema package for validating MusicXML against the definition
import xmlschema
import pkg_resources
from partitura.directions import parse_words
import partitura.score as score

_LOGGER = logging.getLogger(__name__)
_MUSICXML_SCHEMA = pkg_resources.resource_filename('partitura', 'musicxml.xsd')
_XML_VALIDATOR = None
_DYNAMICS_DIRECTIONS = {
    'f': score.ConstantLoudnessDirection,
    'ff': score.ConstantLoudnessDirection,
    'fff': score.ConstantLoudnessDirection,
    'ffff': score.ConstantLoudnessDirection,
    'fffff': score.ConstantLoudnessDirection,
    'ffffff': score.ConstantLoudnessDirection,
    'n': score.ConstantLoudnessDirection,
    'mf': score.ConstantLoudnessDirection,
    'mp': score.ConstantLoudnessDirection,
    'pp': score.ConstantLoudnessDirection,
    'ppp': score.ConstantLoudnessDirection,
    'pppp': score.ConstantLoudnessDirection,
    'ppppp': score.ConstantLoudnessDirection,
    'pppppp': score.ConstantLoudnessDirection,
    'fp': score.ImpulsiveLoudnessDirection,
    'pf': score.ImpulsiveLoudnessDirection,
    'rf': score.ImpulsiveLoudnessDirection,
    'rfz': score.ImpulsiveLoudnessDirection,
    'fz': score.ImpulsiveLoudnessDirection,
    'sf': score.ImpulsiveLoudnessDirection,
    'sffz': score.ImpulsiveLoudnessDirection,
    'sfp': score.ImpulsiveLoudnessDirection,
    'sfzp': score.ImpulsiveLoudnessDirection,
    'sfpp': score.ImpulsiveLoudnessDirection,
    'sfz': score.ImpulsiveLoudnessDirection,
}

def validate_musicxml(xml, debug=False):
    """
    Validate an XML file against an XSD. 
    
    Parameters
    ----------
    xml: str
        Path to XML file
    debug: bool, optional
        If True, raise an exception when the xml is invalid, and print out the
        cause. Otherwise just return True when the XML is valid and False otherwise
    
    Returns
    -------
    bool or None
        None if debug=True, True or False otherwise, signalling validity

    """
    global _XML_VALIDATOR
    if not _XML_VALIDATOR:
        _XML_VALIDATOR = xmlschema.XMLSchema(_MUSICXML_SCHEMA)
    if debug:
        return _XML_VALIDATOR.validate(xml)
    else:
        return _XML_VALIDATOR.is_valid(xml)


def parse_partlist(partlist):
    """
    This parses the <part-list> ... </part-list> element in the beginning
    of the MusicXML file where each instrument is declared, instruments and
    their staves are grouped (braces, brackets), etc.

    Parameters
    ----------
    partlist : etree element
        The part-list etree element

    Returns
    -------
    list:
        list of PartGroup objects
    dict:
        Dictionary of pairs (partid, Part) where the Part objects are
        instantiated with part-name and part-abbreviation if these are specified in
        the part list definition.
    """

    structure = []
    current_group = None
    part_dict = {}

    for e in partlist:
        if e.tag == 'part-group':
            if e.attrib['type'] == 'start':

                gr_name = get_value_from_tag(e, 'group-name', str)
                gr_type = get_value_from_tag(e, 'group-symbol', str)

                new_group = score.PartGroup(gr_type, gr_name)

                if 'number' in e.attrib:
                    new_group.number = int(e.attrib['number'])

                if current_group is None:
                    current_group = new_group
                else:
                    current_group.constituents.append(new_group)
                    new_group.parent = current_group
                    current_group = new_group

            elif e.attrib['type'] == 'stop':
                if current_group.parent is None:
                    structure.append(current_group)
                    current_group = None
                else:
                    current_group = current_group.parent

        elif e.tag == 'score-part':
            part_id = e.attrib['id']
            part = score.Part(part_id)

            # set part name and abbreviation if available
            part.part_name = next(iter(e.xpath('part-name/text()')), None)
            part.part_abbreviation = next(iter(e.xpath('part-abbreviation/text()')), None)

            part_dict[part_id] = part

            if current_group is None:
                structure.append(part)
            else:
                current_group.constituents.append(part)
                sp.parent = current_group

    if current_group:
        LOGGER.warning(
            'part-group {0} was not ended'.format(current_group.number))
        structure.append(current_group)

    return structure, part_dict
    

def load_musicxml(xml, validate=False):
    """
    Parse a MusicXML file and build a composite score ontology
    structure from it (see also scoreontology.py).

    Parameters
    ----------
    xml : str or file-like  object
        Path to the MusicXML file to be parsed, or a file-like object
    validate : bool (optional, default=False)
        When True, the validity of the MusicXML is checked against the MusicXML
        3.1 specification before loading the file. An exception will be raised 
        when the MusicXML is invalid.

    Returns
    -------
    partlist : list
        A list of either Part or PartGroup objects
    """
    if validate:
        validate_musicxml(xml, debug=True)
        # if xml is a file-like object we need to set the read pointer to the
        # start of the file for parsing
        if hasattr(xml, 'seek'):
            xml.seek(0)
    
    parser = etree.XMLParser(resolve_entities=False, huge_tree=False,
                             remove_comments=True, remove_blank_text=True)
    document = etree.parse(xml, parser)
    
    if document.getroot().tag != 'score-partwise':
        raise Exception('Currently only score-partwise structure is supported')

    partlist_el = document.xpath('part-list')
    
    if partlist_el:
        # parse the (hierarchical) structure of score parts
        # (instruments) that are listed in the part-list element
        partlist, part_dict = parse_partlist(partlist_el[0])
        # Go through each <part> to obtain the content of the parts.
        # The Part instances will be modified in place
        parse_parts(document, part_dict)
    else:
        partlist = []
    

    return partlist


def parse_parts(document, part_dict):
    """
    Populate the Part instances that are the values of `part_dict` with the musical content in document.

    Parameters
    ----------
    document : lxml.etree.ElementTree
        The ElementTree representation of the MusicXML document
    part_dict : dict
        A dictionary with key--value pairs (part_id, Part instance), as returned
        by the parse_partlist() function.
    """

    for part_id in document.xpath('/score-partwise/part/@id'):
        part_el = document.xpath('/score-partwise/part[@id="{}"]'.format(part_id))[0]
        part = part_dict.get(part_id, score.Part(part_id))
        populate_part(part_el, part)
        
        # part_builder.finalize()


def populate_part(part_el, part):
    position = 0
    ongoing = {}

    # add new page and system at start of part
    handle_new_page(position, part.timeline, ongoing)
    handle_new_system(position, part.timeline, ongoing)
    
    for measure_el in part_el.xpath('measure'):
        position = handle_measure(measure_el, position, part.timeline, ongoing)


def handle_measure(measure_el, position, timeline, ongoing):
    """
    Parse a <measure>...</measure> element, adding it and its contents to the
    timeline.
    """
    # make a measure object
    measure = make_measure(measure_el)

    # add the start of the measure to the time line
    timeline.add_starting_object(position, measure)

    # keep track of the position within the measure
    # measure_pos = 0
    measure_start = position
    # keep track of the previous note (in case of <chord>)
    prev_note = None
    # used to keep track of the duration of the measure
    measure_maxtime = measure_start

    for i, e in enumerate(measure_el):

        if e.tag == 'backup':
    	# <xs:documentation>The backup and forward elements are required
    	# to coordinate multiple voices in one part, including music on
    	# multiple staves. The backup type is generally used to move
    	# between voices and staves. Thus the backup element does not
    	# include voice or staff elements. Duration values should always
    	# be positive, and should not cross measure boundaries or
    	# mid-measure changes in the divisions value.</xs:documentation>

            duration = get_value_from_tag(e, 'duration', int) or 0
            position -= duration
            # <backup> tags trigger an update of the measure
            # duration up to the measure position (after the
            # <backup> has been processed); This has been found to
            # account for implicit measure durations in
            # Baerenreiter MusicXML files.
            measure_maxtime = max(measure_maxtime, position)

        elif e.tag == 'forward':
            duration = get_value_from_tag(e, 'duration', int) or 0
            position += duration
            measure_maxtime = max(measure_maxtime, position)

        elif e.tag == 'attributes':
            handle_attributes(e, position, timeline)

        elif e.tag == 'direction':
            handle_direction(e, position, timeline, ongoing)

        elif e.tag == 'print':
            # new-page/new-system occurring anywhere in the measure take effect
            # at the start of the measure, so we pass measure_start rather than
            # position
            handle_print(e, measure_start, timeline, ongoing)

        elif e.tag == 'sound':
            handle_sound(e, position, timeline)

        elif e.tag == 'note':
            (position, prev_note) = handle_note(e, position, timeline, ongoing, prev_note)
            measure_maxtime = max(measure_maxtime, position)
            # measure_duration = max(measure_duration, measure_pos)

        elif e.tag == 'barline':
            repeats = e.xpath('repeat')
            if len(repeats) > 0:
                handle_repeat(repeats[0], measure_pos)

            endings = e.xpath('ending')
            if len(endings) > 0:
                handle_ending(endings[0], measure_pos)

        else:
            LOGGER.debug('ignoring tag {0}'.format(e.tag))

    timeline.add_ending_object(measure_maxtime, measure)

    return measure_maxtime


def handle_new_page(position, timeline, ongoing):
    if 'page' in ongoing:
        if position == 0:
            # ignore non-informative new-page at start of score
            return

        timeline.add_ending_object(position,
                                   ongoing['page'])
        page_nr = ongoing['page'].nr + 1
    else:
        page_nr = 1

    page = score.Page(page_nr)
    timeline.add_starting_object(position, page)
    ongoing['page'] = page


def handle_new_system(position, timeline, ongoing):
    if 'system' in ongoing:

        if position == 0:
            # ignore non-informative new-system at start of score
            return

        # end current page
        timeline.add_ending_object(position, ongoing['system'])
        system_nr = ongoing['system'].nr + 1
    else:
        system_nr = 1

    system = score.System(system_nr)
    timeline.add_starting_object(position, system)
    ongoing['system'] = system


def make_measure(xml_measure):
    measure = score.Measure()
    try:
        measure.number = int(xml_measure.attrib['number'])
    except:
        LOGGER.warn('No number attribute found for measure')
    return measure


def handle_attributes(e, position, timeline):
    """

    """

    ts_num = get_value_from_tag(e, 'time/beats', int)
    ts_den = get_value_from_tag(e, 'time/beat-type', int)
    if ts_num and ts_den:
        timeline.add_starting_object(position, score.TimeSignature(ts_num, ts_den))

    fifths = get_value_from_tag(e, 'key/fifths', int)
    mode = get_value_from_tag(e, 'key/mode', str)
    if fifths is not None or mode is not None:
        timeline.add_starting_object(position, score.KeySignature(fifths, mode))

    diat = get_value_from_tag(e, 'transpose/diatonic', int)
    chrom = get_value_from_tag(e, 'transpose/chromatic', int)
    if diat is not None or chrom is not None:
        timeline.add_starting_object(position, score.Transposition(diat, chrom))
        
    divs = get_value_from_tag(e, 'divisions', int)
    if divs:
        timeline.add_starting_object(position, score.Divisions(divs))

    clefs = get_clefs(e)
    for clef in clefs:
        timeline.add_starting_object(position, score.Clef(**clef))


def get_offset(e):
    offset = e.find('offset')

    if offset:

        sounding = offset.attrib.get('sound', 'no')
        return int(offset.text) if sounding == 'yes' else 0

    else:

        return None


def handle_direction(e, position, timeline, ongoing):

    # <!--
    #     A direction is a musical indication that is not attached
    #     to a specific note. Two or more may be combined to
    #     indicate starts and stops of wedges, dashes, etc.

    #     By default, a series of direction-type elements and a
    #     series of child elements of a direction-type within a
    #     single direction element follow one another in sequence
    #     visually. For a series of direction-type children, non-
    #     positional formatting attributes are carried over from
    #     the previous element by default.
    # -->
    # <!ELEMENT direction (direction-type+, offset?,
    #     %editorial-voice;, staff?, sound?)>
    # <!ATTLIST direction
    #     %placement;
    #     %direction;
    # >

    # directions may have an explicit temporal offset (does not
    # affect measure_duration)

    offset = get_value_from_tag(e, 'offset', int) or 0

    starting_directions = []
    ending_directions = []

    sounds = e.xpath('sound')
    if len(sounds) > 0:
        if 'fine' in sounds[0].attrib:
            starting_directions.append(Fine())
        if 'dacapo' in sounds[0].attrib:
            starting_directions.append(DaCapo())

    # <direction-type> ... </...>
    direction_types = e.xpath('direction-type')
    # <!ELEMENT direction-type (rehearsal+ | segno+ | words+ |
    #     coda+ | wedge | dynamics+ | dashes | bracket | pedal |
    #     metronome | octave-shift | harp-pedals | damp | damp-all |
    #     eyeglasses | string-mute | scordatura | image |
    #     principal-voice | accordion-registration | percussion+ |
    #     other-direction)>

    # flag indicating presence of <dashes type=start>
    dashes_start = None
    dashes_end = None

    for dts in direction_types:
        assert len(dts) >= 1

        if len(dts) > 1:
            LOGGER.warning(
                '(FIXME) ignoring trailing direction-types in direction')
        direction_type = dts[0].tag

        if direction_type == 'dynamics':
            # there may be multiple dynamics items in dts, loop:
            for dt in dts:
                # there may be multiple dynamics components
                for dyn in dt:
                    # direction = None # _make_direction(dyn.tag)
                    direction = _DYNAMICS_DIRECTIONS.get(dyn.tag, score.Words)(dyn.tag)
                    if direction is not None:
                        starting_directions.append(direction)

        elif direction_type == 'words':
            # there may be multiple dynamics/words items in dts, loop:
            for dt in dts:
                # try to make a direction out of words
                # TODO: check if we need str in python3
                parse_result = parse_words(str(dt.text))

                if parse_result is not None:

                    if isinstance(parse_result, (list, tuple)):
                        starting_directions.extend(parse_result)
                    else:
                        starting_directions.append(parse_result)

        elif direction_type == 'wedge':
            key = get_wedge_key(dts[0])

            if dts[0].attrib['type'] in ('crescendo', 'diminuendo'):

                o = score.DynamicLoudnessDirection(dts[0].attrib['type'])
                starting_directions.append(o)
                ongoing[key] = o

            elif dts[0].attrib['type'] == 'stop':

                o = ongoing.get(key, None)

                if o is not None:
                    ending_directions.append(o)
                    del ongoing[key]
                else:
                    LOGGER.warning(
                        'did not find a wedge start element for wedge stop!')

        elif direction_type == 'dashes':

            if dts[0].attrib['type'] == 'start':
                dashes_start = dts[0]
            else:
                dashes_end = dts[0]
        else:
            LOGGER.warning('ignoring direction type: {} {}'.format(
                direction_type, dts[0].attrib))

    if dashes_start is not None:

        key = get_dashes_key(dashes_start)
        ongoing[key] = starting_directions

    if dashes_end is not None:

        key = get_dashes_key(dashes_end)
        oo = ongoing.get(key, None)

        if oo is None:
            LOGGER.warning('Dashes end without dashes start')
        else:
            ending_directions.extend(oo)
            del ongoing[key]

    for o in starting_directions:
        timeline.add_starting_object(position, o)

    for o in ending_directions:
        timeline.add_ending_object(position, o)


def get_clefs(e):
    """
    Get the clefs

    Returns
    -------

    (fifths, mode) OR None : tuple of or None.

    """
    clefs = e.xpath('clef')
    result = []
    for clef in clefs:
        result.append(dict(number=get_value_from_attribute(clef, 'number', int),
                           sign=get_value_from_tag(clef, 'sign', str),
                           line=get_value_from_tag(clef, 'line', int),
                           octave_change=get_value_from_tag(clef, 'clef-octave-change', int)))
    return result


def get_value_from_tag(e, tag, as_type, none_on_error=True):
    """
    Return the text contents of a particular tag in element e, cast as a
    particular type. By default the function will return None if either the tag
    is not found or the value cannot be cast to the desired type.
    
    Examples
    --------

    >>> e = lxml.etree.fromstring('<note><duration>2</duration></note>') 
    >>> get_value_from_tag(e, 'duration', int)
    2
    >>> get_value_from_tag(e, 'duration', float)
    2.0

    >>> e = lxml.etree.fromstring('<note><duration>quarter</duration></note>')
    >>> get_value_from_tag(e, 'duration', float)
    None
    

    Parameters
    ----------
    e: etree.Element
        An etree Element instance
    tag: string
        Child tag to retrieve
    as_type: function
        Function that casts the string to the desired type (e.g. int, float)
    none_on_error: bool, optional (default: True)
        When False, an exception is raised when `tag` is not found in `e` or the
        text inside `tag` cannot be cast to an integer. When True, None is returned
        in such cases.
    
    Returns
    -------
    object
        The value read from `tag`, cast as `as_type`
    """
    try:
        return as_type(e.find(tag).text)
    except:
        if none_on_error:
            return None
        else:
            raise


def get_value_from_attribute(e, attr, as_type, none_on_error=True):
    """
    Return the attribute of a particular attribute of element e, cast as a
    particular type. By default the function will return None if either `e` does
    not have the attribute or the value cannot be cast to the desired type.

    Parameters
    ----------
    e: etree.Element
        An etree Element instance
    attr: string
        Attribute to retrieve
    as_type: function
        Function that casts the string to the desired type (e.g. int, float)
    none_on_error: bool, optional (default: True)
        When False, an exception is raised when `tag` is not found in `e` or the
        text inside `tag` cannot be cast to an integer. When True, None is returned
        in such cases.
    
    Returns
    -------
    object or None
        The attribute value, or None
    """
    try:
        return as_type(e.get(attr))
    except:
        if none_on_error:
            return None
        else:
            raise


def get_pitch(e):
    """
    Check whether the element has a pitch. If so return a tuple (pitch, alter,
    octave), otherwise return None.

    Returns
    -------
    tuple : (str, int or None, int) or None
        The tuple contains (pitch, alter, octave)
    """

    pitch = e.find('pitch')
    if pitch:
        step = get_value_from_tag(pitch, 'step', str)
        alter = get_value_from_tag(pitch, 'alter', int)
        octave = get_value_from_tag(pitch, 'octave', int)
        return (step, alter, octave)
    else:
        return None


def handle_print(e, position, timeline, ongoing):
    if "new-page" in e.attrib:
        handle_new_page(position, timeline, ongoing)
        handle_new_system(position, timeline, ongoing)
    if "new-system" in e.attrib:
        handle_new_system(position, timeline, ongoing)

def handle_sound(e, position, timeline):
    if "tempo" in e.attrib:
        timeline.add_starting_object(position, Tempo(int(e.attrib['tempo'])))

def handle_note(e, position, timeline, ongoing, prev_note):

    # prev_note is used when the current note has a <chord/> tag

    # get some common features of element if available
    duration = get_value_from_tag(e, 'duration', int) or 0
    # elements may have an explicit temporal offset
    offset = get_value_from_tag(e, 'offset', int) or 0
    staff = get_value_from_tag(e, 'staff', int) or 0
    voice = get_value_from_tag(e, 'voice', int) or 0

    note_id = get_value_from_attribute(e, 'id', str)

    symbolic_duration = {}
    symbolic_duration['type'] = get_value_from_tag(e, 'type', str)
    symbolic_duration['dots'] = len(e.findall('dot'))
    symbolic_duration['actual_notes'] = get_value_from_tag(e, 'time-modification/actual-notes', int)
    symbolic_duration['normal_notes'] = get_value_from_tag(e, 'time-modification/normal-notes', int)

    chord = e.find('chord') is not None
    if chord:
        # this note starts at the same position as the previous note, and has same duration
        assert prev_note is not None
        position = prev_note.start.t
        # duration = prev_note.duration
        
        # TODO: fix logic for Chord instances

        # if 'chord' in ongoing:
        #     chord = ongoing['chord']
        # else:
        #     chord = score.Chord()
        #     timeline.add_starting_object(prev_note.start.t, chord)
        #     ongoing['chord'] = chord
    else:
        # end the current chord

        # if 'chord' in ongoing:
        #     chord = ongoing['chord']
        #     self.timeline.add_ending_object(
        #         self.position + measure_pos, chord)
        #     del self.ongoing['chord']
        pass

    pitch = e.find('pitch')
    if pitch is not None:

        step = get_value_from_tag(pitch, 'step', str)
        alter = get_value_from_tag(pitch, 'alter', int)
        octave = get_value_from_tag(pitch, 'octave', int)

        grace = e.find('grace')

        if grace is not None:
            grace_type, steal_proportion = get_grace_info(grace)
            note = score.GraceNote(grace_type, step, alter, octave,
                                   note_id, voice, staff, symbolic_duration,
                                   steal_proportion=steal_proportion)
        else:

            note = score.Note(step, alter, octave, note_id, voice, staff, symbolic_duration)

    else:
        # note element is a rest
        note = score.Rest(note_id, voice, staff, symbolic_duration)

    timeline.add_starting_object(position, note)
    timeline.add_ending_object(position+duration, note)

    ties = e.findall('tie')
    if ties:

        # TODO: this fails when note is a rest
        tie_key = ('tie', note.midi_pitch)
        tie_types = set(tie.attrib['type'] for tie in ties)

        if 'stop' in tie_types:

            tie_prev = ongoing.get(tie_key, None)

            if tie_prev:

                note.tie_prev = tie_prev
                tie_prev.tie_next = note
                del ongoing[tie_key]

        if 'start' in tie_types:

            ongoing[tie_key] = note


    # staccato = e.find('notations/articulations/staccato') is not None
    # accent = e.find('notations/articulations/accent') is not None
    # if fermata:
    #     self.timeline.add_starting_object(
    #         position, score.Fermata())
        
    # # look for <notations> tags. Inside them, <tied> and <slur>
    # # may be present. Note that for a tie, a <tied> should be present
    # # here as well as a <tie> tag inside the <note> ... </note> tags
    # # of the same note (this is not looked for here). The code
    # # so far only looks for <slur> here.
    # if len(e.xpath('notations')) > 0:

    #     eslurs = e.xpath('notations/slur')    # list

    #     # TODO: Slur stop can preceed slur start in document order (in the
    #     # case of <backup>). Current implementation does not recognize that.

    #     # this sorts all found slurs by type (either 'start' or 'stop')
    #     # in reverse order, so all with type 'stop' will be before
    #     # the ones with 'start'?!.
    #     eslurs.sort(key=lambda x: x.attrib['type'], reverse=True)

    #     # Now that the slurs are sorted by their type, sort them
    #     # by their numbers; First note that slurs do not always
    #     # have a number attribute, then 1 is implied.
    #     # If, however, either more than one slur starts
    #     # or ends at the same note (!) they must be
    #     # numbered so that they can be distinguished. If however
    #     # a (single) slur ends and the next (single) one starts
    #     # at the same note, none of them needs to be numbered.
    #     eslurs.sort(key=lambda x: int(
    #         x.attrib['number']) if 'number' in x.attrib else 1)

    #     erroneous_stops = []    # gather stray stops that have no beginning?

    #     for eslur in eslurs:    # loop over all found slurs
    #         key = get_slur_key(eslur)    # number that represents the slur
    #         if eslur.attrib['type'] == 'stop':
    #             try:
    #                 o = self.ongoing[key]
    #                 self.timeline.add_ending_object(
    #                     self.position + measure_pos, o)
    #                 del self.ongoing[key]
    #             except KeyError:  # as exception:
    #                 LOGGER.warning(("Part xx, Measure xx: Stopping slur with number {0} was never started (Note ID: {1})"
    #                                 "").format(key[1], note_id))

    #                 erroneous_stops.append(key)

    #         elif eslur.attrib['type'] == 'start':
    #             if key not in self.ongoing:
    #                 o = score.Slur(voice)
    #                 self.timeline.add_starting_object(
    #                     self.position + measure_pos, o)
    #                 self.ongoing[key] = o
    #             else:
    #                 LOGGER.warning(
    #                     "Slur with number {0} started twice; Ignoring the second slur start".format(key[1]))
    #     for k in erroneous_stops:
    #         if k in self.ongoing:
    #             del self.ongoing[k]

    new_position = position + duration

    return new_position, note 


def get_grace_info(grace):
    # grace note handling
    grace_type = 'grace'
    steal_proportion = None

    slash_text = get_value_from_attribute(grace, 'slash', str)
    if slash_text == 'yes':
        grace_type = 'acciaccatura'

    steal_prc = get_value_from_attribute(grace, 'steal-time-following', float)
    if steal_prc is not None:
        steal_proportion = steal_prc / 100
        grace_type = 'appoggiatura'

    steal_prec = get_value_from_attribute(grace, 'steal-time-previous', float)
    if steal_prc is not None:
        steal_proportion = steal_prc / 100
        grace_type = 'acciaccatura'

    return grace_type, steal_proportion
    
