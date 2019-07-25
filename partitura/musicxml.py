#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
import logging

import numpy as np
from lxml import etree
# lxml does XSD validation too but has problems with the MusicXML 3.1 XSD, so we use
# the xmlschema package for validating MusicXML against the definition
import xmlschema

from partitura.directions import parse_words
import partitura.score as score

LOGGER = logging.getLogger(__name__)
DYNAMICS_DIRECTIONS = {
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

XML_VALIDATOR = None

def validate(xsd, xml, debug=False):
    """
    Validate an XML file against an XSD. 
    
    Parameters
    ----------
    xsd: str
        Path to XSD file
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
    global XML_VALIDATOR
    if not XML_VALIDATOR:
        XML_VALIDATOR = xmlschema.XMLSchema(xsd)
    if debug:
        return XML_VALIDATOR.validate(xml)
    else:
        return XML_VALIDATOR.is_valid(xml)
    

# the get_*_key functions are helper functions for musical elements
# spanning a time interval. The start of those elements gets added to
# the timeline, and then they need to be stored until their end is
# encountered (e.g. wedges, ties, slurs). The keys that these
# functions produce must be such that the starting and the stopping
# xml elements produce the same key

def get_wedge_key(e):
    """
    Generate a key to store ongoing wedges temporarily.
    A <wedge ... /> is a so called "hairpin" (ger: Gabel)
    that stands for either crescendo or decrescendo/diminuendo.

    Returns
    -------
    ('wedge', number) OR ('wedge',) : tuple of (str, number) OR (str,)
    """
    if 'number' in e.attrib:
        return ('wedge', int(e.attrib['number']))
    else:
        return ('wedge',)


def get_dashes_key(e):
    """
    Generate a key to store ongoing dashes temporarily. A <dashes ... /> tag
    specifies the temporal range of some textual annotation at the start of the
    dashes.

    Returns
    -------
    ('dashes', number) OR ('wedge',) : tuple of (str, number) OR (str,)
    """
    if 'number' in e.attrib:
        return ('dashes', int(e.attrib['number']))
    else:
        return ('dashes', 1)


def get_repeat_key(e):
    """
    Generate a key to store repeats temporarily
    """
    return ('repeat',)


def get_ending_key(e):
    """
    Generate a key to store repeats temporarily
    """
    return ('ending',)


def get_slur_key(e):
    """
    Generate a key to store slurred notes temporarily

    Returns
    -------
    number
        the slur number if it has one, 1 otherwise
    """
    if 'number' in e.attrib:
        return ('slur', int(e.attrib['number']))
    else:
        return ('slur', 1)


def get_tie_key(e):
    """
    Generate a key to store tied notes temporarily.
    If the element has a pitch then a tuple is returned and
    None otherwise.

    Returns
    -------
    nested tuples (str, (str, number, number, number)) OR None
        Note that `alter` may be None instead.
        The nested tuples are ('tie', (pitch, alter, octave, staff))
        and serve as a key to identify a <tie>.

    Note
    ----

    This key mechanism does not handle ties between enharmonically equivalent
    pitches (such as Eb and D#).
    """

    pitchk = get_pitch(e)    # tuple (step, alter, octave)
    staff = None
    voice = None

    if e.find('staff') is not None:
        staff = int(e.find('staff').text)
    if e.find('voice') is not None:
        voice = int(e.find('voice').text)

    if pitchk is not None:    # element is a note
        return ('tie', tuple(list(pitchk) + [staff]))
    else:
        if e.find('rest') is not None:
            pitch = None
            alter = None
            octave = None
            if e.find('rest/display-step') is not None:
                pitch = e.find('rest/display-step').text
            if e.find('rest/display-octave') is not None:
                octave = e.find('rest/display-octave').text
            return ('tie', (pitch, alter, octave, staff))
        else:
            return None


def get_pitch(e):
    """
    checks whether the element has a pitch and if so it
    returns pitch/alter/octave as a tuple.

    Returns
    -------
    tuple : (str, number, number) OR None
        `alter` may be None instead.
        The tuple contains (pitch, alter, octave)
    """

    if e.find('pitch') is not None:
        # ASSUMPTIONS
        pitch = e.find('pitch/step').text
        if e.find('pitch/alter') is not None:
            alter = int(e.find('pitch/alter').text)
        else:
            alter = None
        octave = int(e.find('pitch/octave').text)
        return (pitch, alter, octave)
    else:
        return None


def get_integer_from_attribute(e, attr, none_on_error=True):
    """
    This convenience function returns an integer value for a particular
    attribute of element e.

    Parameters
    ----------
    e: etree.Element
        An etree Element instance
    attr: string
        Attribute to retrieve
    none_on_error: bool, optional (default: True)
        When False, an exception is raised when `attr` is not found in `e` or the
        text inside `attr` cannot be cast to an integer. When True, None is returned
        in such cases.
    
    Returns
    -------
    integer or None
        The integer read from `attr`
    """
    try:
        return int(e.get(attr))
    except:
        if none_on_error:
            return None
        else:
            raise


def get_integer_from_tag(e, tag, none_on_error=True):
    """
    This convenience function returns an integer value for a particular
    tag in element e. For example, if element e represents
    <note><duration>2</duration></note>, get_integer_from_tag(e,
    'duration') will return 2.

    Parameters
    ----------
    e: etree.Element
        An etree Element instance
    tag: string
        Child tag to retrieve
    none_on_error: bool, optional (default: True)
        When False, an exception is raised when `tag` is not found in `e` or the
        text inside `tag` cannot be cast to an integer. When True, None is returned
        in such cases.
    
    Returns
    -------
    integer or None
        The integer read from `tag`
    """
    d = e.find(tag)
    try:
        return int(d.text)
    except:
        if none_on_error:
            return None
        else:
            raise

def get_string_from_tag(e, tag, none_on_error=True):
    """
    This convenience function returns an integer value for a particular
    tag in element e.

    Parameters
    ----------
    e: etree.Element
        An etree Element instance
    tag: string
        Child tag to retrieve
    none_on_error: bool, optional (default: True)
        When False, an exception is raised when `tag` is not found.
        When True, None is returned in such cases.

    Returns
    -------
    str or None
        The string read from `tag`
    """
    d = e.find(tag)
    try:
        return d.text
    except:
        if none_on_error:
            return None
        else:
            raise


def get_float_from_tag(e, tag):
    """This convenience function returns an float value for a particular
    tag in element e. For example, if element e represents
    <note><duration>2</duration></note>, get_integer_from_tag(e,
    'duration') will return 2

    """
    d = e.find(tag)
    return float(d.text) if d is not None else None


def get_duration(e):
    return get_integer_from_tag(e, 'duration')


def get_offset(e):
    el = e.find('offset')

    if el is None:
        return None

    sounding = el.attrib.get('sound', 'no')

    return int(el.text) if sounding == 'yes' else 0


def get_staff(e):
    return get_integer_from_tag(e, 'staff')


def get_voice(e):
    return get_integer_from_tag(e, 'voice')


def get_divisions(e):
    return get_integer_from_tag(e, 'divisions')

def get_coordinates(e):
    """
    get the coordinates of an element in pixel.
    """
    pixel_e = e.find('coordinates/pixel')

    if pixel_e is not None:
        top = get_float_from_tag(pixel_e, 'top')
        left = get_float_from_tag(pixel_e, 'left')
        return (top, left)
    else:
        return (None, None)


def get_time_signature(e):
    """
    Get the time signature and return

    Returns
    -------
    tuple (mumber, number) OR None
        the tuple is (beats, beat_type). `beats` is the numerator,
        `beat_type` is the denominator of the key signature fraction.
    """

    if e.find('time/beats') is not None and e.find('time/beat-type') is not None:
        beats = int(e.find('time/beats').text)
        beat_type = int(e.find('time/beat-type').text)
        return (beats, beat_type)
    else:
        return None


def get_key_signature(e):
    """
    Get the key signature. Note that the key is defined by the amount
    of fifths (+/-) it is away from c major. This of course equals to
    steps in the circle of fifths.

    Returns
    -------

    (fifths, mode) OR None : tuple of (number, 'str') OR None.
        Note that a tuple can be returned where either of both values
        is None, but not both of them; in the latter case only None is
        returned.
    """

    if e.find('key/fifths') is not None:
        fifths = int(e.find('key/fifths').text)
    else:
        fifths = None

    if e.find('key/mode') is not None:
        mode = e.find('key/mode').text
    else:
        mode = None
    if mode is None and fifths is None:
        return None
    else:
        return (fifths, mode)


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
        result.append(dict(number=get_integer_from_attribute(clef, 'number'),
                           sign=get_string_from_tag(clef, 'sign'),
                           line=get_integer_from_tag(clef, 'line'),
                           octave_change=get_integer_from_tag(clef, 'clef-octave-change')))
    return result

def get_transposition(e):
    """
    Get the transposition of the respective part. This is used for
    instruments that are notated in transposed fashion. Pitches will have
    to be transposed by the given amount to be in "correct" concert pitch
    (and thus in tune with the other instruments).

    Returns
    -------

    (diat, chrom) : tuple of int
        the amount of steps (+/-) that the pitches have to be adjusted.
        `diat` means steps in diatonic scale (could be min 2nd or maj 2nd?),
        `chrom` means steps in chromatic scale, thus half steps (min 2nd)
    """

    diat = None
    chrom = None

    if e.find('transpose/diatonic') is not None:
        diat = int(e.find('transpose/diatonic').text)
    if e.find('transpose/chromatic') is not None:
        chrom = int(e.find('transpose/chromatic').text)
    if diat is None and chrom is None:
        return None
    else:
        return (diat, chrom)


def _make_direction_new(label):
    """
    Parameters
    ----------
    label : str
        A performance direction label such as "crescendo", "ritardando", "allegro"
    Returns
    -------
    Direction object or None
    """
    r = parse_direction(label)
    if isinstance(score.Direction):
        return r
    else:
        return score.Words(label)
    

def make_measure(xml_measure):
    measure = score.Measure()
    try:
        measure.number = int(xml_measure.attrib['number'])
    except:
        LOGGER.warn('No number attribute found for measure')
    return measure


class PartBuilder(object):
    """

    Parameters
    ----------
    part :

    part_etree :

    no_default_note_id : boolean, optional. Default: False


    Attributes
    ----------
    timeline : TimeLine object

    """

    def __init__(self, part, part_etree, no_default_note_id=False):
        self.timeline = part.timeline
        self.part_etree = part_etree
        self.part_id = part.part_id
        self.ongoing = {}
        self.divisions = np.empty((0, 2))
        self.position = 0

        # start a new page by default
        self.handle_new_page(0)
        # start a new system by default
        self.handle_new_system(0)

        if no_default_note_id:
            self.default_note_id = None  # id counter for notes disabled
        else:
            self.default_note_id = 0  # id counter for notes

        self.current_measure = None

    def finalize(self):
        
        if self.ongoing:
            tp = self.timeline.get_or_add_point(self.position)
            for obj in list(self.ongoing.values()):
                if not isinstance(obj, (list, tuple)):
                    obj_list = [obj]
                else:
                    obj_list = obj
    
                for o in obj_list:
                    if not isinstance(o, (score.Page, score.System)):
                        LOGGER.warning(
                            'ongoing object "{}" until end of piece'.format(o))
                    end_pos = tp

                    # special case: 
                    if isinstance(o, score.DynamicTempoDirection):
                        ct = o.start.get_next_of_type(score.ConstantTempoDirection)
                        if any(c.text == 'a_tempo' for c in ct):
                            end_pos = ct[0].start

                    end_pos.add_ending_object(o)

        # when a divisions object occurs at any position other than the start of
        # the part, we need to correct the end time of any object that spans the
        # divisions object. This is because the end time was computed based on
        # the previous divisions object.
        divs = self.timeline.get_all(score.Divisions)
        if len(divs) > 0:
            prev_divs = 1
            for div in divs:
                ongoing = self.timeline.get_all_ongoing_objects(div.start.t)
                for o in ongoing:
                    if isinstance(o, score.Note):
                        new_end = int(div.start.t + div.divs * (o.end.t-div.start.t)/prev_divs)
                        self.timeline.remove_ending_object(o)
                        self.timeline.add_ending_object(new_end, o)
                prev_divs = div.divs

    def end_current_measure_at(self, t_quarter):
        """
        End the Measure object stored in `self.current_measure` at time `t`; if `t`
        > `self.position`, this implies `self.current_measure` is incomplete but
        is treated as a complete measure.
        """

        actual_end_q = np.sum(
            np.diff(self.measure_divs[:, 0]) / self.measure_divs[:-1, 1])
        missing_q = t_quarter - actual_end_q
        t = int(self.measure_divs[-1, 0] +
                missing_q * self.measure_divs[-1, 1])
        tp = self.timeline.get_or_add_point(t)
        tp.add_ending_object(self.current_measure)
        if t > self.position:
            LOGGER.warning('Part {0}, Measure {1}: treating incomplete measure as complete'
                           .format(self.part_id, self.current_measure.number))
            self.position = t

    def add_measure(self, xml_measure):
        """
        Parse a <measure>...</measure> element, adding it and its contents to the
        timeline.
        """
        # make a measure object
        self.current_measure = make_measure(xml_measure)

        # add the start of the measure to the time line
        tp = self.timeline.get_or_add_point(self.position)
        tp.add_starting_object(self.current_measure)

        # keep track of the position within the measure
        measure_pos = 0
        # keep track of the previous note (in case of <chord>)
        prev_note = None
        # keep track of the duration of the measure
        measure_duration = 0
        for i, e in enumerate(xml_measure):

            if e.tag == 'backup':
		# <xs:documentation>The backup and forward elements are required
		# to coordinate multiple voices in one part, including music on
		# multiple staves. The backup type is generally used to move
		# between voices and staves. Thus the backup element does not
		# include voice or staff elements. Duration values should always
		# be positive, and should not cross measure boundaries or
		# mid-measure changes in the divisions value.</xs:documentation>

                duration = get_duration(e) or 0
                measure_pos -= duration
                # <backup> tags trigger an update of the measure
                # duration up to the measure position (after the
                # <backup> has been processed); This has been found to
                # account for implicit measure durations in
                # Baerenreiter MusicXML files.
                measure_duration = max(measure_duration, measure_pos)

            elif e.tag == 'forward':
                duration = get_duration(e) or 0
                measure_pos += duration

            elif e.tag == 'attributes':
                self.handle_attributes(e, measure_pos)

            elif e.tag == 'direction':          # <direction> ... </direction>
                self.handle_direction(e, measure_pos)

            elif e.tag == 'print':
                self.handle_print(e, measure_pos)

            elif e.tag == 'sound':
                self.handle_sound(e, measure_pos)

            elif e.tag == 'note':
                (measure_pos, maybe_prev_note) = self.handle_note(e, measure_pos, prev_note)
                if maybe_prev_note:
                    prev_note = maybe_prev_note
                measure_duration = max(measure_duration, measure_pos)

            elif e.tag == 'barline':
                repeats = e.xpath('repeat')
                if len(repeats) > 0:
                    self.handle_repeat(repeats[0], measure_pos)

                endings = e.xpath('ending')
                if len(endings) > 0:
                    self.handle_ending(endings[0], measure_pos)

            else:
                print((e, type(e)))
                LOGGER.debug('ignoring tag {0}'.format(e.tag))

        self.position += measure_duration

        div = np.array([(self.current_measure.start.t, division.divs) for division
                        in self.current_measure.start.get_next_of_type(score.Divisions, eq=True)])

        if div.shape[0] > 0:
            self.divisions = np.vstack((self.divisions, div))

        start = np.searchsorted(self.divisions[:, 0], self.current_measure.start.t, side='right')
        end = np.searchsorted(self.divisions[:, 0], self.position, side='left')

        if self.divisions[start - 1, 0] < self.current_measure.start.t:
            self.measure_divs = np.vstack(((self.current_measure.start.t, self.divisions[start - 1, 1]),
                                           self.divisions[start:end, :]))
        else:
            self.measure_divs = self.divisions[start - 1:end, :]

        if end == self.divisions.shape[0]:
            self.measure_divs = np.vstack(
                (self.measure_divs, (self.position, self.divisions[end - 1, 1])))

        measure_duration_quarters = np.sum(
            np.diff(self.measure_divs[:, 0] / self.measure_divs[:-1, 1]))

        return self.position, measure_duration_quarters

    # <!--
    #     If a barline is other than a normal single barline, it
    #     should be represented by a barline element that describes
    #     it. This includes information about repeats and multiple
    #     endings, as well as line style. Barline data is on the same
    #     level as the other musical data in a score - a child of a
    #     measure in a partwise score, or a part in a timewise score.
    #     This allows for barlines within measures, as in dotted
    #     barlines that subdivide measures in complex meters. The two
    #     fermata elements allow for fermatas on both sides of the
    #     barline (the lower one inverted).

    #     Barlines have a location attribute to make it easier to
    #     process barlines independently of the other musical data
    #     in a score. It is often easier to set up measures
    #     separately from entering notes. The location attribute
    #     must match where the barline element occurs within the
    #     rest of the musical data in the score. If location is left,
    #     it should be the first element in the measure, aside from
    #     the print, bookmark, and link elements. If location is
    #     right, it should be the last element, again with the
    #     possible exception of the print, bookmark, and link
    #     elements. If no location is specified, the right barline
    #     is the default. The segno, coda, and divisions attributes
    #     work the same way as in the sound element defined in the
    #     direction.mod file. They are used for playback when barline
    #     elements contain segno or coda child elements.
    # -->

    # <!-- Elements -->

    # <!ELEMENT barline (bar-style?, %editorial;, wavy-line?,
    #     segno?, coda?, (fermata, fermata?)?, ending?, repeat?)>
    # <!ATTLIST barline
    #     location (right | left | middle) "right"
    #     segno CDATA #IMPLIED
    #     coda CDATA #IMPLIED
    #     divisions CDATA #IMPLIED
    # >

    def handle_repeat(self, repeat, measure_pos):
        if repeat.attrib['direction'] == 'forward':
            o = score.Repeat()
            self.ongoing[get_repeat_key(repeat)] = o
            self.timeline.add_starting_object(
                self.position + measure_pos, o)

        elif repeat.attrib['direction'] == 'backward':
            key = get_repeat_key(repeat)
            if key in self.ongoing:
                o = self.ongoing[key]
                del self.ongoing[key]
            else:
                # implicit repeat start: create Repeat
                # object and add it at the beginning of
                # the self.timeline retroactively
                o = score.Repeat()
                self.timeline.add_starting_object(0, o)
            self.timeline.add_ending_object(
                self.position + measure_pos, o)

    def handle_ending(self, ending, measure_pos):
        if ending.attrib['type'] == 'start':
            o = score.Ending(ending.attrib['number'])
            self.ongoing[get_ending_key(ending)] = o
            self.timeline.add_starting_object(
                self.position + measure_pos, o)
        elif ending.attrib['type'] in ['stop', 'discontinue']:
            key = get_ending_key(ending)
            if key in self.ongoing:
                o = self.ongoing[key]
                del self.ongoing[key]
                self.timeline.add_ending_object(
                    self.position + measure_pos, o)
            else:
                LOGGER.warning(
                    'Found ending[stop] without a preceding ending[start]')

    def handle_direction(self, e, measure_pos):

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

        offset = get_offset(e) or 0

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
                        direction = DYNAMICS_DIRECTIONS.get(
                            dyn.tag, score.Words)(dyn.tag)
                        if direction is not None:
                            starting_directions.append(direction)

            elif direction_type == 'words':
                # there may be multiple dynamics/words items in dts, loop:
                for dt in dts:
                    # try to make a direction out of words
                    parse_result = parse_words(str(dt.text))

                    if parse_result is not None:

                        # starting_directions.append(direction)

                        if isinstance(parse_result, list) or isinstance(parse_result, tuple):
                            starting_directions.extend(parse_result)
                        else:
                            starting_directions.append(parse_result)

                    # else:
                    #     # if it fails, add the text as Words
                    #     starting_directions.append(Words(unicode(dt.text)))

            elif direction_type == 'wedge':
                key = get_wedge_key(dts[0])
                if dts[0].attrib['type'] in ('crescendo', 'diminuendo'):
                    o = score.DynamicLoudnessDirection(dts[0].attrib['type'])
                    starting_directions.append(o)
                    self.ongoing[key] = o
                elif dts[0].attrib['type'] == 'stop':
                    o = self.ongoing.get(key, None)
                    if o is not None:
                        ending_directions.append(o)
                        del self.ongoing[key]
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
            self.ongoing[key] = starting_directions

        if dashes_end is not None:
            key = get_dashes_key(dashes_end)
            oo = self.ongoing.get(key, None)
            if oo is None:
                LOGGER.warning('Dashes end without dashes start')
            else:
                ending_directions.extend(oo)
                del self.ongoing[key]

        for o in starting_directions:
            self.timeline.add_starting_object(
                self.position + measure_pos, o)

        for o in ending_directions:
            self.timeline.add_ending_object(
                self.position + measure_pos, o)

    def handle_note(self, e, measure_pos, prev_note):

        # get some common features of element if available
        duration = get_duration(e) or 0
        # elements may have an explicit temporal offset
        offset = get_offset(e) or 0
        staff = get_staff(e) or 0
        voice = get_voice(e) or 0

        if 'id' in e.attrib:
            note_id = e.attrib['id']
        else:
            if self.default_note_id is None:
                note_id = None
            else:
                note_id = 'n{:d}'.format(self.default_note_id)
                self.default_note_id += 1 

            # Warning: automatic ID assignment is only intended for
            # situations where none of the notes have IDs; in case both
            # notated and automically assigned IDs are used, there is no
            # checking for duplicate IDs

        # grace note handling
        grace_type = None
        steal_proportion = None

        try:
            gr = e.xpath('grace[1]')[0]
        except IndexError:
            gr = None

        if gr is not None:
            grace_type = 'grace'

            slash_text = gr.attrib.get('slash', None)
            if slash_text == 'yes':
                grace_type = 'acciaccatura'

            app_text = gr.attrib.get('steal-time-following', None)
            if app_text is not None:
                steal_proportion = float(app_text) / 100
                grace_type = 'appoggiatura'

            acc_text = gr.attrib.get('steal-time-previous', None)
            if acc_text is not None:
                steal_proportion = float(acc_text) / 100
                grace_type = 'acciaccatura'

        staccato = len(e.xpath('notations/articulations/staccato')) > 0
        accent = len(e.xpath('notations/articulations/accent')) > 0
        fermata = len(e.xpath('notations/fermata')) > 0
        chord = len(e.xpath('chord')) > 0
        dur_type_els = e.xpath('type')
        n_dur_dots = len(e.xpath('dot'))

        if len(dur_type_els) > 0:
            symbolic_duration = dict(
                type=dur_type_els[0].text, dots=n_dur_dots)
        else:
            symbolic_duration = dict()

        time_mod_els = e.xpath('time-modification')
        if len(time_mod_els) > 0:
            symbolic_duration['actual_notes'] = time_mod_els[0].xpath(
                'actual-notes')[0].text
            symbolic_duration['normal_notes'] = time_mod_els[0].xpath(
                'normal-notes')[0].text

        if chord:
            # this note starts at the same position as the previous note, and has same duration
            assert prev_note is not None
            measure_pos, duration = prev_note.start.t - self.position, prev_note.duration
            
            # add chord
            if 'chord' in self.ongoing:
                chord = self.ongoing['chord']
            else:
                chord = score.Chord()
                self.timeline.add_starting_object(prev_note.start.t, chord)
                self.ongoing['chord'] = chord
        else:
            # end the current chord
            if 'chord' in self.ongoing:
                chord = self.ongoing['chord']
                self.timeline.add_ending_object(
                    self.position + measure_pos, chord)
                del self.ongoing['chord']

        if fermata:
            self.timeline.add_starting_object(
                self.position + measure_pos, score.Fermata())

        # look for <notations> tags. Inside them, <tied> and <slur>
        # may be present. Note that for a tie, a <tied> should be present
        # here as well as a <tie> tag inside the <note> ... </note> tags
        # of the same note (this is not looked for here). The code
        # so far only looks for <slur> here.
        if len(e.xpath('notations')) > 0:

            eslurs = e.xpath('notations/slur')    # list

            # TODO: Slur stop can preceed slur start in document order (in the
            # case of <backup>). Current implementation does not recognize that.

            # this sorts all found slurs by type (either 'start' or 'stop')
            # in reverse order, so all with type 'stop' will be before
            # the ones with 'start'?!.
            eslurs.sort(key=lambda x: x.attrib['type'], reverse=True)

            # Now that the slurs are sorted by their type, sort them
            # by their numbers; First note that slurs do not always
            # have a number attribute, then 1 is implied.
            # If, however, either more than one slur starts
            # or ends at the same note (!) they must be
            # numbered so that they can be distinguished. If however
            # a (single) slur ends and the next (single) one starts
            # at the same note, none of them needs to be numbered.
            eslurs.sort(key=lambda x: int(
                x.attrib['number']) if 'number' in x.attrib else 1)

            erroneous_stops = []    # gather stray stops that have no beginning?

            for eslur in eslurs:    # loop over all found slurs
                key = get_slur_key(eslur)    # number that represents the slur
                if eslur.attrib['type'] == 'stop':
                    try:
                        o = self.ongoing[key]
                        self.timeline.add_ending_object(
                            self.position + measure_pos, o)
                        del self.ongoing[key]
                    except KeyError:  # as exception:

                        note_id = e.attrib.get('ID', None)
                        LOGGER.warning(("Part xx, Measure xx: Stopping slur with number {0} was never started (Note ID: {1})"
                                        "").format(key[1], note_id))

                        erroneous_stops.append(key)

                elif eslur.attrib['type'] == 'start':
                    # # first check if a slur with our key was already
                    # # started (unfortunately, this happens in the
                    # # Zeilingercorpus)
                    # if key in self.ongoing:
                    #     LOGGER.warning("Slur with number {0} started twice; Assuming an implicit stopping of the first slur".format(key[1]))
                    #     o = self.ongoing[key]
                    #     self.timeline.add_ending_object(self.position + measure_pos, o)
                    #     del self.ongoing[key]

                    if key not in self.ongoing:
                        o = score.Slur(voice)
                        self.timeline.add_starting_object(
                            self.position + measure_pos, o)
                        self.ongoing[key] = o
                    else:
                        LOGGER.warning(
                            "Slur with number {0} started twice; Ignoring the second slur start".format(key[1]))
            for k in erroneous_stops:
                if k in self.ongoing:
                    del self.ongoing[k]

        # pitch will be None if there is no <pitch> ... </pitch> tag
        pitch = get_pitch(e)

        if pitch is not None:
            # NON-REST NOTE ELEMENT ###################################
            step, alter, octave = pitch

            # MG: it happens that started ties have no ending
            # tie. Nevertheless, by necessity the next occurrence of the
            # same note must end the tie, so rather than looking for tie
            # end tags, we end any started tie as soon as the tie-key of a
            # note is found in `self.ongoing`

            # generate a key for the tied note
            # `key` looks like: ('tie', ('E', None, 6, 1)),
            # that is ('tie', (step, alter, octave, staff))
            tie_key = get_tie_key(e)

            # according to the definition, there are no more than two tie elements in a note
            ties = e.xpath('tie')
            tie_types = [tie.attrib['type'] for tie in ties]

            if tie_key in self.ongoing:

                note = self.ongoing[tie_key]

                if symbolic_duration:
                    note.symbolic_durations.append(symbolic_duration)

            else:

                note = score.Note(step, alter, octave, voice=get_voice(e), id=note_id,
                               grace_type=grace_type, staccato=staccato, fermata=fermata,
                               steal_proportion=steal_proportion, symbolic_duration=symbolic_duration,
                               accent=accent, staff=staff)

                self.timeline.add_starting_object(self.position + measure_pos, note)
                
            if 'start' in tie_types:
                
                self.ongoing[tie_key] = note

            else:

                self.timeline.add_ending_object(self.position+measure_pos+duration, note)

                if tie_key in self.ongoing:
                    del self.ongoing[tie_key]

        else:
            # note element is a rest
            note = None

        # return the measure_pos after this note, and the pair (measure_pos, duration)
        
        return measure_pos + duration, note # (measure_pos, duration)


    def handle_attributes(self, e, measure_pos):
        """

        """
        ts = get_time_signature(e)

        ts_num = None
        ts_den = None
        if ts is not None:
            ts_num, ts_den = ts
            self.timeline.add_starting_object(self.position + measure_pos,
                                              score.TimeSignature(ts_num, ts_den))
            # self.quarters_per_measure = ts_num * 4 / ts_den

        ks = get_key_signature(e)
        if ks is not None:
            self.timeline.add_starting_object(self.position + measure_pos,
                                              score.KeySignature(ks[0], ks[1]))

        tr = get_transposition(e)
        if tr is not None:
            self.timeline.add_starting_object(self.position + measure_pos,
                                              score.Transposition(tr[0], tr[1]))
            
        divs = get_divisions(e)
        if divs is not None:
            self.timeline.add_starting_object(self.position + measure_pos,
                                              score.Divisions(divs))

        clefs = get_clefs(e)
        for clef in clefs:
            self.timeline.add_starting_object(self.position + measure_pos,
                                              score.Clef(**clef))

        return divs, ts_num, ts_den

    def handle_new_page(self, measure_pos):
        if 'page' in self.ongoing:
            # if self.ongoing['page'].start.t == self.position + measure_pos:
            if self.position + measure_pos == 0:
                # LOGGER.debug('ignoring non-informative new-page at start of score')
                return

            self.timeline.add_ending_object(self.position + measure_pos,
                                            self.ongoing['page'])
            page_nr = self.ongoing['page'].nr + 1
        else:
            page_nr = 1

        page = score.Page(page_nr)
        self.timeline.add_starting_object(self.position + measure_pos, page)
        self.ongoing['page'] = page


    def handle_new_system(self, measure_pos):
        if 'system' in self.ongoing:

            if self.position+measure_pos == 0:
                # LOGGER.debug('ignoring non-informative new-system at start of score')
                return

            # end current page
            self.timeline.add_ending_object(self.position+measure_pos,
                                            self.ongoing['system'])
            system_nr = self.ongoing['system'].nr + 1
        else:
            system_nr = 1

        system = score.System(system_nr)
        self.timeline.add_starting_object(self.position+measure_pos, system)
        self.ongoing['system'] = system


    def handle_print(self, e, measure_pos):
        if "new-page" in e.attrib:
            self.handle_new_page(measure_pos)
            self.handle_new_system(measure_pos)
        if "new-system" in e.attrib:
            self.handle_new_system(measure_pos)

    def handle_sound(self, e, measure_pos):
        if "tempo" in e.attrib:
            self.timeline.add_starting_object(self.position + measure_pos,
                                              Tempo(int(e.attrib['tempo'])))


def parse_parts(document, part_dict):
    """

    Parameters
    ----------
    document : lxml etree object (?)

    part_dict : dictionary
        a dictionary as returned by the parse_partlist() function.
    """

    # initialize a PartBuilder instance for each part
    part_builders = [PartBuilder(part_dict.get(part_id, score.Part(part_id)),
                                      document.xpath('/score-partwise/part[@id="{0}"]'.format(part_id))[0])
                     for part_id in document.xpath('/score-partwise/part/@id')]
    assert len(part_builders) > 0

    # a list of measures of each score part (in a list)
    measures_per_sp = [part_builder.part_etree.xpath('measure')
                       for part_builder in part_builders]

    # equal number of measures in all score parts
    assert len(set(len(measure) for measure in measures_per_sp)) == 1

    # number of measures (in the first score part)
    n_measures = len(measures_per_sp[0])

    # we build the score parts measure by measure in parallel, rather than
    # building the score parts one after another. This makes it possible to deal
    # with the case where the measures of some score parts are shorter than the
    # corresponding measures in other score parts by adjusting the length of
    # incomplete measures to the length of the longest measure.
    pos = 0
    for j in range(n_measures):

        # position_dict = {}
        measure_max_end = 0
        for i, (measure, part_builder) in enumerate(zip(measures_per_sp,
                                                        part_builders)):
            _, measure_end = part_builder.add_measure(measure[j])
            measure_max_end = max(measure_max_end, measure_end)
        # end all measures at max_pos
        for part_builder in part_builders:
            part_builder.end_current_measure_at(measure_max_end)

    for part_builder in part_builders:
        part_builder.finalize()
        print(part_builder.timeline)


def load_musicxml(fn, schema=None):
    """
    Parse a MusicXML file and build a composite score ontology
    structure from it (see also scoreontology.py).

    Parameters
    ----------
    fn : str
        Path to the MusicXML file to be parsed.
    schema : str or None (optional, default=None)
        Path to the MusicXML XSD specification. When specified, the validity of
        the MusicXML is checked before loading the file. An exception will be 
        raised when the MusicXML is invalid.

    Returns
    -------
    partlist : list
        A list of either Part or PartGroup objects

    """
    if schema:
        validate(schema, fn, debug=True)

    parser = etree.XMLParser(resolve_entities=False, huge_tree=False,
                             remove_comments=True, remove_blank_text=True)
    document = etree.parse(fn, parser)
    
    if document.getroot().tag != 'score-partwise':
        raise Exception('Currently only score-partwise structure is supported')

    partlist_el = document.xpath('part-list')
    
    if partlist_el:
        # parse the (hierarchical) structure of score parts
        # (instruments) that are listed in the part-list element
        partlist, part_dict = parse_partlist(partlist_el[0])
        # go through each <part> to obtain the content of the parts
        parse_parts(document, part_dict)
    else:
        partlist = []
    

    return partlist


SCORE_DTYPES = [('pitch', 'i4'), ('onset', 'f4'), ('duration', 'f4')]


def xml_to_notearray(fn, flatten_parts=True, sort_onsets=True,
                     expand_grace_notes=True):
    """
    Get a note array from a MusicXML file

    Parameters
    ----------
    fn : str
        Path to a MusicXML file
    flatten_parts : bool
        If `True`, returns a single array containing all notes.
        Otherwise, returns a list of arrays for each part.
    expand_grace_notes : bool or 'delete'

    Returns
    -------
    score : structured array or list of structured arrays
        Structured array containing the score. The fields are
        'pitch', 'onset' and 'duration'.
    """

    if not isinstance(expand_grace_notes, (bool, str)):
        raise ValueError('`expand_grace_notes` must be a boolean or '
                         '"delete"')
    delete_grace_notes = False
    if isinstance(expand_grace_notes, str):

        if expand_grace_notes in ('omit', 'delete', 'd'):
            expand_grace_notes = False
            delete_grace_notes = True
        else:
            raise ValueError('`expand_grace_notes` must be a boolean or '
                             '"delete"')

    # Parse MusicXML
    parts = load_musicxml(fn)

    score = []
    for part in parts:
        # Unfold timeline to have repetitions
        part.timeline = part.unfold_timeline()

        if expand_grace_notes:
            LOGGER.debug('Expanding grace notes...')
            part.expand_grace_notes()
        # get beat map
        bm = part.beat_map
        # Build score from beat map
        _score = np.array(
            [(n.midi_pitch, bm(n.start.t), bm(n.end.t) - bm(n.start.t))
             for n in part.notes],
            dtype=SCORE_DTYPES)

        # Sort notes according to onset
        if sort_onsets:
            _score = _score[_score['onset'].argsort()]

        if delete_grace_notes:
            LOGGER.debug('Deleting grace notes...')
            _score = _score[_score['duration'] != 0]
        score.append(_score)

    # Return a structured array if the score has only one part
    if len(score) == 1:
        return score[0]
    elif len(score) > 1 and flatten_parts:
        score = np.vstack(score)
        if sort_onsets:
            return score[score['onset'].argsort()]
    else:
        return score


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

                gr_name = None
                group_name = e.xpath('group-name')
                if len(group_name) > 0:
                    gr_name = group_name[0].text

                gr_type = None
                grouping = e.xpath('group-symbol')
                if len(grouping) > 0:
                    gr_type = grouping[0].text

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
            part_id = e.attrib['id']  # like "P1"
            sp = score.Part(part_id, score.TimeLine())

            try:
                sp.part_name = e.xpath('part-name/text()')[0]
            except:
                pass

            try:
                sp.part_abbreviation = e.xpath('part-abbreviation/text()')[0]
            except:
                pass

            part_dict[part_id] = sp

            if current_group is None:
                structure.append(sp)
            else:
                current_group.constituents.append(sp)
                sp.parent = current_group

    if current_group is not None:
        LOGGER.warning(
            'part-group {0} was not ended'.format(current_group.number))
        structure.append(current_group)

    return structure, part_dict


# <xs:group name="music-data">
# 	<xs:annotation>
# 		<xs:documentation>The music-data group contains the basic musical data that is either associated with a part or a measure, depending on whether the partwise or timewise hierarchy is used.</xs:documentation>
# 	</xs:annotation>
# 	<xs:sequence>
# 		<xs:choice minOccurs="0" maxOccurs="unbounded">
# 			<xs:element name="note" type="note"/>
# 			<xs:element name="backup" type="backup"/>
# 			<xs:element name="forward" type="forward"/>
# 			<xs:element name="direction" type="direction"/>
# 			<xs:element name="attributes" type="attributes"/>
# 			<xs:element name="harmony" type="harmony"/>
# 			<xs:element name="figured-bass" type="figured-bass"/>
# 			<xs:element name="print" type="print"/>
# 			<xs:element name="sound" type="sound"/>
# 			<xs:element name="barline" type="barline"/>
# 			<xs:element name="grouping" type="grouping"/>
# 			<xs:element name="link" type="link"/>
# 			<xs:element name="bookmark" type="bookmark"/>
# 		</xs:choice>
# 	</xs:sequence>
# </xs:group>

