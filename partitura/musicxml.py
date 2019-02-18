#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
from collections import defaultdict
import logging

import numpy as np
from lxml import etree

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
    'mf': score.ConstantLoudnessDirection,
    'mp': score.ConstantLoudnessDirection,
    'pp': score.ConstantLoudnessDirection,
    'ppp': score.ConstantLoudnessDirection,
    'pppp': score.ConstantLoudnessDirection,
    'ppppp': score.ConstantLoudnessDirection,
    'pppppp': score.ConstantLoudnessDirection,
    'fp': score.ImpulsiveLoudnessDirection,
    'rf': score.ImpulsiveLoudnessDirection,
    'rfz': score.ImpulsiveLoudnessDirection,
    'fz': score.ImpulsiveLoudnessDirection,
    'sf': score.ImpulsiveLoudnessDirection,
    'sffz': score.ImpulsiveLoudnessDirection,
    'sfp': score.ImpulsiveLoudnessDirection,
    'sfpp': score.ImpulsiveLoudnessDirection,
    'sfz': score.ImpulsiveLoudnessDirection,
}

# the _get_*_key functions are helper functions for musical elements
# spanning a time interval. The start of those elements gets added to
# the timeline, and then they need to be stored until their end is
# encountered (e.g. wedges, ties, slurs). The keys that these
# functions produce must be such that the starting and the stopping
# xml elements produce the same key

def _get_wedge_key(e):
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


def _get_dashes_key(e):
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


def _get_repeat_key(e):
    """
    Generate a key to store repeats temporarily
    """
    return ('repeat',)


def _get_ending_key(e):
    """
    Generate a key to store repeats temporarily
    """
    return ('ending',)


def _get_slur_key(e):
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


def _get_tie_key(e):
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
    """

    pitchk = _get_pitch(e)    # tuple (step, alter, octave)
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


def _get_pitch(e):
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


def _get_integer_from_tag(e, tag):
    """This convenience function returns an integer value for a particular
    tag in element e. For example, if element e represents
    <note><duration>2</duration></note>, _get_integer_from_tag(e,
    'duration') will return 2

    """
    d = e.find(tag)
    return int(d.text) if d is not None else None


def _get_float_from_tag(e, tag):
    """This convenience function returns an float value for a particular
    tag in element e. For example, if element e represents
    <note><duration>2</duration></note>, _get_integer_from_tag(e,
    'duration') will return 2

    """
    d = e.find(tag)
    return float(d.text) if d is not None else None


def _get_duration(e):
    return _get_integer_from_tag(e, 'duration')


def _get_offset(e):
    el = e.find('offset')

    if el is None:
        return None

    sounding = el.attrib.get('sound', 'no')

    return int(el.text) if sounding == 'yes' else 0


def _get_staff(e):
    return _get_integer_from_tag(e, 'staff')


def _get_voice(e):
    return _get_integer_from_tag(e, 'voice')


def _get_divisions(e):
    return _get_integer_from_tag(e, 'divisions')


def _get_coordinates(e):
    """
    get the coordinates of an element in pixel.
    """
    pixel_e = e.find('coordinates/pixel')

    if pixel_e is not None:
        top = _get_float_from_tag(pixel_e, 'top')
        left = _get_float_from_tag(pixel_e, 'left')
        return (top, left)
    else:
        return (None, None)


def _get_time_signature(e):
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


def _get_key_signature(e):
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


def _get_transposition(e):
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

    # page and system attributes are non-standard, but are used by
    # either SharpEye, or Flossmann's code
    try:
        # do some measures have a page number as an attribute? TG
        measure.page = int(xml_measure.attrib['page'])
    except:
        # LOGGER.warn('No page attribute found for measure')
        pass

    try:
        measure.system = int(xml_measure.attrib['system'])
    except:
        # LOGGER.warn('No system attribute found for measure')
        pass

    return measure


class ScorePartBuilder(object):
    """

    Parameters
    ----------
    scorepart :

    part_etree :

    disable_default_note_id : boolean, optional. Default: False


    Attributes
    ----------
    timeline : TimeLine object

    part_etree :

    part_id : str

    ongoing : dictionary

    divisions :

    position :

    page_nr :

    system_nr :

    """

    def __init__(self, scorepart, part_etree, disable_default_note_id=False):
        self.timeline = scorepart.timeline
        self.part_etree = part_etree
        self.part_id = scorepart.part_id
        self.ongoing = {}
        self.divisions = np.empty((0, 2))
        self.position = 0
        self.page_nr = 0
        self.system_nr = 0

        # start a new page by default
        self._do_new_page(0)
        # start a new system by default
        self._do_new_system(0)

        self.disable_default_note_id = disable_default_note_id
        if self.disable_default_note_id:
            self.default_note_id = None  # id counter for notes disabled
        else:
            self.default_note_id = 0  # id counter for notes
        # self.quarters_per_measure = 4
        self.current_measure = None

    def finalize(self):
        tp = self.timeline.get_or_add_point(self.position)
        self.timeline.lock()
        self.timeline.link()
        self.timeline.unlock()
        for obj in list(self.ongoing.values()):
            if not isinstance(obj, (list, tuple)):
                obj_list = [obj]
            else:
                obj_list = obj

            for o in obj_list:
                LOGGER.warning(
                    'ongoing object "{}" until end of piece'.format(o))
                end_pos = tp
                if isinstance(o, score.DynamicTempoDirection):
                    ct = o.start.get_next_of_type(score.ConstantTempoDirection)
                    if any(c.text == 'a_tempo' for c in ct):
                        end_pos = ct[0].start
                end_pos.add_ending_object(o)

        self.timeline.lock()
        # self.timeline.link()

    def close_current_measure_at(self, t_quarter):
        """End the Measure object stored in `self.current_measure` at time
        `t`; if `t` > `self.position`, this implies
        `self.current_measure` is incomplete but is treated as a
        complete measure.

        """

        actual_end_q = np.sum(
            np.diff(self.measure_divs[:, 0]) / self.measure_divs[:-1, 1])
        missing_q = t_quarter - actual_end_q
        t = int(self.measure_divs[-1, 0]
                + missing_q * self.measure_divs[-1, 1])
        tp = self.timeline.get_or_add_point(t)
        tp.add_ending_object(self.current_measure)
        if t > self.position:
            LOGGER.warning('Part {0}, Measure {1}: treating incomplete measure as complete'
                           .format(self.part_id, self.current_measure.number))
            self.position = t

    def add_measure(self, xml_measure):
        """

        """
        # make a measure object
        self.current_measure = make_measure(xml_measure)

        # add the start of the measure to the time line
        tp = self.timeline.get_or_add_point(self.position)
        tp.add_starting_object(self.current_measure)

        # keep track of the position within the measure
        measure_position = 0
        # keep track of the start of the previous note (in case of <chord>)
        prev_note_start_dur = None
        # keep track of the duration of the measure
        measure_duration = 0
        for i, e in enumerate(xml_measure):

            if e.tag == 'backup':
                # print(e.tag)
                duration = _get_duration(e) or 0
                measure_position -= duration
                # <backup> tags trigger an update of the measure
                # duration up to the measure position (after the
                # <backup> has been processed); This has been found to
                # account for implicit measure durations in
                # Baerenreiter MusicXML files.
                measure_duration = max(measure_duration, measure_position)

            elif e.tag == 'forward':
                duration = _get_duration(e) or 0
                measure_position += duration

            elif e.tag == 'note':
                (measure_position, prev_note_start_dur) = self._do_note(
                    e, measure_position, prev_note_start_dur)
                # <note> tags trigger an update of the measure
                # duration up to the measure position (after the
                # <note> has been processed )
                measure_duration = max(measure_duration, measure_position)

            elif e.tag == 'barline':
                repeats = e.xpath('repeat')
                if len(repeats) > 0:
                    self._do_repeat(repeats[0], measure_position)

                endings = e.xpath('ending')
                if len(endings) > 0:
                    self._do_ending(endings[0], measure_position)

            elif e.tag == 'attributes':
                self._do_attributes(e, measure_position)

            elif e.tag == 'direction':          # <direction> ... </direction>
                self._do_direction(e, measure_position)

            elif e.tag == 'print':
                self._do_print(e, measure_position)

            elif e.tag == 'sound':
                self._do_sound(e, measure_position)
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

    def _do_repeat(self, repeat, measure_position):
        if repeat.attrib['direction'] == 'forward':
            o = score.Repeat()
            self.ongoing[_get_repeat_key(repeat)] = o
            self.timeline.add_starting_object(
                self.position + measure_position, o)

        elif repeat.attrib['direction'] == 'backward':
            key = _get_repeat_key(repeat)
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
                self.position + measure_position, o)

    def _do_ending(self, ending, measure_position):
        if ending.attrib['type'] == 'start':
            o = score.Ending(ending.attrib['number'])
            self.ongoing[_get_ending_key(ending)] = o
            self.timeline.add_starting_object(
                self.position + measure_position, o)
        elif ending.attrib['type'] in ['stop', 'discontinue']:
            key = _get_ending_key(ending)
            if key in self.ongoing:
                o = self.ongoing[key]
                del self.ongoing[key]
                self.timeline.add_ending_object(
                    self.position + measure_position, o)
            else:
                LOGGER.warning(
                    'Found ending[stop] without a preceding ending[start]')

    def _do_direction(self, e, measure_position):

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

        offset = _get_offset(e) or 0

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
                key = _get_wedge_key(dts[0])
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
            key = _get_dashes_key(dashes_start)
            self.ongoing[key] = starting_directions

        if dashes_end is not None:
            key = _get_dashes_key(dashes_end)
            oo = self.ongoing.get(key, None)
            if oo is None:
                LOGGER.warning('Dashes end without dashes start')
            else:
                ending_directions.extend(oo)
                del self.ongoing[key]

        for o in starting_directions:
            self.timeline.add_starting_object(
                self.position + measure_position, o)

        for o in ending_directions:
            self.timeline.add_ending_object(
                self.position + measure_position, o)

    def _do_note(self, e, measure_position, prev_note_start_dur):
        # Note: part_builder is only passed as an argument to report the
        # part ID/ measure number when an error is encountered

        # get some common features of element if available
        duration = _get_duration(e) or 0
        # elements may have an explicit temporal offset
        offset = _get_offset(e) or 0
        staff = _get_staff(e) or 0
        voice = _get_voice(e) or 0

        try:
            note_id = e.attrib['ID']
            if self.default_note_id is not None and self.default_note_id > 0:
                raise Exception(('MusicXML file contains both <note> tags with '
                                 'a ID attribute (which is non-standard), and without a ID attribute. '
                                 'The parser is configured to set missing ID attributes automatically, '
                                 'but this is only safe when there are no predefined ID attributes. To '
                                 'deal with this, either: define ID attributes for *all* <note> tags, or '
                                 'disable automatic ID-attribute assignment'))
        except KeyError:   # as ex:
            if self.default_note_id is None:
                note_id = None
            else:
                note_id = str(self.default_note_id)
                self.default_note_id += 1  # notes without id will be numbered consecutively

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
                'actual-notes')[0]
            symbolic_duration['normal_notes'] = time_mod_els[0].xpath(
                'normal-notes')[0]

        if chord:
            # this note starts at the same position as the previous note, and has same duration
            assert prev_note_start_dur is not None
            measure_position, duration = prev_note_start_dur

        if fermata:
            self.timeline.add_starting_object(
                self.position + measure_position, score.Fermata())

        if len(e.xpath('coordinates')) > 0:
            coordinates = _get_coordinates(e)
        else:
            coordinates = None

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
                key = _get_slur_key(eslur)    # number that represents the slur
                if eslur.attrib['type'] == 'stop':
                    try:
                        o = self.ongoing[key]
                        self.timeline.add_ending_object(
                            self.position + measure_position, o)
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
                    #     self.timeline.add_ending_object(self.position + measure_position, o)
                    #     del self.ongoing[key]

                    if key not in self.ongoing:
                        o = score.Slur(voice)
                        self.timeline.add_starting_object(
                            self.position + measure_position, o)
                        self.ongoing[key] = o
                    else:
                        LOGGER.warning(
                            "Slur with number {0} started twice; Ignoring the second slur start".format(key[1]))
            for k in erroneous_stops:
                if k in self.ongoing:
                    del self.ongoing[k]

        # pitch will be None if there is no <pitch> ... </pitch> tag
        pitch = _get_pitch(e)

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
            tie_key = _get_tie_key(e)

            if tie_key in self.ongoing:
                self.timeline.add_ending_object(
                    self.position + measure_position, self.ongoing[tie_key])
                if symbolic_duration is not None:
                    self.ongoing[tie_key].symbolic_durations.append(
                        symbolic_duration)
                del self.ongoing[tie_key]

            if len(e.xpath('tie')) > 0:    # look for a <tie> tag.
                tietypes = [tie.attrib['type'] for tie in e.xpath('tie')]
                tie_key = _get_tie_key(e)

                # TG: NOTE: it may be useful to integrate the `voice`
                # number into the `key`. However, when a multipart, i.e.
                # two or more separate voices (in the same staff?) turn
                # into a <chord> (because their stems are joined
                # together), all notes of the chord will have the same
                # voice tag from now on. This will then break the
                # meaning of the voice tag in the key and will cause
                # problems.

                # we ignore stop ties, since they are redundant

                # if 'stop' in tietypes:

                #     # this is the stop note/rest
                #     # combine notes
                #     if not self.ongoing.has_key(tie_key):
                #         # LOGGER.warning('No tie-start found for tie-stop')
                #         pass

                if 'start' in tietypes:
                    # this is the start note/rest
                    # save the note under the tie_key
                    # until we encounter the stop note/rest
                    o = score.Note(step, alter, octave, voice=_get_voice(e), id=note_id,
                                   grace_type=grace_type, staccato=staccato, fermata=fermata,
                                   steal_proportion=steal_proportion, symbolic_duration=symbolic_duration,
                                   accent=accent, coordinates=coordinates, staff=staff)
                    self.timeline.add_starting_object(self.position + measure_position, o)
                    self.ongoing[tie_key] = o
            else:
                o = score.Note(step, alter, octave, voice=_get_voice(e), id=note_id,
                               grace_type=grace_type, staccato=staccato, fermata=fermata,
                               steal_proportion=steal_proportion, symbolic_duration=symbolic_duration,
                               accent=accent, coordinates=coordinates, staff=staff)
                self.timeline.add_starting_object(self.position + measure_position, o)
                self.timeline.add_ending_object(self.position + measure_position + duration, o)
        else:
            # REST (NOTE) ELEMENT #####################################
            pass

        # return the measure_position after this note, and also the start
        # position of this note (within the measure)
        return measure_position + duration, (measure_position, duration)

    def _do_attributes(self, e, measure_position):
        """

        """
        ts = _get_time_signature(e)

        ts_num = None
        ts_den = None
        if ts is not None:
            ts_num, ts_den = ts
            self.timeline.add_starting_object(self.position + measure_position,
                                              score.TimeSignature(ts_num, ts_den))
            # self.quarters_per_measure = ts_num * 4 / ts_den

        ks = _get_key_signature(e)
        if ks is not None:
            self.timeline.add_starting_object(self.position + measure_position,
                                              score.KeySignature(ks[0], ks[1]))

        tr = _get_transposition(e)
        if tr is not None:
            self.timeline.add_starting_object(self.position + measure_position,
                                              score.Transposition(tr[0], tr[1]))
            
        divs = _get_divisions(e)
        if divs is not None:
            self.timeline.add_starting_object(self.position + measure_position,
                                              score.Divisions(divs))

        return divs, ts_num, ts_den

    def _do_new_page(self, measure_position):
        if 'page' in self.ongoing and self.ongoing['page'].start.t == self.position + measure_position:
            # LOGGER.debug('ignoring non-informative new-page at start of score')
            return

        if 'page' in self.ongoing:
            # end current page
            self.timeline.add_ending_object(self.position + measure_position,
                                            self.ongoing['page'])

        # start new page
        page = score.Page(self.page_nr)
        self.ongoing['page'] = page
        self.timeline.add_starting_object(self.position + measure_position,
                                          self.ongoing['page'])
        self.page_nr += 1

    def _do_new_system(self, measure_position):
        if 'system' in self.ongoing and self.ongoing['system'].start.t == self.position + measure_position:
            LOGGER.debug(
                'ignoring non-informative new-system at start of score')
            return

        if 'system' in self.ongoing:
            # end current system
            self.timeline.add_ending_object(self.position + measure_position,
                                            self.ongoing['system'])

        # start new system
        system = score.System(self.system_nr)
        self.ongoing['system'] = system
        self.timeline.add_starting_object(self.position + measure_position,
                                          self.ongoing['system'])
        self.system_nr += 1

    def _do_print(self, e, measure_position):
        if "new-page" in e.attrib:
            self._do_new_page(measure_position)
            self._do_new_system(measure_position)
        if "new-system" in e.attrib:
            self._do_new_system(measure_position)

    def _do_sound(self, e, measure_position):
        if "tempo" in e.attrib:
            self.timeline.add_starting_object(self.position + measure_position,
                                              Tempo(int(e.attrib['tempo'])))


def parse_parts(document, score_part_dict):
    """

    Parameters
    ----------
    document : lxml etree object (?)

    score_part_dict : dictionary
        a dictionary as returned by the parse_partlist() function.
    """
    # initialize a ScorePartBuilder instance for each scorepart
    part_builders = [ScorePartBuilder(score_part_dict.get(part_id, score.ScorePart(part_id)),
                                      document.xpath('/score-partwise/part[@id="{0}"]'.format(part_id))[0])
                     for part_id in document.xpath('/score-partwise/part/@id')]
    assert len(part_builders) > 0

    # a list of measures of each score part (in a list)
    measures = [part_builder.part_etree.xpath(
        'measure') for part_builder in part_builders]

    # all score parts have an equal number of measures
    assert np.all(np.diff([len(x) for x in measures]) == 0)

    # number of measures (in the first score part)
    n_measures = len(measures[0])

    for j in range(n_measures):

        position_dict = {}
        for i, part_builder in enumerate(part_builders):
            _, position_dict[i] = part_builder.add_measure(measures[i][j])
        if len(position_dict) == 0:
            break
        assert len(position_dict) == len(part_builders), ('Some score-parts' +
                                            ' have less bars than others,' +
                                                          ' don\'t know how to handle this')

        # After processing a measure, the position of all
        # score-builders should be identical, but measures are
        # sometimes incomplete (where the corresponding measures in
        # other score parts are not). To correct this
        # error, we close the measures of all score parts at the same
        # position.
        # get the current positions of all part builders
        positions = np.array([(v, k) for k, v in list(position_dict.items())])
        # print(positions)
        # print(part_builders[33].part_id)
        # take max
        max_pos = np.max(positions[:, 0])
        # close all at max_pos
        for part_builder in part_builders:
            # print('q p m', part_builder.part_id,  part_builder.quarters_per_measure)
            part_builder.close_current_measure_at(max_pos)

    for part_builder in part_builders:
        part_builder.finalize()


def parse_musicxml(fn):
    """
    Parse a MusicXML file and build a composite score ontology
    structure from it (see also scoreontology.py).

    Parameters
    ----------
    fn : str
        file name of the musicXML file to be parsed.

    Returns
    -------
    top_structure : PartGroup object OR empty list (*)
        (*): in case of failure because of xml not being of
        "score-partwise" type.
    """

    parser = etree.XMLParser(resolve_entities=False, huge_tree=False,
                             remove_comments=True, remove_blank_text=True)
    document = etree.parse(fn, parser)

    if document.getroot().tag != 'score-partwise':
        LOGGER.error('Currently only score-partwise structure is supported')
        return []
    else:
        top_structure = score.PartGroup()
        score_part_dict = {}

        try:
            partlist = document.xpath('part-list')[0]
        except IndexError:
            partlist = None

        # parse the (hierarchical) structure of score parts
        # (instruments) that are listed in the part-list element
        if partlist is not None:
            top_structure.constituents, score_part_dict = parse_partlist(
                partlist)
            for c in top_structure.constituents:
                c.parent = top_structure
        # go through each <part> (i.e. more or less instrument).
        # print(document.xpath('part[0]/measure[1]')[0].attrib)
        parse_parts(document, score_part_dict)
        return top_structure

        # return False

        # complete_measures = set()
        # for part in document.iterfind('part'):
        #     part_id = part.attrib['id']
        #     parse_score_part(part,
        #                      score_part_dict.get(part_id, ScorePart(part_id)),
        #                      complete_measures)

        # return top_structure


SCORE_DTYPES = [('pitch', 'i4'), ('onset', 'f4'), ('duration', 'f4')]


def xml_to_notearray(fn, flatten_parts=True, sort_onsets=True):
    """
    Get a note array from a MusicXML file

    Parameters
    ----------
    fn : str
        Path to a MusicXML file
    flatten_parts : bool
        If `True`, returns a single array containing all notes.
        Otherwise, returns a list of arrays for each part.

    Returns
    -------
    score : structured array or list of structured arrays
        Structured array containing the score. The fields are
        'pitch', 'onset' and 'duration'.
    """

    # Parse MusicXML
    xml_parts = parse_music_xml(fn).score_parts

    score = []
    for xml_part in xml_parts:
        # Unfold timeline to have repetitions
        xml_part.timeline = xml_part.unfold_timeline()

        # get beat map
        bm = xml_part.beat_map
        # Build score from beat map
        _score = np.array(
            [(n.midi_pitch, bm(n.start.t), bm(n.end.t) - bm(n.start.t))
             for n in xml_part.notes],
            dtype=SCORE_DTYPES)

        # Sort notes according to onset
        if sort_onsets:
            _score = _score[_score['onset'].argsort()]
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
    This goes through the <part-list> ... </part-list> in the beginning
    of the mxml file where each instrument is declarated, instruments and
    their staves are grouped (braces, brackets), etc.

    Parameters
    ----------
    partlist : list


    Returns
    -------
    structure : list

    score_part_dict : dict

    """

    structure = []
    current_group = None
    score_part_dict = {}

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
            sp = score.ScorePart(part_id, score.TimeLine())

            try:
                sp.part_name = e.xpath('part-name/text()')[0]
            except:
                pass

            try:
                sp.part_abbreviation = e.xpath('part-abbreviation/text()')[0]
            except:
                pass

            score_part_dict[part_id] = sp

            if current_group is None:
                structure.append(sp)
            else:
                current_group.constituents.append(sp)
                sp.parent = current_group

    if current_group is not None:
        LOGGER.warning(
            'part-group {0} was not ended'.format(current_group.number))
        structure.append(current_group)

    # for g in structure:
    #     print(g.pprint())

    return structure, score_part_dict


if __name__ == '__main__':
    xml_fn = sys.argv[1]
    # failing xml: op's 34_1, 63_1
    parts = parse_musicxml(xml_fn)
