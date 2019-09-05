#!/usr/bin/env python

from collections import defaultdict
from lxml import etree
import logging
import partitura.score as score
from operator import itemgetter

from partitura.musicxml import DYN_DIRECTIONS

__all__ = ['save_musicxml']

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger()

DOCTYPE = '''<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">'''
MEASURE_SEP_COMMENT = '======================================================='

def make_note_el(note, dur, counter):
    # child order
    # <grace> | <chord> | <cue>
    # <pitch>
    # <duration>
    # <tie type="stop"/>
    # <voice>
    # <type>
    # <notations>

    note_e = etree.Element('note')

    if note.id is not None:

        note_id = note.id
        # make sure note_id is unique by appending _x to the note_id for the
        # x-th repetition of the id
        counter[note_id] = counter.get(note_id, 0) + 1

        if counter[note_id] > 1:

            note_id += f'_{counter[note_id]}'

        note_e.attrib['id'] = note_id
        

    if isinstance(note, score.Note):

        if isinstance(note, score.GraceNote):
    
            if note.grace_type == 'acciaccatura':
    
                etree.SubElement(note_e, 'grace', slash='yes')
    
            else:
    
                etree.SubElement(note_e, 'grace')
    
        pitch_e = etree.SubElement(note_e, 'pitch')
    
        etree.SubElement(pitch_e, 'step').text = f'{note.step}'
    
        if note.alter is not None:
            etree.SubElement(pitch_e, 'alter').text = f'{note.alter}'
    
        etree.SubElement(pitch_e, 'octave').text = f'{note.octave}'
    
    elif isinstance(note, score.Rest):
        
        etree.SubElement(note_e, 'rest')
            
    if not isinstance(note, score.GraceNote):

        duration_e = etree.SubElement(note_e, 'duration')
        duration_e.text = f'{int(dur):d}'
    
    notations = []

    if note.tie_prev is not None:

        etree.SubElement(note_e, 'tie', type='stop')
        notations.append(etree.Element('tied', type='stop'))

    if note.tie_next is not None:

        etree.SubElement(note_e, 'tie', type='start')
        notations.append(etree.Element('tied', type='start'))

    if note.voice not in (None, 0):

        etree.SubElement(note_e, 'voice').text = f'{note.voice}'

    if note.fermata is not None:

        notations.append(etree.Element('fermata'))
    
    sym_dur = note.symbolic_duration
    
    if sym_dur.get('type') is not None:

        etree.SubElement(note_e, 'type').text = sym_dur['type']

    for i in range(sym_dur['dots']):

        etree.SubElement(note_e, 'dot')

    if (sym_dur.get('actual_notes') is not None
        and sym_dur.get('normal_notes') is not None):

        time_mod_e = etree.SubElement(note_e, 'time-modification')
        actual_e = etree.SubElement(time_mod_e, 'actual-notes')
        actual_e.text = str(sym_dur['actual_notes'])
        normal_e = etree.SubElement(time_mod_e, 'normal-notes')
        normal_e.text = str(sym_dur['normal_notes'])

    if note.staff:
        
        etree.SubElement(note_e, 'staff').text = f'{note.staff}'

    for slur in note.slur_stops:

        number = counter.get(slur, None)

        if number is None:

            number = 1
            counter[slur] = number

        else:

            del counter[slur]

        notations.append(etree.Element('slur', number=f'{number}', type='stop'))

    for slur in note.slur_starts:

        number = counter.get(slur, None)

        if number is None:

            number = 1 + sum(1 for o in counter.keys() if isinstance(o, score.Slur))
            counter[slur] = number

        else:

            del counter[slur]
            
        notations.append(etree.Element('slur', number=f'{number}', type='start'))

    if notations:

        notations_e = etree.SubElement(note_e, 'notations')
        notations_e.extend(notations)

    return note_e

            
def group_notes_by_voice(notes):
    
    if all(hasattr(n, 'voice') for n in notes):

        by_voice = defaultdict(list)

        for n in notes:

            by_voice[n.voice].append(n)

        return by_voice

    else:
        raise NotImplementedError('currently all notes must have a voice attribute for exporting to MusicXML')
    

def do_note(note, measure_end, timeline, counter):
    notes = []
    ongoing_tied = []

    if isinstance(note, score.GraceNote):

        dur_divs = 0

    else:

        dur_divs = note.end.t - note.start.t
        
    note_e = make_note_el(note, dur_divs, counter)

    return (note.start.t, dur_divs, note_e)


def linearize_measure_contents(part, start, end, counter):
    """
    Determine the document order of events starting between `start` (inclusive)
    and `end` (exlusive).  (notes, directions, divisions, time signatures). This
    function finds any mid-measure attribute/divisions and splits up the measure
    into segments by divisions, to be linearized separately and
    concatenated. The actual linearization is done by
    the `linearize_segment_contents` function.
    
    Parameters
    ----------
    start: score.TimePoint
        start
    end: score.TimePoint
        end
    part: score.Part

    Returns
    -------
    list
        The contents of measure in document order
    """

    divisions = part.timeline.get_all(score.Divisions, start=start, end=end)
    splits = [start]

    for d in divisions:

        if d.start != splits[-1]:

            splits.append(d.start)

    splits.append(end)
    contents = []

    for i in range(1, len(splits)):

        contents.extend(linearize_segment_contents(part, splits[i-1], splits[i], counter))
    
    return contents


def linearize_segment_contents(part, start, end, counter):
    """
    Determine the document order of events starting between `start` (inclusive) and `end` (exlusive).
    (notes, directions, divisions, time signatures).
    """
    notes = part.timeline.get_all(score.GenericNote,
                                  start=start, end=end,
                                  include_subclasses=True)
    notes_by_voice = group_notes_by_voice(notes)
    voices_e = defaultdict(list)
    voices = set(notes_by_voice.keys())

    for voice in sorted(voices):

        voice_notes = notes_by_voice[voice]
        # grace notes should precede other notes at the same onset
        voice_notes.sort(key=lambda n: not isinstance(n, score.GraceNote))
        voice_notes.sort(key=lambda n: n.start.t)

        for n in voice_notes:

            note_e = do_note(n, end.t, part.timeline, counter)
            voices_e[voice].append(note_e)
            
        add_chord_tags(voices_e[voice])
        
    attributes_e = do_attributes(part, start, end)
    directions_e = do_directions(part, start, end)
    prints_e = do_prints(part, start, end)
    # TODO: Page/System (i.e. everything print)
    # TODO: Tempo (i.e. everything sound)

    barline_e = do_barlines(part, start, end)
    other_e = directions_e + barline_e + prints_e

    contents = merge_measure_contents(voices_e, attributes_e, other_e, start.t)
    
    return contents

def do_prints(part, start, end):
    pages = part.timeline.get_all(score.Page, start, end)
    systems = part.timeline.get_all(score.System, start, end)
    by_onset = defaultdict(dict)
    for page in pages:
        by_onset[page.start.t]['new-page'] = 'yes'
    for system in systems:
        by_onset[system.start.t]['new-system'] = 'yes'
    result = []
    for onset, attrs in by_onset.items():
        result.append((onset, None, etree.Element('print', **attrs)))
    return result


def do_barlines(part, start, end):
    # all fermata that are not linked to a note (fermata at time end may be part
    # of the current or the next measure, depending on the location attribute
    # (which is stored in fermata.ref)).
    fermata = ([ferm for ferm in part.timeline.get_all(score.Fermata, start, end)
                if ferm.ref in (None, 'left', 'middle', 'right')] +
               [ferm for ferm in part.timeline.get_all(score.Fermata, end, end.next)
                if ferm.ref in (None, 'right')])
    repeat_start = part.timeline.get_all(score.Repeat, start, end)
    repeat_end = part.timeline.get_all_ending(score.Repeat, start.next, end.next)
    ending_start = part.timeline.get_all(score.Ending, start, end)
    ending_end = part.timeline.get_all_ending(score.Ending, start.next, end.next)
    by_onset = defaultdict(list)

    for obj in fermata:

        by_onset[obj.start.t].append(etree.Element('fermata'))

    for obj in repeat_start:

        if obj.start is not None:

            by_onset[obj.start.t].append(etree.Element('repeat',
                                                       direction='forward'))

    for obj in ending_start:

        if obj.start is not None:

            by_onset[obj.start.t].append(etree.Element('ending', type='start',
                                                       number=str(obj.number)))

    for obj in repeat_end:

        if obj.end is not None:

            by_onset[obj.end.t].append(etree.Element('repeat',
                                                     direction='backward'))

    for obj in ending_end:

        if obj.end is not None:

            by_onset[obj.end.t].append(etree.Element('ending', type='stop',
                                                     number=str(obj.number)))

    result = []

    for onset in sorted(by_onset.keys()):

        attrib = {}

        if onset == start.t:

            attrib['location'] = 'left'

        elif onset == end.t:

            attrib['location'] = 'right'

        else:

            attrib['location'] = 'middle'

        barline_e = etree.Element('barline', **attrib)

        barline_e.extend(by_onset[onset])
        result.append((onset, None, barline_e))

    return result


def add_chord_tags(notes):

    prev = None
    for onset, _, note in notes:

        if onset == prev:

            note.insert(0, etree.Element('chord'))

        if any(e.tag == 'grace' for e in note):
            # if note is a grace note we don't want to trigger a chord for the
            # next note
            prev = None

        else:

            prev = onset

def forward_backup_if_needed(t, t_prev):
    result = []
    gap = 0

    if t > t_prev:

        gap = t - t_prev
        e = etree.Element('forward')
        ee = etree.SubElement(e, 'duration')
        ee.text = f'{int(gap):d}'
        result.append((t_prev, gap, e))

    elif t < t_prev:

        gap = t_prev - t
        e = etree.Element('backup')
        ee = etree.SubElement(e, 'duration')
        ee.text = f'{int(gap):d}'
        result.append((t_prev, -gap, e))

    return result, gap


def merge_with_voice(notes, other, measure_start):
    by_onset = defaultdict(list)

    for onset, dur, el in notes:

        by_onset[onset].append((dur, el))

    for onset, dur, el in other:

        by_onset[onset].append((dur, el))

    result = []
    last_t = measure_start
    fb_cost = 0
    # order to insert simultaneously starting elements; it is important to put
    # notes last, since they update the position, and thus would lead to
    # backup/forward
    order = {'barline': 0, 'attributes': 1, 'direction':2,
             'print':3, 'sound':4, 'note': 5}
    last_note_onset = measure_start

    for onset in sorted(by_onset.keys()):

        elems = by_onset[onset]
        elems.sort(key=lambda x: order.get(x[1].tag, len(order)))

        for dur, el in elems:

            if el.tag == 'note':

                if el.find('chord') is not None:

                    last_t = last_note_onset

                last_note_onset = onset
                
            els, cost = forward_backup_if_needed(onset, last_t)
            fb_cost += cost
            result.extend(els)
            result.append((onset, dur, el))
            last_t = onset + (dur or 0)
    
    return result, fb_cost
            

def merge_measure_contents(notes, attributes, other, measure_start):
    merged = {}
    # cost (measured as the total forward/backup jumps needed to merge) all
    # elements in `other` into each voice
    cost = {} 

    
    for voice in sorted(notes.keys()):
        # merge `other` with each voice, and keep track of the cost
        merged[voice], cost[voice] = merge_with_voice(notes[voice], other, measure_start)

    # get the voice for which merging notes and other has lowest cost
    merge_voice = sorted(cost.items(), key=itemgetter(1))[0][0]

    result = []
    pos = measure_start
    for i, voice in enumerate(sorted(notes.keys())):

        if voice == merge_voice:

            elements = merged[voice]

        else:

            elements = notes[voice]

        if i == 0:

            elements, _ =  merge_with_voice(elements, attributes, measure_start)

        # backup/forward when switching voices if necessary
        if elements:

            gap = elements[0][0] - pos

            if gap < 0:

                e = etree.Element('backup')
                ee = etree.SubElement(e, 'duration')
                ee.text = f'{-int(gap):d}'
                result.append(e)

            elif gap > 0:

                e = etree.Element('forward')
                ee = etree.SubElement(e, 'duration')
                ee.text = f'{gap:d}'
                result.append(e)

        result.extend([e for _, _, e in elements])

        # update current position
        if elements:
            pos = elements[-1][0] + (elements[-1][1] or 0)

    return result
        

def do_directions(part, start, end):
    tempos = part.timeline.get_all(score.Tempo, start, end)
    directions = part.timeline.get_all(score.Direction, start, end,
                                       include_subclasses=True)
    
    result = []
    for tempo in tempos:
        # <sound tempo="qpm">
        # tempo.bpm, tempo.unit
        e0 = etree.Element('direction')
        e1 = etree.SubElement(e0, 'direction-type')
        e2 = etree.SubElement(e1, 'words')
        unit = 'q' if tempo.unit is None else tempo.unit
        e2.text = f'{unit}={tempo.bpm}'
        # e3 = etree.SubElement(e0, 'staff')
        # e3.text = "1"
        result.append((tempo.start.t, None, e0))
        e3 = etree.Element('sound', tempo=f'{int(score.to_quarter_tempo(unit, tempo.bpm))}')
        result.append((tempo.start.t, None, e3))

    for direction in directions:

        text = direction.raw_text or direction.text
        e0 = etree.Element('direction')
        e1 = etree.SubElement(e0, 'direction-type')

        if text in DYN_DIRECTIONS:

            e2 = etree.SubElement(e1, 'dynamics')
            etree.SubElement(e2, text)

        else:

            e2 = etree.SubElement(e1, 'words')
            e2.text = text
            
        if direction.end is not None:

            e3 = etree.SubElement(e0, 'direction-type')
            etree.SubElement(e3, 'dashes', type='start')


        if direction.staff is not None:

            e5 = etree.SubElement(e0, 'staff')
            e5.text = str(direction.staff)

        elem = (direction.start.t, None, e0)
        result.append(elem)

        if direction.end is not None:

            e6 = etree.Element('direction')
            e7 = etree.SubElement(e6, 'direction-type')
            etree.SubElement(e7, 'dashes', type='stop')

            elem = (direction.end.t, None, e6)
            result.append(elem)

    return result
    

def do_attributes(part, start, end):
    """
    Produce xml objects for non-note measure content 
    
    Parameters
    ----------
    others: type
        Description of `others`
    
    Returns
    -------
    type
        Description of return value
    """
    attributes = (part.timeline.get_all(score.Divisions, start, end)
                  +part.timeline.get_all(score.KeySignature, start, end)
                  +part.timeline.get_all(score.TimeSignature, start, end)
                  +part.timeline.get_all(score.Clef, start, end))

    by_start = defaultdict(list)

    for o in attributes:

        by_start[o.start.t].append(o)

    result = []

    for t in sorted(by_start.keys()):

        attr_e = etree.Element('attributes')

        for o in by_start[t]:

            if isinstance(o, score.Divisions):

                etree.SubElement(attr_e, 'divisions').text = f'{o.divs}'

            elif isinstance(o, score.KeySignature):

                ks_e = etree.SubElement(attr_e, 'key')
                etree.SubElement(ks_e, 'fifths').text = f'{o.fifths}'

                if o.mode:

                    etree.SubElement(ks_e, 'mode').text = f'{o.mode}'

            elif isinstance(o, score.TimeSignature):

                ts_e = etree.SubElement(attr_e, 'time')
                etree.SubElement(ts_e, 'beats').text = f'{o.beats}'
                etree.SubElement(ts_e, 'beat-type').text = f'{o.beat_type}'

            elif isinstance(o, score.Clef):

                clef_e = etree.SubElement(attr_e, 'clef')

                if o.number:

                    clef_e.set('number', f'{o.number}')

                etree.SubElement(clef_e, 'sign').text = f'{o.sign}'
                etree.SubElement(clef_e, 'line').text = f'{o.line}'

                if o.octave_change:

                    etree.SubElement(clef_e, 'clef-octave-change').text = f'{o.octave_change}'

        result.append((t, None, attr_e))

    return result
    

def save_musicxml(parts, out=None):
    if isinstance(parts, (score.Part, score.PartGroup)):

        parts = [parts]
        
    root = etree.Element('score-partwise')
    
    partlist_e = etree.SubElement(root, 'part-list')
    counter = {}

    for part in parts:

        scorepart_e = etree.SubElement(partlist_e, 'score-part', id=part.part_id)

        if part.part_name:

            partname_e = etree.SubElement(scorepart_e, 'part-name')
            partname_e.text = part.part_name

        part_e = etree.SubElement(root, 'part', id=part.part_id)
        # store quarter_map in a variable to avoid re-creating it for each call
        quarter_map = part.quarter_map
        beat_map = part.beat_map
        ts = part.list_all(score.TimeSignature)

        for measure in part.list_all(score.Measure):

            part_e.append(etree.Comment(MEASURE_SEP_COMMENT))
            attrib = {}

            if measure.number is not None:

                attrib['number'] = str(measure.number)

            measure_e = etree.SubElement(part_e, 'measure', **attrib)
            contents = linearize_measure_contents(part, measure.start, measure.end, counter)
            measure_e.extend(contents)
            
    if out:

        if hasattr(out, 'write'):

            out.write(etree.tostring(root.getroottree(), encoding='UTF-8', xml_declaration=True,
                                     pretty_print=True, doctype=DOCTYPE))

        else:

            with open(out, 'wb') as f:

                f.write(etree.tostring(root.getroottree(), encoding='UTF-8', xml_declaration=True,
                                       pretty_print=True, doctype=DOCTYPE))

    else:

        return etree.tostring(root.getroottree(), encoding='UTF-8', xml_declaration=True,
                              pretty_print=True, doctype=DOCTYPE)

