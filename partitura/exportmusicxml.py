#!/usr/bin/env python

from collections import defaultdict
from lxml import etree
import logging
import partitura.score as score
from operator import itemgetter

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
        note_e.attrib['id'] = note.id

    if isinstance(note, score.Note):
        # if not note.grace_type:
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
            
    # if not note.grace_type:
    if not isinstance(note, score.GraceNote):

        duration_e = etree.SubElement(note_e, 'duration')
        duration_e.text = f'{int(dur):d}'
    

    notations = []

    if note.tie_prev:
        etree.SubElement(note_e, 'tie', type='stop')
        notations.append(etree.Element('tied', type='stop'))

    if note.tie_next:
        etree.SubElement(note_e, 'tie', type='start')
        notations.append(etree.Element('tied', type='start'))

    if note.voice:
        etree.SubElement(note_e, 'voice').text = f'{note.voice}'

    sym_dur = note.symbolic_duration
    
    etree.SubElement(note_e, 'type').text = sym_dur['type']

    # TODO: add actual/normal notes

    for i in range(sym_dur['dots']):

        etree.SubElement(note_e, 'dot')

    if note.staff:
        
        etree.SubElement(note_e, 'staff').text = f'{note.staff}'

    for slur in note.slur_stops:
        number = counter.get(slur, None)
        if number is None:
            number = 1
            counter[slur] = number
            pass # LOGGER.warning('unmatched slur')
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

    # make_note_el
    if isinstance(note, score.GraceNote):

        dur_divs = 0

    else:

        divs = next(iter(note.start.get_prev_of_type(score.Divisions, eq=True)),
                    score.Divisions(1)).divs
        dur_divs = divs * score._LABEL_DURS[note.symbolic_duration['type']]
        dur_divs *= score._DOT_MULTIPLIERS[note.symbolic_duration['dots']]

    note_e = make_note_el(note, dur_divs, counter)

    return (note.start.t, dur_divs, note_e)


def linearize_measure_contents(part, start, end, counter):
    """
    Determine the document order of events starting between `start` (inclusive) and `end` (exlusive).
    (notes, directions, divisions, time signatures)
    
    Parameters
    ----------
    start: score.TimePoint
        start
    end: score.TimePoint
        end
    part: score.Part

    Returns
    -------
    type
        Description of return value
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
    notes = part.timeline.get_all(score.GenericNote, start=start, end=end, include_subclasses=True)

    notes_by_voice = group_notes_by_voice(notes)

    voices_e = defaultdict(list)

    voices = set(notes_by_voice.keys())

    for voice in sorted(voices):

        voice_notes = notes_by_voice[voice]

        # grace notes should precede other notes at the same onset
        voice_notes.sort(key=lambda n: not isinstance(n, score.GraceNote))
        # vnotes.sort(key=lambda n: n.duration, reverse)
        voice_notes.sort(key=lambda n: n.start.t)

        for n in voice_notes:

            note_e = do_note(n, end.t, part.timeline, counter)
            voices_e[voice].append(note_e)
            
        add_chord_tags(voices_e[voice])
        
    attributes = (part.timeline.get_all(score.Divisions, start, end)
                  +part.timeline.get_all(score.KeySignature, start, end)
                  +part.timeline.get_all(score.TimeSignature, start, end)
                  +part.timeline.get_all(score.Clef, start, end))
    
    attributes_e = do_attributes(attributes)
    
    directions = part.timeline.get_all(score.Direction, start, end,
                                       include_subclasses=True)
    # TODO: deal with directions
    directions_e = []
    
    # merge other and contents
    contents = merge_measure_contents(voices_e, attributes_e, directions_e, start.t)
    
    # print(other)
    return contents # , save_for_next_measure


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
        result.append((t, None, e))
    elif t < t_prev:
        gap = t_prev - t
        e = etree.Element('backup')
        ee = etree.SubElement(e, 'duration')
        ee.text = f'{int(gap):d}'
        result.append((t, None, e))
    return result, gap

    
def merge_with_voice(notes, other, measure_start):
    other = iter(other)
    notes = iter(notes)
    result = []
    end_token = (None, None, None)
    n_t, n_dur, n = next(notes, end_token)
    o_t, o_dur, o = next(other, end_token)
    last_t = measure_start
    fb_cost = 0
    while o is not None or n is not None:
        if o is None or n_t < o_t:
            # if n has a chord tag it can be assumed to start at the same time
            # as the previous note, so we don't need to forward or backup
            if n[0].tag != 'chord':
                els, cost = forward_backup_if_needed(n_t, last_t)
                fb_cost += cost
                result.extend(els)
            result.append((n_t, n_dur, n))
            last_t = n_t + (n_dur or 0)
            n_t, n_dur, n = next(notes, end_token)
        # elif n is None or o_t >= n_t:
        else:
            els, cost = forward_backup_if_needed(o_t, last_t)
            fb_cost += cost
            result.extend(els)
            result.append((o_t, o_dur, o))
            last_t = o_t
            o_t, _, o = next(other, end_token)

        # elif n_t < o_t:
        #     # we have n and o, and n comes first
        #     # if n has a chord tag it can be assumed to start at the same time
        #     # as the previous note, so we don't need to forward or backup
        #     if n[0].tag != 'chord':
        #         els, cost = forward_backup_if_needed(n_t, last_t)
        #         fb_cost += cost
        #         result.extend(els)
        #     result.append((n_t, n_dur, n))
        #     last_t = n_t + (n_dur or 0)
        #     n_t, n_dur, n = next(notes, end_token)
        # else: 
        #     # we have n and o, and o comes first
        #     els, cost = forward_backup_if_needed(o_t, last_t)
        #     fb_cost += cost
        #     result.extend(els)
        #     result.append((o_t, o_dur, o))
        #     last_t = o_t
        #     o_t, _, o = next(other, end_token)

    return result, fb_cost
    
def merge_measure_contents(notes, attributes, other, measure_start):
    merged = {}
    # cost (measured as the total forward/backup jumps needed to merge) all elements in `other` into each voice
    cost = {} 
    assert 1 in notes
    
    for voice in sorted(notes.keys()):
        # merge `other` with each voice, and keep track of the cost
        merged[voice], cost[voice] = merge_with_voice(notes[voice], other, measure_start)

    merge_voice = sorted(cost.items(), key=itemgetter(1))[0][0]

    result = []
    pos = measure_start
    for voice in sorted(notes.keys()):
        if voice == merge_voice:
            elements = merged[voice]
        else:
            elements = notes[voice]
        if voice == 1:
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

    # attributes take effect in score order, not document order, so we add them to the first voice.
    return result
        

def do_attributes(attributes):
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

    by_start = defaultdict(list)
    for o in attributes:
        by_start[o.start.t].append(o)

    # attribute_classes = (score.Divisions, score.KeySignature, score.TimeSignature)
    result = []
    for t in sorted(by_start.keys()):
        # attributes = [o for o in by_start[t] if isinstance(o, attribute_classes)]
        # if attributes:
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

        # directions = [o for o in by_start[t] if isinstance(o, score.Direction)]
        # TODO: handle directions
    return result


# def to_musicxml(part, out=None):
#     root = etree.Element('score-partwise')
    
#     partlist_e = etree.SubElement(root, 'part-list')
#     scorepart_e = etree.SubElement(partlist_e, 'score-part', id=part.part_id)
#     partname_e = etree.SubElement(scorepart_e, 'part-name')
#     if part.part_name:
#         partname_e.text = part.part_name
#     part_e = etree.SubElement(root, 'part', id=part.part_id)
#     # store quarter_map in a variable to avoid re-creating it for each call
#     quarter_map = part.quarter_map
#     beat_map = part.beat_map
#     ts = part.list_all(score.TimeSignature)
#     for measure in part.list_all(score.Measure):
#         part_e.append(etree.Comment(MEASURE_SEP_COMMENT))
#         measure_e = etree.SubElement(part_e, 'measure', number='{}'.format(measure.number))
#         # contents, saved_from_prev = linearize_measure_contents(measure, part)
#         contents = linearize_measure_contents(measure, part)
#         measure_e.extend(contents)
        
#     if out:
#         if hasattr(out, 'write'):
#             out.write(etree.tostring(root.getroottree(), encoding='UTF-8', xml_declaration=True,
#                                      pretty_print=True, doctype=DOCTYPE))
#         else:
#             with open(out, 'wb') as f:
#                 f.write(etree.tostring(root.getroottree(), encoding='UTF-8', xml_declaration=True,
#                                        pretty_print=True, doctype=DOCTYPE))
#     else:
#         # print(etree.tostring(root.getroottree(), encoding='unicode', pretty_print=True, doctype=DOCTYPE))
#         return etree.tostring(root.getroottree(), encoding='UTF-8', xml_declaration=True,
#                               pretty_print=True, doctype=DOCTYPE)
def to_musicxml(parts, out=None):
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
            measure_e = etree.SubElement(part_e, 'measure', number='{}'.format(measure.number))
            # contents, saved_from_prev = linearize_measure_contents(measure, part)
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
        # print(etree.tostring(root.getroottree(), encoding='unicode', pretty_print=True, doctype=DOCTYPE))
        return etree.tostring(root.getroottree(), encoding='UTF-8', xml_declaration=True,
                              pretty_print=True, doctype=DOCTYPE)

