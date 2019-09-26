#!/usr/bin/env python

import math
from collections import defaultdict
from lxml import etree
import logging
import partitura.score as score
from operator import itemgetter, attrgetter

from partitura.importmusicxml import DYN_DIRECTIONS
from partitura.utils import partition, iter_current_next, to_quarter_tempo

__all__ = ['save_musicxml']

LOGGER = logging.getLogger(__name__)

DOCTYPE = '''<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">'''
MEASURE_SEP_COMMENT = '======================================================='

def filter_string(s):
    """
    Make (unicode) string fit for passing it to lxml, which means (at least)
    removing null characters.
    """
    return s.replace('\x00', '')

def make_note_el(note, dur, voice, counter):
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

            note_id += '_{}'.format(counter[note_id])

        note_e.attrib['id'] = filter_string(note_id)
        

    if isinstance(note, score.Note):

        if isinstance(note, score.GraceNote):
    
            if note.grace_type == 'acciaccatura':
    
                etree.SubElement(note_e, 'grace', slash='yes')
    
            else:
    
                etree.SubElement(note_e, 'grace')
    
        pitch_e = etree.SubElement(note_e, 'pitch')
    
        etree.SubElement(pitch_e, 'step').text = '{}'.format(note.step)
    
        if note.alter not in (None, 0):
            etree.SubElement(pitch_e, 'alter').text = '{}'.format(note.alter)
    
        etree.SubElement(pitch_e, 'octave').text = '{}'.format(note.octave)
    
    elif isinstance(note, score.Rest):
        
        etree.SubElement(note_e, 'rest')
            
    if not isinstance(note, score.GraceNote):

        duration_e = etree.SubElement(note_e, 'duration')
        duration_e.text = '{:d}'.format(int(dur))
    
    notations = []

    if note.tie_prev is not None:

        etree.SubElement(note_e, 'tie', type='stop')
        notations.append(etree.Element('tied', type='stop'))

    if note.tie_next is not None:

        etree.SubElement(note_e, 'tie', type='start')
        notations.append(etree.Element('tied', type='start'))

    if voice not in (None, 0):

        etree.SubElement(note_e, 'voice').text = '{}'.format(voice)

    if note.fermata is not None:

        notations.append(etree.Element('fermata'))
    
    sym_dur = note.symbolic_duration or {}
    
    if sym_dur.get('type') is not None:

        etree.SubElement(note_e, 'type').text = sym_dur['type']

    for i in range(sym_dur.get('dots', 0)):

        etree.SubElement(note_e, 'dot')

    if (sym_dur.get('actual_notes') is not None
        and sym_dur.get('normal_notes') is not None):

        time_mod_e = etree.SubElement(note_e, 'time-modification')
        actual_e = etree.SubElement(time_mod_e, 'actual-notes')
        actual_e.text = str(sym_dur['actual_notes'])
        normal_e = etree.SubElement(time_mod_e, 'normal-notes')
        normal_e.text = str(sym_dur['normal_notes'])

    if note.staff:
        
        etree.SubElement(note_e, 'staff').text = '{}'.format(note.staff)

    for slur in note.slur_stops:

        number = counter.get(slur, None)

        if number is None:

            number = 1
            counter[slur] = number

        else:

            del counter[slur]

        notations.append(etree.Element('slur', number='{}'.format(number), type='stop'))

    for slur in note.slur_starts:

        number = counter.get(slur, None)

        if number is None:

            number = 1 + sum(1 for o in counter.keys() if isinstance(o, score.Slur))
            counter[slur] = number

        else:

            del counter[slur]
            
        notations.append(etree.Element('slur', number='{}'.format(number), type='start'))

    if notations:

        notations_e = etree.SubElement(note_e, 'notations')
        notations_e.extend(notations)

    return note_e

def do_note(note, measure_end, timeline, voice, counter):
    notes = []
    ongoing_tied = []

    if isinstance(note, score.GraceNote):

        dur_divs = 0

    else:

        dur_divs = note.end.t - note.start.t
        
    note_e = make_note_el(note, dur_divs, voice, counter)

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
    splits = [start]
    q_times = part.timeline.quarter_durations(start.t, end.t)
    if len(q_times) > 0:
        quarter = start.quarter
        tp = start.next
        while tp and tp != end:
            if tp.quarter != quarter:
                splits.append(tp)
                quarter = tp.quarter
            tp = tp.next
    
    splits.append(end)
    contents = []

    for i in range(1, len(splits)):
        contents.extend(linearize_segment_contents(part, splits[i-1], splits[i], counter))

    return contents


def remove_voice_polyphony_single(notes, voice_spans):
    """
    Test wether a list of notes satisfies the MusicXML constraints on voices that:
    - all notes starting at the same time have the same duration
    - no <backup> is required to specify the voice in document order
    whenever a note violates the constraints change its voice (choosing a new voice that is not currently in use)

    Parameters
    ----------
    notes: list
        List of notes in a voice
    
    Returns
    -------
    type
        Description of return value
    """

    extraneous = defaultdict(list)

    by_onset = defaultdict(list)
    for note in notes:
        by_onset[note.start.t].append(note)
    onsets = sorted(by_onset.keys())

    for o in onsets:

        chord_dur = min(n.duration for n in by_onset[o])

        for n in by_onset[o]:

            if n.duration > chord_dur:

                voice = find_free_voice(voice_spans, n.start.t, n.end.t)
                voice_spans.append((n.start.t, n.end.t, voice))
                extraneous[voice].append(n)
                notes.remove(n)

    # now remove any notes that exceed next onset
    by_onset = defaultdict(list)
    for note in notes:
        by_onset[note.start.t].append(note)
    onsets = sorted(by_onset.keys())

    for o1, o2 in iter_current_next(onsets):

        for n in by_onset[o1]:

            if o1 + n.duration > o2:

                voice = find_free_voice(voice_spans, n.start.t, n.end.t)
                voice_spans.append((n.start.t, n.end.t, voice))
                extraneous[voice].append(n)
                notes.remove(n)

    return extraneous


def find_free_voice(voice_spans, start, end):
    free_voice = min(voice for _, _, voice in voice_spans) + 1

    for vstart, vend, voice in voice_spans:

        if ((end > vstart) and (start < vend)):

            free_voice = max(free_voice, voice+1)

    return free_voice
    
def remove_voice_polyphony(notes_by_voice):
    voice_spans = [(-math.inf, math.inf, max(notes_by_voice.keys()))]
    extraneous = defaultdict(list)
    # n_orig = sum(len(nn) for nn in notes_by_voice.values())

    for voice, vnotes in notes_by_voice.items():

        v_extr = remove_voice_polyphony_single(vnotes, voice_spans)

        for new_voice, new_vnotes in v_extr.items():
            extraneous[new_voice].extend(new_vnotes)
            
    # n_1 = sum(len(nn) for nn in notes_by_voice.values())
    # n_2 = sum(len(nn) for nn in extraneous.values())
    # n_new = n_1 + n_2
    # assert n_orig == n_new
    # assert len(set(notes_by_voice.keys()).intersection(set(extraneous.keys()))) == 0
    for v, vnotes in extraneous.items():
        notes_by_voice[v] = vnotes
        

def fill_gaps_with_rests(notes_by_voice, start, end, part):
    for voice, notes in notes_by_voice.items():
        if len(notes) == 0:
            rest = score.Rest(voice=voice or None)
            part.timeline.add(rest, start.t, end.t)
        else:
            t = start.t
            for note in notes:
                if note.start.t > t:
                    rest = score.Rest(voice=voice or None)
                    part.timeline.add(rest, t, note.start.t)
                t = note.end.t
            if note.end.t < end.t:
                rest = score.Rest(voice=voice or None)
                part.timeline.add(rest, note.end.t, end.t)
                

def linearize_segment_contents(part, start, end, counter):
    """
    Determine the document order of events starting between `start` (inclusive) and `end` (exlusive).
    (notes, directions, divisions, time signatures).
    """
    notes = part.timeline.iter_all(score.GenericNote,
                                  start=start, end=end,
                                  include_subclasses=True)

    notes_by_voice = partition(lambda n: n.voice or 0, notes)
    if len(notes_by_voice) == 0:
        # if there are no notes in this segment, we add a rest
        # NOTE: altering the part instance while exporting is bad!
        # rest = score.Rest()
        # part.add(start.t, rest, end.t)
        # notes_by_voice = {0: [rest]}
        notes_by_voice[None] = []
        
    # make sure there is no polyphony within voices by assigning any violating
    # notes to a new (free) voice.
    remove_voice_polyphony(notes_by_voice)

    # fill_gaps_with_rests(notes_by_voice, start, end, part)

    # # redo
    # notes = part.timeline.get_all(score.GenericNote,
    #                               start=start, end=end,
    #                               include_subclasses=True)
    # notes_by_voice = partition(lambda n: n.voice or 0, notes)

    voices_e = defaultdict(list)

    for voice in sorted(notes_by_voice.keys()):

        voice_notes = notes_by_voice[voice]
        # grace notes should precede other notes at the same onset
        voice_notes.sort(key=lambda n: not isinstance(n, score.GraceNote))
        # voice_notes.sort(key=lambda n: -n.duration)
        voice_notes.sort(key=lambda n: n.start.t)
        
        for n in voice_notes:
            # make rest: duration, voice, type
            note_e = do_note(n, end.t, part.timeline, voice, counter)
            voices_e[voice].append(note_e)
            
        add_chord_tags(voices_e[voice])

    attributes_e = do_attributes(part, start, end)
    directions_e = do_directions(part, start, end)
    prints_e = do_prints(part, start, end)
    barline_e = do_barlines(part, start, end)

    other_e = attributes_e + directions_e + barline_e + prints_e

    contents = merge_measure_contents(voices_e, other_e, start.t)
    
    return contents

def do_prints(part, start, end):
    pages = part.timeline.iter_all(score.Page, start, end)
    systems = part.timeline.iter_all(score.System, start, end)
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
    fermata = ([ferm for ferm in part.timeline.iter_all(score.Fermata, start, end)
                if ferm.ref in (None, 'left', 'middle', 'right')] +
               [ferm for ferm in part.timeline.iter_all(score.Fermata, end, end.next)
                if ferm.ref in (None, 'right')])
    repeat_start = part.timeline.iter_all(score.Repeat, start, end)
    repeat_end = part.timeline.iter_all(score.Repeat, start.next, end.next,
                                       mode='ending')
    ending_start = part.timeline.iter_all(score.Ending, start, end)
    ending_end = part.timeline.iter_all(score.Ending, start.next, end.next,
                                       mode='ending')
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
    prev_dur = None
    prev = None
    for onset, dur, note in notes:
        if onset == prev:
            if dur == prev_dur:
                note.insert(0, etree.Element('chord'))
                
        if any(e.tag == 'grace' for e in note):
            # if note is a grace note we don't want to trigger a chord for the
            # next note
            prev = None
        else:
            prev = onset
            prev_dur = dur


def forward_backup_if_needed(t, t_prev):
    result = []
    gap = 0

    if t > t_prev:

        gap = t - t_prev
        e = etree.Element('forward')
        ee = etree.SubElement(e, 'duration')
        ee.text = '{:d}'.format(int(gap))
        result.append((t_prev, gap, e))

    elif t < t_prev:

        gap = t_prev - t
        e = etree.Element('backup')
        ee = etree.SubElement(e, 'duration')
        ee.text = '{:d}'.format(int(gap))
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
    # needless backup/forward insertions
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
            

def merge_measure_contents(notes, other, measure_start):
    merged = {}
    # cost (measured as the total forward/backup jumps needed to merge) all
    # elements in `other` into each voice
    cost = {} 

    for voice in sorted(notes.keys()):
        # merge `other` with each voice, and keep track of the cost
        merged[voice], cost[voice] = merge_with_voice(notes[voice], other, measure_start)

    if not merged:
        merged[0] = []
        cost[0] = 0

    # get the voice for which merging notes and other has lowest cost
    merge_voice = sorted(cost.items(), key=itemgetter(1))[0][0]
    result = []
    pos = measure_start
    for i, voice in enumerate(sorted(notes.keys())):
        
        if voice == merge_voice:

            elements = merged[voice]

        else:

            elements = notes[voice]

        # backup/forward when switching voices if necessary
        if elements:

            gap = elements[0][0] - pos

            if gap < 0:

                e = etree.Element('backup')
                ee = etree.SubElement(e, 'duration')
                ee.text = '{:d}'.format(-int(gap))
                result.append(e)

            elif gap > 0:

                e = etree.Element('forward')
                ee = etree.SubElement(e, 'duration')
                ee.text = '{:d}'.format(gap)
                result.append(e)

        result.extend([e for _, _, e in elements])

        # update current position
        if elements:
            pos = elements[-1][0] + (elements[-1][1] or 0)

    return result
        

def do_directions(part, start, end):
    tempos = part.timeline.iter_all(score.Tempo, start, end)
    directions = part.timeline.iter_all(score.Direction, start, end,
                                       include_subclasses=True)
    
    result = []
    for tempo in tempos:
        # e0 = etree.Element('direction')
        # e1 = etree.SubElement(e0, 'direction-type')
        # e2 = etree.SubElement(e1, 'words')
        unit = 'q' if tempo.unit is None else tempo.unit
        # e2.text = '{}={}'.format(unit, tempo.bpm)
        # result.append((tempo.start.t, None, e0))
        e3 = etree.Element('sound', tempo='{}'.format(int(to_quarter_tempo(unit, tempo.bpm))))
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
            e2.text = filter_string(text)
            
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

    by_start = defaultdict(list)

    # for o in part.timeline.iter_all(score.Divisions, start, end):
    #     by_start[o.start.t].append(o)
    for t, quarter in part.timeline.quarter_durations(start.t, end.t):
        by_start[t].append(int(quarter))
    for o in part.timeline.iter_all(score.KeySignature, start, end):
        by_start[o.start.t].append(o)
    for o in part.timeline.iter_all(score.TimeSignature, start, end):
        by_start[o.start.t].append(o)
    for o in part.timeline.iter_all(score.Clef, start, end):
        by_start[o.start.t].append(o)
                  
    result = []

    for t in sorted(by_start.keys()):

        attr_e = etree.Element('attributes')

        for o in by_start[t]:

            if isinstance(o, int):

                etree.SubElement(attr_e, 'divisions').text = '{}'.format(o)

            elif isinstance(o, score.KeySignature):

                ks_e = etree.SubElement(attr_e, 'key')
                etree.SubElement(ks_e, 'fifths').text = '{}'.format(o.fifths)

                if o.mode:

                    etree.SubElement(ks_e, 'mode').text = '{}'.format(o.mode)

            elif isinstance(o, score.TimeSignature):

                ts_e = etree.SubElement(attr_e, 'time')
                etree.SubElement(ts_e, 'beats').text = '{}'.format(o.beats)
                etree.SubElement(ts_e, 'beat-type').text = '{}'.format(o.beat_type)

            elif isinstance(o, score.Clef):

                clef_e = etree.SubElement(attr_e, 'clef')

                if o.number:

                    clef_e.set('number', '{}'.format(o.number))

                etree.SubElement(clef_e, 'sign').text = '{}'.format(o.sign)
                etree.SubElement(clef_e, 'line').text = '{}'.format(o.line)

                if o.octave_change:

                    etree.SubElement(clef_e, 'clef-octave-change').text = '{}'.format(o.octave_change)

        result.append((t, None, attr_e))

    return result
    

def save_musicxml(parts, out=None):
        
    root = etree.Element('score-partwise')
    
    partlist_e = etree.SubElement(root, 'part-list')
    counter = {}

    group_stack = []

    def close_group_stack():
        while group_stack:
            # close group
            etree.SubElement(partlist_e, 'part-group',
                             number='{}'.format(group_stack[-1].number),
                             type='stop')
            # remove from stack
            group_stack.pop()
        
    def handle_parents(part):
        # 1. get deepest parent that is in group_stack (keep track of parents to
        # add)
        pg = part.parent
        to_add = []
        while pg:
            if pg in group_stack:
                break
            to_add.append(pg)
            pg = pg.parent

        
        # close groups while not equal to pg
        while group_stack:
            if pg == group_stack[-1]:
                break
            else:
                # close group
                etree.SubElement(partlist_e, 'part-group',
                                 number='{}'.format(group_stack[-1].number),
                                 type='stop')
                # remove from stack
                group_stack.pop()

        # start all parents in to_add
        for pg in reversed(to_add):
            # start group
            pg_e = etree.SubElement(partlist_e, 'part-group',
                                    number='{}'.format(pg.number),
                                    type='start')
            if pg.group_symbol is not None:
                symb_e = etree.SubElement(pg_e, 'group-symbol')
                symb_e.text = pg.group_symbol
            if pg.group_name is not None:
                name_e = etree.SubElement(pg_e, 'group-name')
                name_e.text = pg.group_name

            group_stack.append(pg)


    for part in score.iter_parts(parts):

        handle_parents(part)
        
        # handle part list entry
        scorepart_e = etree.SubElement(partlist_e, 'score-part', id=part.id)

        partname_e = etree.SubElement(scorepart_e, 'part-name')
        if part.part_name:
            partname_e.text = filter_string(part.part_name)

        if part.part_abbreviation:
            partabbrev_e = etree.SubElement(scorepart_e, 'part-abbreviation')
            partabbrev_e.text = filter_string(part.part_abbreviation)


        # write the part itself
        
        part_e = etree.SubElement(root, 'part', id=part.id)
        # store quarter_map in a variable to avoid re-creating it for each call
        quarter_map = part.quarter_map
        beat_map = part.beat_map
        # ts = part.timeline.get_all(score.TimeSignature)

        for measure in part.timeline.iter_all(score.Measure):

            part_e.append(etree.Comment(MEASURE_SEP_COMMENT))
            attrib = {}

            if measure.number is not None:

                attrib['number'] = str(measure.number)

            measure_e = etree.SubElement(part_e, 'measure', **attrib)
            contents = linearize_measure_contents(part, measure.start, measure.end, counter)
            measure_e.extend(contents)
            
    close_group_stack()

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
