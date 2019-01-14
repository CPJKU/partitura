#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import numpy as np
from lxml import etree
from collections import Counter, defaultdict, OrderedDict
from operator import itemgetter
import logging

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

CHILD_ORDER = {'//pitch': ['step', 'alter', 'octave'],
               '//score-partwise': ['work', 'movement-number', 'movement-title',
	                          'identification', 'defaults', 'credit',
                                  'part-list', 'part'],
               '//work': ['work-number', 'work-title', 'opus'],
               '//direction': ['direction-type', 'offset', 'footnote', 'level',
                               'voice', 'staff', 'sound'],
               '//barline': ['bar-style', 'footnote', 'level', 'wavy-line',
                             'segno', 'coda', 'fermata', 'ending', 'repeat'],
               '//identification': ['creator', 'rights', 'encoding', 'source',
                                    'relation, miscellaneous'],
               '//attributes': ['divisions', 'key', 'time', 'staves', 'part-symbol',
                                'instruments', 'clef', 'staff-details', 'transpose',
                                'directive', 'measure-style'],
               # TODO: add note child order (tricky, because can be different,
               # depending presence/absence of children)
               }

# min max numbers allowed in number attribute of slurs, wedges, dashes, etc
MIN_RANGE = 1
MAX_RANGE = 6

class Colors(object):
    """Escape codes for color in logging
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_elements_by_position(measures, target):
    """
    Return a dictionary indexed by position (in `divisions`), where each
    position key returns a list of elements matching xpath expression `target`
    at that position in the MusicXML file.  `measures`.
    """
    pos = 0
    result = defaultdict(list)
    for m in measures:
        for e in m:
            result[pos].extend(e.xpath(target))
            dur = 0
            if len(e.xpath('./chord[1]')) == 0:
                try:
                    dur = int(e.xpath('./duration[1]/text()')[0])
                except IndexError:
                    pass
                if e.tag == 'backup':
                    dur = -dur
            pos += dur
    for k, v in list(result.items()):
        if len(v) == 0:
            del result[k]
    return dict(result)
    

def fix_order(doc, xpath, order=None):
    """
    Fix the order of children in elements returned by `doc`.xpath(`xpath`). The
    order of the children is given by a dictionary order with element tags as
    keys, and a sorting number as value. Elements whose tags are not in `order`
    are placed at the end in the order of original occurrence.
    """
    if order is None:
        order = CHILD_ORDER.get(xpath, {})

    if not isinstance(order, dict):
        order = dict(list(zip(order, list(range(len(order))))))

    elements = doc.xpath(xpath)
    for e in elements:
        e[:] = sorted(e, key=lambda x: order.get(x.tag, len(order)))


def fix_direction(doc):
    """
    Directions need a direction-type element. This function looks for directions
    that have no direction-type and if they have a sound element, tries to infer
    a direction-type from the attributes of the sound element. If no
    direction-type can be inferred, an Exception is raised.
    """
    directions = doc.xpath('//direction')
    to_dt = {'dacapo': dict(type='words', text='da capo'),
             'fine': dict(type='words', text='fine'),
             'segno': dict(type='segno'),
             'coda': dict(type='coda')
             }
    for d in directions:
        childtags = [e.tag for e in d]
        has_dt = 'direction-type' in childtags
        try:
            sound_idx = childtags.index('sound')
        except ValueError:
            continue
        if not has_dt:
            dts = set.intersection(set(to_dt.keys()),
                                   set(d[sound_idx].attrib.keys()))
            if len(dts) > 0:
                for w in dts:
                    dt = etree.Element('direction-type')
                    e = etree.Element(to_dt[w]['type'])
                    if 'text' in to_dt[w]:
                        e.text = to_dt[w]['text']
                    dt.append(e)
                    d.append(dt)
            else:
                raise Exception("Do not know how to deal with direction without direction-type (children: {})"
                                .format(childtags))
            #     chtagset = set(childtags).difference(set(('offset', 'staff', 'sound'
            #                                               'footnote', 'level', 'voice')))
            #     if len(chtagset)
            #     s = d[sound_idx]
            #     d.remove(s)
            #     p = d.getparent()
            #     p.insert(p.index(d), s)
            #     p.remove(d)


def get_position_info(e, return_dict=False):
    """
    Build a string that provides position information for the element, only
    intended for manual inspection.
    """
    # TODO, don't assume e is inside a measure
    m = [x for x in e.iterancestors() if x.tag == 'measure'][0]
    note = [x for x in e.iterancestors() if x.tag == 'note']
    if len(note) > 0:
        noteinfo = ' (note id: {}; left/top: {:.0f} / {:.0f})'.format(
            note[0].get('ID'),
            float(note[0].xpath('coordinates/pixel/left')[0].text),
            float(note[0].xpath('coordinates/pixel/top')[0].text))
    else:
        noteinfo = ''
    if return_dict:
        return dict(measure=m.get('number'), page=m.get('page'), system=m.get('system'))
    else:
        return ('measure number {}, page {}, system {}{}'
                .format(m.get('number'), m.get('page'), m.get('system'),
                        noteinfo))


def try_compress_numbers_to_range(elements):
    """
    Map the "number" attribute of any element in `elements` to the most compact
    range possible (starting from 1). If the resulting numbers are within
    [MIN_RANGE, MAX_RANGE], return True, otherwise, return False. If it is not
    possible to obtain a mapping within [MIN_RANGE, MAX_RANGE], the number
    attributes not modified.
    """

    numbers = set(int(e.get('number')) for e in elements)

    if len(numbers) <= ((MAX_RANGE - MIN_RANGE) + 1):
        actual_nrs = sorted(numbers)
        ideal_nrs = list(range(MIN_RANGE, MIN_RANGE + len(numbers)))

        if np.any(np.array(actual_nrs) != np.array(ideal_nrs)):
            nr_map = dict(list(zip(actual_nrs, ideal_nrs)))
            LOGGER.debug('compressing number range {}'
                         .format(', '.join('{} → {}'.format(k, v) for k, v in list(nr_map.items()))))
            for e in elements:
                old_nr = int(e.get('number'))
                new_nr = nr_map[old_nr]
                e.set('number', str(new_nr))
        all_within_range = True
    else:
        all_within_range = False
    return all_within_range


def get_first_non_blacklisted(blacklist):
    """Return the first integer not in `blacklist`.
    """
    i = 1
    while i in blacklist:
        i += 1
    return i

def get_note_ancestor(e):
    """
    Return the first ancestor of `e` with tag "note", or None if there is no such
    ancestor.
    """
    for a in e.iterancestors():
        if a.tag == 'note':
            return a
    return None

# def get_pitch(n):
#     """
#     """
#     pitch = n.xpath('pitch')[0]
#     return '{} {} {}'.format(''.join(pitch.xpath('step/text()')),
#                              ''.join(pitch.xpath('alter/text()')),
#                              ''.join(pitch.xpath('octave/text()')))

def remove_redundant_ranges(ebp, pbe):
    matches, false_starts, false_stops = match_start_stop_elements(ebp)
    be_note_pairs = defaultdict(list)
    # be_pos_pairs = defaultdict(list)
    for e_start, e_end in list(matches.items()):
        n_start = get_note_ancestor(e_start)
        n_end = get_note_ancestor(e_end)
        # if n_start and n_end and get_pitch(n_start) == get_pitch(n_end):
        #     print('tie?')
        # be_pos_pairs[(pbe[e_start], pbe[e_end])].append(e_start)
        be_note_pairs[(n_start, n_end)].append(e_start)
    if (None, None) in be_note_pairs:
        del be_note_pairs[(None, None)]
    for ee in list(be_note_pairs.values()):
        if len(ee) > 1:
            LOGGER.debug('removing {} redundant items'.format(len(ee) -1))
        for e_start in ee[1:]:
            e_start.getparent().remove(e_start)
            e_end = matches[e_start]
            e_end.getparent().remove(e_end)

def fix_start_stop_numbers(doc, xpath):
    """
    Change the "number" attributes of elements matching `xpath` to lie within
    the interval [MIN_RANGE, MAX_RANGE]. As opposed to the function
    `try_compress_numbers_to_range` this function keeps track of which numbered
    ranges are active, and reuses numbers if there are no ongoing ranges with
    that number. It does not use numbers occurring in erroneous ranges (ranges
    for which there are unmatched start or stop elements).
    """
    measures = doc.xpath('.//measure')
    ebp = get_elements_by_position(measures, xpath)
    pbe = dict((e, p) for p, ee in list(ebp.items()) for e in ee)

    LOGGER.debug('mismatches before corrections:')
    ebp = get_elements_by_position(measures, xpath)
    pbe = dict((e, p) for p, ee in list(ebp.items()) for e in ee)
    check_mismatches(ebp)

    # fn = '/tmp/{}_before.txt'.format(re.sub('\W', '', xpath))
    # plot_ranges(ebp, fn)

    remove_redundant_ranges(ebp, pbe)
    # ebp = get_elements_by_position(measures, xpath)
    # pbe = dict((e, p) for p, ee in ebp.items() for e in ee)

    all_within_range = try_compress_numbers_to_range([e for ee in list(ebp.values())
                                                      for e in ee])

    if all_within_range:
        LOGGER.debug('{}all within range{}'.format(Colors.OKGREEN, Colors.ENDC))
        return True
    else:
        LOGGER.debug('not all within range')
    
    matches, false_starts, false_stops = match_start_stop_elements(ebp)
    inv_matches = dict((v, k) for k, v in list(matches.items()))
    # print('false st/st', len(false_starts), len(false_stops))
    blacklist = set([int(e.get('number', -1)) for e in
                     false_starts + false_stops
                     if MIN_RANGE <= int(e.get('number', -1)) <= MAX_RANGE])

    LOGGER.debug('blacklist: {}'.format(blacklist))
    active = defaultdict(set)
    active_at_pos = {}
    positions = np.array(sorted(ebp.keys()))
    for p_np in positions:
        p = int(p_np)
        ee = sorted(ebp[p], key=lambda x: x.get('type'), reverse=True)
        for e in ee:
            n = int(e.get('number', -1))
            t = e.get('type')
            if t in ('start', 'crescendo', 'diminuendo'):
                active[n].add(e)
            elif t == 'stop':
                try:
                    active[n].remove(inv_matches[e])
                except KeyError:
                    pass
            else:
                raise Exception()
        active_at_pos[p] = set(x for y in list(active.values()) for x in y)
    
    # for p in positions:
    #     active = active_at_pos[int(p)]
    #     if len(active) > 0:
    #         print('{}: {}'.format(int(p), [int(e.get('number')) for e in active]))

    for b, e in list(matches.items()):
        n = int(b.get('number'))
        if n in blacklist:
            continue
        start = pbe[b]
        end = pbe[e]
        span = positions[np.logical_and(positions >= start, positions < end)]
        concurrent = set(int(x.get('number', -1)) for p in span
                         for x in active_at_pos[int(p)].difference(set((b,))))
        blacklist_for_range = blacklist.union(concurrent)
        new_n = get_first_non_blacklisted(blacklist_for_range)
        if not MIN_RANGE <= new_n <= MAX_RANGE:
            #LOGGER.warning('Cannot renumber start-stop into valid range',start, end, n, new_n, concurrent)
            LOGGER.warning('Cannot renumber start-stop into valid range (orig. nr: {}; new nr: {}; co-occuring numbers: {})'.format(n, new_n, concurrent))
            LOGGER.warning('Position: ' + get_position_info(b))
        else:
            if n != new_n:
                LOGGER.debug('renumbering {} → {} (blacklisted: {})'.format(n, new_n, list(blacklist_for_range)))
                # print(start, end, n, '->', new_n, blacklist_for_range)
                b.set('number', str(new_n))
                e.set('number', str(new_n))
        
    LOGGER.debug('mismatches after corrections:')
    ebp = get_elements_by_position(measures, xpath)
    check_mismatches(ebp)

    fn = '/tmp/{}_after.txt'.format(re.sub('\W', '', xpath))
    plot_ranges(ebp, fn)

def check_mismatches(ebp):
    """
    Check if there are any mis-matched range start/stop elements. Only for
    informational purposes.
    """

    matches, false_starts, false_stops = match_start_stop_elements(ebp)
    pbe = dict((e, p) for p, ee in list(ebp.items()) for e in ee)
    false_start_by_nr = Counter()
    false_stop_by_nr = Counter()
    for e in false_starts:
        false_start_by_nr.update((int(e.get('number', -1)),))
    for e in false_stops:
        false_stop_by_nr.update((int(e.get('number', -1)),))
    nrs = set.union(set(false_start_by_nr.keys()),
                    set(false_stop_by_nr.keys()))
    
    for n in nrs:
        LOGGER.debug('{}number {} has {} false starts and {} false stops{}'.format(
            Colors.FAIL, n, false_start_by_nr[n],
            false_stop_by_nr[n], Colors.ENDC).encode('utf8'))

    # counter = defaultdict(lambda: defaultdict(lambda: 0))
    # for p in sorted(ebp.keys()):
    #     # sort elements by reverse lexical order of attribute type, to ensure
    #     # that "stop" elements are handled before "start" elements (nevermind
    #     # "diminuendo", "crescendo")
    #     for e in sorted(ebp[p], key=lambda x: x.get('type'), reverse=True):
    #         n = int(e.get('number', -1))
    #         t = e.get('type')
    #         counter[n][t] += 1

    # for k, ss in counter.items():
    #     # print(k,ss['start'], ss['stop'])
    #     # if ss['start'] != ss['stop']:
    #     nstart = sum(ss[t] for t in ('start', 'diminuendo', 'crescendo'))
    #     if not nstart == ss['stop']:
    #         level = Colors.FAIL
    #     else:
    #         level = Colors.OKGREEN
    #     print(u'{}number {} has {} starts and {} stops{}'.format(
    #         level, k, nstart, ss['stop'], Colors.ENDC).encode('utf8'))

# def plot_ss(ebp, outfile, false_starts=[], false_stops=[]):
#     symb = dict(start='<', stop='>')
#     with open(outfile, 'w') as f:
#         for p in sorted(ebp.keys()):
#             for e in ebp[p]:
#                 n = int(e.get('number', -1))
#                 t = e.get('type')
#                 s = symb[t]
#                 f.write('{} {} {}\n'.format(p, n, s))

# def plot_ranges(ebp, outfile):
#     symb = dict(start='<',
#                 crescendo='<',
#                 diminuendo='<',
#                 stop='>',
#                 false_start='X',
#                 false_stop='Y'
#     )
#     matches, false_starts, false_stops = match_start_stop_elements(ebp)
#     # print('false starts', len(false_starts))
#     # print('false stops', len(false_stops))
#     pbe = dict((e, p) for p, ee in ebp.items() for e in ee)
#     data = []
#     active = defaultdict(list)
#     n_active = {}
#     for p in sorted(ebp.keys()):
#         for e in  sorted(ebp[p], key=lambda x: x.get('type'), reverse=True):
#             n = int(e.get('number', -1))
#             t = e.get('type')
#             if t in ('start', 'crescendo', 'diminuendo'):
#                 active[n].append(e)
#             elif t == 'stop':
#                 try:
#                     active[n].pop()
#                 except:
#                     pass
#             else:
#                 raise Exception()
            
#         n_active[p] = sum(len(x) for x in active.values())

#     for s, e in matches.items():
#         data.append((pbe[s], int(s.get('number')), symb[s.get('type')], n_active[pbe[s]]))
#         data.append((pbe[e], int(e.get('number')), symb[e.get('type')], n_active[pbe[e]]))
#     for s in false_starts:
#         # print(pbe[s], get_position_info(s))
#         data.append((pbe[s], int(s.get('number')), symb['false_start'], n_active[pbe[s]]))
#     for s in false_stops:
#         data.append((pbe[s], int(s.get('number')), symb['false_stop'], n_active[pbe[s]]))
#     data.sort(key=itemgetter(0))
#     with open(outfile, 'w') as f:
#         for row in data:
#             f.write('{} {} {} {}\n'.format(*row))

    
def match_start_stop_elements(ebp):
    """
    Return a list of matching start/stop elements, as well as a list of
    non-matched start and non-matched stop elements, occurring in the values of
    dictionary `ebp` (as returned by `get_elements_by_position`).
    """
    matches = {}
    started = defaultdict(list)
    false_starts = []
    false_stops = []
    for p in sorted(ebp.keys()):
        # sort elements by reverse lexical order of attribute type, to ensure
        # that "stop" elements are handled before "start" elements (nevermind
        # "diminuendo", "crescendo")

        for e in sorted(ebp[p], key=lambda x: x.get('type'), reverse=True):
            n = int(e.get('number', -1))
            t = e.get('type')
            if t in ('start', 'crescendo', 'diminuendo'):
                if len(started[n]) > 0:
                    # print('false start')
                    false_starts.append(e)
                else:
                    started[n].append(e)
            elif t == 'stop':
                if len(started[n]) == 0:
                    # print('false stop')
                    false_stops.append(e)
                else:
                    es = started[n].pop()
                    matches[es] = e
            else:
                raise
    for ss in list(started.values()):
        false_starts.extend(ss)
    return matches, false_starts, false_stops
