#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

The class `Instrument` represents an instrument, to be associated to a
unit of musical information, such as a MIDI track, or a staff in a
musical score. The class has both a name and a canonical name. Its
name is the textual representation of the instrument that originates
from the data. In case of reading a MIDI file, the name of the
instrument may be read from a InstrumentName or TrackName meta event
in the MIDI track. In case of analysing a musical score, it is
typically the instrument name as it is written in front of the staff
(note that in case of automatic OMR, the name may contain OCR
errors). In other words, the name of an `Instrument` is a free form
string, it may be an the instrument name in any language, or an
abreviated name.

The canonical name on the other hand is the English name for the
instrument, and is not free form, but rather one of a list of known
instrument names.

The mapping of free form instrument names to canonical instrument
names is done by the class `InstrumentMapper`. This mapper contains a
list of InstrumentName instances (`instruments`), whose purpose it is
to quantify how similar an arbitrary free form string is to the
canonical instrument name it represents. It does so by assigning a
distance value to a limited number of (user defined) normalized
strings (normalized means converted to lower case, and white-space
characters removed). If the normalized free form string starts with
any of the defined strings, the associated distance value is returned,
other wise it returns a distance of infinity, that represents no match
is possible. When an Instrument class is instantiated with a freeform
string, the Instrument class uses an InstrumentMapper instance to
guess the most likely canonical instrument name for that freeform
string.

The `InstrumentMapper` also defines a fixed mapping from canonical
instrument names to MIDI channels and patches, so that musical
information associated to an instrument can exported to MIDI without
manual intervention to set the correct MIDI patch for the
instrument. This is done through a dictionary `channel_patch_pairs`,
that maps canonical instrument names to a pair of MIDI channel, and
MIDI patch.

Finally, the `Instrument` class defines a member `rank`. The rank of an
instrument is a number that may be associated to the instrument, in
case there are multiple instruments of the same class. If during OMR a
staff with the name "Violin I,II" is encountered, two Instrument
instances will be created, both with canonical name "Violin", but one
with rank 1, and the other with rank 2.

"""


import numpy as np
import re
import logging

LOGGER = logging.getLogger(__name__)


# Note: the canonical names given here as strings have to be inserted
# into the ScoreDB's "Instrument" table exactly as written here!
PIANO = 'Piano'
HARPSICHORD = 'Harpsichord'  # ger.: Cembalo
CELESTA = 'Celesta'
ORGAN = 'Organ'
FLUTE = 'Flute'
PICCOLO = 'Piccolo'
OBOE = 'Oboe'
ENGLISHHORN = 'English Horn'
HECKELPHONE = 'Heckelphone'  # ger.: Heckelphon
CLARINET = 'Clarinet'
BASSCLARINET = 'Bass Clarinet'
BASSOON = 'Bassoon'
CONTRABASSOON = 'Contrabassoon'
SOPRANOSAXOPHONE = 'Soprano Saxophone'
ALTOSAXOPHONE = 'Alto Saxophone'
TENORSAXOPHONE = 'Tenor Saxophone'
BARITONESAXOPHONE = 'Baritone Saxophone'
FRENCHHORN = 'French Horn'
WAGNER_TUBA = 'Wagner Tuba'  # Wagner Tuba, special kind of horn. It is NOT a tenor horn! # TO DO: put in Instrument table if necessary
TRUMPET = 'Trumpet'
CORNET = 'Cornet'
TROMBONE = 'Trombone'
ALTOTROMBONE = 'Alto Trombone'
TENORTROMBONE = 'Tenor Trombone'
BASSTROMBONE = 'Bass Trombone'
TUBA = 'Tuba'  # this is actually the basstuba
CONTRABASSTUBA = 'Contrabass Tuba'  # TO DO: put in Instrument table if necessary
OPHICLEIDE = 'Ophicleide'  # predecessor to the Tuba
GLOCKENSPIEL = 'Glockenspiel'
TIMPANI = 'Timpani'
HARP = 'Harp'
VIOLIN = 'Violin'
VIOLA = 'Viola'
CELLO = 'Cello'
CONTRABASS = 'Contrabass'
SLEIGHBELLS = 'Sleigh Bells'  # aka. jingle bells; ger.: Schellen
TAMTAM = 'Tam-Tam'
CYMBALS = 'Cymbals'  # ger.: Becken
TRIANGLE = 'Triangle'
BASSDRUM = 'Bass Drum'    # ger.: große Trommel
SNAREDRUM = 'Snare Drum'  # kleine Trommel
SOPRANO_VOICE = 'Soprano Voice'
MEZZOSOPRANO_VOICE = 'Mezzosoprano Voice'
ALTO_VOICE = 'Alto Voice'
CONTRAALTO_VOICE = 'Contraalto Voice'  # rare?!
TENOR_VOICE = 'Tenor Voice'
BARITONE_VOICE = 'Baritone Voice'
BASS_VOICE = 'Bass Voice'


NON_ALPHA_NUM_PAT = re.compile(r'\W', re.UNICODE)
# pattern that matches any leading numbers from instrument names (such
# as in '2 Violoncelli).
LEADING_NUMBERS_PAT = re.compile(r'^([0-9\s]*)', re.UNICODE)


def normalize_string(s):
    """

    takes a string and returns it as an all lowercase string?

    Parameters
    ----------
    s : str
        the string to be normalized.

    Returns
    -------
    str
        the normalized string, i.e. all lowercase letters?
    """
    # return s.lower().replace(' ', '')
    return LEADING_NUMBERS_PAT.sub(r'', NON_ALPHA_NUM_PAT.sub(r'', s.lower()))


class InstrumentNameMapper(object):
    """
    """

    class InstrumentName(object):
        """
        This internal class is used to link "canonical" (english)
        instrument names to versions of the instrument names in other
        languages, and also to degraded instrument names, such as
        abbreviations, possibly with OCR errors. Each of the degraded
        name variants can gets assigned a (manually chosen) distance,
        used for ranking candidate canonical instrument names for a
        particular freeform instrument name string.

        Parameters
        ----------
        name : str
            the instrument's name in english, such as 'Piano',
            'Clarinet', 'French Horn', 'Contrabassoon', etc.

        alt_dist : dictionary
            this is a dictionary of pairs of alternative instrument
            names (and/or abbreviations of such) as expected to occur
            throughout scores, plus (manually) assigned distances
            to the "original" name that is given by `name`.
            Example: {'piano': 0} as e.g. the alternative name / distance
            pair to the "canonical" name 'Piano'.
        """

        def __init__(self, name, alt_dist):
            self.name = name
            self.alt_dist = alt_dist

        def dist(self, s):
            x = normalize_string(s)
            dist = np.inf
            for l, v in list(self.alt_dist.items()):
                if x.startswith(l):
                    dist = min(dist, v)
            return dist

    def __init__(self):
        # in this place all possible OCR errors are reactively
        # coerced to the right instruments...
        # set up a list of InstrumentName objects that contain dicts
        # of alternative names and their respective distances to the
        # full "canonical" name
        self.instruments = [
            self.InstrumentName(PIANO, {'piano': 0}),
            self.InstrumentName(HARPSICHORD, {'harpsichord': 0,
                                              'cembalo': 0}),
            self.InstrumentName(CELESTA, {'celesta': 0,
                                          'cel': 1}),
            self.InstrumentName(ORGAN, {'organ': 0,
                                        'orgel': 0}),
            self.InstrumentName(FLUTE, {'flute': 0,
                                        'flûte': 0,  # french
                                        'flöte': 0,
                                        'grfl': 0,
                                        'großefl': 0,
                                        'grossefl': 0,
                                        'flaut': 1,
                                        'fl': 2,
                                        'f1': 2}),
            self.InstrumentName(PICCOLO, {'picc': 0,
                                          'flautopiccolo': 0,
                                          'flautipiccoli': 0,
                                          'petiteflute': 0,  # french
                                          'petiteflûte': 0,
                                          'klfl': 0,
                                          'kleinefl': 0}),
            self.InstrumentName(OBOE, {'oboe': 0,
                                       'oboi': 0,
                                       'hautbois': 0,  # french
                                       'ob': 2,
                                       'hob': 3}),
            self.InstrumentName(ENGLISHHORN, {'englishhorn': 0,
                                              'coranglais': 0,
                                              'cornoingles': 0,
                                              'engl': 1}),
            self.InstrumentName(HECKELPHONE, {'heckelphon': 0,
                                              'heck': 1}),
            self.InstrumentName(CLARINET, {'clar': 1,
                                           'klar': 1,
                                           'kla': 1,      # check if this works
                                           'clarinet': 0,
                                           'klarinet': 0,
                                           'petiteclarinet': 1,  # let's see if that works (Eb clarinet)
                                           'cl': 2,
                                           'c1': 2}),
            self.InstrumentName(BASSCLARINET, {'bassclar': 0,
                                               'bassklar': 0,
                                               'baßklar': 0,
                                               'bcl': 0,
                                               'bkl': 0,
                                               'clarinettobasso': 0}),
            self.InstrumentName(BASSOON, {'fag': 0,
                                          'bassoon': 0,
                                          'basson': 0,  # french
                                          'bs': 2}),
            self.InstrumentName(CONTRABASSOON, {'contrafag': 0,
                                                'kontrafag': 0,
                                                'ctrfg': 0,
                                                'ctrfag': 0,
                                                'cfag': 0,
                                                'kfag': 0,
                                                'contrabassoon': 0,
                                                'doublebassoon': 0}),  # problems with double bass?
            self.InstrumentName(SOPRANOSAXOPHONE, {'sopranosax': 0,
                                                   'sopransax': 0}),
            self.InstrumentName(ALTOSAXOPHONE, {'altosax': 0,
                                                'altsax': 0}),
            self.InstrumentName(TENORSAXOPHONE, {'tenorsax': 0}),
            self.InstrumentName(BARITONESAXOPHONE, {'baritonesax': 0,
                                                    'baritonsax': 0}),
            self.InstrumentName(FRENCHHORN, {'frenchhorn': 0,
                                             'horn': 0,
                                             'hörn': 0,
                                             'hn': 1,
                                             'cor': 1,
                                             'como': 2,
                                             'c0r': 2}),
            self.InstrumentName(WAGNER_TUBA, {'wagnertub': 0,
                                              'tenortub': 0,       # c.f. e.g. Alpensinfonie
                                              'wagnerhorn': 0,     # not very common?
                                              'bayreuthtub': 0}),  # not very common?
            self.InstrumentName(TRUMPET, {'trumpet': 0,
                                          'clarin': 0,  # clarino, clarini
                                          'tromba': 0,
                                          'trombe': 0,
                                          'trompet': 0,
                                          'trp': 1,
                                          'tpt': 2,
                                          'clno': 2}),
            self.InstrumentName(CORNET, {'kornett': 0,
                                         'cornet': 0}),
            self.InstrumentName(TROMBONE, {'trombo': 1,
                                           'posaun': 0,
                                           'tbn': 1,
                                           'tr': 2}),
            self.InstrumentName(ALTOTROMBONE, {'trombonealt': 0,
                                               'altpos': 0,
                                               'atr': 0,
                                               'atbn': 0,
                                               'alttr': 0,
                                               'altotr': 0}),
            self.InstrumentName(TENORTROMBONE, {'trombonetenor': 0,
                                                'tenorpos': 0,
                                                'ttr': 0,
                                                'ttbn': 0,
                                                'tentr': 0,
                                                'tenortr': 0}),
            self.InstrumentName(BASSTROMBONE, {'trombonebass': 0,
                                               'basspos': 0,
                                               'basstbn': 0,
                                               'tbnbasso': 0,
                                               'tbnbass': 0,
                                               'basstromb': 0}),
            self.InstrumentName(TUBA, {'tuba': 0,
                                       'tb': 2,
                                       'basstuba': 1,
                                       'baßtuba': 1}),
            self.InstrumentName(CONTRABASSTUBA, {'contrabasstub': 0,
                                                 'kontrabasstub': 0,
                                                 'kontrabaßtuba': 0,
                                                 'tubacontrebass': 0,
                                                 'tubacontrabass': 0}),
            self.InstrumentName(OPHICLEIDE, {'ophicleide': 0,
                                             'ophicléide': 0,  # french
                                             'ophikleide': 0}),
            self.InstrumentName(GLOCKENSPIEL, {'glockenspiel': 0,
                                               'bells': 0,
                                               'glspl': 0}),
            self.InstrumentName(TIMPANI, {'timp': 0,
                                          'timbale': 0,
                                          'pauk': 0}),
            self.InstrumentName(HARP, {'harp': 0,
                                       'harfe': 0,
                                       'hfe': 1}),
            self.InstrumentName(VIOLIN, {'violin': 0,
                                         'violon': 1,  # french, balance with violoncell!
                                         'geige': 0,   # german
                                         'gg': 1,      # german, abbrev. "Gg." for "Geige"
                                         'vln': 2,
                                         'vni': 2,
                                         'soloviolin': 2}),
            self.InstrumentName(VIOLA, {'viola': 0,
                                        'viole': 0,
                                        'bratsche': 0,
                                        'br': 1,
                                        'vla': 2,
                                        'vle': 2}),
            self.InstrumentName(CELLO, {'violoncell': 0,
                                        'voiloncello': 2,
                                        'cello': 0,
                                        'vlc': 2,
                                        'vc': 2}),
            self.InstrumentName(CONTRABASS, {'contrabass': 0,
                                             'contrebass': 0,  # french
                                             'doublebass': 0,
                                             'kontrabass': 0,
                                             'kontraba': 1,
                                             'bassi': 0,
                                             'basso': 1,
                                             'cb': 0,
                                             'db': 1}),
            self.InstrumentName(SLEIGHBELLS, {'sleighbell': 0,
                                              'jinglebell': 0,
                                              'schelle': 0,
                                              'sche': 2,
                                              'sch': 3}),  # dangerous?
            self.InstrumentName(TAMTAM, {'tamtam': 0,
                                         'tam-tam': 0,  # necessary?
                                         'tam': 2}),
            self.InstrumentName(CYMBALS, {'cymb': 0,
                                          'beck': 0,
                                          'cin': 2,  # cinelli, etc.
                                          'bck': 2}),
            self.InstrumentName(TRIANGLE, {'triang': 0,
                                           'tri': 1,
                                           'trgl': 2}),
            self.InstrumentName(BASSDRUM, {'bassdr': 0,
                                           'großetr': 0,
                                           'grossetr': 0,
                                           'grantamb': 0,
                                           'grtamb': 0,
                                           'tamburogr': 0,
                                           'grcaisse': 0,
                                           'grossecaisse': 0,
                                           'grancassa': 0,
                                           'grcassa': 0,
                                           'grtr': 3}),
            self.InstrumentName(SNAREDRUM, {'tambour': 0,  # french
                                            'snaredrum': 0}),
            self.InstrumentName(SOPRANO_VOICE, {'sopran': 0}),
            self.InstrumentName(MEZZOSOPRANO_VOICE, {'mezzosopran': 0}),
            self.InstrumentName(ALTO_VOICE, {'alt': 0}),
            self.InstrumentName(CONTRAALTO_VOICE, {'contraalt': 0}),
            self.InstrumentName(TENOR_VOICE, {'tenor': 0}),
            self.InstrumentName(BARITONE_VOICE, {'bariton': 0}),
            self.InstrumentName(BASS_VOICE, {'bass': 1,
                                             'bassi': 3})  # check with contrabass
            ]

        # channel numbers are in the range 1-16,
        # program (patch) numbers are in the range 0-127.
        self.channel_patch_pairs = {PIANO: (1, 0),
                                    HARPSICHORD: (1, 6),
                                    CELESTA: (1, 8),
                                    ORGAN: (1, 19),
                                    FLUTE: (1, 73),
                                    PICCOLO: (1, 72),  # has own program number
                                    OBOE: (6, 68),
                                    ENGLISHHORN: (5, 69),
                                    HECKELPHONE: (6, 68),  # mapped to oboe
                                    CLARINET: (3, 71),
                                    BASSCLARINET: (3, 71),  # mapped to clarinet
                                    BASSOON: (2, 70),
                                    CONTRABASSOON: (2, 70),  # mapped to basson
                                    SOPRANOSAXOPHONE: (4, 64),
                                    ALTOSAXOPHONE: (4, 65),
                                    TENORSAXOPHONE: (4, 66),
                                    BARITONESAXOPHONE: (4, 67),
                                    FRENCHHORN: (5, 60),
                                    WAGNER_TUBA: (5, 60),  # mapped to french horn
                                    TRUMPET: (13, 56),
                                    CORNET: (13, 56),  # mapped to trumpet
                                    TROMBONE: (7, 57),
                                    ALTOTROMBONE: (7, 57),   # mapped to trombone
                                    TENORTROMBONE: (7, 57),  # mapped to trombone
                                    BASSTROMBONE: (7, 57),   # mapped to trombone
                                    TUBA: (14, 58),
                                    CONTRABASSTUBA: (14, 58),  # mapped to tuba
                                    OPHICLEIDE: (14, 58),      # mapped to tuba
                                    GLOCKENSPIEL: (13, 9),
                                    TIMPANI: (12, 47),
                                    HARP: (14, 46),
                                    VIOLIN: (9, 48),
                                    VIOLA: (9, 48),
                                    CELLO: (11, 48),
                                    CONTRABASS: (4, 48),
                                    SLEIGHBELLS: (10, 48),  # (channel 10 reserved for percussion, 48 = orchestral percussion kit)
                                    TAMTAM: (10, 48),
                                    CYMBALS: (10, 48),
                                    TRIANGLE: (10, 48),
                                    BASSDRUM: (10, 48),
                                    SNAREDRUM: (10, 48),
                                    SOPRANO_VOICE: (16, 53),  # all voices are mapped to patch 'Voice Oohs'
                                    MEZZOSOPRANO_VOICE: (16, 53),
                                    ALTO_VOICE: (16, 53),
                                    CONTRAALTO_VOICE: (16, 53),
                                    TENOR_VOICE: (16, 53),
                                    BARITONE_VOICE: (16, 53),
                                    BASS_VOICE: (16, 53),
                                    }

    def map_instrument(self, s):
        """
        function that tries to map the given string to a canonical
        instrument name.

        Parameters
        ----------
        s : str
            the freeform name (identifier, label) of the instrument.

        Returns
        -------
        str or None
            the canonical instrument name if one was found.
            Else, None is returened.
        """

        dists = np.zeros(len(self.instruments))
        for i, instr in enumerate(self.instruments):
            dists[i] = instr.dist(s)
        i = np.argmin(dists)
        if dists[i] < np.inf:    # smaller than infinity?
            return self.instruments[i].name
        else:
            # LOGGER.warning(u'No canonical name could be found for "{0}"'.format(s))
            return None


class UnkownTranspositionException(BaseException):
    pass


class Instrument(object):
    """
    A class that represents an individual instrument.

    It is instantiated from a MIDI track name or an OCR-extracted staff
    name. The name is coerced (if possible) to a canonical instrument
    name, and if applicable, the "rank" of the instrument (e.g. Viola
    1, Viola 2) is determined.

    Parameters
    ----------
    name : str
        the instrument's name (identifier, label) to be coerced to
        a "canonical" name

    Attributes
    ----------
    name : str

    canonical_name : str or None

    transposition : integer or None
        the transposition of the instrument in +/- semitones.

    rank : number or None
        the rank here means e.g. Viola 1 vs Viola 2.

    channel : number
        the MIDI channel number assigned to the instrument. Should be
        in the range of 1-16. Note that channel 10 is reserved for
        unpitched percussion instruments in the General Midi standard.

    patch : number
        the MIDI patch (program) number assigned to the instrument.
        This should be the General MIDI (GM) number for that instrument,
        usually in the range of 0-127.
        Note that unpitched percussion instruments (should be set to chn 10)
        will typically receive a patch number that is used to select a
        drum set (e.g. patch 48 for orchestral percussion set).
    """

    im = InstrumentNameMapper()
    rank_pat = re.compile('\s(?:(3|III|iii)|(2|ii|II)|([iI1]))($|\W)')

    def __init__(self, name):
        self.name = name
        self.canonical_name = self.im.map_instrument(normalize_string(name))
        # self.transposition = estimate_transposition(self.canonical_name, self.name)
        self.rank = self.estimate_rank()
        self.channel, self.patch = self.im.channel_patch_pairs.get(self.canonical_name, (None, None))

    def __unicode__(self):
        return '{0} ({1}/{2}/{3})'.format(self.name, self.canonical_name,
                                           self.rank, self.transposition)

    def __str__(self):
        return self.__unicode__().encode('utf8')

    def estimate_rank(self):
        """
        estimate the rank of the instrument, e.g. first Violin vs second
        Violin, etc.

        Returns
        -------
        number or None
        """

        # use the reg ex given above to estimate the rank
        m = self.rank_pat.search(self.name)
        if m:
            if m.group(1):
                return 3
            elif m.group(2):
                return 2
            elif m.group(3):
                return 1
        else:
            return None

    @property  # check whether this breaks anything
    def transposition(self):
        return estimate_transposition(self.canonical_name, self.name)


def decompose_name(s):
    """
    If `s` is a conjunction of two instrument names (e.g. "Violoncello
    e Contrabasso"), return both parts. Furthermore, if `s` contains a
    conjunction, return both parts, e.g. return ("Viola 1", "Viola 2")
    for "Viola 1,2".

    Parameters
    ----------
    s : str

    Returns
    -------
    list of str
    """

    s_lower = s.lower()
    conjunction_pat = re.compile('(?:\+|&| e | i )')
    parts = conjunction_pat.split(s_lower)

    if len(parts) > 1:
        return parts

    onetwo_pat = re.compile(r'(.*)(?:([iI1])[&,-\u2014]\s?(2|ii|II))(.*)', re.UNICODE)
    m = onetwo_pat.search(s)
    if m:
        pre, g1, g2, post = m.groups()
        if not pre.endswith(' '):
            pre = pre + ' '
        return [''.join((pre, g1, post)),
                ''.join((pre, g2, post))]
    else:
        return [s]


def assign_instruments(instr_sc_channel_map, extracted_staff_names):
    """
    This function is used for mapping OMR/OCR information about staffs
    to a list of instruments. This function should be elsewhere, not
    in instrument_assignment.py ...

    Parameters
    ----------
    instr_sc_channel_map :

    extracted_staff_names :

    Returns
    -------
    score_channels : list

    """

    sc_instruments = [(Instrument(k), v) for k, v in list(instr_sc_channel_map.items())]

    sc_instruments_map = dict(((instr.canonical_name, instr.rank), (instr, sc))
                              for instr, sc in sc_instruments)

    for k, v in list(sc_instruments_map.items()):
        print((k, v[0].name))
    print('')
    for k in extracted_staff_names:
        print(k)
    print('')

    score_channels = []

    for k in extracted_staff_names:
        names = decompose_name(k)
        instruments = [Instrument(n) for n in names]
        print(('{0}:'.format(k)))
        esc = []
        #print(k, names, instruments)
        for instr in instruments:
            #print((instr.canonical_name, instr.rank))
            mapped_instr, score_channel = sc_instruments_map.get((instr.canonical_name, instr.rank), (None, None))
            if mapped_instr:
                print(('\t{0}/{1} ({2})'.format(mapped_instr.canonical_name, mapped_instr.rank, score_channel)))
                esc.append(score_channel)
            else:
                found_cn = False
                if instr.rank is None:
                    # it may be that there are rank 1,2 instruments, and this staff refers to both implicitly
                    for (cn, r), (sc_i, sc_channel) in list(sc_instruments_map.items()):
                        if cn == instr.canonical_name:
                            print(('\t{0}/{1} ({2})'.format(cn, r, sc_channel)))
                            found_cn = True
                            esc.append(sc_channel)
                if instr.rank is not None or not found_cn:
                    print(('ERROR', k, instr.name, instr.canonical_name, instr.rank))
        score_channels.append(esc)
        print('')

    print(score_channels)

    return score_channels

# TRANSPOSITIONS contains the possible transpositions of an instrument
# type, and regular expression patterns that trigger the
# transpositions in a freeform instrument name (e.g. "Horn in
# F"). TRANSPOSITIONS is a dictionary where keys are canonical
# instrument names (defined as global variables earlier in this
# file). The values are either:
#
# 1. An integer, representing the transposition in semitones for this
#    instrument (without inspecting the freeform name)
#
# 2. A tuple of pairs, where the first element of each pair is a
#    regular expression, and the second element is either of form
#    1. or of form 2.
#
# The nesting of the regular expressions defines a nested structure of
# `if-elif` clauses

TRANSPOSITIONS = {
    CLARINET:  # is this ignored for e.g. a C clarinet (that doesn't need transposition)?
    (
        (re.compile('(sul|in|en) (Sib|B)', re.UNICODE), -2),
        (re.compile('(sul|in|en) (A|La)', re.UNICODE), -3),
        (re.compile('(sul|in|en) (Es|Mib)', re.UNICODE), +3),
        (re.compile('(sul|in|en) (D|Re)', re.UNICODE), +2)
    ),
    TRUMPET:
    (
        (re.compile('(sul|in|en) (Sib|B)', re.UNICODE), -2),
        (re.compile('(sul|in|en) (D|Re)', re.UNICODE), +2),
        (re.compile('(sul|in|en) (Fa|F)', re.UNICODE), +5),
        (re.compile('(sul|in|en) (Es|Mib)', re.UNICODE), +3)
    ),
    CORNET:
    (   # This here is introduced first time for Berlioz symphonie fantastique
        # which calls for "Cornet a Pistons", which seems to be an ancestor to
        # the modern Cornet. The score calls for a "Cornet a Pistons en Sol",
        # therefore a transpostion for key of G is given here, derived from
        # how a trumpet in G is notated.
        (re.compile('(sul|in|en) (Sib|B)', re.UNICODE), -2),
        (re.compile('(sul|in|en) (G|Sol)', re.UNICODE), +7)   # like a trumpet in G would be. Correct?
    ),
    FRENCHHORN:
    (
        (re.compile('(sul|in|en) (Fa|F)', re.UNICODE), -7),
        (re.compile('(sul|in|en) (Es|Mib)', re.UNICODE), -9),
        (re.compile('(sul|in|en) (E|Mi)', re.UNICODE), -8),     # e.g. symphonie fantastique
        (re.compile('(sul|in|en) (D|Re)', re.UNICODE), -10),
        (re.compile('(sul|in|en) (Sib|B)', re.UNICODE),
            (
                (re.compile('basso|tief|grave', re.UNICODE), -14),
                (re.compile('alto|hoch|haut', re.UNICODE), -2),
                # this is  a fallback that should
                # give at least the correct key.
                (re.compile('', re.UNICODE), -2)
            )
        ),
    ),
    WAGNER_TUBA:
    (   # The Tenor Wagner Tuba in B may in modern notation be written
        # like a French Horn and would thus have to be transposed by
        # -7 semitones instead of the -2. This would have to be checked
        # and some workaround would have to be used.
        # TO DO: for the future: make it possible to override the
        # transposition defined here from the outside when the specific
        # piece/score requieres it?
        (re.compile('(sul|in|en) (Sib|B)', re.UNICODE), -2),  # Tenor Wagner Tuba
        (re.compile('(sul|in|en) (Fa|F)', re.UNICODE), -7)    # Bass Wagner Tuba
    ),
    BASSCLARINET:
    (
        (re.compile('(sul|in|en) (Sib|B)', re.UNICODE), -14),
        (re.compile('(sul|in|en) (La|A)', re.UNICODE), -15)
    ),
    PICCOLO: +12,
    # (
    #     (re.compile(u'(sul|in|en) (Do|C)', re.UNICODE), +12),
    #     (re.compile(u'(sul|in|en) (Des|Reb)', re.UNICODE), +13)  # may occur
    # )
    ENGLISHHORN: -7,
    CONTRABASS: -12,
    CONTRABASSOON: -12
}


def _est_transp_recursive(transp, name):
    """
    Internal function used by `estimate_transposition`

    """

    if isinstance(transp, int):
        return transp
    else:
        try:
            for pattern, result in transp:
                if pattern.search(name) is not None:
                    # print('matching name to', name, pattern.pattern)
                    est = _est_transp_recursive(result, name)
                    return est
            return None
        except:
            Exception(('Format error in TRANSPOSITIONS detected while '
                       'determining transposition for instrument "{0}"')
                      .format(name))


def estimate_transposition(canonical_name, name):
    """
    Lookup transpositions of instruments, given the instrument type
    `canonical_name` (should match one of the global variable
    instrument names), and a freeform string `name`. This function
    checks `name` for transposition information (such as "in Mi"), and
    returns a transposition (in semitones) accordingly.

    This function relies on a list of nested pairs of regular
    expressions, and transpositions, defined in a global variable
    TRANSPOSITIONS

    Returns
    -------
    integer or None
        the transposition in +/- semitones.
    """

    instr_transp = TRANSPOSITIONS.get(canonical_name, None)
    if instr_transp is None:
        return None
    else:
        return _est_transp_recursive(instr_transp, name)

