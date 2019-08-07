#!/usr/bin/env python

"""
Parse textual directions that occur in a score (in a MusicXML they are
 encoded as <words></words>), and if possible, convert them to a specific
 score.Direction class or subclass. For example "cresc." will produce a
 `score.DynamicLoudnessDirection` instance, and "Allegro molto" will produce a
 `score.ConstantTempoDirection` instance. If the meaning of the direction cannot
 be inferred, a `score.Words` instance is returned.

The functionality is provided by the function `parse_words`
"""

import logging

LOGGER = logging.getLogger(__name__)

try:
    from lark import Lark
    HAVE_LARK = True
except ImportError as e:
    LOGGER.warning('''package "lark" not found; Textual directions will not be 
parsed to form `score.Direction` objects but included as `score.Words` 
instead; Install using "pip install lark-parser"''')
    HAVE_LARK = False

# TODO: interpret roman numerals
# import roman
# regexp: '(viii|iii|vii|ii|iv|vi|ixi|v|x|i)\.?'
# convert: '{}'.format(roman.fromRoman(t.value.upper()))

import partitura.score as score


def join_items(items):
    """
    form a terminal from a list of strings, by joining them using | and
    enclosing them in double quotes, unless they start with /, implying they are
    regular expressions. Furthermore, each item is flagged with "i" for
    case-insensitivity.
    """
    return ' | '.join(f'{item}i' if item.startswith('/') else f'"{item}"i'
               for item in items)


INC_LOUDNESS_ADJ = [
    '/(crescendo|cresc\.?)/',
    ]

DEC_LOUDNESS_ADJ = [
    "raddolcendo",
    '/(smorzando|smorz\.?)/',
    'calando',
    '/(diminuendo|decrescendo|(decresc|decr|dimin|dim)\.?)/',
    ]

INC_TEMPO_ADJ = [
    '/(accelerando|(acceler|accel|acc)\.?)/',
    'rubato',
    ]
DEC_TEMPO_ADJ = [
    '/(ritenuto|ritenente|riten\.?)/',
    '/(ritardando|(ritard|rit)\.?)/',
    '/(rallentando|(rallent|rall)\.?)/',
    ]

DYNAMIC_QUANTIFIER = [
    "poi a poi",
    "poco a poco",
    "sempre piu",
    "sempre più",
]

CONSTANT_QUANTIFIER = [
    "molto",
    "di molto",
    "poco",
    "un poco",
    "ben",
    "piu",
    "più",
    "pio",
    "meno",
    "gran",
    "mezzo",
    "quasi",
    "assai",
    "doppio",
    "troppo",
    "tanto",
    "sempre",
    ", ma non troppo",
    "ma non troppo",
    "non troppo",
    "non tanto",
    "etwas",
]

CONSTANT_ADJ = [
    "adagio",
    "agitato",
    "andante",
    "andantino",
    "animato",
    "appassionato",
    "arioso",
    "brillante",
    "cantabile",
    "comodo",
    "delicatissimo",
    "religioso",
    "dolce",
    "dolente",
    "energico",
    "forte",
    "grave",
    "grazioso",
    "langsamer",
    "larghetto",
    "largo",
    '/(leggiermente|leggiero|leggierissimo|(leggieriss|legg)\.?)/',
    "lento",
    "ligato",
    "lusingando",
    "maestoso",
    "mancando",
    "marcato",
    "mesto",
    "moderato",
    "mosso",
    "pesante",
    "piacevole",
    "piano",
    "prestissimo",
    "presto",
    "risoluto",
    "risvegliato",
    "scherzando",
    "secco",
    "/s[ei]mplice/",
    "slentando",
    "staccato",
    "stretto",
    "stringendo",
    "teneramente",
    '/(tenute|tenuto|ten\.?)/',
    "tranquilamente",
    "tranquilo",
    "recitativo",
    '/(vivo|vivacissimamente|vivace)/',
    '/(allegro|allegretto)/',
    '/(espressivo|espress\.?)/',
    '/(legato|legatissimo|ligato|ligatissimo)/',
    '/(rinforzando|(rinforz|rinf|rfz|rf)\.?)/',
    '/(sostenuto|(sosten|sost)\.?)/',
    "pp",
    "p",
    "f",
]

NOUN = [
    "brio",
    "espressione",
    "marcia",
    "bravura",
    "variation",
    "moto",
    "fine",
    "fuoco",
    "duolo",
    "anima",
    "forza",
    "voce",
    "tempo",
    "sentimento",
    "movimento",
]

GENRE = [
    "scherzo",
    "marcia",
    "tedesca",
    "mazurka",
    "menuetto",
]
# TODO: support "tempo di GENRE"

PREP = [
    "con",
    "alla",
    "al",
    "del",
    "sotto",
    "senza"
]


DIRECTION_GRAMMAR = f"""
start: direction -> do_first
     | direction conj direction -> conj
     | direction "("? tempo_indication ")"?  -> do_both
     | neg direction

direction: ap
         | pp 
         | ap pp -> do_first
         | tempo_reset
         | "("? tempo_indication ")"?
         | genre

genre: GENRE
     | GENRE ap

pp: PREP np -> words
  | PREP GENRE -> do_second

np: NOUN
  | _quantifier NOUN

ap: _adj      -> do_first    // allegro
  | _adj _adj -> do_first    // allegro grazioso | lento sostenuto 
  | _quantifier _adj -> do_second
  | _adj _quantifier -> do_first
  | _quantifier pp -> do_second

_adj: constant_adj
    | inc_loudness_adj
    | dec_loudness_adj
    | inc_tempo_adj
    | dec_tempo_adj

constant_adj: CONSTANT_ADJ
inc_loudness_adj: INC_LOUDNESS_ADJ
dec_loudness_adj: DEC_LOUDNESS_ADJ
inc_tempo_adj: INC_TEMPO_ADJ
dec_tempo_adj: DEC_TEMPO_ADJ

_quantifier: constant_quantifier
           | dynamic_quantifier

constant_quantifier: CONSTANT_QUANTIFIER
dynamic_quantifier: DYNAMIC_QUANTIFIER

conj: CONJ
neg: NEG

tempo_indication: NOTE "=" BPM
tempo_reset: "("? TEMPO_RESET ")"?

TEMPO_RESET: /((in|a) )?tempo (i|1|primo)/i | /(in|a) tempo/i | "doppio movimento"i

NOTE: /[qhe]\.*/i
BPM: /[0-9]+/

NOUN: {join_items(NOUN)}
PREP: {join_items(PREP)}
CONSTANT_ADJ: {join_items(CONSTANT_ADJ)}
INC_LOUDNESS_ADJ: {join_items(INC_LOUDNESS_ADJ)}
DEC_LOUDNESS_ADJ: {join_items(DEC_LOUDNESS_ADJ)}
INC_TEMPO_ADJ: {join_items(INC_TEMPO_ADJ)}
DEC_TEMPO_ADJ: {join_items(DEC_TEMPO_ADJ)}
CONSTANT_QUANTIFIER: {join_items(CONSTANT_QUANTIFIER)}
DYNAMIC_QUANTIFIER: {join_items(DYNAMIC_QUANTIFIER)}
GENRE: {join_items(GENRE)}
CONJ: "ed"i | "e"i | "und"i | ","i
NEG: "non"i

%import common.WS
%ignore WS

"""

def create_directions(tree, string):
    """
    Recursively walk the parse tree of `string` to create a `score.Direction` or `score.Words` instance.

    """
    
    if tree.data == 'conj':
        return (create_directions(tree.children[0], string),
                create_directions(tree.children[2], string))

    elif tree.data in ('direction', 'do_first'):
        return create_directions(tree.children[0], string)

    elif tree.data == 'do_second':
        return create_directions(tree.children[1], string)

    elif tree.data == 'do_both':
        return (create_directions(tree.children[0], string),
                create_directions(tree.children[1], string))

    elif tree.data == 'constant_adj':
        return score.Direction(" ".join(tree.children), string)

    elif tree.data in ('inc_loudness_adj', 'dec_loudness_adj'):
        return score.DynamicLoudnessDirection(" ".join(tree.children), string)

    elif tree.data in ('inc_tempo_adj', 'dec_tempo_adj'):
        return score.DynamicTempoDirection(" ".join(tree.children), string)

    elif tree.data == 'genre':
        if len(tree.children) > 1:
            # this can be something like "scherzo vivace" or "marcia funebre"
            pass
        return score.ConstantTempoDirection(tree.children[0], string)

    elif tree.data == 'tempo_indication':
        bpm = int(tree.children[1])
        unit = tree.children[0]
        return score.Tempo(bpm, unit)

    elif tree.data == 'tempo_reset':
        return score.ResetTempoDirection(" ".join(tree.children), string)

    elif tree.data == 'words':
        return score.Words(string)

    else:
        LOGGER.warning('unhandled: {}'.format(string))
        return score.Words(string)

if HAVE_LARK:
    DEFAULT_PARSER = Lark(DIRECTION_GRAMMAR, start='start', ambiguity='explicit', propagate_positions=True)
else:
    DEFAULT_PARSER = None

def parse_words(string, parser=DEFAULT_PARSER):
    if DEFAULT_PARSER:
        try:
            parse_result = parser.parse(string)
            direction = create_directions(parse_result, string)
        except Exception as e:
            LOGGER.warning('error parsing "{}" ({})'.format(string, e))
            direction = score.Words(string)
    else:
        direction = score.Words(string)
    return direction


    
