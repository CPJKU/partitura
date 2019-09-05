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

try:
    from lark import Lark
    HAVE_LARK = True
except ImportError:
    logging.getLogger(__name__).warning('''package "lark" not found; Textual directions will not be
parsed to form `score.Direction` objects but included as `score.Words`
instead; Install using "pip install lark-parser"''')
    HAVE_LARK = False

__all__ = ['parse_words']

# TODO: interpret roman numerals
# import roman
# regexp: '(viii|iii|vii|ii|iv|vi|ixi|v|x|i)\.?'
# convert: '{}'.format(roman.fromRoman(t.value.upper()))

import partitura.score as score

LOGGER = logging.getLogger(__name__)


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
    r'/(crescendo|cresc\.?)/',
]

DEC_LOUDNESS_ADJ = [
    "raddolcendo",
    r'/(smorzando|smorz\.?)/',
    'perdendosi',
    'calando',
    r'/(diminuendo|decrescendo|(decresc|decr|dimin|dim)\.?)/',
]

INC_TEMPO_ADJ = [
    r'/(accelerando|(acceler|accel|acc)\.?)/',
    'rubato',
]
DEC_TEMPO_ADJ = [
    r'/(ritenuto|ritenente|riten\.?)/',
    r'/(ritardando|(ritard|rit)\.?)/',
    r'/(rallentando|(rallent|rall)\.?)/',
]

DYNAMIC_QUANTIFIER = [
    "poi a poi",
    "poco a poco",
    "sempre piu",
    "sempre più",
]

CONSTANT_QUANTIFIER = [
    r'/molt[oa]/',
    "di molto",
    r'/poc[oa]/',
    "un poco",
    "ben",
    "piu",
    "più",
    "pio",
    "meno",
    "gran",
    r'/mezz[oa]/',
    "quasi",
    "assai",
    "/doppi[oa]/",
    "troppo",
    "tanto",
    "sempre",
    ", ma non troppo",
    "ma non troppo",
    "non troppo",
    "non tanto",
    "etwas",
]

CONSTANT_LOUDNESS_ADJ = [
    r'/(dolcissimo|dolciss\.?|dolce)/',
    "forte",
    "delicato",
    "energico",
    "piano",
    "pp",
    "p",
    "f",
    r'/(sostenuto|(sosten|sost)\.?)/',
]

CONSTANT_TEMPO_ADJ = [
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
    r'/(delicatissimo|delicatiss\.?)/',
    "religioso",
    "dolente",
    "funebre",
    "grave",
    "grazioso",
    "langsamer",
    "larghetto",
    "largo",
    r'/(leggieramente|leggiermente|leggiero|leggierissimo|(leggieriss|legg)\.?)/',
    "lento",
    "lusingando",
    "maestoso",
    "mancando",
    "marcato",
    "mesto",
    "moderato",
    "mosso",
    "pesante",
    "piacevole",
    "prestissimo",
    "presto",
    "risoluto",
    "risvegliato",
    r'/(scherzando|scherz\.?)/',
    "secco",
    "/s[ei]mplice/",
    "slentando",
    "stretto",
    "stringendo",
    "teneramente",
    r'/(tenute|tenuto|ten\.?)/',
    "tranquilamente",
    "tranquilo",
    "recitativo",
    r'/(vivo|vivacissimamente|vivace)/',
    r'/(allegro|allegretto)/',
    r'/(espressivo|espress\.?)/',
]


CONSTANT_ARTICULATION_ADJ = [
    r'/(staccato|staccatissimo)/',
    r'/(legato|legatissimo|ligato|ligatissimo)/',
]

# adjectives that may express a combination of tempo loudness and articulation directions
CONSTANT_MIXED_ADJ = [
    r'/(rinforzando|(rinforz|rinf|rfz|rf)\.?)/',
]


NOUN = [
    "brio",
    "espressione",
    "marcia",
    "bravura",
    "variation",
    "moto",
    "fine",
    "energia",
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


DIRECTION_GRAMMAR = rf"""
start: direction -> do_first
     | direction conj direction -> conj
     | direction "("? tempo_indication ")"?  -> do_both
     | neg direction

direction: ap
         | pp
         | np
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

_adj: constant_tempo_adj
    | constant_loudness_adj
    | constant_articulation_adj
    | constant_mixed_adj
    | inc_loudness_adj
    | dec_loudness_adj
    | inc_tempo_adj
    | dec_tempo_adj

constant_tempo_adj: CONSTANT_TEMPO_ADJ
constant_loudness_adj: CONSTANT_LOUDNESS_ADJ
constant_articulation_adj: CONSTANT_ARTICULATION_ADJ
constant_mixed_adj: CONSTANT_MIXED_ADJ
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
CONSTANT_TEMPO_ADJ: {join_items(CONSTANT_TEMPO_ADJ)}
CONSTANT_LOUDNESS_ADJ: {join_items(CONSTANT_LOUDNESS_ADJ)}
CONSTANT_ARTICULATION_ADJ: {join_items(CONSTANT_ARTICULATION_ADJ)}
CONSTANT_MIXED_ADJ: {join_items(CONSTANT_MIXED_ADJ)}
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
    start = tree.column - 1
    end = tree.end_column - 1
    
    if tree.data == 'conj':
        return (create_directions(tree.children[0], string)
                + create_directions(tree.children[2], string))

    elif tree.data in ('direction', 'do_first'):
        return create_directions(tree.children[0], string)

    elif tree.data == 'do_second':
        return create_directions(tree.children[1], string)

    elif tree.data == 'do_both':
        return (create_directions(tree.children[0], string)
                + create_directions(tree.children[1], string))

    elif tree.data == 'constant_tempo_adj':
        return [score.ConstantTempoDirection(" ".join(tree.children), string[start:end])]

    elif tree.data == 'constant_loudness_adj':
        return [score.ConstantLoudnessDirection(" ".join(tree.children), string[start:end])]

    elif tree.data == 'constant_articulation_adj':
        return [score.ConstantArticulationDirection(" ".join(tree.children), string[start:end])]

    elif tree.data == 'constant_mixed_adj':
        return [score.Direction(" ".join(tree.children), string[start:end])]

    elif tree.data in ('inc_loudness_adj', 'dec_loudness_adj'):
        return [score.DynamicLoudnessDirection(" ".join(tree.children), string[start:end])]

    elif tree.data in ('inc_tempo_adj', 'dec_tempo_adj'):
        return [score.DynamicTempoDirection(" ".join(tree.children), string[start:end])]

    elif tree.data == 'genre':
        if len(tree.children) > 1:
            # this can be something like "scherzo vivace" or "marcia funebre"
            pass
        return [score.ConstantTempoDirection(tree.children[0], string[start:end])]

    elif tree.data == 'tempo_indication':
        bpm = int(tree.children[1])
        unit = tree.children[0]
        return [score.Tempo(bpm, unit)]

    elif tree.data == 'tempo_reset':
        return [score.ResetTempoDirection(" ".join(tree.children), string[start:end])]

    elif tree.data == 'words':
        return [score.Words(string[start:end])]

    else:
        LOGGER.warning('unhandled: {}'.format(string[start:end]))
        return [score.Words(string[start:end])]


if HAVE_LARK:
    DEFAULT_PARSER = Lark(DIRECTION_GRAMMAR, start='start',
                          ambiguity='explicit', propagate_positions=True)
else:
    DEFAULT_PARSER = None


def parse_words(string, parser=DEFAULT_PARSER):
    """
    Description
    
    Parameters
    ----------
    string: type
        Description of `string`
    parser: type, optional
        Description of `parser`
    
    Returns
    -------
    type
        Description of return value
    """
    
    if DEFAULT_PARSER:
        try:
            parse_result = parser.parse(string)
            direction = create_directions(parse_result, string)
        except Exception as e:
            LOGGER.warning('error parsing "{}" ({})'.format(
                string, type(e).__name__))
            direction = [score.Words(string)]
    else:
        direction = [score.Words(string)]
    return direction

