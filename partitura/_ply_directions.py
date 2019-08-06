#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import re
import logging

LOGGER = logging.getLogger(__name__)

try:
    import ply.lex as lex
    import ply.yacc as yacc
except ImportError as e:
    LOGGER.warning('package "ply" not found')
    raise e

try:
    import roman
except ImportError as e:
    LOGGER.warning('package "roman" not found')
    raise e

import partitura.score as score

TEMPOHINT_PARSER = re.compile('explicittempo:(?P<durtype>.+)=(?P<bpm>.+)')


class TokenizeException(Exception):
    pass

# grep t_ annotation_tokenizer.py | perl -pe 's/t_([^ ]+).*/    "\1",/'


class AnnotationTokenizer(object):
    tokens = (
        "P",
        "PP",
        "PPP",
        "MP",
        "MF",
        "F",
        "FF",
        "FFF",
        "A",
        "ACCELERANDO",
        "ADAGIO",
        "AGITATO",
        "ALLA",
        "ALLEGRO",
        "ALLORA",
        "ALS",
        "AMABILITA",
        "AND",
        "ANDANTE",
        "ANIMA",
        "ANIMATO",
        "APPASSIONATO",
        "ARIOSO",
        "ASSAI",
        "AUSDRUCK",
        "BAR",
        "BEN",
        "BRAVURA",
        "BRILLANTE",
        "BRIO",
        "CALANDO",
        "CANTABILE",
        "CARLOSCOMMENT",
        "COMMA",
        "COMODO",
        "CON",
        "CORDA",
        "CORDE",
        "CRESC",
        "CRESCENDO",
        "DAS",
        "DASHES",
        "DEL",
        "DELICATISSIMO",
        "DELICATO",
        "DELLA",
        "DER",
        "DES",
        "DI",
        "DIE",
        "DIMINUENDO",
        "DOCH",
        "DOLCE",
        "DOLENTE",
        "DOPPIO",
        "DUE",
        "DUN",
        "DURCHAUS",
        "EMPFINDUNG",
        "END",
        "ENERGIA",
        "ENERGICO",
        "ENTSCHLOSSENHEIT",
        "ERSTEN",
        "ESPRESSIONE",
        "ESPRESSIVO",
        "ETWAS",
        "FORTE",
        "FORZA",
        "FUGA",
        "FUNEBRE",
        "FUOCO",
        "GESANGVOLL",
        "GESCHWIND",
        "GRAN",
        "GRAVE",
        "GRAZIOSO",
        "IN",
        "INNIGST",
        "ISTESSO",
        "LANGSAM",
        "LANGSAMER",
        "LARGHETTO",
        "LARGO",
        "LE",
        "LEBHAFT",
        "LEBHAFTIGKEIT",
        "LEGATO",
        "LEGGIERO",
        "LENTO",
        "LIGATO",
        "LUSINGANDO",
        "MA",
        "MAESTOSO",
        "MANCANDO",
        "MARCATO",
        "MARCIA",
        "MARSCHMAESSIG",
        "MENO",
        "MENUETTO",
        "MESTO",
        "MEZZO",
        "MINUETTO",
        "MIT",
        "MODERATO",
        "MOVIMENTO",
        "MOLTO",
        "MOSSO",
        "MOTO",
        "NOT",
        "NUOVO",
        "ONE",
        "PARENCLOSE",
        "PARENOPEN",
        "PESANTE",
        "PIACEVOLE",
        "PIANO",
        "PIO",
        "PIU",
        "POCO",
        "POI",
        "POSSIBILE",
        "PRESTISSIMO",
        "PRESTO",
        "PRIMO",
        "QUASI",
        "RALLENTANDO",
        "RINFORZANDO",
        "RISOLUTO",
        "RISVEGLIATO",
        "RITARDANDO",
        "RITENUTO",
        "ROMAN_NUMBER",
        "RUBATO",
        "SCHERZANDO",
        "SCHERZO",
        "SECCO",
        "SEHNSUCHTVOLL",
        "SEMPLICE",
        "SEMPRE",
        "SENTIMENTO",
        "SENZA",
        "SINGBAR",
        "SMORZANDO",
        "SOSTENUTO",
        "SOTTO",
        "SPACE",
        "START",
        "STRETTO",
        "STRINGENDO",
        "STUECKES",
        "SUL",
        "TEDESCA",
        "TEMA",
        "TEMPO",
        "TEMPOHINT",
        "TENERAMENTE",
        "TENUTO",
        "THEMA",
        "THEME",
        "TRANQUILO",
        "TRE",
        "TROPPO",
        "TUTTE",
        "UN",
        "VARIATION",
        "VELOCE",
        "VIVACE",
        "VIVACISSIMAMENTE",
        "VIVENTE",
        "VIVO",
        "VOCE",
        "VORGETRAGEN",
        "WORDS",
        "ZEITMASS",
        "ZU",
    )

    # token definitions
    t_P = r'p'
    t_PP = r'pp'
    t_PPP = r'ppp'
    t_MP = r'mp'
    t_MF = r'mf'
    t_F = r'f'
    t_FF = r'ff'
    t_FFF = r'fff'
    t_A = r'a'
    t_ACCELERANDO = r'(accelerando|(acceler|accel|acc)\.?)'
    t_ADAGIO = r'adagio'
    t_AGITATO = r'agitato'
    t_ALLA = r'alla'
    t_ALLEGRO = r'(allegro|allegretto)'
    t_ALLORA = r'allora'
    t_ALS = r'als'
    t_AMABILITA = r'amabilita'
    t_AND = r'(ed|e|und)'
    t_ANDANTE = r'(andante|andantino)'
    t_ANIMA = r'anima'
    t_ANIMATO = r'animato'
    t_APPASSIONATO = r'(appassionato|appasionato)'
    t_ARIOSO = r'arioso'
    t_ASSAI = r'assai'
    t_AUSDRUCK = r'ausdruck'
    t_BAR = r'bar'
    t_BEN = r'ben'
    t_BRAVURA = r'bravura'
    t_BRILLANTE = r'brillante'
    t_BRIO = r'brio'
    t_CALANDO = r'calando'
    t_CANTABILE = r'cantabile'
    t_CARLOSCOMMENT = r'\(r_cc.*\)'
    t_COMMA = r','
    t_COMODO = r'comodo'
    t_CON = r'(con|mit)'
    t_CORDA = r'corda'
    t_CORDE = r'corde'
    t_CRESC = r'cresc'
    t_CRESCENDO = r'(crescendo|cresc\.?)'
    t_DAS = r'das'
    t_DEL = r'del'
    t_DELICATISSIMO = r'delicatiss.'
    t_DELICATO = r'delicato'
    t_DELLA = r'della'
    t_DER = r'der'
    t_DES = r'des'
    t_DI = r'di'
    t_DIE = r'die'
    t_DIMINUENDO = r'(diminuendo|decrescendo|(decresc|decr|dimin|dim)\.?)'
    t_DOCH = r'doch'
    t_DOLCE = r'(dolcissimo|dolciss\.?|dolce)'
    t_DOLENTE = r'dolente'
    t_DOPPIO = r'doppi[ao]'
    t_DUE = r'due'
    t_DUN = r"d'un"
    t_DURCHAUS = r'durchaus'
    t_ENTSCHLOSSENHEIT = r'entschlossenheit'
    t_FORTE = r'forte'
    t_FORZA = r'forza'
    t_FUGA = r'fuga'
    t_FUNEBRE = r'funebre'
    t_FUOCO = r'fuoco'
    t_GESANGVOLL = r'gesangvoll'
    t_GESCHWIND = r'geschwinde?'
    t_GRAN = r"(grande|grand'?|gran)"
    t_GRAVE = r'grave'
    t_GRAZIOSO = r'grazioso'
    t_LANGSAM = r'langsam'
    t_LARGHETTO = r'larghetto'
    t_LARGO = r'largo'
    t_LE = r'le'
    t_LEBHAFT = r'lebhaft'
    t_LEBHAFTIGKEIT = r'lebhaftigkeit'
    t_LEGATO = r'(legato|legatissimo|ligato|ligatissimo)'
    t_LEGGIERO = r'(leggiermente|leggiero|leggierissimo|(leggieriss|legg)\.?)'
    t_LENTO = r'lento'
    t_LIGATO = r'ligato'
    t_LUSINGANDO = r'lusingando'
    t_MA = r'ma'
    t_MAESTOSO = r'maestoso'
    t_MANCANDO = r'mancando'
    t_MARCATO = r'marcato'
    t_MARCIA = r'marcia'
    t_MARSCHMAESSIG = r'marschmaessig'
    t_MENO = r'meno'
    t_MENUETTO = r'menuetto'
    t_MESTO = r'mesto'
    t_MEZZO = r'mezz[ao]'
    t_MINUETTO = r'minuetto'
    t_MODERATO = r'moderato'
    t_MOLTO = r'(molt[oa]|sehr)'
    t_MOSSO = r'mosso'
    t_MOTO = r'moto'
    t_MOVIMENTO = r'movimento'
    t_NOT = r'(non|nicht)'
    t_NUOVO = r'nuovo'
    t_ONE = r'una?'
    t_PARENCLOSE = r'\)'
    t_PARENOPEN = r'\('
    t_PESANTE = r'pesante'
    t_PIACEVOLE = r'piacevole'
    t_PIANO = r'piano'
    t_PIO = r'pio'
    t_PIU = '(piu|più|piú)'
    t_POCO = r'poco'
    t_POI = r'poi'
    t_POSSIBILE = r'possibile'
#    t_PP = r'pp'
    t_PRESTISSIMO = r'prestissimo'
    t_PRESTO = r'presto'
    t_QUASI = r'quasi'
    t_RALLENTANDO = r'(rallentando|(rallent|rall)\.?)'
    t_RINFORZANDO = r'(rinforzando|(rinforz|rinf|rfz|rf)\.?)'
    t_RISOLUTO = r'risoluto'
    t_RISVEGLIATO = r'risvegliato'
    t_RITARDANDO = r'(ritardando|(ritard|rit)\.?)'
    t_RUBATO = r'rubato'
    t_SECCO = r'secco'
    t_SEHNSUCHTVOLL = r'sehnsuchtvoll'
    t_SEMPLICE = r'semplice'
    t_SEMPRE = r'sempre'
    t_SENTIMENTO = r'sentimento'
    t_SENZA = r'senza'
    t_SINGBAR = r'singbar'
    t_SMORZANDO = r'(smorzando|smorz\.?)'
    t_SOSTENUTO = r'(sostenuto|(sosten|sost)\.?)'
    t_SOTTO = r'sotto'
    t_START = r'start'
    t_STRETTO = r'stretto'
    t_STRINGENDO = r'stringendo'
    t_STUECKES = r'stueckes'
    t_SUL = r'sul'
    t_TEDESCA = r'tedesca'
    t_TEMA = r'tema'
    t_TEMPO = r'tempo'
    t_TENUTO = r'(tenute|tenuto|ten\.?)'
    t_THEMA = r'thema'
    t_THEME = r'theme'
    t_TRANQUILO = r'(tranquilo|tranquilamente)'
    t_TRE = r'tre'
    t_TROPPO = r'troppo'
    t_TUTTE = r'tutte'
    t_UN = r'un'
    t_WORDS = r'words'
    t_ZEITMASS = r'zeitmass'
    t_ZU = r'zu'

    # Ignored characters
    t_ignore = " \t"

    def __init__(self, on_error='raise'):  # on_error = { raise, skip }
        self.build()
        self.on_error = on_error

    def t_LANGSAMER(self, t):
        r'langsamer'
        return t

    def t_DASHES(self, t):
        r'dashes'
        return t

    def t_ERSTEN(self, t):
        r'erst(en|es|er|e)'
        return t

    def t_EMPFINDUNG(self, t):
        r'empfindung'
        return t

    def t_END(self, t):
        r'end'
        return t

    def t_ENERGIA(self, t):
        r'energia'
        return t

    def t_ENERGICO(self, t):
        r'energico'
        return t

    def t_ESPRESSIONE(self, t):
        r'espressione'
        return t

    def t_ESPRESSIVO(self, t):
        r'(espressivo|espress\.?)'
        return t

    def t_ETWAS(self, t):
        r'etwas'
        return t

    def t_PRIMO(self, t):
        r'primo'
        t.value = '1'
        return t

    def t_INNIGST(self, t):
        r'innigste[nrs]'
        return t

    def t_IN(self, t):
        r'in'
        return t

    def t_ISTESSO(self, t):
        r"l'istesso"
        return t

    def t_RITENUTO(self, t):
        r'(ritenuto|ritenente|riten\.?)'
        return t

    def t_SCHERZO(self, t):
        r'scherzo'
        return t

    def t_SCHERZANDO(self, t):
        r'(scherzando|scherz\.?)'
        return t

    def t_TENERAMENTE(self, t):
        r'teneramente'
        return t

    def t_VARIATION(self, t):
        r'(variazioni|variazione|var\.?)'
        return t

    def t_VELOCE(self, t):
        r'veloce'
        return t

    def t_VIVACISSIMAMENTE(self, t):
        r'vivacissimamente'
        return t

    def t_VIVACE(self, t):
        r'vivace'
        return t

    def t_VIVENTE(self, t):
        r'vivente'
        return t

    def t_VIVO(self, t):
        r'vivo'
        return t

    def t_VOCE(self, t):
        r'voce'
        return t

    def t_VORGETRAGEN(self, t):
        r'vorgetragen'
        return t

    def t_ROMAN_NUMBER(self, t):
        r'(viii|iii|vii|ii|iv|vi|ixi|v|x|i)\.?'
        t.value = '{}'.format(roman.fromRoman(t.value.upper()))
        return t

    def t_TEMPOHINT(self, t):
        r'(s|q|e|h|w)\.*\s*=\s*([0-9]+)'
        t.value = 'explicittempo:{}'.format(t.value)
        return t

    def t_error(self, t):
        s = (t.lexer.lexdata[:t.lexpos] +
             '\033[1m' + t.lexer.lexdata[t.lexpos:t.lexpos + 1] + '\033[0m'
             + t.lexer.lexdata[t.lexpos + 1:])
        if self.on_error == 'skip':
            raise TokenizeException("Skipping illegal character '{}' at position {} in \"{}\""
                                    .format(t.value[0], t.lexpos, s).encode('utf8'))
            t.lexer.skip(1)
        else:
            raise TokenizeException("Illegal character '{}' at position {} in \"{}\""
                                    .format(t.value[0], t.lexpos, s).encode('utf8'))

    def build(self, **kwargs):
        # Build the lexer
        self.lexer = lex.lex(module=self, **kwargs)

    def tokenize(self, s):
        s_lower = s.lower()
        self.lexer.input(s_lower)
        toktypes = []
        while True:
            tok = self.lexer.token()
            if not tok:
                break
            toktypes.append(tok)
        # print(' '.join(toktypes))
        return toktypes



def p_annotation0(p):
    ''' annotation : ap
                   | temporeset
                   | pp
    '''
    # lex.input(p[1])
    # print(p[1], lex.token())
    p[0] = p[1]


def p_annotation1(p):
    ''' annotation : annotation PARENOPEN TEMPOHINT PARENCLOSE
    '''
    if isinstance(p[1], score.ConstantTempoDirection):
        m = TEMPOHINT_PARSER.search(p[3])
        if m:
            p[1].duration = m.group('durtype')
            p[1].bpm = m.group('bpm')
        else:
            LOGGER.warning("Cannot interpret tempohint: {}".format(p[3]))
    p[0] = p[1]

# def p_annotation1(p):
#     ''' annotation : annotation AND annotation
#     '''
#     p[0] = u'{},{}'.format(p[1], p[3])


def p_ap0(p):
    ''' ap : a
           | a adv
           | a pp
           | a ap
           | adv
    '''
    p[0] = p[1]


def p_ap1(p):
    ''' ap : adv a
    '''
    p[0] = p[2]


def p_ap2(p):
    ''' ap : ap AND ap
           | ap AND pp
           | pp AND ap
           | pp AND pp
    '''
    # p[0] = '{}, {}'.format(p[1], p[3])
    p[0] = (p[1], p[3])

# def p_ap3(p):
#     ''' ap :
#     '''
# p[0] = ','.join((p[1], p[3]))
#     p[0] = '{}, {}'.format(p[1], p[3])


def p_adv(p):
    ''' adv : quantifier
    '''


def p_quantifier(p):
    ''' quantifier : MOLTO
                   | DI MOLTO
                   | POCO
                   | UN POCO
                   | POCO A POCO
                   | POI A POI
                   | BEN
                   | PIU
                   | PIO
                   | MENO
                   | MEZZO
                   | QUASI
                   | DOPPIO
                   | TROPPO
                   | SEMPRE
                   | MA NOT TROPPO
                   | COMMA MA NOT TROPPO
    '''


def p_quantifier1(p):
    ''' quantifier : quantifier quantifier
    '''


def p_pp(p):
    ''' pp : p np
    '''
    p[0] = score.ConstantTempoDirection('{}_{}'.format(p[1], p[2]).lower())


def p_p(p):
    ''' p : CON
          | ALLA
          | DEL
          | SOTTO
    '''
    p[0] = p[1]


def p_a_dynamic_tempo(p):
    ''' a : ACCELERANDO
          | RITENUTO
          | RITARDANDO
          | RALLENTANDO
          | SMORZANDO
    '''
    p[0] = score.DynamicTempoDirection(p.slice[-1].type.lower())

def p_a_dynamic_loudness(p):
    ''' a : CRESCENDO
          | RINFORZANDO
          | CALANDO
          | DIMINUENDO
    '''
    p[0] = score.DynamicLoudnessDirection(p.slice[-1].type.lower())
    

def p_a_constant_loudness(p):
    ''' a : PPP
          | PP
          | P
          | MP
          | MF
          | F
          | FF
          | FFF
          | DOLCE
          | GRAZIOSO
    '''
    p[0] = score.ConstantLoudnessDirection(p.slice[-1].type.lower())

def p_a_constant_tempo(p):
    ''' a : ESPRESSIVO
          | PIACEVOLE
          | ANDANTE
          | ADAGIO
          | ANIMATO
          | STRETTO
          | LARGHETTO
          | LENTO
          | PESANTE
          | TRANQUILO
          | RISVEGLIATO
          | PRESTO
          | LARGO
          | PIANO
          | VIVACE
          | ALLEGRO
          | ARIOSO
          | GRAVE
          | DOLENTE
          | ASSAI
          | SEMPLICE
          | AGITATO
          | ENERGICO
          | CANTABILE
          | VIVO
          | COMODO
          | SOSTENUTO
          | MAESTOSO
          | MANCANDO
          | MODERATO
          | MOSSO
          | APPASSIONATO
          | PRESTISSIMO
          | RISOLUTO
          | LEGGIERO
          | MARCATO
          | LEGATO
          | LIGATO
          | TENUTO
          | SECCO
          | TENERAMENTE
          | STRINGENDO
          | VIVACISSIMAMENTE
    '''
    # print(p.slice[-1].type,)
    # p[0] = p[1]
    # print()
    p[0] = score.ConstantTempoDirection(p.slice[-1].type.lower())


def p_np(p):
    '''  np : noun
            | quantifier noun
    '''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = p[2]


def p_noun(p):
    '''  noun : BRIO
              | ESPRESSIONE
              | MARCIA
              | VARIATION
              | MOTO
              | FUOCO
              | ANIMA
              | FORZA
              | VOCE
              | TEMPO
              | SENTIMENTO
              | MOVIMENTO
    '''
    p[0] = p[1]


def p_temporeset(p):
    ''' temporeset : TEMPO ROMAN_NUMBER
                   | TEMPO PRIMO
                   | A TEMPO
                   | IN TEMPO
    '''
    p[0] = score.ResetTempoDirection('_'.join(p[1:]))


def p_error(p):
    if p is not None:
        LOGGER.warning("Syntax error at '{}' in '{}'"
                       .format(p.value, p.lexer.lexdata))

def parse_words(words):
    lwords = words.lower()
    parser = yacc.yacc()
    try:
        r = parser.parse(lwords)
    except TokenizeException as e:
        LOGGER.warning('Cannot tokenize: "{}"'.format(lwords))
        r = None

    if isinstance(r, score.Direction):
        r.raw = lwords
    elif r is None:
        LOGGER.warning('Cannot not convert "{}" into a direction'.format(lwords))
        r = score.Words(words)
    return r


def main():
    parser = argparse.ArgumentParser(
        description="""
        Test a list of annotations from a file (one per line) against the annotation parser
        """)

    parser.add_argument("annotations", help="annotations file")
    parser.add_argument("--test", type=str, help="test string")

    args = parser.parse_args()

    annotations = [l.strip() for l in open(args.annotations).read().decode('utf8').split('\n')
                   if not l.strip().startswith('#')]
    tokens = AnnotationTokenizer.tokens
    parser = yacc.yacc()
    # if args.test:
    for words in annotations:
        ltest = words.lower()
        print('')
        print(('parsing "{}"'.format(ltest).encode('utf8')))
        # lex.input(ltest)
        # toktypes = []
        # while True:
        #     tok = lex.token()
        #     if not tok:
        #         break
        #     toktypes.append(tok.type)

        # print(' '.join(toktypes))
        try:
            r = parser.parse(ltest)
        except TokenizeException as e:
            print(('cannot tokenize: "{}"'.format(
                ltest.lower()).encode('utf8')))
            continue
        if isinstance(r, score.Direction):
            r.raw = words
        print(('correctly parsed into: "{}"'.format(r).encode('utf8')))
        # return True

tokenizer = AnnotationTokenizer()
tokens = tokenizer.tokens

if __name__ == '__main__':
    pass
