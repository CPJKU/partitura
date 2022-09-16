import partitura as pt
import numpy as np
from fractions import Fraction

from partitura.utils import generic
try:
    import music21 as m21
except ImportError:
    m21 = None

def load_music21(mei_path: str) -> list:
    """
    Loads a Mei score from path and returns a list of Partitura.Part

    Parameters
    ----------
    mei_path : str
        The path to an MEI score.

    Returns
    -------
    part_list : list
        A list of Partitura Part or GroupPart Objects.
    """
    if m21 is None:
        raise ImportError("Music21 must be installed for this function to work")

    parser = M21Parser(mei_path)
    # create parts from the specifications in the mei
    parser.create_parts()
    # fill parts with the content from the mei
    parser.fill_parts()
    return parser.parts


class M21Parser:

    def __init__(self, m21_score):
        self.m21_score = m21_score
        self.ppq = self.find_ppq()

    def create_parts(self):
        # create the part list
        self.parts = [pt.score.Part(m21_part.id, m21_part.partName, quarter_duration=self.ppq) for m21_part in self.m21_score.parts]
        
    def fill_parts(self):
        # fill parts with the content of the score
        for part_idx, m21_part, pt_part in enumerate(zip(self.m21_score.parts, self.parts)):
            # fill notes
            self.fill_part_notes(m21_part,pt_part, part_idx)
            # fill rests
            self.fill_part_rests(self, m21_part,pt_part, part_idx)

    def fill_part_rests(self, m21_part,pt_part, part_idx):
        for rest in m21_part.recurse().getElementsByClass(m21.note.Rest):
            rest = pt.score.Rest(
                id=rest.id,
                voice=self.find_voice(rest),
                staff=part_idx + 1,
                symbolic_duration=rest.duration.type,
                articulations=None,
            )
            # add rest to the part
            position = int(rest.getOffsetInHierarchy(self.m21_score)*self.ppq)
            duration = int(rest.duration.quarterLength*self.ppq)
            pt_part.add(rest, position, position + duration)

    def fill_part_notes(self, m21_part,pt_part, part_idx):
        for generic_note in m21_part.recurse().notes:
            for pitch in generic_note.pitches:
                if generic_note.duration.is_grace:
                    note = pt.score.GraceNote(
                        grace_type="acciaccatura" if generic_note.duration.slash else "appoggiatura",
                        step=pitch.step,
                        octave=pitch.octave,
                        alter=pitch.accidental.alter if pitch.accidental is not None else None,
                        id=pitch.id,
                        voice=self.find_voice(generic_note),
                        staff=part_idx + 1,
                        symbolic_duration=generic_note.duration.type,
                        articulations=None,  # TODO : add articulation
                    )
                else:
                    note= pt.score.Note(
                        step= pitch.step,
                        octave= pitch.octave,
                        alter=pitch.accidental.alter if pitch.accidental is not None else None,
                        id=pitch.id,
                        voice= self.find_voice(generic_note),
                        staff= part_idx + 1,
                        symbolic_duration=generic_note.duration.type,
                        articulations=None,  # TODO : add articulation
                    )
                position = int(generic_note.getOffsetInHierarchy(self.m21_score)*self.ppq)
                duration = int(generic_note.duration.quarterLength*self.ppq)
                pt_part.add(note, position, position + duration)
                
    def find_voice(self, m21_general_note):
        """Return the voice for an music21 general note"""
        return 1 if type(m21_general_note.activeSite) is m21.stream.Measure else m21_general_note.activeSite.id

    def find_staff(self, m21_general_note):
        """Return the staff for an music21 general note"""
        return [e for e in m21_general_note.containerHierarchy() if type(e) is m21.score.Staff][0].


    def find_ppq(self):
        """Finds the ppq """
        
        def fractional_gcd(inputs):
            denoms = np.array([Fraction(e).denominator for e in inputs],dtype = int)
            denoms_lcm = np.lcm.reduce(denoms)
            return Fraction(1,denoms_lcm)

        durs = [el.duration.quarterLength for el in self.m21_score.recurse().getElementsByClass('Music21Object') if el.duration.quarterLength!=0 ]
        return fractional_gcd(durs).denominator


        


        


