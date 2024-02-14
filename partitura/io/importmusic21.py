import partitura as pt
import numpy as np
from fractions import Fraction

try:
    import music21 as m21
    from music21.stream import Score as M21Score
except ImportError:
    m21 = None

    class M21Score(object):
        pass


def load_music21(m21_score: M21Score) -> pt.score.Score:
    """
    Loads a music21 score object and returns a Partitura Score object.

    Parameters
    ----------
    m21_score : :class:`music21.stream.Score`
        The music21 score object, produced for example with the `music21.converter.parse()` function.

    Returns
    -------
    scr: :class:`partitura.score.Score`
        A `Score` object
    """
    if m21 is None:
        raise ImportError("Music21 must be installed for this function to work")

    parser = M21Parser(m21_score)
    # create parts from the specifications in the music21 object
    parser.create_parts()
    # fill parts with the content from the music21 object
    parser.fill_parts()
    # create the score
    doc_name = (
        m21_score.metadata.title if m21_score.metadata.title is not None else "score"
    )
    scr = pt.score.Score(
        id=doc_name,
        partlist=parser.parts,
    )
    return scr


class M21Parser:
    """
    Class to parse a music21 score object and create a partitura score object from it.
    """

    def __init__(self, m21_score):
        self.m21_score = m21_score
        self.ppq = self.find_ppq()

    def create_parts(self):
        # create the part list
        self.parts = [
            pt.score.Part(m21_part.id, m21_part.partName, quarter_duration=self.ppq)
            for m21_part in self.m21_score.parts
        ]

    def fill_parts(self):
        # fill parts with the content of the score
        for part_idx, (m21_part, pt_part) in enumerate(
            zip(self.m21_score.parts, self.parts)
        ):
            # fill notes
            self.fill_part_notes(m21_part, pt_part, part_idx)
            # fill rests
            self.fill_part_rests(m21_part, pt_part, part_idx)
            # fill key signatures
            self.fill_part_ks(m21_part, pt_part, part_idx)
            # fill time signatures
            self.fill_part_ts(m21_part, pt_part, part_idx)
            # fill with clefs
            self.fill_part_clefs(m21_part, pt_part, part_idx)
        # handle ties
        self.tie_notes(self.m21_score, self.parts)

    def fill_part_rests(self, m21_part, pt_part, part_idx):
        for m21_rest in m21_part.recurse().getElementsByClass(m21.note.Rest):
            pt_rest = pt.score.Rest(
                id=m21_rest.id,
                voice=self.find_voice(m21_rest),
                staff=part_idx + 1,
                symbolic_duration=m21_rest.duration.type,
                articulations=None,
            )
            # add rest to the part
            position = int(
                m21_rest.getOffsetBySite(self.m21_score.recurse()) * self.ppq
            )
            duration = int(m21_rest.duration.quarterLength * self.ppq)
            pt_part.add(pt_rest, position, position + duration)

    def fill_part_notes(self, m21_part, pt_part, part_idx):
        for generic_note in m21_part.recurse().notes:
            for i_pitch, pitch in enumerate(generic_note.pitches):
                if generic_note.duration.isGrace:
                    note = pt.score.GraceNote(
                        grace_type=(
                            "acciaccatura"
                            if generic_note.duration.slash
                            else "appoggiatura"
                        ),
                        step=pitch.step,
                        octave=pitch.octave,
                        alter=(
                            pitch.accidental.alter
                            if pitch.accidental is not None
                            else None
                        ),
                        # id="{}_{}".format(generic_note.id, i_pitch),
                        id=generic_note.id,
                        voice=self.find_voice(generic_note),
                        staff=part_idx + 1,
                        symbolic_duration=generic_note.duration.type,
                        articulations=None,  # TODO : add articulation
                    )
                else:
                    note = pt.score.Note(
                        step=pitch.step,
                        octave=pitch.octave,
                        alter=(
                            pitch.accidental.alter
                            if pitch.accidental is not None
                            else None
                        ),
                        # id="{}_{}".format(generic_note.id, i_pitch),
                        id=generic_note.id,
                        voice=self.find_voice(generic_note),
                        staff=part_idx + 1,
                        symbolic_duration=generic_note.duration.type,
                        articulations=None,  # TODO : add articulation
                    )
                position = int(
                    generic_note.getOffsetInHierarchy(self.m21_score) * self.ppq
                )
                duration = int(generic_note.duration.quarterLength * self.ppq)
                pt_part.add(note, position, position + duration)

    def fill_part_ts(self, m21_part, pt_part, part_idx):
        """Fills the part with time signatures"""
        for ts in m21_part.recurse().getElementsByClass(m21.meter.TimeSignature):
            new_time_signature = pt.score.TimeSignature(ts.numerator, ts.denominator)
            position = int(ts.getOffsetInHierarchy(self.m21_score) * self.ppq)
            pt_part.add(new_time_signature, position)

    def fill_part_ks(self, m21_part, pt_part, part_idx):
        """Fills the part with key signatures"""
        for ks in m21_part.recurse().getElementsByClass(m21.key.KeySignature):
            new_key_signature = pt.score.KeySignature(ks.sharps, None)
            position = int(ks.getOffsetInHierarchy(self.m21_score) * self.ppq)
            pt_part.add(new_key_signature, position)

    def tie_notes(self, m21_score, pt_part_list):
        """Fills the part with ties"""
        # create a dict of id : note, to speed up search
        all_notes = [
            note for part in pt_part_list for note in part.iter_all(cls=pt.score.Note)
        ]
        all_notes_dict = {note.id: note for note in all_notes}
        for m21_note in m21_score.recurse().getElementsByClass(m21.note.Note):
            if m21_note.tie is not None:
                if m21_note.tie.type == "start":
                    start_id = m21_note.id
                    end_id = m21_note.next("Note").id
                elif m21_note.tie.type == "stop":
                    pass  # music21 don't require the stop to be set
                elif m21_note.tie.type == "continue":
                    start_id = m21_note.id
                    end_id = m21_note.next("Note").id

                # set tie prev and tie next in partitura note objects
                all_notes_dict[start_id].tie_next = all_notes_dict[end_id]
                all_notes_dict[end_id].tie_prev = all_notes_dict[start_id]

    def fill_part_clefs(self, m21_part, pt_part, part_idx):
        """Fills the part with clefs"""
        for m21_clef in m21_part.recurse().getElementsByClass(m21.clef.Clef):
            pt_clef = pt.score.Clef(
                int(part_idx) + 1,
                m21_clef.sign,
                int(m21_clef.line),
                m21_clef.octaveChange,
            )
            position = int(m21_clef.getOffsetInHierarchy(self.m21_score) * self.ppq)
            pt_part.add(pt_clef, position)

    def find_voice(self, m21_general_note):
        """Return the voice for an music21 general note"""
        return (
            1
            if type(m21_general_note.activeSite) is m21.stream.Measure
            else m21_general_note.activeSite.id
        )

    def find_ppq(self):
        """Finds the ppq"""

        def fractional_gcd(inputs):
            denoms = np.array([Fraction(e).denominator for e in inputs], dtype=int)
            denoms_lcm = np.lcm.reduce(denoms)
            return Fraction(1, denoms_lcm)

        durs = [
            el.duration.quarterLength
            for el in self.m21_score.recurse().getElementsByClass("Music21Object")
            if el.duration.quarterLength != 0
        ]
        return fractional_gcd(durs).denominator
