import numpy as np

fn = './data_examples/Three-Part_Invention_No_13_(fragment).xml'


from partitura.musicxml import xml_to_notearray
from partitura.music_utils import voice_estimation

if __name__ == '__main__':
    notearray = xml_to_notearray(fn)
    v_notearray = voice_estimation(notearray)
