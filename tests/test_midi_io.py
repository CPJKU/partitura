


import ipdb
import os
import partitura
from partitura import midi


output_fn = './tests/data_examples/test_midi_output.mid'


# print(os.cwd())
sp = midi.load_midi('./tests/data_examples/test_basic_midi.mid')


midi.write_midi(output_fn, sp)


ipdb.set_trace()




