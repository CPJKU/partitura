import argparse
import os
import subprocess
from pydub import AudioSegment

'''
This file is a modified version of midi2audio.py from https://github.com/bzamecnik/midi2audio
Author: Bohumír Zámečník (@bzamecnik)
License: MIT, see the LICENSE file
'''

__all__ = ['FluidSynth']

DEFAULT_SOUND_FONT = '~/.fluidsynth/default_sound_font.sf2'
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_GAIN = 0.2

class FluidSynth():
    def __init__(self, sound_font=DEFAULT_SOUND_FONT, sample_rate=DEFAULT_SAMPLE_RATE, gain=DEFAULT_GAIN):
        self.sample_rate = sample_rate
        self.sound_font = os.path.expanduser(sound_font)
        self.gain = gain

    def midi_to_audio(self, midi_file: str, audio_file: str, verbose=True):
        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
        
        # Convert MIDI to WAV
        subprocess.call(
            ['fluidsynth', '-ni', '-g', str(self.gain), self.sound_font, midi_file, '-F', audio_file, '-r', str(self.sample_rate)], 
            stdout=stdout
        )

        # Convert WAV to MP3
        mp3_path = audio_file.replace('.wav', '.mp3')
        AudioSegment.from_wav(audio_file).export(mp3_path, format="mp3")
        
        # Delete the temporary WAV file
        os.remove(audio_file)

    def play_midi(self, midi_file):
        subprocess.call(['fluidsynth', '-i', '-g', str(self.gain), self.sound_font, midi_file, '-r', str(self.sample_rate)])

def parse_args(allow_synth=True):
    parser = argparse.ArgumentParser(description='Convert MIDI to audio via FluidSynth')
    parser.add_argument('midi_file', metavar='MIDI', type=str)
    if allow_synth:
        parser.add_argument('audio_file', metavar='AUDIO', type=str, nargs='?')
    parser.add_argument('-s', '--sound-font', type=str,
        default=DEFAULT_SOUND_FONT,
        help='path to a SF2 sound font (default: %s)' % DEFAULT_SOUND_FONT)
    parser.add_argument('-r', '--sample-rate', type=int, nargs='?',
        default=DEFAULT_SAMPLE_RATE,
        help='sample rate in Hz (default: %s)' % DEFAULT_SAMPLE_RATE)
    return parser.parse_args()

def main(allow_synth=True):
    args = parse_args(allow_synth)
    fs = FluidSynth(args.sound_font, args.sample_rate)
    if allow_synth and args.audio_file:
        fs.midi_to_audio(args.midi_file, args.audio_file)
    else:
        fs.play_midi(args.midi_file)

def main_play():
    """
    A method for the `midiplay` entry point. It omits the audio file from args.
    """
    main(allow_synth=False)

if __name__ == '__main__':
    main()