import numpy as np
import re
from collections import OrderedDict

class WormFile:
    VERSION = '1.05'
    attr_types = {
        'WORM Version':   str,
        'AudioFile':      str,
        'Composer':       str,
        'Piece':  str,
        'Key':    str,
        'Indication': str,    
        'Performer':      str,
        'YearOfRecording':    int,
        'FrameLength':    float,
        'LoudnessUnits':  str,
        'Axis': str,
        'Smoothing': str,
        'StartBarNumber': int,
        'TempoLate':      int,
        'BeatLevel':      str,
        'TrackLevel':     float,
        'Upbeat': float,
        'BeatsPerBar':    int,
        'Length': int
        }

    def __getitem__(self,v):
        return self.header[v]

    def __init__(self,wormfile=None):
        self.header = None
        self.data = None

        if wormfile:
            self.read_file(wormfile)

    def read_file(self,filename):
        self.filename = filename
        header_pat = re.compile('(?P<attribute>[^:]+):(?P<value>.+)$',re.I)

        self.header = OrderedDict()

        with open(filename, 'r') as f:

            line = f.readline()

            while line:

                m = header_pat.match(line)

                if m:
                    attr = m.group('attribute').strip()

                    try:
                        value = self.attr_types[attr](m.group('value').strip())
                    except KeyError:
                        print(('Error: Unknown header attribute at line "{0}"'.format(line)))
                        return False
                    except ValueError:
                        print(('Error: Invalid attribute value at line "{0}"'.format(line)))
                        return False

                    self.header[attr] = value

                    if attr == 'Length':
                        break

                else:

                    print(('Error reading WORM file header at line "{0}"'.format(line)))
                    return False

                line = f.readline()

            self.data = np.genfromtxt(f)

    @property
    def beatlevel_float(self):
        """
        Return the beat level value as a float (rather than a string
        depicting a rational)

        :returns: a float

        """

        parts = self.header['BeatLevel'].split('/')
        return float(parts[0])/float(parts[1])

    def write_file(self,filename):
        """
        Write the current header and data to a file

        :param filename: the name of the file to write to
        
        """
        
        with open(filename, 'w') as f:
            tab_width = np.max([len(k) for k in list(self.header.keys())])
            for k,v in list(self.header.items()):
                f.write('{0}:\t{1}\n'.format(k, v).encode('utf8').expandtabs(tab_width+2))
            np.savetxt(f, self.data, fmt ='%f %f %f %d')
