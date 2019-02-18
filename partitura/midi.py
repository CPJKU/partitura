#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sys import stderr
from collections import OrderedDict
import logging
from functools import reduce
import numpy as np
from scipy.interpolate import interp1d

from mxm.midifile.MidiOutStream import MidiOutStream
from mxm.midifile.MidiInFile import MidiInFile
from mxm.midifile.MidiOutFile import MidiOutFile

LOGGER = logging.getLogger(__name__)


def partition(func, iterable):
    """
    Return a dictionary containing the equivalence classes (actually bags)
    of iterable, partioned according to func. The value of a key k is the 
    list of all elements e from iterable such that k = func(e)
    """
    result = defaultdict(list)
    for v in iterable:
        result[func(v)].append(v) 
    return result


def tic2ms(t, tempo, div):
    """
    Convert `t` from midi tics to milliseconds, given `tempo`
    (specified as period in microseconds), and beat division `div`

    Parameters
    ----------

    t :

    tempo :

    div :

    Returns
    -------
    number

    """

    return tempo * t / (1000.0 * div)


def tic2beat(t, div):
    """
    Convert `t` from midi tics to beats given beat division `div`

    Parameters
    ----------

    t : number

    div : number

    Returns
    -------
    number
    """

    return t / float(div)


######################### Midi Ontology
class MidiTimeEvent(object):
    """
    abstract class for Midi events that have a time stamp.

    Parameters
    ----------
    time : number
        the time in MIDI units.

    """
    def __init__(self, time):
        self.time = time

    #def __str__(self):
    #    return 'asdf'

    def _send(self, midi):
        pass


class MidiMetaEvent(MidiTimeEvent):
    """
    abstract class for Midi meta events
    """
    def __init__(self, time):
        MidiTimeEvent.__init__(self, time)


class EndOfTrackEvent(MidiMetaEvent):
    def __init__(self, time):
        MidiMetaEvent.__init__(self, time)

    def _send(self, midi):
        midi.update_time(self.time, relative=0)
        midi.end_of_track()


class TextEvent(MidiMetaEvent):
    """
    Parameters
    ----------
    time :

    text :
    """

    def __init__(self, time, text):
        MidiMetaEvent.__init__(self, time)
        self.text = text

    def _send(self, midi):
        midi.update_time(self.time, relative=0)
        midi.text(self.text.encode("utf8"))

    def __str__(self):
        return '{0} Meta Text "{1}"'.format(self.time, self.text)


class MarkerEvent(MidiMetaEvent):
    def __init__(self, time, text):
        MidiMetaEvent.__init__(self, time)
        self.text = text

    def _send(self, midi):
        midi.update_time(self.time, relative=0)
        midi.marker(self.text.encode("utf8"))

    def __str__(self):
        return '{0} Meta Marker "{1}"'.format(self.time, self.text)


class TrackNameEvent(MidiMetaEvent):
    """

    Parameters
    ----------
    time :

    name : str
    """

    def __init__(self, time, name):
        MidiMetaEvent.__init__(self, time)
        self.name = name

    def _send(self, midi):
        midi.update_time(self.time, relative=0)
        midi.sequence_name(self.name.encode("utf8"))

    def __str__(self):
        return '{0} Meta TrkName "{1}"'.format(self.time, self.name)


class InstrumentNameEvent(MidiMetaEvent):
    """

    Parameters
    ----------
    time :

    name :

    """

    def __init__(self, time, name):
        MidiMetaEvent.__init__(self, time)
        self.name = name

    def _send(self, midi):
        midi.update_time(self.time, relative=0)
        midi.instrument_name(self.name.encode("utf8"))

    def __str__(self):
        return '{0} Meta InstrName "{1}"'.format(self.time, self.name)


class TempoEvent(MidiMetaEvent):
    def __init__(self, time, val):
        MidiMetaEvent.__init__(self, time)
        self.tempo = val

    def _send(self, midi):
        midi.update_time(self.time, relative=0)
        midi.tempo(self.tempo)


class KeySigEvent(MidiMetaEvent):
    def __init__(self, time, key, scale):
        MidiMetaEvent.__init__(self, time)
        self.key = key
        self.scale = scale

    def _send(self, midi):
        midi.update_time(self.time, relative=0)
        midi.key_signature(self.key, self.scale)


class TimeSigEvent(MidiMetaEvent):
    """
    Parameters
    ----------

    TO DO: check the docstring!

    time : number

    num : number
        the numerator of the time signature

    den : number
        the denominator of the time signature. CHECK: Is this simply
        the note value?! Could be power of two thing where e.g. 3 means
        eigth note because 8 = 2^3.

    metro : number, optional. Default: 24
        the number of MIDI Clock ticks in a metronome click.

    thirtysecs : number, optional. Default: 8
        the number of 32nd notes in a "MIDI quarter note", i.e. in 24 MIDI
        clock ticks. CHECK THIS!
    """
    def __init__(self, time, num, den, metro=24, thirtysecs=8):
        MidiMetaEvent.__init__(self, time)
        self.num = num
        self.den = den
        self.metro = metro
        self.thirtysecs = thirtysecs

    def _send(self, midi):
        midi.update_time(self.time, relative=0)
        midi.time_signature(self.num, self.den, self.metro, self.thirtysecs)


class MidiEvent(MidiTimeEvent):
    """
    abstract class for Midi events, having a time stamp and a channel

    TO DO: check and fix docstring; check for channel number bug ?

    Parameters
    ----------
    time : number

    channel : number
        the MIDI channel number. Should be in the range 0-15. ????
    """

    def __init__(self, time, channel):
        MidiTimeEvent.__init__(self, time)
        self.channel = channel

    def _getChannelForSend(self):
        """
        Ugly workaround for apparent bug in mxm code:
        interprets channels as one lower as supposed to be
        """

        # DEBUGGING
#        import pdb
#
#        if type(self.channel) != int:
#            pdb.set_trace()

        # NOTE: the problem seems to occur on a PatchChangeEvent!
        # There seems to be something wrong with the setting of its
        # channel number? CHECK this!

        return self.channel - 1


class PatchChangeEvent(MidiEvent):
    """
    switch to a certain MIDI patch. A patch is basically equivalent to
    one particular instrument (one particular sound).

    Parameters
    ----------
    time : number

    channel : number
        the MIDI channel number.

    patch : number
        the MIDI patch (program) number to change to.
    """

    def __init__(self, time, channel, patch):
        MidiEvent.__init__(self, time, channel)
        self.patch = patch

    def _send(self, midi):
        midi.update_time(self.time, relative=0)
        midi.patch_change(self._getChannelForSend(), self.patch)

    def __str__(self):
        return '{0} PrCh ch={1} p={2}'.format(self.time, self.channel,
                                              self.patch)


class PitchBendEvent(MidiEvent):
    """

    """

    def __init__(self, time, channel, value):
        MidiEvent.__init__(self, time, channel)
        self.value = value

    def _send(self, midi):
        midi.update_time(self.time, relative=0)
        midi.pitch_bend(self._getChannelForSend(), self.value)


class NoteOnOffEvent(MidiEvent):
    """
    super class for `NoteOnEvent` and `NoteOffEvent`.

    Parameters
    ----------
    time :

    ch :

    note :

    velocity :
        MIDI velocity value. Should be in the range 0-127.
    """
    def __init__(self, time, ch, note, velocity):
        MidiEvent.__init__(self, time, ch)
        self.note = note
        self.velocity = velocity


class NoteOnEvent(NoteOnOffEvent):
    """
    represents a 'note on' command.
    """
    def _send(self, midi):
        midi.update_time(self.time, relative=0)
        midi.note_on(channel=self._getChannelForSend(),
                     note=self.note,
                     velocity=self.velocity)

    def __str__(self):
        return '{0} On ch={1} n={2} v={3}'.format(self.time, self.channel,
                                                  self.note, self.velocity)


class NoteOffEvent(NoteOnOffEvent):
    """
    represents a 'note off' command.
    """
    def _send(self, midi):
        midi.update_time(self.time, relative=0)
        midi.note_off(channel=self._getChannelForSend(),
                      note=self.note,
                      velocity=self.velocity)

    def __str__(self):
        return '{0} Off ch={1} n={2} v={3}'.format(self.time, self.channel,
                                                   self.note, self.velocity)


class AftertouchEvent(NoteOnOffEvent):
    """

    """
    def _send(self, midi):
        midi.update_time(self.time, relative=0)
        midi.aftertouch(channel=self._getChannelForSend(),
                        note=self.note,
                        velocity=self.velocity)


class ControllerEvent(MidiEvent):
    """

    Parameters
    ----------
    time :

    ch :

    controller :

    value :
    """
    def __init__(self, time, ch, controller, value):
        MidiEvent.__init__(self, time, ch)
        self.controller = controller
        self.value = value

    def __str__(self):
        return '%s Par %s %s %s' % (self.time, self.channel,
                                    self.controller, self.value)

    def _send(self, midi):
        midi.update_time(self.time, relative=0)
        midi.continuous_controller(channel=self._getChannelForSend(),
                                   controller=self.controller,
                                   value=self.value)


class MidiNote:
    """
    Class that bundles midi on and off events, mostly to have a
    convenient way of getting note durations from midi data

    Parameters
    ----------
    onset_event :

    offset_event :

    """
    def __init__(self, onset_event, offset_event):
        self._onset = onset_event
        self._offset = offset_event

    @classmethod
    def from_data(cls, onset, note, duration, channel, velocity):
        "Initialize MidiNote from note information"
        return cls(NoteOnEvent(time=onset, ch=channel, note=note, velocity=velocity),
                   NoteOffEvent(time=onset + duration, ch=channel, note=note, velocity=0))

    @property
    def onset(self):
        """
        the onset time of the note
        """
        return self._onset.time

    @onset.setter
    def onset(self, time):
        self._onset.time = time

    @property
    def offset(self):
        """
        the onset time of the note
        """
        return self._offset.time

    @offset.setter
    def offset(self, time):
        self._offset.time = time

    @property
    def duration(self):
        """
        the onset time of the note
        """
        return self.offset - self.onset

    @property
    def channel(self):
        """
        the Midi channel of the note
        """
        return self._onset.channel

    @channel.setter
    def channel(self, ch):
        self._onset.channel = ch
        self._offset.channel = ch

    @property
    def note(self):
        """
        the Midi pitch of the note
        """
        return self._onset.note

    @note.setter
    def note(self, note):
        self._onset.note = note
        self._offset.note = note

    @property
    def velocity(self):
        """
        the Midi velocity of the note
        """
        return self._onset.velocity

    @velocity.setter
    def velocity(self, vel):
        self._onset.velocity = vel

    def __str__(self):
        return '%s %s %s %s %s' % (self.onset, self.offset,
                                   self.channel, self.note, self.velocity)


## some events still missing, like aftertouch etc
class MidiFile:
    """

    TO DO: fix docstring

    Parameters
    ----------
    in_file : None OR str, optional. Default: None

    zero_vel_on_is_off : optional. Default: None

    Attributes
    ----------

    """

    def __init__(self, in_file=None, zero_vel_on_is_off=False):
        self.header = None
        self.tracks = []
        self.zero_vel_on_is_off = zero_vel_on_is_off
        if in_file is not None:
            self.read_file(in_file)

    def summarize(self):
        """
        summarize the contents of the MidiFile object, by printing
        header information, and summarizing information about each
        track

        """
        out = [self.header.summarize()]
        for t in self.tracks:
            out.append(t.summarize())
        return '\n'.join(out)

    def midi_ticks2seconds(self, times, default_bpm = 120):
        """
        Convert a sequence of positions expressed in MIDI ticks to
        their corresponding times in seconds, taking into account the
        tempo events in the MIDI file. If the first tempo event occurs
        later than time 0, that tempo event is duplicated at time
        0. If the file contains no tempo events at all, the specified
        `default_bpm` value is assumed from time 0.

        Parameters
        ----------
        times : iterable
            A sequence of positions, in MIDI ticks
    
        default_bpm : float (default: 120)

            The tempo value to use when no tempo events are present in
            the file

        Returns
        -------
    
        ndarray
            An array containing the positions in seconds

        """

        return interp1d(*self._time_map(default_bpm), bounds_error = False)(times)
        
    def seconds2midi_ticks(self, times, default_bpm = 120):
        """
        Convert a sequence of positions expressed in seconds to their
        corresponding MIDI ticks, taking into account the tempo events
        in the MIDI file. If the first tempo event occurs later than
        time 0, that tempo event is duplicated at time 0. If the file
        contains no tempo events at all, the specified `default_bpm`
        value is assumed from time 0.

        Parameters
        ----------
    
        times : iterable
            A sequence of positions, in seconds
    
        default_bpm : float (default: 120)

            The tempo value to use when no tempo events are present in
            the file

        Returns
        -------
    
        ndarray
            An array containing the positions in MIDI ticks

        default_tempo : number, optional. Default: 1000000
            the default tempo. The default value is 10^6 microseconds,
            i.e. 1 second.

        Returns
        -------
        numpy array
            timestamps in seconds. The array should be as long as `times`,
            i.e. the same number of timestamps should be returned as were
            given.
        """

        # return interp1d(*(self._time_map(default_bpm)[::-1]), bounds_error = False, fill_value=0.0)(times)

        time_map = self._time_map(default_bpm)
        ticks = time_map[0]
        seconds = time_map[1]
        return interp1d(seconds, ticks, bounds_error=True)(times)

    def _time_map(self, default_bpm = 100):
        """
        Return a map of midi tick times and second times for each
        tempo event in the file

        """

        first_time = 0
        last_time = self.last_time_of(MidiTimeEvent) # self.last_off

        events = [(e.time, e.tempo) for t in self.tracks
                  for e in t.get_events(TempoEvent) if first_time <= e.time < last_time]

        if len(events) == 0:
            init_tempo = 10**6*(60./default_bpm)
            last_tempo = init_tempo
        else:
            init_tempo = events[0][1]
            last_tempo = events[-1][1]

        events.insert(0, (first_time, init_tempo))
        events.append((last_time, last_tempo))

        time_tempo = np.array(list(OrderedDict([e for e in events]).items()))
        
        return (time_tempo[:, 0],
                (np.r_[0, np.cumsum(time_tempo[:-1, 1] * np.diff(time_tempo[:, 0]))] / 
                 (10**6 * float(self.header.time_division))))

    def first_time_of(self, cls):
        try:
            return np.min([t.first_time_of(cls) for t in self.tracks])
        except ValueError:
            return np.inf

    def last_time_of(self, cls):
        try:
            return np.max([t.last_time_of(cls) for t in self.tracks])
        except ValueError:
            return -np.inf

    @property
    def first_on(self):
        return self.first_time_of(NoteOnEvent)

    @property
    def last_off(self):
        return self.last_time_of(NoteOffEvent)

    def get_track(self, i = 0):
        """
        get a specific track, or the first, if no index is specified

        Parameters
        ----------
        i : number, optional. Default: 0
            the index of the track (a MidiTrack object) to return

        Returns
        -------
        MidiTrack object

#        :param i: the index of the track to return
#
#        :returns: a MidiTrack object
        """

        return self.tracks[i]

    def replace_track(self, i, track):
        """
        replace an existing track with the given track

        Parameters
        ----------
        i : number
            the index of the track to be replaced

        track : MidiTrack object
            a MidiTrack object to replace the i-th track
        """

        #self.tracks[n] = track
        self.tracks[i] = track

    def add_track(self, track):
        """
        add a track after (possibly already) existing tracks. A MidiTrack
        object is added to `tracks`, which is a list.

        Parameters
        ----------
        track: MidiTrack object
            track to be added to `tracks` (list).
        """
        self.tracks.append(track)

    def read_file(self, filename):
        """
        instantiate the MidiObject with data read from a file

        Parameters
        ----------
        filename : str
            the file to read from

#        :param filename: the file to read from
        """
        self.midi_in = MidiInFile(_MidiHandler(self, self.zero_vel_on_is_off), filename)
        ## header and tracks get instantiated through the midi event handler
        self.midi_in.read()

    def correct_header(self):
        n = len(self.tracks)
        if n > 1:
            if self.header.format == 0:
                LOGGER.warning('Multiple tracks found, changing Midi file format from type 0 to type 1')
                self.header.format = 1
        if self.header.number_of_tracks is not n:
            if self.header.number_of_tracks is not None:
                # wrong nr of tracks specified, warn
                LOGGER.warning(('Header specification does not match '
                                'number of tracks ({0} vs. {1}), correcting header'
                                '').format(self.header.number_of_tracks, n))
            self.header.number_of_tracks = n

    def write_file(self, filename):
        """
        write the MidiObject to a file

        Parameters
        ----------
        filename : str
            the file to write the data to (may include the path to
            a folder/directory).
        """

        # DEBUGGING
        count = 0
        for track in self.tracks:
            #track.summarize()    # this doesn't really do anything?
            myTrack = self.get_track()
            myTrack.summarize()
            #print(('info on track number {0}').format(count))
            count = count + 1

        midi = MidiOutFile(filename)
        self.correct_header()
        self.header._send(midi)

        # `self.tracks` is a list of MidiTrack objects that were
        # previously added (via the 'add_track()' method)

        # DEBUGGING
        count = 0
        for track in self.tracks:
            track.summarize()
            track._send(midi)
            #print(('successfully wrote track number {0}').format(count))
            count = count + 1

        # DEBUGGING TG
        #[track._send(midi) for track in self.tracks]    # SOMEWHERE HERE is the problem
        midi.eof()    # NOTE: where is this defined?



    # OBSOLETE METHODS:

    def compute_time(self,time,track=0,default_tempo=1000000):
        """
        OBSOLETE, USE `midi_ticks2seconds`

        compute the time in seconds for `time` (in midi units), taking
        into account the beat division, and tempo events in `track`

        :param time: a time value in midi units
        :param track: the index of the midi track to get tempo events from
        :param default_tempo: the default tempo to use in the absence of tempo events 

        :returns: the time in seconds that corresponds to `time`

        """

        try:
            return self.compute_times([time],track=track,default_tempo=default_tempo)[0]
        except IndexError:
            print('No Tempo events found in file, could not compute time.')

    def compute_times(self, times, track = 0, default_tempo = 1000000):
        """
        OBSOLETE, USE `midi_ticks2seconds`

        compute the time in seconds for all values in `time` (in midi
        units), taking into account the beat division, and tempo
        events in `track`

        :param times: a list of times in midi units
        :param track: the index of the midi track to get tempo events from
        :param default_tempo: the default tempo to use in the absence of tempo events 

        :returns: a list of times in seconds (one value for each element in `times`)

        """


        try:
            events = self.get_track(track).get_events(TempoEvent)
        except:
            print(('midi file has no track %d'.format(track)))
            return False
        if len(events) > 0 and events[0].time > 0:
            ## assume default tempo until first tempo event
            events.insert(0,TempoEvent(0, default_tempo))
        mtime = max(times)
        time_tempo = np.array(list(OrderedDict([(e.time, e.tempo) for e in events if e.time < mtime]).items()))
        tempo_times = np.column_stack((time_tempo[:, 0],
                                      np.r_[0, np.cumsum(time_tempo[:-1, 1] * np.diff(time_tempo[:, 0]))],
                                      time_tempo[:, 1]))
        j = 0
        result = [0]*len(times)
        for i in np.argsort(np.array(times)):
            while j < tempo_times.shape[0] and tempo_times[j, 0] > times[i]:
                j = j+1
            result[i] = (tempo_times[j-1, 1] + (times[i] - tempo_times[j-1, 0]) * tempo_times[j-1,  2])/\
                        (10**6 * float(self.header.time_division))
        return np.array(result)

class MidiHeader:
    """
    TO DO: check and fix docstring

    Parameters
    ----------
    format : number

    number_of_tracks : number ????, optional. Default: None

    time_division : number, optional. Default: 480

    """

    def __init__(self, format, number_of_tracks=None, time_division=480):
        self.format = format
        self.number_of_tracks = number_of_tracks
        self.time_division = time_division

    def __str__(self):
        return 'MFile %d %d %d\n' % (self.format, self.number_of_tracks,
                                     self.time_division)

    def _send(self, midi):
        midi.header(format=self.format,
                    nTracks=self.number_of_tracks,
                    division=self.time_division)

    def summarize(self):
        out = "Midi Header\n"
        out += "        Format: %s\n" % self.format
        out += " Nr. of Tracks: %s\n" % self.number_of_tracks
        out += "      Division: %s\n" % self.time_division
        return out


class MidiTrack:
    """

    Parameters
    ----------
    events : list, optional. Default: empty list []

    Attributes
    ----------
    first_on :

    def on_offs :

    on_off_eq_classes :

    notes :

    homophonic_slices :

    """

    def __init__(self, events=[]):
        self.events = events

    def summarize(self):
        """
        print information about the track, such as its name, the
        number of events/metaevents, channels, and the types of events

        """
        events = partition(lambda x: isinstance(x, MidiMetaEvent), self.get_events())
        ev = list(set([e.__class__.__name__ for e in events.get(False, [])]))
        mev = list(set([e.__class__.__name__ for e in events.get(True, [])]))
        midiChannels = list(set([e.channel for e in events.get(False, [])]))
        tname = self.get_events(TrackNameEvent)
        out = "Midi Track\n"
        if len(tname) > 0:
            out += "        Track Name: %s\n" % tname[0].name
        out += "     Nr. of Events: %s\n" % len(events.get(False, []))
        out += " Nr. of MetaEvents: %s\n" % len(events.get(True, []))
        out += "    Event Channels: %s\n" % midiChannels
        out += "            Events: %s\n" % ev
        out += "       Meta Events: %s\n" % mev
        return out

    def add_event(self, event):
        """
        add `event` to the track

        """
        self.events.append(event)

    def close(self):
        """
        sort events and add an EndOfTrack event to the track if it's missing

        """
        self.sort_events()
        endTime = int(self.get_events()[-1].time if len(self.get_events()) > 0 else 0)
        if len(self.get_events()) == 0 or not isinstance(self.get_events()[-1], EndOfTrackEvent):
            self.add_event(EndOfTrackEvent(endTime))

    def sort_events(self):
        """
        Sort the events in the track according to their time stamp,
        this is done in place ?
        TODO: check
        """
        self.events.sort(key=lambda x: x.time)

    def get_events(self, event_type=None, filters=None):
        """
        get all events of type event_type from track that
        return True on all filter predicates

        Parameters
        ----------
        event_type : ????, Default: None

        filters : ????, Default: None

        Returns
        -------
        result :
        """

        result = self.events
        if not event_type is None:
            result = [e for e in result if isinstance(e, event_type)]
        if not filters is None:
            result = [e for e in result if all([f(e) for f in filters])]

        return result


    def first_time_of(self, cls):
        try:
            return np.min([x.time for x in self.get_events(cls)])
        except ValueError:
            return np.inf

    def last_time_of(self, cls):
        try:
            return np.max([x.time for x in self.get_events(cls)])
        except ValueError:
            return -np.inf

    @property
    def first_on(self):
        return self.first_time_of(NoteOnEvent)

    @property
    def last_off(self):
        return self.first_time_of(NoteOffEvent)

        
    def _send(self,midi):
        midi.start_of_track()
        self.close()
        [e._send(midi) for e in self.get_events()]

    @property
    def on_offs(self):
        """
        all NoteOnEvents and NoteOffEvents from track

        """
        onoffs = self.get_events(NoteOffEvent)+self.get_events(NoteOnEvent)
        onoffs.sort(key=lambda x: x.time)
        return onoffs

    @property
    def on_off_eq_classes(self):
        """
        all NoteOnEvents and NoteOffEvents from track, grouped by time

        """
        return partition(lambda x: x.time, self.on_offs)

    @property
    def notes(self):
        """
        a list of MidiNotes

        """
        onoffs = self.on_off_eq_classes
        eventTimes = list(onoffs.keys())
        eventTimes.sort()
        acc = []
        sounding = {}
        errors = 0
        for t in eventTimes:
            onOffsByChannel = partition(lambda x: x.channel, onoffs[t])
            for ch, evs in list(onOffsByChannel.items()):
                onOffsByNote = partition(lambda x: x.note, evs)
                for note, v in list(onOffsByNote.items()):
                    isSounding = (ch, note) in sounding
                    onOff = partition(lambda x: isinstance(x, NoteOnEvent) and x.velocity > 0, v)
                    ons = onOff.get(True, [])
                    offs = onOff.get(False, [])
                    ons = partition(lambda x: x.velocity == 0, ons)
                    onsZeroVel = ons.get(True, [])
                    ons = ons.get(False, [])
                    nOns = len(ons)
                    nOnsZeroVel = len(onsZeroVel)
                    nOffs = len(offs) - nOnsZeroVel

                    if nOffs >= 0: # there's an off for every 0vel On
                        if nOnsZeroVel > 0:
                            acc.append(MidiNote(onsZeroVel[0], offs[0]))
                        else:
                            pass # there are no ons 0vel
                    else: # there's 0vel ons without off, treat as note off
                        if isSounding:
                            on = sounding[(ch, note)]
                            acc.append(MidiNote(on, onsZeroVel[0])) ## any off will do
                            del sounding[(ch, note)]
                        else:
                            ## Warn spurious note on with 0 vel
                            pass
                    if nOffs > 0:
                        if isSounding:
                            on = sounding[(ch, note)]
                            acc.append(MidiNote(on, offs[0])) ## any off will do
                            del sounding[(ch, note)]
                            isSounding = False
                        else:
                            ## Warn spurious note off
                            pass
                    if nOns > 0:
                        if isSounding:
                            ## Warn implicit note off by new note on
                            on = sounding[(ch, note)]
                            acc.append(MidiNote(on, ons[0])) ## any off will do
                            del sounding[(ch, note)]
                        sounding[(ch, note)] = ons[0]

        acc.sort(key=lambda x: x.onset)
        return acc

    @property
    def homophonic_slices(self):
        """
        Return the homophonic slicing (Pickens, 2001) of the
        piano-roll, as a list of (on, off, (pitch_0, ..., pitch_N))
        triples

        """
        onoffs = self.on_off_eq_classes
        times = list(onoffs.keys())
        times.sort()
        sounding = set()
        acc = []
        prev_time = 0
        for t in times:
            if not sounding == {}:
                acc.append((prev_time, t, [i[1] for i in sounding]))
                prev_time = t
            onsOffs = partition(lambda x: isinstance(x, NoteOnEvent), onoffs[t])
            ons = set([(e.channel, e.note) for e in onsOffs.get(True, [])])
            offs = set([(e.channel, e.note) for e in onsOffs.get(False, [])])

            on = ons - offs - sounding
            off = offs - ons - sounding
            snd = sounding - ons - offs
            onOff = set.intersection(ons, offs) - sounding
            offSnd = set.intersection(sounding, offs) - ons
            onSnd = set.intersection(ons ,sounding) - offs
            #onOffSnd = list(set.intersection(set.intersection(ons,offs),snds))
            onOffSnd = set.intersection(onOff, offSnd)
            for i in list(on):
                sounding.add(i) # add to sounding
            for i in list(offSnd):
                sounding.remove(i) # remove from sounding
            for i in list(onOff):
                pass #print('Warning, ignoring zero duration (grace) note at time %d' % t)
            if len(onSnd) > 1:
                print(('Warning, ignoring unexpected note on at time %d' % t))
                print(onSnd)
            if len(off) > 1:
                print(('Warning, ignoring unexpected %s note off at time %d' % (off, t)))

        acc.sort(key=lambda x: x[0])
        return acc


#################################################
## some convenience functions for midi files

def convert_midi_to_type_0(input_midifile):
    """
    return a new MidiFile with all tracks merged into a single one,
    effectively converting a type 1 midi file into a type 0 file

    :param input_midifile: MidiFile object to convert

    :returns: MidiFile object of type 0

    """
    output_midifile = MidiFile()
    header = MidiHeader(format=0,
                        number_of_tracks=1,
                        time_division=input_midifile.header.time_division)
    output_midifile.header = header
    track = MidiTrack()
    allEvents = partition(
        lambda x: isinstance(x, EndOfTrackEvent),
        reduce(lambda x, y: x+y, [tr.events for tr in input_midifile.tracks]))

    eot_events, non_eot_events = [allEvents[i] for i in (True, False)]
    eot_events.sort(key=lambda x: x.time)
    track.events = non_eot_events + [eot_events[-1]]

    output_midifile.add_track(track)

    return output_midifile


def convert_midifile_to_type_0(input_filename, output_filename):
    """
    convert a type 1 midi file into a type 0 midi file by merging
    all tracks into a single one

    :param input_midifile: filename of midifile to convert
    :param output_midifile: filename of midifile to write converted midi to

    """
    convert_midi_to_type_0(MidiFile(input_filename)).write_file(output_filename)


class _MidiHandler(MidiOutStream):
    """
    event handler that constructs a MidiFile object (with MidiHeader
    and MidiTrack objects) from a midifile; this is for internal use
    of the MidiFile class

    Parameters
    ----------
    midiFile :

    zero_vel_on_is_off :
    """

    def __init__(self, midiFile, zero_vel_on_is_off):
        MidiOutStream.__init__(self)
        self.midiFile = midiFile
        self.zero_vel_on_is_off = zero_vel_on_is_off
        self.events = []

    def channel_message(self, message_type, channel, data):
        stderr.write('ch msg, type: %s, ch: %s, data: %s' % (message_type,
                                                             channel, data))

    def pitch_bend(self, channel, value):
        channel += 1
        self.events.append(PitchBendEvent(self.abs_time(), channel, value))

    def start_of_track(self, n_track=0):
        self.events = []

    def end_of_track(self):
        self.events.append(EndOfTrackEvent(self.abs_time()))
        self.midiFile.tracks.append(MidiTrack(self.events))

    def continuous_controller(self, channel, controller, value):
        channel += 1
        self.events.append(ControllerEvent(self.abs_time(), channel, controller, value))

    def note_on(self, channel=1, note=0x40, velocity=0x40):
        channel += 1
        if self.zero_vel_on_is_off and velocity == 0:
            self.note_off(channel, note, velocity)
        else:
            self.events.append(NoteOnEvent(self.abs_time(), channel, note, velocity))

    def note_off(self, channel=1, note=0x40, velocity=0x40):
        channel += 1
        self.events.append(NoteOffEvent(self.abs_time(), channel, note, velocity))

    def header(self, format=0, nTracks=1, division=480):
        self.midiFile.header = MidiHeader(format=format, number_of_tracks=nTracks,
                                          time_division=division)

    def text(self, text):
        self.events.append(TextEvent(self.abs_time(), text))

    def marker(self, text):
        self.events.append(MarkerEvent(self.abs_time(), text))

    def sequence_name(self, text):
        self.events.append(TrackNameEvent(self.abs_time(), text))

    def instrument_name(self, text):
        self.events.append(InstrumentNameEvent(self.abs_time(), text))

    def key_signature(self, sf, mi):
        self.events.append(KeySigEvent(self.abs_time(), key=sf, scale=mi))

    def time_signature(self, nn, dd, cc, bb):
        self.events.append(TimeSigEvent(self.abs_time(), num=nn, den=dd,
                                        metro=cc, thirtysecs=bb))

    def tempo(self, value):
        self.events.append(TempoEvent(self.abs_time(), value))

    def patch_change(self, channel, patch):
        channel += 1
        self.events.append(PatchChangeEvent(self.abs_time(), channel, patch))

    # unsupported events

    def sysex_event(self, data):
        pass

    def device_name(self, *args, **kwargs):
        pass

    def program_name(self, *args, **kwargs):
        pass




if __name__ == '__main__':
    pass
