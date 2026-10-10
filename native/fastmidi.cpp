// Fast MIDI parsing backend for partitura.
//
// This extension does not depend on (or vendor) symusic itself. It is built
// directly on top of "minimidi" (https://github.com/lzqlzzq/minimidi,
// vendored at 3rdparty/minimidi), the small, dependency-free, header-only
// MIDI parser that symusic itself is built on. The note on/off pairing and
// tempo-map logic below intentionally mirrors
// partitura/io/importmidi.py:load_performance_midi and load_score_midi so
// that the mido and minimidi backends of each produce equivalent output.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "minimidi/MiniMidi.hpp"

namespace py = pybind11;
namespace mm = minimidi;

namespace {

// ---------------------------------------------------------------------
// Output record layouts (registered as numpy structured dtypes below).
// ---------------------------------------------------------------------

struct NoteRecord {
    int32_t midi_pitch;
    double note_on;
    int64_t note_on_tick;
    double note_off;
    int64_t note_off_tick;
    int32_t track;
    int32_t channel;
    int32_t velocity;
};

struct ControlRecord {
    double time;
    int64_t time_tick;
    int32_t number;
    int32_t value;
    int32_t track;
    int32_t channel;
};

struct ProgramRecord {
    double time;
    int64_t time_tick;
    int32_t program;
    int32_t track;
    int32_t channel;
};

struct TimeSignatureRecord {
    double time;
    int64_t time_tick;
    int32_t beats;
    int32_t beat_type;
    int32_t track;
};

struct KeySignatureRecord {
    double time;
    int64_t time_tick;
    int32_t fifths;
    int32_t mode;  // 0 = major, 1 = minor
    int32_t track;
};

// Used only by the score-import path (load_score_midi's backend='minimidi'),
// which works entirely in ticks and has no use for a seconds field.
struct TempoRecord {
    int64_t tick;
    int32_t mpq;
};

// One MIDI event handed to write_midi() by save_performance_midi's
// backend='minimidi' path, already converted to ticks. The meaning of
// `a`/`b` depends on `kind` (see EventKind).
struct EventRecord {
    int64_t tick;
    int32_t kind;
    int32_t channel;
    int32_t a;
    int32_t b;
};

enum EventKind : int32_t {
    kNoteOn = 0,      // a = pitch, b = velocity
    kNoteOff = 1,     // a = pitch, b = velocity
    kControl = 2,     // a = controller number, b = value
    kProgram = 3,     // a = program
    kTimeSig = 4,     // a = numerator, b = denominator
    kKeySig = 5,      // a = fifths, b = 1 if minor else 0
    kRawMeta = 6,     // a = index into write_midi()'s raw_metas
};

// ---------------------------------------------------------------------
// Tempo map: converts absolute MIDI ticks to seconds via a single,
// order-independent, piecewise-linear breakpoint table built from every
// SetTempo event in the file (equivalent in spirit to
// importmidi.py:adjust_time, but O(log n) per lookup instead of O(n)).
// ---------------------------------------------------------------------

class TempoMap {
public:
    TempoMap(std::vector<std::pair<uint32_t, uint32_t>> tempo_events, uint32_t ppq, uint32_t default_mpq)
        : ppq_(ppq) {
        std::stable_sort(tempo_events.begin(), tempo_events.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        breakpoints_.push_back({0, default_mpq, 0.0});
        for (const auto& [tick, mpq] : tempo_events) {
            if (tick == 0) {
                // A tempo event at tick 0 overrides the caller-supplied default.
                breakpoints_[0].mpq = mpq;
                continue;
            }
            if (mpq == breakpoints_.back().mpq) continue;
            const auto& prev = breakpoints_.back();
            const double seconds = prev.seconds + delta_seconds(tick - prev.tick, prev.mpq);
            breakpoints_.push_back({tick, mpq, seconds});
        }
    }

    double tick_to_seconds(uint32_t tick) const {
        auto it = std::upper_bound(
            breakpoints_.begin(),
            breakpoints_.end(),
            tick,
            [](uint32_t t, const Breakpoint& bp) { return t < bp.tick; }
        );
        --it;
        return it->seconds + delta_seconds(tick - it->tick, it->mpq);
    }

private:
    struct Breakpoint {
        uint32_t tick;
        uint32_t mpq;
        double seconds;
    };

    double delta_seconds(uint32_t delta_ticks, uint32_t mpq) const {
        return static_cast<double>(delta_ticks) * static_cast<double>(mpq) / (static_cast<double>(ppq_) * 1e6);
    }

    std::vector<Breakpoint> breakpoints_;
    uint32_t ppq_;
};

// ---------------------------------------------------------------------
// Per-track (or merged) aggregation, mirroring the message loop in
// importmidi.py:load_performance_midi (note_hash-based on/off pairing,
// control/program/time-signature/key-signature collection).
// ---------------------------------------------------------------------

struct TrackResult {
    std::vector<NoteRecord> notes;
    std::vector<ControlRecord> controls;
    std::vector<ProgramRecord> programs;
    std::vector<TimeSignatureRecord> time_signatures;
    std::vector<KeySignatureRecord> key_signatures;
    uint32_t dropped_note_offs = 0;
};

inline int note_hash(int channel, int pitch) { return channel * 128 + pitch; }

// A single flattened, already tick-ordered message from one (or, when
// merging, several concatenated and re-sorted) original track(s).
struct FlatMessage {
    uint32_t tick;
    int32_t track;
    mm::Message<mm::container::SmallBytes> msg;
};

TrackResult pair_events(const std::vector<FlatMessage>& events, const TempoMap& tempo_map) {
    TrackResult result;

    struct SoundingNote {
        uint32_t tick;
        uint8_t velocity;
    };
    std::unordered_map<int, SoundingNote> sounding_notes;

    for (const auto& fm : events) {
        const auto& msg = fm.msg;
        const auto type = msg.type();

        if (type == mm::MessageType::NoteOn || type == mm::MessageType::NoteOff) {
            const uint8_t channel = msg.channel();
            uint8_t pitch, velocity;
            if (type == mm::MessageType::NoteOn) {
                const auto& note_on = msg.cast<mm::NoteOn>();
                pitch = note_on.pitch();
                velocity = note_on.velocity();
            } else {
                const auto& note_off = msg.cast<mm::NoteOff>();
                pitch = note_off.pitch();
                velocity = note_off.velocity();
            }

            const int key = note_hash(channel, pitch);
            const bool is_note_on = (type == mm::MessageType::NoteOn) && velocity > 0;
            const bool is_note_off = (type == mm::MessageType::NoteOff) || (type == mm::MessageType::NoteOn && velocity == 0);

            if (is_note_on) {
                sounding_notes[key] = {fm.tick, velocity};
            } else if (is_note_off) {
                auto it = sounding_notes.find(key);
                if (it == sounding_notes.end()) {
                    ++result.dropped_note_offs;
                    continue;
                }
                result.notes.push_back(NoteRecord{
                    static_cast<int32_t>(pitch),
                    tempo_map.tick_to_seconds(it->second.tick),
                    static_cast<int64_t>(it->second.tick),
                    tempo_map.tick_to_seconds(fm.tick),
                    static_cast<int64_t>(fm.tick),
                    fm.track,
                    static_cast<int32_t>(channel),
                    static_cast<int32_t>(it->second.velocity),
                });
                sounding_notes.erase(it);
            }
        } else if (type == mm::MessageType::ControlChange) {
            const auto& cc = msg.cast<mm::ControlChange>();
            result.controls.push_back(ControlRecord{
                tempo_map.tick_to_seconds(fm.tick),
                static_cast<int64_t>(fm.tick),
                static_cast<int32_t>(cc.control_number()),
                static_cast<int32_t>(cc.control_value()),
                fm.track,
                static_cast<int32_t>(msg.channel()),
            });
        } else if (type == mm::MessageType::ProgramChange) {
            const auto& pc = msg.cast<mm::ProgramChange>();
            result.programs.push_back(ProgramRecord{
                tempo_map.tick_to_seconds(fm.tick),
                static_cast<int64_t>(fm.tick),
                static_cast<int32_t>(pc.program()),
                fm.track,
                static_cast<int32_t>(msg.channel()),
            });
        } else if (type == mm::MessageType::Meta) {
            const auto& meta = msg.cast<mm::Meta>();
            const auto meta_type = meta.meta_type();
            if (meta_type == mm::MetaType::TimeSignature) {
                const auto& ts = msg.cast<mm::TimeSignature>();
                result.time_signatures.push_back(TimeSignatureRecord{
                    tempo_map.tick_to_seconds(fm.tick),
                    static_cast<int64_t>(fm.tick),
                    static_cast<int32_t>(ts.numerator()),
                    static_cast<int32_t>(ts.denominator()),
                    fm.track,
                });
            } else if (meta_type == mm::MetaType::KeySignature) {
                const auto& ks = msg.cast<mm::KeySignature>();
                result.key_signatures.push_back(KeySignatureRecord{
                    tempo_map.tick_to_seconds(fm.tick),
                    static_cast<int64_t>(fm.tick),
                    static_cast<int32_t>(ks.key()),
                    static_cast<int32_t>(ks.tonality()),
                    fm.track,
                });
            }
            // SetTempo is handled separately (tempo map is built up front);
            // all other meta message types (track name, lyrics, markers,
            // ...) are not collected by the fast path -- they are exposed
            // as `meta_other` by the mido backend "just for completeness"
            // and are not used elsewhere in partitura.
        }
    }

    // Match importmidi.py's per-track note ordering (onset, pitch, offset,
    // channel); track is constant within a single pair_events() call.
    std::stable_sort(result.notes.begin(), result.notes.end(), [](const NoteRecord& a, const NoteRecord& b) {
        return std::tie(a.note_on_tick, a.midi_pitch, a.note_off_tick, a.channel)
             < std::tie(b.note_on_tick, b.midi_pitch, b.note_off_tick, b.channel);
    });

    return result;
}

template<typename Record>
py::array to_structured_array(const std::vector<Record>& records) {
    py::array_t<Record> arr(static_cast<py::ssize_t>(records.size()));
    if (!records.empty()) {
        std::memcpy(arr.mutable_data(), records.data(), records.size() * sizeof(Record));
    }
    return arr;
}

py::dict track_result_to_dict(const TrackResult& tr) {
    py::dict d;
    d["notes"] = to_structured_array(tr.notes);
    d["controls"] = to_structured_array(tr.controls);
    d["programs"] = to_structured_array(tr.programs);
    d["time_signatures"] = to_structured_array(tr.time_signatures);
    d["key_signatures"] = to_structured_array(tr.key_signatures);
    d["dropped_note_offs"] = tr.dropped_note_offs;
    return d;
}

}  // namespace

namespace {

// Extracts the text of the first TrackName meta message in `track`, or ""
// if there is none -- matches mido's MidiTrack.name property, which
// load_score_midi's backend='mido' path uses for part naming.
std::string extract_track_name(const mm::Track<>& track) {
    for (const auto& msg : track) {
        if (msg.type() == mm::MessageType::Meta) {
            const auto& meta = msg.cast<mm::Meta>();
            if (meta.meta_type() == mm::MetaType::TrackName) {
                const auto bytes = meta.meta_value();
                return std::string(bytes.begin(), bytes.end());
            }
        }
    }
    return "";
}

py::dict parse_midi(const std::string& path, double default_bpm, bool merge_tracks) {
    auto midi = mm::MidiFile<>::from_file(path);
    const uint32_t ppq = midi.ticks_per_quarter();
    const uint32_t default_mpq = static_cast<uint32_t>(std::llround(60.0 * 1e6 / default_bpm));

    // Global, order-independent tempo map built from every SetTempo event
    // in the file (see TempoMap docstring above for how this differs from
    // the mido backend's sequential-per-track tempo_changes list).
    std::vector<std::pair<uint32_t, uint32_t>> tempo_events;
    for (const auto& track : midi.tracks) {
        for (const auto& msg : track) {
            if (msg.type() == mm::MessageType::Meta) {
                const auto& meta = msg.cast<mm::Meta>();
                if (meta.meta_type() == mm::MetaType::SetTempo) {
                    tempo_events.emplace_back(msg.time, msg.cast<mm::SetTempo>().tempo());
                }
            }
        }
    }

    // Exposed as-is (tick, mpq) for load_score_midi's backend='minimidi'
    // path, which works entirely in ticks and builds its own Tempo
    // objects directly -- unlike the TempoMap below, no seconds
    // conversion or default_bpm-driven fallback is meaningful here.
    std::vector<TempoRecord> tempo_records;
    tempo_records.reserve(tempo_events.size());
    for (const auto& [tick, mpq] : tempo_events) {
        tempo_records.push_back(TempoRecord{static_cast<int64_t>(tick), static_cast<int32_t>(mpq)});
    }

    const TempoMap tempo_map(std::move(tempo_events), ppq, default_mpq);

    py::list track_list;
    // Populated only in the non-merged branch (load_score_midi never
    // merges tracks): every original track's time/key signatures and
    // name, regardless of whether that track also has notes/controls/
    // programs -- unlike `track_list` above, a note-less "meta" track's
    // time/key signatures still matter to the score-import path (they
    // become *global* time/key signatures there), so these must not be
    // filtered the way `track_list` is.
    std::vector<TimeSignatureRecord> all_time_signatures;
    std::vector<KeySignatureRecord> all_key_signatures;
    py::list track_names;

    if (merge_tracks) {
        std::vector<FlatMessage> events;
        for (size_t i = 0; i < midi.tracks.size(); ++i) {
            for (const auto& msg : midi.tracks[i]) {
                events.push_back(FlatMessage{msg.time, 0, msg});
            }
        }
        std::stable_sort(events.begin(), events.end(), [](const FlatMessage& a, const FlatMessage& b) {
            return a.tick < b.tick;
        });
        const TrackResult tr = pair_events(events, tempo_map);
        if (!tr.notes.empty() || !tr.controls.empty() || !tr.programs.empty()) {
            py::dict d = track_result_to_dict(tr);
            d["track_index"] = 0;
            track_list.append(d);
        }
    } else {
        for (size_t i = 0; i < midi.tracks.size(); ++i) {
            std::vector<FlatMessage> events;
            events.reserve(midi.tracks[i].size());
            for (const auto& msg : midi.tracks[i]) {
                events.push_back(FlatMessage{msg.time, static_cast<int32_t>(i), msg});
            }
            const TrackResult tr = pair_events(events, tempo_map);
            if (!tr.notes.empty() || !tr.controls.empty() || !tr.programs.empty()) {
                py::dict d = track_result_to_dict(tr);
                d["track_index"] = static_cast<int>(i);
                track_list.append(d);
            }
            all_time_signatures.insert(all_time_signatures.end(), tr.time_signatures.begin(), tr.time_signatures.end());
            all_key_signatures.insert(all_key_signatures.end(), tr.key_signatures.begin(), tr.key_signatures.end());
            track_names.append(extract_track_name(midi.tracks[i]));
        }
    }

    py::dict out;
    out["ppq"] = static_cast<int>(ppq);
    out["num_tracks"] = static_cast<int>(midi.tracks.size());
    out["tracks"] = track_list;
    out["tempos"] = to_structured_array(tempo_records);
    out["all_time_signatures"] = to_structured_array(all_time_signatures);
    out["all_key_signatures"] = to_structured_array(all_key_signatures);
    out["track_names"] = track_names;
    return out;
}

using EventArray = py::array_t<EventRecord, py::array::c_style | py::array::forcecast>;

// Serializes already tick-converted, insertion-ordered events into a
// Standard MIDI File. minimidi's to_bytes() stable-sorts each track by tick
// and appends the end-of-track marker, so the caller only has to emit events
// in the order they should appear among events sharing a tick.
//
// `tracks` holds one EventRecord array per output track; the global tempo is
// written at tick 0 of the first track. `raw_metas` holds fully encoded meta
// events, [type, length, payload...] (i.e. mido's MetaMessage.bytes()
// without the leading 0xFF), referenced by kRawMeta events.
py::bytes write_midi(
    const std::vector<EventArray>& tracks,
    const std::vector<std::string>& raw_metas,
    int ppq,
    uint32_t mpq,
    int midi_type
) {
    mm::Tracks<> out_tracks;
    out_tracks.reserve(tracks.size());

    for (size_t ti = 0; ti < tracks.size(); ++ti) {
        mm::Track<> track;
        auto& msgs = track.messages;
        const EventRecord* events = tracks[ti].data();
        const size_t n = static_cast<size_t>(tracks[ti].size());
        msgs.reserve(n + 1);

        if (ti == 0) msgs.push_back(mm::SetTempo<>(0, mpq));

        for (size_t i = 0; i < n; ++i) {
            const EventRecord& e = events[i];
            const uint32_t tick = static_cast<uint32_t>(e.tick);
            const uint8_t channel = static_cast<uint8_t>(e.channel);
            switch (e.kind) {
                case kNoteOn:
                    msgs.push_back(mm::NoteOn<>(tick, channel, static_cast<uint8_t>(e.a), static_cast<uint8_t>(e.b)));
                    break;
                case kNoteOff:
                    msgs.push_back(mm::NoteOff<>(tick, channel, static_cast<uint8_t>(e.a), static_cast<uint8_t>(e.b)));
                    break;
                case kControl:
                    msgs.push_back(mm::ControlChange<>(tick, channel, static_cast<uint8_t>(e.a), static_cast<uint8_t>(e.b)));
                    break;
                case kProgram:
                    msgs.push_back(mm::ProgramChange<>(tick, channel, static_cast<uint8_t>(e.a)));
                    break;
                case kTimeSig:
                    msgs.push_back(mm::TimeSignature<>(tick, static_cast<uint8_t>(e.a), static_cast<uint8_t>(e.b)));
                    break;
                case kKeySig:
                    msgs.push_back(mm::KeySignature<>(tick, static_cast<int8_t>(e.a), static_cast<uint8_t>(e.b)));
                    break;
                case kRawMeta: {
                    const std::string& raw = raw_metas.at(static_cast<size_t>(e.a));
                    const auto* begin = reinterpret_cast<const uint8_t*>(raw.data());
                    msgs.emplace_back(tick, static_cast<uint8_t>(0xFF), begin, begin + raw.size());
                    break;
                }
                default:
                    throw std::invalid_argument("write_midi: unknown event kind " + std::to_string(e.kind));
            }
        }
        out_tracks.push_back(std::move(track));
    }

    const auto format = midi_type == 0 ? mm::MidiFormat::SingleTrack : mm::MidiFormat::MultiTrack;
    mm::MidiFile<> midi(std::move(out_tracks), format, 0, static_cast<uint16_t>(ppq));
    const auto bytes = midi.to_bytes();
    return py::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

}  // namespace

PYBIND11_MODULE(_minimidi, m) {
    // pybind11 >= 3 registers structured numpy dtypes at runtime rather than
    // via a template specialization, so these calls must live inside the
    // module init function rather than at namespace scope.
    PYBIND11_NUMPY_DTYPE(NoteRecord, midi_pitch, note_on, note_on_tick, note_off, note_off_tick, track, channel, velocity);
    PYBIND11_NUMPY_DTYPE(ControlRecord, time, time_tick, number, value, track, channel);
    PYBIND11_NUMPY_DTYPE(ProgramRecord, time, time_tick, program, track, channel);
    PYBIND11_NUMPY_DTYPE(TimeSignatureRecord, time, time_tick, beats, beat_type, track);
    PYBIND11_NUMPY_DTYPE(KeySignatureRecord, time, time_tick, fifths, mode, track);
    PYBIND11_NUMPY_DTYPE(TempoRecord, tick, mpq);
    PYBIND11_NUMPY_DTYPE(EventRecord, tick, kind, channel, a, b);

    m.doc() =
        "Fast MIDI reading and writing backend for partitura, built on the "
        "vendored minimidi library (see native/fastmidi.cpp). Serves the "
        "backend='minimidi' paths of load_performance_midi (seconds-based), "
        "load_score_midi (ticks-based) and save_performance_midi.";
    m.def(
        "write_midi",
        &write_midi,
        py::arg("tracks"),
        py::arg("raw_metas"),
        py::arg("ppq"),
        py::arg("mpq"),
        py::arg("midi_type"),
        "Serialize per-track EventRecord arrays (ticks, insertion-ordered) "
        "into the bytes of a Standard MIDI File."
    );
    m.def(
        "parse_midi",
        &parse_midi,
        py::arg("path"),
        py::arg("default_bpm") = 120.0,
        py::arg("merge_tracks") = false,
        "Parse a MIDI file into numpy structured arrays suitable for bulk "
        "partitura.performance.PerformedPart / partitura.score.Part "
        "construction."
    );
}
