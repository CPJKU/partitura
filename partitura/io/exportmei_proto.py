import partitura
import partitura.score as score
from lxml import etree
from partitura.utils.generic import partition
from partitura.utils.music import estimate_symbolic_duration
import sys


part = score.Part("P0","Test")

# test case blues lick
# part.set_quarter_duration(0,10)
# part.add(score.KeySignature(-3,"minor"),start=0)
# part.add(score.TimeSignature(6,8),start=0)
# part.add(score.Clef(sign="F",line=4, octave_change=0, number=1),start=0)
# part.add(score.Clef(sign="G",line=2, octave_change=0, number=2),start=0)
# n0 = score.Note(id="n0",step="C",octave=2,voice=1, staff=1)
# n1 =score.Note(id="n1",step="E",octave=2,voice=1, staff=1, alter=-1)
# n2 =score.Note(id="n2",step="D",octave=2,voice=1, staff=1)
# part.add(n0,start=0,end=5)
# part.add(n1,start=5,end=10)
# part.add(n2,start=10,end=15)
# n0q = score.Note(id="n0q",step="G",octave=2,voice=1, staff=1)
# n1q =score.Note(id="n1q",step="G",octave=2,voice=1, staff=1)
# n2q =score.Note(id="n2q",step="G",octave=2,voice=1, staff=1)
# part.add(n0q,start=0,end=5)
# part.add(n1q,start=5,end=10)
# part.add(n2q,start=10,end=15)
# part.add(score.Note(id="n3",step="C",octave=2,voice=1, staff=1),start=15,end=40)
# n0s2 = score.Note(id="n0s2",step="C",octave=4,voice=1, staff=2)
# n1s2 =score.Note(id="n1s2",step="E",octave=4,voice=1, staff=2, alter=-1)
# n2s2 =score.Note(id="n2s2",step="D",octave=4,voice=1, staff=2)
# part.add(n0s2,start=0,end=5)
# part.add(n1s2,start=5,end=10)
# part.add(n2s2,start=10,end=15)
# part.add(score.Slur(n0s2,n2s2),start=0)
# part.add(score.Note(id="n3s2",step="C",octave=4,voice=1, staff=2),start=15,end=40)


# # testing crossing measures and tieing notes together
# part.set_quarter_duration(0,16)
# part.add(score.KeySignature(-3,"minor"),start=0)
# part.add(score.TimeSignature(4,4),start=0)
# part.add(score.Clef(sign="F",line=4, octave_change=0, number=1),start=0)
#
# part.add(score.Rest(id="r0",staff=1, voice=1),start=0,end=1)
# part.add(score.Note(id="n0",step="C",octave=2, staff=1, voice=1),start=1,end=1+3*16+4+1)
# part.add(score.Note(id="n2",step="G",octave=2, staff=1, voice=1),start=1,end=1+3*16+4+1)
# part.add(score.Note(id="n3",step="E",octave=3, staff=1, voice=1),start=1,end=1+3*16+4+1)
#
# score.add_measures(part)

# using this feature?
# making ties then becomes about looking at the tiegroup tag of notes
# which is fine, however the sum of powers of 2 idea seems better than estimating symbolic duration
# without this feature, notes crossing measure boundaries have to be handled
#score.tie_notes(part)

part = partitura.load_musicxml("../../tests/data_examples/Three-Part_Invention_No_13_(fragment).xml", force_note_ids=True)




# create MEI file from this ;D

def addChild(parent,childName):
    return etree.SubElement(parent,childName)

def setAttributes(elem, *list_attrib_val):
    for pair in list_attrib_val:
        elem.set(pair[0],str(pair[1]))

nameSpace = "http://www.music-encoding.org/ns/mei"

xmlIdString = "{http://www.w3.org/XML/1998/namespace}id"



mei = etree.Element("mei")

meiHead=addChild(mei,"meiHead")
music = addChild(mei,"music")



meiHead.set("xmlns",nameSpace)
fileDesc = addChild(meiHead,"fileDesc")
titleStmt=addChild(fileDesc,"titleStmt")
pubStmt=addChild(fileDesc,"pubStmt")
title=addChild(titleStmt,"title")
title.set("type","main")
title.text="TEST"

body = addChild(music,"body")
mdiv=addChild(body,"mdiv")
mei_score=addChild(mdiv,"score")

scoreDef = addChild(mei_score,"scoreDef")

keySig = next(part.iter_all(score.KeySignature),None)

if keySig!=None:
    key = keySig.name
    pname = key[0]
    mode = "major"

    if len(key)==2:
        mode="minor"

    fifths = str(abs(keySig.fifths))

    if keySig.fifths<0:
        fifths+="f"
    else:
        fifths+="s"

    setAttributes(scoreDef,("key.sig",fifths),("key.mode", mode),("key.pname",pname))

timeSig = next(part.iter_all(score.TimeSignature),None)

if timeSig!=None:
    setAttributes(scoreDef,("meter.count",timeSig.beats),("meter.unit",timeSig.beat_type))

section = addChild(mei_score,"section")

# might want to count staff numbers during processing and update staffGrp if count isn't consistent with clefs
staffGrp = addChild(scoreDef,"staffGrp")

def partition_handleNone(func, iter, partitionAttrib):
    p = partition(func,iter)

    if None in p.keys():
        print("PARTITION ERROR: some elements of set do not have partition attribute \""+partitionAttrib+"\"")
        sys.exit()

    return p

addDefaultStaffDef = True

clefs = partition_handleNone(lambda c:c.number, part.iter_all(score.Clef), "number")

for c in clefs.values():
    clef = c[0]

    for i in range(1,len(c)):
        if c[i].start.t < clef.start.t:
            clef = c[i]


    staffDef = addChild(staffGrp,"staffDef")
    setAttributes(staffDef,("n",clef.number),("lines",5),("clef.shape",clef.sign),("clef.line",clef.line))
    addDefaultStaffDef = False

if addDefaultStaffDef:
    staffDef = addChild(staffGrp,"staffDef")






measure_counter = 0

notes_nextMeasure_perStaff = {}

for m in part.iter_all(score.Measure):
    measure=addChild(section,"measure")
    setAttributes(measure,("n",m.number))

    #for i in range(len(quarterDurations)):
    # handle differing quarter durations within a single measure?

    if m.start.quarter!= m.end.quarter:
        print("something's rotten in the state of Denmark")
        print("quarter duration is different at the end of the measure from when it starts")
        sys.exit()

    quarterDur = m.start.quarter

    notes_withinMeasure = part.iter_all(score.GenericNote, m.start, m.end, include_subclasses=True)

    notes_withinMeasure_perStaff = partition_handleNone(lambda n:n.staff, notes_withinMeasure, "staff")

    notes_nextMeasure_perStaff_keys=set(notes_nextMeasure_perStaff.keys())
    notes_withinMeasure_perStaff_keys=set(notes_withinMeasure_perStaff.keys())

    staff_intersection_keys = notes_nextMeasure_perStaff_keys.intersection(notes_withinMeasure_perStaff_keys)
    notes_nextMeasure_perStaff_keys = notes_nextMeasure_perStaff_keys.difference(staff_intersection_keys)
    notes_withinMeasure_perStaff_keys=notes_withinMeasure_perStaff_keys.difference(staff_intersection_keys)
    staff_disjunction = {}

    for nnm in notes_nextMeasure_perStaff_keys:
        staff_disjunction[nnm]=notes_nextMeasure_perStaff[nnm]

    for nwm in notes_withinMeasure_perStaff_keys:
        staff_disjunction[nwm]=notes_withinMeasure_perStaff[nwm]

    def create_staff(notes_perStaff, ties_perStaff):
        for s in notes_perStaff.keys():
            staff=addChild(measure,"staff")

            setAttributes(staff,("n",s))

            notes_withinMeasure_perStaff_perVoice = partition_handleNone(lambda n:n.voice, notes_perStaff[s], "voice")

            ties_perStaff_perVoice={}

            for voice,notes in notes_withinMeasure_perStaff_perVoice.items():
                layer=addChild(staff,"layer")

                setAttributes(layer,("n",voice))

                ties={}

                chords = [[notes[0]]]



                for i in range(1,len(notes)):
                    rep = chords[-1][0]
                    n = notes[i]
                    # what to do with notes that start same time but have different durations?
                    # should multiple Rests with same start time or a mix of Rests and Notes with same start times be ignored, trigger a warning or be an error?
                    if isinstance(rep,score.Note) and isinstance(n, score.Note) and rep.start.t==n.start.t:
                        if rep.duration==n.duration:
                            chords[-1].append(n)
                        else:
                            print("In staff "+str(s)+",")
                            print("in measure "+str(m.number)+",")
                            print("for voice "+str(voice)+",")
                            print("2 notes start at time "+str(n.start.t)+",")
                            print("but have different durations, namely "+n.id+" has duration "+str(n.duration)+" and "+rep.id+" has duration "+str(rep.duration))
                            print("change to same duration for a chord or change voice of one of the notes for something else")
                            sys.exit()
                    else:
                        chords.append([n])

                def calc_dur_dots_splitNote(n):
                    note_duration = n.duration
                    splitNote = None

                    if n.start.t+n.duration>m.end.t:
                        note_duration = m.end.t - n.start.t
                        # just some value different from None, for later checking
                        splitNote = True


                    # what if note doesn't have any ID?
                    # maybe n.id = generateId()

                    # estimate_symbolic_duration ?
                    # or just this:
                    # n.duration is some fraction of quarterDur and that fraction is a sum of powers of 2
                    # note n breaks up into multiple notes for every disjunct sequence of consecutive powers of 2
                    # where each note has a duration of int(4/the largest power of 2 in the sequence)
                    # and the number of dots is (the length of the sequence - 1)

                    fraction = note_duration/quarterDur
                    intPart = int(fraction)
                    fracPart = fraction - intPart

                    untiedDurations = []
                    powOf_2 = 1

                    while intPart>0:
                        untiedDurations.insert(0,(intPart%2)*powOf_2)
                        intPart=intPart//2
                        powOf_2*=2

                    powOf_2 = 1/2

                    while fracPart > 0:
                        fracPart*=2
                        bit = int(fracPart)
                        fracPart-=bit
                        untiedDurations.append(bit*powOf_2)
                        powOf_2/=2


                    def powerOf2_toDur(p):
                        return int(4/p)

                    dur_dots = []

                    curr_dur = 0
                    curr_dots = 0

                    i=0

                    while i<len(untiedDurations):
                        if curr_dur!=0:
                            if untiedDurations[i]==0:
                                dur_dots.append((powerOf2_toDur(curr_dur), curr_dots))
                                curr_dots=0
                                curr_dur=0
                            else:
                                curr_dots+=1
                        else:
                            curr_dur = untiedDurations[i]

                        i+=1

                    if curr_dur!=0:
                        dur_dots.append((powerOf2_toDur(curr_dur), curr_dots))

                    return dur_dots,splitNote


                openBeam=False
                parent = layer

                next_dur_dots, next_splitNote = calc_dur_dots_splitNote(chords[0][0])

                for i in range(len(chords)):
                    chordNotes = chords[i]
                    rep = chordNotes[0]
                    dur_dots,splitNote = next_dur_dots, next_splitNote

                    # hack right now, don't need to check every iteration, good time to factor out inside of loop
                    if i < len(chords)-1:
                        next_dur_dots, next_splitNote = calc_dur_dots_splitNote(chords[i+1][0])

                    if isinstance(rep,score.Note):
                        if not openBeam and dur_dots[0][0]>=8 and next_dur_dots[0][0]>=8 and i < len(chords)-1:
                            parent = addChild(layer,"beam")
                            openBeam = True

                        if len(chordNotes)>1:
                            chord = addChild(parent,"chord")
                            setAttributes(chord,("dur",dur_dots[0][0]),("dots",dur_dots[0][1]))

                            for n in chordNotes:
                                note=addChild(chord,"note")
                                setAttributes(note,(xmlIdString,n.id),("pname",n.step.lower()),("oct",n.octave))
                        else:
                            note=addChild(parent,"note")
                            setAttributes(note,(xmlIdString,rep.id),("pname",rep.step.lower()),("oct",rep.octave),("dur",dur_dots[0][0]),("dots",dur_dots[0][1]))

                        # additional ties will have to work together with ties specified in partitura
                        if len(dur_dots)>1:
                            # how to interleave beam and chords?

                            for n in chordNotes:
                                ties[n.id]=[n.id]

                            for i in range(1,len(dur_dots)):
                                if not openBeam and dur_dots[i][0]>=8:
                                    parent = addChild(layer,"beam")
                                    openBeam = True

                                if len(chordNotes)>1:
                                    chord = addChild(parent,"chord")
                                    setAttributes(chord,("dur",dur_dots[i][0]),("dots",dur_dots[i][1]))

                                    for n in chordNotes:
                                        note=addChild(chord,"note")

                                        id = n.id+"-"+str(i)

                                        ties[n.id].append(id)

                                        setAttributes(note,(xmlIdString,id),("pname",n.step.lower()),("oct",n.octave))


                                else:
                                    note=addChild(parent,"note")

                                    id = rep.id+"-"+str(i)

                                    ties[rep.id].append(id)

                                    setAttributes(note,(xmlIdString,id),("pname",n.step.lower()),("oct",n.octave),("dur",dur_dots[i][0]),("dots",dur_dots[i][1]))

                        if splitNote!=None:
                            splitNote = []

                            for n in chordNotes:
                                splitNote.append(score.Note(n.step,n.octave, id=n.id+"s"))


                            if len(dur_dots)>1:
                                for n in chordNotes:
                                    ties[n.id].append(n.id+"s")
                            else:
                                for n in chordNotes:
                                    ties[n.id]=[n.id, n.id+"s"]

                    elif isinstance(rep,score.Rest):
                        if splitNote!=None:
                            splitNote = [score.Rest(id=rep.id+"s")]

                        rest = addChild(layer,"rest")

                        setAttributes(rest,(xmlIdString,rep.id),("dur",dur_dots[0][0]),("dots",dur_dots[0][1]))

                        if len(dur_dots)>1:
                            for i in range(1,len(dur_dots)):
                                rest=addChild(layer,"rest")

                                id = rep.id+str(i)

                                setAttributes(rest,(xmlIdString,id),("dur",dur_dots[i][0]),("dots",dur_dots[i][1]))

                    if splitNote!=None:
                        for sn in splitNote:
                            sn.voice = rep.voice
                            sn.start = m.end
                            sn.end = score.TimePoint(rep.start.t+rep.duration)

                            if s in notes_nextMeasure_perStaff.keys():
                                notes_nextMeasure_perStaff[s].append(sn)
                            else:
                                notes_nextMeasure_perStaff[s]=[sn]


                # for now all notes are beamed, however some rules should be obeyed there, see Note Beaming and Grouping


                ties_perStaff_perVoice[voice]=ties

            ties_perStaff[s]=ties_perStaff_perVoice

    ties_perStaff = {}

    create_staff(staff_disjunction, ties_perStaff)


    # check if there are notes that overlap with same staff and same voice during this measure (is that a problem or is it just another case to be handled?)
#     for s in staff_intersection_keys:
#         notes_perVoice_keys, voice_intersection_keys = splitSets(notes_nextMeasure_perStaff[s].keys(), notes_withinMeasure_perStaff[s].keys())
#         print(voice_intersection_keys)


    for slur in part.iter_all(score.Slur, m.start, m.end):
        s = addChild(measure,"slur")
        setAttributes(s, ("staff",slur.start_note.staff), ("startid","#"+slur.start_note.id), ("endid","#"+slur.end_note.id))


    for s,tps in ties_perStaff.items():

        for v,tpspv in tps.items():

            for ties in tpspv.values():

                for i in range(len(ties)-1):
                    tie = addChild(measure, "tie")

                    setAttributes(tie, ("staff",s), ("startid","#"+ties[i]), ("endid","#"+ties[i+1]))

    measure_counter+=1






(etree.ElementTree(mei)).write("testResult.mei",pretty_print=True)

#print(etree.tostring(mei,pretty_print=True))
#partitura.render(part)