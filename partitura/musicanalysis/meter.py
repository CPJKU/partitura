#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Meter numerator, Beat, and Tempo estimation.

Implementation adapted from Jakob Woegerbauer 
based on a model published by Simon Dixon.[1]

References
----------

.. [1] Simon Dixon (2001), Automatic extraction of 
    tempo and beat from expressive performances.
    Journal of New Music Research, 30(1):39–58
"""

import warnings
import numpy as np
# import scipy.spatial.distance as distance
# from scipy.interpolate import interp1d

from partitura.utils import get_time_units_from_note_array, ensure_notearray, add_field


# Scaling factors
MAX = 9999999999999
MIN_INTERVAL = 0.01
MAX_INTERVAL = 2  # in seconds
CLUSTER_WIDTH = 1/12  # in seconds
N_CLUSTERS = 100
INIT_DURATION = 10  # in seconds
TIMEOUT = 10  # in seconds
TOLERANCE_POST = 0.4  # propotion of beat_interval
TOLERANCE_PRE = 0.2  # proportion of beat_interval
TOLERANCE_INNER = 1/12 
CORRECTION_FACTOR = 1/4  # higher => more correction (speed changes)
MAX_AGENTS = 100  # delete low-scoring agents when there are more than MAX_AGENTS
CHORD_SPREAD_TIME = 1/12 # for onset aggregation


class MultipleAgents():
    """
    Class to compute inter onset interval clusters
    and to instantiate a number of agents to 
    approximate beat positions.
    """
    def run(self, onsets, salience):
        self.clusters = []
        self.agents = []
        onsets = np.array(onsets)
        salience = np.array(salience)
        self.setup_clusters(onsets)
        self.init_tracking(onsets, salience)
        self.track(onsets, salience)

    def getTempo(self):
        if len(self.agents) == 0:
            return 120
        return self.agents[0].getTempo()

    def getNum(self):
        if len(self.agents) == 0:
            return 4
        return self.agents[0].getTimeSignatureNum()

    def getBeats(self):
        if len(self.agents) > 0:
            return self.agents[0].history
        return []

    def setup_clusters(self, onsets):
        # create inter-onset interval clusters
        self.clusters = []
        for i in range(len(onsets)):
            for j in range(i+1, len(onsets)):
                ioi = onsets[j]-onsets[i]
                if ioi < MIN_INTERVAL: 
                    continue
                if ioi > MAX_INTERVAL:
                    break
                c_min = False
                for c in self.clusters:
                    k = c.getK(ioi)
                    if k:
                        c_min = c
                if c_min:
                    c_min.addIoi(ioi)
                else:
                    self.clusters.append(Cluster(ioi))

        # merge clusters
        i = 0
        while i < len(self.clusters):
            c_i = self.clusters[i]
            i = i+1
            j = i
            while j < len(self.clusters):
                if abs(c_i.interval - self.clusters[j].interval) < CLUSTER_WIDTH:
                    c_i.addIoi(self.clusters[j].iois)
                    self.clusters.remove(self.clusters[j])
                else:
                    j += 1

        # sanitize
        for c in self.clusters:
            if c.interval <= 0:
                continue
            while c.interval < MIN_INTERVAL:
                c.interval *= 2
            while c.interval > MAX_INTERVAL:
                c.interval /= 2
                
        # merge again
        i = 0
        while i < len(self.clusters):
            c_i = self.clusters[i]
            i = i+1
            j = i
            while j < len(self.clusters):
                if abs(c_i.interval - self.clusters[j].interval) < CLUSTER_WIDTH:
                    c_i.addIoi(self.clusters[j].iois)
                    self.clusters.remove(self.clusters[j])
                else:
                    j += 1

        # calculate cluster scores
        for c_i in self.clusters:
            for c_j in self.clusters:
                n = round(c_j.interval / c_i.interval)
                if abs(c_i.interval - n*c_j.interval) < CLUSTER_WIDTH:
                    c_i.score += Cluster.relationship_factor(n) * len(c_j.iois)

        self.clusters = sorted(self.clusters, key=lambda x: x.score, reverse=True)[
            :N_CLUSTERS]

    def init_tracking(self, onsets, salience):
        self.agents = []
        for c in self.clusters:
            i = 0
            while i < len(onsets) and onsets[i] < INIT_DURATION:
                a = Agent()
                a.beat_interval = c.interval
                a.history.append((onsets[i], salience[i]))
                a.prediction = onsets[i] + c.interval
                a.score = salience[i]
                self.agents.append(a)
                i += 1

    def track(self, onsets, salience):  
        for e_i in range(len(onsets)):
            e = onsets[e_i]
            new_agents = []
            remove_agents = []
            for a in self.agents:
                if e - a.lastBeat() > TIMEOUT:
                    remove_agents.append(a)
                else:
                    while a.prediction + TOLERANCE_POST*a.beat_interval < e:
                        a.history.append((a.prediction, 0)) 
                        a.prediction += a.beat_interval
                    if a.prediction - TOLERANCE_PRE*a.beat_interval <= e and e <= a.prediction + TOLERANCE_POST*a.beat_interval:
                        if abs(a.prediction - e) > TOLERANCE_INNER:
                            a_new = Agent()
                            a_new.beat_interval = a.beat_interval
                            a_new.history = a.history[:]
                            a_new.prediction = a.prediction
                            a_new.score = a.score
                            new_agents.append(a_new)
                        err = e - a.prediction
                        a.beat_interval = a.beat_interval + err*CORRECTION_FACTOR
                        a.prediction = e + a.beat_interval
                        a.history.append((e, salience[e_i]))
                        a.score += (1-abs(err/a.beat_interval)/2.) * salience[e_i]

            for a in remove_agents:
                self.agents.remove(a)
            self.agents = self.agents + new_agents

            # remove duplicate agents
            duplicate = np.zeros(len(self.agents))
            agents_all = self.agents[:]
            self.agents = []
            for i in range(len(agents_all)):
                for j in range(i+1, len(agents_all)):
                    if duplicate[i] > 0 or duplicate[j] > 0:
                        continue

                    if abs(agents_all[i].beat_interval - agents_all[j].beat_interval) < 0.01 \
                            and abs(agents_all[i].lastBeat() - agents_all[j].lastBeat()) < 0.02:
                        if agents_all[i].score > agents_all[j].score:
                            duplicate[j] += 1
                        else:
                            duplicate[i] += 1
                            break

            self.agents = sorted(np.asarray(agents_all)[(duplicate < 1)].tolist(
            ), key=lambda x: x.score, reverse=True)[:MAX_AGENTS]

        self.agents = sorted(self.agents, key=lambda x: x.score, reverse=True)


class Cluster():
    """
    Class for inter onset interval clusters.
    
    Parameters
    ----------
    ioi : float
        an initial inter onset interval 
    
    """
    
    def __init__(self, ioi) -> None:
        self.iois = np.zeros(0)
        self.score = 0
        self.interval = 0
        self.addIoi(ioi)

    def getK(self, ioi):
        diff = abs(self.interval-ioi)
        if diff < CLUSTER_WIDTH:
            return diff
        return False

    def addIoi(self, ioi):
        self.iois = np.append(self.iois, ioi)
        self.interval = np.sum(self.iois)/len(self.iois)

    @staticmethod
    def relationship_factor(d):
        if 1 <= d and d <= 4:
            return 6-d
        elif 5 <= d and d <= 8:
            return 1
        return 0

class Agent():
    """
    Class for beat induction agents.
    """

    def __init__(self) -> None:
        self.beat_interval = 0
        self.prediction = 0
        self.history = []
        self.score = 0

    def lastBeat(self):
        i = len(self.history)-1
        while i > 0 and self.history[i][1] == 0:
            i-=1
        return self.history[i][0]

    def getTempo(self):
        return 60.0 * (len(self.history)-1) / (self.history[-1][0]-self.history[0][0])
    
    def getTimeSignatureNum(self):
        possibleNums = [2, 3, 4, 6, 9, 12, 24]
        bestVal = {num:0 for num in possibleNums}
        salience = list(zip(*self.history))[1]
        sumSalience = sum(salience)
        f = 1.005
        for num in possibleNums:
            for startIdx in range(num):    
                dbs = len(salience[startIdx::num])
                if dbs > 1:
                    downbeatSalience = sum(salience[startIdx::num])/dbs
                    sumSalience = sum(salience[:(dbs-1)*num]) 
                    otherSalience = (sumSalience-downbeatSalience*dbs)/((num-1)*(dbs-1))
                else:
                    downbeatSalience = 0
                    otherSalience = 1
                    
                ratio = downbeatSalience/otherSalience
                bestVal[num] = max(bestVal[num], ratio)

        bestNum = max(bestVal, key=bestVal.get)    
            
        return bestNum


def estimate_time(note_info):
    """
    Estimate tempo, meter (currently only time signature numerator), and beats
    
    Parameters
    ----------
    note_info : structured array, `Part` or `PerformedPart`
        Note information as a `Part` or `PerformedPart` instances or
        as a structured array. If it is a structured array, it has to
        contain the fields generated by the `note_array` properties
        of `Part` or `PerformedPart` objects. If the array contains
        onset and duration information of both score and performance,
        (e.g., containing both `onset_beat` and `onset_sec`), the score
        information will be preferred.
    
    Returns
    -------
    dict
        Tempo, meter, and beat information
    """         

    note_array = ensure_notearray(note_info)
    onset_kw, _ = get_time_units_from_note_array(note_array)
    onsets_raw = note_array[onset_kw]
    
    # aggregate notes in clusters
    aggregated_notes = [(0,0)]
    for note_on in onsets_raw:
        prev_note_on = aggregated_notes[-1][0]
        prev_note_salience = aggregated_notes[-1][1]
        if abs(note_on - prev_note_on) < CHORD_SPREAD_TIME:
            aggregated_notes[-1] = (note_on, prev_note_salience + 1)
        else:
            aggregated_notes.append((note_on, 1))
        
    print(aggregated_notes)    
    onsets, saliences = list(zip(*aggregated_notes))
    
    ma = MultipleAgents()
    ma.run(onsets, saliences)
    
    return dict(tempo=ma.getTempo(),
                meter_numerator=ma.getNum(),
                beats=ma.getBeats())
    