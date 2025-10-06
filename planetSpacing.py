import numpy as np
import math
from math import *
import itertools
from collections import Counter


# module = 0.5
# addendum = module # I think this should be right

class Planetary3k():
    def __init__(self, module, SGTeeth, P1Teeth, p2Teeth, R1Teeth, R2Teeth, nPlanets = 2) -> None:

        self.module = module
        self.addendum = module

        self.sun = SGTeeth
        self.planet1 = P1Teeth
        self.planet2 = p2Teeth
        self.ring1 = R1Teeth
        self.ring2 = R2Teeth

        self.sun_idle = self.ring2-(self.planet2*2)

        self.num_planets = nPlanets

        self.planet_carrier_d = (self.sun+self.planet1)*module

        self.uniform_angle = 360/self.num_planets

    def min_angle(self):
        planet1_od = self.planet1*self.module + 2*self.addendum
        self.minimum_angle = 2*np.arcsin((planet1_od/2)/(self.planet_carrier_d/2))
        self.minimum_angle = self.minimum_angle*180/np.pi # in degrees
        return self.minimum_angle

    def imbalance(self, angles):
        r = self.planet_carrier_d*0.5
        sumr = 0
        sump = 0
        total_angle = 0
        total_angles = []
        angles.insert(0,0)
        for angle in angles[:-1]:
            total_angle += angle
            total_angles.append(total_angle)
            new_sumr = sqrt(sumr**2+r**2+2*sumr*r*np.cos(np.radians(total_angle-np.degrees(sump))))

            sump = sump + np.arctan2(r*np.sin(np.radians(total_angle-np.degrees(sump))),
                                     sumr+r*np.cos(np.radians(total_angle-np.degrees(sump))))
            sumr = new_sumr
        imbalanceDist = sumr/r
        imbalanceDir = np.min(np.array([abs(np.radians(sump)-angle) for angle in total_angles]))
        return imbalanceDist, imbalanceDir

def possibleAngles(planet: Planetary3k):

    theta1 = []
    theta2 = []

    for k in range(100):
        theta1.append((360*k)/(planet.sun + planet.ring1))
        theta2.append((360*k)/(planet.sun_idle + planet.ring2))

    angles = []
    for i in range(len(theta1)):
        for j in range(len(theta2)):
            if theta2[j] == theta1[i]:
                angles.append(theta2[j])

    if len(angles) == 0:
        print("No possible combanations were found")
    minimum_angle = planet.min_angle()
    angles = [angle for angle in angles if ((angle < 360) and (angle > minimum_angle))]
    return angles

def lower_angle_pair_combos(planet: Planetary3k, tickAngles):

    minimum_angle = planet.min_angle()

    lowerAngle = max([angle for angle in tickAngles if angle < 360/planet.num_planets], default=None)
    upperAngle = min([angle for angle in tickAngles if angle > 360/planet.num_planets], default=None)

    tickAngles = [lowerAngle, upperAngle]
    combinations = [seq for seq in itertools.product(tickAngles, repeat=planet.num_planets-1)]
    valids = []
    for combination in combinations:
        if np.sum(np.array(combination))+minimum_angle <= 360:
            valids.append(list(combination))

    repeats = []
    formats = []
    for i, valid in enumerate(valids):
        valid.append(360-np.sum(np.array(valid)))
        valids[i] = valid
        counts = Counter(valids[i])
        format = {}
        for val, cnt, in sorted(counts.items(), key=lambda x: x[1]):
            format[val] = cnt
        if format in formats:
            repeats.append(i)
        else:
            pass
        formats.append(format)

    valids = [valid for i, valid in enumerate(valids) if i not in repeats]
    return valids

def choose_most_balanced(planet: Planetary3k, schemes):
    minImbalance = planet.planet_carrier_d*0.5*10
    bestScheme = None
    for scheme in schemes:
        dist, dir = planet.imbalance(scheme)
        imbalance = np.linalg.norm([dist, dir]) # equal weights for equal units
        if dist < minImbalance:
            bestScheme = scheme
            bestImbalance = imbalance
            imbalanceDir = dir
            minImbalance = dist
    return bestScheme, bestImbalance, imbalanceDir

planetary = Planetary3k(0.5,16,17,16,50,49,5)

tickAngles                    = possibleAngles(planetary)
trialSets                     = lower_angle_pair_combos(planetary, tickAngles)
print(trialSets)
bestSet, magnitude, direction = choose_most_balanced(planetary, trialSets)
print(bestSet, magnitude*planetary.planet_carrier_d*.5, np.degrees(direction))

testSchemes = [[72,72,72,72,72]]
bestSet, magnitude, direction = choose_most_balanced(planetary, testSchemes)
print(bestSet, magnitude*planetary.planet_carrier_d*.5, np.degrees(direction))
