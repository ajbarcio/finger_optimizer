import numpy as np
import math
from math import *
import itertools
from collections import Counter


module = 0.5
addendum = module # I think this should be right

sun = 16
planet1 = 17
planet2 = 16
ring1 = 50
ring2 = 49

num_planets = 5

sun_idle = (ring2 - (planet2*2))
 
planet_carrier_d = (sun+planet1)*module
planet1_od = planet1*module + 2*addendum

minimum_angle = 2*np.arcsin((planet1_od/2)/(planet_carrier_d/2))
minimum_angle = minimum_angle*180/np.pi # in degrees

theta1 = []
theta2 = []
 
for k in range(100):
    theta1.append((360*k)/(sun + ring1))
    theta2.append((360*k)/(sun_idle + ring2))
# print('theta1', theta1)
# print('theta2', theta2)
 
angles = []
for i in range(len(theta1)):
    for j in range(len(theta2)):
        if theta2[j] == theta1[i]:
            angles.append(theta2[j])
 
if len(angles) == 0:
    print("No possible combanations were found")

angles = [angle for angle in angles if ((angle < 360) and (angle > minimum_angle))]

lowerAngle = max([angle for angle in angles if angle < 360/num_planets], default=None)
upperAngle = min([angle for angle in angles if angle > 360/num_planets], default=None)

angles = [lowerAngle, upperAngle]
combinations = [seq for seq in itertools.product(angles, repeat=num_planets-1)]
# print(combinations)
valids = []
for combination in combinations:
    if np.sum(np.array(combination))+minimum_angle <= 360:
        valids.append(list(combination))
# print(valids)

repeats = []
formats = []
for i, valid in enumerate(valids):
    valid.append(360-np.sum(np.array(valid)))
    valids[i] = valid
    counts = Counter(valids[i])
    format = {}
    for val, cnt, in sorted(counts.items(), key=lambda x: x[1]):
        format[val] = cnt
    # print(format)
    if format in formats:
        repeats.append(i)
        # print(valids[i])
    else:
        print(valid, format)
    formats.append(format)

valids = [valid for i, valid in enumerate(valids) if i not in repeats]

print(valids)

# means = []
minError = 100
for valid in valids:
    mean = (np.mean(np.array(valid)))
    error = abs(mean-360/num_planets)
    if error < minError:
        bestValid = valid
        minError = error
        bestMean = mean
print(bestValid, error, bestMean)





# even = np.ones(num_planets-1)*(360/num_planets)
# minError = 1000
# for valid in valids:
#     if np.linalg.norm(np.array(valids)-even) < minError:
#         bestSol = valid
#         minError = np.linalg.norm(np.array(valids)-even)
# print(bestSol)
# # print(len(combinations))
# # print(lowerAngle, upperAngle)

