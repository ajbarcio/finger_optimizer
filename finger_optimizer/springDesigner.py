import numpy as np
from scipy.spatial.distance import minkowski

data = np.loadtxt('AvailableSprings.csv', delimiter=',')

stiffnesses = data[:,2]
springData  = data

# print(springData)

def resultStiffnesses(stiffnesses, springData):
    result = []
    posSprings = []
    negSprings = []
    for i in range(len(stiffnesses)):
        for j in range(len(stiffnesses)):
            result.append(stiffnesses[i]-stiffnesses[j])
            posSprings.append(springData[i,:])
            negSprings.append(springData[j,:])
    return result, posSprings, negSprings

def jointDesigner(preloadRequired, targetStiffness, possibleStiffnesses, posSprings, negSprings):
    allData = np.hstack([np.atleast_2d((possibleStiffnesses)).T, posSprings, negSprings])
    minError = 1
    minStrength = 1000
    minSize     = 1000
    bestCombo = None
    for i, stiffness in enumerate(possibleStiffnesses):
        preloadedSpringStiffness = negSprings[i][-1]
        # print(preloadedSpringStiffness)
        availablePreloadDeflection = negSprings[i][0]
        requiredPreloadDeflection = preloadRequired/preloadedSpringStiffness
        remainingDeflection = availablePreloadDeflection-requiredPreloadDeflection
        error = abs(stiffness - targetStiffness)
        if targetStiffness != 0:
            percentError = error/targetStiffness
            if (remainingDeflection > 90) and (percentError<=minError):
                minError = percentError
                bestCombo = allData[i]
                returnPreload = requiredPreloadDeflection
        else:
            strength = np.mean([negSprings[i][1], posSprings[i][1]])
            size     = np.mean([negSprings[i][0], posSprings[i][0]])
            if (remainingDeflection > 90) and (error <= minError) and (strength <= minStrength) and (size <= minSize):
                minError = error
                minStrength = strength
                minSize = size
                bestCombo = allData[i]
                returnPreload = requiredPreloadDeflection
    return bestCombo, minError, returnPreload


possibleStiffnesses, posSprings, negSprings = resultStiffnesses(stiffnesses, springData)

j1Best, j1Err, j1Preload = jointDesigner(22.78,  0.268,   possibleStiffnesses, posSprings, negSprings)
j2Best, j2Err, j2Preload = jointDesigner(14.462, 0.08325, possibleStiffnesses, posSprings, negSprings)
j3Best, j3Err, j3Preload = jointDesigner(6.15,   0.000,   possibleStiffnesses, posSprings, negSprings)

print(f'The closest possible stiffness for J1 with the provided springs is {j1Best[0]:1.5f}, with error of {j1Err*100:2.2f}%')
print(f'This uses a {j1Best[1]:3.0f}° spring with max torque {j1Best[2]:2.3f} in.lbs for the positive spring (LH)')
print(f'      and a {j1Best[4]:3.0f}° spring with max torque {j1Best[5]:2.3f} in.lbs for the negative spring (RH), preloaded to {j1Preload:3.5f}°')
print(f'The closest possible stiffness for J2 with the provided springs is {j2Best[0]:1.5f}, with error of {j2Err*100:2.2f}%')
print(f'This uses a {j2Best[1]:3.0f}° spring with max torque {j2Best[2]:2.3f} in.lbs for the positive spring (LH)')
print(f'      and a {j2Best[4]:3.0f}° spring with max torque {j2Best[5]:2.3f} in.lbs for the negative spring (RH), preloaded to {j2Preload:3.5f}°')
print(f'The closest possible stiffness for J3 with the provided springs is {j3Best[0]:1.5f}, with absolute error of {j3Err:1.5f}')
print(f'This uses a {j3Best[1]:3.0f}° spring with max torque {j3Best[2]:2.3f} in.lbs for the positive spring (LH)')
print(f'      and a {j3Best[4]:3.0f}° spring with max torque {j3Best[5]:2.3f} in.lbs for the negative spring (RH), preloaded to {j3Preload:3.5f}°')

# print(j1Best)
# print(j2Best)
# print(j3Best)

# print(np.hstack([np.atleast_2d((possibleStiffnesses)).T, posSprings, negSprings]))