from strucMatrices import testbedFinger4
from scipy.integrate import quad
import numpy as np

testStructure = testbedFinger4

effort_lookup = {obj.idx: obj for obj in testStructure.effortFunctions}
totalExcursions = []

def control_law(q, theta, d_theta):
    return np.linalg.pinv(testStructure(theta)) @ testStructure(theta) @ q

# def excursion_from_theta_to_theta(tendon, t_from, t_to):
#     totalExcursion = 0.0
#     for joint in range(testStructure.numJoints):
#         if (joint, tendon) in effort_lookup.keys():
#             addition = quad(effort_lookup.get((joint,tendon)),
#                             t_from, t_to)[0] * testStructure.D[joint, tendon]
#             totalExcursion+=addition
#     return totalExcursion

# def worst_case_excursions_in_theta_range(tendon, t_from, t_to):
#     excursionsFlexion = 0.0
#     excursionsExtension = 0.0
#     for joint in range(testStructure.numJoints):
#         if (joint, tendon) in effort_lookup.keys():
#             joint_tendon_excursion = quad(effort_lookup.get((joint,tendon)),
#                             t_from, t_to)[0] * testStructure.D[joint, tendon]
#             if joint_tendon_excursion < 0:
#                 excursionsExtension+=joint_tendon_excursion
#             elif joint_tendon_excursion > 0:
#                 excursionsFlexion+=joint_tendon_excursion
#             else:
#                 print("this should not have happened")
#     return excursionsExtension, excursionsFlexion

# R_A = np.diag([r]*testStructure.numTendons)

r = 0.273
# doing one bit of a distal cycle
start_angle = np.array([0,0,0])*np.pi/180
end_angle =   np.array([70,70,70])*np.pi/180
# end_angle = np,add

for i in range(testStructure.numTendons):
    # totalExcursion = 0.0
    # tendonExcursion = excursion_from_theta_to_theta(i,30*np.pi/180,60*np.pi/180)
    min_excursion, max_excursion = testStructure.tendon_excursion_limits(i,
                                                                    start_angle,
                                                                      end_angle)
    min_turn = min_excursion/r*180/np.pi
    max_turn = max_excursion/r*180/np.pi
    total_excursion_range = max_excursion - min_excursion
    total_turn_range = total_excursion_range/r*180/np.pi

    print(f"Tendon {i} may need to let out by as much as {min_excursion}, \
requires {min_turn} degrees of turn")
    print(f"Tendon {i} may need to take in by as much as {max_excursion}, \
requires {max_turn} degrees of turn")
    print(f"This results in an overall arc of {total_turn_range}")

extra_let_out = 45+51+10
print(f"for {extra_let_out} more degrees of let out, cut {extra_let_out*np.pi/180*r} in of tube")


start_angles = np.linspace(0,90,120)*np.pi/180
worst_excursion = 0
for start in start_angles:
    end = start+0.03 # max command in single timestep hopefully
    for j in range(testStructure.numJoints):
        for i in range(testStructure.numTendons):
            excursion = testStructure.tjp_excursion(j, i, start, end)
            if excursion > worst_excursion:
                worst_excursion = excursion
print(f"Biggest expected excursion in single step is {worst_excursion}")
