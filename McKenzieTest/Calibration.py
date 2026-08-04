import numpy as np

target_prox_T1 = -2.174503356588717; target_prox_T2 = -2.418878165370875

target_term_T1 = -0.562503356588717; target_term_T2 = -1.650078165370875

prox_slip = -3.479205370541947; term_slip = -1.5000089509032453
prop_prox = 0.625


## Finding Correct Bifurcation Angle ##
a = target_prox_T2 / prox_slip
b = target_term_T2 / term_slip

S = np.arccos((1 - a*a - b*b) / (2*a*b))

angle_bot = np.arcsin(a * np.sin(S))
angle_top = S - angle_bot

# print(np.degrees(theta_top))
# print(np.degrees(theta_bot))

## Using Prox_T1 to find correct proximal slip from proportionality ##
adj_prox_slip_T1 = target_prox_T1/prop_prox
adj_term_slip_T1 = target_term_T1/(1-prop_prox)

# adj_prop_prox_1=target_prox_T1/prox_slip
# adj_prop_prox_2=target_term_T1/term_slip
# print(adj_prox_T1)

T2_diag = np.sin(angle_top)/np.sin(angle_top+angle_bot); T2_lat  = np.sin(angle_bot)/np.sin(angle_top+angle_bot)
prox_T2 = prox_slip*T2_lat # T2 Group (PI, LUM)
term_T2 = term_slip*T2_diag # ^^

prox_T1 = prox_slip*prop_prox # T1 group (EIP, EDC)
term_T1 = term_slip*(1-prop_prox) # ^^

print(adj_prox_slip_T1,adj_term_slip_T1)
# np.set_printoptions(precision=6, formatter={'float_kind':'{:.5f}'.format})

# print(f"M_paper:\n{M_paper}");print(f"M:\n{M}");print(f"Difference:\n\n{M-M_paper}")
# print(f"\tDI\tPI\tEIP\tLUM\tEDC");print(f"R_paper:\n{R_paper[-2:, -4:]*1e3}");print(f"R:\n{R[-2:, -4:]*1e3}");print(prox_T2, term_T2, prox_T1, term_T1)
# print(f"R_paper:\n{R_paper*1e3}"); print(f"R:\n{R*1e3}"); print(f"Difference:\n\n{(R-R_paper)*1e3}")
print(np.degrees(np.array([angle_top,angle_bot])))


if __name__ == "__main__":
    pass
    # diff=(abs(f_test-f[point-1])/(f_test+f[point-1]))*100
    # print(f"point: {point}\t{np.degrees(q2)}° {np.degrees(q3)}° {np.degrees(q4)}°")
    # print(f"{M@e[point-1]}\n{f[point-1]}");print(f"Percentage Difference:\n{diff}")

    # print(R)