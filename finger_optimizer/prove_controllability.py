import sympy as sp
a, b, c = sp.symbols('a b c ')


f = a*(b-c)+b*(c-2) + 1
g = -a*(b-c)+c*(b-2)+1
h = a*(b-c-2)-b*c +1

indicator = sp.Piecewise(
    (1, sp.And(f > 0, g > 0, h > 0)),
    (0, True)
)

prob = sp.integrate(indicator, (a, 0, 1), (b, 0, 1), (c, 0, 1))

print(prob)
print(prob.evalf())
