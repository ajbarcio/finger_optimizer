# RESULTS:

Please note best grips are reported in units relative to max pulley radius and tendon force
Maximum pulley radius is normalized to unitless 1 unless otherwise noted
Maximum force         is normalized to unitless 1 unless otherwise noted

## For 2:1 flexion to extension ratio:

### Type 1:

Best grip magnitude: 3.741
Pulley radii:
[[ 1.     1.     1.    -0.875]
 [ 0.     1.     1.    -0.375]
 [ 0.     0.     1.    -0.125]]
(Objective to maximize flexure grip strength)

Best grip magnitude: 1.590
Pulley Radii:
[[ 0.971  0.033  0.151 -0.568]
 [ 0.     0.968  0.141 -0.531]
 [ 0.     0.     0.953 -0.42 ]]
(Objective to maximize individual joint flexure ability)
NOTE: this wants to approach a diagonal matrix with the trivial solution:
[[ 1.     0.     0.    -0.5]
 [ 0.     1.     0.    -0.5]
 [ 0.     0.     1.    -0.5]]
(duh)

### Type 2:

best grip magnitude: 2.303
Pulley radii:

[[ 0.944 -0.47   0.844 -0.878]
 [ 0.    -0.243  1.078 -0.85 ]
 [ 0.     0.     0.974 -0.974]]
Note: Only succeeds in one axis. Impossible?


Better result from COBYQA:

2.449489742783178
[[ 1.    -0.626  1.    -0.678]
 [ 0.    -0.538  1.    -0.157]
 [ 0.     0.     1.    -0.668]]


### Type 3

MaxGrip (degenerate)
2.449489742783178
[[ 1.     1.    -0.739 -0.   ]
 [ 0.     1.    -0.25  -0.   ]
 [ 0.     0.    -0.491  1.   ]]

MaxJoint (degenerate)
1.7320508075688772
[[ 1.     0.    -0.42  -0.138]
 [ 0.     1.    -0.284 -0.419]
 [ 0.     0.    -0.489  1.   ]]

### Hollow

Also only on one axis but not degenerate
3.523690269047387
[[ 0.     1.025  0.994 -1.001]
 [ 1.029  0.     1.041 -0.761]
 [ 1.002  1.013  0.    -1.013]]

1.678967148865948
[[ 0.     1.048  1.037 -0.778]
 [ 1.044  0.     0.998 -1.017]
 [ 1.019  1.026  0.    -1.019]]

### Diagonal

Grip
1.7320508075688772
[[ 1.   0.   0.  -0.5]
 [ 0.   1.   0.  -0.5]
 [ 0.   0.   1.  -0.5]]

Joint
1.7320508075688772
[[ 1.   0.   0.  -0.5]
 [ 0.   1.   0.  -0.5]
 [ 0.   0.   1.  -0.5]]

### Free

3.039242794745715
[[-1.     1.     1.    -0.034]
 [ 1.     0.1    1.    -0.992]
 [ 1.     1.    -0.044 -0.923]]

 3.039242794745715
[[-1.     1.     1.    -0.034]
 [ 1.     0.1    1.    -0.992]
 [ 1.     1.    -0.044 -0.923]]

### Free constrained


# Controllability results:

## Upper Triangular:

### All Unique well-posed matrices

Matrix 0:
[[ 1 -1 -1  1]
 [ 0  1 -1 -1]
 [ 0  0  1 -1]]

Matrix 1:
[[ 1 -1 -1 -1]
 [ 0  1 -1 -1]
 [ 0  0 -1  1]]

Matrix 2:
[[ 1  1 -1  1]
 [ 0  1 -1  1]
 [ 0  0  1 -1]]

Matrix 3:
[[ 1  1 -1  1]
 [ 0  1  1 -1]
 [ 0  0  1 -1]]

Matrix 4:
[[ 1  1 -1 -1]
 [ 0  1 -1  1]
 [ 0  0  1 -1]]

Matrix 5:
[[ 1  1 -1 -1]
 [ 0  1 -1 -1]
 [ 0  0 -1  1]]

Matrix 6:
[[ 1 -1  1  1]
 [ 0  1 -1  1]
 [ 0  0  1 -1]]

Matrix 7:
[[ 1 -1 -1 -1]
 [ 0  1 -1  1]
 [ 0  0  1 -1]]

## Free Matrices

