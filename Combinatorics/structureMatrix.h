#ifndef STRUCTURE_MATRIX_H
#define STRUCTURE_MATRIX_H

#include <stdlib.h>

// --- configuration constants ---
#define NROWS     (4)
#define NCOLS     (NROWS + 1)
#define numBytes  (((NROWS * NCOLS) * 2 + 7) / 8)

// macro def to extract a value from array, for efficiency:
#define getValue(col, row, nRows)    (((col & (0b11 << 2*(nRows-(row+1))))     \
                                                        >> 2*(nRows-(row+1)))-1)
// macro def to set a value to array, replaces whole column but conserves other
// values in row
#define setValue(col,row,nRows,v)    (((col) & ~(0b11 << (2*(nRows-(row+1))))) \
                                     | (((v+1) & 0b11) << (2*(nRows-(row+1)))) )

// --- typedef for your fixed-size bit-packed matrix ---
typedef unsigned char structureMatrix[numBytes];

// We're breaking tons of rules here
// trying to define a legit 2-bit data type but mayyyybe its ok? because I will
// never address it individually?
typedef struct __attribute__((packed)) {
    unsigned char entry : 2;
} twoBitEntry;

// wE'RE doIN IT
typedef twoBitEntry Col[NCOLS];

typedef Col twoBitSquareArray[NCOLS];

// structureMatrix zeroMatrix = {0,0,0,0,0};
#define zeroMatrix (structureMatrix){0,0,0,0,0}

#endif