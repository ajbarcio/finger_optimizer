#ifndef STRUCTURE_MATRIX_H
#define STRUCTURE_MATRIX_H

#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>

// --- configuration constants ---
#define NROWS     (4)
#define NCOLS     (NROWS + 1)
#define numBytes  (((NROWS * NCOLS) * 2 + 7) / 8)

// #define perms[120][5]

// macro def to extract a value from array, for efficiency:
#define getValue(col, row, nRows)    (((col & (0b11 << 2*(nRows-(row+1))))     \
                                                        >> 2*(nRows-(row+1)))-1)
// macro def to set a value to array, replaces whole column but conserves other
// values in row
#define setValue(col,row,nRows,v)    (((col) & ~(0b11 << (2*(nRows-(row+1))))) \
                                     | (((v+1) & 0b11) << (2*(nRows-(row+1)))) )

#define permute_columns(in, out, perm) \
{ \
    for (int c = 0; c< NCOLS; c++) { \
        out[c] = in[perm[c]]; \
    } \
}

#define compare_matrices_lexicographical(a, b)  ({     \               \
int rv = 0; \
  for (int c = 0; c < NCOLS; c++) \
  { \
    if (a[c] < b[c]) rv = -1; \
    if (a[c] > b[c]) rv = 1; \
  } \
  rv = 0; \
rv; \
                                                })

#define count_signs(m, neg, pos)                      \
{                                                     \
  *neg = *pos = 0;                                    \
  int v = 0;                                          \
  unsigned char col = m[0];                           \
  for (int c = 0; c < 5; c++)                         \
  {                                                   \
    col = m[c];                                       \
    for (int r = 0; r < 4; r++)                       \
    {                                                 \
      v = getValue(col, r, NROWS);                    \
      if (v < 0) (*neg)++;                            \
      else if (v > 0) (*pos)++;                       \
    }                                                 \
  }                                                   \
}                                                                   

typedef unsigned char structureMatrix[numBytes];

#define zeroMatrix (structureMatrix){0,0,0,0,0}

// function declarations to expose to main
void print_ascii (structureMatrix out);;
unsigned char* generate_row_signs(int nRows);
void generate_perms_rec(int depth, int usedMask, int* cur, int (*out)[5], int* count);
void generate_perms(int out[120][5]);
unsigned char apply_row_signs_to_column(unsigned char row_sign, unsigned char col);
void apply_row_signs_to_matrix(const structureMatrix in, structureMatrix out, const unsigned char row_sign);
void canonical_form (const structureMatrix in, const unsigned char row_signs[NROWS*NROWS], const int perms[120][5], structureMatrix out);
bool isUnique(structureMatrix candidate, unsigned long numUnique, unsigned char* allUnique);

#endif