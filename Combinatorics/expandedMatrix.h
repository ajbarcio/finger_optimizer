// expandedMatrix.h

#ifndef EXPANDED_MATRIX_H
#define EXPANDED_MATRIX_H

#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>

#include "structureMatrix.h"

void expand_structMatrix(const structureMatrix in, char out[NROWS][NCOLS]);
void print_expanded(const char expanded[NROWS][NCOLS]);
void multiply_char_matrices(
    const char *A, size_t rowsA, size_t colsA,
    const char *B, size_t colsB,
    char *C );

#endif