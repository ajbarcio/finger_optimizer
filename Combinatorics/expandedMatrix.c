// expandedMatrix.c

#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>

#include "structureMatrix.h"
#include "expandedMatrix.h"

void multiply_char_matrices(
    const char *A, size_t rowsA, size_t colsA,
    const char *B, size_t colsB,
    char *C                  // int output to avoid overflow
) {
    for (size_t i = 0; i < rowsA; i++) {
        for (size_t j = 0; j < colsB; j++) {
            int sum = 0;
            for (size_t k = 0; k < colsA; k++) {
                sum += A[i*colsA + k] * B[k*colsB + j];
            }
            C[i*colsB + j] = sum;
        }
    }
}

void expand_structMatrix(const structureMatrix in, char out[NROWS][NCOLS]) {
    char value;

    for (int i = 0; i<NCOLS; i++) {
        for (int j = 0; j < NROWS; j++) {
            value = (unsigned char)getValue(in[i], j, NROWS);
            out[j][i] = value;
        }
    }
}

void print_expanded(const char expanded[NROWS][NCOLS]) {
        printf("\n");
    for (int j = 0; j<NROWS; j++) {
        for (int i = 0; i < NCOLS; i++) {
            printf("%2d ", expanded[j][i]);
        }
        printf("\n");
    }
}