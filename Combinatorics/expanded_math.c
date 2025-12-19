#include <stdio.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <stdbool.h>
#include <stdint.h>
#include <omp.h>
// #include <gsl_block.h>
#include <gsl/gsl_block.h>

#include "hash.h"
#include "structureMatrix.h"
#include "expandedMatrix.h"

// #include <gsl/gsl_sf_bessel.h>


char ***signings(int *numDs) {
    *numDs = 1 << NROWS; // 2^m
    char ***Ds = malloc(*numDs * sizeof(char **));
    
    for (int i = 0; i < *numDs; i++) {
        // allocate m x m matrix
        Ds[i] = malloc(NROWS * sizeof(char *));
        for (int row = 0; row < NROWS; row++) {
            Ds[i][row] = calloc(NROWS, sizeof(char)); // zero init
            Ds[i][row][row] = (i & (1 << row)) ? 1 : -1;
        }
    }
    return Ds;
}

/* Free memory allocated by signings */
void free_signings(char ***Ds, int numDs) {
    for (int i = 0; i < numDs; i++) {
        for (int row = 0; row < NROWS; row++) free(Ds[i][row]);
        free(Ds[i]);
    }
    free(Ds);
}

bool identify_SSC(char **S) {
    bool success = true, valid;
    int numDs;
    char ***Ds = signings(&numDs); // We need Ds[i] to be an mxm matrix
                                            // but you gotta learn how pointers
                                            // work man
    char check[NROWS][NCOLS];
    for (int k = 0; k < numDs; k++) {
        
        multiply_char_matrices(Ds[k], NROWS, NROWS, S, NCOLS, check);

        valid = false;
        for (int l; l < NCOLS; l++) {
            bool all_ge_zero = true;
            bool any_nonzero = false;
            for (int row = 0; row < NROWS; row++) {
                if (check[row][l] < 0) all_ge_zero = false;
                if (check[row][l] != 0) any_nonzero = true;
            }
            if (all_ge_zero && any_nonzero) valid = true;
        }
    }
    free_signings(Ds, numDs);
}

void main() {
    // Load in list of unique structure matrices (we will only be expanding these)
    FILE*fptr;
    fptr = fopen("unique4x5.out", "rb");
    // Check length
    fseek(fptr, 0, SEEK_END);
    size_t length = ftell(fptr);
    rewind(fptr);
    // Allocate memory for all [n] matrices
    size_t numMatrices = length/sizeof(structureMatrix);
    structureMatrix *uniqueList = malloc(numMatrices * numBytes);
    // read 'em in, johnny
    size_t read_bytes = fread(uniqueList, numBytes, numMatrices, fptr);
    structureMatrix cur;
    char ecur[NROWS][NCOLS];
    numMatrices = 10;
    for (int k = 0; k < numMatrices; k++) {
        // structureMatrix cur = uniqueList[k];
        memcpy(cur, uniqueList[k], numBytes);
        expand_structMatrix(cur, ecur);
        print_expanded(ecur);

    }
}