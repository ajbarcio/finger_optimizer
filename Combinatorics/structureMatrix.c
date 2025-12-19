// structureMatrix.c

#include "structureMatrix.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <omp.h>


void print_ascii (structureMatrix out) {
# ifdef TRACE
printf("print_ascii()\n");
# endif // TRACE
    int cell;
    for (int j = 0; j < NROWS; j++)
    {
        for (int i = 0; i < NCOLS; i++) {
            cell = getValue(out[i], j, NROWS);
            printf("%2d ", cell);
        }
        printf("\n");
    }
}

unsigned char* generate_row_signs(int nRows) {
# ifdef TRACE
    printf("generate_row_signs()\n");
# endif // TRACE
    unsigned char* row_signs = (unsigned char*)malloc(nRows*nRows*sizeof(unsigned char));
    for (int mask = 0; mask < nRows*nRows; mask++) {
        // printf("%d \n", mask);
        row_signs[mask]=0u;
        for (int r = 0; r < nRows; r++) {
            unsigned bit = (unsigned)(mask >> r) & 1u;
            unsigned tri = 1u | (bit << 1);
            row_signs[mask] |= (unsigned char)(tri << (r * 2));
        }
    }
    return row_signs;
}

void generate_perms_rec(int depth, int usedMask, int* cur, int (*out)[5], int* count) {
# ifdef TRACE
    printf("generate_row_signs()\n");
# endif // TRACE
    if (depth == 5) { memcpy(out[(*count)++], cur, sizeof(int)*5); return; }
    for (int i = 0; i < 5; i++) if (!(usedMask & (1<<i))) {
        cur[depth] = i;
        generate_perms_rec(depth+1, usedMask | (1<<i), cur, out, count);
    }
}

void generate_perms(int out[120][5]) {
    int cur[5];
    int count = 0;

    generate_perms_rec(0, 0, cur, out, &count);
}

unsigned char apply_row_signs_to_column(unsigned char row_sign, unsigned char col) {
# ifdef TRACE
    printf("apply_row_signs_to_column()\n");
# endif // TRACE
    unsigned char a0 = col & 0x55;        // LSB of each field
    unsigned char a1 = (col & 0xAA) >> 1; // MSB of each field
    unsigned char b0 = row_sign & 0x55;        // LSB of each field
    unsigned char b1 = (row_sign & 0xAA) >> 1; // MSB of each field
 
    // Compute C0 and C1 per the scalar truth table logic
    unsigned char c0 = (~a1 & a0);                        // LSB
    unsigned char c1 = ((~a1 & ~a0 & b1) | (a1 & ~a0 & ~b1)); // MSB
 
    c0 &= 0x55; // keep in LSB positions
    c1 &= 0x55; // MSB in LSB positions first
    c1 <<= 1;   // move to MSB positions of each field
 
    return c1 | c0;
}

void apply_row_signs_to_matrix(const structureMatrix in, structureMatrix out, const unsigned char row_sign) {
# ifdef TRACE
    printf("apply_row_signs_to_matrix()\n");
# endif // TRACE
    for (int c=0; c<NCOLS; c++) {
        out[c] = apply_row_signs_to_column(row_sign, in[c]);
    }
}

// void permute_columnsF(const structureMatrix in, structureMatrix out, const int perm[NCOLS]) {
// # ifdef TRACE
//     printf("permute_columnsF()\n");
// # endif // TRACE
//     for (int c=0; c<NCOLS; c++) {
//         out[c] = in[perm[c]];
//     }
// }

// int compare_matrices_lexicographicalF(const structureMatrix a, const structureMatrix b) {
// # ifdef TRACE
//     printf("compare_matrices_lexicographical()\n");
// # endif // TRACE
//     for (int c=0; c<NCOLS; c++) {
//         if (a[c] < b[c]) return -1;
//         if (a[c] > b[c]) return 1;
//     }
//     return 0;
// }

// void count_signsF(const structureMatrix m, int* neg, int* pos) {
// # ifdef TRACE
//     printf("count_signs()\n");
// # endif // TRACE
//     *neg = *pos = 0;
//     for (int c = 0; c < 5; c++) {
//         unsigned char col = m[c];
//         for (int r = 0; r < 4; r++) {
//             int v = getValue(col, r, NROWS);
//             if (v < 0) (*neg)++;
//             else if (v > 0) (*pos)++;
//         }
//     }
// }

void canonical_form (const structureMatrix in, const unsigned char row_signs[NROWS*NROWS], const int perms[120][5], structureMatrix out) {
# ifdef TRACE
    printf("canonical_form()\n");
# endif // TRACE
    structureMatrix best;
    int bestInit = 0;
    int bestPosCount = -1;

    structureMatrix temp1, temp2, temp3;

    unsigned char flipAllRows = 0b11111111;

    int neg, pos;
    int lex_primacy = 0;

    for (int sign_pattern = 0; sign_pattern < NROWS*NROWS; sign_pattern++) {
        apply_row_signs_to_matrix(in, temp1, row_signs[sign_pattern]);
        for (int permutation = 0; permutation < 120; permutation++) {
            permute_columns(temp1, temp2, perms[permutation]);
            count_signs(temp2, &neg, &pos);
            // if (neg > pos ){
            //     apply_row_signs_to_matrix(temp2, temp3, flipAllRows);
            // }
            // if (!bestInit || pos > bestPosCount || (pos == bestPosCount) && (compare_matrices_lexicographical(temp2, best) < 0)) {
            //     memcpy(best, temp2, sizeof(structureMatrix));
            //     bestInit = 1;
            //     bestPosCount = pos;
                // printf("%3d %3d \n", sign_pattern, permutation);
                // print_ascii(best);
            // }
            lex_primacy = 0;
            // We've defunctionalized this bit cause it's only called once. 
            // Why not call the macro?
            for (int c = 0; c< NCOLS; c++) {
                if (temp2[c] < best[c]) {lex_primacy =  -1; break;}
                if (temp2[c] > best[c]) {lex_primacy =  1; break;}
            }
            if (!bestInit || pos > bestPosCount || (pos == bestPosCount) && (lex_primacy < 0)) {
                memcpy(best, temp2, sizeof(structureMatrix));
                bestInit = 1;
                bestPosCount = pos;
            }
            if (permutation==119 && sign_pattern==15){
                // printf("done \n");
            }
        }
    }
    memcpy(out, best, sizeof(structureMatrix));
    // printf("\n");
}

bool isUnique(structureMatrix candidate, unsigned long numUnique, unsigned char* allUnique) {
# ifdef TRACE
    printf("isUnique()\n");
# endif // TRACE
    unsigned char* ptr = 0;
    for (int i=0; i < numUnique; i++) {
        ptr = allUnique + i * numBytes;
        if (memcmp(candidate, ptr, numBytes) == 0) {
            return false;
        }
    }
    return true;
}
