#include <stdio.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <stdbool.h>

#define NROWS (4)
#define NCOLS (NROWS + 1)
#define numBytes (((NROWS * NCOLS) * 2 + 7) / 8)

// macro def to extract a value from array, for efficiency:
#define getValue(col, row, nRows)    (((col & (0b11 << 2*(nRows-(row+1))))     \
                                                        >> 2*(nRows-(row+1)))-1)
// macro def to set a value to array, replaces whole column but conserves other
// values in row
#define setValue(col,row,nRows,v)    (((col) & ~(0b11 << (2*(nRows-(row+1))))) \
                                     | (((v+1) & 0b11) << (2*(nRows-(row+1)))) )

// Create data type which is a char array of requisite length
typedef unsigned char structureMatrix[numBytes];

// const int numColPerms = (int)tgamma(nCols+1)

// This struct is inefficient
    // typedef struct bitmatrix
    // {
    //     unsigned char col0;
    //     unsigned char col1;
    //     unsigned char col2;
    //     unsigned char col3;
    //     unsigned char col4;
    // } Bitmatrix;
    // cell =                          ((twoBitArray[i] & (0b11 << 2*(nRows-(j+1)))) >> 2*(nRows-(j+1)));



// struct perms {
//     int row_signs[nRows*nRows][nRows];
//     int column_perms[(int)tgamma(nCols+1)][nCols];
// }

void print_ascii (structureMatrix out) {
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

int perms[120][5];
int cur[5];
int count = 0;

void generate_perms_rec(int depth, int usedMask, int* cur, int (*out)[5], int* count) {
    if (depth == 5) { memcpy(out[(*count)++], cur, sizeof(int)*5); return; }
    for (int i = 0; i < 5; i++) if (!(usedMask & (1<<i))) {
        cur[depth] = i;
        generate_perms_rec(depth+1, usedMask | (1<<i), cur, out, count);
    }
}

unsigned char apply_row_signs_to_column(unsigned char row_sign, unsigned char col) {
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
    for (int c=0; c<NCOLS; c++) {
        out[c] = apply_row_signs_to_column(row_sign, in[c]);
    }
}

void permute_columns(const structureMatrix in, structureMatrix out, const int perm[NCOLS]) {
    for (int c=0; c<NCOLS; c++) {
        out[c] = in[perm[c]];
    }
}

int compare_matrices_lexicographical(const structureMatrix a, const structureMatrix b) {
    for (int c=0; c<NCOLS; c++) {
        if (a[c] < b[c]) return -1;
        if (a[c] > b[c]) return 1;
    }
    return 0;
}

void count_signs(const structureMatrix m, int* neg, int* pos) {
    *neg = *pos = 0;
    for (int c = 0; c < 5; c++) {
        unsigned char col = m[c];
        for (int r = 0; r < 4; r++) {
            int v = getValue(col, r, NROWS);
            if (v < 0) (*neg)++;
            else if (v > 0) (*pos)++;
        }
    }
}

void canonical_form (const structureMatrix in, const unsigned char row_signs[NROWS*NROWS], structureMatrix out) {
    structureMatrix best;
    int bestInit = 0;
    int bestPosCount = -1;

    structureMatrix temp1, temp2, temp3;

    unsigned char flipAllRows = 0b11111111;

    for (int sign_pattern = 0; sign_pattern < NROWS*NROWS; sign_pattern++) {
        apply_row_signs_to_matrix(in, temp1, row_signs[sign_pattern]);
        for (int permutation = 0; permutation < 120; permutation++) {
            permute_columns(temp1, temp2, perms[permutation]);
            int neg, pos;
            count_signs(temp2, &neg, &pos);
            // if (neg > pos ){
            //     apply_row_signs_to_matrix(temp2, temp3, flipAllRows);
            // }
            if (!bestInit || pos > bestPosCount || (pos == bestPosCount) && (compare_matrices_lexicographical(temp2, best) < 0)) {
                memcpy(best, temp2, sizeof(structureMatrix));
                bestInit = 1;
                bestPosCount = pos;
                // printf("%3d %3d \n", sign_pattern, permutation);
                // print_ascii(best);
            }
            if (permutation==119 && sign_pattern==15){
                // printf("done \n");
            }
        }
    }
    memcpy(out, best, sizeof(structureMatrix));
    // printf("\n");
}

bool isUnique(structureMatrix candidate, unsigned long numUnique, unsigned char* allUnique){
    for (int i=0; i < numUnique; i++) {
        unsigned char* ptr = allUnique + i *numBytes;
        if (memcmp(candidate, ptr, numBytes) == 0) {
            return false;
        }
    }
    return true;
}

void generate_matrix_from_index(unsigned long index, structureMatrix twoBitArray) {
    // create a structure matrix and set it to zero
    // structureMatrix* twoBitArray = (structureMatrix*)malloc(sizeof(structureMatrix));  
    memset(twoBitArray, 0x00, numBytes);
    // printf("======================================\n");
    // printf("%ld\n", index);
    // printf("======================================\n");
    
    // Assign a value to each element in column i and row j of the structure
    // matrix

    // decode each index slice into a base 3 representation
    unsigned long quotient = index;
    for (int i = 0; i < NCOLS; i++)
    // col i
    {
        for (int j = 0; j < NROWS; j++) {
        // row j
            // each digit of the base 3 number is val
            int val = quotient % 3;
            
            // OLD FORMAT WHERE WE ENCODE NEGATIVE NUMBERS DIRECTLY
            // val -= 1;
            
            // move to the next place-value in base 3
            quotient /= 3;
            // stick it in the array in the appropriate position
            twoBitArray[i] |= (val & 0b00000011) << (NROWS-(j+1))*2;
        }
    }
}

void main() {
    // Define amount of matrices to generate (current implementation is for
    // all possible combinations)
    unsigned long numMatrices = pow(3, (NROWS*NCOLS));
    unsigned long numUnique   = (numMatrices / 1920) + 1000;
    // unsigned long numMatrices = 1000;
    // allocate memory for a big (possibly multiple gigabytes) array containing 
    // each structure matrix
    unsigned char *combinationsList = (unsigned char *)malloc(numMatrices*numBytes);
    
    unsigned char *allUnique        = (unsigned char *)malloc(numUnique*numBytes);
    // Set everything in the big long list to 0
    
    memset(combinationsList, 0x00, numMatrices*numBytes);
    memset(allUnique, 0x00, numUnique*numBytes);
    
    // for each slice (1-d representation of a 2-d structure matrix) of the list
    // of all combinations
    unsigned char* rowSigns = generate_row_signs(4);
    generate_perms_rec(0,0,cur, perms, &count);
    numUnique = 0;
    for (unsigned long slice = 0; slice < numMatrices; slice++)
    {
        structureMatrix rawMatrix;
        generate_matrix_from_index(slice, rawMatrix);
        structureMatrix canonicalForm;
        canonical_form(rawMatrix, rowSigns, canonicalForm);
        // extract values in two different ways
        // print_ascii(rawMatrix);
        // printf("--------------------------------------------\n");
        // print_ascii(canonicalForm);
        // printf("--------------------------------------------\n");
        bool dup = isUnique(canonicalForm, numUnique, allUnique);
        if (dup) {
            memcpy(allUnique+(numUnique*numBytes), canonicalForm, sizeof(canonicalForm));
            numUnique++;
            // allUnique[numUnique] = canonicalForm;
            
        }
        // printf("%d \n", dup);
        // printf("--------------------------------------------\n");
        // print()
        // stick each structure matrix in the master list
        memcpy(combinationsList+(slice*numBytes), rawMatrix, sizeof(rawMatrix));
        if ((slice % 10000) == 0){
            printf("total processed %ld, unique found %ld                     \r", slice, numUnique);
            fflush(stdout);
        }
    } // end for slice
    // check size to make sure its consistent
    printf("%ld\n", numUnique);

    FILE *all = fopen("all4x5.out","w");
    size_t numAllElements = fwrite(combinationsList, numBytes, numMatrices, all);
    fclose(all);
    printf("wrote %ld elements to all\n", numAllElements);
    
    FILE *unique = fopen("unique4x5.out","w");
    size_t numUniqueElements = fwrite(allUnique, numBytes, numUnique, unique);
    fclose(unique);
    printf("wrote %ld elements to unique\n", numUniqueElements);

    // fwrite
}

// void main() {
//     unsigned char* rowSigns = generate_row_signs(4);
//     structureMatrix test, test1, test2, test3;  
//     memset(test, 0x00, numBytes);
//     // unsigned char  test_col  = 0x48;
//     for (int i = 0; i < NCOLS; i++)
//         {
//             for (int j = 0; j < NROWS; j++) {
//                 // test[i] += ((((j+i) % 3)) & 0b00000011) << (NROWS-(j+1))*2;
//                 if (j==0 && i==0) {
//                     test[i] += ((0b01) & 0b00000011) << (NROWS-(j+1))*2;
//                 }
//                 else {
//                     test[i] += ((0b00) & 0b00000011) << (NROWS-(j+1))*2;
//                 }
//             }
//         }
    
//     print_ascii(test);
//     printf("--------------------------------------------\n");
//     printf("--------------------------------------------\n");

//     // for (int i = 0; i < 16; i++) {
//     //     apply_row_signs_to_matrix(test, test1, rowSigns[i]);
//     //     // printf("%02X %02X %02X \n", row_signs[i], test_col, result);
//     //     print_ascii(test1);
//     //     printf("--------------------------------------------\n");
//     // }
//     // printf("--------------------------------------------\n");
//     generate_perms_rec(0, 0, cur, perms, &count);
//     // for (int i = 0; i < 10; i++) {
//     //     int *p = perms[i];
//     //     permute_columns(test, test2, p);
//     //     print_ascii(test2);
//     //     printf("--------------------------------------------\n");
//     // }
//     canonical_form(test, rowSigns, test3);
//     print_ascii(test3);
// }