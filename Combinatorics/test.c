#include <stdio.h>

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "hash.h"
#include "structureMatrix.h"
#include "expandedMatrix.h"

typedef struct TwoBit
{
    char value;
} TwoBit;

// typedef twoBitInt towBitIntArray

// unsigned int test : 2;

// void print_ascii (structureMatrix out) {
// # ifdef TRACE
// printf("print_ascii()\n");
// # endif // TRACE
//     int cell;
//     for (int j = 0; j < NROWS; j++)
//     {
//         for (int i = 0; i < NCOLS; i++) {
//             cell = getValue(out[i], j, NROWS);
//             printf("%2d ", cell);
//         }
//         printf("\n");
//     }
// }



void main() {
    // int result = cantor(19, 27);
    // printf("%2d, \n", result);
    // printf("%ld: size of the stupid square array you're trying to make\n", sizeof(twoBitSquareArray));


    // check_entry_size();

    // structureMatrix testMatrix =  {0,0,0,0,0};
    // structureMatrix testMatrix2 = {0,0,0,0,1};
    // structureMatrix testMatrix = {0,0,0,1,1};
    // structureMatrix testMatrix4 = {0,0,1,0,1};
    // structureMatrix testMatrix5 = {0,1,0,0,1};
    
    FILE*fptr;
    fptr = fopen("unique4x5.out", "rb");
    fseek(fptr, 0, SEEK_END);
    size_t length = ftell(fptr);
    rewind(fptr);

    size_t numMatrices = length/sizeof(structureMatrix);
    printf("%ld \n", numMatrices);

    // *buffer = (unsigned char*)malloc(length + 1);
    structureMatrix *uniqueList = malloc(numMatrices * numBytes);
    size_t read_bytes = fread(uniqueList, numBytes, numMatrices, fptr);
    // printf("%d, \n", uniqueList[0][0]);
    print_ascii(uniqueList[0]);
    char expanded[4][5];
    expand_structMatrix(uniqueList[0], expanded);
    print_expanded(expanded);
    // PRINT_ARRAY(expandedTestMatrix, NROWS, NCOLS);

    // printf("")

    // if (testMatrix == NULL) {
    //     printf("waow\n");
    // }

    // int cell;
    // for (int j = 0; j < NROWS; j++)
    // {
    //     for (int i = 0; i < NCOLS; i++) {
    //         cell = getValue(testMatrix[i], j, NROWS);
    //         printf("%2d ", cell);
    //     }
    //     printf("\n");
    // }
    // int value = memcmp(testMatrix, zeroMatrix, numBytes);
    // printf("%d value means true, they're the same\n", value);
    // value = memcmp(testMatrix, testMatrix2, numBytes);
    // printf("%d value means flase, they're different\n", value);
    // Set *testSet = create_set(4);
    // printf("%ld of %ld now\n", count_set(testSet), capacity_set(testSet));
    // int ret = set_insert(testSet, testMatrix);
    // printf("%ld of %ld now\n", count_set(testSet), capacity_set(testSet));
    // ret = set_insert(testSet, testMatrix2);
    // printf("%ld of %ld now\n", count_set(testSet), capacity_set(testSet));
    // ret = set_insert(testSet, testMatrix2);
    // printf("%ld of %ld now\n", count_set(testSet), capacity_set(testSet));
    // ret = set_insert(testSet, testMatrix3);
    // printf("%ld of %ld now\n", count_set(testSet), capacity_set(testSet));
    // ret = set_insert(testSet, testMatrix4);
    // printf("%ld of %ld now\n", count_set(testSet), capacity_set(testSet));
    // ret = set_insert(testSet, testMatrix5);
    // printf("%ld of %ld now\n", count_set(testSet), capacity_set(testSet));
    // ret = set_insert(testSet, testMatrix6);
    // printf("%ld of %ld now\n", count_set(testSet), capacity_set(testSet));
    // ret = set_insert(testSet, testMatrix6);
    // printf("%ld of %ld now\n", count_set(testSet), capacity_set(testSet));
    // ret = set_insert(testSet, testMatrix2);
    // printf("%ld of %ld now\n", count_set(testSet), capacity_set(testSet));

    // TwoBit twoBitArray[4][5];

    

    // for (int i = 0; i < 4; i++) {
    //     for (int j = 0; j < 5; j++) {
    //         twoBitArray[i][j].value = ((i+j) % 3) -1;
    //     }
    // }
    // for (int i = 0; i < 4; i++) {
    //     for (int j = 0; j < 5; j++) {
    //         printf("%2d ", twoBitArray[i][j].value);
    //     }
    //     printf("\n");
    // }
    // printf("%ld\n", sizeof(twoBitArray));
    // printf("%ld\n", sizeof(TwoBit));
    // // printf("%ld\n", sizeof(test));

}