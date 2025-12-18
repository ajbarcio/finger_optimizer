#include <stdio.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <stdbool.h>
#include <stdint.h>
#include <omp.h>

#include "hash.h"
#include "structureMatrix.h"

// define trace to reveal function calls
#undef TRACE
// #define TRACE

// define test to run for shorter nuber of matrices
#undef TEST
// #define TEST

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

int perms[120][5];
int cur[5];
int count = 0;

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

#define permute_columns(in, out, perm) \
{ \
    for (int c = 0; c< NCOLS; c++) { \
        out[c] = in[perm[c]]; \
    } \
}

void permute_columnsF(const structureMatrix in, structureMatrix out, const int perm[NCOLS]) {
# ifdef TRACE
    printf("permute_columnsF()\n");
# endif // TRACE
    for (int c=0; c<NCOLS; c++) {
        out[c] = in[perm[c]];
    }
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

int compare_matrices_lexicographicalF(const structureMatrix a, const structureMatrix b) {
# ifdef TRACE
    printf("compare_matrices_lexicographical()\n");
# endif // TRACE
    for (int c=0; c<NCOLS; c++) {
        if (a[c] < b[c]) return -1;
        if (a[c] > b[c]) return 1;
    }
    return 0;
}

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


void count_signsF(const structureMatrix m, int* neg, int* pos) {
# ifdef TRACE
    printf("count_signs()\n");
# endif // TRACE
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

void generate_matrix_from_index(unsigned long index, structureMatrix twoBitArray) {
# ifdef TRACE
    printf("generate_matrix_from_index()\n");
# endif // TRACE
    // create a structure matrix and set it to zero
    // structureMatrix* twoBitArray = (structureMatrix*)malloc(sizeof(structureMatrix));  
    memset(twoBitArray, 0x00, numBytes);
    // printf("======================================\n");
    // printf("%ld\n", index);
    // printf("======================================\n");
    
    // Assign a value to each element in column i and row j of the structure
    // matrix

    // decode each index encodedIndex into a base 3 representation
    unsigned long quotient = index;
    int val = quotient % 3;
    for (int i = 0; i < NCOLS; i++)
    // col i
    {
        for (int j = 0; j < NROWS; j++) {
        // row j
            // each digit of the base 3 number is val
            val = quotient % 3;
            
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
# ifdef TRACE
    printf("main()\n");
# endif // TRACE
    // Define amount of matrices to generate (current implementation is for
    // all possible combinations)
    # ifdef TEST
        unsigned long numMatrices = 1000000000/2;
        // unsigned long expectedUniques = numMatrices/2;
        // unsigned long expected
        // unsigned long numUnique = (unsigned long)pow(2,(int)log2(expectedUniques));
        // printf("%ld \n", numUnique);
    # else
        unsigned long numMatrices = pow(3, (NROWS*NCOLS));
        // unsigned long expectedUniques = numMatrices/1920*2;
        // unsigned long numUnique   = (unsigned long)(pow(2,log2(expectedUniques)));
    # endif

    
    int numThreads = omp_get_max_threads(); // this is set by an environment variable
    // Create pointers to a bunch of different Sets, lists of Combinations, and Counts, one for each thread
    Set **threadSets = malloc(numThreads * sizeof(Set *));
    // printf("%ld: bytes memory set aside for hash tables\n", numThreads * sizeof(Set *));
    unsigned char **threadCombLists = malloc(numThreads * sizeof(unsigned char *));
    // printf("%ld: bytes memory set aside for lists\n", numThreads * sizeof(unsigned char *));
    unsigned long *threadCounts = calloc(numThreads, sizeof(unsigned long));
    // printf("%ld\n", numThreads * sizeof(unsigned long));
    // Initialize the lists and sets
    unsigned long entriesPerThread = (unsigned long)pow(2,floor(log2(numMatrices/numThreads)));
    unsigned long totalSetMemory = 0;
    unsigned long totalCombMemory = 0;
    printf("each of %d thread gets %ld entries in its hash table \n", numThreads, entriesPerThread);
    for (int t=0; t < numThreads; t++) {
        threadSets[t] = create_set(entriesPerThread);
        totalSetMemory += set_memory_usage(threadSets[t]);
        
        threadCombLists[t] = malloc((numMatrices/numThreads+1) * numBytes);
        totalCombMemory+=(numMatrices/numThreads+1) * numBytes;
    }
    printf("%lu: size set aside for all Sets  I think\n", totalSetMemory);
    printf("%lu: size set aside for all Lists I think\n", totalCombMemory);
    // printf("%ld: size of all Sets I think\n", sizeof(*threadSets));
    // printf("%ld: size of all master Arrays I think\n", sizeof(*threadCombLists));

    ///////////////////////////// THIS IS ALL OLD, SERIAL CODE ///////////////////////////////
    // // allocate memory for a big (possibly multiple gigabytes) array containing 
    // // each structure matrix
    // unsigned char *combinationsList = (unsigned char *)malloc(numMatrices*numBytes);
    // // Set everything in the big long list to 0
    // memset(combinationsList, 0x00, numMatrices*numBytes);
    
    // // create a hash set for the unique structure matrices
    // Set *uniqueSet = create_set(numUnique);

    // unsigned char *allUnique        = (unsigned char *)malloc(numUnique*numBytes);
    // memset(allUnique, 0x00, numUnique*numBytes);
    
    /////////////////////////////// END SERIAL CODE ///////////////////////////////////////

    //prepare permutations and row signs for the isomorphisms
    unsigned char* rowSigns = generate_row_signs(4);
    generate_perms_rec(0,0,cur, perms, &count);
    //print for each thread every 1% they're done
    unsigned long printInterval = numMatrices / numThreads / 1000;
    // unsigned long total_across_all_threads, localCount;
    unsigned long localCount;
    
    #pragma omp parallel private(localCount) //shared(total_across_all_threads)
    {
        // total_across_all_threads = 0;
        int thread_id = omp_get_thread_num();
        
        Set *localSet = threadSets[thread_id];
        unsigned char *localCombList = threadCombLists[thread_id];
        localCount = 0;

        // fprintf(stderr, "[tid %d] parallel region start\n", thread_id);
        // fflush(stderr);

        #pragma omp for schedule(static)
        for (unsigned long encodedIndex=0; encodedIndex < numMatrices; encodedIndex++) {
            structureMatrix rawMatrix;
            structureMatrix canonicalForm;
            
            generate_matrix_from_index(encodedIndex, rawMatrix);
            canonical_form(rawMatrix, rowSigns, canonicalForm);

            set_insert(localSet, canonicalForm);

            memcpy(&localCombList[localCount*numBytes], rawMatrix, sizeof(rawMatrix));
            localCount++;

            if (encodedIndex % printInterval == 0) {
                printf("thread %d is %.2f%% done, processed %ld matrices\n", 
                                    omp_get_thread_num(), (float)localCount/(numMatrices/numThreads)*100, localCount);
            }
        }
        // #pragma omp critical
        // {
        //     total_across_all_threads += localCount;
        //     printf("%ld total matrices have been processed \n", 
        //                         total_across_all_threads);
            
        // }
        threadCounts[thread_id] = localCount;
        // fprintf(stderr, "[tid %d] parallel region end, localCount = %ld\n", thread_id, localCount);
        // fflush(stderr);
    } // END THREADS
    // printf("'escaped for loop'");
    Set *globalSet = create_set(entriesPerThread*numThreads);

    for (int t = 0; t < numThreads; t++) {
        merge_set_into(globalSet, threadSets[t]);
        free_set(threadSets[t]);
    }

    unsigned char *combinationsList = (unsigned char *)malloc(numMatrices*numBytes);
    unsigned long offset = 0;
    for (int t = 0; t < numThreads; t++) {
        memcpy(&combinationsList[offset * numBytes], 
                threadCombLists[t],
                threadCounts[t] * numBytes);
        offset +=(threadCounts[t]);
        free(threadCombLists[t]);
    }
    free(threadCombLists);

    FILE *all = fopen("all4x5.out","w");
    size_t numAllElements = fwrite(combinationsList, numBytes, numMatrices, all);
    fclose(all);
    printf("wrote %ld elements to all\n", numAllElements);
    
    write_out_set_values(globalSet, "unique4x5.out");

    // fwrite
}

