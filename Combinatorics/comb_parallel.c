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
// #undef TEST
#define TEST

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

