#include "hash.h"
#include "structureMatrix.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <omp.h>

int cantor(int a, int b) {
    return 1/2*(a+b)*(a+b+1)+b;
}

//attempting packed struct:
typedef struct __attribute__((packed)) {
    uint32_t hashValue;
    structureMatrix key;
} Entry ;

// typedef struct {
//     unsigned long hashValue;
//     structureMatrix key;
// } Entry;

struct Set {
    Entry *entries;
    size_t capacity;
    size_t count;
};

Set *create_set(size_t cap) {
    Set *s = malloc(sizeof(Set));
    if (!s) {
        printf("set malloc failed\n");
    }
    s->capacity = cap;
    s->count = 0;
    s->entries = calloc(cap, sizeof(Entry));
    return s;
}

/* FNV-1 hash from wikipedia https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function:  */
uint32_t fnv_1_hash(const structureMatrix array) {
    unsigned long hash = 0xcbf29ce484222325;
    for (int byte = 0; byte < NCOLS; byte++) {
        hash ^= array[byte];
        hash *= 0x100000001b3;
    }
    return (uint32_t)(hash >> 32);
}

int set_insert_entry(Set *s, const structureMatrix matrix) {
    // printf("creating hash, ");
    uint32_t hash = fnv_1_hash(matrix);

    size_t i = hash & (s->capacity - 1); // start at the right 'bucket'
    
    // Comment these out for alternate attempt, we will be doing this in a
    // smarter way for this
    // size_t probes = 0;
    // const size_t max_probes = s->capacity;  // one full loop is the limit
    
    // ALTERNATE ATTEMPT
    
    // printf("attempting insert, ");
    Entry *e = &s->entries[i];
    // Loop until we find an empty entry
    while (memcmp(e->key, zeroMatrix, numBytes) != 0) {
        // Check if we've found a duplicate
        if (memcmp(e->key, matrix, numBytes) == 0) {
            return 0; // duplicate
        }
        // If not, move to next slot and wrap if we have to
        i = (i + 1) & (s->capacity - 1);
        //  Update which entry we're looking at
        e = &s->entries[i];
    }
    // Didn't find any duplicates, so we can put an entry in this empty slot
    
    e->hashValue = hash;
    memcpy(e->key, matrix, numBytes);
    s->count++;
    return 1; // inserted new value
}

// returns a true on success and a false on failure
bool set_expand(Set* s) {
    printf("called set_expand\n");
    size_t new_capacity = s->capacity * 2;
    size_t old_capacity = s->capacity;
    if (new_capacity < s-> capacity) {
        printf("data type overflow for capacity %ld\n", new_capacity);
        return false; // overflow
    }
    Entry *new_entries = realloc(s->entries, new_capacity*sizeof(Entry));
    if (new_entries == NULL){
        printf("not enough memory for %ld matrices in hash table\n", new_capacity);
        return false; // Failure if calloc fails?
    }

    s->entries = new_entries;
    s->capacity = new_capacity;

    memset(s->entries + old_capacity, 0, old_capacity * sizeof(Entry));
    return true;

    // for (size_t i = 0; i<s->capacity; i++) {
    //     Entry *e = &s->entries[i];
    //     if (memcmp(e->key, zeroMatrix, numBytes) != 0) {
    //         // return true;
    //         set_insert_entry()
    //     }
    // }
}


int set_insert(Set *s, const structureMatrix matrix) {
    // printf("creating hash, ");
    if (s->count+1 > s->capacity) {
       if (!set_expand(s)) {
        fprintf(stderr, "[tid %d] set_insert FAILED: ran out of memory.\n capacity=%zu count=%zu \n",
                            omp_get_thread_num(), s->capacity, s->count);
        fflush(stderr);
        return -1; // error
       } 
    }
    
    return set_insert_entry(s, matrix);

    // OLD IMPLEMENTATION

    // for (;;) {
    //     // printf(".");
    //     Entry *e = &s->entries[i];

    //     if (!e->hashValue) {
    //         e->hashValue = hash;
    //         memcpy(e->key, matrix, numBytes);
    //         s->count++;
    //         return 1; // new entry
    //     }

    //     if (e->hashValue == hash &&
    //         memcmp(e->key, matrix, numBytes) == 0) {
    //         return 0; // duplicate
    //     }

    //     if (++probes >= max_probes) {
    //         // diagnostic — include thread id so you know who triggered it
    //         fprintf(stderr, "[tid %d] set_insert FAILED: capacity=%zu count=%zu probes=%zu\n",
    //                 omp_get_thread_num(), s->capacity, s->count, probes);
    //         fflush(stderr);
    //         return -1; // signal error
    //     }

    //     i = (i + 1) & (s->capacity - 1);
    // }
}

size_t set_memory_usage(const Set *s) {
    if (!s) return 0;

    size_t structuSize = sizeof(*s);
    size_t entriesSize = s->capacity * sizeof(Entry);

    // printf("%ld \n", structuSize);
    // printf("%ld * %ld\n", s->capacity , sizeof(Entry));

    return structuSize+ entriesSize;
}

size_t count_set(const Set *s) {
    return s->count;
}

size_t capacity_set(const Set *s) {
    return s->capacity;
}

void write_out_set_values(const Set *s, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) { perror("fopen"); return; }

    size_t numWritten = 0;
    for (size_t i = 0; i < s->capacity; i++) {
        Entry *e = &s->entries[i];
        if (e->hashValue != 0) {
            fwrite(e->key, numBytes, 1, f);
            numWritten++;
        }
    }

    fclose(f);
    printf("Wrote %zu unique matrices to %s\n", numWritten, filename);
}


void merge_set_into(Set *dest, const Set *src)
{
    for (size_t i = 0; i < src->capacity; i++) {
        Entry *e = &src->entries[i];
        if (e->hashValue != 0) {
            set_insert(dest, e->key);
        }
    }
}

void check_entry_size() {
    printf("An entry takes up %ld bytes", sizeof(Entry));
}

void free_set(Set *s)
{
    if (!s) return;
    free(s->entries);
    free(s);
}