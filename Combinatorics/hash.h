#ifndef HASH_FUNCTIONS_H
#define HASH_FUNCTIONS_H

#include <stddef.h>
#include "structureMatrix.h"

// make hashSet structure available
typedef struct Set Set;

// create a set of a certain capacity
Set *create_set(size_t cap);

// dumb function I used for testing stuff
int cantor(int a, int b);

// insert a matrix into the set
int set_insert(Set *s, const structureMatrix key);

// count entries in set
size_t count_set(const Set *s);

// count capacity of set
size_t capacity_set(const Set *s);

// write out
void write_out_set_values(const Set *s, const char *filename);

// free set memory
void free_set(Set *s);

// merge sets
void merge_set_into(Set *dest, const Set *src);

// return memory usage of a given Set
size_t set_memory_usage(const Set *s);

// Prove
void check_entry_size();

#endif // HASH_FUNCTIONS_H