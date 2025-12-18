#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct {
    uint64_t *keys;
    size_t capacity;
    size_t count;
} HashSet40;

/* simple multiplicative hash */
static inline uint64_t hash40(uint64_t x) {
    return (x * 0x9E3779B97F4A7C15ULL) >> 24;  // mix, keep high bits
}

HashSet40 *hs40_create(size_t cap) {
    // must be >=2× expected number of uniques
    HashSet40 *s = malloc(sizeof(HashSet40));
    s->capacity = cap;
    s->count    = 0;
    s->keys     = calloc(cap, sizeof(uint64_t));  // zero = empty slot
    return s;
}

static inline uint64_t read40(const unsigned char bytes[5]) {
    uint64_t v = 0;
    memcpy(&v, bytes, 5);  // lower 40 bits
    return v;
}

// returns 1 if new unique inserted, 0 if already present
int hs40_insert(HashSet40 *s, uint64_t key) {
    if (key == 0) key = 1;              // reserve 0 as "empty"

    uint64_t h = hash40(key);
    size_t i = h & (s->capacity - 1);   // power-of-2 capacity recommended

    while (s->keys[i] != 0) {
        if (s->keys[i] == key) return 0; // already present
        i = (i + 1) & (s->capacity - 1);
    }

    s->keys[i] = key;
    s->count++;
    return 1;
}


