#ifndef TEST_MAT_UTIL_H_
#define TEST_MAT_UTIL_H_

#include <string.h>

#include "mat.h"
#include "test_util.h"

static inline struct Mat *mat_from_bytes(int rows, int cols, int type,
                                         const void *bytes) {
    struct Mat *m = alloc_mat(rows, cols, type);
    memcpy(m->data, bytes, (size_t)rows * m->step[0]);
    return m;
}

static inline void fill_pattern(struct Mat *m, unsigned seed) {
    size_t n = (size_t)m->rows * m->step[0];
    for (size_t i = 0; i < n; i++) {
        m->data[i] = (uchar)(i * 37u + seed);
    }
}

// Uniform pseudo-random bytes (xorshift32). Unlike fill_pattern, neighbouring
// values are unrelated, which exercises every comparator of a sorting network
// and every rounding remainder.
static inline void fill_xorshift(struct Mat *m, uint32_t seed) {
    uint32_t s = seed * 2654435761u + 1u;
    size_t   n = (size_t)m->rows * m->step[0];
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        m->data[i] = (uchar)(s >> 24);
    }
}

#define CHECK_MAT_BYTES(mat, expected)                                         \
    CHECK_EQ_BYTES((mat)->data, (expected), (size_t)(mat)->rows *(mat)->step[0])

#endif // TEST_MAT_UTIL_H_
