#ifndef TEST_EXPECT_ASSERT_H_
#define TEST_EXPECT_ASSERT_H_

#include <setjmp.h>

#include "test_util.h"
#include "utils.h"

// EXPECT_ASSERT(stmt) runs stmt with an assert handler that longjmps back and
// checks that exactly one ASSERT fired. Memory that stmt took from fast_malloc
// before the ASSERT is the caller's concern; the ops under test free their
// scratch before failing.
static jmp_buf expect_assert_jump;
static int     expect_assert_count;

static void expect_assert_handler(const char *expr, const char *file,
                                  int line) {
    (void)expr;
    (void)file;
    (void)line;
    expect_assert_count++;
    longjmp(expect_assert_jump, 1);
}

#define EXPECT_ASSERT(stmt)                                                    \
    do {                                                                       \
        expect_assert_count = 0;                                               \
        improc_set_assert_handler(expect_assert_handler);                      \
        if (setjmp(expect_assert_jump) == 0) {                                 \
            stmt;                                                              \
        }                                                                      \
        improc_set_assert_handler(NULL);                                       \
        if (expect_assert_count != 1) {                                        \
            TEST_FAIL_("expected one ASSERT from %s, got %d", #stmt,           \
                       expect_assert_count);                                   \
        }                                                                      \
    } while (0)

#endif // TEST_EXPECT_ASSERT_H_
