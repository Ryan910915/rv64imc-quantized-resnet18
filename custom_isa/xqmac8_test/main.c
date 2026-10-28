#include <stdint.h>
#include "xqmac8.h"

static uint64_t pack8(const int8_t x[8])
{
    uint64_t v = 0;

    for (int i = 0; i < 8; ++i) {
        v |= ((uint64_t)(uint8_t)x[i]) << (8 * i);
    }

    return v;
}

static int64_t reference_dot8(const int8_t a[8],
                              const int8_t b[8])
{
    int64_t sum = 0;

    for (int i = 0; i < 8; ++i) {
        sum += (int64_t)a[i] * (int64_t)b[i];
    }

    return sum;
}

static int check(const int8_t a[8], const int8_t b[8])
{
    int64_t expected = reference_dot8(a, b);
    int64_t actual = xqmac8(pack8(a), pack8(b));

    return actual == expected;
}


/* Deterministic RNG for randomized verification. */
static uint32_t rng_state = 0x12345678u;

static uint32_t rng_next(void)
{
    uint32_t x = rng_state;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;

    rng_state = x;
    return x;
}


int main(void)
{
    /* Directed test 1: simple positive values. */
    {
        const int8_t a[8] = {1,2,3,4,5,6,7,8};
        const int8_t b[8] = {1,1,1,1,1,1,1,1};

        if (!check(a, b))
            return 1;
    }

    /* Directed test 2: mixed signs. */
    {
        const int8_t a[8] = {1,-2,3,-4,5,-6,7,-8};
        const int8_t b[8] = {-8,7,-6,5,-4,3,-2,1};

        if (!check(a, b))
            return 2;
    }

    /* Directed test 3: zero. */
    {
        const int8_t a[8] = {0,0,0,0,0,0,0,0};
        const int8_t b[8] = {-128,127,-1,1,50,-50,10,-10};

        if (!check(a, b))
            return 3;
    }

    /* Directed test 4: positive extreme. */
    {
        const int8_t a[8] =
            {-128,-128,-128,-128,-128,-128,-128,-128};

        const int8_t b[8] =
            {-128,-128,-128,-128,-128,-128,-128,-128};

        if (!check(a, b))
            return 4;
    }

    /* Directed test 5: negative extreme. */
    {
        const int8_t a[8] =
            {-128,-128,-128,-128,-128,-128,-128,-128};

        const int8_t b[8] =
            {127,127,127,127,127,127,127,127};

        if (!check(a, b))
            return 5;
    }

    /* Randomized verification. */
    for (int test = 0; test < 1000; ++test) {
        int8_t a[8];
        int8_t b[8];

        for (int i = 0; i < 8; ++i) {
            a[i] = (int8_t)(rng_next() & 0xff);
            b[i] = (int8_t)(rng_next() & 0xff);
        }

        if (!check(a, b))
            return 10;
    }

    return 0;
}