#include <stdint.h>
#include "xqmac8.h"

static uint64_t pack8(const int8_t x[8])
{
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i)
        v |= ((uint64_t)(uint8_t)x[i]) << (8 * i);
    return v;
}

static int32_t reference_mac8(int32_t acc,
                              const int8_t a[8],
                              const int8_t b[8])
{
    int64_t sum = acc;
    for (int i = 0; i < 8; ++i)
        sum += (int64_t)a[i] * (int64_t)b[i];
    return (int32_t)sum;
}

static int check(int32_t acc, const int8_t a[8], const int8_t b[8])
{
    int32_t expected = reference_mac8(acc, a, b);
    int32_t actual = xqmac8(acc, pack8(a), pack8(b));
    return actual == expected;
}

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
    {
        const int8_t a[8] = {1,2,3,4,5,6,7,8};
        const int8_t b[8] = {1,1,1,1,1,1,1,1};
        if (!check(100, a, b)) return 1; /* 100 + 36 */
    }

    {
        const int8_t a[8] = {1,-2,3,-4,5,-6,7,-8};
        const int8_t b[8] = {-8,7,-6,5,-4,3,-2,1};
        if (!check(-777, a, b)) return 2;
    }

    {
        const int8_t a[8] = {-128,-128,-128,-128,-128,-128,-128,-128};
        const int8_t b[8] = {-128,-128,-128,-128,-128,-128,-128,-128};
        if (!check(123456, a, b)) return 3;
    }

    {
        const int8_t a[8] = {-128,-128,-128,-128,-128,-128,-128,-128};
        const int8_t b[8] = {127,127,127,127,127,127,127,127};
        if (!check(-123456, a, b)) return 4;
    }

    for (int test = 0; test < 1000; ++test) {
        int8_t a[8], b[8];
        int32_t acc = (int32_t)rng_next();

        /* Keep randomized accumulators in a conservative range for this test. */
        acc >>= 8;

        for (int i = 0; i < 8; ++i) {
            a[i] = (int8_t)(rng_next() & 0xff);
            b[i] = (int8_t)(rng_next() & 0xff);
        }

        if (!check(acc, a, b)) return 10;
    }

    return 0;
}
