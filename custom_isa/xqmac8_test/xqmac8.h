#ifndef XQMAC8_H
#define XQMAC8_H

#include <stdint.h>

static inline int64_t xqmac8(uint64_t a, uint64_t b)
{
    int64_t result;

    __asm__ volatile (
        ".insn r 0x0b, 0, 0, %0, %1, %2"
        : "=r"(result)
        : "r"(a), "r"(b)
    );

    return result;
}

#endif
