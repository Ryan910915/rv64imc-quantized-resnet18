#ifndef XQMAC8_H
#define XQMAC8_H

#include <stdint.h>

/*
 * XQMAC8 — packed signed INT8 multiply-accumulate
 *
 * Encoding:
 *   CUSTOM_0 opcode = 0x0b
 *   funct3 = 0
 *   funct7 = 0
 *
 * Architectural semantics:
 *
 *   rd_old = signed value currently held in rd
 *   rd = rd_old
 *        + int8(rs1[ 7: 0]) * int8(rs2[ 7: 0])
 *        + int8(rs1[15: 8]) * int8(rs2[15: 8])
 *        + ...
 *        + int8(rs1[63:56]) * int8(rs2[63:56])
 *
 * Thus one instruction performs eight signed INT8 multiplies and accumulates
 * all eight products into the destination accumulator.
 */
static inline int32_t xqmac8(int32_t acc, uint64_t packed_a, uint64_t packed_w)
{
#ifdef __riscv
    /*
     * "+r" is essential: the register selected for rd contains the old
     * accumulator before the instruction and receives the new accumulator
     * afterwards.
     */
    int64_t rd = (int64_t)acc;

    __asm__ volatile (
        ".insn r 0x0b, 0, 0, %0, %1, %2"
        : "+r"(rd)
        : "r"(packed_a), "r"(packed_w)
    );

    return (int32_t)rd;
#else
    /* Host reference path for make host-test. */
    int64_t sum = (int64_t)acc;

    for (int i = 0; i < 8; ++i) {
        int8_t av = (int8_t)((packed_a >> (8 * i)) & 0xffu);
        int8_t wv = (int8_t)((packed_w >> (8 * i)) & 0xffu);
        sum += (int64_t)av * (int64_t)wv;
    }

    return (int32_t)sum;
#endif
}

/*
 * Load eight contiguous INT8 values as one packed 64-bit operand.
 *
 * On the RISC-V target this is deliberately one LD rather than a byte-packing
 * loop.  The selected ResNet-18 export uses aligned arena/blob bases and
 * aligned channel blocks for all XQMAC8-enabled convolutions.
 *
 * The host path assembles bytes explicitly so the host test has no alignment
 * or strict-aliasing dependency.
 */
static inline uint64_t xqmac8_load8(const int8_t *p)
{
#ifdef __riscv
    uint64_t value;

    __asm__ volatile (
        "ld %0, 0(%1)"
        : "=r"(value)
        : "r"(p)
        : "memory"
    );

    return value;
#else
    uint64_t value = 0;

    for (int i = 0; i < 8; ++i)
        value |= ((uint64_t)(uint8_t)p[i]) << (8 * i);

    return value;
#endif
}

#endif
