#ifndef XREQUANT_H
#define XREQUANT_H

#include <stdint.h>

static inline uint64_t xrequant_pack_params(
    int32_t multiplier,
    int32_t shift,
    int32_t zero_point,
    int32_t act_min,
    int32_t act_max)
{
    return
        ((uint64_t)(uint32_t)multiplier) |
        ((uint64_t)(uint8_t)(int8_t)shift
            << 32) |
        ((uint64_t)(uint8_t)(int8_t)zero_point
            << 40) |
        ((uint64_t)(uint8_t)(int8_t)act_min
            << 48) |
        ((uint64_t)(uint8_t)(int8_t)act_max
            << 56);
}


static inline int8_t xrequant_apply(
    int32_t accumulator,
    uint64_t params)
{
#ifdef __riscv

    int64_t result;

    __asm__ volatile (
        ".insn r 0x0b, 1, 0, %0, %1, %2"
        : "=r"(result)
        : "r"((int64_t)accumulator),
          "r"(params)
    );

    return (int8_t)result;

#else

    const int32_t multiplier =
        (int32_t)(uint32_t)
        (params & UINT64_C(0xffffffff));

    const int32_t shift =
        (int32_t)(int8_t)
        ((params >> 32) & UINT64_C(0xff));

    const int32_t zero_point =
        (int32_t)(int8_t)
        ((params >> 40) & UINT64_C(0xff));

    const int32_t act_min =
        (int32_t)(int8_t)
        ((params >> 48) & UINT64_C(0xff));

    const int32_t act_max =
        (int32_t)(int8_t)
        ((params >> 56) & UINT64_C(0xff));


    int left  = shift > 0 ? shift : 0;
    int right = shift < 0 ? -shift : 0;

    int64_t expanded =
        (int64_t)accumulator *
        (INT64_C(1) << left);

    int64_t product =
        expanded * (int64_t)multiplier;

    int64_t nudge =
        product >= 0
            ? (INT64_C(1) << 30)
            : 1 - (INT64_C(1) << 30);

    int32_t high =
        (int32_t)(
            (product + nudge) /
            (INT64_C(1) << 31));


    int32_t scaled = high;

    if (right != 0) {

        int64_t magnitude =
            high < 0
                ? -(int64_t)high
                :  (int64_t)high;

        magnitude =
            (magnitude +
             (INT64_C(1) << (right - 1)))
            >> right;

        scaled =
            (int32_t)(
                high < 0
                    ? -magnitude
                    : magnitude);
    }


    int64_t result =
        (int64_t)scaled +
        zero_point;

    if (result < act_min)
        result = act_min;

    if (result > act_max)
        result = act_max;

    return (int8_t)result;

#endif
}

#endif