#ifndef RESNET_PLATFORM_H
#define RESNET_PLATFORM_H
#include <stdint.h>
void platform_puts(const char *text);
void platform_u64(uint64_t value);
void platform_i64(int64_t value);
void platform_hex(uint64_t value);
uint64_t platform_instret(void);
_Noreturn void platform_trap(uint64_t cause,uint64_t pc,uint64_t value);
#endif
