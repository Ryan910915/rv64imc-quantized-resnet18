#include "platform.h"
#ifdef HOST_TEST
#include <stdio.h>
#include <stdlib.h>
static void putchar_raw(char c) { (void)fputc(c,stdout); }
#else
extern volatile uint64_t tohost,fromhost;
static void putchar_raw(char c) {
    /* HTIF basic character device = 1, write command = 1.
     * The Spike BCD write handler clears tohost but sends no fromhost reply.
     * This is console I/O, NOT a proxy kernel or runtime filesystem. */
    while(tohost!=0) { }
    __asm__ volatile("fence rw,rw" ::: "memory");
    tohost=(UINT64_C(1)<<56)|(UINT64_C(1)<<48)|(unsigned char)c;
    while(tohost!=0) { }
}
#endif
void platform_puts(const char *s) {
    while(*s) putchar_raw(*s++);
#ifdef HOST_TEST
    (void)fflush(stdout);
#endif
}
void platform_u64(uint64_t v) {
    char text[21]; int n=0;
    do { text[n++]=(char)('0'+v%10);v/=10; } while(v);
    while(n) putchar_raw(text[--n]);
}
void platform_i64(int64_t v) {
    if(v<0) { putchar_raw('-'); platform_u64((uint64_t)(-(v+1))+1); }
    else platform_u64((uint64_t)v);
}
void platform_hex(uint64_t v) {
    const char *digits="0123456789abcdef";
    platform_puts("0x");
    for(int n=15;n>=0;--n) putchar_raw(digits[(v>>(n*4))&15]);
}
#if defined(HOST_TEST) || !PROFILE
uint64_t platform_instret(void) { return 0; }
#endif
_Noreturn void platform_trap(uint64_t cause,uint64_t pc,uint64_t value) {
    platform_puts("\nTRAP cause=");platform_hex(cause);
    platform_puts(" pc=");platform_hex(pc);
    platform_puts(" tval=");platform_hex(value);platform_puts("\n");
#ifdef HOST_TEST
    exit(125);
#else
    while(tohost) { }
    tohost=(UINT64_C(125)<<1)|1;
    for(;;) { }
#endif
}
