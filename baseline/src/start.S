    .section .text.init,"ax",@progbits
    .globl _start
    .type _start,@function
_start:
    .option push
    .option norelax
    la gp,__global_pointer$
    .option pop
    la sp,__stack_top
    la t0,trap_entry
    /* csrw mtvec,t0. Explicit encoding is accepted by both old and new
     * assemblers while workload code stays -march=rv64imc.
     * Spike is configured with the standard Zicsr system extension. */
    .word 0x30529073
    la t0,__bss_start
    la t1,__bss_end
1:  bgeu t0,t1,2f
    sd zero,0(t0)
    addi t0,t0,8
    j 1b
2:
#if PROFILE
    /* csrw mcountinhibit,zero: enable standard machine-mode counters. */
    .word 0x32001073
#endif
    call main
    slli a0,a0,1
    ori a0,a0,1
    la t0,tohost
3:  ld t1,0(t0)
    bnez t1,3b
    fence rw,rw
    sd a0,0(t0)
4:  j 4b
    .size _start,.-_start
    .balign 4
trap_entry:
    /* csrr a0,mcause; csrr a1,mepc; csrr a2,mtval */
    .word 0x34202573
    .word 0x341025f3
    .word 0x34302673
    call platform_trap
    j 4b
#if PROFILE
    .globl platform_instret
    .type platform_instret,@function
platform_instret:
    /* csrr a0,minstret. No cycle-accurate CV-Wally claim is made. */
    .word 0xb0202573
    ret
    .size platform_instret,.-platform_instret
#endif
    .section .tohost,"aw",@progbits
    .balign 64
    .globl tohost
tohost: .dword 0
    .balign 64
    .globl fromhost
fromhost: .dword 0
    .section .note.GNU-stack,"",@progbits
