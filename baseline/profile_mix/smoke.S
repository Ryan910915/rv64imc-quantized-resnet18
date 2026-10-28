/* Two deliberately small, independently countable ROIs: 24 and 13 retired
 * instructions. Counter reads serialize in Spike; observer must not double count.
 * Also checks compressed loads/stores, multiply, branch, divide, and fence. */
.option norelax
.option norvc
.section .text.init,"ax",@progbits
.globl _start
.type _start,@function
_start:
    la sp,stack_top
    .word 0x32001073
    call platform_instret
    mv s0,a0
    li t0,3
    la t1,scratch
    li t2,7
1:
    sw t2,0(t1)
    lw t3,0(t1)
    mul t4,t3,t2
    addi t0,t0,-1
    bnez t0,1b
    call platform_instret
    sub t0,a0,s0
    li t1,24
    bne t0,t1,fail
    call platform_instret
    mv s0,a0
.option push
.option rvc
    c.li a1,5
    c.swsp a1,0(sp)
    c.lwsp a2,0(sp)
    c.add a2,a1
    c.slli a2,1
    div a3,a2,a1
    rem a4,a2,a1
    fence rw,rw
.option pop
    call platform_instret
    sub t0,a0,s0
    li t1,13
    bne t0,t1,fail
    li t0,1
    j exit_test
fail:
    li t0,7
exit_test:
    la t1,tohost
    fence rw,rw
    sd t0,0(t1)
2:  j 2b
.size _start,.-_start
.balign 4
.globl platform_instret
.type platform_instret,@function
platform_instret:
    .word 0xb0202573
    ret
.size platform_instret,.-platform_instret
.section .tohost,"aw",@progbits
.balign 64
.globl tohost
tohost:.dword 0
.balign 64
.globl fromhost
fromhost:.dword 0
.section .bss,"aw",@nobits
.balign 16
scratch:.space 16
stack_area:.space 4096
stack_top:.space 16
.section .note.GNU-stack,"",@progbits
