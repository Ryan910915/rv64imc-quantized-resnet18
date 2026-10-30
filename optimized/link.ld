OUTPUT_ARCH(riscv)
ENTRY(_start)
/* Spike test memory, NOT a claim about the CV-Wally physical memory map. */
MEMORY { RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 64M }
SECTIONS {
    . = ORIGIN(RAM);
    .text : ALIGN(16) { KEEP(*(.text.init)) *(.text .text.*) } > RAM
    .rodata : ALIGN(16) { *(.rodata .rodata.* .srodata .srodata.*) } > RAM
    .tohost : ALIGN(64) { KEEP(*(.tohost)) } > RAM
    .data : ALIGN(16) {
        PROVIDE(__global_pointer$ = . + 0x800);
        *(.data .data.* .sdata .sdata.*)
    } > RAM
    .bss (NOLOAD) : ALIGN(16) {
        __bss_start = .;
        *(.bss .bss.* .sbss .sbss.*) *(COMMON)
        . = ALIGN(16); __bss_end = .;
    } > RAM
    .stack (NOLOAD) : ALIGN(16) {
        __stack_bottom = .; . += 256K; __stack_top = .;
    } > RAM
    ASSERT(__stack_top <= ORIGIN(RAM)+LENGTH(RAM), "Full model exceeds Spike RAM")
    /DISCARD/ : { *(.comment .note .note.* .eh_frame .eh_frame.*) }
}
