.section .rodata.nn_blobs,"a",@progbits
.balign 16
.global nn_model_blob
.type nn_model_blob, @object
nn_model_blob:
.incbin "generated/model.bin"
.size nn_model_blob, .-nn_model_blob
.balign 16
.global nn_reference_blob
.type nn_reference_blob, @object
nn_reference_blob:
.incbin "generated/reference.bin"
.size nn_reference_blob, .-nn_reference_blob
.section .note.GNU-stack,"",@progbits
