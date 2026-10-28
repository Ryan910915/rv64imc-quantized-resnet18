#ifndef RESNET_FULL_NN_H
#define RESNET_FULL_NN_H
#include <stdint.h>
#include <stddef.h>
/* Dimensions are stored in their original order, padded with trailing ones.
 * Conv/pooling activations are NHWC; Conv filters are OHWI. */
typedef struct {
    int32_t dims[4];
    uint32_t rank, count, offset, bytes;
    int32_t zero_point;
    uint32_t is_const;
} Tensor;
typedef enum {
    NN_ADD=0, NN_AVGPOOL=1, NN_CONV=3, NN_FC=9, NN_MAXPOOL=17,
    NN_RELU=19, NN_RELU1=20, NN_RELU6=21, NN_RESHAPE=22,
    NN_PAD=34, NN_TRANSPOSE=39, NN_PADV2=60
} OpKind;
typedef struct {
    int32_t kind, in[3], out;
    int32_t stride[2], dilation[2], kernel[2], pad[4];
    int32_t act_min, act_max;
    uint32_t q_offset, q_count;
    /* ADD: input1 multiplier/shift, input2 multiplier/shift,
     * output multiplier/shift, common left shift. */
    int32_t add[7], perm[4], pads[8], fill;
    uint64_t dense_macs, valid_macs;
    uint32_t ref_offset;
    /* BUILTIN_REF FC uses binary64 scaling then round-away-from-zero.
     * Store the scale as integer significand/exponent, not target FP values. */
    uint32_t fc_reference, fc_scale_offset;
} Op;
extern const Tensor nn_tensors[];
extern const Op nn_ops[];
extern const unsigned char nn_model_blob[], nn_reference_blob[];
extern unsigned char nn_arena[];
const int8_t *nn_tensor_input(int32_t id);
int8_t *nn_tensor_output(int32_t id);
const char *nn_kind_name(int32_t kind);
int32_t nn_requantize(int32_t x, int32_t multiplier, int32_t shift);
int32_t nn_requantize_fc_reference(int32_t x, uint64_t significand,
                                    int32_t exponent);
int nn_execute(const Op *op);
#endif
