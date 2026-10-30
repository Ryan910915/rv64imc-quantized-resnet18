#include "nn.h"
#include "xqmac8.h"
#include "xrequant.h"
#include <limits.h>

const int8_t *nn_tensor_input(int32_t id) {
    const Tensor *t=&nn_tensors[id];
    return (const int8_t *)(t->is_const ? nn_model_blob+t->offset : nn_arena+t->offset);
}
int8_t *nn_tensor_output(int32_t id) {
    return (int8_t *)(nn_arena+nn_tensors[id].offset);
}
static size_t nhwc(const Tensor *t,int n,int h,int w,int c) {
    return (((size_t)n*(size_t)t->dims[1]+(size_t)h)*(size_t)t->dims[2]+(size_t)w)*(size_t)t->dims[3]+(size_t)c;
}
static int8_t clamp(int64_t value,const Op *p) {
    if(value<p->act_min) value=p->act_min;
    if(value>p->act_max) value=p->act_max;
    return (int8_t)value;
}
/* Integer equivalent of the positive Q31 multiplier, TWO-stage rounding path.
 * The exporter proves INT32 accumulator and pre-left-shift bounds for MAC ops.
 * ADD uses a fixed safe 20-bit pre-shift. Do not substitute float arithmetic.
 * Division of signed integers follows C11 truncation toward zero. */
int32_t nn_requantize(int32_t x,int32_t multiplier,int32_t shift) {
    int left=shift>0?shift:0, right=shift<0?-shift:0;
    int64_t expanded=(int64_t)x*(INT64_C(1)<<left);
    int64_t product=expanded*(int64_t)multiplier;
    int64_t nudge=product>=0?(INT64_C(1)<<30):1-(INT64_C(1)<<30);
    int32_t high=(int32_t)((product+nudge)/(INT64_C(1)<<31));
    if(right==0) return high;
    int64_t magnitude=high<0?-(int64_t)high:(int64_t)high;
    magnitude=(magnitude+(INT64_C(1)<<(right-1)))>>right;
    return (int32_t)(high<0?-magnitude:magnitude);
}
/* Return the bit length without compiler runtime helpers or ISA extensions. */
static unsigned fc_bit_length(uint64_t x) {
    unsigned n=0;
    if(x>>32) { x>>=32; n+=32; }
    if(x>>16) { x>>=16; n+=16; }
    if(x>>8)  { x>>=8;  n+=8; }
    if(x>>4)  { x>>=4;  n+=4; }
    if(x>>2)  { x>>=2;  n+=2; }
    if(x>>1)  { x>>=1;  n+=1; }
    return n+(unsigned)x;
}
/* Match the LiteRT BUILTIN_REF INT8 FC expression
 * round(double(acc) * effective_output_scale), entirely with integers.
 *
 * The HOST exports a positive normal binary64 scale exactly as
 * significand * 2^exponent, with a 53-bit significand. This function forms the
 * exact (at most 84-bit) product using two 64-bit limbs. It first rounds the
 * product to binary64 precision (nearest, ties-to-even), then rounds to an
 * integer (ties-away-from-zero), as the reference kernel does.
 *
 * This is NOT Q31 single-rounding, nor a switch of golden/reference data.
 * No float/double, __int128, dynamic allocation, or FP runtime is used here.
 * The exporter limits supported FC scaled accumulators to the INT32 range;
 * saturation below is defensive and also makes this helper safe in tests. */
int32_t nn_requantize_fc_reference(int32_t x,uint64_t significand,int32_t exponent) {
    if(x==0 || significand==0) return 0;
    const int negative=x<0;
    const uint64_t a=negative?(uint64_t)(-(int64_t)x):(uint64_t)x;
    const uint64_t low_part=(significand & UINT64_C(0xffffffff))*a;
    const uint64_t high_part=(significand>>32)*a;
    const uint64_t lo=low_part+(high_part<<32);
    const uint64_t hi=(high_part>>32)+(lo<low_part);
    const unsigned bits=hi?64+fc_bit_length(hi):fc_bit_length(lo);
    const unsigned discard=bits>53?bits-53:0; /* 0..31 for these operand widths. */
    uint64_t rounded=lo;
    if(discard) {
        rounded=(lo>>discard)|(hi<<(64-discard));
        const uint64_t remainder=lo & ((UINT64_C(1)<<discard)-1);
        const uint64_t halfway=UINT64_C(1)<<(discard-1);
        if(remainder>halfway || (remainder==halfway && (rounded&1))) ++rounded;
    }
    /* rounded * 2^power is exactly the rounded binary64 product. */
    const int32_t power=exponent+(int32_t)discard;
    const uint64_t limit=negative?UINT64_C(2147483648):UINT64_C(2147483647);
    uint64_t magnitude;
    if(power>=0) {
        if(power>=31 || rounded>(limit>>(unsigned)power))
            return negative?INT32_MIN:INT32_MAX;
        magnitude=rounded<<(unsigned)power;
    } else {
        const unsigned right=(unsigned)(-(int64_t)power);
        /* rounded <= 2^53, including a possible carry from ties-to-even. */
        if(right>54) return 0;
        magnitude=(rounded+(UINT64_C(1)<<(right-1)))>>right;
    }
    if(magnitude>=limit) return negative?INT32_MIN:INT32_MAX;
    return negative?-(int32_t)magnitude:(int32_t)magnitude;
}
/*
 * XQMAC8 integration for the selected ResNet-18.
 *
 * XQMAC8 is a packed signed INT8 multiply-accumulate instruction:
 *   acc = acc + sum_{i=0..7} int8(a_i) * int8(w_i)
 *
 * The destination register is a read/write accumulator operand.
 *
 * The quantized Conv reference expression is:
 *   sum_i (a_i - input_zero_point) * w_i
 *
 * Because the weights are symmetric (weight zero-point = 0), for each valid
 * spatial kernel position we can rewrite this exactly as:
 *
 *   sum_i a_i*w_i - input_zero_point * sum_i w_i
 *
 * Weight sums depend only on (output-channel, ky, kx), so they are computed
 * once per Conv invocation and reused for all output pixels. This is
 * especially important for SAME-padding border pixels: only weight sums for
 * spatial kernel positions that are actually in bounds are applied.
 *
 * The largest filter-position table in this ResNet-18 is
 *   512 output channels * 3 * 3 = 4608 entries.
 * Keep the storage static so the bare-metal workload still uses no heap.
 */
#define NN_XQMAC8_MAX_WEIGHT_SUMS 4608
static int32_t nn_xqmac8_weight_sums[NN_XQMAC8_MAX_WEIGHT_SUMS];

/*
 * XREQUANT parameters are constant for each Conv output channel.
 * Pack them once per Conv invocation, then reuse the 64-bit parameter word
 * for every output element of that channel.
 */
#define NN_XREQUANT_MAX_CHANNELS 512
static uint64_t nn_xrequant_params[NN_XREQUANT_MAX_CHANNELS];

static int conv(const Op *p) {
    const Tensor *x=&nn_tensors[p->in[0]], *w=&nn_tensors[p->in[1]], *y=&nn_tensors[p->out];
    const int8_t *a=nn_tensor_input(p->in[0]), *b=nn_tensor_input(p->in[1]);
    const int32_t *bias=p->in[2]<0?NULL:(const int32_t *)(const void *)nn_tensor_input(p->in[2]);
    const int32_t *q=(const int32_t *)(const void *)(nn_model_blob+p->q_offset);
    const int32_t *sh=q+p->q_count;
    int8_t *out=nn_tensor_output(p->out);
    int ih=x->dims[1], iw=x->dims[2], ci=x->dims[3], co=y->dims[3];
    int kh=w->dims[1], kw=w->dims[2];

    /*
     * The first ResNet Conv has ci=3, so it stays on the original scalar path.
     * The dominant later convolutions have ci=64/128/256/512 and use XQMAC8.
     */
    const int use_xqmac8=(ci>=8);

    /*
     * Pre-pack the complete output requantization state used by XREQUANT:
     * multiplier, shift, output zero-point, activation min and activation max.
     * ResNet-18 has at most 512 Conv output channels.
     */
    if(co>NN_XREQUANT_MAX_CHANNELS) return 11;
    for(int oc=0;oc<co;++oc) {
        nn_xrequant_params[oc]=xrequant_pack_params(
            q[oc],
            sh[oc],
            y->zero_point,
            p->act_min,
            p->act_max);
    }

    /*
     * Precompute sum(weights) for each (oc,ky,kx) only when the asymmetric
     * activation zero point actually needs a correction.
     */
    if(use_xqmac8 && x->zero_point!=0) {
        size_t weight_sum_count=(size_t)co*(size_t)kh*(size_t)kw;
        if(weight_sum_count>NN_XQMAC8_MAX_WEIGHT_SUMS) return 10;

        for(int oc=0;oc<co;++oc) {
            for(int ky=0;ky<kh;++ky) {
                for(int kx=0;kx<kw;++kx) {
                    size_t wi=(((size_t)oc*(size_t)kh+(size_t)ky)*
                               (size_t)kw+(size_t)kx)*(size_t)ci;
                    int32_t sum=0;

                    for(int ic=0;ic<ci;++ic)
                        sum+=(int32_t)b[wi+(size_t)ic];

                    nn_xqmac8_weight_sums[
                        ((size_t)oc*(size_t)kh+(size_t)ky)*(size_t)kw+
                        (size_t)kx
                    ]=sum;
                }
            }
        }
    }

    for(int n=0;n<y->dims[0];++n) for(int oy=0;oy<y->dims[1];++oy)
    for(int ox=0;ox<y->dims[2];++ox) for(int oc=0;oc<co;++oc) {
        int32_t acc=bias?bias[oc]:0;

        for(int ky=0;ky<kh;++ky) {
            int iy=oy*p->stride[0]-p->pad[0]+ky*p->dilation[0];
            if(iy<0||iy>=ih) continue;

            for(int kx=0;kx<kw;++kx) {
                int ix=ox*p->stride[1]-p->pad[2]+kx*p->dilation[1];
                if(ix<0||ix>=iw) continue;

                size_t ai=nhwc(x,n,iy,ix,0);
                size_t wi=(((size_t)oc*(size_t)kh+(size_t)ky)*
                           (size_t)kw+(size_t)kx)*(size_t)ci;

                if(use_xqmac8) {
                    int ic=0;

                    /*
                     * Activations and OHWI weights are contiguous in the
                     * channel dimension.  The selected model's arena/model
                     * blob bases are 16-byte aligned, relevant tensor offsets
                     * are 8-byte aligned, and ic advances by 8, so target-side
                     * xqmac8_load8() can use one RV64 LD per packed operand.
                     */
                    for(;ic+7<ci;ic+=8) {
                        uint64_t av=xqmac8_load8(&a[ai+(size_t)ic]);
                        uint64_t bv=xqmac8_load8(&b[wi+(size_t)ic]);
                        acc=xqmac8(acc,av,bv);
                    }

                    /* Generic scalar tail for channel counts not divisible by 8. */
                    for(;ic<ci;++ic)
                        acc+=(int32_t)a[ai+(size_t)ic]*
                             (int32_t)b[wi+(size_t)ic];

                    /*
                     * Apply the activation zero-point correction once for this
                     * valid (ky,kx) position. Out-of-bounds positions are
                     * skipped above, matching the original Conv behavior.
                     */
                    if(x->zero_point!=0) {
                        size_t si=((size_t)oc*(size_t)kh+(size_t)ky)*
                                  (size_t)kw+(size_t)kx;
                        acc-=x->zero_point*nn_xqmac8_weight_sums[si];
                    }
                } else {
                    /*
                     * Original scalar path, retained for small channel counts
                     * such as the first RGB Conv (ci=3).
                     */
                    for(int ic=0;ic<ci;++ic) {
                        int32_t xv=(int32_t)a[ai+(size_t)ic]-x->zero_point;
                        acc+=xv*(int32_t)b[wi+(size_t)ic];
                    }
                }
            }
        }

        /*
         * XREQUANT performs the whole output post-processing path in one
         * custom instruction on RV64:
         *   fixed-point multiply -> rounding -> shift
         *   -> output zero-point -> saturation.
         *
         * The host path in xrequant.h executes the identical software
         * semantics for make host-test.
         */
        out[nhwc(y,n,oy,ox,oc)]=
            xrequant_apply(acc,nn_xrequant_params[oc]);
    }

    return 0;
}

static int fc(const Op *p) {
    const Tensor *x=&nn_tensors[p->in[0]], *w=&nn_tensors[p->in[1]], *y=&nn_tensors[p->out];
    const int8_t *a=nn_tensor_input(p->in[0]), *b=nn_tensor_input(p->in[1]);
    const int32_t *bias=p->in[2]<0?NULL:(const int32_t *)(const void *)nn_tensor_input(p->in[2]);
    const int32_t *q=(const int32_t *)(const void *)(nn_model_blob+p->q_offset), *sh=q+p->q_count;
    const uint64_t *sig=p->fc_reference?
        (const uint64_t *)(const void *)(nn_model_blob+p->fc_scale_offset):NULL;
    const int32_t *exp=p->fc_reference?
        (const int32_t *)(const void *)(sig+p->q_count):NULL;
    int8_t *out=nn_tensor_output(p->out);
    int co=w->dims[0], ci=w->dims[1];
    uint32_t batches=y->count/(uint32_t)co;
    for(uint32_t n=0;n<batches;++n) for(int oc=0;oc<co;++oc) {
        int32_t acc=bias?bias[oc]:0;
        for(int ic=0;ic<ci;++ic)
            acc+=((int32_t)a[(size_t)n*(size_t)ci+(size_t)ic]-x->zero_point)*(int32_t)b[(size_t)oc*(size_t)ci+(size_t)ic];
        const int32_t scaled=p->fc_reference?
            nn_requantize_fc_reference(acc,sig[oc],exp[oc]):
            nn_requantize(acc,q[oc],sh[oc]);
        out[(size_t)n*(size_t)co+(size_t)oc]=clamp((int64_t)scaled+y->zero_point,p);
    }
    return 0;
}
static int add(const Op *p) {
    const Tensor *x=&nn_tensors[p->in[0]], *z=&nn_tensors[p->in[1]], *y=&nn_tensors[p->out];
    const int8_t *a=nn_tensor_input(p->in[0]), *b=nn_tensor_input(p->in[1]);
    int8_t *out=nn_tensor_output(p->out);

    /*
     * Only the final ADD output requantization can be fused into XREQUANT.
     * The two input-side requantizations below must remain wide INT32 values
     * because they are added together before the final output saturation.
     */
    const uint64_t out_params=xrequant_pack_params(
        p->add[4],
        p->add[5],
        y->zero_point,
        p->act_min,
        p->act_max);

    /* Exporter rejects broadcast ADD; this graph uses same-shape residuals. */
    for(uint32_t i=0;i<y->count;++i) {
        int32_t av=((int32_t)a[i]-x->zero_point)*(INT32_C(1)<<p->add[6]);
        int32_t bv=((int32_t)b[i]-z->zero_point)*(INT32_C(1)<<p->add[6]);
        int32_t sum=nn_requantize(av,p->add[0],p->add[1])+nn_requantize(bv,p->add[2],p->add[3]);
        out[i]=xrequant_apply(sum,out_params);
    }
    return 0;
}
static int pool(const Op *p) {
    const Tensor *x=&nn_tensors[p->in[0]], *y=&nn_tensors[p->out];
    const int8_t *a=nn_tensor_input(p->in[0]); int8_t *out=nn_tensor_output(p->out);
    for(int n=0;n<y->dims[0];++n) for(int oy=0;oy<y->dims[1];++oy)
    for(int ox=0;ox<y->dims[2];++ox) for(int c=0;c<y->dims[3];++c) {
        int32_t sum=0, maximum=-128, count=0;
        for(int ky=0;ky<p->kernel[0];++ky) {
            int iy=oy*p->stride[0]-p->pad[0]+ky;
            if(iy<0||iy>=x->dims[1]) continue;
            for(int kx=0;kx<p->kernel[1];++kx) {
                int ix=ox*p->stride[1]-p->pad[2]+kx;
                if(ix<0||ix>=x->dims[2]) continue;
                int32_t v=a[nhwc(x,n,iy,ix,c)];
                sum+=v; if(v>maximum) maximum=v; ++count;
            }
        }
        if(!count) return 20;
        /* Average excludes out-of-bounds elements; ties round away from zero. */
        int32_t v=p->kind==NN_MAXPOOL?maximum:(sum>0?(sum+count/2)/count:(sum-count/2)/count);
        out[nhwc(y,n,oy,ox,c)]=clamp(v,p);
    }
    return 0;
}
static int transpose(const Op *p) {
    const Tensor *x=&nn_tensors[p->in[0]], *y=&nn_tensors[p->out];
    const int8_t *a=nn_tensor_input(p->in[0]); int8_t *out=nn_tensor_output(p->out);
    uint32_t strides[4]={1,1,1,1};
    for(int j=(int)x->rank-2;j>=0;--j) strides[j]=strides[j+1]*(uint32_t)x->dims[j+1];
    for(uint32_t i=0;i<y->count;++i) {
        uint32_t remain=i, source=0;
        for(int j=(int)y->rank-1;j>=0;--j) {
            uint32_t coord=remain%(uint32_t)y->dims[j]; remain/=(uint32_t)y->dims[j];
            source+=coord*strides[p->perm[j]];
        }
        out[i]=a[source];
    }
    return 0;
}
static int pad(const Op *p) {
    const Tensor *x=&nn_tensors[p->in[0]], *y=&nn_tensors[p->out];
    const int8_t *a=nn_tensor_input(p->in[0]); int8_t *out=nn_tensor_output(p->out);
    uint32_t strides[4]={1,1,1,1};
    for(int j=(int)x->rank-2;j>=0;--j) strides[j]=strides[j+1]*(uint32_t)x->dims[j+1];
    for(uint32_t i=0;i<y->count;++i) {
        uint32_t remain=i, source=0; int inside=1;
        for(int j=(int)y->rank-1;j>=0;--j) {
            int coord=(int)(remain%(uint32_t)y->dims[j])-p->pads[2*j];
            remain/=(uint32_t)y->dims[j];
            if(coord<0||coord>=x->dims[j]) inside=0;
            else source+=(uint32_t)coord*strides[j];
        }
        /* PAD's real zero is the quantized zero point, NOT necessarily integer 0. */
        out[i]=inside?a[source]:(int8_t)p->fill;
    }
    return 0;
}
__attribute__((noinline)) int nn_execute(const Op *p) {
    switch(p->kind) {
        case NN_CONV: return conv(p);
        case NN_FC: return fc(p);
        case NN_ADD: return add(p);
        case NN_AVGPOOL: case NN_MAXPOOL: return pool(p);
        case NN_TRANSPOSE: return transpose(p);
        case NN_PAD: case NN_PADV2: return pad(p);
        case NN_RESHAPE: case NN_RELU: case NN_RELU1: case NN_RELU6: {
            const int8_t *a=nn_tensor_input(p->in[0]); int8_t *out=nn_tensor_output(p->out);
            for(uint32_t i=0;i<nn_tensors[p->out].count;++i) out[i]=clamp(a[i],p);
            return 0;
        }
        default: return 99;
    }
}
const char *nn_kind_name(int32_t k) {
    switch(k) {
        case NN_CONV:return "CONV_2D"; case NN_FC:return "FULLY_CONNECTED";
        case NN_ADD:return "ADD"; case NN_AVGPOOL:return "AVERAGE_POOL_2D";
        case NN_MAXPOOL:return "MAX_POOL_2D"; case NN_TRANSPOSE:return "TRANSPOSE";
        case NN_PAD:return "PAD"; case NN_PADV2:return "PADV2";
        case NN_RESHAPE:return "RESHAPE"; case NN_RELU:return "RELU";
        case NN_RELU1:return "RELU_N1_TO_1"; case NN_RELU6:return "RELU6";
        default:return "UNSUPPORTED";
    }
}
