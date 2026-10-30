#include "nn.h"
#include "model_data.h"
#include "platform.h"
#ifdef HOST_TEST
#include <stdio.h>
#endif
/* Live debugger-visible results. No reference tensor is ever a kernel input. */
volatile uint32_t current_operator=UINT32_MAX;
volatile uint64_t operator_instructions[MODEL_OP_COUNT];
volatile uint64_t full_output_checksum;
static uint64_t checksum(const int8_t *a,uint32_t count) {
    uint64_t h=UINT64_C(14695981039346656037);
    for(uint32_t j=0;j<count;++j) {h^=(uint8_t)a[j];h*=UINT64_C(1099511628211);}
    return h;
}
int main(void) {
    platform_puts("FULL MODEL start ops=");platform_u64(MODEL_OP_COUNT);
    platform_puts(" model_sha256=" MODEL_SHA256 "\n");
    uint64_t sum_instructions=0;
    for(uint32_t i=0;i<MODEL_OP_COUNT;++i) {
        current_operator=i;
        const Op *p=&nn_ops[i];
        const Tensor *y=&nn_tensors[p->out];
        platform_puts("BEGIN op=");platform_u64(i);
        platform_puts(" kind=");platform_puts(nn_kind_name(p->kind));platform_puts("\n");
        uint64_t before=0,after=0;
#if PROFILE && !defined(HOST_TEST)
        before=platform_instret();
#endif
        int status=nn_execute(p);
#if PROFILE && !defined(HOST_TEST)
        after=platform_instret();
#endif
        uint64_t instructions=after-before;
        operator_instructions[i]=instructions;sum_instructions+=instructions;
        if(status) {
            platform_puts("FAIL kernel op=");platform_u64(i);
            platform_puts(" status=");platform_i64(status);platform_puts("\n");return 2;
        }
        /* Verification is OUTSIDE the measured region. Intermediate results
         * remain in nn_arena and are consumed by subsequent operators. */
        const int8_t *actual=nn_tensor_input(p->out);
        const int8_t *expected=(const int8_t *)(nn_reference_blob+p->ref_offset);
        uint32_t bad=0,first=0,max_diff=0;
        for(uint32_t j=0;j<y->count;++j) {
            int diff=(int)actual[j]-(int)expected[j];
            unsigned magnitude=(unsigned)(diff<0?-diff:diff);
            if(magnitude>max_diff)max_diff=magnitude;
            if(diff) { if(!bad)first=j; ++bad; }
        }
        platform_puts(bad?"FAIL op=":"CHECK op=");platform_u64(i);
        platform_puts(" elements=");platform_u64(y->count);
        platform_puts(" mismatches=");platform_u64(bad);
        platform_puts(" max_abs_diff=");platform_u64(max_diff);
        if(bad) {
            platform_puts(" first_bad=");platform_u64(first);
            platform_puts(" actual=");platform_i64(actual[first]);
            platform_puts(" expected=");platform_i64(expected[first]);
        }
        platform_puts("\n");
        if(bad)return 3;
#if PROFILE && !defined(HOST_TEST)
        platform_puts("PROFILE op=");platform_u64(i);
        platform_puts(" kind=");platform_puts(nn_kind_name(p->kind));
        platform_puts(" instret=");platform_u64(instructions);
        platform_puts(" dense_macs=");platform_u64(p->dense_macs);
        platform_puts(" valid_macs=");platform_u64(p->valid_macs);platform_puts("\n");
#endif
    }
    const int8_t *final=nn_tensor_input(MODEL_OUTPUT_ID);
    full_output_checksum=checksum(final,MODEL_FINAL_ELEMENTS);
#ifdef HOST_TEST
    FILE *f=fopen("results/host_output.bin","wb");
    if(!f) {platform_puts("FAIL cannot write results/host_output.bin\n");return 4;}
    size_t written=fwrite(final,1,MODEL_FINAL_ELEMENTS,f);
    int close_status=fclose(f);
    if(written!=MODEL_FINAL_ELEMENTS||close_status!=0)return 4;
#endif
    platform_puts("PASS full_model operators=");platform_u64(MODEL_OP_COUNT);
    platform_puts(" outputs=");platform_u64(MODEL_FINAL_ELEMENTS);
    platform_puts(" mismatches=0 checksum=");platform_u64(full_output_checksum);
    platform_puts("\n");
    if(sum_instructions) {
        platform_puts("PROFILE_SUM dispatch_regions_instret=");platform_u64(sum_instructions);platform_puts("\n");
    }
    return 0;
}
