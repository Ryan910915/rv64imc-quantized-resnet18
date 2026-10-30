#include <stdint.h>
#include "xrequant.h"

static int8_t reference_requant(
    int32_t x,
    int32_t multiplier,
    int32_t shift,
    int32_t zero_point,
    int32_t act_min,
    int32_t act_max)
{
    int left=shift>0?shift:0;
    int right=shift<0?-shift:0;

    int64_t expanded=(int64_t)x*(INT64_C(1)<<left);
    int64_t product=expanded*(int64_t)multiplier;
    int64_t nudge=product>=0?(INT64_C(1)<<30):1-(INT64_C(1)<<30);
    int32_t high=(int32_t)((product+nudge)/(INT64_C(1)<<31));

    int32_t scaled=high;

    if(right!=0) {
        int64_t magnitude=high<0?-(int64_t)high:(int64_t)high;
        magnitude=(magnitude+(INT64_C(1)<<(right-1)))>>right;
        scaled=(int32_t)(high<0?-magnitude:magnitude);
    }

    int64_t out=(int64_t)scaled+zero_point;

    if(out<act_min) out=act_min;
    if(out>act_max) out=act_max;

    return (int8_t)out;
}

static int check(
    int32_t x,
    int32_t multiplier,
    int32_t shift,
    int32_t zero_point,
    int32_t act_min,
    int32_t act_max)
{
    uint64_t params=xrequant_pack_params(
        multiplier,shift,zero_point,act_min,act_max);

    int8_t expected=reference_requant(
        x,multiplier,shift,zero_point,act_min,act_max);

    int8_t actual=xrequant_apply(x,params);

    return actual==expected;
}

static uint32_t rng_state=0x6d5a56e9u;

static uint32_t rng_next(void)
{
    uint32_t x=rng_state;
    x^=x<<13;
    x^=x>>17;
    x^=x<<5;
    rng_state=x;
    return x;
}

int main(void)
{
    if(!check(1000,1073741824,0,0,-128,127)) return 1;
    if(!check(-1000,1073741824,0,0,-128,127)) return 2;
    if(!check(1234567,1500000000,-5,-17,-128,127)) return 3;
    if(!check(1234,1200000000,3,11,-128,127)) return 4;
    if(!check(214748,2147483647,0,120,-128,127)) return 5;
    if(!check(-214748,2147483647,0,-120,-128,127)) return 6;
    if(!check(1000000,1800000000,-3,0,-6,6)) return 7;

    for(int t=0;t<10000;++t) {
        int32_t x=(int32_t)(rng_next()&UINT32_C(0x001fffff));

        if(rng_next()&1u)
            x=-x;

        int32_t multiplier=(int32_t)(
            UINT32_C(0x40000000)+
            (rng_next()&UINT32_C(0x3fffffff)));

        int32_t shift=(int32_t)(rng_next()%17u)-8;
        int32_t zero_point=(int32_t)(int8_t)(rng_next()&0xffu);

        int32_t act_min=-128+(int32_t)(rng_next()%32u);
        int32_t act_max=127-(int32_t)(rng_next()%32u);

        if(act_min>act_max)
            return 90;

        if(!check(
            x,multiplier,shift,zero_point,act_min,act_max))
            return 10;
    }

    return 0;
}
