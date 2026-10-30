#!/usr/bin/env python3
"""Test integer FC scaling against a separate native C binary64/round oracle.
The oracle uses FP on the HOST only. nn.c contains the target integer code.
"""
from __future__ import annotations
import argparse, ctypes, json, math, os, random, subprocess, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STUB=r'''
#include "nn.h"
#include <math.h>
#include <limits.h>
const Tensor nn_tensors[1]={0};
const Op nn_ops[1]={0};
const unsigned char nn_model_blob[16]={0}, nn_reference_blob[16]={0};
unsigned char nn_arena[16]={0};
int32_t native_fc_oracle(int32_t x,double scale) {
    const double v=round((double)x*scale);
    if(v>=(double)INT32_MAX) return INT32_MAX;
    if(v<=(double)INT32_MIN) return INT32_MIN;
    return (int32_t)v;
}
'''

def build_library(work: Path):
    stub=work/'fc_oracle.c';stub.write_text(STUB)
    so=work/'fc_test.so'
    subprocess.run([os.environ.get('HOST_CC','cc'),'-O2','-std=c11',
                    '-Wall','-Wextra','-Werror','-shared','-fPIC',
                    '-I'+str(ROOT/'src'),str(ROOT/'src/nn.c'),str(stub),
                    '-lm','-o',str(so)],check=True)
    lib=ctypes.CDLL(str(so))
    lib.nn_requantize_fc_reference.argtypes=[ctypes.c_int32,ctypes.c_uint64,ctypes.c_int32]
    lib.nn_requantize_fc_reference.restype=ctypes.c_int32
    lib.native_fc_oracle.argtypes=[ctypes.c_int32,ctypes.c_double]
    lib.native_fc_oracle.restype=ctypes.c_int32
    lib.nn_requantize.argtypes=[ctypes.c_int32]*3
    lib.nn_requantize.restype=ctypes.c_int32
    return lib

def encode(scale: float):
    f,e=math.frexp(scale)
    return int(math.ldexp(f,53)),e-53

def run_tests(count=100000):
    rng=random.Random(7142026);tested=0
    with tempfile.TemporaryDirectory(prefix='fc-rounding-') as td:
        lib=build_library(Path(td))
        def check(x,scale):
            nonlocal tested
            m,e=encode(scale)
            actual=lib.nn_requantize_fc_reference(x,m,e)
            expected=lib.native_fc_oracle(x,scale)
            assert actual==expected, (x,scale,m,e,actual,expected)
            tested+=1
        for _ in range(count):
            x=rng.randint(-(1<<31),(1<<31)-1)
            m=rng.randrange(1<<52,1<<53)
            scale=math.ldexp(float(m),rng.randint(-100,-53))
            check(x,scale)
        # Explicit half-way and adjacent-binary64 cases, including INT_MIN.
        xs=[0,1,2,3,7,127,1023,(1<<24)-1,1<<24,(1<<24)+1,
            (1<<30)-1,1<<30,(1<<31)-1,-(1<<31)]
        xs += [rng.randint(1,(1<<31)-1) for _ in range(150)]
        for x in xs:
            for sign in (-1,1):
                a=sign*x
                if a < -(1<<31) or a > (1<<31)-1: continue
                for scale in [0.25,0.5,0.75,1.0,math.ldexp(1.0,-1022)]:
                    check(a,scale)
                if a:
                    for half in (0.5,1.5,7.5,63.5,127.5):
                        center=half/abs(a)
                        for scale in (math.nextafter(center,0.0),center,
                                      math.nextafter(center,math.inf)):
                            check(a,scale)
        # An explicit case proves Q31 double-rounding is not interchangeable.
        assert lib.nn_requantize(1,1<<30,-1)==1
        m,e=encode(.25)
        assert lib.nn_requantize_fc_reference(1,m,e)==0
    return {'test':'integer_fc_vs_native_c_binary64_round',
            'cases':tested,'mismatches':0,'q31_counterexample':'pass',
            'full_model_tested':False,'spike_tested':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=100000)
    p.add_argument('--json',type=Path);args=p.parse_args()
    result=run_tests(args.count)
    text=json.dumps(result,indent=2)+'\n';print(text,end='')
    if args.json:
        args.json.parent.mkdir(parents=True,exist_ok=True)
        args.json.write_text(text)
