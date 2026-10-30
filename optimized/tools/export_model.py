#!/usr/bin/env python3
"""Compile a supported static INT8 TFLite graph into an offline C workload.

Only the HOST uses NumPy/LiteRT. The RV64IMC program has no model parser,
filesystem, Python, LiteRT runtime, dynamic allocator, or floating-point math.
Unknown/unsupported semantics are errors, never silently approximated.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import numpy as np
from flatbuffer_min import FlatBuffer

KNOWN_SHA = '8fea1acb62829e120b4953b6fbf0f842e0032909254f51c2d989703b66534683'
NAMES = {0:'ADD',1:'AVERAGE_POOL_2D',3:'CONV_2D',9:'FULLY_CONNECTED',
         17:'MAX_POOL_2D',19:'RELU',20:'RELU_N1_TO_1',21:'RELU6',22:'RESHAPE',
         34:'PAD',39:'TRANSPOSE',60:'PADV2'}
MAX_VERSION = {0:5,1:3,3:8,9:14,17:3,19:3,20:3,21:3,22:2,34:4,39:6,60:4}

def need(condition, message):
    if not condition:
        raise ValueError(message)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def round_away(x):
    return math.floor(x+0.5) if x>=0 else math.ceil(x-0.5)

def quantize_multiplier(scale):
    need(math.isfinite(scale) and scale>=0,'Invalid effective multiplier')
    if scale==0:
        return 0,0
    mantissa, shift = math.frexp(scale)
    q = round_away(mantissa*(1<<31))
    if q==(1<<31):
        q//=2; shift+=1
    if shift < -31:
        return 0,0
    need(shift<=30, 'Multiplier shift exceeds this INT32 baseline')
    return q,shift

def round_binary64_away(value):
    """C round() semantics; avoid the extra rounding introduced by x + 0.5."""
    magnitude=abs(float(value))
    base=math.floor(magnitude)
    rounded=base+int(magnitude-base>=0.5)
    return -rounded if value<0 else rounded

def fc_reference_scales(node,meta):
    """Mirror the BUILTIN_REF FC binary64 operation order on the HOST."""
    x=meta[node['inputs'][0]];w=meta[node['inputs'][1]];y=meta[node['output']]
    sx,_=per_tensor(x);sy,_=per_tensor(y);co=w['shape'][0]
    ws=w['scales'] if len(w['scales'])>1 else w['scales']*co
    return [float(sx)*float(sw)/float(sy) for sw in ws]

def fc_reference_blob(node,meta):
    """Export binary64 values losslessly as integer significands/exponents."""
    significands=[];exponents=[]
    for scale in fc_reference_scales(node,meta):
        need(math.isfinite(scale) and scale>=sys.float_info.min,
             'FC reference needs a finite positive normal binary64 scale')
        frac,exp=math.frexp(scale)
        significands.append(int(math.ldexp(frac,53)))
        exponents.append(exp-53)
    return (np.asarray(significands,dtype='<u8').tobytes()+
            np.asarray(exponents,dtype='<i4').tobytes())

def per_tensor(t):
    need(t['dtype']=='int8' and len(t['scales'])==1 and len(t['zero_points'])==1,
         f"Tensor {t['index']} requires per-tensor INT8 quantization")
    s,z = t['scales'][0],t['zero_points'][0]
    need(math.isfinite(s) and s>0 and -128<=z<=127,'Invalid INT8 scale/zero point')
    return s,z

def equal_quant(a,b):
    need(per_tensor(a)==per_tensor(b), 'Copy/pool operation requires identical quantization')

def activation(t, code):
    s,z=per_tensor(t)
    # LiteRT activation threshold division is float32, not Python binary64.
    def qt(f):
        return z+round_away(float(np.float32(f)/np.float32(s)))
    if code==0:return -128,127
    if code==1:return max(-128,qt(0)),127
    if code==2:return max(-128,qt(-1)),min(127,qt(1))
    if code==3:return max(-128,qt(0)),min(127,qt(6))
    raise ValueError(f'Unsupported fused activation enum {code}')

def pad_axis(size,kernel,stride,dilation,out,mode):
    need(min(size,kernel,stride,dilation,out)>0,'Nonpositive Conv/pool dimension')
    eff=(kernel-1)*dilation+1
    if mode==0:
        expected=(size+stride-1)//stride
        total=max((out-1)*stride+eff-size,0)
        pads=(total//2,total-total//2)
    elif mode==1:
        expected=max((size-eff)//stride+1,0); pads=(0,0)
    else:raise ValueError('Unknown padding enum')
    need(out==expected,f'Output size {out} does not match inferred {expected}')
    return pads

def graph_schema(data):
    fb=FlatBuffer(data)
    need(fb.value(fb.root,0,'I')==3,'Only schema version 3 is supported')
    sgs=fb.tables(fb.root,2)
    need(len(sgs)==1,'Exactly one subgraph is required (no control flow)')
    sg=sgs[0]; ts=fb.tables(sg,0); codes=fb.tables(fb.root,1); buffers=fb.tables(fb.root,4)
    constants=set()
    for i,t in enumerate(ts):
        need(not fb.value(t,5,'B') and fb.reference(t,6) is None and not fb.value(t,10,'I'),
             f'Tensor {i}: variable/sparse/external tensor unsupported')
        need(all(d>0 for d in fb.ints(t,0)),f'Tensor {i}: invalid static shape')
        # Export specializes the actual allocated shape, including batch=1.
        b=fb.value(t,2,'I'); need(b<len(buffers),'Buffer index out of range')
        need(fb.value(buffers[b],1,'Q')<=1,'Out-of-FlatBuffer weights unsupported')
        _,length=fb.vector(buffers[b],0,1)
        if length:constants.add(i)
        qp=fb.reference(t,4)
        if qp is not None:
            need(fb.value(qp,4,'B')==0, 'Custom/blockwise quantization unsupported')
    nodes=[]
    for idx,op in enumerate(fb.tables(sg,3)):
        oi=fb.value(op,0,'I'); need(oi<len(codes),'Opcode index out of range')
        c=codes[oi]; k=max(fb.value(c,0,'b'),fb.value(c,3,'i'))
        need(k in NAMES,f'Operator {idx}: unsupported builtin code {k}')
        ver=fb.value(c,2,'i',1)
        need(1<=ver<=MAX_VERSION[k],f'Operator {idx}: review unsupported {NAMES[k]} version {ver}')
        ins=fb.ints(op,1); outs=fb.ints(op,2)
        need(all(-1<=i<len(ts) for i in ins) and all(0<=i<len(ts) for i in outs),'Bad tensor index')
        need(1<=len(ins)<=3 and ins[0]>=0 and len(outs)==1,'Unsupported operator arity')
        opt=fb.reference(op,4); tag=fb.value(op,3,'B')
        def get(slot,fmt='i',default=0):
            return default if opt is None else fb.value(opt,slot,fmt,default)
        options={}
        if k in (3,1,17):
            need(opt is not None and tag==(1 if k==3 else 5),'Unexpected Conv/Pool options union')
            options={'padding':get(0,'b'),'stride':[get(2),get(1)]}
            if k==3:
                options.update(dilation=[get(5,'i',1),get(4,'i',1)],activation=get(3,'b'))
                need(get(6,'b') in (0,2),'Only INT32 bias/accumulator supported')
            else:
                options.update(kernel=[get(4),get(3)],activation=get(5,'b'))
        elif k==9:
            need(opt is not None and tag==8,'Unexpected FullyConnected options union')
            need(get(1,'b')==0 and not get(3,'B') and get(4,'b') in (0,2),
                 'Unsupported FC shuffled/dynamic-quant/accumulator setting')
            _,nquant=fb.vector(opt,5,1)
            need(nquant==0,'Unsupported FC quant_spec')
            options={'activation':get(0,'b'),'keep_num_dims':bool(get(2,'B'))}
        elif k==0:
            need(tag in (0,11),'Unexpected Add options union')
            options={'activation':get(0,'b')}
        nodes.append({'index':idx,'kind':k,'name':NAMES[k],'version':ver,'inputs':ins,'output':outs[0],'options':options})
    inputs=fb.ints(sg,1); outputs=fb.ints(sg,2)
    need(len(inputs)==1 and len(outputs)==1,'Exactly one model input/output required')
    return nodes,constants,inputs[0],outputs[0],len(ts)

def summarize(d):
    q=d['quantization_parameters']
    return {'index':int(d['index']),'name':str(d['name']),
            'shape':[int(v) for v in d['shape']], 'dtype':np.dtype(d['dtype']).name,
            'scales':[float(v) for v in q['scales']],
            'zero_points':[int(v) for v in q['zero_points']],
            'quantized_dimension':int(q['quantized_dimension'])}

def prepare_op(node,meta,arrays):
    k=node['kind']; ins=node['inputs']; y=meta[node['output']]; x=meta[ins[0]]; opt=node['options']
    per_tensor(x); per_tensor(y)
    need(0<len(x['shape'])<=4 and 0<len(y['shape'])<=4,'Activation rank must be 1..4')
    arity={0:(2,),1:(1,),3:(2,3),9:(2,3),17:(1,),19:(1,),20:(1,),21:(1,),22:(1,2),34:(2,),39:(2,),60:(3,)}
    need(len(ins) in arity[k],f'{node["name"]}: unsupported number of inputs')
    p={'kind':k,'in':(ins+[-1]*3)[:3],'out':node['output'],'stride':[1,1],
       'dilation':[1,1],'kernel':[1,1],'pad':[0]*4,'act_min':-128,'act_max':127,
       'q_offset':0,'q_count':0,'add':[0]*7,'perm':[0,1,2,3],'pads':[0]*8,'fill':0,
       'dense_macs':0,'valid_macs':0,'ref_offset':0,
       'fc_reference':int(k==9),'fc_scale_offset':0}
    qvalues=None
    if k in (3,9):
        need(ins[1] in arrays,'MAC weights must be static constants')
        w=meta[ins[1]]; need(w['dtype']=='int8','MAC weights must be INT8')
        co=w['shape'][0]; sx,zx=per_tensor(x); sy,_=per_tensor(y)
        ws=w['scales']; wz=w['zero_points']
        need(len(ws) in (1,co) and all(math.isfinite(v) and v>0 for v in ws),'Invalid weight scales')
        need(len(wz) in (1,co) and all(z==0 for z in wz),'Weights must be symmetric')
        need(len(ws)==1 or w['quantized_dimension']==0,'Weight per-channel axis must be output channel (0)')
        scales=[ws[0]]*co if len(ws)==1 else ws
        bias_id=p['in'][2]
        if bias_id>=0:
            b=meta[bias_id]
            need(bias_id in arrays and b['dtype']=='int32' and b['shape']==[co],'Bias must be static INT32[O]')
            need(not b['zero_points'] or all(z==0 for z in b['zero_points']),'Nonzero bias zero point unsupported')
            if b['scales']:
                bs=np.asarray(b['scales']); target=np.asarray(scales)*sx
                need(len(bs) in (1,co) and np.allclose(bs,target,rtol=1e-5,atol=1e-12),'Bias scale mismatch')
            bv=arrays[bias_id].astype(np.int64).reshape(co)
        else:bv=np.zeros(co,dtype=np.int64)
        multipliers,shifts=zip(*(quantize_multiplier(sx*s/sy) for s in scales))
        qvalues=np.asarray(list(multipliers)+list(shifts),dtype='<i4')
        p['q_count']=co
        # Prove absence of signed overflow even for arbitrary INT8 inputs.
        sumabs=np.abs(arrays[ins[1]].astype(np.int64).reshape(co,-1)).sum(axis=1)
        bound=np.abs(bv)+max(abs(-128-zx),abs(127-zx))*sumabs
        shifted_bound=bound*np.asarray([1<<max(0,s) for s in shifts],dtype=np.int64)
        need(np.all(bound<=2147483647) and np.all(shifted_bound<=2147483647),
             'INT32 accumulator/pre-shift bound fails; this model needs a reviewed wider path')
        if k==9:
            # C++ reference casts the scaled result to INT32 before clamping.
            # Prove the cast is in range for any supported INT8 input.
            need(all(float(b)*scale+abs(y['zero_points'][0])+1 < 2147483647
                     for b,scale in zip(bound,fc_reference_scales(node,meta))),
                 'FC binary64 reference scaled accumulator exceeds INT32')
        p['act_min'],p['act_max']=activation(y,opt.get('activation',0))
        if k==3:
            need(len(x['shape'])==len(w['shape'])==len(y['shape'])==4,'Conv needs rank-4 tensors')
            n,ih,iw,ci=x['shape']; oc,kh,kw,wci=w['shape']; ny,oh,ow,cy=y['shape']
            need(n==ny and ci==wci and oc==cy,'Conv shape/group mismatch')
            p['stride']=opt['stride']; p['dilation']=opt['dilation']; p['kernel']=[kh,kw]
            p['pad']=list(pad_axis(ih,kh,p['stride'][0],p['dilation'][0],oh,opt['padding']))+list(pad_axis(iw,kw,p['stride'][1],p['dilation'][1],ow,opt['padding']))
            p['dense_macs']=n*oh*ow*oc*kh*kw*ci
            vh=sum(0<=oy*p['stride'][0]-p['pad'][0]+ky*p['dilation'][0]<ih for oy in range(oh) for ky in range(kh))
            vw=sum(0<=ox*p['stride'][1]-p['pad'][2]+kx*p['dilation'][1]<iw for ox in range(ow) for kx in range(kw))
            p['valid_macs']=n*oc*ci*vh*vw
        else:
            need(len(w['shape'])==2,'FC requires rank-2 weights [O,I]')
            ci=w['shape'][1]; batches=math.prod(x['shape'])//ci
            need(math.prod(x['shape'])%ci==0 and math.prod(y['shape'])==batches*co,'FC flatten mismatch')
            if opt.get('keep_num_dims'):
                need(x['shape'][-1]==ci and y['shape']==x['shape'][:-1]+[co],'FC keep_num_dims mismatch')
            else:need(y['shape']==[batches,co],'FC output shape mismatch')
            p['dense_macs']=p['valid_macs']=batches*co*ci
    elif k==0:
        z=meta[ins[1]]; s1,_=per_tensor(x); s2,_=per_tensor(z); sy,_=per_tensor(y)
        need(x['shape']==z['shape']==y['shape'],'Broadcast ADD is not supported in this exporter')
        mx=2*max(s1,s2)
        qm=[quantize_multiplier(s1/mx),quantize_multiplier(s2/mx),quantize_multiplier(mx/((1<<20)*sy))]
        need(all(s<=0 for _,s in qm),'ADD rescale must be <=1 for this implementation')
        p['add']=[v for pair in qm for v in pair]+[20]
        p['act_min'],p['act_max']=activation(y,opt.get('activation',0))
    elif k in (1,17):
        equal_quant(x,y)
        need(len(x['shape'])==len(y['shape'])==4 and x['shape'][0]==y['shape'][0] and x['shape'][3]==y['shape'][3],'Pool shape mismatch')
        p['stride']=opt['stride']; p['kernel']=opt['kernel']
        p['pad']=list(pad_axis(x['shape'][1],p['kernel'][0],p['stride'][0],1,y['shape'][1],opt['padding']))+list(pad_axis(x['shape'][2],p['kernel'][1],p['stride'][1],1,y['shape'][2],opt['padding']))
        need(math.prod(p['kernel'])*128<2147483647,'Pool accumulator overflow bound')
        p['act_min'],p['act_max']=activation(y,opt.get('activation',0))
    elif k in (34,60):
        equal_quant(x,y); need(ins[1] in arrays,'PAD widths must be constant')
        pads=arrays[ins[1]]
        need(pads.dtype==np.int32 and pads.shape==(len(x['shape']),2) and np.all(pads>=0),'Invalid PAD widths')
        need(y['shape']==[v+int(a)+int(b) for v,(a,b) in zip(x['shape'],pads)],'PAD output shape mismatch')
        p['pads']=pads.reshape(-1).astype(int).tolist()+[0]*(8-pads.size)
        p['fill']=per_tensor(y)[1]
        if k==60:
            need(ins[2] in arrays and arrays[ins[2]].dtype==np.int8 and arrays[ins[2]].size==1,'PADV2 fill must be constant INT8 scalar')
            p['fill']=int(arrays[ins[2]].reshape(-1)[0])
    elif k==39:
        equal_quant(x,y); need(ins[1] in arrays,'TRANSPOSE permutation must be constant')
        a=arrays[ins[1]]
        need(a.dtype==np.int32 and a.ndim==1,'TRANSPOSE permutation must be INT32 vector')
        perm=a.astype(int).tolist(); need(sorted(perm)==list(range(len(x['shape']))),'Invalid permutation')
        need(y['shape']==[x['shape'][i] for i in perm],'TRANSPOSE shape mismatch')
        p['perm']=perm+[0]*(4-len(perm))
    elif k in (19,20,21,22):
        equal_quant(x,y)
        need(math.prod(x['shape'])==math.prod(y['shape']),'Element count mismatch')
        if k!=22:
            need(x['shape']==y['shape'],'Activation shape mismatch')
            p['act_min'],p['act_max']=activation(y,{19:1,20:2,21:3}[k])
        elif len(ins)==2:need(ins[1] in arrays,'Dynamic reshape unsupported')
    return p,qvalues

def append_blob(blob,data):
    blob.extend(b'\0'*((-len(blob))%16)); at=len(blob); blob.extend(data); return at

def c_array(values):
    return '{'+','.join(str(int(v)) for v in values)+'}'

def emit_artifacts(out,meta,nodes,arrays,references,input_id,output_id,info,tensor_count=None,incbin_dir='generated'):
    """Separately testable compiler; arrays include constants and ONE graph input.
    references are never inserted into arrays or the writable activation arena.
    """
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    count=tensor_count or max(meta)+1
    used={i for n in nodes for i in n['inputs']+[n['output']] if i>=0}|{input_id,output_id}
    need(all(i in meta for i in used),'Missing tensor metadata')
    need(input_id in arrays,'No model input')
    need(output_id not in arrays,'Model output must be computed, not constant')
    available=set(arrays); seen=set()
    for n in nodes:
        need(all(i<0 or i in available for i in n['inputs']),f"op {n['index']}: input not produced")
        need(n['output'] not in available and n['output'] not in seen,'Aliased/in-place/multiple tensor producers unsupported')
        available.add(n['output']);seen.add(n['output'])
    need(output_id in seen,'Model output is not produced by the schedule')
    model=bytearray(); gold=bytearray(); arena=0; entries=[]; tensor_report=[]
    for i in range(count):
        if i not in used:
            entries.append('{{1,1,1,1},0,0,0,0,0,0}'); continue
        t=meta[i]; shape=t['shape']; dtype=np.dtype(t['dtype']); size=math.prod(shape)*dtype.itemsize
        need(0<=len(shape)<=4 and all(v>0 for v in shape),'Invalid tensor dimensions')
        need(dtype in (np.dtype('int8'),np.dtype('int32')),'Only INT8/INT32 tensors supported')
        need(size<2**31,'Tensor too large')
        if i in arrays:
            a=arrays[i]
            need(list(a.shape)==shape and a.dtype==dtype,f'Tensor {i}: array shape/dtype mismatch')
            raw=np.ascontiguousarray(a.astype(dtype.newbyteorder('<'),copy=False)).tobytes()
            at=append_blob(model,raw); constant=1
        else:
            need(dtype==np.int8,'Writable activations must be INT8')
            arena=(arena+15)//16*16; at=arena; arena+=size; constant=0
        z=t['zero_points'][0] if t['zero_points'] else 0
        entries.append('{'+c_array(shape+[1]*(4-len(shape)))+f',{len(shape)},{math.prod(shape)},{at},{size},{z},{constant}'+'}')
        tensor_report.append(dict(t,storage='model_blob' if constant else 'arena',offset=at,bytes=size))
    ops=[]; reports=[]
    for n in nodes:
        try:p,q=prepare_op(n,meta,arrays)
        except ValueError as e:raise ValueError(f"op {n['index']} {n['name']}: {e}") from e
        if q is not None:p['q_offset']=append_blob(model,q.tobytes())
        if n['kind']==9:p['fc_scale_offset']=append_blob(model,fc_reference_blob(n,meta))
        a=references[n['output']]; y=meta[n['output']]
        need(a.dtype==np.int8 and list(a.shape)==y['shape'],'Reference shape/dtype mismatch')
        p['ref_offset']=append_blob(gold,a.tobytes(order='C'))
        ops.append(p)
        reports.append(dict(n,compiled=p,output_elements=int(a.size),reference_sha256=digest(a.tobytes())))
    required=len(model)+len(gold)+arena+1024*1024
    need(required<64*1024*1024,f'64 MiB linker budget exceeded: conservative requirement {required} bytes')
    (out/'model.bin').write_bytes(model)
    (out/'reference.bin').write_bytes(gold)
    # This assembler inclusion happens at BUILD time. No files are opened by RV64IMC.
    assembly=['.section .rodata.nn_blobs,"a",@progbits']
    for symbol,name in [('nn_model_blob','model.bin'),('nn_reference_blob','reference.bin')]:
        path=f'{incbin_dir}/{name}'.replace('\\','\\\\').replace('"','\\"')
        assembly+=['.balign 16',f'.global {symbol}',f'.type {symbol}, @object',f'{symbol}:',f'.incbin "{path}"',f'.size {symbol}, .-{symbol}']
    assembly+=['.section .note.GNU-stack,"",@progbits']
    (out/'model_blob.S').write_text('\n'.join(assembly)+'\n')
    hdr=f'''#ifndef RESNET_GENERATED_MODEL_DATA_H
#define RESNET_GENERATED_MODEL_DATA_H
#include "nn.h"
#define MODEL_OP_COUNT {len(ops)}
#define MODEL_TENSOR_COUNT {count}
#define MODEL_INPUT_ID {input_id}
#define MODEL_OUTPUT_ID {output_id}
#define MODEL_ARENA_BYTES {max(16,(arena+15)//16*16)}
#define MODEL_FINAL_ELEMENTS {math.prod(meta[output_id]['shape'])}
#define MODEL_SHA256 "{info.get('model_sha256','synthetic')}"
#endif
'''
    (out/'model_data.h').write_text(hdr)
    op_lines=[]
    for p in ops:
        terms=[]
        for key,val in p.items():
            literal=c_array(val) if isinstance(val,list) else str(int(val))
            if key in ('dense_macs','valid_macs'):literal+='ULL'
            terms.append('.'+key+'='+literal)
        op_lines.append('    {'+','.join(terms)+'}')
    src='#include "model_data.h"\nunsigned char nn_arena[MODEL_ARENA_BYTES] __attribute__((aligned(16)));\n'
    src+='const Tensor nn_tensors[MODEL_TENSOR_COUNT]={\n'+',\n'.join('    '+v for v in entries)+'\n};\n'
    src+='const Op nn_ops[MODEL_OP_COUNT]={\n'+',\n'.join(op_lines)+'\n};\n'
    (out/'model_data.c').write_text(src)
    np.save(out/'input.npy',arrays[input_id],allow_pickle=False)
    np.save(out/'expected_output.npy',references[output_id],allow_pickle=False)
    report=dict(info,format='rv64imc_full_graph_v2_fc_ref',
        rounding='Conv/ADD: Q31 two-stage; FC: integer-emulated binary64 then ties-away',
        numerical_fix='fc_builtin_ref_binary64_v1',
        input_id=input_id,output_id=output_id,operator_counts=dict(Counter(n['name'] for n in nodes)),
        tensors=tensor_report,operators=reports,
        storage={'model_blob_bytes':len(model),'reference_blob_bytes':len(gold),'activation_arena_bytes':arena,
                 'linker_ram_bytes':64*1024*1024,'stack_bytes':256*1024,
                 'note':'No activation-buffer reuse; reference data is verification-only. FC uses integer-only binary64-compatible requantization.'},
        dense_macs=sum(p['dense_macs'] for p in ops),valid_macs=sum(p['valid_macs'] for p in ops),
        blob_sha256=digest(model),reference_blob_sha256=digest(gold))
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'operators.txt').write_text('\n'.join(f"{n['index']:02d} {n['name']:20s} in={n['inputs']} out={n['output']} shape={meta[n['output']]['shape']} dense_MACs={p['dense_macs']} valid_MACs={p['valid_macs']}" for n,p in zip(nodes,ops))+'\n')
    (out/'.stamp').write_text(digest(model)+':'+digest(gold)+'\n')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,default=Path('../models/resnet18_int8_channelwise.tflite'))
    p.add_argument('--input-npy',type=Path)
    p.add_argument('--seed',type=int,default=20261006)
    p.add_argument('--out',type=Path,default=Path('generated'))
    args=p.parse_args()
    need(args.model.is_file(),f'Model not found: {args.model}')
    data=args.model.read_bytes(); sha=digest(data)
    nodes,constants,input_id,output_id,nt=graph_schema(data)
    print(f'Model SHA256: {sha}',flush=True)
    if sha!=KNOWN_SHA:print('NOTICE: model differs from the previously inspected ResNet; validating the actual graph.',flush=True)
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    interp=Interpreter(model_path=str(args.model),num_threads=1,
                       experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
                       experimental_preserve_all_tensors=True)
    interp.allocate_tensors()
    meta={int(t['index']):summarize(t) for t in interp.get_tensor_details()}
    need(input_id==int(interp.get_input_details()[0]['index']) and output_id==int(interp.get_output_details()[0]['index']),'Schema/runtime I/O mismatch')
    t=meta[input_id];per_tensor(t)
    if args.input_npy:
        value=np.load(args.input_npy,allow_pickle=False)
        need(value.dtype==np.int8 and list(value.shape)==t['shape'],'Input must already match the model INT8 dtype/layout/shape')
        source={'kind':'npy','path':str(args.input_npy)}
    else:
        value=np.random.default_rng(args.seed).integers(-128,128,size=t['shape'],dtype=np.int8)
        source={'kind':'synthetic','seed':args.seed,'accuracy_test':False}
    arrays={i:interp.get_tensor(i).copy() for i in constants if i in meta}
    arrays[input_id]=value.copy()
    # Validate the complete graph before the expensive reference run.
    for n in nodes:
        try:prepare_op(n,meta,arrays)
        except ValueError as e:raise ValueError(f"op {n['index']} {n['name']}: {e}") from e
    interp.set_tensor(input_id,value)
    print(f'Running one complete HOST LiteRT reference ({len(nodes)} operators)...',flush=True)
    interp.invoke()
    refs={n['output']:interp.get_tensor(n['output']).copy() for n in nodes}
    info={'model_path':str(args.model),'model_sha256':sha,'input_source':source,
          'input_sha256':digest(value.tobytes()),'numpy_version':np.__version__,
          'litert_version':importlib.metadata.version('ai-edge-litert'),
          'resolver':'BUILTIN_REF','num_threads':1,'preserve_all_tensors':True,
          'host_reference_complete':True,'c_full_model_verified':False,
          'note':'Export is not C/Spike verification; run host-test and spike.'}
    # Do not replace a valid previous export on a parse/validation failure.
    args.out.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='.export-',dir=str(args.out.parent)))
    try:
        report=emit_artifacts(temp,meta,nodes,arrays,refs,input_id,output_id,info,nt,args.out.as_posix())
        args.out.mkdir(parents=True,exist_ok=True)
        (args.out/'.stamp').unlink(missing_ok=True)
        for f in temp.iterdir():
            if f.name!='.stamp':shutil.move(str(f),str(args.out/f.name))
        shutil.move(str(temp/'.stamp'),str(args.out/'.stamp'))
    finally:shutil.rmtree(temp,ignore_errors=True)
    print('EXPORT COMPLETE:',args.out)
    print('Operators:',report['operator_counts'])
    print('Dense MACs:',report['dense_macs'],'Valid MACs:',report['valid_macs'])
    print('Storage:',report['storage'])
    print('Next: make host-test (C computes the ENTIRE graph; no reference reinjection).')
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,ImportError,OSError,KeyError) as e:
        print(f'EXPORT FAILED: {e}',file=sys.stderr)
        print('Stop here; do not substitute a different model or relax the golden check.',file=sys.stderr)
        sys.exit(2)
