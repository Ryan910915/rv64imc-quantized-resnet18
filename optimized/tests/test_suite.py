#!/usr/bin/env python3
"""Synthetic semantic/integration tests. Optional --case checks uploaded op14.
These tests do NOT claim a complete ResNet/LiteRT/Spike execution.
"""
from __future__ import annotations
import argparse,ctypes,json,math,os,shutil,struct,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from export_model import emit_artifacts,prepare_op,quantize_multiplier,graph_schema,fc_reference_scales,round_binary64_away

RNG=np.random.default_rng(4130)

def rq(value,mult,shift):
    """Independent gemmlowp-style arithmetic using floor/remainders."""
    v=np.asarray(value,dtype=np.int64)
    left=np.maximum(np.asarray(shift,dtype=np.int64),0)
    right=np.maximum(-np.asarray(shift,dtype=np.int64),0)
    high=(v*(np.int64(1)<<left)*np.asarray(mult,dtype=np.int64)+(1<<30))//(1<<31)
    mask=(np.int64(1)<<right)-1
    threshold=(mask>>1)+(high<0)
    return (high>>right)+((high&mask)>threshold)

class Fixture:
    def __init__(self):self.meta={};self.arrays={};self.values={};self.nodes=[];self.input_id=None
    def tensor(self,a,scale=0.125,zp=0,constant=False):
        a=np.asarray(a);i=len(self.meta)
        scales=[float(np.float32(v)) for v in np.atleast_1d(scale)]
        zps=[int(v) for v in np.atleast_1d(zp)]
        self.meta[i]={'index':i,'name':f't{i}','shape':list(a.shape),'dtype':a.dtype.name,
                      'scales':scales,'zero_points':zps,'quantized_dimension':0}
        self.values[i]=a.copy()
        if constant:self.arrays[i]=a.copy()
        return i
    def input(self,shape,scale=.125,zp=0):
        i=self.tensor(RNG.integers(-128,128,size=shape,dtype=np.int8),scale,zp,True)
        if self.input_id is None:self.input_id=i
        return i
    def op(self,kind,ins,shape,scale=.125,zp=0,options=None):
        y=self.tensor(np.zeros(shape,dtype=np.int8),scale,zp)
        from export_model import NAMES
        n={'index':len(self.nodes),'name':NAMES[kind],'kind':kind,'version':1,'inputs':ins,'output':y,'options':options or {}}
        p,q=prepare_op(n,self.meta,self.arrays)
        self.values[y]=self.reference(n,p,q)
        self.nodes.append(n)
        return y
    def reference(self,n,p,q):
        x=self.values[n['inputs'][0]];xm=self.meta[n['inputs'][0]];ym=self.meta[n['output']]
        y=np.zeros(ym['shape'],dtype=np.int64);k=n['kind'];xzp=xm['zero_points'][0];yzp=ym['zero_points'][0]
        if k in (3,9):
            w=self.values[n['inputs'][1]].astype(np.int64);co=w.shape[0]
            b=self.values[n['inputs'][2]].astype(np.int64) if len(n['inputs'])==3 and n['inputs'][2]>=0 else np.zeros(co,dtype=np.int64)
            if k==9:
                acc=(x.astype(np.int64).reshape(-1,w.shape[1])-xzp)@w.T+b
                scales=fc_reference_scales(n,self.meta)
                y=np.asarray([[round_binary64_away(float(v)*scale)+yzp
                               for v,scale in zip(row,scales)] for row in acc],
                             dtype=np.int64).reshape(ym['shape'])
            else:
                for batch in range(y.shape[0]):
                    for oy in range(y.shape[1]):
                        for ox in range(y.shape[2]):
                            patch=np.zeros(w.shape[1:],dtype=np.int64)
                            for ky in range(w.shape[1]):
                                iy=oy*p['stride'][0]-p['pad'][0]+ky*p['dilation'][0]
                                for kx in range(w.shape[2]):
                                    ix=ox*p['stride'][1]-p['pad'][2]+kx*p['dilation'][1]
                                    if 0<=iy<x.shape[1] and 0<=ix<x.shape[2]:
                                        patch[ky,kx]=x[batch,iy,ix].astype(np.int64)-xzp
                            acc=np.sum(w*patch,axis=(1,2,3))+b
                            y[batch,oy,ox]=rq(acc,q[:co],q[co:])+yzp
        elif k==0:
            z=self.values[n['inputs'][1]];zzp=self.meta[n['inputs'][1]]['zero_points'][0]
            a=(x.astype(np.int64)-xzp)*(1<<20);b=(z.astype(np.int64)-zzp)*(1<<20)
            y=rq(rq(a,*p['add'][0:2])+rq(b,*p['add'][2:4]),*p['add'][4:6])+yzp
        elif k in (1,17):
            for batch in range(y.shape[0]):
                for oy in range(y.shape[1]):
                    for ox in range(y.shape[2]):
                        points=[]
                        for ky in range(p['kernel'][0]):
                            iy=oy*p['stride'][0]-p['pad'][0]+ky
                            for kx in range(p['kernel'][1]):
                                ix=ox*p['stride'][1]-p['pad'][2]+kx
                                if 0<=iy<x.shape[1] and 0<=ix<x.shape[2]:points.append(x[batch,iy,ix].astype(np.int64))
                        a=np.asarray(points)
                        if k==17:y[batch,oy,ox]=a.max(axis=0)
                        else:
                            total=a.sum(axis=0);count=len(points)
                            y[batch,oy,ox]=np.sign(total)*((np.abs(total)+count//2)//count)
        elif k==39:y=np.transpose(x,p['perm'][:x.ndim])
        elif k in (34,60):y=np.pad(x,np.asarray(p['pads'][:x.ndim*2]).reshape(-1,2),constant_values=p['fill'])
        else:y=x.reshape(ym['shape'])
        return np.clip(y,p['act_min'],p['act_max']).astype(np.int8)
    def conv(self,x,co,kh=3,stride=1,padding=0,dilation=1,act=0,scale=.15,zp=-12):
        shape=self.meta[x]['shape'];n,h,w,ci=shape
        eff=(kh-1)*dilation+1
        oh=(h+stride-1)//stride if padding==0 else (h-eff)//stride+1
        ow=(w+stride-1)//stride if padding==0 else (w-eff)//stride+1
        ws=RNG.uniform(.0004,.04,size=co).astype(np.float32)
        wid=self.tensor(RNG.integers(-30,31,size=(co,kh,kh,ci),dtype=np.int8),ws,[0]*co,True)
        bid=self.tensor(RNG.integers(-300,301,size=co,dtype=np.int32),self.meta[x]['scales'][0]*ws,[0]*co,True)
        return self.op(3,[x,wid,bid],[n,oh,ow,co],scale,zp,
                       {'stride':[stride,stride],'dilation':[dilation,dilation],'padding':padding,'activation':act})
    def pool(self,x,k,kh=2,stride=2,padding=1,act=0):
        sh=self.meta[x]['shape'];n,h,w,c=sh
        oh=(h+stride-1)//stride if padding==0 else (h-kh)//stride+1
        ow=(w+stride-1)//stride if padding==0 else (w-kh)//stride+1
        return self.op(k,[x],[n,oh,ow,c],self.meta[x]['scales'][0],self.meta[x]['zero_points'][0],
                       {'stride':[stride,stride],'kernel':[kh,kh],'padding':padding,'activation':act})

def fixture():
    f=Fixture()
    # A CONNECTED synthetic residual graph exercises real intermediate flow.
    x=f.input((1,3,8,8),.05,-14)
    perm=f.tensor(np.array([0,2,3,1],np.int32),[],[],True)
    x=f.op(39,[x,perm],[1,8,8,3],.05,-14)
    pd=f.tensor(np.array([[0,0],[1,1],[1,1],[0,0]],np.int32),[],[],True)
    x=f.op(34,[x,pd],[1,10,10,3],.05,-14)
    x=f.conv(x,4,3,1,1,act=1,scale=.12,zp=-128)
    a=f.conv(x,4,3,1,0,act=0,scale=.21,zp=11)
    x=f.op(0,[x,a],[1,8,8,4],.2,-128,{'activation':1})
    x=f.pool(x,17)
    x=f.pool(x,1,kh=4,stride=1)
    w=f.tensor(RNG.integers(-20,21,size=(5,4),dtype=np.int8),[.012]*5,[0]*5,True)
    b=f.tensor(np.arange(5,dtype=np.int32),[np.float32(.2)*np.float32(.012)]*5,[0]*5,True)
    x=f.op(9,[x,w,b],[1,5],.3,-3,{'activation':0,'keep_num_dims':False})
    # Independent randomized fixtures exercise the same full compiled runner.
    xx=f.input((1,15,16,3),.04,-17)
    x=f.conv(xx,5,kh=7,stride=2,padding=1,act=1)
    xx=f.input((1,10,11,2),.08,12)
    x=f.conv(xx,3,kh=7,stride=2,padding=0,act=0)
    for trial in range(40):
        size=int(RNG.integers(5,9));ci=int(RNG.integers(1,7));co=int(RNG.integers(1,7))
        xx=f.input((1,size,size,ci),float(RNG.uniform(.01,.3)),int(RNG.integers(-128,128)))
        x=f.conv(xx,co,kh=[1,3][trial%2],stride=1+trial%2,padding=trial%2,dilation=1+(trial%3==0),act=trial%4)
    for trial in range(20):
        xx=f.input((1,5,6,3),.2,-23)
        x=f.pool(xx,1 if trial%2 else 17,kh=2+trial%2,stride=1+trial%2,padding=trial%2,act=trial%4)
    for trial in range(15):
        a=f.input((1,2,3,4),float(RNG.uniform(.01,.5)),int(RNG.integers(-128,128)))
        b=f.input((1,2,3,4),float(RNG.uniform(.01,.5)),int(RNG.integers(-128,128)))
        x=f.op(0,[a,b],[1,2,3,4],.3,-11,{'activation':trial%4})
    for trial in range(15):
        ci=int(RNG.integers(1,15));co=int(RNG.integers(1,10));a=f.input((2,ci),.07,-38)
        w=f.tensor(RNG.integers(-20,21,size=(co,ci),dtype=np.int8),[.012]*co,[0]*co,True)
        x=f.op(9,[a,w,-1],[2,co],.14,12,{'activation':trial%4,'keep_num_dims':False})
    a=f.input((2,3),.1,12);pads=f.tensor(np.array([[1,0],[2,1]],np.int32),[],[],True)
    fill=f.tensor(np.array(-7,np.int8),[],[],True)
    x=f.op(60,[a,pads,fill],[3,6],.1,12)
    x=f.op(22,[x],[1,18],.1,12)
    x=f.op(21,[x],[1,18],.1,12)
    return f,x

# Small forward FlatBuffer builder used ONLY to test the structural reader.
class Ref:
    def __init__(self,f):self.f=f
class FB:
    def __init__(self):self.b=bytearray(b'\0\0\0\0TFL3')
    def align(self,n):self.b.extend(b'\0'*((-len(self.b))%n))
    def table(self,fields):
        self.align(2);v=len(self.b);sz=4+2*len(fields);osz=8+8*len(fields)
        self.b.extend(struct.pack('<HH',sz,osz))
        for i,item in enumerate(fields):self.b.extend(struct.pack('<H',8+8*i if item is not None else 0))
        self.align(8);obj=len(self.b);self.b.extend(b'\0'*osz);struct.pack_into('<i',self.b,obj,obj-v)
        for i,item in enumerate(fields):
            if item is None:continue
            pos=obj+8+8*i
            if isinstance(item,Ref):
                target=item.f();struct.pack_into('<I',self.b,pos,target-pos)
            else:struct.pack_into('<'+item[0],self.b,pos,item[1])
        return obj
    def ints(self,values,fmt='i'):
        self.align(4);pos=len(self.b);self.b.extend(struct.pack('<I',len(values)))
        for v in values:self.b.extend(struct.pack('<'+fmt,v))
        return pos
    def tables(self,items):
        self.align(4);pos=len(self.b);self.b.extend(struct.pack('<I',len(items))+b'\0'*(4*len(items)))
        for i,fn in enumerate(items):
            at=pos+4+4*i;target=fn();struct.pack_into('<I',self.b,at,target-at)
        return pos

def schema_test():
    b=FB();R=Ref
    def tensor(shape,dtype,buf):return lambda:b.table([R(lambda:b.ints(shape)),('b',dtype),('I',buf)])
    def options():return b.table([('b',1),('i',1),('i',1),('b',0)])
    def op():return b.table([('I',0),R(lambda:b.ints([0,1,2])),R(lambda:b.ints([3])),('B',1),R(options)])
    def sg():return b.table([R(lambda:b.tables([tensor([1,3,3,1],9,0),tensor([1,1,1,1],9,1),tensor([1],2,2),tensor([1,3,3,1],9,0)])),R(lambda:b.ints([0])),R(lambda:b.ints([3])),R(lambda:b.tables([op]))])
    def buf(vals):return lambda:b.table([R(lambda:b.ints(vals,'B'))])
    root=b.table([('I',3),R(lambda:b.tables([lambda:b.table([('b',3),None,('i',1),('i',3)])])),R(lambda:b.tables([sg])),None,R(lambda:b.tables([buf([]),buf([1]),buf([0,0,0,0])]))])
    struct.pack_into('<I',b.b,0,root)
    nodes,constants,x,y,nt=graph_schema(bytes(b.b))
    assert (x,y,nt)==(0,3,4) and constants=={1,2}
    assert nodes[0]['options']['dilation']==[1,1]
    for raw in (b'',bytes(b.b[:20]),b'XXXXXXXX'):
        try:graph_schema(raw)
        except (ValueError,IndexError):pass
        else:raise AssertionError('Malformed input was accepted')

def build_run(work,meta,nodes,arrays,refs,inp,out,tag):
    gen=work/'generated';gen.mkdir(exist_ok=True);(work/'results').mkdir(exist_ok=True)
    emit_artifacts(gen,meta,nodes,arrays,refs,inp,out,{'model_sha256':tag,'test_scope':tag},incbin_dir=str(gen))
    exe=work/'test'
    flags=['-O2','-std=c11','-Wall','-Wextra','-Werror','-DHOST_TEST','-DPROFILE=0','-I'+str(ROOT/'src'),'-I'+str(gen)]
    compiler=os.environ.get('HOST_CC','cc')
    sources=[str(ROOT/'src'/s) for s in ['main.c','nn.c','platform.c']]+[str(gen/'model_data.c'),str(gen/'model_blob.S')]
    subprocess.run([compiler,*flags,*sources,'-o',str(exe)],check=True)
    result=subprocess.run([str(exe)],cwd=work,text=True,capture_output=True)
    (work/'run.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout[-3000:]+result.stderr
    assert 'PASS full_model' in result.stdout
    return gen,exe,flags,sources

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path);p.add_argument('--keep',type=Path)
    args=p.parse_args();schema_test();f,last=fixture()
    work=args.keep or Path(tempfile.mkdtemp(prefix='resnet-tests-'))
    work.mkdir(parents=True,exist_ok=True)
    gen,exe,flags,sources=build_run(work,f.meta,f.nodes,f.arrays,f.values,f.input_id,last,'synthetic_kernel_suite')
    lib=work/'kernels.so';compiler=os.environ.get('HOST_CC','cc')
    subprocess.run([compiler,*flags,'-shared','-fPIC',str(ROOT/'src/nn.c'),str(gen/'model_data.c'),str(gen/'model_blob.S'),'-o',str(lib)],check=True)
    fn=ctypes.CDLL(str(lib)).nn_requantize;fn.argtypes=[ctypes.c_int32]*3;fn.restype=ctypes.c_int32
    for _ in range(10000):
        shift=int(RNG.integers(-31,6));limit=(1<<30)>>max(shift,0)
        value=int(RNG.integers(-limit,limit));mult=int(RNG.integers(1,1<<31))
        assert fn(value,mult,shift)==int(rq(value,mult,shift))
    # A wrong golden byte must fail the runner, not print a success banner.
    original=(gen/'reference.bin').read_bytes();bad=bytearray(original);bad[0]^=1
    (gen/'reference.bin').write_bytes(bad)
    subprocess.run([compiler,*flags,*sources,'-o',str(exe)],check=True)
    failed=subprocess.run([str(exe)],cwd=work,text=True,capture_output=True)
    assert failed.returncode!=0 and 'FAIL op=0' in failed.stdout and 'PASS full_model' not in failed.stdout
    (gen/'reference.bin').write_bytes(original)
    # Reject unsupported broadcasting, rather than producing a plausible result.
    n=dict(f.nodes[4]);n['inputs']=[f.input_id,f.nodes[3]['output']]
    try:prepare_op(n,f.meta,f.arrays)
    except ValueError:pass
    else:raise AssertionError('Bad residual shape was accepted')
    result={'synthetic_operators':len(f.nodes),'requantization_cases':10000,
            'flatbuffer_structural_tests':'pass','bad_reference_failure':'pass','unsupported_shape_rejection':'pass',
            'connected_residual_graph':'pass','real_op14':None,
            'full_resnet_tested':False,'spike_tested':False}
    if args.case:
        case=args.case;j=json.loads((case/'case.json').read_text());op=j['selected_operator'];m={}
        arrays={};refs={}
        for role,t in op['tensors'].items():
            if t is None:continue
            m[t['index']]=t
            a=np.load(case/({'output':'expected_output.npy'}.get(role,role+'.npy')),allow_pickle=False)
            if role=='output':refs[t['index']]=a
            else:arrays[t['index']]=a
        o=op['options'];node={'index':0,'kind':3,'name':'CONV_2D','version':op['version'],
                'inputs':op['inputs'],'output':op['outputs'][0],
                'options':{'padding':{'SAME':0,'VALID':1}[o['padding']],
                           'stride':[o['stride_h'],o['stride_w']],
                           'dilation':[o['dilation_h'],o['dilation_w']],
                           'activation':{'NONE':0,'RELU':1,'RELU_N1_TO_1':2,'RELU6':3}[o['fused_activation']]}}
        cw=work/'real_op14';cw.mkdir(exist_ok=True)
        build_run(cw,m,[node],arrays,refs,op['inputs'][0],op['outputs'][0],'uploaded_real_op14')
        result['real_op14']={'elements':int(next(iter(refs.values())).size),'mismatches':0}
    (work/'test_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));print('Test workspace:',work)
    if not args.keep:shutil.rmtree(work)
if __name__=='__main__':main()
