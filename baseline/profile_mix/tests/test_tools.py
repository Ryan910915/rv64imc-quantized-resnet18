#!/usr/bin/env python3
"""Offline unit tests; NOT a claim that Spike or full ResNet was executed here."""
from __future__ import annotations
import collections,contextlib,csv,io,json,os,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from common import ELF,CATEGORIES,classify
from setup_spike import patch_execute
from summarize import read_counts,summarize

class Tests(unittest.TestCase):
    def test_01_base_loads(self):
        for f,w in enumerate((1,2,4,8,1,2,4)):
            self.assertEqual(classify((f<<12)|0x03,4)[::2],('load',w))
    def test_02_base_stores(self):
        for f,w in enumerate((1,2,4,8)):
            self.assertEqual(classify((f<<12)|0x23,4)[::2],('store',w))
    def test_03_multiply(self):
        for f in range(4):self.assertEqual(classify((1<<25)|(f<<12)|0x33,4)[0],'multiply')
        self.assertEqual(classify((1<<25)|0x3b,4)[1],'mulw')
    def test_04_divide(self):
        for op in (0x33,0x3b):
            for f in range(4,8):self.assertEqual(classify((1<<25)|(f<<12)|op,4)[0],'divide_remainder')
    def test_05_control_and_custom(self):
        for f in (0,1,4,5,6,7):self.assertEqual(classify((f<<12)|0x63,4)[0],'branch')
        for op in (0x67,0x6f):self.assertEqual(classify(op,4)[0],'jump_call_return')
        for op in (0x0b,0x2b,0x5b,0x7b):self.assertEqual(classify(op,4)[0],'custom')
        self.assertEqual(classify(0xb0202573,4)[0],'csr')
        self.assertEqual(classify(0x0330000f,4)[0],'fence')
    def test_06_compressed(self):
        for q in (0,2):
            for f,c,w in ((2,'load',4),(3,'load',8),(6,'store',4),(7,'store',8)):
                self.assertEqual(classify((f<<13)|q,2)[::2],(c,w))
        for f in (6,7):self.assertEqual(classify((f<<13)|1,2)[0],'branch')
        self.assertEqual(classify(0x8082,2)[0],'jump_call_return')
        self.assertEqual(classify(0x9002,2)[0],'system')
    def test_07_no_fp_guess(self):
        for op in (0x07,0x27,0x53,0x57,0x2f):self.assertEqual(classify(op,4)[0],'unknown')
        self.assertEqual(classify(0,2)[0],'unknown')
    def test_08_patch_layout(self):
        fixture='''#include <cassert>
static inline reg_t execute_insn_logged(processor_t* p, reg_t pc, insn_fetch_t fetch) {
 reg_t npc;
 npc = fetch.func(p, fetch.insn, pc);
 if (npc != PC_SERIALIZE_BEFORE) { /* commit log */ }
  p->update_histogram(pc);
 return npc;
}
'''
        result=patch_execute(fixture)
        self.assertIn('mix.retired(p->get_id()',result)
        self.assertIn('if (npc != PC_SERIALIZE_BEFORE)\n      mix.retired',result)
        with self.assertRaises(ValueError):patch_execute(result)
        with self.assertRaises(ValueError):patch_execute('not a known version')
    def test_09_observer_cpp(self):
        compiler=shutil.which('g++')
        if not compiler:self.skipTest('g++ absent')
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);src=d/'observer_test.cc';binary=d/'test';counts=d/'counts.csv'
            src.write_text('''#include "observer.h"
int main() {
 auto& x=resnet_profile_mix::instance();
 x.retired(0,0x1000,0x13,4); // outside region
 x.retired(0,0x1100,0xffffffffb0202573ULL,4); // serialized retry already excluded by caller
 x.retired(0,0x1000,0x00012583,4); // lw
 for(int i=0;i<3;++i)x.retired(0,0x1004,0x026283b3,4); // mul
 x.retired(0,0x1008,0xffffffffffffc606ULL,2); // sign-extended compressed store
 x.retired(0,0x1100,0xb0202573,4); // closing read excluded
 x.retired(0,0x1000,0x13,4); // outside region
 return 0;
}
''')
            subprocess.run([compiler,'-std=c++17','-Wall','-Wextra','-Werror','-I',str(ROOT),str(src),'-o',str(binary)],check=True)
            env=os.environ.copy();env.update(RESNET_MIX_COUNTER_PC='0x1100',RESNET_MIX_TEXT_BASE='0x1000',
                RESNET_MIX_TEXT_LIMIT='0x1200',RESNET_MIX_REGIONS='1',RESNET_MIX_OUTPUT=str(counts))
            subprocess.run([str(binary)],env=env,check=True)
            txt=counts.read_text();self.assertIn('# CLOSE op=0 total=6',txt)
            self.assertIn('# END regions=1 expected=1 active=0',txt)
            self.assertIn('0xc606,2,1',txt);self.assertNotIn('0xffffffff',txt)
    def test_10_cross_elf_and_report(self):
        cross=shutil.which('riscv64-unknown-elf-gcc');clang=shutil.which('clang')
        if not cross and not clang:self.skipTest('No RISC-V cross compiler')
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);elf_path=d/'smoke.elf'
            cmd=[cross] if cross else [clang,'--target=riscv64-unknown-elf','-fuse-ld=lld']
            cmd+=['-march=rv64imc','-mabi=lp64','-nostdlib','-Wl,--no-relax','-T',str(ROOT/'smoke.ld'),str(ROOT/'smoke.S'),'-o',str(elf_path)]
            subprocess.run(cmd,check=True)
            elf=ELF(elf_path);gate=elf.counter_pc();self.assertEqual(elf.instruction(gate,4),0xb0202573)
            # Feed actual compiled smoke instructions through a tiny test-only
            # interpreter of the subset used here; this is NOT Spike.
            trace,expected=emulate_smoke(elf)
            self.assertEqual([sum(c.values()) for c in expected],[24,13])
            counts=d/'roi_counts.csv';log=d/'spike.log';manifest=d/'manifest.json'
            lines=[f'# RESNET_MIX_V1 gate={gate:#x} expected=2','op,pc,bits,length,count']
            for i,t in enumerate(trace):
                for (pc,bits,length),n in sorted(t.items()):lines.append(f'{i},{pc:#x},{bits:#x},{length},{n}')
                lines.append(f'# CLOSE op={i} total={sum(t.values())}')
            lines.append('# END regions=2 expected=2 active=0');counts.write_text('\n'.join(lines)+'\n')
            log.write_text('PROFILE op=0 kind=TEST instret=24 dense_macs=3 valid_macs=3\nPROFILE op=1 kind=TEST instret=13 dense_macs=0 valid_macs=0\nPASS full_model operators=2 outputs=1 mismatches=0 checksum=1\nPROFILE_SUM dispatch_regions_instret=37\n')
            manifest.write_text(json.dumps({'operators':[{'compiled':{'dense_macs':3,'valid_macs':3}},{'compiled':{'dense_macs':0,'valid_macs':0}}]}))
            with contextlib.redirect_stdout(io.StringIO()):
                summary=summarize(counts,log,elf_path,manifest,d/'summary')
            self.assertEqual(summary['dispatch_regions_instret'],37)
            self.assertEqual(summary['loads'],4);self.assertEqual(summary['stores'],4)
            self.assertEqual(summary['instruction_mix']['multiply'],3)
            # Fail closed on changed target total and incomplete capture.
            log.write_text(log.read_text().replace('instret=24','instret=25'))
            with self.assertRaises(ValueError):summarize(counts,log,elf_path,manifest,d/'bad')
            counts.write_text(counts.read_text().replace('active=0','active=1'))
            with self.assertRaises(ValueError):read_counts(counts,elf)

MASK=(1<<64)-1
def sx(v,n):return (v&((1<<(n-1))-1))-(v&(1<<(n-1)))
def emulate_smoke(elf):
    """Small independent test interpreter. Only the smoke's opcodes are accepted."""
    x=[0]*32;mem={};pc=elf.symbols['_start'];retired=0;gate=elf.counter_pc();active=False
    traces=[];groups=[]
    def get(a,n):return sum(mem.get(a+j,0)<<(8*j) for j in range(n))
    def put(a,n,v):
        for j in range(n):mem[a+j]=(v>>(8*j))&255
    for step in range(1000):
        low=elf.instruction(pc,2);length=4 if low&3==3 else 2;b=elf.instruction(pc,length);npc=pc+length
        if pc==gate:
            if active:active=False
            else:active=True;traces.append(collections.Counter());groups.append(collections.Counter())
        if active:
            traces[-1][(pc,b,length)]+=1;groups[-1][classify(b,length)[0]]+=1
        if length==4:
            op=b&127;rd=(b>>7)&31;f=(b>>12)&7;r1=(b>>15)&31;r2=(b>>20)&31;imm=sx(b>>20,12)
            if op==0x17:x[rd]=(pc+sx(b&0xfffff000,32))&MASK
            elif op==0x13:
                if f!=0:raise ValueError('Smoke op_imm not addi')
                x[rd]=(x[r1]+imm)&MASK
            elif op==0x67:npc=(x[r1]+imm)&~1;x[rd]=pc+4
            elif op==0x6f:
                j=((b>>31)<<20)|(((b>>12)&255)<<12)|(((b>>20)&1)<<11)|(((b>>21)&1023)<<1)
                npc=pc+sx(j,21);x[rd]=pc+4
            elif op==0x63:
                j=((b>>31)<<12)|(((b>>7)&1)<<11)|(((b>>25)&63)<<5)|(((b>>8)&15)<<1)
                cond=(x[r1]!=x[r2]) if f==1 else (x[r1]==x[r2]) if f==0 else None
                if cond is None:raise ValueError('Unsupported smoke branch')
                if cond:npc=pc+sx(j,13)
            elif op==0x33:
                f7=b>>25
                if f7==0x20 and f==0:x[rd]=(x[r1]-x[r2])&MASK
                elif f7==1 and f==0:x[rd]=(x[r1]*x[r2])&MASK
                elif f7==1 and f==4:x[rd]=x[r1]//x[r2]
                elif f7==1 and f==6:x[rd]=x[r1]%x[r2]
                else:raise ValueError('Unsupported smoke ALU')
            elif op==0x03:
                n=1<<f;x[rd]=sx(get(x[r1]+imm,n),8*n)&MASK
            elif op==0x23:
                si=sx(((b>>25)<<5)|((b>>7)&31),12);a=x[r1]+si;n=1<<f
                put(a,n,x[r2])
                if a==elf.symbols['tohost']:
                    if x[r2]!=1:raise ValueError('Smoke assembly self-check returned failure')
                    return traces,groups
            elif op==0x73:
                csr=b>>20
                if csr==0xb02:x[rd]=retired
                elif csr!=0x320:raise ValueError('Unexpected CSR')
            elif op!=0x0f:raise ValueError(f'Unsupported smoke instruction {b:#x}')
        else:
            q=b&3;f=b>>13;rd=(b>>7)&31;r2=(b>>2)&31
            if q==1 and f==2:x[rd]=sx(((b>>12)&1)<<5|r2,6)&MASK
            elif q==2 and f==6:
                off=((b>>9)&15)<<2|((b>>7)&3)<<6;put(x[2]+off,4,x[r2])
            elif q==2 and f==2:
                off=((b>>12)&1)<<5|((b>>4)&7)<<2|((b>>2)&3)<<6;x[rd]=sx(get(x[2]+off,4),32)&MASK
            elif q==2 and f==4 and b&(1<<12):x[rd]=(x[rd]+x[r2])&MASK
            elif q==2 and f==0:x[rd]=(x[rd]<<(((b>>12)&1)<<5|r2))&MASK
            else:raise ValueError(f'Unsupported compressed smoke opcode {b:#x}')
        x[0]=0;retired+=1;pc=npc
    raise ValueError('Smoke test interpreter exceeded step limit')

if __name__=='__main__':unittest.main(verbosity=2)
