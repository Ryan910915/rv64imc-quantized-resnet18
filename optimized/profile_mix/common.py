"""ELF and RV64IMC decoding helpers. Uses the Python standard library only."""
from __future__ import annotations
import hashlib, struct
from pathlib import Path

CATEGORIES = ('load','store','integer_alu','multiply','branch','jump_call_return',
              'divide_remainder','csr','fence','system','custom','unknown')

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

class ELF:
    def __init__(self, path: Path):
        self.path=path; self.data=path.read_bytes()
        if self.data[:6]!=b'\x7fELF\x02\x01':
            raise ValueError('Expected a little-endian ELF64 file')
        h=struct.unpack_from('<16sHHIQQQIHHHHHH',self.data)
        if h[2]!=243: raise ValueError('ELF is not RISC-V')
        off,entsize,num,shstr=h[6],h[11],h[12],h[13]
        if entsize!=64 or num==0:raise ValueError('Unsupported ELF section table')
        raw=[struct.unpack_from('<IIQQQQIIQQ',self.data,off+i*entsize) for i in range(num)]
        names=self.data[raw[shstr][4]:raw[shstr][4]+raw[shstr][5]]
        def zstr(buf,start):return buf[start:buf.index(0,start)].decode('utf-8','replace')
        self.sections=[]
        for x in raw:
            self.sections.append(dict(name=zstr(names,x[0]),type=x[1],flags=x[2],addr=x[3],
                                      offset=x[4],size=x[5],link=x[6],entsize=x[9]))
        self.executable=[s for s in self.sections if s['flags']&4 and s['size']]
        if not self.executable:raise ValueError('ELF has no executable section')
        self.base=min(s['addr'] for s in self.executable)
        self.limit=max(s['addr']+s['size'] for s in self.executable)
        self.symbols={};self.functions=[]
        for s in self.sections:
            if s['type']!=2:continue
            strings=self.sections[s['link']]
            buf=self.data[strings['offset']:strings['offset']+strings['size']]
            if s['entsize']!=24:raise ValueError('Unsupported ELF64 symbol entry')
            for pos in range(s['offset'],s['offset']+s['size'],24):
                name,info,other,index,value,size=struct.unpack_from('<IBBHQQ',self.data,pos)
                if not name or not index:continue
                text=zstr(buf,name)
                self.symbols[text]=value
                if info&15==2 and size:self.functions.append((value,value+size,text))
        self.functions.sort()
    def instruction(self,pc: int,length: int) -> int:
        for s in self.executable:
            if s['addr']<=pc and pc+length<=s['addr']+s['size']:
                o=s['offset']+pc-s['addr']
                return int.from_bytes(self.data[o:o+length],'little')
        raise ValueError(f'PC {pc:#x} is not in an executable ELF section')
    def counter_pc(self) -> int:
        pc=self.symbols.get('platform_instret')
        if pc is None or self.instruction(pc,4)!=0xb0202573:
            raise ValueError('Expected PROFILE=1 ELF with platform_instret starting in csrrs a0,minstret,x0')
        return pc
    def function(self,pc: int) -> str:
        hits=[f for f in self.functions if f[0]<=pc<f[1]]
        return min(hits,key=lambda f:(f[1]-f[0],f[2]))[2] if hits else '(no function symbol)'

def classify(bits: int,length: int) -> tuple[str,str,int]:
    """Return disjoint category, instruction-family name, memory width in bytes.

    Branch means conditional branch; jumps/calls/returns have a separate bucket.
    Compressed forms count as one instruction, not half. Raw encoding avoids
    objdump alias ambiguity. Custom is an opcode-space count, NOT a semantic
    dot-product or requantization count. Unsupported encoding remains unknown.
    """
    if length==2 and bits&3!=3:
        q=bits&3; f=(bits>>13)&7
        if q==0:
            if f in (2,3):return 'load',('c.lw' if f==2 else 'c.ld'),(4 if f==2 else 8)
            if f in (6,7):return 'store',('c.sw' if f==6 else 'c.sd'),(4 if f==6 else 8)
            if f==0 and bits>>5:return 'integer_alu','c.addi4spn',0
        elif q==1:
            if f in (0,1,2,3,4):return 'integer_alu',('c.addi/nop','c.addiw','c.li','c.lui/addi16sp','c.arith')[f],0
            if f==5:return 'jump_call_return','c.j',0
            if f in (6,7):return 'branch',('c.beqz' if f==6 else 'c.bnez'),0
        elif q==2:
            if f==0:return 'integer_alu','c.slli',0
            if f in (2,3):return 'load',('c.lwsp' if f==2 else 'c.ldsp'),(4 if f==2 else 8)
            if f in (6,7):return 'store',('c.swsp' if f==6 else 'c.sdsp'),(4 if f==6 else 8)
            if f==4:
                rd=(bits>>7)&31;rs2=(bits>>2)&31;hi=(bits>>12)&1
                if rs2:return 'integer_alu',('c.add' if hi else 'c.mv'),0
                if rd:return 'jump_call_return',('c.jalr' if hi else 'c.jr'),0
                if hi:return 'system','c.ebreak',0
        return 'unknown','unsupported16',0
    if length!=4 or bits&3!=3:return 'unknown','unsupported_length',0
    op=bits&127; f3=(bits>>12)&7; f7=(bits>>25)&127
    if op==0x03 and f3<=6:
        return 'load',('lb','lh','lw','ld','lbu','lhu','lwu')[f3],(1,2,4,8,1,2,4)[f3]
    if op==0x23 and f3<=3:return 'store',('sb','sh','sw','sd')[f3],(1,2,4,8)[f3]
    if op in (0x33,0x3b) and f7==1:
        if f3<=3 and (op==0x33 or f3==0):
            return 'multiply',('mul','mulh','mulhsu','mulhu')[f3]+('w' if op==0x3b else ''),0
        if f3>=4:return 'divide_remainder',('div','divu','rem','remu')[f3-4]+('w' if op==0x3b else ''),0
        return 'unknown','reserved_m',0
    if op in (0x13,0x1b,0x33,0x3b,0x37,0x17):
        return 'integer_alu',{0x13:'op_imm',0x1b:'op_imm_32',0x33:'op',0x3b:'op_32',0x37:'lui',0x17:'auipc'}[op],0
    if op==0x63 and f3 in (0,1,4,5,6,7):return 'branch',{0:'beq',1:'bne',4:'blt',5:'bge',6:'bltu',7:'bgeu'}[f3],0
    if op in (0x6f,0x67):return 'jump_call_return',('jal' if op==0x6f else 'jalr'),0
    if op==0x73:
        if f3 in (1,2,3,5,6,7):return 'csr','csr',0
        if f3==0:return 'system','system',0
    if op==0x0f:return 'fence','fence',0
    if op in (0x0b,0x2b,0x5b,0x7b):return 'custom',f'custom_opcode_{op:#x}',0
    return 'unknown',f'unsupported_opcode_{op:#x}',0
