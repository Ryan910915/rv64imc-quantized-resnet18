"""Bounds-checked subset reader for the TFLite version-3 FlatBuffer schema.
Not a security verifier or general FlatBuffers runtime.
"""
from __future__ import annotations
import struct

class FlatBuffer:
    """Minimal read-only FlatBuffer accessor; not a general model verifier."""

    def __init__(self, data: bytes):
        if len(data) < 8 or data[4:8] != b'TFL3':
            raise ValueError('Expected a non-size-prefixed TFL3 model.')
        self.data = data
        self.root = self.indirect(0)

    def scalar(self, pos: int, fmt: str) -> int:
        size = struct.calcsize('<' + fmt)
        if pos < 0 or pos + size > len(self.data):
            raise ValueError('Invalid/truncated FlatBuffer offset.')
        return int(struct.unpack_from('<' + fmt, self.data, pos)[0])

    def indirect(self, pos: int) -> int:
        offset = self.scalar(pos, 'I')
        target = pos + offset
        if offset == 0 or target < 0 or target + 4 > len(self.data):
            raise ValueError('Invalid FlatBuffer reference.')
        return target

    def field(self, table: int, slot: int) -> int | None:
        vtable = table - self.scalar(table, 'i')
        vsize = self.scalar(vtable, 'H')
        osize = self.scalar(vtable + 2, 'H')
        if vsize < 4 or vsize % 2 or vtable + vsize > len(self.data):
            raise ValueError('Invalid FlatBuffer vtable.')
        if osize < 4 or table + osize > len(self.data):
            raise ValueError('Invalid FlatBuffer table size.')
        at = 4 + 2 * slot
        if at + 2 > vsize:
            return None
        offset = self.scalar(vtable + at, 'H')
        if offset == 0:
            return None
        if offset < 4 or offset >= osize:
            raise ValueError('Invalid FlatBuffer field offset.')
        return table + offset

    def value(self, table: int, slot: int, fmt: str, default: int = 0) -> int:
        pos = self.field(table, slot)
        return default if pos is None else self.scalar(pos, fmt)

    def reference(self, table: int, slot: int) -> int | None:
        pos = self.field(table, slot)
        return None if pos is None else self.indirect(pos)

    def vector(self, table: int, slot: int, item_size: int = 4) -> tuple[int, int]:
        pos = self.reference(table, slot)
        if pos is None:
            return 0, 0
        length = self.scalar(pos, 'I')
        start = pos + 4
        if length > (len(self.data) - start) // item_size:
            raise ValueError('Invalid FlatBuffer vector length.')
        return start, length

    def tables(self, table: int, slot: int) -> list[int]:
        start, length = self.vector(table, slot)
        return [self.indirect(start + 4 * i) for i in range(length)]

    def ints(self, table: int, slot: int) -> list[int]:
        start, length = self.vector(table, slot)
        return [self.scalar(start + 4 * i, 'i') for i in range(length)]

