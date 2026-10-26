"""
A from-scratch GGUF (GPT-Uniform-File, version 3) writer.

GGUF is the model format used by llama.cpp (and LM Studio, Ollama and
friends). The format is delightfully simple:

    header:  magic 'GGUF', version, tensor_count, kv_count
    KVs:     key string, value type, value   (metadata)
    tensors: name, n_dims, ne[], type, offset
    data:    raw tensor bytes, each tensor aligned to 32 bytes

Only what Lito's exporter needs is implemented: strings, ints, floats,
bools and arrays of them, plus F32/F16 tensors. No dependencies beyond
numpy.
"""

import struct

GGUF_MAGIC = 0x46554747  # 'GGUF' little-endian
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32

# GGUF metadata value types
UINT8 = 0
INT8 = 1
UINT16 = 2
INT16 = 3
UINT32 = 4
INT32 = 5
FLOAT32 = 6
BOOL = 7
STRING = 8
ARRAY = 9
UINT64 = 10
INT64 = 11
FLOAT64 = 12

# GGML tensor types
TYPE_F32 = 0
TYPE_F16 = 1

_TYPE_FMT = {
    UINT8: '<B', INT8: '<b', UINT16: '<H', INT16: '<h',
    UINT32: '<I', INT32: '<i', UINT64: '<Q', INT64: '<q',
    FLOAT32: '<f', FLOAT64: '<d', BOOL: '<B',
}


def _pack_string(s):
    b = s.encode('utf-8')
    return struct.pack('<Q', len(b)) + b


def _pack_value(vtype, value):
    if vtype == STRING:
        return _pack_string(value)
    if vtype == BOOL:
        return struct.pack('<B', 1 if value else 0)
    if vtype in _TYPE_FMT:
        return struct.pack(_TYPE_FMT[vtype], value)
    raise ValueError(f'unsupported value type {vtype}')


def _pack_array(item_type, values):
    out = struct.pack('<I', item_type) + struct.pack('<Q', len(values))
    for v in values:
        out += _pack_value(item_type, v)
    return out


class GGUFWriter:
    """Collects metadata and tensors, then writes the file."""

    def __init__(self, arch, name=''):
        self.kv = []      # (key, value_type, packed_value)
        self.tensors = []  # (name, [ne...], ggml_type, ndarray)
        self.data = bytearray()
        if arch:
            self.add_string('general.architecture', arch)
        if name:
            self.add_string('general.name', name)

    # ----------------------------- metadata ---------------------------
    def _add(self, key, vtype, packed):
        # replace any existing entry for the key (keys must be unique)
        self.kv = [(k, t, v) for (k, t, v) in self.kv if k != key]
        self.kv.append((key, vtype, packed))

    def add_string(self, key, value):
        self._add(key, STRING, _pack_string(value))

    def add_uint32(self, key, value):
        self._add(key, UINT32, struct.pack('<I', value))

    def add_uint64(self, key, value):
        self._add(key, UINT64, struct.pack('<Q', value))

    def add_int32(self, key, value):
        self._add(key, INT32, struct.pack('<i', value))

    def add_float32(self, key, value):
        self._add(key, FLOAT32, struct.pack('<f', value))

    def add_bool(self, key, value):
        self._add(key, BOOL, struct.pack('<B', 1 if value else 0))

    def add_array(self, key, item_type, values):
        self._add(key, ARRAY, _pack_array(item_type, values))

    def add_strings(self, key, values):
        self.add_array(key, STRING, list(values))

    def add_uint32s(self, key, values):
        self.add_array(key, UINT32, list(values))

    def add_int32s(self, key, values):
        self.add_array(key, INT32, list(values))

    # ----------------------------- tensors ----------------------------
    def add_tensor(self, name, array, tensor_type=TYPE_F32):
        """Add a tensor. `array` is a numpy array in *torch* shape
        (rows = output features, cols = input features); GGUF dimension
        order is the reverse of torch's, matching every other converter.
        1-D tensors (norms) are written as-is.
        """
        import numpy as np
        array = np.ascontiguousarray(array)
        if array.ndim == 1:
            ne = [array.shape[0]]
        elif array.ndim == 2:
            ne = [array.shape[1], array.shape[0]]  # reversed: [in, out]
        else:
            raise ValueError('only 1-D and 2-D tensors are supported')
        if tensor_type == TYPE_F16:
            array = array.astype(np.float16)
        else:
            array = array.astype(np.float32)
        self.tensors.append((name, ne, tensor_type, array))

    # ----------------------------- output -----------------------------
    def write(self, path):
        out = bytearray()
        out += struct.pack('<IIQQ', GGUF_MAGIC, GGUF_VERSION,
                           len(self.tensors), len(self.kv))

        # Metadata must be sorted? Not required by the spec, but llama.cpp
        # writes sorted keys - keep them sorted for tidiness.
        for key, vtype, packed in sorted(self.kv, key=lambda kv: kv[0]):
            out += _pack_string(key) + struct.pack('<I', vtype) + packed

        # Tensor infos, then the data section (32-byte aligned offsets).
        offset = 0
        infos = bytearray()
        self.data = bytearray()
        for name, ne, tensor_type, array in self.tensors:
            infos += _pack_string(name)
            infos += struct.pack('<I', len(ne))
            for d in ne:
                infos += struct.pack('<Q', d)
            infos += struct.pack('<I', tensor_type)
            infos += struct.pack('<Q', offset)
            b = array.tobytes()
            self.data += b
            padded = (len(b) + GGUF_ALIGNMENT - 1) // GGUF_ALIGNMENT * GGUF_ALIGNMENT
            self.data += b'\x00' * (padded - len(b))
            offset += padded

        out += infos
        # The tensor data section starts at the next multiple of the
        # alignment (GGUF spec); pad the file to it. Tensor offsets are
        # relative to that aligned start.
        pad = -len(out) % GGUF_ALIGNMENT
        out += b'\x00' * pad
        out += self.data

        with open(path, 'wb') as f:
            f.write(bytes(out))
        return offset  # size of the tensor data section
