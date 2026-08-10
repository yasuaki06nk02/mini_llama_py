import struct
from dataclasses import dataclass


# GGML type ids used by GGUF.
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q6_K = 14


@dataclass
class TensorInfo:
    name: str
    shape: tuple
    ggml_type: int
    offset: int


class GGUFReader:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        self.metadata = {}
        self.tensors = {}
        self.data_offset = 0

        self._read_header()
        self._read_metadata()
        self._read_tensor_infos()

        self.data_offset = self._align(self.f.tell(), self.alignment)

    def close(self):
        self.f.close()

    def _read(self, fmt):
        size = struct.calcsize(fmt)
        data = self.f.read(size)
        if len(data) != size:
            raise EOFError("unexpected end of GGUF")
        return struct.unpack(fmt, data)[0]

    def _read_string(self):
        n = self._read("<Q")
        data = self.f.read(n)
        if len(data) != n:
            raise EOFError("unexpected end of GGUF string")
        return data.decode("utf-8")

    def _read_value(self, value_type):
        # GGUF metadata value types.
        if value_type == 0:
            return self._read("<B")
        if value_type == 1:
            return self._read("<b")
        if value_type == 2:
            return self._read("<H")
        if value_type == 3:
            return self._read("<h")
        if value_type == 4:
            return self._read("<I")
        if value_type == 5:
            return self._read("<i")
        if value_type == 6:
            return self._read("<f")
        if value_type == 7:
            return bool(self._read("<B"))
        if value_type == 8:
            return self._read_string()
        if value_type == 9:
            item_type = self._read("<I")
            n = self._read("<Q")
            return [self._read_value(item_type) for _ in range(n)]
        if value_type == 10:
            return self._read("<Q")
        if value_type == 11:
            return self._read("<q")
        if value_type == 12:
            return self._read("<d")
        raise ValueError(f"unsupported GGUF metadata type: {value_type}")

    def _read_header(self):
        magic = self.f.read(4)
        if magic != b"GGUF":
            raise ValueError("not a GGUF file")

        self.version = self._read("<I")
        self.n_tensors = self._read("<Q")
        self.n_metadata = self._read("<Q")

    def _read_metadata(self):
        for _ in range(self.n_metadata):
            key = self._read_string()
            value_type = self._read("<I")
            self.metadata[key] = self._read_value(value_type)

        self.alignment = int(self.metadata.get("general.alignment", 32))

    def _read_tensor_infos(self):
        for _ in range(self.n_tensors):
            name = self._read_string()
            n_dims = self._read("<I")

            # GGUF stores dimensions in little-endian uint64.
            dims = tuple(self._read("<Q") for _ in range(n_dims))
            ggml_type = self._read("<I")
            offset = self._read("<Q")

            # GGUF dimension order is reversed relative to the natural
            # NumPy shape when interpreted as a conventional tensor.
            shape = tuple(reversed(dims))

            self.tensors[name] = TensorInfo(
                name=name,
                shape=shape,
                ggml_type=ggml_type,
                offset=offset,
            )

    @staticmethod
    def _align(value, alignment):
        return (value + alignment - 1) // alignment * alignment

    def tensor(self, name):
        if name not in self.tensors:
            raise KeyError(f"tensor not found: {name}")

        info = self.tensors[name]
        self.f.seek(self.data_offset + info.offset)
        return info

    def tensor_bytes(self, info, nbytes):
        self.f.seek(self.data_offset + info.offset)
        data = self.f.read(nbytes)
        if len(data) != nbytes:
            raise EOFError(f"short tensor data: {info.name}")
        return data
