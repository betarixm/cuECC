import ctypes
import struct
from dataclasses import dataclass
from typing import Tuple

CtypeUint256 = ctypes.c_uint64 * 4


@dataclass
class Point:
    x: int
    y: int


class CtypePoint(ctypes.Structure):
    _fields_ = [("x", CtypeUint256), ("y", CtypeUint256)]


def as_uint256(value: int) -> Tuple[int, int, int, int]:
    return tuple(reversed(struct.unpack(">4Q", value.to_bytes(32, "big", signed=False))))  # type: ignore


def as_ctype_uint256(value: int) -> CtypeUint256:
    return CtypeUint256(*as_uint256(value))


def as_python_int(value: CtypeUint256) -> int:
    return int.from_bytes(
        b"".join(reversed([v.to_bytes(8, "big", signed=False) for v in value])),
        "big",
        signed=False,
    )
