import ctypes
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import List, Protocol

from bindings.hooks import use_get_public_key_by_private_key
from bindings.utils import (
    CtypePoint,
    CtypeUint256,
    Point,
    as_ctype_uint256,
    as_python_int,
)


class EccProtocol(Protocol):
    def get_public_key_by_private_key(
        self,
        private_keys: List[int],
        kernel_context: AbstractContextManager[None] | None = None,
    ) -> List[Point]:
        ...


class Ecc(EccProtocol):
    def __init__(self, library_path: Path) -> None:
        self._library_path = library_path
        self._library = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)

        self._get_public_key_by_private_key = use_get_public_key_by_private_key(
            self._library
        )

    def get_public_key_by_private_key(
        self,
        private_keys: List[int],
        kernel_context: AbstractContextManager[None] | None = None,
    ) -> List[Point]:
        if kernel_context is None:
            kernel_context = nullcontext()

        n = len(private_keys)

        args = (
            (CtypePoint * n)(),
            (CtypeUint256 * (n * 4))(*[as_ctype_uint256(key) for key in private_keys]),
            n,
        )

        with kernel_context:
            self._get_public_key_by_private_key(*args)

        points = [
            Point(x=as_python_int(point.x), y=as_python_int(point.y))
            for point in args[0]
        ]

        return points
