from contextlib import AbstractContextManager, nullcontext
from typing import List

from benchmark.reference.curve import Curve, CurvePoint, Fp
from bindings.ecc import EccProtocol
from bindings.utils import Point


class Ecc(EccProtocol):
    def __init__(self) -> None:
        self.p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
        self.curve = Curve(0, 7, self.p)
        self.generator = CurvePoint(
            0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
            0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
            self.curve,
        )

    def get_public_key_by_private_key(
        self,
        private_keys: List[int],
        kernel_context: AbstractContextManager[None] | None = None,
    ) -> List[Point]:
        if kernel_context is None:
            kernel_context = nullcontext()

        private_keys_on_field = [Fp(key, self.p) for key in private_keys]

        with kernel_context:
            public_keys = [self.generator * key for key in private_keys_on_field]

        points = [Point(x=point.x.value, y=point.y.value) for point in public_keys]

        return points
