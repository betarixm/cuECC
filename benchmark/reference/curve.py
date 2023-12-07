# From: https://github.com/mohanson/cryptography-python/blob/master/secp256k1.py

from __future__ import annotations


class Fp:
    def __init__(self, value: int, p: int) -> None:
        self.p = p
        self.value = value % self.p

    def __repr__(self):
        return f"Fp(0x{self.value:064x}, p=0x{self.p:064x})"

    def __eq__(self, that: object) -> bool:
        if not isinstance(that, Fp):
            raise NotImplementedError()

        assert self.p == that.p
        return self.value == that.value

    def __add__(self, that: Fp) -> Fp:
        assert self.p == that.p
        return self.__class__((self.value + that.value) % self.p, self.p)

    def __sub__(self, that: Fp) -> Fp:
        assert self.p == that.p
        return self.__class__((self.value - that.value) % self.p, self.p)

    def __mul__(self, that: Fp) -> Fp:
        assert self.p == that.p
        return self.__class__((self.value * that.value) % self.p, self.p)

    def __pow__(self, that: int) -> Fp:
        return self.__class__(pow(self.value, that, self.p), self.p)

    def __truediv__(self, that: Fp) -> Fp:
        return self * that**-1

    def __pos__(self) -> Fp:
        return self

    def __neg__(self) -> Fp:
        return self.__class__(self.p - self.value, self.p)

    def is_zero(self) -> bool:
        return self.value == 0


class Curve:
    def __init__(self, a: int, b: int, p: int) -> None:
        self.p = p
        self.a: Fp = Fp(a, p)
        self.b: Fp = Fp(b, p)

    def __eq__(self, that: object) -> bool:
        if not isinstance(that, Curve):
            raise NotImplementedError()

        return self.a == that.a and self.b == that.b and self.p == that.p  # type: ignore

    def forward(self, x: Fp) -> Fp:
        assert x.p == self.p
        return x**3 + self.a * x + self.b

    @property
    def identity(self) -> CurvePoint:
        return CurvePoint(0, 0, self)


class CurvePoint:
    def __init__(self, x: int, y: int, curve: Curve) -> None:
        self.x = Fp(x, curve.p)
        self.y = Fp(y, curve.p)
        self.curve = curve

        if x != 0 and y != 0:
            assert self.curve.forward(self.x) == self.y**2

    def __eq__(self, that: object) -> bool:
        if not isinstance(that, CurvePoint):
            raise NotImplementedError()

        assert self.curve == that.curve
        return self.x == that.x and self.y == that.y

    def __add__(self, that: CurvePoint) -> CurvePoint:
        if self.x.is_zero() and self.y.is_zero():
            return that

        if that.x.is_zero() and that.y.is_zero():
            return self

        if self.x == that.x and self.y == -that.y:
            return self.curve.identity

        x1, x2 = self.x, that.x
        y1, y2 = self.y, that.y

        if y1 == y2:
            squared_x1 = x1 * x1
            s = (squared_x1 + squared_x1 + squared_x1 + self.curve.a) / (y1 + y1)
        else:
            s = (y2 - y1) / (x2 - x1)

        x3 = s * s - x1 - x2
        y3 = s * (x1 - x3) - y1

        return CurvePoint(x3.value, y3.value, self.curve)

    def __neg__(self) -> CurvePoint:
        return CurvePoint(self.x.value, -self.y.value, self.curve)

    def __sub__(self, that: CurvePoint) -> CurvePoint:
        return self + -that

    def __mul__(self, k: Fp) -> CurvePoint:
        assert k.p == self.curve.p

        n = k.value
        result = self.curve.identity
        addend = self

        while n:
            b = n & 1
            if b:
                result += addend
            addend += addend
            n >>= 1

        return result

    def __pos__(self) -> CurvePoint:
        return self
