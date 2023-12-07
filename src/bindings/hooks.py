import ctypes

from bindings.utils import CtypePoint, CtypeUint256


def use_get_public_key_by_private_key(library: ctypes.CDLL):
    f = library.getPublicKeyByPrivateKey
    f.argtypes = (
        ctypes.POINTER(CtypePoint),
        ctypes.POINTER(CtypeUint256),
        ctypes.c_uint,
    )

    return f
