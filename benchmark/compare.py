from typing import List

from benchmark.adapter import adapt_get_public_key_by_private_key
from bindings.ecc import EccProtocol


def compare_get_public_key_by_private_key(
    participants: List[EccProtocol], private_keys: List[int]
):
    runnables = [adapt_get_public_key_by_private_key(ecc) for ecc in participants]

    results = [runnable(private_keys) for runnable in runnables]

    return results
