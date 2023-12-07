from contextlib import contextmanager
from time import process_time
from typing import List

from bindings.ecc import EccProtocol


def adapt_get_public_key_by_private_key(ecc: EccProtocol):
    def f(private_keys: List[int]):
        mutable_elapsed_time = 0.0

        @contextmanager
        def use_timer():
            nonlocal mutable_elapsed_time
            start_time = process_time()
            yield
            mutable_elapsed_time = process_time() - start_time

        public_keys = ecc.get_public_key_by_private_key(
            private_keys, kernel_context=use_timer()
        )

        return public_keys, mutable_elapsed_time

    return f
