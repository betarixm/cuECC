import csv
import itertools
import random
from typing import TextIO

import click

from benchmark.adapter import adapt_get_public_key_by_private_key
from benchmark.reference.ecc import Ecc as ReferenceEcc
from benchmark.settings import LIBCUECC_SO_PATH
from benchmark.utils import primes as prime_generator
from bindings.ecc import Ecc as CuEcc


def report(out: TextIO, start_from: int = 1, end_at: int = 30):
    print("[*] Benchmark: get_public_key_from_private_key")

    writer = csv.DictWriter(out, fieldnames=["n", "name", "elapsed_time", "is_same"])
    writer.writeheader()

    cu_ecc = CuEcc(LIBCUECC_SO_PATH)
    reference_ecc = ReferenceEcc()

    print("- Generating primes...")
    primes = list(itertools.islice(prime_generator(), 2**18))

    runnables = [
        (name, adapt_get_public_key_by_private_key(ecc))
        for name, ecc in [("CUDA", cu_ecc), ("Python", reference_ecc)]
    ]

    for n in range(start_from, end_at + 1):
        print(f"- For 2**n where n = {n}...")

        private_keys = [random.choice(primes) for _ in range(2**n)]

        results = [(name, *runnable(private_keys)) for name, runnable in runnables]

        is_same = all(
            [
                all([public_key == row[0] for public_key in row])
                for row in zip(*[public_keys for _, public_keys, _ in results])
            ]
        )

        writer.writerows(
            [
                {"n": n, "name": name, "elapsed_time": elapsed_time, "is_same": is_same}
                for name, _, elapsed_time in results
            ]
        )

        out.flush()

        for name, public_keys, elapsed_time in results:
            print(f"  - {name}: {elapsed_time} (Sample: {public_keys[0]})")


@click.command()
@click.option("--attempt-key", required=False)
@click.option("--start-from", required=False, default=1)
@click.option("--end-at", required=False, default=30)
def main(attempt_key: str | None, start_from: int, end_at: int):
    from uuid import uuid4

    from benchmark.settings import BUILD_DIR

    if attempt_key is None:
        attempt_key = uuid4().hex[:6]

    print("[*] Attempt key:", attempt_key)

    with open(BUILD_DIR / f"report-public-keys-{attempt_key}.csv", "w") as out:
        report(out, start_from, end_at)


if __name__ == "__main__":
    main()
