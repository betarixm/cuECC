import itertools


def primes():
    visited = [2]
    for candidate in itertools.count(start=3, step=2):
        if all(
            candidate % prime
            for prime in itertools.takewhile(lambda p: p**2 <= candidate, visited)
        ):
            yield visited[-1]
            visited.append(candidate)
