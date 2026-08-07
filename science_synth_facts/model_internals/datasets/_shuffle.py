"""Hash-based deterministic shuffle.

Byte-for-byte the same implementation as ``science_synth_facts.utils`` (so the
sampled rows are identical), lifted here because importing that module pulls in
``together`` and ``safetytooling`` -- neither of which the probing path needs.
"""

import hashlib
from typing import Any, Callable, Iterable, TypeVar

T = TypeVar("T")


def deterministic_shuffle_sort_fn(key: str, _: Any = None) -> int:
    hash = hashlib.sha256(key.encode("utf-8"))
    return int(hash.hexdigest(), 16)


def deterministic_shuffle(data: Iterable[T], key: Callable[[T], str]) -> list[T]:
    return sorted(data, key=lambda t: deterministic_shuffle_sort_fn(key(t), None))
