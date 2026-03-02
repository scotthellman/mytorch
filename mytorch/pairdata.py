# Ok so we want a heap of the pair counts
# But the problem is we're going to be manipulating them, so we also need a
# map into them
# We could track auxiliary index information, but that seems difficult and error prone.
# SO to the rescue: build a heap out of an object that points to the actual data
# One issue: ideally we would delete stale pairs after a merge, but that does
# require proper index tracking. Rather, just track which pairs as stale, and if
# we would pop them, throw htem out and pop again
# https://stackoverflow.com/a/55345010
import heapq
from dataclasses import dataclass
from functools import total_ordering

heapq


@dataclass
class TokenData:
    token: bytes
    count: int
    locs: list[int]
    stale: bool = False

    def __eq__(self, other) -> bool:
        # Be careful, we're technically inconsistent here since
        # bytes aren't treated the same as TokenData. This is safe
        # in our specific case because we will only have one TokenData
        # in our maps at any one time, but is _not_ safe in general.
        # also hashing breaks if you try to mix bytes in and TokenData
        if isinstance(other, bytes):
            return self.token == other
        if isinstance(other, TokenData):
            return self.token == other.token and self.stale == other.stale
        return False

    def __hash__(self) -> int:
        return hash(self.token) + hash(self.stale)


@total_ordering
@dataclass
class TokenPointer:
    """A pointer to PairData that emits that PairData's count for heapifying"""

    referent: TokenData

    def __eq__(self, other) -> bool:
        if not isinstance(other, TokenPointer):
            return False
        return self.referent.count == other.referent.count

    def __lt__(self, other) -> bool:
        if not isinstance(other, TokenPointer):
            return False
        return self.referent.count < other.referent.count


class TokenHeap:
    def __init__(self):
        self.heap = []
        self.pairs = []
        self.pair_lookup = {}
        self.pointer_lookup = {}
        self.stale = set()

    def insert_token(self, pair: TokenData):
        assert not pair.stale
        self.pair_lookup[pair] = len(self.pairs)
        self.pairs.append(pair)
        pointer = TokenPointer(pair)
        heapq.heappush(self.heap, pointer)
        self.pointer_lookup[pair] = pointer

    def pop(self) -> TokenData:
        pointer = heapq.heappop(self.heap)
        data = pointer.referent
        if data in self.stale:
            return self.pop()
        return data

    def mark_as_stale(self, stale_values: list[bytes]):
        for stale in stale_values:
            self.stale.add(self.pointer_lookup[stale].referent)
