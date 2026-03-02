from collections import Counter
from dataclasses import dataclass

from mytorch.pairdata import TokenData


@dataclass
class ArrayNode:
    index: int
    value: bytes
    prev_node: "ArrayNode | None"
    next_node: "ArrayNode | None" = None


class LinkedArray:
    def __init__(self, initial_values: bytes):
        self.array = []
        previous = None
        for i in range(len(initial_values)):
            # we have to take slices or we'll end up with an int rather than a bytes
            val = initial_values[i : i + 1]
            node = ArrayNode(index=i, prev_node=previous, value=val)
            if previous:
                previous.next_node = node
            self.array.append(node)
            previous = node

    def __getitem__(self, index: int) -> ArrayNode | None:
        return self.array[index]

    def merge(self, left_index: int) -> ArrayNode | None:
        left_node = self[left_index]
        if left_node is None:
            return None
        right_node = left_node.next_node
        if right_node is None:
            print("seems like this should never happen")
            return None

        left_node.value += right_node.value
        left_node.next_node = right_node.next_node
        self.array[right_node.index] = None

        new_right_node = right_node.next_node
        if new_right_node is not None:
            new_right_node.prev_node = left_node
        return left_node

    def merge_all(self, indices: list[int]) -> tuple[list[TokenData], Counter[bytes]]:
        new_pair_counts = {}
        stale_counts = Counter()
        frontier_index = 0
        for i in indices:
            if i < frontier_index:
                # this token got merged
                continue
            new_node = self.merge(i)
            if new_node is None:
                raise IndexError("Bad index passed to merge_all", i)
            # both left and right were impacted
            if new_node.prev_node is not None:
                stale_counts[new_node.prev_node.value] += 1
                new_pair = new_node.prev_node.value + new_node.value
                if new_pair not in new_pair_counts:
                    new_pair_counts[new_pair] = TokenData(new_pair, 0, [])
                data = new_pair_counts[new_pair]
                data.count += 1
                data.locs.append(i - 1)
            if new_node.next_node is not None:
                stale_counts[new_node.next_node.value] += 1
                new_pair = new_node.next_node.value + new_node.value
                if new_pair not in new_pair_counts:
                    new_pair_counts[new_pair] = TokenData(new_pair, 0, [])
                data = new_pair_counts[new_pair]
                data.count += 1
                data.locs.append(i - 1)
                frontier_index = new_node.next_node.index
        return list(new_pair_counts.values()), stale_counts

    def get_next_index(self, index: int) -> int | None:
        node = self[index]
        if node is None or node.next_node is None:
            return None
        return node.next_node.index

    def get_prev_index(self, index: int) -> int | None:
        node = self[index]
        if node is None or node.prev_node is None:
            return None
        return node.prev_node.index
