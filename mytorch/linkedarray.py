from dataclasses import dataclass


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
        self.array[right_node.index] = None

        new_right_node = right_node.next_node
        if new_right_node is not None:
            new_right_node.prev_node = left_node

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
