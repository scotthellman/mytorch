from collections import Counter
from dataclasses import dataclass

from mytorch.tokenizers.pairdata import TokenData


@dataclass
class ArrayNode:
    index: int
    value: bytes
    prev_node: "ArrayNode | None"
    next_node: "ArrayNode | None" = None
    old_left: bytes | None = None
    old_right: bytes | None = None


@dataclass
class UpdateInfo:
    node_index: int
    old_right_index: int
    # The actual values of the two tokens we merged
    old_left_val: bytes
    old_right_val: bytes
    # The values of the left's prev and the right's next, before merging
    old_prev_val: bytes | None
    old_next_val: bytes | None


class LinkedArray:
    """A linked list that also allows for random access.

    For the specific use-case of learning a BPE vocabulary, we do not need
    to support inserting new entries, only merging existing entries. This means
    that we can initialize as a list that contains linked list nodes (ArrayNodes)
    and then as we merge, we merge the two values into the left hand side and
    null out the right hand side.
    """

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

    def merge(self, left_index: int) -> tuple[ArrayNode, ArrayNode] | None:
        left_node = self[left_index]
        if left_node is None:
            return None
        right_node = left_node.next_node
        if right_node is None:
            print("seems like this should never happen")
            return None

        left_node.old_left = left_node.value
        left_node.old_right = right_node.value

        left_node.value += right_node.value
        left_node.next_node = right_node.next_node
        self.array[right_node.index] = None

        new_right_node = right_node.next_node
        if new_right_node is not None:
            new_right_node.prev_node = left_node
        return left_node, right_node

    def merge_all(
        self, indices: list[int], word_index: int
    ) -> tuple[list[TokenData], Counter[bytes]]:
        # TODO: I have to imagine this could all be written in a cleaner way
        new_pair_counts = {}
        stale_counts = Counter()
        frontier_index = -1
        last_merged_index = None
        previous_right = None
        merge_info = []
        for i in indices:
            if i == frontier_index:
                # We're trying to merge something that got merged as a right
                # This happens eg if we merge the "aa"s in "aaaa" - index 1
                # will be in indices but it will have been subsumed by 0
                # by the time we get here
                continue
            old_node_value = self[i].value
            old_prev = self[i].prev_node
            merge_result = self.merge(i)
            if merge_result is None:
                raise IndexError("Bad index passed to merge_all", i)
            new_node, abandoned_node = merge_result
            # We did destroy this pair (made it its own token)
            # stale_counts[new_node.value] += 1
            stale_counts[abandoned_node.value] += 1
            stale_counts[old_node_value] += 1  # FIXME: do we need this?
            # We have to be careful here - old_prev might have just been merged,
            # but this is for computing the stale values so we need its
            # old right in that case
            if old_prev is None:
                old_prev_val = None
            else:
                if old_prev.index == last_merged_index:
                    old_prev_val = old_prev.old_right
                else:
                    old_prev_val = old_prev.value
            abandoned_next = abandoned_node.next_node
            abandoned_next_val = (
                None if abandoned_next is None else abandoned_next.value
            )
            info = UpdateInfo(
                node_index=i,
                old_right_index=abandoned_node.index,
                old_left_val=old_node_value,
                old_right_val=abandoned_node.value,
                old_prev_val=old_prev_val,
                old_next_val=abandoned_next_val,
            )
            merge_info.append(info)
            previous_right = abandoned_node
            frontier_index = abandoned_node.index
            last_merged_index = i
        # Then, we count pairs
        frontier_index = -1
        for info in merge_info:
            # both left and right were impacted
            new_node = self[info.node_index]
            if new_node.prev_node is not None:
                if new_node.prev_node.index > frontier_index:
                    # if we aren't ahead of the frontier, we've already accounted for this
                    # when building the last next_node's pair
                    stale_counts[info.old_prev_val + info.old_left_val] += 1
                # make sure we we haven't already counted this one
                if new_node.prev_node.index != frontier_index:
                    new_pair = new_node.prev_node.value + new_node.value
                    if new_pair not in new_pair_counts:
                        new_pair_counts[new_pair] = TokenData(new_pair, 0, [])
                    data = new_pair_counts[new_pair]
                    data.count += 1
                    data.locs.append((word_index, new_node.prev_node.index))
                    assert self.array[data.locs[-1][1]] is not None
            if new_node.next_node is not None:
                stale_counts[info.old_right_val + info.old_next_val] += 1
                new_pair = new_node.value + new_node.next_node.value
                if new_pair not in new_pair_counts:
                    new_pair_counts[new_pair] = TokenData(new_pair, 0, [])
                data = new_pair_counts[new_pair]
                data.count += 1
                data.locs.append((word_index, new_node.index))
                assert self.array[data.locs[-1][1]] is not None
                frontier_index = new_node.index
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
