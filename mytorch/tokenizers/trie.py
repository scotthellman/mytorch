from tqdm import tqdm


class Trie:
    def __init__(self, value: int):
        self.children = {}
        self.value = value

    def insert(self, key: bytes, value: int) -> "Trie":
        if len(key) == 0:
            self.value = value
            return self
        if key[0] not in self.children:
            self.children[key[0]] = Trie(-1)
        next_node = self.children[key[0]]
        return next_node.insert(key[1:], value)

    def __getitem__(self, key: bytes) -> int | None:
        if len(key) == 0:
            return self.value
        if key[0] not in self.children:
            return None
        return self.children[key[0]][key[1:]]

    def traversal_value(self, key: bytes) -> tuple[int, bytes]:
        if len(key) == 0 or key[0] not in self.children:
            return (self.value, key)
        result = self.children[key[0]].traversal_value(key[1:])
        if result[0] == -1:
            return (self.value, key)
        return result

    def traversal_value_iterative(self, key: bytes) -> tuple[int, bytes]:
        current_val = self.value
        best_index = 1
        current_node = self
        for i in range(len(key)):
            current_tok = key[i]
            if current_tok not in current_node.children:
                break
            current_node = current_node.children[current_tok]
            if current_node.value != -1:
                current_val = current_node.value
                best_index = i + 1
        return current_val, key[best_index:]

    def tokenize(self, key: bytes, missing_value: int = -2, pbar_pos=None) -> list[int]:
        pbar = tqdm(total=len(key), position=pbar_pos, leave=False)
        current_length = len(key)
        index, leftover = self.traversal_value_iterative(key)
        delta = current_length - len(leftover)
        pbar.update(delta)
        result = [index]
        while len(leftover) > 0:
            current_length = len(leftover)
            # index, leftover = self.traversal_value(leftover)
            index, leftover = self.traversal_value_iterative(leftover)
            delta = current_length - len(leftover)
            pbar.update(delta)
            result.append(index)
        pbar.close()
        return result
