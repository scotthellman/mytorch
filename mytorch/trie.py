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

    def tokenize(self, key: bytes, missing_value: int = -2) -> list[int]:
        index, leftover = self.traversal_value(key)
        result = [index]
        while len(leftover) > 0:
            index, leftover = self.traversal_value(leftover)
            result.append(index)
        return result
