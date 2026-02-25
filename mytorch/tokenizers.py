from collections import Counter


class NaiveBPE:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocabulary = None
        self.trie = None
        self.index_lookup = {}

    def get_word_counts(self, texts: list[bytes]) -> Counter:
        result = Counter()
        for text in texts:
            for i, tok in enumerate(text.split(b" ")):
                if i == 0:
                    result[tok] += 1
                else:
                    result[b" " + tok] += 1
        return result

    def fit(self, texts: list[bytes]):
        word_counts = self.get_word_counts(texts)

        # naive implementation:
        # get tokens out of words
        # find most common token pair
        # merge
        # repeat until at vocab size

        # set up the trie
        self.trie = Trie(0)

        pair_counts = Counter()
        seen = set()
        for word in word_counts.keys():
            for tok in word:
                # we really want to view these as bytes, not ints
                tok = int.to_bytes(tok)
                if tok not in seen:
                    self.trie.insert(tok, len(self.index_lookup))
                    self.index_lookup[len(self.index_lookup)] = tok
                    seen.add(tok)

        while len(self.index_lookup) < self.vocab_size:
            if len(self.index_lookup) % 500 == 0:
                print(len(self.index_lookup))
            # get the new counts
            pair_counts = Counter()
            for word, word_count in word_counts.items():
                toks = self.trie.tokenize(word)
                for l, r in zip(toks, toks[1:]):
                    pair_counts[(l, r)] += word_count
            if len(pair_counts) == 0:
                # we've formed every pair
                print("out of pairs")
                break
            # find the most common pair
            top_pair, count = pair_counts.most_common(1)[0]
            actual_bytes = (
                self.index_lookup[top_pair[0]] + self.index_lookup[top_pair[1]]
            )

            self.trie.insert(actual_bytes, len(self.index_lookup))
            self.index_lookup[len(self.index_lookup)] = actual_bytes

    def tokenize(self, text: bytes) -> list[int]:
        if self.trie is None:
            raise ValueError("Tokenizer must be fit to text before use")
        return self.trie.tokenize(text)

    def untokenize(self, toks: list[int]) -> bytes:
        if self.index_lookup is None:
            raise ValueError("Tokenizer must be fit to text before use")
        return b"".join(self.index_lookup[t] for t in toks)


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
