from collections import Counter

from mytorch.linkedarray import LinkedArray
from mytorch.pairdata import TokenData, TokenHeap
from mytorch.trie import Trie


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


class BPE:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocabulary = None
        self.trie = None
        self.array = None
        self.index_lookup = {}
        self.heap = TokenHeap()

    def fit(self, text: bytes):
        self.trie = Trie(0)
        self.array = LinkedArray(text)
        token_counts = Counter()
        # FIXME: so we've lost any notion of word chunks
        initial_counts = {}
        # First, init our vocab to be all bytes
        # for i in range(256):
        #    value = int.to_bytes(i)

        # now, we need to know our starting pair counts
        seen = set()
        for i in range(len(text)):
            pair = text[i : i + 2]
            if len(pair) != 2:
                continue
            for j in range(2):
                tok = pair[j : j + 1]
                if tok not in seen:
                    seen.add(tok)
                    self.trie.insert(tok, len(self.index_lookup))
                    self.index_lookup[len(self.index_lookup)] = tok
                # inverted math due to min heap
                token_counts[tok] -= 1
            if pair not in initial_counts:
                initial_counts[pair] = TokenData(token=pair, count=0, locs=[])
            data = initial_counts[pair]
            data.count += 1
            data.locs.append(i)
        # Now we can build our heap
        for data in initial_counts.values():
            # we have a min heap
            data.count = -data.count
            self.heap.insert_token(data)
            token_counts[data.token] += data.count

        # We should be set up for the iterative stage now
        while len(self.index_lookup) < self.vocab_size:
            # pop from the heap
            # add to the trie
            # merge in the array
            # need to manage global counts so that we can compute the new counts
            # insert new counts and mark old counts as stale
            data = self.heap.pop()
            print("Currently the vocab is")
            print(list(self.index_lookup.values()))
            print(token_counts)
            print("now looking at")
            print(data.token)
            print(data.locs)
            self.trie.insert(data.token, len(self.index_lookup))
            self.index_lookup[len(self.index_lookup)] = data.token

            impacted_indices = data.locs
            new_counts, stale_counts = self.array.merge_all(impacted_indices)
            print(stale_counts)
            stale_pairs = []
            # The only thing missing is some sort of global counter. We need this because
            # e.g. let's say we had "a a b c d" and merged to "a a bc d"
            # from merge_all, we know the counts for abc and bcd
            # but we _don't_ know what the new count for "a" is
            # we need to update our counts now
            for pair, count in stale_counts.items():
                if len(pair) > 1:
                    stale_pairs.append(pair)
                # again, min heap, so our math is inverted
                token_counts[pair] += count
                assert token_counts[pair] <= 0

            self.heap.mark_as_stale(stale_pairs)
            for data in new_counts:
                data.count = -data.count
                self.heap.insert_token(data)
                token_counts[data.token] = data.count

    def tokenize(self, text: bytes) -> list[int]:
        if self.trie is None:
            raise ValueError("Tokenizer must be fit to text before use")
        return self.trie.tokenize(text)

    def untokenize(self, toks: list[int]) -> bytes:
        if self.index_lookup is None:
            raise ValueError("Tokenizer must be fit to text before use")
        return b"".join(self.index_lookup[t] for t in toks)
