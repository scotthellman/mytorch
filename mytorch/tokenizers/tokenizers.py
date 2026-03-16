"""Efficient tokenization and vocab construction for byte-pair encoding.

(Well, in the BPE class. NaiveBPE, as the name suggests, is a very
naive implementation.)
"""

from collections import Counter

from mytorch.tokenizers.linkedarray import LinkedArray
from mytorch.tokenizers.pairdata import TokenData, TokenHeap
from mytorch.tokenizers.trie import Trie


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
        self.index_lookup = {}

    def get_word_counts(self, text: bytes) -> Counter:
        result = Counter()
        for i, tok in enumerate(text.split(b" ")):
            if i == 0:
                result[tok] += 1
            else:
                result[b" " + tok] += 1
        return result

    def fit(self, text: bytes):
        word_counts = self.get_word_counts(text)
        words = list(word_counts.keys())

        word_arrays = []
        # we need a linked array for every word
        for word in words:
            word_arrays.append(LinkedArray(word))

        self.trie = Trie(0)
        heap = TokenHeap()
        token_counts = Counter()
        initial_counts = {}

        # now, we need to know our starting pair counts
        seen = set()
        for w_idx, word in enumerate(words):
            for i in range(len(word)):
                pair = word[i : i + 2]
                if len(pair) != 2:
                    continue
                for j in range(2):
                    tok = pair[j : j + 1]
                    if tok not in seen:
                        seen.add(tok)
                        self.trie.insert(tok, len(self.index_lookup))
                        self.index_lookup[len(self.index_lookup)] = tok
                    token_counts[tok] += 1
                if pair not in initial_counts:
                    initial_counts[pair] = TokenData(token=pair, count=0, locs=[])
                data = initial_counts[pair]
                data.count += 1
                data.locs.append((w_idx, i))
        # Now we can build our heap
        for data in initial_counts.values():
            token_counts[data.token] += data.count
            # we have a min heap
            data.count = -data.count
            heap.insert_token(data)

        # We should be set up for the iterative stage now
        while len(self.index_lookup) < self.vocab_size:
            # pop from the heap
            # add to the trie
            # merge in the array
            # need to manage global counts so that we can compute the new counts
            # insert new counts and mark old counts as stale
            data = heap.pop()
            self.trie.insert(data.token, len(self.index_lookup))
            self.index_lookup[len(self.index_lookup)] = data.token

            # have to reorient the locs. they're (w,i) but we need {w:i}
            word_loc_map = {}
            for w_idx, s_idx in data.locs:
                if w_idx not in word_loc_map:
                    word_loc_map[w_idx] = []
                word_loc_map[w_idx].append(s_idx)

            for w_idx, locs in word_loc_map.items():
                word = words[w_idx]
                word_count = word_counts[word]
                array = word_arrays[w_idx]
                new_counts, stale_counts = array.merge_all(locs, w_idx)
                stale_pairs = []
                for pair, count in stale_counts.items():
                    if len(pair) > 1:
                        stale_pairs.append(pair)
                    # again, min heap, so our math is inverted
                    token_counts[pair] -= count
                    assert token_counts[pair] >= 0

                heap.mark_as_stale(stale_pairs)
                for data in new_counts:
                    token_counts[data.token] += data.count
                    # careful here, we need to weight by the word's freq
                    # also it's a min heap so we need to invert
                    data.count = -data.count * word_count
                    heap.insert_token(data)

    def tokenize(self, text: bytes, pbar_pos=None) -> list[int]:
        if self.trie is None:
            raise ValueError("Tokenizer must be fit to text before use")
        return self.trie.tokenize(text, pbar_pos=pbar_pos)

    def untokenize(self, toks: list[int]) -> bytes:
        if self.index_lookup is None:
            raise ValueError("Tokenizer must be fit to text before use")
        return b"".join(self.index_lookup[t] for t in toks)
