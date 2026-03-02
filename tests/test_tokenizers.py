from mytorch.tokenizers import NaiveBPE


def test_naive_tokenizer():
    texts = [
        b"The dog ran.",
        b"the dog ran",
        b"do",
        b"the cat runs?",
        b"a rock didn't (does not!) run",
    ]

    singletons = set()
    for text in texts:
        for t in text:
            singletons.add(int.to_bytes(t))
    print(singletons)

    # " r" should be the most common pair
    tokenizer = NaiveBPE(len(singletons) + 1)

    tokenizer.fit(texts)
    assert len(tokenizer.index_lookup) == tokenizer.vocab_size

    assert len(tokenizer.tokenize(b"the rock")) == 7

    for t in texts:
        assert t == tokenizer.untokenize(tokenizer.tokenize(t))
