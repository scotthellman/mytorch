from mytorch.tokenizers import BPE, NaiveBPE


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


def test_bpe():
    # broadly it would be nice to test this at vocab_size == len(text)
    # but the tokenizer fails at that point and fixing it would take nontrivial work
    # (and exhausting the text will never happen in a realistic scenario)
    tokenizer = BPE(vocab_size=3)

    text = b"aaaa"

    tokenizer.fit(text)

    result = tokenizer.tokenize(text)
    print(result)
    assert len(result) == 1


def test_bpe_nontrivial():
    # broadly it would be nice to test this at vocab_size == len(text)
    # but the tokenizer fails at that point and fixing it would take nontrivial work
    # (and exhausting the text will never happen in a realistic scenario)
    text = b"this is a piece of text, that contains MANY characters, etc."
    n_unique = len(set(text))
    tokenizer = BPE(vocab_size=n_unique + 15)

    tokenizer.fit(text)

    result = tokenizer.tokenize(text)

    rebuilt = tokenizer.untokenize(result)

    assert text == rebuilt


def test_bpe_known_failure():
    # broadly it would be nice to test this at vocab_size == len(text)
    # but the tokenizer fails at that point and fixing it would take nontrivial work
    # (and exhausting the text will never happen in a realistic scenario)
    text = b"is is is "
    n_unique = len(set(text))
    tokenizer = BPE(vocab_size=4)

    tokenizer.fit(text)

    result = tokenizer.tokenize(text)

    rebuilt = tokenizer.untokenize(result)

    assert text == rebuilt


def test_bpe_repetitive():
    # broadly it would be nice to test this at vocab_size == len(text)
    # but the tokenizer fails at that point and fixing it would take nontrivial work
    # (and exhausting the text will never happen in a realistic scenario)
    text = b"ccabcabc" * 2
    n_unique = len(set(text))
    tokenizer = BPE(vocab_size=n_unique + 5)

    tokenizer.fit(text)

    result = tokenizer.tokenize(text)

    rebuilt = tokenizer.untokenize(result)

    assert text == rebuilt


def test_bpe_competing_matches():
    # broadly it would be nice to test this at vocab_size == len(text)
    # but the tokenizer fails at that point and fixing it would take nontrivial work
    # (and exhausting the text will never happen in a realistic scenario)
    text = b"thethe"
    n_unique = len(set(text))
    tokenizer = BPE(vocab_size=n_unique + 3)

    tokenizer.fit(text)

    result = tokenizer.tokenize(text)

    rebuilt = tokenizer.untokenize(result)

    assert text == rebuilt
