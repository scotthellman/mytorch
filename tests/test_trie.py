from mytorch.trie import Trie


def test_trie():
    root = Trie(0)

    insertions = {b"foo": 1, b"bar": 2, b"baz": 3, b"ba": 4, b"food": 5}

    for key, val in insertions.items():
        root.insert(key, val)

    for key, val in insertions.items():
        assert root[key] == val


def test_trie_traversal():
    root = Trie(0)

    insertions = {b"foo": 1, b"bar": 2, b"baz": 3, b"ba": 4, b"food": 5}

    for key, val in insertions.items():
        root.insert(key, val)

    result, leftover = root.traversal_value(b"back")
    assert result == insertions[b"ba"]
    assert leftover == b"ck"


def test_trie_tokenize():
    root = Trie(0)

    insertions = {b"a": 1, b"b": 2, b"ab": 3, b"ba": 4, b"c": 5}
    for key, val in insertions.items():
        root.insert(key, val)

    test_string = b"abbacaa"
    expected = [3, 4, 5, 1, 1]

    actual = root.tokenize(test_string)
    assert expected == actual


def test_trie_missing_interior():
    root = Trie(0)

    insertions = {b"a": 1, b"b": 2, b"abb": 3, b"aba": 4}
    for key, val in insertions.items():
        root.insert(key, val)

    test_string = b"ab"
    expected = [1, 2]

    actual = root.tokenize(test_string)
    assert expected == actual
