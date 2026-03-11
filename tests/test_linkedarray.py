from mytorch.tokenizers.linkedarray import LinkedArray


def test_linked_array():
    test = "abc".encode()
    arr = LinkedArray(test)

    for i in range(len(test)):
        node = arr[i]
        assert node is not None
        assert node.value == test[i : i + 1]

    arr.merge(0)

    assert arr[0].value == test[:2]
    assert arr[0].next_node.index == 2
    assert arr[1] is None
    assert arr[2].value == test[2:]
    assert arr[2].prev_node.index == 0


def test_linked_array_merge_all():
    test = "is is is".encode()
    arr = LinkedArray(test)
    # let's say we're merging "is"
    indices = [0, 3, 6]
    new_counts, stale_counts = arr.merge_all(indices, 0)

    expected_new_counts = {b"is ": 2, b" is": 2}
    expected_stale = {b"s ": 2, b" i": 2, "s": 3, "i": 3}

    assert len(new_counts) == len(expected_new_counts)
    for data in new_counts:
        assert data.count == expected_new_counts[data.token]

    assert len(stale_counts) == len(expected_stale)
    for value, count in expected_stale.items():
        assert expected_stale[value] == count


def test_linked_array_repeated_merging_correctness():
    test = "the specifics don't matter here, just need some text".encode()
    arr = LinkedArray(test)
    indices = [0, 3, 6]
    arr.merge_all(indices, 0)
    indices = [6, 13, 16]
    arr.merge_all(indices, 0)

    node = arr[0]
    reconstructed = [node.value]
    next_node = node.next_node
    while next_node:
        reconstructed.append(next_node.value)
        next_node = next_node.next_node
    rejoined = b"".join(reconstructed)
    assert rejoined == test
