from mytorch.linkedarray import LinkedArray


def test_linked_array():
    test = "abc".encode()
    arr = LinkedArray(test)

    for i in range(len(test)):
        node = arr[i]
        assert node is not None
        assert node.value == test[i : i + 1]

    arr.merge(0)

    assert arr[0].value == test[:2]
    assert arr[1] is None
    assert arr[2].value == test[2:]
