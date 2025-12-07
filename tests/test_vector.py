import pytest
from python_geometry.vector import Vector, DimensionMismatchException


def test_vector_length():
    v = Vector(1, 2, 3)
    assert len(v) == 3


def test_vector_iteration():
    v = Vector(1, 2, 3)
    assert list(v) == [1, 2, 3]


def test_vector_indexing():
    v = Vector(10, 20, 30)
    assert v[0] == 10
    assert v[2] == 30


def test_vector_repr():
    v = Vector(1, 2)
    assert repr(v) == "Vector(1, 2)"


def test_vector_equality():
    v1 = Vector(1, 2, 3)
    v2 = Vector(1, 2, 3)
    v3 = Vector(3, 2, 1)

    assert v1 == v2
    assert v1 != v3
    assert v1 != "not a vector"


def test_vector_addition():
    v1 = Vector(1, 2, 3)
    v2 = Vector(4, 5, 6)

    result = v1 + v2
    assert result == Vector(5, 7, 9)


def test_vector_addition_dimension_mismatch():
    v1 = Vector(1, 2)
    v2 = Vector(
        1,
    )

    with pytest.raises(DimensionMismatchException):
        _ = v1 + v2


def test_vector_subtraction():
    v1 = Vector(5, 6, 7)
    v2 = Vector(1, 2, 3)

    result = v1 - v2
    assert result == Vector(4, 4, 4)


def test_vector_scalar_multiplication():
    v = Vector(1, -2, 3)
    result = v * 2
    assert result == Vector(2, -4, 6)


def test_vector_scalar_rmul():
    v = Vector(1, 2, 3)
    result = 3 * v
    assert result == Vector(3, 6, 9)
