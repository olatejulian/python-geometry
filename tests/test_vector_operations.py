import pytest
from math import isclose, pi

from python_geometry.vector import Vector, DimensionMismatchException
from python_geometry.vector_operations import (
    inner_product,
    norm,
    distance,
    angle_between,
    projection,
    normalize,
    ZeroLengthVectorException,
)


def test_inner_product_basic():
    u = Vector(1, 2, 3)
    v = Vector(4, 5, 6)

    assert inner_product(u, v) == 32


def test_inner_product_orthogonal():
    u = Vector(1, 0)
    v = Vector(0, 1)

    assert inner_product(u, v) == 0


def test_inner_product_dimension_mismatch():
    u = Vector(1, 2)
    v = Vector(
        1,
    )

    with pytest.raises(DimensionMismatchException):
        inner_product(u, v)


def test_norm_basic():
    v = Vector(3, 4)
    assert norm(v) == 5


def test_norm_zero():
    v = Vector(0, 0, 0)
    assert norm(v) == 0


def test_distance_basic():
    u = Vector(1, 2)
    v = Vector(4, 6)

    assert distance(u, v) == 5


def test_angle_parallel_vectors():
    u = Vector(1, 0)
    v = Vector(3, 0)

    assert isclose(angle_between(u, v), 0.0)


def test_angle_orthogonal_vectors():
    u = Vector(1, 0)
    v = Vector(0, 1)

    assert isclose(angle_between(u, v), pi / 2)


def test_angle_opposite_vectors():
    u = Vector(1, 0)
    v = Vector(-1, 0)

    assert isclose(angle_between(u, v), pi)


def test_angle_with_zero_vector():
    u = Vector(0, 0)
    v = Vector(1, 0)

    with pytest.raises(ZeroLengthVectorException):
        angle_between(u, v)


def test_projection_basic():
    u = Vector(2, 2)
    v = Vector(1, 0)

    proj = projection(u, v)
    assert proj == Vector(2, 0)


def test_projection_zero_vector():
    u = Vector(1, 2)
    v = Vector(0, 0)

    with pytest.raises(ZeroLengthVectorException):
        projection(u, v)


def test_normalize_basic():
    v = Vector(3, 4)
    v_hat = normalize(v)

    assert isclose(norm(v_hat), 1.0)


def test_normalize_zero_vector():
    v = Vector(0, 0)

    with pytest.raises(ZeroLengthVectorException):
        normalize(v)
