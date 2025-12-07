from math import acos, isclose
from functools import reduce
from .vector import Vector, DimensionMismatchException


class ZeroLengthVectorException(Exception):
    pass


def inner_product(u: Vector, v: Vector) -> float:
    if len(u) != len(v):
        raise DimensionMismatchException

    return reduce(
        lambda acc, pair: acc + pair[0] * pair[1],
        zip(u, v),
        0.0,
    )


def norm(v: Vector) -> float:
    return inner_product(v, v) ** 0.5


def distance(u: Vector, v: Vector) -> float:
    return norm(u - v)


def angle_between(u: Vector, v: Vector) -> float:
    if len(u) != len(v):
        raise DimensionMismatchException

    dot_product = inner_product(u, v)

    norm_u = norm(u)
    norm_v = norm(v)

    if isclose(norm_u, 0.0) or isclose(norm_v, 0.0):
        raise ZeroLengthVectorException

    cos_theta = dot_product / (norm_u * norm_v)
    cos_theta = max(-1.0, min(1.0, cos_theta))

    return acos(cos_theta)


def projection(u: Vector, v: Vector) -> Vector:
    norm_v_squared = inner_product(v, v)

    if isclose(norm_v_squared, 0.0):
        raise ZeroLengthVectorException

    scalar_projection = inner_product(u, v) / norm_v_squared

    return v * scalar_projection


def normalize(v: Vector) -> Vector:
    norm_v = norm(v)

    if isclose(norm_v, 0.0):
        raise ZeroLengthVectorException

    return v * (1 / norm_v)
