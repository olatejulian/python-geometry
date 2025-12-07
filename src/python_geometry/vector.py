from typing import Iterable


class DimensionMismatchException(Exception):
    pass


class Vector:
    def __init__(self, *components: float):
        self.__components = components

    def __len__(self) -> int:
        return len(self.__components)

    def __iter__(self) -> Iterable[float]:
        for component in self.__components:
            yield component

    def __getitem__(self, index: int) -> float:
        return self.__components[index]

    def __add__(self, other: Vector) -> Vector:
        if len(self.__components) != len(other):
            raise DimensionMismatchException

        dimension = len(self)

        return Vector.from_iterable(map(lambda i: self[i] + other[i], range(dimension)))

    def __sub__(self, other: Vector) -> Vector:
        if len(self.__components) != len(other.__components):
            raise DimensionMismatchException

        dimension = len(self)

        return Vector.from_iterable(map(lambda i: self[i] - other[i], range(dimension)))

    def __mul__(self, scalar: float) -> Vector:
        return Vector.from_iterable(
            map(lambda component: component * scalar, self.__components)
        )

    def __rmul__(self, scalar: float) -> Vector:
        return self.__mul__(scalar)

    @staticmethod
    def from_iterable(iterable: Iterable[float]) -> Vector:
        return Vector(*iterable)

    def to_tuple(self) -> tuple[float, ...]:
        return tuple(self.__components)

    def to_list(self) -> list[float]:
        return list(self.__components)
