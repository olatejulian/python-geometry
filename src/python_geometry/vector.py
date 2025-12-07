from typing import Iterator


class DimensionMismatchException(Exception):
    pass


class Vector:
    def __init__(self, *components: float):
        self.__components = components

    def __len__(self) -> int:
        return len(self.__components)

    def __iter__(self) -> Iterator[float]:
        return iter(self.__components)

    def __getitem__(self, index: int) -> float:
        return self.__components[index]

    def __repr__(self) -> str:
        return f"Vector{self.__components}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented

        if len(self) != len(other):
            return False

        return all(self[i] == other[i] for i in range(len(self)))

    def __add__(self, other: Vector) -> Vector:
        if len(self) != len(other):
            raise DimensionMismatchException

        dimension = len(self)

        return Vector(*map(lambda i: self[i] + other[i], range(dimension)))

    def __sub__(self, other: Vector) -> Vector:
        if len(self) != len(other):
            raise DimensionMismatchException

        dimension = len(self)

        return Vector(*map(lambda i: self[i] - other[i], range(dimension)))

    def __mul__(self, scalar: float) -> Vector:
        return Vector(*map(lambda component: component * scalar, self.__components))

    def __rmul__(self, scalar: float) -> Vector:
        return self.__mul__(scalar)
