from typing import Iterable


class MatrixDimensionMismatchException(Exception):
    pass


class Matrix:
    def __init__(self, components: Iterable[Iterable[float]]):
        rows = [list(row) for row in components]

        if not rows:
            raise ValueError

        num_cols = len(rows[0])
        if num_cols == 0:
            raise ValueError

        for row in rows:
            if len(row) != num_cols:
                raise MatrixDimensionMismatchException

        self.__components = tuple(tuple(row) for row in rows)
        self.__num_rows = len(rows)
        self.__num_cols = num_cols

    def __iter__(self) -> Iterable[float]:
        for row in self.__components:
            for value in row:
                yield value

    def __getitem__(self, index: tuple[int, int]) -> float:
        row, col = index
        return self.__components[row][col]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return NotImplemented
        return self.__components == other.__components

    def __add__(self, other: "Matrix") -> "Matrix":
        if not self.__check_dimension_match(other):
            raise MatrixDimensionMismatchException

        return Matrix(
            [
                [self[i, j] + other[i, j] for j in range(self.__num_cols)]
                for i in range(self.__num_rows)
            ]
        )

    def __sub__(self, other: "Matrix") -> "Matrix":
        if not self.__check_dimension_match(other):
            raise MatrixDimensionMismatchException

        return Matrix(
            [
                [self[i, j] - other[i, j] for j in range(self.__num_cols)]
                for i in range(self.__num_rows)
            ]
        )

    def __mul__(self, scalar: float) -> "Matrix":
        return Matrix(
            [
                [self[i, j] * scalar for j in range(self.__num_cols)]
                for i in range(self.__num_rows)
            ]
        )

    def __rmul__(self, scalar: float) -> "Matrix":
        return self.__mul__(scalar)

    @property
    def shape(self) -> tuple[int, int]:
        return self.__num_rows, self.__num_cols

    def __check_dimension_match(self, other: "Matrix") -> bool:
        return self.shape == other.shape
