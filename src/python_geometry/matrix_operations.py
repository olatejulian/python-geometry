from copy import deepcopy
from math import isclose
from typing import List, Tuple

from .vector import Vector
from .matrix import Matrix


class NonSquareMatrixException(Exception):
    pass


class MatrixDimensionMismatchException(Exception):
    pass


class SingularMatrixException(Exception):
    pass


EPS = 1e-12


def transpose(matrix: Matrix) -> Matrix:
    rows, cols = matrix.shape
    components = [[matrix[j, i] for j in range(rows)] for i in range(cols)]
    return Matrix(components)


def matrix_multiplication(a: Matrix, b: Matrix) -> Matrix:
    if a.shape[1] != b.shape[0]:
        raise MatrixDimensionMismatchException

    rows_a, cols_a = a.shape
    _, cols_b = b.shape

    result_components = [
        [sum(a[i, k] * b[k, j] for k in range(cols_a)) for j in range(cols_b)]
        for i in range(rows_a)
    ]

    return Matrix(result_components)


def _to_float_matrix(matrix: Matrix) -> List[List[float]]:
    n, m = matrix.shape
    return [[float(matrix[i, j]) for j in range(m)] for i in range(n)]


def determinant(matrix: Matrix) -> float:
    n, m = matrix.shape
    if n != m:
        raise NonSquareMatrixException

    # Use Gaussian elimination with partial pivoting for O(n^3)
    A = _to_float_matrix(matrix)
    det_sign = 1.0
    det = 1.0
    for k in range(n):
        # partial pivot
        pivot_row = max(range(k, n), key=lambda i: abs(A[i][k]))
        if abs(A[pivot_row][k]) < EPS:
            return 0.0
        if pivot_row != k:
            A[k], A[pivot_row] = A[pivot_row], A[k]
            det_sign *= -1.0
        pivot = A[k][k]
        det *= pivot
        # eliminate below
        for i in range(k + 1, n):
            factor = A[i][k] / pivot
            for j in range(k + 1, n):
                A[i][j] -= factor * A[k][j]
            A[i][k] = 0.0
    return det_sign * det


def trace(matrix: Matrix) -> float:
    n, m = matrix.shape
    if n != m:
        raise NonSquareMatrixException
    return sum(float(matrix[i, i]) for i in range(n))


def identity(size: int) -> Matrix:
    return Matrix([[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)])


def linear_transform(matrix: Matrix, vector: Vector) -> Vector:
    if matrix.shape[1] != len(vector):
        raise MatrixDimensionMismatchException

    result = [
        sum(matrix[i, j] * vector[j] for j in range(matrix.shape[1]))
        for i in range(matrix.shape[0])
    ]
    return Vector(*result)


def cofactor(matrix: Matrix) -> Matrix:
    n, m = matrix.shape
    if n != m:
        raise NonSquareMatrixException

    cof = []
    for i in range(n):
        row = []
        for j in range(n):
            # build minor
            minor = Matrix(
                [[matrix[r, c] for c in range(n) if c != j] for r in range(n) if r != i]
            )
            row.append(((-1) ** (i + j)) * determinant(minor))
        cof.append(row)
    return Matrix(cof)


def adjugate(matrix: Matrix) -> Matrix:
    return transpose(cofactor(matrix))


def inverse(matrix: Matrix) -> Matrix:
    """
    Compute inverse with Gauss-Jordan elimination (partial pivoting).
    More stable and O(n^3) vs adjugate/determinant which is slower and numerically bad.
    """
    n, m = matrix.shape
    if n != m:
        raise NonSquareMatrixException

    A = _to_float_matrix(matrix)
    aug = [row + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < EPS:
            raise SingularMatrixException
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot_val = aug[col][col]
        aug[col] = [val / pivot_val for val in aug[col]]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if abs(factor) < EPS:
                continue
            aug[r] = [aug[r][c] - factor * aug[col][c] for c in range(2 * n)]

    inv_components = [row[n:] for row in aug]
    return Matrix(inv_components)


def is_symmetric(matrix: Matrix) -> bool:
    n, m = matrix.shape
    if n != m:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if not isclose(matrix[i, j], matrix[j, i], abs_tol=1e-9):
                return False
    return True


def is_orthogonal(matrix: Matrix) -> bool:
    n, m = matrix.shape
    if n != m:
        return False
    transposed = transpose(matrix)
    product = matrix_multiplication(transposed, matrix)  # A^T A
    idn = identity(n)
    for i in range(n):
        for j in range(n):
            if not isclose(product[i, j], idn[i, j], abs_tol=1e-9):
                return False
    return True


def rank(matrix: Matrix) -> int:
    A = _to_float_matrix(matrix)
    n_rows = len(A)
    n_cols = len(A[0]) if n_rows > 0 else 0
    row = 0
    for col in range(n_cols):
        if row >= n_rows:
            break
        sel = max(range(row, n_rows), key=lambda r: abs(A[r][col]))
        if abs(A[sel][col]) < EPS:
            continue
        if sel != row:
            A[row], A[sel] = A[sel], A[row]
        pivot = A[row][col]
        A[row] = [val / pivot for val in A[row]]
        for r in range(row + 1, n_rows):
            factor = A[r][col]
            if abs(factor) < EPS:
                continue
            A[r] = [A[r][c] - factor * A[row][c] for c in range(n_cols)]
        row += 1
    return row
