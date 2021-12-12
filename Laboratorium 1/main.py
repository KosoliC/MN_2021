from math import pi
import numpy as np
from typing import List

def cylinder_area(r: float, h: float):
    """Obliczenie pola powierzchni walca. 
    Szczegółowy opis w zadaniu 1.

    Parameters:
    r (float): promień podstawy walca 
    h (float): wysokosć walca

    Returns:
    float: pole powierzchni walca 
    """

    if (r > 0) and (h > 0):
        area = 2 * pi * r * r + 2 * pi * r * h
        return area
    else:
        return np.NaN


def fib(n: int):
    """Obliczenie pierwszych n wyrazów ciągu Fibonnaciego. 
    Szczegółowy opis w zadaniu 3.

    Parameters:
    n (int): liczba określająca ilość wyrazów ciągu do obliczenia 

    Returns:
    np.ndarray: wektor n pierwszych wyrazów ciągu Fibonnaciego.
    """

    fib_list = np.array([1, 1])

    if n <= 0:
        return None

    if isinstance(n, int):

        if n == 1:
            return np.array([1])
        elif n == 2:
            return fib_list
        for i in range(2, n):
            next_elem = fib_list[-1] + fib_list[-2]
            fib_list = np.append(fib_list, [next_elem])

        return np.reshape(fib_list, (1, n))

    else:
        return None


def matrix_calculations(a: float):
    """Funkcja zwraca wartości obliczeń na macierzy stworzonej 
    na podstawie parametru a.  
    Szczegółowy opis w zadaniu 4.

    Parameters:
    a (float): wartość liczbowa 

    Returns:
    touple: krotka zawierająca wyniki obliczeń 
    (Minv, Mt, Mdet) - opis parametrów w zadaniu 4.
    """

    matrix = np.array([[a, 1, -a], [0, 1, 1], [-a, a, 1]])
    Mdet = np.linalg.det(matrix)
    if Mdet == 0:
        Minv = np.NaN
    else:
        Minv = np.linalg.inv(matrix)
    Mt = np.transpose(matrix)

    return Minv, Mt, Mdet


def custom_matrix(m: int, n: int) -> List[List[int]]:
    """Funkcja zwraca macierz o wymiarze mxn zgodnie 
    z opisem zadania 7.  

        Parameters:
    m (int): ilość wierszy macierzy
    n (int): ilość kolumn macierzy  

    Returns:
    np.ndarray: macierz zgodna z opisem z zadania 7.
    """

    if m < 0 or n < 0:
        return None

    if isinstance(m, int) and isinstance(n, int):
        matrix = np.zeros((m, n), dtype=int)
        for i in range(m):
            for j in range(n):
                if i > j:
                    matrix[i][j] = i
                else:
                    matrix[i][j] = j

        return matrix

    else:
        return None

