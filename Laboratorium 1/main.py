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

    import math
    if (r > 0) and (h > 0):
        area = 2 * pi * r * r + 2 * pi * r * h
        return area
    return None


def fib(n: int):
    """Obliczenie pierwszych n wyrazów ciągu Fibonnaciego. 
    Szczegółowy opis w zadaniu 3.

    Parameters:
    n (int): liczba określająca ilość wyrazów ciągu do obliczenia 

    Returns:
    np.ndarray: wektor n pierwszych wyrazów ciągu Fibonnaciego.
    """

    if n < 1:
        return None
    vect = np.ndarray(shape=(1, n))
    vect[0][0] = 1
    if n == 1:
        return vect[0]
    else:
        vect[0][1] = 1
    for x in range(2, n):
        vect[0][x] = (vect[0][x - 2] + vect[0][x - 1])
    return vect


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

    M = np.array([[a, 1, -a], [0, 1, 1], [-a, a, 1]])

    Mt = np.transpose(M)
    Mdet = np.linalg.det(M)

    if Mdet == 0:
        return (None, Mt, Mdet)
    else:
        Minv = np.invert(M)
        return (Minv, Mt, Mdet)


def custom_matrix(m: int, n: int) -> List[List[int]]:
    """Funkcja zwraca macierz o wymiarze mxn zgodnie 
    z opisem zadania 7.  

        Parameters:
    m (int): ilość wierszy macierzy
    n (int): ilość kolumn macierzy  

    Returns:
    np.ndarray: macierz zgodna z opisem z zadania 7.
    """

    if n > 0 and m > 0:
        result = np.ndarray([m, n])
        for row, col in result:
            for elem, _ in col:
                if row > col:
                    result[row][col] = row
                else:
                    result[row][col] = col
        return result
    else:
        return None

print("Macierz wynikowa o wymiarach {0}x{1}: \n{2}".format(4, 5, custom_matrix(4,5)))
