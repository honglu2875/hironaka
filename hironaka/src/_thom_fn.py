from itertools import combinations, combinations_with_replacement

import numpy as np
import sympy as sp
from sympy import Matrix, MatrixSymbol, Poly


def quadratic_part(order_of_jets: int):
    """
    Get the quadratic form q(v)=v_1v_{k-1}+v_2v_{k-2}+...+v_{k-1}v_1 for v_i in C^k in a matrix form:
    the (i,j) entry of the upper triangular kxk matrix quadratic_part(k) is the coefficient of e-ie_j in q(v).
    """

    W = [Matrix(np.zeros((order_of_jets, order_of_jets)))]
    B = MatrixSymbol('b', order_of_jets, order_of_jets)
    blower = np.tril(Matrix(B).transpose())
    for i in range(1, order_of_jets):
        blower[i][0] = 0
    blower = Matrix(blower)
    for s in range(order_of_jets - 1):
        V = Matrix(np.zeros((order_of_jets, order_of_jets)))
        for t in range(s + 1):
            V = V + Matrix((blower.row(t).transpose() * blower.row(s - t)))
        W.append(Matrix(np.triu(V + V.transpose() - Matrix(np.diag(np.array(V.diagonal())[0])))))
    return [blower, W]


def quadratic_fixed_points(order_of_jets: int):
    """
        Get the torus fixed points...
    """
    Q = []
    P = []
    for s in range(2, order_of_jets + 1):
        for lin_set in combinations(list(range(order_of_jets)), s):
            P = list(list(L) for L in list(combinations_with_replacement(list(range(order_of_jets)), 2)) if
                     L[0] + L[1] < order_of_jets - 1)
            if len(P) >= order_of_jets - s:
                for comb in list(combinations(list(P), order_of_jets - s)):
                    Q.append([list(lin_set), list(comb)])
    return Q


def thom_monomial_ideal(order_of_jets: int):
    """
        Get the generators of the monomial ideal of the rational map in the Thom polynomial paper
        E.g for k=3 this is a homogeneous degree 2 monomial ideal in C[t,b_22,b_23,b_33]:
        I_3=(b_22*b_33,t*b_33,t*b_23,b_22^2,t^2)
    """
    minors = []
    QP = quadratic_part(order_of_jets)
    for Q in quadratic_fixed_points(order_of_jets):
        minor = []
        """ Form the minor for all fixed points in the test curve model"""
        for lin_column in Q[0]:
            minor.append(list(QP[0][:, lin_column]))
        for quad_column in Q[1]:
            minor.append([QP[1][s][quad_column[0], quad_column[1]] for s in range(order_of_jets)])
        minors.append(Matrix(np.array(minor)))
    return [sp.det(Matrix(minor)) for minor in minors]


def thom_points(order_of_jets: int):
    """
    Get the points corresponding to the monomials in the Thom monomial ideal
    """
    Q = thom_monomial_ideal(order_of_jets)
    if len(Q) == 0:
        return [(0,)]
    else:
        P = Q[0]  # QUESTION: Really starts at 1?????
        for i in range(0, len(Q)):
            P = P + Q[i]
    return Poly(P).monoms()


def thom_points_homogeneous(order_of_jets: int):
    """
    Get the points corresponding to the monomials in the Thom monomial ideal
    """
    TP = thom_points(order_of_jets)
    P = []
    max_deg = np.amax([np.sum(TP[i][1:]) for i in range(len(TP))])
    for point in TP:
        pt = np.array(point)
        pt[0] = max_deg - np.sum(point[1:])
        P.append(pt.tolist())
    return P
