from typing import List, Tuple
import sympy as sp
from sympy import symbols, IndexedBase, Idx, MatrixSymbol, Matrix, Poly, Sum
import numpy as np
from itertools import chain, combinations




def getNewtonPolytope_approx(points: List[Tuple[int]]):
    """
        A simple-minded quick-and-dirty method to obtain an approximation of Newton Polytope disregarding convexity.
    """
    assert points

    if len(points) == 1:
        return points
    DIM = len(points[0])

    points = sorted(points)
    result = []
    for i in range(len(points)):
        contained = False
        for j in range(i):
            if sum([points[j][k] > points[i][k] for k in range(DIM)]) == 0:
                contained = True
                break
        if not contained:
            result.append(points[i])
    return result


def getNewtonPolytope(points: List[Tuple[int]]):
    """
        Get the Newton Polytope for a set of points.
    """
    return getNewtonPolytope_approx(points)  # TODO: change to a more precise algo to obtain Newton Polytope


def shift(points: List[Tuple[int]], coords: List[int], axis: int):
    """
        Shift a set of points according to the rule of Hironaka game.
    """
    assert axis in coords
    assert points

    if len(points) == 1:
        return points
    DIM = len(points[0])

    return [tuple([
        sum([x[k] for k in coords]) if i == axis else x[i]
        for i in range(DIM)])
        for x in points]


def QuadraticPart(Ord: int):
    """Get the quadratic form q(v)=v_1v_{k-1}+v_2v_{k-2}+...+v_{k-1}v_1 for v_i in C^k in a matrix form: the (i,j) entry of
       the upper triangular kxk matrix QuadraticPart(k) is the coefficient of e-ie_j in q(v).
    """

    i = Idx('i', Ord)
    j = Idx('j', Ord)
    b = IndexedBase('b')
    s = symbols('s', integer=True)
    V = sp.zeros(Ord,Ord)
    D = []
    B = MatrixSymbol('b',Ord,Ord)
    Blower = Matrix(np.tril(Matrix(B).transpose()))
    for s in range(1,Ord):
        V = V + Blower.row(s-1).transpose()*Blower.row(Ord-s-1)
    return Matrix(np.triu(V+V.transpose()-Matrix(np.diag(np.array(V.diagonal())[0]))))

print(QuadraticPart(4))
print(sp.flatten([QuadraticPart(4)[i,i:] for i in range(QuadraticPart(4).rows)]))

def QuadraticFixedPoints(Ord: int):
    """
        Get the torus fixed points...
    """
    Q = []
    for s in range(2,Ord+1):
        for LinSet in combinations(list(range(Ord)),s):
            print(combinations(list(LinSet),2))
            Q.append([list(LinSet), (L for L in combinations(list(LinSet), 2) if L[0] + L[1] < Ord)])
    return Q

print(QuadraticFixedPoints(3))


def ThomMonomialIdeal(Ord: int):
    """
        Get the monomial ideal of the rational map in the Thom polynomial paper
        E.g for k=3 this is a homogeneous degree 2 monomial ideal in C[t,b_22,b_23,b_33]:
        I_3=(b_22*b_33,t*b_33,t*b_23,b_22^2,t^2)
    """




#P = Poly(np.array(Blower.row(2).transpose()*Blower.row(1))[0][0]+b[1,2])
#print(P,P.monoms())
