from typing import List, Tuple
import sympy as sp
from sympy import symbols, IndexedBase, Idx, MatrixSymbol, Matrix, Poly, Sum
import numpy as np
from itertools import chain, combinations, combinations_with_replacement




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
    W = [np.zeros((Ord,Ord))]
    B = MatrixSymbol('b',Ord,Ord)
    Blower = np.tril(Matrix(B).transpose())
    for i in range(1,Ord):
        Blower[i][0] = 0
    Blower = Matrix(Blower)
    for s in range(Ord-1):
        V = Matrix(np.zeros((Ord,Ord)))
        for t in range(s+1):
            V = V + Matrix((Blower.row(t).transpose()*Blower.row(s-t)))
        W.append(Matrix(np.triu(V+V.transpose()-Matrix(np.diag(np.array(V.diagonal())[0])))))
    return [Blower, W]

#print(QuadraticPart(3))
#print(sp.flatten([QuadraticPart(3)[1][i,i:] for i in range(QuadraticPart(3)[1].rows)]))

def QuadraticFixedPoints(Ord: int):
    """
        Get the torus fixed points...
    """
    Q = []
    P = []
    for s in range(2,Ord+1):
        for LinSet in combinations(list(range(Ord)),s):
            P = list(list(L) for L in list(combinations_with_replacement(list(LinSet), 2)) if L[0] + L[1] < Ord-1)
            if len(P) >= Ord-s:
                for Comb in list(combinations(list(P),Ord-s)):
                    Q.append([list(LinSet), list(Comb)])
    return Q

#print(QuadraticFixedPoints(3))

def ThomMonomialIdeal(Ord: int):
    """
        Get the generators of the monomial ideal of the rational map in the Thom polynomial paper
        E.g for k=3 this is a homogeneous degree 2 monomial ideal in C[t,b_22,b_23,b_33]:
        I_3=(b_22*b_33,t*b_33,t*b_23,b_22^2,t^2)
    """
    Ideal = []
    minors = [[]]
    QP = QuadraticPart(Ord)
    for Q in QuadraticFixedPoints(Ord):
        minor = [[]]
        """ Form the minor for all fixed point in the test curve model"""
        for LinColumn in Q[0]:
            minor.append(np.array(QP[0])[:,LinColumn])
        for QuadColumn in Q[1]:
            minor.append([np.array(QP[1][s])[QuadColumn[0]][QuadColumn[1]] for s in range(Ord)])
        minors.append(Matrix(minor))
    return [sp.det(Matrix(minor)) for minor in minors]

#print(ThomMonomialIdeal(3))


def ThomPoints(Ord: int):
    """
    Get the points corresponding to the monomials in the Thom monomial ideal
    """
    P = 0
    Q = ThomMonomialIdeal(Ord)
    for i in range(1,len(ThomMonomialIdeal(Ord))):
        P = P+Q[i]
    return Poly(P).monoms()

def ThomPointsHomogeneous(Ord: int):
    """
    Get the points corresponding to the monomials in the Thom monomial ideal
    """
    TP = ThomPoints(Ord)
    P = []
    MaxDeg = np.amax([np.sum(TP[i][1:]) for i in range(len(TP))])
    #DegVector = [np.sum(TP[i][1:]) for i in range(len(TP))]
    for point in TP:
        pt = np.array(point)
        pt[0] = MaxDeg-np.sum(point[1:])
        P.append(tuple(pt))
    return P


#print(ThomPoints(4))
##print(ThomPoints(4)[-1])
#print(ThomPointsHomogeneous(4)[-1])
#for k in range(2,4):
#    print(len(ThomPoints(k)[0]),ThomPoints(k))