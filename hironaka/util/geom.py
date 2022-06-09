def getNewtonPolytope_approx(points: List[Tuple[int]], DIM=4):
    """
        A simple-minded quick-and-dirty method to obtain an approximation of Newton Polytope disregarding convexity.
    """
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
