from hironaka.core import ListPoints


class Points:
    """
        A simple wrapper allowing for simple usage of basic functions of `ListPoints`
    """
    points_config = {
        'use_precise_newton_polytope': True,
        'value_threshold': 1e8
    }

    def __init__(self, points, distinguished_point=None):
        distinguished_point = [distinguished_point] if distinguished_point is not None else None
        self.points = ListPoints([points], distinguished_points=distinguished_point, **self.points_config)
        self.dimension = self.points.dimension

        self.points.get_newton_polytope()
        self.track_dist_point = distinguished_point is not None

    def step(self, host_coordinates, agent_coordinate) -> bool:
        if self.points.ended or (self.track_dist_point and self.points.distinguished_points[0] is None):
            return False
        self.points.shift([host_coordinates], [agent_coordinate])
        self.points.get_newton_polytope()
        return True

    def __repr__(self):
        return str(self.points.points[0])

    @property
    def ended(self):
        return self.points.ended
