import abc
from typing import Any, Optional, List


class PointsBase(abc.ABC):
    """
        This is the abstract interface representing points.
        Points must internally have 3 dimensions.

        The first axis is the batch axis (self.batch_size):
         - We allow the storage of multiple set of points that are independent of each other.
        The second axis is the point axis (self.max_num_points):
         - Each slice represents a point in a certain batch.
         - The maximal number of points should be capped by self.max_num_points.
         - If the implementation represents points by tensors (numpy, torch, tf, ...), when there are fewer points
                than self.max_num_points in a batch, the remaining places should be padded by -1.
        The third axis is the dimension axis (self.dim).

        The default implementation is the class 'Points' which uses nested lists, i.e.,
            self.points: List[List[List[int]]]
        But one should feel free to implement it in any other different data types.

        Must implement:
            _get_shape
            _get_newton_polytope
            _reposition
            _shift
            _rescale
            _point_copy
            _add_batch_axis
            _get_batch_ended
        Feel free to override:
            get_features
            _get_max_num_points
    """
    # You MUST define `config_keys` when inheriting.
    # Keys in `config_keys` will be tracked when calling the `copy()` method.
    config_keys: List[str]

    def __init__(self,
                 points: Any,
                 **kwargs):
        """
            Comments:
                Arguments when inheriting __init__:
                    self, <keys in config_keys>, config_kwargs=None, **kwargs
                Then please combine `config_kwargs` and `kwargs` into a dict (say, `config`), and call
                    super().__init__(points, **config)
                at the end.
        """
        if hasattr(self, 'config'):
            self.config = {**kwargs, **self.config}
        else:
            self.config = kwargs

        # Update keys if modified or created in subclass
        for key in self.config_keys:
            self.config[key] = getattr(self, key)

        # Check the shape of `points`.
        shape = self._get_shape(points)
        if len(shape) == 2:
            print("Input is required to be 3-dimensional: batch, number_of_points, coordinates.")
            print("A batch dimension is automatically added.")
            shape = (1, *shape)
            points = self._add_batch_axis(points)
        if len(shape) != 3:
            raise Exception("input dimension must be 2 or 3.")

        self.points = points

        self.batch_size = self.config.get('points_batch_size', shape[0])
        self.dimension = self.config.get('dimension', shape[2])
        self.max_num_points = self.config.get('max_number_points', self._get_max_num_points())

        # self.ended represents whether the whole game (for all batches) has ended
        # will be updated on point-changing modifications including `get_newton_polytope`
        self.ended = False

        # self.ended_each_batch represents the game status of each batch
        # will also be updated on point-changing modifications including `get_newton_polytope`
        self.ended_each_batch = [False] * self.batch_size

    def copy(self, points: Optional[Any] = None):
        """
            Copy the object.
            Parameters:
                points: the point data. If None, copy self.points.
            Returns:
                the cloned object.
        """
        if points is None:
            new_points = self._points_copy(self.points)
        else:
            new_points = self._points_copy(points)
        return self.__class__(new_points, **self.config)

    def shift(self, coords: List[List[int]], axis: List[int], inplace=True):
        """
            Shift each batch according to the list of coords and axis.
        """
        r = self._shift(self.points, coords, axis, inplace=inplace)
        if inplace:
            return None
        else:
            return self.copy(points=r)

    def reposition(self, inplace=True):
        """
            Reposition batches of points so that each batch touches all the coordinate planes.
        """
        r = self._reposition(self.points, inplace=inplace)
        if inplace:
            return None
        else:
            return self.copy(points=r)

    def get_newton_polytope(self, inplace=True):
        """
            Get the Newton Polytope for points in each batch.
        """
        r = self._get_newton_polytope(self.points, inplace=inplace)
        ended_each_batch = self._get_batch_ended(self.points)
        ended = all(ended_each_batch)

        if inplace:
            self.ended_each_batch = ended_each_batch
            self.ended = ended
            return None
        else:
            new_points = self.copy(points=r)
            new_points.ended_each_batch = ended_each_batch
            new_points.ended = ended
            return new_points

    def rescale(self, inplace=True):
        """
            Rescale each batch.
        """
        r = self._rescale(self.points, inplace=inplace)
        if inplace:
            return None
        else:
            return self.copy(points=r)

    def get_features(self):
        """
            An alias of getting the points.
            Feel free to override if you want to make some transformations and do feature engineering.
        """
        return self.points

    def get_batch(self, b: int):
        """
            Returns: the b-th batch.
            (note: it is hard-coded instead of being an abstract method because for list/numpy/torch/tf, the syntax
                are all the same: `self.points[b]`)
        """
        return self.points[b]

    @abc.abstractmethod
    def _get_shape(self, points: Any):
        """
            Get the shape of input data. E.g., if it is implemented for nested lists, just recursively return the
            lengths. If it is implemented for numpy array, just return `points.shape`.
            Parameters:
                points: the point data.
            Returns:
                Tuple
        """
        pass

    @abc.abstractmethod
    def _get_newton_polytope(self, points: Any, inplace: Optional[bool] = True):
        """
            Get the Newton polytope.
            Parameters:
                points: the point data (must be mutable).
                inplace: True, directly make modifications on the reference `points`. False, return as a new object.
            Returns:
                None if inplace==True. The new object if inplace==False.
        """
        pass

    @abc.abstractmethod
    def _shift(self,
               points: Any,
               coords: List[List[int]],
               axis: List[int],
               inplace: Optional[bool] = True):
        """
            Shift the points.
            Parameters:
                points: the point data (must be mutable).
                coords: the list of chosen coordinates.
                axis: the list of chose axis.
                inplace: True, directly make modifications on the reference `points`. False, return as a new object.
            Returns:
                None if inplace==True. The new object if inplace==False.
        """
        pass

    @abc.abstractmethod
    def _reposition(self, points: Any, inplace: Optional[bool] = True):
        """
            Reposition batches of points so that each batch touches all the coordinate planes.
            Parameters:
                points: the point data.
                inplace: True, directly make modifications on the reference `points`. False, return as a new object.
            Returns:
                None if inplace==True. The new object if inplace==False.
        """
        pass

    @abc.abstractmethod
    def _rescale(self, points: Any, inplace: Optional[bool] = True):
        """
            Rescale the points.
            Parameters:
                points: the point data (must be mutable).
                inplace: True, directly make modifications on the reference `points`. False, return as a new object.
            Returns:
                None if inplace==True. The new object if inplace==False.
        """
        pass

    @abc.abstractmethod
    def _points_copy(self, points: Any):
        """
            Make a copy of the points object.
            Parameters:
                points: the point data.
            Returns:
                The cloned object of `points`.
        """
        pass

    @abc.abstractmethod
    def _add_batch_axis(self, points: Any):
        """
            Add a batch dimension to the front (always return a new object).
            Parameters:
                points: the point data.
            Returns:
                the new point data with an additional batch axis (size=1).
        """
        pass

    @abc.abstractmethod
    def _get_batch_ended(self, points: Any) -> List[bool]:
        """
            Get a list of bool representing whether each batch of points has ended (having only 1 point left).
            Parameters:
                points: the point data.
            Returns: List[bool]
        """
        pass

    def _get_max_num_points(self) -> int:
        """
            Get the maximal number of points in all batches
            Returns: int
        """
        max_num_points = 0
        for b in range(self.batch_size):
            max_num_points = max(max_num_points, len(self.get_batch(b)))
        return max_num_points
