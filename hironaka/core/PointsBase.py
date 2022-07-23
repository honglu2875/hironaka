import abc
from copy import deepcopy
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
    subcls_config_keys: List[str]
    # Keys in `copied_attributes` will be directly copied during `copy()`. They MUST be initialized.
    copied_attributes: List[str]
    # Keys in `base_attributes` will be copied. But they are shared in all subclasses and do not need to re-initialize.
    base_attributes = ['batch_size', 'max_num_points', 'dimension']

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

        # Check the shape of `points`.
        shape = self._get_shape(points)
        if len(shape) == 2:
            print("Input is required to be 3-dimensional: batch, max_num_points, coordinates.")
            print("A batch dimension is automatically added.")
            shape = (1, *shape)
            points = self._add_batch_axis(points)
        if len(shape) != 3:
            raise Exception("input dimension must be 2 or 3.")
        self.points = points

        self.batch_size = self.config.get('points_batch_size', shape[0])
        self.dimension = self.config.get('dimension', shape[2])
        self.max_num_points = self.config.get('max_num_points', self._get_max_num_points())

        # Update keys in `self.copied_attributes`
        for key in self.copied_attributes:
            if hasattr(self, key):
                self.config[key] = getattr(self, key)
            else:
                raise Exception("Must initialize keys in 'subcls_config_keys' before calling super().__init__.")

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
        new_points = self.__class__(new_points, **self.config)

        for key in self.copied_attributes + self.base_attributes:
            if hasattr(self, key):
                setattr(new_points, key, deepcopy(getattr(self, key)))
            else:
                raise Exception(f"Attribute {key} is not initialized.")
        return new_points

    def shift(self, coords: List[List[int]], axis: List[int], inplace=True, **kwargs):
        """
            Shift each batch according to the list of coords and axis.
        """
        r = self._shift(self.points, coords, axis, inplace=inplace, **kwargs)
        if inplace:
            return None
        else:
            return self.copy(points=r)

    @property
    def ended(self):
        # self.ended represents whether the whole game (for all batches) has ended
        return all(self._get_batch_ended(self.points))

    @property
    def ended_batch(self):
        # self.ended_each_batch represents the game status of each batch
        return self._get_batch_ended(self.points)

    def reposition(self, inplace=True, **kwargs):
        """
            Reposition batches of points so that each batch touches all the coordinate planes.
        """
        r = self._reposition(self.points, inplace=inplace, **kwargs)
        if inplace:
            return None
        else:
            return self.copy(points=r)

    def get_newton_polytope(self, inplace=True, **kwargs):
        """
            Get the Newton Polytope for points in each batch.
        """
        r = self._get_newton_polytope(self.points, inplace=inplace, **kwargs)

        if inplace:
            return None
        else:
            new_points = self.copy(points=r)
            return new_points

    def rescale(self, inplace=True, **kwargs):
        """
            Rescale each batch.
        """
        r = self._rescale(self.points, inplace=inplace, **kwargs)
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
    def _get_newton_polytope(self, points: Any, inplace: Optional[bool] = True, **kwargs):
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
               inplace: Optional[bool] = True,
               **kwargs):
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
    def _reposition(self, points: Any, inplace: Optional[bool] = True, **kwargs):
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
    def _rescale(self, points: Any, inplace: Optional[bool] = True, **kwargs):
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
