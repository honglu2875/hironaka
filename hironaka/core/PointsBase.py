import abc
import logging
from copy import deepcopy
from typing import Any, List, Optional, Tuple, Union


class PointsBase(abc.ABC):
    """
    This is the abstract interface representing points.
    Points must internally have 3 dimensions with shape (self.batch_size, <=self.max_num_points, self.dimension).

    The first axis is the batch axis (self.batch_size):
     - We allow the storage of multiple set of points that are independent of each other.
    The second axis is the point axis (self.max_num_points):
     - Each item represents a point in a certain batch.
     - The maximal number of points should be capped by self.max_num_points.
     - If the implementation represents points by tensors (numpy, torch, tf, ...), when there are fewer points
            than self.max_num_points in a batch, the remaining places should be padded by negative numbers (such as -1).
    The third axis is the dimension axis (self.dimension).

    Must implement:
        _get_shape
        _get_newton_polytope
        _reposition
        _shift
        _rescale
        _get_batch_ended
    Feel free to override/implement:
        get_features  # Output customized forms of the points (feature engineering for ML training, etc.).
        _get_max_num_points  # Recommended to override if points are not lists for better performances.
        _add_batch_axis  # If not implemented, the input cannot ignore batch dimension.
        _points_copy  # Override if the deepcopy process is different from a simple call of `deepcopy()`.

    Caution: Put `super().__init__(points, **kwargs)` after where `subcls_config_keys` and `running_attributes` are set.
    """

    # Keys in `base_attributes` will be copied. But they are shared in all subclasses and DO NOT need to re-initialize.
    base_config_keys = ["max_num_points"]

    # You MUST define `subcls_config_keys` when subclassing (will be used when initializing a new object in `copy()`).
    subcls_config_keys: List[str]
    # Keys in `running_attributes` change throughout the life-cycle of the object. Will also be directly copied.
    running_attributes: List[str]

    def __init__(self, points: Any, **kwargs):
        """
        Parameters:
            max_num_points: (Optional) maximum number of points in a set of points.
        """
        self.logger = logging.getLogger(__class__.__name__)

        self._check_class_attributes()

        # Check the shape of `points`.
        # Note: `_get_shape` does not necessarily check the validity of the whole array/tensor.
        #   E.g., in the case of nested-lists, it may just recursively get the size of the first item.
        self.points = points
        shape = self._check_points_shape()

        self.batch_size, _, self.dimension = shape
        self.max_num_points = self._get_max_num_points()
        if "max_num_points" in kwargs:
            if kwargs["max_num_points"] < self.max_num_points:
                self.logger.warning("Specified max_num_points is smaller than the one in input. Ignored.")
            else:
                self.max_num_points = kwargs["max_num_points"]

        self.config = {}  # Initialize the collection of parameters used to construct copies of this object.
        # Update keys in `self.copied_attributes`
        for key in self.subcls_config_keys + self.base_config_keys:
            if hasattr(self, key):
                self.config[key] = getattr(self, key)
            else:
                raise Exception("Must initialize keys in 'subcls_config_keys' before calling super().__init__.")

    def copy(self, points=None) -> "PointsBase":
        """
        Copy the object.
        """
        points_input = self._points_copy(self.points) if points is None else points
        new_points = self.__class__(points_input, **self.config)

        for key in self.running_attributes:
            if hasattr(self, key):
                setattr(new_points, key, deepcopy(getattr(self, key)))
            else:
                raise Exception(f"Attribute {key} is not initialized.")
        return new_points

    def shift(self, coords: List[List[int]], axis: List[int], inplace=True, **kwargs) -> Union["PointsBase", None]:
        """
        Shift each batch according to the list of coords and axis.
        """
        r = self._shift(self.points, coords, axis, inplace=inplace, **kwargs)
        if inplace:
            return None
        else:
            return self.copy(points=r)

    @property
    def ended(self) -> bool:
        # self.ended represents whether the whole game (for all batches) has ended
        return all(self._get_batch_ended(self.points))

    @property
    def ended_batch(self) -> Any:
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

    def __getitem__(self, item: int) -> List[List[Any]]:
        """
        Returns: the `item`-th batch.
        (note: it is not an abstract method because for list/numpy/torch/tf, the syntax
        are all the same: `self.points[item]`)
        """
        return self.points[item]

    @staticmethod
    def _points_copy(points):
        return deepcopy(points)

    @abc.abstractmethod
    def _get_shape(self, points: Any):
        """
        Get the shape of input data. E.g., if it is implemented for nested lists, just recursively return the
        lengths. If it is implemented for numpy array, just return `points.shape`.
        Parameters:
            points: the point data.
        Returns:
            Tuple of integers
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
    def _shift(self, points: Any, coords: List[List[int]], axis: List[int], inplace: Optional[bool] = True, **kwargs):
        """
        Shift the points.
        Parameters:
            points: the point data (must be mutable).
            coords: the list of chosen coordinates.
            axis: the list of chose axis.
            inplace: True, directly make modifications on the reference `points`. False, return as a new object.
        Returns:
            None if inplace==True. New point data if inplace==False.
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

    def _add_batch_axis(self, points: Any):
        """
        Add a batch dimension to the front (always return a new object).
        Parameters:
            points: the point data.
        Returns:
            the new point data with an additional batch axis (size=1).
        """
        raise NotImplementedError

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
            max_num_points = max(max_num_points, len(self[b]))
        return max_num_points

    def _check_class_attributes(self):
        """
        Check the mandatory class attributes (e.g., `subcls_config_keys`) are initialized.
        """
        mandatory_keys = ["subcls_config_keys", "running_attributes"]
        for key in mandatory_keys:
            if not hasattr(self, key):
                raise NotImplementedError(f"{key} must be initialized when subclassing.")

    def _check_points_shape(self) -> Tuple[int, int, int]:
        shape = self._get_shape(self.points)

        if len(shape) == 2:
            try:
                self.points = self._add_batch_axis(self.points)
                self.logger.warning(
                    "Points are 3-dimensional: batch, max_num_points, coordinates. "
                    "A batch dimension is automatically added."
                )
                shape = (1, *shape)
            except NotImplementedError:
                raise ValueError("Points must be 3-dimensional: batch, max_num_points, coordinates.")

        if len(shape) != 3:
            raise ValueError("Input dimension must be 2 or 3.")

        return shape
