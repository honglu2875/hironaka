# hironaka.core
This is the core functionality whose classes
 - save collections of points,
 - perform transformations (Newton polytope, shift, rescale, etc.),
 - provide features, states, etc.

The base class is `PointsBase`, and the subclasses are currently `ListPoints` and `TensorPoints`. 

`NumpyPoints` is currently unnecessary and the implementation is postponed.

## .PointsBase
This interface is an abstraction of collection of points used in the Hironaka games and its variations.

A couple important notes:
 - The points are stored as 3d objects (nested lists, tensors, etc.). The 3 axis represent:
   - (batches, points, point coordinates)
 - Subclasses must define `subcls_config_keys, copied_attributes`.
   - `subcls_config_keys` defines the config keys that will be tracked and initialized when creating a copy object in `.copy()` method. In other words, they are configs that stay unchanged throughout space transformations.
   - `copied_attributes` defines the keys of the attributes that will be directly copied after initialization in `.copy()`. In other words, they may be changed during space transformations and is partially the information of the state (e.g., `.ended`).
 - Shape of points can be specified with config parameters `points_batch_size, dimension, max_num_points`. If they are not given, it will look over the point data and initialize `.batch_size, .dimension, .max_num_points` attributes.

Must implement: 
` 
_get_shape
_get_newton_polytope
_reposition
_shift
_rescale
_point_copy
_add_batch_axis
_get_batch_ended
`

Feel free to override:
`
get_features
_get_max_num_points
`

## .ListPoints
It stores the points in nested lists and perform list-based transformations. The nested lists do not have to be of uniform shape (not padded).
For example:

```
[
   [
      (7, 5, 3, 8), (8, 9, 8, 18), (8, 3, 17, 8),
      (11, 11, 1, 19), (11, 12, 18, 6), (16, 11, 5, 6)
   ],
   [
      (0, 1, 0, 1), (0, 2, 0, 0), (1, 0, 0, 1),
      (1, 0, 1, 0), (1, 1, 0, 0)
   ]
]
```

This is a batch of 2 separate sets of points. They are of different sizes.

## .TensorPoints
It performs everything using PyTorch tensors.

Major differences between `TensorPoints` and `ListPoints`
 - `TensorPoints.points` have a fixed shape. Removed points will be replaced by `padded_value` which must be a negative number.
 - In `ListPoints.points`, the order of points may change during `get_newton_polytope()`. But for `TensorPoints.points`, the order will never change. When points are removed, we still maintain the original orders without moving surviving points next to each other. As a result, `distinguished_points` marks the distinguished point in each set of points, but is never changed under any operations.
