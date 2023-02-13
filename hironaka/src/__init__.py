from ._borrowed_fn import polyak_update, zip_equal, zip_longest
from ._fn import (
    Experience,
    HostActionEncoder,
    batched_coord_list_to_binary,
    coord_list_to_binary,
    decode_action,
    generate_batch_points,
    generate_points,
    get_batched_padded_array,
    get_gym_version_in_float,
    get_padded_array,
    get_python_version_in_float,
    get_shape,
    lst_cpy,
    make_nested_list,
    mask_encoded_action,
    merge_experiences,
    remove_repeated,
    scale_points,
)
from ._jax_ops import (
    calculate_rescale,
    get_equal,
    get_interior,
    get_newton_polytope_approx_jax,
    get_newton_polytope_jax,
    is_repeated,
    remove_repeated_jax,
    reposition_jax,
    rescale_jax,
    shift_jax,
    shift_single_batch,
    subtract_min,
)
from ._list_ops import get_newton_polytope_approx_lst, get_newton_polytope_lst, get_shape, reposition_lst, shift_lst
from ._thom_fn import quadratic_fixed_points, quadratic_part, thom_monomial_ideal, thom_points, thom_points_homogeneous
from ._torch_ops import (
    batched_coord_list_to_binary,
    get_newton_polytope_approx_torch,
    get_newton_polytope_torch,
    remove_repeated,
    reposition_torch,
    rescale_torch,
    shift_torch,
)

from_borrowed_fn = ["zip_longest", "zip_equal", "polyak_update"]
from_fn = [
    "get_shape",
    "make_nested_list",
    "lst_cpy",
    "get_padded_array",
    "get_batched_padded_array",
    "coord_list_to_binary",
    "batched_coord_list_to_binary",
    "get_gym_version_in_float",
    "get_python_version_in_float",
    "scale_points",
    "decode_action",
    "mask_encoded_action",
    "generate_points",
    "generate_batch_points",
    "remove_repeated",
    "Experience",
    "merge_experiences",
    "HostActionEncoder",
]
from_jax_ops = [
    "get_equal",
    "is_repeated",
    "remove_repeated_jax",
    "get_interior",
    "get_newton_polytope_approx_jax",
    "get_newton_polytope_jax",
    "shift_single_batch",
    "shift_jax",
    "calculate_rescale",
    "rescale_jax",
    "subtract_min",
    "reposition_jax",
]
from_list_ops = ["get_shape", "get_newton_polytope_approx_lst", "get_newton_polytope_lst", "shift_lst", "reposition_lst"]
from_thom_fn = ["quadratic_part", "quadratic_fixed_points", "thom_monomial_ideal", "thom_points", "thom_points_homogeneous"]
from_torch_ops = [
    "batched_coord_list_to_binary",
    "remove_repeated",
    "get_newton_polytope_approx_torch",
    "get_newton_polytope_torch",
    "shift_torch",
    "reposition_torch",
    "rescale_torch",
]

__all__ = from_borrowed_fn + from_fn + from_jax_ops + from_list_ops + from_thom_fn + from_torch_ops
