import numpy as np


def get_shape(o):
    """
        o is supposed to be a nested object consisting of list and tuple.
        It will recursively search for o[0][0]... until it cannot proceed. output len(..) at each level.
            - Example: o = [[1,2,3],[2,3,4]], it will return (2,3)

        If the nested list/tuple objects are not of uniform shape, this function becomes pointless.
        Therefore, being uniform is an assumption before using this snippet.
            - Anti-example: o = [ ([1,2,3],2,3),(2,3,4) ], it will return (2,3,3).
        It's intuitively wrong but this function is not responsible for checking the uniformity.

        Also, if it hits a length-0 object, it will just stop.
    """

    unwrapped = o
    shape = []
    while isinstance(unwrapped, (list, tuple)) and unwrapped:
        shape.append(len(unwrapped))
        unwrapped = unwrapped[0]

    return tuple(shape)


def make_nested_list(o):
    """
        This will make a nested list-like object a nested list.
        It operates in a recursive fashion, and we do not wish to use it in standard class operations.
        ***It's only for testing and scripting purposes.***

        (comment: __dir__() is super super slow!)
    """
    if '__len__' not in o.__dir__() or len(o) == 0:
        return o

    return [make_nested_list(i) for i in o]


def lst_cpy(dest, orig):
    """
        This copies the content of orig to dest. Both need to be list-like and mutable.
        Furthermore, we assume len(dest)>=len(orig).
    """
    for i in range(len(orig)):
        dest[i] = orig[i]
    diff = len(dest) - len(orig)
    for i in range(diff):
        dest.pop()
