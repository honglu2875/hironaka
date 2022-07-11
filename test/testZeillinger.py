import unittest

from hironaka.abs import Points
from hironaka.host import Zeillinger
from hironaka.src import make_nested_list


class TestZeillinger(unittest.TestCase):
    def test_select_coord(self):
        host = Zeillinger()
        points = Points(make_nested_list([(0, 0, 4), (5, 0, 1), (1, 5, 1), (0, 25, 0)]))

        assert tuple(host.select_coord(points, debug=True)[0]) == (0, 2)

    def test_char_vector(self):
        host = Zeillinger()
        points = Points(make_nested_list([(0, 0, 4), (5, 0, 1), (1, 5, 1), (0, 25, 0)]))

        assert host.get_char_vector((5, 0, -3)) == (8, 2)
