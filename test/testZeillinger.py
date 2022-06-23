import unittest
from hironaka.host import Zeillinger
from hironaka.types import Points


class TestZeillinger(unittest.TestCase):
    def test_selectCoord(self):
        host = Zeillinger()
        points = Points([(0, 0, 4), (5, 0, 1), (1, 5, 1), (0, 25, 0)])

        assert tuple(host.selectCoord(points, debug=True)[0]) == (0, 2)

    def test_charVector(self):
        host = Zeillinger()
        points = Points([(0, 0, 4), (5, 0, 1), (1, 5, 1), (0, 25, 0)])

        assert host.getCharVector((5, 0, -3)) == (8, 2)
