import abc
import random
import logging
from itertools import combinations
from typing import Optional, Union

import numpy as np
from collections import defaultdict

from hironaka.core import ListPoints
from hironaka.points import Points
from hironaka.policy.policy import Policy


class Host(abc.ABC):
    """
    A host that returns the subset of coordinates according to the given set of points.
    Must implement:
        _select_coord
    """

    def __init__(self, ignore_batch_dimension=False, **kwargs):
        self.logger = logging.getLogger(__class__.__name__)

        # If the agent only has one batch and wants to ignore batch dimension in the parameters, set it to True.
        self.ignore_batch_dimension = ignore_batch_dimension

    def select_coord(self, points: Union[ListPoints, Points], debug=False):
        if isinstance(points, Points):
            points = points.points

        if self.ignore_batch_dimension:
            return self._select_coord(points)[0]
        else:
            return self._select_coord(points)

    @abc.abstractmethod
    def _select_coord(self, points: ListPoints):
        pass


class RandomHost(Host):
    def _select_coord(self, points: ListPoints):
        dim = points.dimension
        return [np.random.choice(list(range(dim)), size=2, replace=False).tolist() for _ in range(points.batch_size)]


class AllCoordHost(Host):
    def _select_coord(self, points: ListPoints):
        dim = points.dimension
        return [list(range(dim)) for _ in range(points.batch_size)]


class Zeillinger(Host):
    # noinspection PyPep8Naming
    @staticmethod
    def get_char_vector(vt):
        """
        Character vector (L, S),
            L: maximum coordinate - minimum coordinate
            S: sum of the numbers of maximum coordinates and minimum coordinates
        e.g., (1, 1, -1, -1) -> (L=2, S=4)
        """
        mx = max(vt)
        mn = min(vt)
        L = mx - mn
        S = sum([vt[i] == mx for i in range(len(vt))]) + sum([vt[i] == mn for i in range(len(vt))])
        return L, S

    def _select_coord(self, points: ListPoints):
        assert not points.ended

        dim = points.dimension
        result = []
        for b in range(points.batch_size):
            pts = points[b]
            if len(pts) <= 1:
                result.append([])
                continue
            pairs = combinations(pts, 2)
            char_vectors = []
            for pair in pairs:
                vector = tuple([pair[0][i] - pair[1][i] for i in range(dim)])
                char_vectors.append((vector, self.get_char_vector(vector)))
            char_vectors.sort(key=(lambda x: x[1]))

            result.append(self._get_coord(char_vectors))
        return result

    def _get_coord(self, char_vectors):
        r = [np.argmin(char_vectors[0][0]), np.argmax(char_vectors[0][0])]
        if r[0] != r[1]:
            return r
        else:  # if all coordinates are the same, return the first two.
            return [0, 1]


class PolicyHost(Host):
    def __init__(self, policy: Policy, use_discrete_actions_for_host: Optional[bool] = False, **kwargs):
        self._policy = policy
        self.use_discrete_actions_for_host = kwargs.get("use_discrete_actions_for_host", use_discrete_actions_for_host)

        super().__init__(**kwargs)

    def _select_coord(self, points: ListPoints):
        features = points.get_features()

        # calling `predict` to return multi-binary array
        coords = self._policy.predict(features)
        result = []
        for b in range(coords.shape[0]):
            result.append(np.where(coords[b] == 1)[0].tolist())
        return result


class ZeillingerLex(Zeillinger):
    def _get_coord(self, char_vectors):  # TODO: efficiency can be improved
        coords = []
        for char_vector in char_vectors:
            if char_vector[1] == char_vectors[0][1]:
                r = [np.argmin(char_vector[0]), np.argmax(char_vector[0])]
                if r[0] != r[1]:
                    coords.append(r)
                else:  # if all coordinates are the same, return the first two.
                    coords.append([0, 1])
        coords.sort()
        return coords[0]


class Spivakovsky(Host):
    def __init__(self, dim=16):
        super().__init__()
        self.dim = dim
        return

    def _parallel_move(self, points):
        # Move each batch of points so that for each coordinate there is a point with 0 in this coordinate.
        moved_points = []
        pts = points
        dim = len(pts[0])
        min_length = [-1 for _ in range(dim)]
        for i in range(dim):
            for pt in pts:
                if min_length[i] == -1 or pt[i] < min_length[i]:
                    min_length[i] = pt[i]

        points_in_this_batch = []
        for pt in pts:
            new_pt = []
            for i in range(dim):
                new_pt.append(pt[i] - min_length[i])

            points_in_this_batch.append(new_pt)

        return points_in_this_batch

    def _pick_coords(self, points):
        # Choose coordinates from points, also tell us whether the process is over.
        result = []
        pts = points
        dim = len(pts[0])
        S = set()
        minimal_distance = float('inf')
        for point in pts:
            sum_of_coords = sum(point)
            if sum_of_coords < minimal_distance:
                minimal_distance = sum_of_coords
                S = set()

            if sum_of_coords == minimal_distance:
                for coord, num in enumerate(point):
                    if num > 0:
                        S = S | {coord}

        I = set(range(dim)) - S

        return (S, I, minimal_distance)

    def _contains_zero(self, pts):
        dim = len(pts[0])
        zero_point = [0.0 for i in range(dim)]
        if zero_point in pts:
            return True
        return False

    def _last_step(self, pts):
        # The last step of Spivakovsky, find a minimal permissible set.
        # The commented version is the minimal hitting set.
        dim = len(pts[0])
        total_coords = list(range(dim))
        for i in range(1, dim):
            comb = combinations(total_coords, i)
            for coords in comb:
                list_coords = list(coords)

                # Verify if list_coords is a hitting set for all non-zero points
                # contained = [0 for _ in pts]
                # for i,pt in enumerate(pts):
                #     if sum(pt) < 0.0001:
                #         contained[i] = 1
                #         continue
                #     for coord in list_coords:
                #         if pt[coord] > 0:
                #             contained[i] = 1
                #
                # if sum(contained) == len(pts):
                #     return set(coords)

                # # Verify if it is permissible.
                if_permissible = True
                for pt in pts:
                    sum = 0
                    for coord in coords:
                        sum += pt[coord]
                    if sum < 0.999:
                        if_permissible = False

                if if_permissible:
                    return set(coords)

        return set()

    def _select_coord(self, points: ListPoints):
        # Select coordinates based on Spivakovsky's method
        result = []
        for b in range(points.batch_size):
            result_this_batch = set()
            pts = points[b]
            dim = len(pts[0])
            coord_dict = list(range(dim))

            print('Points are', pts)
            print('Number of points is', len(pts))

            while pts and (not self._contains_zero(self._parallel_move(pts))):
                S, I, minimal_distance = self._pick_coords(self._parallel_move(pts))
                for coord in S:
                    result_this_batch = result_this_batch | {coord_dict[coord]}

                if I:
                    I = list(I)
                    I.sort()
                    new_dict = []
                    for coord in I:
                        new_dict.append(coord_dict[coord])
                    coord_dict = new_dict

                moved_pts = self._parallel_move(pts)
                new_pts = [[coord / minimal_distance for coord in pt] for pt in moved_pts]
                project_pts = []
                # Choose points to project to the smaller dimension indexed by I
                # This process might cause repetition, but it is ok.
                for pt in new_pts:
                    sum = 0.0
                    for coord in S:
                        sum += pt[coord]

                    if sum < 0.999:
                        project_pts.append(pt)

                # for pt in pts:
                #     sum = 0.0
                #     for coord in S:
                #         sum += pt[coord]
                #
                #     if sum < 0.999:
                #         project_pts.append(pt)
                # Projection
                low_dim_pts = []

                for pt in project_pts:
                    scale = 0
                    for coord in S:
                        scale += pt[coord]
                    scale = 1 - scale
                    new_pt = [pt[i] / scale for i in I]
                    low_dim_pts.append(new_pt)

                pts = low_dim_pts
            # If we get a non-empty set, then we need to find a minimal permissible set, not a hitting set.
            if pts:
                # print('At the final step, current points are:', pts)
                final_set = self._last_step(pts)
                for coord in final_set:
                    result_this_batch = result_this_batch | {coord_dict[coord]}
            # print('Points in the last step are:', pts)
            print('Current coordinate selection is', result_this_batch)

            result.append(list(result_this_batch))
        return result


class RandomSpivakovsky(Host):
    def __init__(self, dim=16):
        super().__init__()
        subset_route = []
        for i in range(1, pow(2, dim)):
            if (i & (i - 1)):
                subset_route.append((bin(i).count("1"), i))

        subset_route.sort()

        self.subset_route = subset_route

        return

    def _select_coord(self, points: ListPoints):
        assert not points.ended
        result = []
        for b in range(points.batch_size):
            pts = points[b]
            if len(pts) <= 1:
                result.append([])
                continue

            point_hash = defaultdict()
            for point in pts:
                this_hash = 0
                place = 1
                for i in range(len(point)):
                    if point[i] != 0:
                        this_hash += place
                    place *= 2

                point_hash[this_hash] = 1

            subsets = point_hash.keys()
            # Find the smallest hitting set.
            answers = []
            bitcount = -1
            for i in self.subset_route:
                hit = True
                for subset in subsets:
                    if not (i[1] & subset):
                        hit = False
                        break
                if hit:
                    if (bitcount > 0) and (bitcount < i[0]):
                        break
                    bitcount = i[0]
                    answers.append(i[1])

            ans = answers[random.randint(0, len(answers) - 1)]
            # Decode the hitting set from int to a set.
            batch_result = []
            counter = 0
            while ans:
                if ans % 2:
                    batch_result.append(counter)
                ans = ans // 2
                counter += 1
            result.append(batch_result)

        return result


class WeakSpivakovsky(Host):
    def _select_coord(self, points: ListPoints):
        assert not points.ended
        result = []
        for b in range(points.batch_size):
            pts = points[b]
            if len(pts) <= 1:
                result.append([])
                continue
            # For each point we store the subset of nonzero coordinates
            subsets = [set(np.nonzero(point)[0]) for point in pts]
            # Find a minimal hitting set, brute-force
            U = set.union(*subsets)
            for i in range(2, len(U) + 1):
                combs = combinations(U, i)
                for c in combs:
                    if all(set(c) & coord for coord in subsets):
                        result.append(list(c))
                        break
                if len(result) > b:  # Found result for this batch. Break.
                    break
        return result


class WeakSpivakovskyRandom(Host):
    def __init__(self, dim=16):
        super().__init__()
        subset_route = []
        for i in range(1, pow(2, dim)):
            if (i & (i - 1)):
                subset_route.append((bin(i).count("1"), i))

        subset_route.sort()

        self.subset_route = subset_route

        return

    def _select_coord(self, points: ListPoints):
        # todo: tune this host to be randomly choosing smallest hitting set.
        assert not points.ended
        result = []
        for b in range(points.batch_size):
            pts = points[b]
            if len(pts) <= 1:
                result.append([])
                continue

            point_hash = defaultdict()
            for point in pts:
                this_hash = 0
                place = 1
                for i in range(len(point)):
                    if point[i] != 0:
                        this_hash += place
                    place *= 2

                point_hash[this_hash] = 1

            subsets = point_hash.keys()
            # Find the smallest hitting set.
            ans = 65536
            for i in self.subset_route:
                hit = True
                for subset in subsets:
                    if not (i[1] & subset):
                        hit = False
                        break
                if hit:
                    ans = i[1]
                    break
            # Decode the hitting set from int to a set.
            batch_result = []
            counter = 0
            while ans:
                if ans % 2:
                    batch_result.append(counter)
                ans = ans // 2
                counter += 1
            result.append(batch_result)

        return result

        #     # For each point we store the subset of nonzero coordinates
        #     subsets = [set(np.nonzero(point)[0]) for point in pts]
        #     # Find a minimal hitting set, brute-force
        #     U = set.union(*subsets)
        #     for i in range(2, len(U) + 1):
        #         combs = combinations(U, i)
        #         for c in combs:
        #             if all(set(c) & l for l in subsets):
        #                 result.append(list(c))
        #                 break
        #         if len(result) > b:  # Found result for this batch. Break.
        #             break
        # print(result)
        # return result


if __name__ == '__main__':
    host = Spivakovsky()
    host2 = WeakSpivakovsky(dim=4)
    points = ListPoints([[[2, 0, 0, 0, 0, 0, 0], [1, 3, 0, 0, 1, 1, 0], [1, 2, 0, 0, 1, 1, 1], [0, 5, 1, 0, 2, 1, 0]]])
    print(host._select_coord(points))
