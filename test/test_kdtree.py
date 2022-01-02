import unittest

from spatial import AABB, CoordinateAxes, Facet, KDTreeNode, Ray, Vector3


class TestKDTree(unittest.TestCase):
    def setUp(self) -> None:
        self.aabb = AABB([Vector3(-1, -1, -1), Vector3(1, 1, 1)])
        self.vertices = self.aabb.corners

        self.facets = [
            # Top facets
            Facet([self.vertices[0], self.vertices[1], self.vertices[2]]),
            Facet([self.vertices[2], self.vertices[3], self.vertices[0]]),
            # Right facets
            Facet([self.vertices[0], self.vertices[3], self.vertices[5]]),
            Facet([self.vertices[5], self.vertices[6], self.vertices[0]]),
            # Left facets
            Facet([self.vertices[1], self.vertices[7], self.vertices[4]]),
            Facet([self.vertices[1], self.vertices[4], self.vertices[2]]),
            # Bottom facets
            Facet([self.vertices[4], self.vertices[6], self.vertices[5]]),
            Facet([self.vertices[4], self.vertices[7], self.vertices[6]]),
        ]

        self.empty_node = KDTreeNode(AABB(), [])
        self.root_node = KDTreeNode(self.aabb, self.facets)

    def test__init__creates_a_node_with_no_children(self) -> None:
        self.assertEqual(len(self.root_node.children), 0)

    def test_is_leaf_returns_true_if_there_are_no_children(self) -> None:
        self.assertTrue(self.empty_node.is_leaf)

        self.empty_node.children.append(KDTreeNode(AABB(), []))
        self.assertFalse(self.empty_node.is_leaf)

    def test_branch_splits_the_node_into_two_children_and_has_no_facets(self) -> None:
        self.root_node.branch()
        self.assertEqual(len(self.root_node.children), 2)
        self.assertEqual(len(self.root_node.facets), 0)

    def test_branch_does_not_split_if_there_are_no_facets(self) -> None:
        self.empty_node.branch()
        self.assertEqual(len(self.empty_node.children), 0)

    def test_can_branch_returns_true_if_the_node_can_split(self) -> None:
        self.assertTrue(self.root_node.can_branch(0))

    def test_can_branch_returns_false_if_there_are_no_facets(self) -> None:
        self.assertFalse(self.empty_node.can_branch(0))

    def test_can_branch_returns_false_if_the_depth_limit_is_exceeded(self) -> None:
        self.assertFalse(self.root_node.can_branch(1000))

    def test_intersect_returns_the_intersection_of_a_ray_with_the_triangles(self) -> None:
        self.root_node.branch()

        miss = self.root_node.intersect(Ray(Vector3(-0.5, 0.5, 3), Vector3.Z()))
        self.assertFalse(miss.hit)

        hit = self.root_node.intersect(Ray(Vector3(-0.5, 0.5, 3), -Vector3.Z()))
        self.assertTrue(hit.hit)

    def test_split_facets_partitions_facets_into_left_and_right(self) -> None:
        left, right = self.root_node.split_facets(CoordinateAxes.X, 0)

        # Facet indices expected in the left and right splits.
        left_expected = [0, 1, 4, 5, 6, 7]
        right_expected = [0, 1, 2, 3, 6, 7]

        self.assertEqual(len(left), len(left_expected))
        for index in left_expected:
            self.assertTrue(self.facets[index] in left)

        self.assertEqual(len(right), len(right_expected))
        for index in right_expected:
            self.assertTrue(self.facets[index] in right)

    def test_splitting_plane_returns_the_splitting_plane_at_a_given_depth(self) -> None:
        for depth in range(3):
            plane_axis, plane_value = self.root_node.splitting_plane(depth)
            self.assertEqual(plane_axis, CoordinateAxes(depth))
            self.assertEqual(plane_value, self.aabb.center[depth])
