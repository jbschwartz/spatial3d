import unittest

from spatial import Edge, Vector3


class TestEdge(unittest.TestCase):
    def setUp(self) -> None:
        self.start = Vector3(1, 2, 3)
        self.end = Vector3(-1, -2, -3)
        self.middle = Vector3(0, 0, 0)
        self.edge = Edge(self.start, self.end)

    def test__init__accepts_endpoints(self) -> None:
        self.assertEqual(self.edge.start, self.start)
        self.assertEqual(self.edge.end, self.end)

    def test__eq__returns_true_for_edges_regardless_of_direction(self) -> None:
        same_edge = Edge(self.start, self.end)
        self.assertEqual(self.edge, same_edge)

        opposite_edge = Edge(self.end, self.start)
        self.assertEqual(self.edge, opposite_edge)

        other_edge = Edge(self.start, self.middle)
        self.assertNotEqual(other_edge, self.edge)

    def test__eq__returns_notimplemented_for_incompatible_types(self) -> None:
        self.assertTrue(self.edge.__eq__(2) == NotImplemented)
        self.assertTrue(self.edge.__eq__("string") == NotImplemented)

    def test_length_returns_the_length_of_the_edge(self) -> None:
        self.assertEqual(self.edge.length, (self.start - self.end).length())

    def test_vector_returns_the_vector_between_the_edges_endpoints(self) -> None:
        self.assertEqual(self.edge.vector, self.end - self.start)
