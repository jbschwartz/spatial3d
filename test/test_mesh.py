import math
import unittest
from unittest import mock

from spatial import AABB, Facet, KDTreeNode, Mesh, Transform, Vector3


class TestMesh(unittest.TestCase):
    def setUp(self) -> None:
        aabb = AABB([Vector3(-1, -1, -1), Vector3(1, 1, 1)])
        self.vertices = aabb.corners

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
        self.mesh = Mesh("Square", self.facets)
        self.accelerator = mock.Mock()

    def test__init__defaults_to_an_empty_mesh_with_no_accelerator(self) -> None:
        m = Mesh()

        self.assertEqual(len(m.facets), 0)
        self.assertEqual(m.aabb.min, Vector3(math.inf, math.inf, math.inf))
        self.assertIsNone(self.mesh.accelerator)

    def test__init__computes_the_aabb_of_the_facets(self) -> None:
        self.assertEqual(self.mesh.aabb.min, self.vertices[4])
        self.assertEqual(self.mesh.aabb.max, self.vertices[0])

    def test_from_file_returns_a_list_of_meshes_with_an_accelerator(self) -> None:
        mock_parser = mock.Mock()
        mock_parser.parse.return_value = [Mesh(), Mesh()]
        file_name = "test_file_name.stl"

        results = Mesh.from_file(mock_parser, file_name)

        mock_parser.parse.assert_called_once_with(file_name)

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], Mesh)
        self.assertIsInstance(results[0].accelerator, KDTreeNode)

    def test_accelerator_setter_initializes_accelerator(self) -> None:
        self.mesh.accelerator = self.accelerator
        self.accelerator.assert_called_once_with(self.mesh.aabb, self.mesh.facets)

    def test_append_adds_a_facet_to_the_mesh(self) -> None:
        num_facets = len(self.mesh.facets)
        self.mesh.append(Facet([2 * Vector3.X(), 2 * Vector3.Y(), 2 * Vector3.Z()]))
        self.assertEqual(len(self.mesh.facets), num_facets + 1)

    def test_append_updates_the_aabb_and_accelerator(self) -> None:
        self.mesh.accelerator = KDTreeNode
        original_accelerator = self.mesh.accelerator

        self.mesh.append(Facet([2 * Vector3.X(), 2 * Vector3.Y(), 2 * Vector3.Z()]))

        self.assertEqual(self.mesh.aabb.max, Vector3(2, 2, 2))
        self.assertIsNot(self.mesh.accelerator, original_accelerator)

    def test_intersect_brute_forces_an_intersection_against_all_facets(self) -> None:
        ray = mock.Mock()
        self.mesh.intersect(ray)

        ray.closest_intersection.assert_called_once_with(self.mesh.facets)

    def test_intersect_calls_the_accelerator_intersection_if_one_exists(self) -> None:
        self.mesh.accelerator = self.accelerator

        ray = mock.Mock()
        self.mesh.intersect(ray)

        ray.closest_intersection.assert_not_called()
        self.mesh.accelerator.intersect.assert_called_once_with(ray)

    def test_scale_returns_a_scaled_mesh(self) -> None:
        scale = 3
        scaled = self.mesh.scale(scale)

        self.assertEqual(scaled.aabb.min, scale * self.mesh.aabb.min)
        self.assertEqual(scaled.aabb.max, scale * self.mesh.aabb.max)
        self.assertEqual(scaled.facets[0].vertices[1], scale * self.mesh.facets[0].vertices[1])

    def test_transform_returns_a_transformed_mesh(self) -> None:
        t = Transform.from_axis_angle_translation(translation=Vector3(1, 2, 3))

        transformed = self.mesh.transform(t)

        self.assertEqual(transformed.aabb.min, t.transform(self.mesh.aabb.min))
        self.assertEqual(transformed.aabb.max, t.transform(self.mesh.aabb.max))
        self.assertEqual(
            transformed.facets[0].vertices[1], t.transform(self.mesh.facets[0].vertices[1])
        )

    def test_vertices_returns_the_list_of_vertices_in_the_mesh(self) -> None:
        self.assertEqual(len(list(self.mesh.vertices)), 3 * len(self.mesh.facets))
        for vertex in self.vertices:
            self.assertIsInstance(vertex, Vector3)
