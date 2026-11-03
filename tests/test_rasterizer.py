import unittest

import numpy as np

from rasterizer import antialias_edges, draw_edges, rasterize


class RasterizerTests(unittest.TestCase):
    def render(self, points, inverse_depths, colors):
        pixels = np.zeros((16, 16, 3), dtype=np.uint8)
        depth = np.zeros((16, 16), dtype=np.float32)
        rasterize(np.asarray(points, dtype=float), np.asarray(inverse_depths, dtype=float),
                  np.asarray(colors, dtype=float), pixels, depth)
        return pixels, depth

    def test_near_triangle_occludes_far_triangle_in_either_draw_order(self):
        triangle = [[1, 1], [14, 1], [1, 14]]
        points = np.array([triangle, triangle], dtype=float)
        depths = np.array([[1, 1, 1], [0.5, 0.5, 0.5]])
        colors = np.array([[[255, 0, 0]] * 3, [[0, 0, 255]] * 3])
        forward, _ = self.render(points, depths, colors)
        reverse, _ = self.render(points[::-1], depths[::-1], colors[::-1])
        np.testing.assert_array_equal(forward, reverse)
        np.testing.assert_array_equal(forward[3, 3], [255, 0, 0])

    def test_crossing_triangles_change_visibility_per_pixel(self):
        triangle = [[1, 1], [14, 1], [1, 14]]
        points = np.array([triangle, triangle], dtype=float)
        depths = np.array([[1, 0.25, 1], [0.3, 0.9, 0.3]])
        colors = np.array([[[255, 0, 0]] * 3, [[0, 0, 255]] * 3])
        forward, _ = self.render(points, depths, colors)
        reverse, _ = self.render(points[::-1], depths[::-1], colors[::-1])
        np.testing.assert_array_equal(forward, reverse)
        np.testing.assert_array_equal(forward[2, 2], [255, 0, 0])
        np.testing.assert_array_equal(forward[2, 10], [0, 0, 255])

    def test_vertex_colors_are_perspective_correct(self):
        pixels, _ = self.render([[[0, 0], [8, 0], [0, 8]]], [[1, 0.5, 0.25]],
                                [[[255, 0, 0], [0, 255, 0], [0, 0, 255]]])
        np.testing.assert_array_equal(pixels[1, 1], [208, 31, 16])

    def test_both_windings_cover_the_same_pixels(self):
        points = np.array([[[1, 1], [14, 1], [1, 14]]], dtype=float)
        colors = np.array([[[255, 20, 10], [20, 255, 10], [10, 20, 255]]], dtype=float)
        forward, _ = self.render(points, [[1, 1, 1]], colors)
        reverse, _ = self.render(points[:, ::-1], [[1, 1, 1]], colors[:, ::-1])
        np.testing.assert_array_equal(forward.any(axis=2), reverse.any(axis=2))
        # Floating-point ties can round an interpolated channel one byte apart.
        np.testing.assert_allclose(forward, reverse, atol=1, rtol=0)

    def test_degenerate_and_offscreen_triangles_are_safe(self):
        for points in ([[[2, 2], [2, 2], [2, 2]]], [[[30, 30], [40, 30], [30, 40]]]):
            with self.subTest(points=points):
                pixels, depth = self.render(points, [[1, 1, 1]], [[[255, 255, 255]] * 3])
                self.assertFalse(pixels.any())
                self.assertFalse(depth.any())

    def test_hidden_edges_do_not_draw_over_nearer_surface(self):
        points = np.array([[[1, 1], [14, 1], [1, 14]]], dtype=float)
        pixels = np.zeros((16, 16, 3), dtype=np.uint8)
        depth = np.full((16, 16), 2.0, dtype=np.float32)
        draw_edges(points, np.array([[1., 1., 1.]]), pixels, depth)
        self.assertFalse(pixels.any())

    def test_edge_filter_keeps_uniform_interiors_unchanged(self):
        pixels = np.full((16, 16, 3), 128, dtype=np.uint8)
        output = np.zeros_like(pixels)
        depth = np.ones((16, 16), dtype=np.float32)
        antialias_edges(pixels, depth, output, (0, 0, 16, 16))
        np.testing.assert_array_equal(output, pixels)


if __name__ == '__main__':
    unittest.main()
