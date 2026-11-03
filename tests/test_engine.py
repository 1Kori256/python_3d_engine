from pathlib import Path
import tempfile
import unittest

import numpy as np

from engine import Camera, Mesh, clip_polygon, frustum_planes, prepare_frame, project, rotation_y

ROOT = Path(__file__).resolve().parents[1]


class MeshTests(unittest.TestCase):
    def load(self, source, normalize=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mesh.obj'
            path.write_text(source)
            return Mesh.from_obj(path, normalize=normalize)

    def test_quads_slash_indices_comments_and_relative_indices(self):
        mesh = self.load('''
            v 0 0 0
            v 1 0 0
            v 1 1 0
            v 0 1 0
            f -4/1/1 -3/2/1 -2/3/1 -1/4/1 # a quad
        ''')
        np.testing.assert_array_equal(mesh.faces, [[0, 1, 2], [0, 2, 3]])
        normal = np.cross(mesh.vertices[1] - mesh.vertices[0], mesh.vertices[2] - mesh.vertices[0])
        self.assertGreater(normal[2], 0)

    def test_invalid_obj_has_line_context(self):
        sources = ['v 1 2', 'v nan 0 0', 'v 0 0 0\nf 0 1 1',
                   'v 0 0 0\nf 1 2 3', 'v 0 0 0\nf 1 1']
        for source in sources:
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, r'mesh.obj:\d+:'):
                self.load(source)

    def test_empty_and_zero_extent_models_are_rejected(self):
        for source in ('# empty', 'v 1 1 1\nf 1 1 1'):
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.load(source, normalize=True)

    def test_bundled_models_are_centered_and_fitted(self):
        for path in (ROOT / 'objects').glob('*.obj'):
            with self.subTest(path=path.name):
                mesh = Mesh.from_obj(path)
                low, high = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
                np.testing.assert_allclose(low + high, 0, atol=1e-12)
                self.assertAlmostEqual(float(np.max(high - low)), 2.8)
                self.assertEqual(mesh.faces.shape[1], 3)


class CameraTests(unittest.TestCase):
    def test_origin_and_camera_position_transform_consistently(self):
        camera = Camera(yaw=0, pitch=0, distance=4)
        np.testing.assert_allclose(camera.world_to_view([[0, 0, 0], camera.position]),
                                   [[0, 0, 4], [0, 0, 0]], atol=1e-12)
        np.testing.assert_allclose(camera.world_to_view([[1, 1, 0]]), [[1, 1, 4]])

    def test_orbit_basis_is_orthonormal_including_poles(self):
        for yaw, pitch in ((0, 0), (0.7, 0.4), (1, np.pi / 2)):
            with self.subTest(yaw=yaw, pitch=pitch):
                camera = Camera(yaw=yaw, pitch=pitch)
                basis = camera.basis()
                np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)
                np.testing.assert_allclose(camera.world_to_view([[0, 0, 0]]),
                                           [[0, 0, camera.distance]], atol=1e-12)

    def test_rotation_uses_full_angle(self):
        np.testing.assert_allclose(rotation_y(np.pi / 2) @ [1, 0, 0], [0, 0, -1], atol=1e-12)
        np.testing.assert_allclose(rotation_y(2 * np.pi), np.eye(3), atol=1e-12)

    def test_zoom_and_pitch_stay_bounded(self):
        camera = Camera()
        camera.zoom(100)
        self.assertGreaterEqual(camera.distance, 0.4)
        camera.zoom(-100)
        self.assertLessEqual(camera.distance, 20)
        camera.orbit(0, 10000)
        self.assertLess(abs(camera.pitch), np.pi / 2)


class ProjectionTests(unittest.TestCase):
    def test_center_and_vertical_axis(self):
        screen = project(np.array([[0, 0, 4], [0, 1, 4]]), 800, 600)
        np.testing.assert_allclose(screen[0], [400, 300])
        self.assertLess(screen[1, 1], 300)

    def test_wide_viewport_does_not_stretch_geometry(self):
        points = np.array([[1, 1, 4]])
        standard = project(points, 800, 600)[0] - [400, 300]
        wide = project(points, 1600, 600)[0] - [800, 300]
        np.testing.assert_allclose(standard, wide)
        self.assertAlmostEqual(abs(standard[0]), abs(standard[1]))

    def test_points_behind_camera_are_not_projected(self):
        for z in (0, -1):
            with self.subTest(z=z), self.assertRaises(ValueError):
                project([[0, 0, z]], 800, 600)

    def test_near_plane_splits_crossing_triangle_without_discarding_it(self):
        triangle = np.array([[0, 0, 0], [0.1, 0, 1], [0, 0.1, 1]])
        polygon = clip_polygon(triangle, [[0, 0, 1, -0.1]])
        self.assertEqual(len(polygon), 4)
        self.assertTrue(np.all(polygon[:, 2] >= 0.1 - 1e-12))
        self.assertEqual(np.count_nonzero(np.isclose(polygon[:, 2], 0.1)), 2)

    def test_full_frustum_keeps_screen_coordinates_in_bounds(self):
        polygon = clip_polygon([[-10, -10, -1], [10, -10, 2], [0, 10, 2]],
                               frustum_planes(800, 600))
        self.assertGreaterEqual(len(polygon), 3)
        screen = project(polygon, 800, 600)
        self.assertTrue(np.isfinite(screen).all())
        self.assertTrue(np.all(screen >= -1e-8))
        self.assertTrue(np.all(screen <= [800 + 1e-8, 600 + 1e-8]))


class FrameTests(unittest.TestCase):
    def test_cube_culls_back_faces_and_keeps_camera_depth(self):
        mesh = Mesh.from_obj(ROOT / 'objects/cube.obj')
        frame = prepare_frame(mesh, Camera(yaw=0, pitch=0), 800, 600)
        self.assertEqual(len(frame.points), 2)
        np.testing.assert_allclose(frame.depths, (1 / frame.inverse_depths).mean(axis=1))
        self.assertTrue(np.all((frame.shades >= 0.24) & (frame.shades <= 1)))

    def test_depth_data_uses_camera_space_after_orbit(self):
        mesh = Mesh.from_obj(ROOT / 'objects/cube.obj')
        camera = Camera(yaw=1.2, pitch=0.4)
        frame = prepare_frame(mesh, camera, 800, 600)
        self.assertEqual(len(frame.points), 6)
        np.testing.assert_allclose(frame.depths, (1 / frame.inverse_depths).mean(axis=1))
        self.assertTrue(np.isfinite(frame.points).all())

    def test_lighting_follows_camera_on_opposite_sides(self):
        mesh = Mesh.from_obj(ROOT / 'objects/cube.obj')
        front = prepare_frame(mesh, Camera(yaw=0, pitch=0), 800, 600)
        back = prepare_frame(mesh, Camera(yaw=np.pi, pitch=0), 800, 600)
        self.assertAlmostEqual(float(front.colors.mean()), float(back.colors.mean()), places=8)

    def test_corner_normals_keep_cube_creases(self):
        mesh = Mesh.from_obj(ROOT / 'objects/cube.obj')
        for face, corners in zip(mesh.face_normals, mesh.corner_normals):
            np.testing.assert_allclose(corners, np.tile(face, (3, 1)), atol=1e-12)

    def test_clip_interpolates_vertex_colors(self):
        polygon = clip_polygon([[0, 0, 0, 0, 0, 0], [1, 0, 1, 100, 100, 100],
                                [0, 1, 1, 100, 100, 100]], [[0, 0, 1, -0.5]])
        crossings = polygon[np.isclose(polygon[:, 2], 0.5)]
        self.assertEqual(len(crossings), 2)
        np.testing.assert_allclose(crossings[:, 3:], 50)

    def test_zero_area_faces_and_empty_visible_set_are_safe(self):
        mesh = Mesh(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float),
                    np.array([[0, 1, 2]]), 'degenerate')
        frame = prepare_frame(mesh, Camera(), 800, 600)
        self.assertEqual(frame.points.shape, (0, 3, 2))
        self.assertEqual(len(frame.depths), 0)

    def test_extreme_zoom_is_clipped_to_viewport(self):
        mesh = Mesh.from_obj(ROOT / 'objects/sphere.obj')
        frame = prepare_frame(mesh, Camera(distance=0.4), 800, 600, cull=False)
        self.assertTrue(np.isfinite(frame.points).all())
        self.assertTrue(np.all(frame.points >= -1e-8))
        self.assertTrue(np.all(frame.points <= [800 + 1e-8, 600 + 1e-8]))


if __name__ == '__main__':
    unittest.main()
