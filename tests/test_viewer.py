import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import numpy as np
import pygame

from main import ROOT, SIZE, Viewer, main


class ViewerTests(unittest.TestCase):
    def setUp(self):
        pygame.display.init()
        pygame.font.init()
        self.viewer = Viewer(pygame.Surface(SIZE))

    def tearDown(self):
        pygame.quit()

    def key(self, key):
        return self.viewer.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key))

    def test_elapsed_time_pause_and_drag_control_animation(self):
        self.viewer.update(1)
        self.assertAlmostEqual(self.viewer.angle, 0.35)
        self.key(pygame.K_SPACE)
        self.viewer.update(1)
        self.assertAlmostEqual(self.viewer.angle, 0.35)
        self.key(pygame.K_SPACE)
        self.viewer.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                                  pos=(300, 300), button=1))
        before = self.viewer.camera.position.copy()
        self.viewer.handle_event(pygame.event.Event(pygame.MOUSEMOTION, rel=(30, 10)))
        self.assertFalse(np.allclose(before, self.viewer.camera.position))
        self.viewer.update(1)
        self.assertAlmostEqual(self.viewer.angle, 0.35)
        self.viewer.handle_event(pygame.event.Event(pygame.WINDOWFOCUSLOST))
        self.assertFalse(self.viewer.dragging)

    def test_every_mesh_and_display_mode_renders(self):
        for _ in self.viewer.models:
            for mode in range(3):
                with self.subTest(mesh=self.viewer.mesh.name, mode=mode):
                    self.viewer.mode = mode
                    frame = self.viewer.render()
                    self.assertTrue(np.isfinite(frame.points).all())
                    self.assertGreater(len(frame.points), 0)
            self.viewer.next_model()

    def test_render_draws_each_triangle_once(self):
        with patch('pygame.draw.polygon', wraps=pygame.draw.polygon) as draw:
            frame = self.viewer.render()
        self.assertEqual(draw.call_count, len(frame.points))
        self.assertGreater(draw.call_count, 0)

    def test_toolbar_mode_pause_model_and_reset(self):
        def click(name):
            self.viewer.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                                      pos=self.viewer.buttons[name].center, button=1))
            self.assertFalse(self.viewer.dragging)
        click('pause')
        self.assertTrue(self.viewer.paused)
        click('mode')
        self.assertEqual(self.viewer.mode, 1)
        previous = self.viewer.mesh.name
        click('model')
        self.assertNotEqual(self.viewer.mesh.name, previous)
        self.viewer.camera.zoom(5)
        self.viewer.camera.orbit(40, 30)
        click('reset')
        self.assertAlmostEqual(self.viewer.camera.yaw, 0.6)
        self.assertEqual(self.viewer.angle, 0)
        self.assertFalse(self.key(pygame.K_ESCAPE))

    def test_resize_rebuilds_viewport_and_surface(self):
        self.viewer.resize(pygame.Surface((850, 600)))
        frame = self.viewer.render()
        self.assertEqual(self.viewer.scene.get_width(), self.viewer.viewport.width * 2)
        self.assertGreater(len(frame.points), 0)

    def test_screenshot_cli_works_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'preview.png'
            result = subprocess.run([sys.executable, str(ROOT / 'main.py'), '--screenshot', str(path)],
                                    cwd=directory, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(pygame.image.load(str(path)).get_size(), SIZE)

    def test_missing_model_has_clear_cli_error(self):
        result = subprocess.run([sys.executable, str(ROOT / 'main.py'),
                                 '--model', str(ROOT / 'missing.obj'),
                                 '--screenshot', '/tmp/unused-3d-preview.png'],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 1)
        self.assertNotIn('Traceback', result.stderr)
        self.assertIn('missing.obj', result.stderr)

    def test_quit_event_exits_main_cleanly(self):
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        main([])
        self.assertFalse(pygame.display.get_init())


if __name__ == '__main__':
    unittest.main()
