"""Inspect OBJ meshes with a small software renderer."""

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

import numpy as np
import pygame

from engine import Camera, Mesh, prepare_frame

ROOT = Path(__file__).resolve().parent
SIZE = (1180, 780)
BACKGROUND = (0, 0, 0)
TEXT = (224, 224, 224)
MUTED = (135, 135, 135)
BORDER = (36, 36, 36)
MODES = ('Shaded', 'Edges', 'Wireframe')


class Viewer:
    def __init__(self, surface, model=ROOT / 'objects/pokus3.obj'):
        self.font = pygame.font.SysFont('helveticaneue,arial,liberationsans', 18)
        self.small = pygame.font.SysFont('helveticaneue,arial,liberationsans', 15)
        self.title = pygame.font.SysFont('georgia,dejavuserif', 31)
        self.models = list(dict.fromkeys([Path(model).resolve(),
                                         *sorted((ROOT / 'objects').glob('*.obj'))]))
        self.model_index = 0
        self.mesh = Mesh.from_obj(self.models[0])
        self.camera = Camera()
        self.angle = 0.0
        self.paused = False
        self.dragging = False
        self.mode = 0
        self.fps = 0.0
        self.message = 'Drag to orbit. Scroll to zoom.'
        self.resize(surface)
        self.reset()

    def resize(self, surface):
        self.surface = surface
        width, height = surface.get_size()
        self.viewport = pygame.Rect(32, 88, width - 64, height - 226)
        self.scene = pygame.Surface((self.viewport.width * 2, self.viewport.height * 2))
        self.buttons = {
            'model': pygame.Rect(width - 310, 26, 180, 34),
            'reset': pygame.Rect(width - 110, 26, 78, 34),
            'pause': pygame.Rect(40, height - 93, 86, 36),
            'mode': pygame.Rect(180, height - 93, 120, 36),
        }

    def reset(self):
        self.camera = Camera()
        radius = float(np.linalg.norm(self.mesh.vertices, axis=1).max())
        self.camera.distance = radius / math.sin(math.radians(25)) * 1.12
        self.angle = 0.0
        self.dragging = False

    def next_model(self):
        index = (self.model_index + 1) % len(self.models)
        try:
            mesh = Mesh.from_obj(self.models[index])
        except (OSError, ValueError) as error:
            self.message = str(error)
            return
        self.model_index, self.mesh = index, mesh
        self.message = 'Drag to orbit. Scroll to zoom.'
        self.reset()

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            if event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif event.key == pygame.K_w:
                self.mode = (self.mode + 1) % len(MODES)
            elif event.key == pygame.K_m:
                self.next_model()
            elif event.key == pygame.K_r:
                self.reset()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.buttons['pause'].collidepoint(event.pos):
                self.paused = not self.paused
            elif self.buttons['mode'].collidepoint(event.pos):
                self.mode = (self.mode + 1) % len(MODES)
            elif self.buttons['model'].collidepoint(event.pos):
                self.next_model()
            elif self.buttons['reset'].collidepoint(event.pos):
                self.reset()
            elif self.viewport.collidepoint(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.camera.orbit(*event.rel)
        elif event.type == pygame.MOUSEWHEEL:
            self.camera.zoom(event.y)
        elif event.type == pygame.WINDOWFOCUSLOST:
            self.dragging = False
        return True

    def update(self, seconds):
        if not self.paused and not self.dragging:
            self.angle = (self.angle + seconds * 0.35) % math.tau

    def text(self, value, position, color=TEXT, font=None):
        self.surface.blit((font or self.font).render(value, True, color), position)

    def button(self, name, label):
        rect = self.buttons[name]
        if rect.collidepoint(pygame.mouse.get_pos()):
            pygame.draw.rect(self.surface, (24, 24, 24), rect, border_radius=4)
        while self.font.size(label)[0] > rect.width - 12 and len(label) > 4:
            label = label[:-4] + '...'
        rendered = self.font.render(label, True, TEXT)
        self.surface.blit(rendered, rendered.get_rect(center=rect.center))

    def draw_axes(self):
        center = np.array([self.viewport.right - 45, self.viewport.bottom - 40])
        basis = self.camera.basis()
        colors = ((157, 99, 91), (99, 133, 105), (95, 123, 157))
        for label, direction, color in zip('XYZ', np.eye(3) @ basis, colors):
            end = center + direction[:2] * [28, -28]
            pygame.draw.aaline(self.surface, color, center, end)
            self.text(label, tuple(end + [4, -8]), color, self.small)

    def render(self):
        width, height = self.surface.get_size()
        self.surface.fill(BACKGROUND)
        self.scene.fill(BACKGROUND)
        frame = prepare_frame(self.mesh, self.camera, *self.scene.get_size(),
                              angle=self.angle, cull=self.mode != 2)
        colors = np.clip(frame.shades[:, None] * [151, 174, 181], 0, 255).astype(int)
        # Each triangle is drawn once; prepare_frame already sorted the complete scene.
        for points, color in zip(frame.points, colors):
            if self.mode != 2:
                pygame.draw.polygon(self.scene, tuple(color), points)
            if self.mode != 0:
                edge = (80, 104, 115) if self.mode == 2 else tuple((color * 0.72).astype(int))
                pygame.draw.aalines(self.scene, edge, True, points)
        self.surface.blit(pygame.transform.smoothscale(self.scene, self.viewport.size), self.viewport)
        self.draw_axes()
        self.text('3D engine', (40, 24), TEXT, self.title)
        hint = self.small.render(self.message, True, MUTED)
        self.surface.blit(hint, hint.get_rect(center=(width // 2, 43)))
        self.button('model', self.mesh.name + '  >')
        self.button('reset', 'Reset')
        pygame.draw.line(self.surface, BORDER, (40, height - 120), (width - 40, height - 120))
        self.button('pause', 'Play' if self.paused else 'Pause')
        self.button('mode', MODES[self.mode])
        self.text(f'{len(self.mesh.vertices)} vertices   /   {len(self.mesh.faces)} triangles',
                  (356, height - 87), MUTED)
        status = f'{self.fps:.0f} fps' if self.fps else 'Software renderer'
        rendered = self.small.render(status, True, MUTED)
        self.surface.blit(rendered, (width - 40 - rendered.get_width(), height - 83))
        self.text('Space  pause     /     W  display     /     M  model     /     R  reset',
                  (40, height - 34), MUTED, self.small)
        self.text('Esc  quit', (width - 94, height - 34), MUTED, self.small)
        return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'objects/pokus3.obj', help='OBJ file to open')
    parser.add_argument('--screenshot', type=Path, help='Render a PNG without opening a window')
    args = parser.parse_args(argv)
    if args.screenshot:
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    pygame.display.init()
    pygame.font.init()
    try:
        surface = (pygame.Surface(SIZE) if args.screenshot
                   else pygame.display.set_mode(SIZE, pygame.RESIZABLE))
        pygame.display.set_caption('Python 3D engine')
        try:
            app = Viewer(surface, args.model)
        except (OSError, ValueError) as error:
            parser.exit(1, f'{parser.prog}: {error}\n')
        if args.screenshot:
            app.render()
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(surface, str(args.screenshot))
            return
        clock = pygame.time.Clock()
        running = True
        while running:
            seconds = min(clock.tick(60) / 1000, 0.1)
            for event in pygame.event.get():
                if event.type == pygame.VIDEORESIZE:
                    surface = pygame.display.set_mode((max(850, event.w), max(600, event.h)),
                                                      pygame.RESIZABLE)
                    app.resize(surface)
                elif not app.handle_event(event):
                    running = False
                    break
            if running:
                app.update(seconds)
                app.fps = clock.get_fps()
                app.render()
                pygame.display.flip()
    finally:
        pygame.quit()


if __name__ == '__main__':
    main()
