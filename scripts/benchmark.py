"""Measure warm render time without a window or the display's refresh-rate cap."""

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pygame

from engine import Mesh
from main import SIZE, Viewer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=80)
    parser.add_argument('--quality', choices=('standard', 'high'), default='standard')
    args = parser.parse_args()
    if args.frames < 1:
        parser.error('--frames must be positive')
    pygame.display.init()
    pygame.font.init()
    results = {}
    try:
        app = Viewer(pygame.Surface(SIZE), quality=2 if args.quality == 'high' else 1)
        for name in ('pokus3.obj', 'sphere.obj'):
            app.mesh = Mesh.from_obj(ROOT / 'objects' / name)
            app.reset()
            samples = []
            for index in range(args.frames + 10):
                app.angle = index * 0.013
                start = time.perf_counter()
                app.render()
                elapsed = (time.perf_counter() - start) * 1000
                if index >= 10:  # Exclude JIT compilation, cache setup, and cold frames.
                    samples.append(elapsed)
            results[name] = {
                'median_ms': round(statistics.median(samples), 3),
                'p95_ms': round(float(np.percentile(samples, 95)), 3),
                'triangles': len(app.mesh.faces),
            }
        samples = []
        for _ in range(args.frames):
            start = time.perf_counter()
            app.render()
            samples.append((time.perf_counter() - start) * 1000)
        results['stationary_scene'] = {'median_ms': round(statistics.median(samples), 3)}
    finally:
        pygame.quit()
    print(json.dumps({'quality': args.quality, 'size': SIZE, 'results': results}, indent=2))


if __name__ == '__main__':
    main()
