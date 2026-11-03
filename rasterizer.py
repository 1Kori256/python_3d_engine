"""Compiled triangle rasterization with perspective-correct color and depth."""

import math

from numba import njit


@njit(cache=True, nogil=True)
def rasterize(points, inverse_depths, colors, pixels, depth):
    """Fill reusable H x W buffers; larger reciprocal depth is closer."""
    pixels.fill(0)
    depth.fill(0)
    height, width = depth.shape
    for triangle in range(len(points)):
        x0, y0 = points[triangle, 0]
        x1, y1 = points[triangle, 1]
        x2, y2 = points[triangle, 2]
        area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        if abs(area) < 1e-10:
            continue
        left = max(0, int(math.floor(min(x0, x1, x2))))
        right = min(width - 1, int(math.ceil(max(x0, x1, x2))))
        top = max(0, int(math.floor(min(y0, y1, y2))))
        bottom = min(height - 1, int(math.ceil(max(y0, y1, y2))))
        a0, b0 = (y1 - y2) / area, (x2 - x1) / area
        a1, b1 = (y2 - y0) / area, (x0 - x2) / area
        c0 = (x1 * y2 - x2 * y1) / area
        c1 = (x2 * y0 - x0 * y2) / area
        z0, z1, z2 = inverse_depths[triangle]
        for y in range(top, bottom + 1):
            w0 = a0 * (left + 0.5) + b0 * (y + 0.5) + c0
            w1 = a1 * (left + 0.5) + b1 * (y + 0.5) + c1
            for x in range(left, right + 1):
                w2 = 1 - w0 - w1
                if w0 >= -1e-10 and w1 >= -1e-10 and w2 >= -1e-10:
                    inverse_z = w0 * z0 + w1 * z1 + w2 * z2
                    if inverse_z > depth[y, x]:
                        depth[y, x] = inverse_z
                        scale = 1 / inverse_z
                        for channel in range(3):
                            value = (w0 * colors[triangle, 0, channel] * z0
                                     + w1 * colors[triangle, 1, channel] * z1
                                     + w2 * colors[triangle, 2, channel] * z2) * scale
                            pixels[y, x, channel] = min(255, max(0, int(value + 0.5)))
                w0 += a0
                w1 += a1


@njit(cache=True, nogil=True)
def draw_edges(points, inverse_depths, pixels, depth, hidden=True):
    """Draw wire edges; solid overlays respect the depth buffer."""
    height, width = depth.shape
    for triangle in range(len(points)):
        for edge in range(3):
            following = (edge + 1) % 3
            x0, y0 = points[triangle, edge]
            x1, y1 = points[triangle, following]
            z0 = inverse_depths[triangle, edge]
            z1 = inverse_depths[triangle, following]
            steps = max(1, int(math.ceil(max(abs(x1 - x0), abs(y1 - y0)))))
            for step in range(steps + 1):
                t = step / steps
                x = int(round(x0 + (x1 - x0) * t))
                y = int(round(y0 + (y1 - y0) * t))
                if 0 <= x < width and 0 <= y < height:
                    inverse_z = z0 + (z1 - z0) * t
                    if not hidden or inverse_z * 1.002 >= depth[y, x]:
                        pixels[y, x, 0] = 83 if hidden else 153
                        pixels[y, x, 1] = 105 if hidden else 177
                        pixels[y, x, 2] = 115 if hidden else 188


@njit(cache=True, nogil=True)
def antialias_edges(pixels, depth, output, bounds):
    """Soften diagonal silhouette steps without blurring surface shading."""
    output[:] = pixels
    height, width = depth.shape
    left, top, right, bottom = bounds
    for y in range(max(1, top), min(height - 1, bottom)):
        for x in range(max(1, left), min(width - 1, right)):
            horizontal = (depth[y, x - 1] > 0) != (depth[y, x + 1] > 0)
            vertical = (depth[y - 1, x] > 0) != (depth[y + 1, x] > 0)
            if horizontal and vertical:
                for channel in range(3):
                    value = (2 * int(pixels[y, x, channel])
                             + int(pixels[y - 1, x, channel]) + int(pixels[y + 1, x, channel])
                             + int(pixels[y, x - 1, channel]) + int(pixels[y, x + 1, channel]))
                    output[y, x, channel] = value // 6
