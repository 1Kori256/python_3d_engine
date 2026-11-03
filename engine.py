"""A small CPU rendering pipeline: OBJ loading, camera, clipping, and projection."""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    name: str

    @classmethod
    def from_obj(cls, path, normalize=True):
        path = Path(path)
        vertices, faces = [], []
        for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            fields = line.partition('#')[0].split()
            if not fields:
                continue
            try:
                if fields[0] == 'v':
                    if len(fields) < 4:
                        raise ValueError('vertex needs three coordinates')
                    vertex = [float(value) for value in fields[1:4]]
                    if not np.isfinite(vertex).all():
                        raise ValueError('vertex coordinates must be finite')
                    vertices.append(vertex)
                elif fields[0] == 'f':
                    if len(fields) < 4:
                        raise ValueError('face needs at least three vertices')
                    indices = []
                    for corner in fields[1:]:
                        index = int(corner.split('/')[0])
                        if index == 0:
                            raise ValueError('OBJ indices start at 1')
                        index = index - 1 if index > 0 else len(vertices) + index
                        if not 0 <= index < len(vertices):
                            raise ValueError('face references an undefined vertex')
                        indices.append(index)
                    # Fan triangulation preserves the winding of convex polygons.
                    faces.extend((indices[0], indices[i], indices[i + 1])
                                 for i in range(1, len(indices) - 1))
            except ValueError as error:
                raise ValueError(f'{path.name}:{line_number}: {error}') from error
        if not vertices or not faces:
            raise ValueError(f'{path.name}: mesh needs vertices and faces')
        vertices = np.asarray(vertices, dtype=float)
        if normalize:
            low, high = vertices.min(axis=0), vertices.max(axis=0)
            extent = float(np.max(high - low))
            if extent <= 1e-12:
                raise ValueError(f'{path.name}: mesh has zero extent')
            vertices = (vertices - (low + high) / 2) * (2.8 / extent)
        return cls(vertices, np.asarray(faces, dtype=int), path.name)


def rotation_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


@dataclass
class Camera:
    yaw: float = 0.6
    pitch: float = 0.25
    distance: float = 6.0

    @property
    def position(self):
        horizontal = self.distance * math.cos(self.pitch)
        return np.array([horizontal * math.sin(self.yaw),
                         self.distance * math.sin(self.pitch),
                         -horizontal * math.cos(self.yaw)])

    def basis(self):
        """Columns are right, up, forward; the camera looks along positive Z."""
        forward = -self.position
        length = np.linalg.norm(forward)
        if length <= 1e-12:
            raise ValueError('Camera distance must be positive')
        forward /= length
        up = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(up, forward)) > 0.999:
            up = np.array([0.0, 0.0, 1.0])
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        return np.column_stack((right, np.cross(forward, right), forward))

    def world_to_view(self, points):
        return (np.asarray(points) - self.position) @ self.basis()

    def orbit(self, dx, dy):
        self.yaw -= dx * 0.008
        self.pitch = float(np.clip(self.pitch + dy * 0.008, -1.5, 1.5))

    def zoom(self, steps):
        self.distance = float(np.clip(self.distance * math.exp(-steps * 0.12), 0.4, 20))


def frustum_planes(width, height, fov=50, near=0.1, far=100):
    if width <= 0 or height <= 0 or not 0 < fov < 180 or not 0 < near < far:
        raise ValueError('Invalid viewport or perspective parameters')
    vertical = math.tan(math.radians(fov) / 2)
    horizontal = vertical * width / height
    # A point is inside when dot(normal, point) + offset >= 0.
    return np.array([[0, 0, 1, -near], [0, 0, -1, far],
                     [1, 0, horizontal, 0], [-1, 0, horizontal, 0],
                     [0, 1, vertical, 0], [0, -1, vertical, 0]])


def clip_polygon(polygon, planes):
    """Sutherland-Hodgman clipping in view space, before perspective division."""
    polygon = list(np.asarray(polygon, dtype=float))
    for plane in planes:
        if not polygon:
            break
        output = []
        previous = polygon[-1]
        previous_distance = float(previous @ plane[:3] + plane[3])
        for current in polygon:
            distance = float(current @ plane[:3] + plane[3])
            if (distance >= 0) != (previous_distance >= 0):
                t = previous_distance / (previous_distance - distance)
                output.append(previous + t * (current - previous))
            if distance >= 0:
                output.append(current)
            previous, previous_distance = current, distance
        polygon = output
    return np.asarray(polygon, dtype=float).reshape(-1, 3)


def project(points, width, height, fov=50):
    points = np.asarray(points, dtype=float)
    if np.any(points[..., 2] <= 0):
        raise ValueError('Clip points behind the camera before projecting')
    factor = height / (2 * math.tan(math.radians(fov) / 2))
    result = points[..., :2] * (factor / points[..., 2, None])
    result[..., 0] += width / 2
    result[..., 1] = height / 2 - result[..., 1]
    return result


@dataclass
class Frame:
    points: np.ndarray
    shades: np.ndarray
    depths: np.ndarray


def prepare_frame(mesh, camera, width, height, angle=0, cull=True):
    """Transform vertices in batches and sort visible triangles once per frame."""
    planes = frustum_planes(width, height)
    world = mesh.vertices @ rotation_y(angle).T
    triangles = world[mesh.faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    if cull:
        valid &= np.einsum('ij,ij->i', normals, triangles[:, 0] - camera.position) < 0
    normals = normals[valid] / lengths[valid, None]
    light = np.array([-0.4, 0.8, -0.6])
    light /= np.linalg.norm(light)
    shades = 0.24 + 0.76 * np.clip(normals @ light, 0, 1)
    viewed = camera.world_to_view(world)[mesh.faces[valid]]
    distances = viewed @ planes[:, :3].T + planes[:, 3]
    outside = np.any(np.all(distances < 0, axis=1), axis=1)
    viewed, shades, distances = viewed[~outside], shades[~outside], distances[~outside]
    inside = np.all(distances >= 0, axis=(1, 2))
    parts = [viewed[inside]]
    part_shades = [shades[inside]]
    for triangle, shade in zip(viewed[~inside], shades[~inside]):
        polygon = clip_polygon(triangle, planes)
        if len(polygon) >= 3:
            clipped = np.array([[polygon[0], polygon[i], polygon[i + 1]]
                                for i in range(1, len(polygon) - 1)])
            parts.append(clipped)
            part_shades.append(np.full(len(clipped), shade))
    triangles = np.concatenate(parts)
    shades = np.concatenate(part_shades)
    depths = triangles[:, :, 2].mean(axis=1)
    order = np.argsort(-depths, kind='stable')
    return Frame(project(triangles[order], width, height), shades[order], depths[order])
