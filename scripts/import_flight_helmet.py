"""Rebuild the bundled helmet OBJ. Extra dependency: fast-simplification==0.2.0."""

import hashlib
import json
from pathlib import Path
import urllib.request

import fast_simplification
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVISION = '8f492390ddc7a2ae221a1369ef560589cf8c70fb'
BASE = f'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/{REVISION}'
MODEL = f'{BASE}/Models/FlightHelmet/glTF'


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def main():
    description = fetch(f'{MODEL}/FlightHelmet.gltf')
    model = json.loads(description)
    binary = fetch(f'{MODEL}/FlightHelmet.bin')

    def read_accessor(index):
        accessor = model['accessors'][index]
        view = model['bufferViews'][accessor['bufferView']]
        dtype = np.dtype({5123: '<u2', 5125: '<u4', 5126: '<f4'}[accessor['componentType']])
        columns = {'SCALAR': 1, 'VEC3': 3}[accessor['type']]
        offset = view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
        stride = view.get('byteStride', columns * dtype.itemsize)
        return np.ndarray((accessor['count'], columns), dtype=dtype, buffer=binary,
                          offset=offset, strides=(stride, dtype.itemsize)).copy()

    vertices, faces = [], []
    offset = 0
    nodes = model['scenes'][model.get('scene', 0)]['nodes']
    for index in nodes:
        node = model['nodes'][index]
        if any(key in node for key in ('matrix', 'translation', 'rotation', 'scale', 'children')):
            raise ValueError('This asset conversion expects the pinned, untransformed mesh nodes')
        for primitive in model['meshes'][node['mesh']]['primitives']:
            if primitive.get('mode', 4) != 4:
                raise ValueError('Expected triangle primitives')
            positions = read_accessor(primitive['attributes']['POSITION']).astype(float)
            indices = read_accessor(primitive['indices']).reshape(-1, 3).astype(np.int32)
            vertices.append(positions)
            faces.append(indices + offset)
            offset += len(positions)
    vertices, faces = np.concatenate(vertices), np.concatenate(faces)
    original_faces = len(faces)
    # Texture seams split vertices in glTF. Weld positions before simplifying geometry.
    _, unique, inverse = np.unique(np.round(vertices, 7), axis=0,
                                   return_index=True, return_inverse=True)
    vertices, faces = vertices[unique], inverse[faces]
    vertices, faces = fast_simplification.simplify(vertices, faces, target_count=16000)
    output = ROOT / 'objects/flight_helmet.obj'
    with output.open('w', encoding='utf-8') as stream:
        stream.write('# Flight Helmet, Khronos sample assets. CC0-1.0. See ATTRIBUTION.md.\n')
        for x, y, z in vertices:
            stream.write(f'v {x:.8g} {y:.8g} {z:.8g}\n')
        for a, b, c in faces + 1:
            stream.write(f'f {a} {b} {c}\n')
    metadata = {
        'source_revision': REVISION,
        'source': f'{MODEL}/FlightHelmet.gltf',
        'license': 'CC0-1.0',
        'source_gltf_sha256': hashlib.sha256(description).hexdigest(),
        'source_bin_sha256': hashlib.sha256(binary).hexdigest(),
        'obj_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'original_triangles': original_faces,
        'triangles': len(faces),
        'vertices': len(vertices),
        'conversion': 'Positions welded at 7 decimal places; geometry simplified; materials omitted.',
    }
    output.with_suffix('.source.json').write_text(json.dumps(metadata, indent=2) + '\n')
    license_path = ROOT / 'objects/licenses/CC0-1.0.txt'
    license_path.parent.mkdir(exist_ok=True)
    license_text = fetch(f'{BASE}/LICENSES/CC0-1.0.txt').decode('utf-8')
    license_path.write_text('\n'.join(line.rstrip() for line in license_text.splitlines()) + '\n')
    print(f'Wrote {output.name}: {len(vertices):,} vertices, {len(faces):,} triangles')


if __name__ == '__main__':
    main()
