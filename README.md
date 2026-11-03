# Python 3D engine

A software renderer for OBJ meshes, with smooth lighting and a per-pixel depth
buffer. Drag to orbit and scroll to zoom.

![Mesh viewer](docs/preview.png)

## Run

Requires Python 3.10+. The first launch compiles the rasterizer with Numba and can
take a few seconds; subsequent launches reuse the compiled code.

```sh
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python main.py
```

Open another mesh with `python main.py --model objects/cube.obj`. Paths passed to
`--model` are relative to your working directory; the default mesh is found
relative to the application. `main2.py` opens the same viewer.

| Control | Action |
| --- | --- |
| Drag | Orbit the camera |
| Scroll | Zoom |
| Space / Pause | Pause rotation |
| W / Display button | Smooth, flat, edges, or wireframe |
| M / Filename button | Next bundled mesh |
| R / Reset | Fit the model and reset the camera |
| Q / Quality button | Standard or High antialiasing |
| Esc | Quit |

## Rendering

`engine.py` transforms vertices in batches, culls back faces, clips geometry and
vertex colors against all six frustum planes, and applies perspective projection.
`rasterizer.py` uses compiled CPU loops to resolve depth per pixel and interpolate
colors with perspective correction. Intersecting triangles no longer depend on
draw order. Equal-depth surfaces can still exhibit ordinary Z-fighting.

Smooth normals preserve sharp creases. A camera-relative key and fill light keep
the visible side lit while orbiting. Lighting includes a subtle specular highlight.
Flat mode keeps the original faceted appearance; wireframe shows all edges.

Standard quality uses native resolution with a silhouette edge filter. High uses
2x supersampling. Buffers and mesh normals are reused, and stationary views reuse
the completed frame. `main.py` handles input and the Pygame window.

The OBJ loader supports vertex positions, slash-separated and negative indices,
and fan triangulation of convex polygons. Meshes are centered and scaled to fit.
Textures, materials, and imported vertex normals are not used. Concave polygons
should be triangulated before export.

The original head has only 100 vertices and 196 triangles. Smooth shading improves
its surface appearance; finer facial features require a more detailed OBJ mesh.
There are no textures, transparency, or cast shadows.

## Tests and preview

```sh
python -m unittest discover -s tests -v
python main.py --screenshot docs/preview.png
python scripts/benchmark.py
```

Tests cover camera transforms, clipping, OBJ parsing, perspective interpolation,
crossing-triangle visibility in either draw order, and headless interaction.
The screenshot command saves a PNG without opening a window. Benchmarks exclude
compilation and measure warm render time without the display's refresh-rate cap.
