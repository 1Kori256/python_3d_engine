# Flight Helmet

Source: [Khronos glTF Sample Assets — Flight Helmet](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/8f492390ddc7a2ae221a1369ef560589cf8c70fb/Models/FlightHelmet).

The source asset is released under [CC0 1.0 Universal](licenses/CC0-1.0.txt).
Khronos credits Gary Hsu for the Maya conversion (2018).

`flight_helmet.obj` contains geometry only. Texture seams were welded and the
94,722-triangle source was simplified to about 16,000 triangles. Materials,
textures, and transparency were omitted. Source revision and checksums are in
`flight_helmet.source.json`.

Rebuild with `python scripts/import_flight_helmet.py` after installing the optional
conversion tool with `python -m pip install fast-simplification==0.2.0`. This tool
is not needed to run the viewer.

The other OBJ files were already present in the original repository.
