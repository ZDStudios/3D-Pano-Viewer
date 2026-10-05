"""Export an equirectangular panorama as 3D models (inward-facing sphere).

Outputs (all Y-up, so Blender's importers convert them to Z-up for you):
  * <name>.glb             – single file, texture embedded, unlit material
  * <name>.obj/.mtl/.jpg   – classic OBJ + material + texture
  * <name>_equirect.jpg    – the flat 360 image (use as a Blender World/HDRI)
"""
import json
import math
import os
import shutil
import struct

import numpy as np


def sphere_mesh(radius=10.0, segments=128, rings=64):
    """Sphere seen from the inside. Returns positions, normals, uvs (v up),
    and triangle indices wound so faces point toward the centre."""
    lon = np.linspace(-math.pi, math.pi, segments + 1)
    lat = np.linspace(math.pi / 2, -math.pi / 2, rings + 1)
    LON, LAT = np.meshgrid(lon, lat)
    # right-handed, Y up, looking down +Z = centre of the image, east to the right
    d = np.stack([-np.sin(LON) * np.cos(LAT), np.sin(LAT), np.cos(LON) * np.cos(LAT)], -1)
    pos = (d * radius).reshape(-1, 3).astype(np.float32)
    nrm = (-d).reshape(-1, 3).astype(np.float32)
    u = (LON + math.pi) / (2 * math.pi)
    v = (LAT + math.pi / 2) / math.pi
    uv = np.stack([u, v], -1).reshape(-1, 2).astype(np.float32)

    idx = []
    row = segments + 1
    for r in range(rings):
        for s in range(segments):
            a = r * row + s
            b = a + row
            idx += [a, a + 1, b, a + 1, b + 1, b]
    idx = np.array(idx, np.uint32).reshape(-1, 3)
    # make sure winding faces inward (drop degenerate pole triangles)
    p = pos[idx]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    area = np.linalg.norm(n, axis=1)
    keep = area > 1e-9
    idx, n, p = idx[keep], n[keep], p[keep]
    if np.mean(np.einsum("ij,ij->i", n, p.mean(axis=1))) > 0:
        idx = idx[:, [0, 2, 1]]
    return pos, nrm, uv, idx


def export_obj(folder, name, jpg_path, radius=10.0):
    pos, nrm, uv, idx = sphere_mesh(radius)
    tex = f"{name}.jpg"
    shutil.copyfile(jpg_path, os.path.join(folder, tex))
    with open(os.path.join(folder, f"{name}.mtl"), "w") as fh:
        fh.write(f"newmtl pano\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nNs 0\nillum 0\nmap_Kd {tex}\nmap_Ke {tex}\nKe 1 1 1\n")
    lines = [f"# 360 panorama sphere (inside-facing), radius {radius}\n",
             f"mtllib {name}.mtl\n", f"o {name}\n"]
    lines += [f"v {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in pos]
    lines += [f"vt {a:.6f} {b:.6f}\n" for a, b in uv]
    lines += [f"vn {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in nrm]
    lines += ["usemtl pano\n", "s 1\n"]
    lines += [f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n" for a, b, c in idx + 1]
    with open(os.path.join(folder, f"{name}.obj"), "w") as fh:
        fh.writelines(lines)


def export_glb(path, jpg_path, radius=10.0):
    pos, nrm, uv, idx = sphere_mesh(radius)
    uv = uv.copy()
    uv[:, 1] = 1.0 - uv[:, 1]  # glTF UV origin is top-left
    with open(jpg_path, "rb") as fh:
        img = fh.read()

    chunks, views = [], []
    offset = 0

    def add(data, target=None):
        nonlocal offset
        pad = (-len(data)) % 4
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target:
            view["target"] = target
        views.append(view)
        chunks.append(data + b"\0" * pad)
        offset += len(data) + pad
        return len(views) - 1

    v_pos = add(pos.tobytes(), 34962)
    v_nrm = add(nrm.tobytes(), 34962)
    v_uv = add(uv.tobytes(), 34962)
    v_idx = add(idx.astype(np.uint32).tobytes(), 34963)
    v_img = add(img)

    gltf = {
        "asset": {"version": "2.0", "generator": "3D Pano Viewer"},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "Panorama"}],
        "meshes": [{"name": "PanoramaSphere", "primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
            "indices": 3, "material": 0}]}],
        "materials": [{"name": "Panorama", "doubleSided": False,
                       "pbrMetallicRoughness": {"baseColorTexture": {"index": 0},
                                                "metallicFactor": 0.0, "roughnessFactor": 1.0},
                       "extensions": {"KHR_materials_unlit": {}}}],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": v_img, "mimeType": "image/jpeg"}],
        "accessors": [
            {"bufferView": v_pos, "componentType": 5126, "count": len(pos), "type": "VEC3",
             "min": pos.min(0).tolist(), "max": pos.max(0).tolist()},
            {"bufferView": v_nrm, "componentType": 5126, "count": len(nrm), "type": "VEC3"},
            {"bufferView": v_uv, "componentType": 5126, "count": len(uv), "type": "VEC2"},
            {"bufferView": v_idx, "componentType": 5125, "count": idx.size, "type": "SCALAR"},
        ],
        "bufferViews": views,
        "buffers": [{"byteLength": offset}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    binary = b"".join(chunks)
    total = 12 + 8 + len(js) + 8 + len(binary)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, total))
        fh.write(struct.pack("<I4s", len(js), b"JSON") + js)
        fh.write(struct.pack("<I4s", len(binary), b"BIN\0") + binary)


def export_all(out_dir, name, jpg_path, formats=("glb", "obj", "equirect")):
    os.makedirs(out_dir, exist_ok=True)
    written = []
    if "glb" in formats:
        p = os.path.join(out_dir, f"{name}.glb")
        export_glb(p, jpg_path)
        written.append(p)
    if "obj" in formats:
        export_obj(out_dir, name, jpg_path)
        written.append(os.path.join(out_dir, f"{name}.obj"))
    if "equirect" in formats:
        p = os.path.join(out_dir, f"{name}_equirect.jpg")
        shutil.copyfile(jpg_path, p)
        written.append(p)
    with open(os.path.join(out_dir, "README - Blender import.txt"), "w") as fh:
        fh.write(
            "Blender import tips\n"
            "===================\n\n"
            f"GLB (recommended): File > Import > glTF 2.0 > {name}.glb\n"
            "  The sphere is 20 m across with the camera meant to sit at the centre (0,0,0).\n"
            "  The material is unlit (emission-like) so it looks right without any lights.\n\n"
            f"OBJ: File > Import > Wavefront (.obj) > {name}.obj (keep the .mtl and .jpg next to it).\n\n"
            f"As a sky / environment: World Properties > Color > Environment Texture > {name}_equirect.jpg\n"
            "  This lights your whole scene with the drone panorama.\n")
    return written
