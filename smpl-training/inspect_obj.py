"""Read-only OBJ/UV audit using the local HKTex load_mesh function, without CUDA imports."""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import types

import numpy as np
from PIL import Image
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def components(n, edges):
    e = np.asarray(edges).reshape(-1, 2)
    graph = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    return connected_components(graph, directed=False)


def audit(mesh_path, loader_path):
    # Execute only the reviewed mesh-loading function from the user's local repo.
    # No package __init__, eigen computation, mesh repairs, or cache writes.
    text = loader_path.read_text()
    fn = next(x for x in ast.parse(text).body
              if isinstance(x, ast.FunctionDef) and x.name == 'load_mesh')
    ns = {'trimesh': trimesh, 'np': np, 'hktex': types.SimpleNamespace(info=print)}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(loader_path), 'exec'), ns)
    rows = [line.split() for line in mesh_path.read_text().splitlines() if line.strip()]
    counts = Counter(row[0] for row in rows)
    polygons = Counter(len(row) - 1 for row in rows if row[0] == 'f')
    mtls = [mesh_path.parent / ' '.join(row[1:]) for row in rows if row[0] == 'mtllib']
    if len(mtls) != 1 or not mtls[0].is_file():
        raise ValueError('Expected one existing MTL file for this sample OBJ')
    maps = [mtls[0].parent / line.split(maxsplit=1)[1]
            for line in mtls[0].read_text().splitlines() if line.startswith('map_Kd ')]
    if len(maps) != 1 or not maps[0].is_file():
        raise ValueError('Expected one existing map_Kd colour image; check relative paths')
    mesh = ns['load_mesh'](str(mesh_path), merge_tex=True, bake_vert_colors=False)
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    uv, tf = np.asarray(mesh.original_uv), np.asarray(mesh.original_faces)
    image = getattr(mesh.visual.material, 'baseColorTexture', None)
    if image is None:
        image = getattr(mesh.visual.material, 'image', None)
    if image is None:
        raise ValueError('No loaded texture image')
    with Image.open(maps[0]) as expected:
        texture_matches = np.array_equal(np.asarray(expected.convert('RGB')),
                                        np.asarray(image.convert('RGB')))
    edges, edge_counts = np.unique(np.sort(mesh.edges, axis=1), axis=0, return_counts=True)
    nc, labels = components(len(v), edges)
    uvtri = uv[tf]
    a, b = uvtri[:, 1] - uvtri[:, 0], uvtri[:, 2] - uvtri[:, 0]
    uvarea = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) * .5
    position_error = float(np.max(np.abs(mesh.original_vertices[tf] - v[f])))
    # A manifold vertex link must be one path or one cycle.
    incident = defaultdict(list)
    for face in f:
        for i in range(3):
            incident[int(face[i])].append([int(face[(i+1)%3]), int(face[(i+2)%3])])
    bad_vertices = []
    for vid, link in incident.items():
        used, inverse = np.unique(link, return_inverse=True)
        degree = np.bincount(inverse.ravel(), minlength=len(used))
        nlink, _ = components(len(used), inverse.reshape(-1, 2))
        if nlink != 1 or np.any(degree > 2) or np.count_nonzero(degree == 1) not in (0, 2):
            bad_vertices.append(vid)
    angles = np.degrees(mesh.face_angles).min(axis=1)
    info = {
        'mesh_path': str(mesh_path),
        'mesh_sha256': hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        'loader_path': str(loader_path),
        'loader_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'trimesh_version': trimesh.__version__,
        'obj_vertices': counts['v'], 'obj_uv_entries': counts['vt'],
        'obj_face_sizes': dict(polygons),
        'vertices_after_welding': len(v), 'triangle_faces': len(f),
        'components': int(nc),
        'component_vertex_counts': sorted(np.bincount(labels).tolist(), reverse=True),
        'watertight': bool(mesh.is_watertight),
        'winding_consistent': bool(mesh.is_winding_consistent),
        'boundary_edges': int((edge_counts == 1).sum()),
        'nonmanifold_edges': int((edge_counts > 2).sum()),
        'nonmanifold_vertices': len(bad_vertices),
        'nonfinite_vertices': int((~np.isfinite(v)).sum()),
        'zero_or_near_zero_area_faces': int((mesh.area_faces <= 1e-15).sum()),
        'duplicate_faces': len(f) - len(np.unique(np.sort(f, axis=1), axis=0)),
        'min_angle_degrees': float(angles.min()),
        'faces_min_angle_lt1deg': int((angles < 1).sum()),
        'faces_min_angle_lt5deg': int((angles < 5).sum()),
        'uv_nonfinite': int((~np.isfinite(uv)).sum()),
        'uv_degenerate_faces': int((uvarea < 1e-12).sum()),
        'uv_range': [float(uv.min()), float(uv.max())],
        'uv_face_geometry_max_error': position_error,
        'texture_path': str(maps[0]), 'texture_size': list(image.size),
        'texture_sha256': hashlib.sha256(maps[0].read_bytes()).hexdigest(),
        'loaded_texture_matches_map_Kd': bool(texture_matches),
        'texture_rgb_std': np.asarray(image.convert('RGB').resize((256, 256))).std(axis=(0, 1)).tolist(),
        'limits': 'Static checks only; no self-intersection, eigen, CUDA tracing, or training validation.',
    }
    failures = []
    for key in ('nonmanifold_edges', 'nonmanifold_vertices', 'nonfinite_vertices',
                'zero_or_near_zero_area_faces', 'duplicate_faces', 'uv_nonfinite',
                'uv_degenerate_faces'):
        if info[key]:
            failures.append(key)
    if not texture_matches: failures.append('loaded_texture_mismatch')
    if position_error > 1e-6: failures.append('uv_face_geometry_mismatch')
    if not info['winding_consistent']: failures.append('inconsistent_winding')
    warnings = []
    if info['boundary_edges']: warnings.append('Open boundaries: inspect before tracing')
    if nc != 1: warnings.append('Multiple connected components: inspect before tracing')
    if info['faces_min_angle_lt1deg']: warnings.append('Very thin triangles: retain tracer checks')
    if any(int(k) != 3 for k in polygons): warnings.append('OBJ polygons are triangulated by trimesh')
    info.update(failures=failures, warnings=warnings, ready_for_smoke=not failures)
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', default='/root/hktex-work/hktex')
    p.add_argument('--mesh', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    report = audit(Path(a.mesh).resolve(), Path(a.repo).resolve() / 'hktex/utils/mesh.py')
    out = Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print('REPORT:', out)
    if not report['ready_for_smoke']:
        raise SystemExit('Static checks failed; review report before training')


if __name__ == '__main__':
    main()
