"""CPU-only geometry/data contract for VHAP checks. No UV or synthetic-light assumptions."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

SCHEMA = 'hktex-vhap-static-fullface-v1'
REGIONS = {1: 'base_face', 2: 'eye_surround', 3: 'left_eye', 4: 'right_eye', 5: 'lips', 6: 'nose', 7: 'ears', 8: 'neck'}

def face_regions(f, vertex_count, groups):
    """Anatomical regions; all original triangles remain present for visibility."""
    def union(names):
        mask=np.zeros(vertex_count,bool)
        for name in names:mask[np.asarray(groups[name],dtype=np.int64)]=True
        return mask
    eyes=union(['left_eyeball','right_eyeball'])
    common_exclude=union(['neck','scalp','boundary','lips','lip_inside','teeth'])
    old=union(['face'])&~common_exclude&~eyes&~union(['eye_region'])
    skin=union(['face','eye_region'])&~common_exclude&~eyes
    labels=np.zeros(len(f),np.uint8)
    labels[skin[f].all(1)]=2
    labels[old[f].all(1)]=1
    for key,name in [(3,'left_eyeball'),(4,'right_eyeball')]:
        take=union([name])[f].all(1)
        if (labels[take]>0).any():raise ValueError('Overlapping anatomical labels')
        labels[take]=key
    if any(not (labels==k).any() for k in REGIONS):raise ValueError('Missing anatomical face region')
    return labels

def region_counts(labels):
    return {name:int((labels==k).sum()) for k,name in REGIONS.items()}

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); tmp.replace(path)

def inside(root, relative):
    root = Path(root).resolve(); p = (root/relative).resolve()
    if root not in p.parents: raise ValueError('Asset escapes dataset: '+str(relative))
    return p

def read_obj(path):
    v, f = [], []
    for line in Path(path).read_text().splitlines():
        a = line.split()
        if not a: continue
        if a[0] == 'v': v.append([float(x) for x in a[1:4]])
        if a[0] == 'f':
            if len(a) != 4: raise ValueError('Expected triangular OBJ, no automatic triangulation')
            ids = [int(x.split('/')[0]) for x in a[1:]]
            f.append([i-1 if i > 0 else len(v)+i for i in ids])
    v = np.asarray(v, np.float64); f = np.asarray(f, np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all(): raise ValueError('Bad vertices')
    if f.ndim != 2 or f.shape[1] != 3 or f.min() < 0 or f.max() >= len(v): raise ValueError('Bad faces')
    return v, f

def topology(v, f):
    cross = np.cross(v[f[:,1]]-v[f[:,0]], v[f[:,2]]-v[f[:,0]])
    lengths = np.linalg.norm(cross, axis=1)
    if (lengths <= 0).any(): raise ValueError('Zero-area triangle; inspect mesh before continuing')
    fn = cross/lengths[:,None]; vn = np.zeros_like(v)
    for j in range(3): np.add.at(vn, f[:,j], cross)
    norm = np.linalg.norm(vn, axis=1)
    if (norm == 0).any(): raise ValueError('Unused vertex or undefined averaged normal')
    vn /= norm[:,None]
    edges = defaultdict(list); adj = defaultdict(set)
    for fi, tri in enumerate(f):
        for a,b in zip(tri,np.roll(tri,-1)):
            a,b = int(a),int(b)
            edges[tuple(sorted((a,b)))].append((fi,a,b)); adj[a].add(b); adj[b].add(a)
    labels = np.full(len(v),-1,np.int32); sizes = []
    for i in range(len(v)):
        if labels[i] >= 0: continue
        ci=len(sizes); stack=[i]; labels[i]=ci; size=0
        while stack:
            a=stack.pop(); size+=1
            for b in adj[a]:
                if labels[b] < 0: labels[b]=ci; stack.append(b)
        sizes.append(size)
    nonmanifold = sum(len(x)>2 for x in edges.values())
    winding = sum(len(x)==2 and x[0][1:]==x[1][1:] for x in edges.values())
    boundary = [(a,b,items[0][0]) for (a,b),items in edges.items() if len(items)==1]
    report = dict(vertices=len(v), faces=len(f), components=len(sizes), component_vertices=sizes,
        boundary_edges=len(boundary), nonmanifold_edges=nonmanifold, inconsistent_winding_edges=winding,
        bounds=[v.min(0).tolist(),v.max(0).tolist()], area=float(lengths.sum()/2),
        minimum_triangle_area=float(lengths.min()/2), geometry_modified=False,
        self_intersections_checked=False, vertex_manifoldness_checked=False)
    return report, fn, vn, labels[f[:,0]], np.asarray(boundary,dtype=np.int64).reshape(-1,3)

def srgb_to_linear(x):
    x=np.asarray(x,np.float32)
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4).astype(np.float32)

def linear_to_srgb(x):
    x=np.clip(x,0,1)
    return np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055)

def validate_camera(r):
    T=np.asarray(r['transform_matrix'],np.float64)
    if T.shape!=(4,4) or not np.isfinite(T).all(): raise ValueError('Invalid c2w')
    if not np.allclose(T[3],[0,0,0,1],atol=1e-7): raise ValueError('Non-affine c2w')
    if not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-5) or abs(np.linalg.det(T[:3,:3])-1)>1e-5: raise ValueError('Invalid rotation')
    if min(r['fl_x'],r['fl_y'],r['w'],r['h'])<=0: raise ValueError('Invalid intrinsics')
    return T

def project(v,r):
    T=validate_camera(r)
    c=np.c_[v,np.ones(len(v))]@np.linalg.inv(T).T
    depth=-c[:,2]
    uv=np.c_[r['fl_x']*c[:,0]/depth+r['cx'],r['cy']-r['fl_y']*c[:,1]/depth]
    return uv,depth

def rays(r,ids):
    y,x=np.divmod(np.asarray(ids),r['w']); T=validate_camera(r)
    d=np.stack([(x+.5-r['cx'])/r['fl_x'],-(y+.5-r['cy'])/r['fl_y'],-np.ones(len(x))],1)
    d=d@T[:3,:3].T; d/=np.linalg.norm(d,axis=1,keepdims=True)
    return np.broadcast_to(T[:3,3],d.shape).copy(),d

def rasterize(v,f,r):
    """Perspective-correct CPU z-buffer at pixel centres, ALL faces occlude; no culling."""
    uv,z=project(v,r); h,w=r['h'],r['w']
    if (z<=1e-7).any(): raise ValueError('Mesh intersects camera near plane; clipping is not implemented')
    depth=np.full((h,w),np.inf); face=np.full((h,w),-1,np.int32); bary=np.zeros((h,w,3),np.float32)
    for fi,tri in enumerate(f):
        p=uv[tri]; lo=np.maximum(np.ceil(p.min(0)-.5).astype(int),0); hi=np.minimum(np.floor(p.max(0)-.5).astype(int),[w-1,h-1])
        if (hi<lo).any():continue
        x,y=np.meshgrid(np.arange(lo[0],hi[0]+1)+.5,np.arange(lo[1],hi[1]+1)+.5)
        den=(p[1,1]-p[2,1])*(p[0,0]-p[2,0])+(p[2,0]-p[1,0])*(p[0,1]-p[2,1])
        if abs(den)<1e-14:continue
        a=((p[1,1]-p[2,1])*(x-p[2,0])+(p[2,0]-p[1,0])*(y-p[2,1]))/den
        b=((p[2,1]-p[0,1])*(x-p[2,0])+(p[0,0]-p[2,0])*(y-p[2,1]))/den
        screen=np.stack([a,b,1-a-b],-1); valid=(screen>=-1e-10).all(-1)
        invz=(screen/z[tri]).sum(-1)
        zz=np.divide(1,invz,out=np.full_like(invz,np.inf),where=invz>0)
        sl=np.s_[lo[1]:hi[1]+1,lo[0]:hi[0]+1]; take=valid&(zz<depth[sl])
        persp=screen/z[tri]; persp/=np.where(invz!=0,invz,1)[...,None]
        depth[sl][take]=zz[take];face[sl][take]=fi;bary[sl][take]=persp[take]
    depth[face<0]=0
    return face,bary,depth.astype(np.float32)

def verify_dataset(root):
    root=Path(root).resolve(); m=json.loads((root/'manifest.json').read_text())
    if m['schema']!=SCHEMA:raise ValueError('Wrong schema; not a synthetic offline dataset')
    for rel,digest in m['sha256'].items():
        if sha(inside(root,rel))!=digest: raise ValueError('Changed asset: '+rel)
    return m
