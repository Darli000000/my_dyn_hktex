"""VHAP OBJ UV layout and train-camera visibility, in image (top-left) coordinates."""
from pathlib import Path
import numpy as np
from PIL import Image
from common import project, linear_to_srgb, srgb_to_linear

def read_layout(obj,g):
    lines=obj.read_text().splitlines()
    def rows(tag):
        return [line.split()[1:] for line in lines if line.startswith(tag+' ')]
    v=np.asarray(rows('v'),float)[:,:3]
    uv=np.asarray(rows('vt'),float)[:,:2]
    corners=[[x.split('/') for x in face] for face in rows('f')]
    faces=np.array([[int(c[0])-1 for c in face] for face in corners])
    tids=np.array([[int(c[1])-1 for c in face] for face in corners])
    if not np.array_equal(faces,g['faces']) or v.shape!=g['vertices'].shape:
        raise ValueError('OBJ topology differs from training geometry')
    delta=g['vertices']-v
    if np.abs(delta-delta.mean(0)).max()>1e-5:raise ValueError('OBJ must be the same posed frame up to translation')
    if tids.min()<0 or tids.max()>=len(uv) or not np.all((uv>=0)&(uv<=1)):raise ValueError('Invalid UV')
    mtl=obj.parent/' '.join(rows('mtllib')[0])
    tex=mtl.parent/next(l.split(maxsplit=1)[1].strip() for l in mtl.read_text().splitlines() if l.startswith('map_Kd '))
    return uv[tids],tex,delta.mean(0)

def raster(tris,size):
    owner=np.full((size,size),-1,np.int32)
    bary=np.zeros((size,size,3),np.float32)
    conflict=np.zeros_like(owner,bool)
    for fid,t in enumerate(tris):
        q=t.copy();q[:,1]=1-q[:,1];q=q*size-.5
        lo=np.maximum(np.ceil(q.min(0)).astype(int),0);hi=np.minimum(np.floor(q.max(0)).astype(int),size-1)
        if (lo>hi).any():continue
        a,b=q[1]-q[0],q[2]-q[0];den=a[0]*b[1]-a[1]*b[0]
        if abs(den)<1e-12:continue
        x,y=np.meshgrid(np.arange(lo[0],hi[0]+1)-q[0,0],np.arange(lo[1],hi[1]+1)-q[0,1])
        s=(x*b[1]-y*b[0])/den;t=(a[0]*y-a[1]*x)/den
        w=np.stack((1-s-t,s,t),axis=-1);inside=w.min(2)>=-1e-9
        sl=np.s_[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        conflict[sl]|=inside&(w.min(2)>1e-7)&(owner[sl]>=0)&(bary[sl].min(2)>1e-7)
        take=inside&(owner[sl]<0)
        owner[sl][take]=fid;bary[sl][take]=w[take]
    return owner,bary,conflict

def support(dataset,m,g,owner,bary,conflict):
    ids=np.flatnonzero((owner>=0)&~conflict)
    f=owner.ravel()[ids];p=(g['vertices'][g['faces'][f]]*bary.reshape(-1,3)[ids,:,None]).sum(1)
    counts=np.zeros(len(f),np.uint8)
    for row in m['views']:
        if row['split']!='train':continue
        mask=np.asarray(Image.open(dataset/row['mask']))>0
        with np.load(dataset/row['geometry']) as z:hit=z['face_ids']
        for start in range(0,len(p),100000):
            sl=slice(start,start+100000);xy,depth=project(p[sl],row['camera']);x,y=np.floor(xy).astype(int).T
            h,w=mask.shape;ok=(depth>0)&(x>=0)&(x<w)&(y>=0)&(y<h)
            y=y.clip(0,h-1);x=x.clip(0,w-1)
            counts[sl]+=ok&mask[y,x]&(hit[y,x]==f[sl])
    image=np.zeros(owner.shape,np.uint8);image.ravel()[ids]=counts
    return image,p[counts>0],f[counts>0]

def sample(image,uv):
    h,w=image.shape[:2]
    x=(uv[:,0]*w-.5).clip(0,w-1);y=((1-uv[:,1])*h-.5).clip(0,h-1)
    i=x.astype(int);j=y.astype(int);ii=(i+1).clip(max=w-1);jj=(j+1).clip(max=h-1)
    dx=(x-i)[:,None];dy=(y-j)[:,None]
    return (1-dy)*((1-dx)*image[j,i]+dx*image[j,ii])+dy*((1-dx)*image[jj,i]+dx*image[jj,ii])

def encode(x):
    return np.rint(linear_to_srgb(np.clip(x,0,1))*255).astype(np.uint8)
def decode(x):
    return srgb_to_linear(x.astype(np.float32)/255)

def assets(out,obj,tex,owner,conflict,counts,values):
    hit=counts>0;size=len(hit)
    ref=np.asarray(Image.open(tex).convert('RGB'))
    if ref.shape[:2]!=(size,size):raise ValueError('Expected 2048-square native VHAP texture')
    pure=np.full_like(ref,128);mix=ref.copy();pure[hit]=mix[hit]=encode(values)
    # Filtering gutter on unoccupied atlas only; never borrow from unsupported surfaces.
    reached=hit.copy()
    for _ in range(3):
        frontier=reached.copy()
        for dy,dx in [(1,0),(-1,0),(0,1),(0,-1)]:
            src=(slice(max(0,-dy),size-max(0,dy)),slice(max(0,-dx),size-max(0,dx)))
            dst=(slice(max(0,dy),size-max(0,-dy)),slice(max(0,dx),size-max(0,-dx)))
            use=(owner[dst]<0)&~frontier[dst]&reached[src]
            pure[dst][use]=pure[src][use];mix[dst][use]=pure[dst][use];frontier[dst][use]=True
        reached=frontier
    Image.fromarray(counts).save(out/'train_camera_count.png')
    Image.fromarray((hit*255).astype('uint8')).save(out/'support.png')
    for name,im in [('vhap',ref),('hktex_supported',pure),('hktex_hybrid',mix)]:
        folder=out/name;folder.mkdir();Image.fromarray(im).save(folder/'texture.png')
        (folder/'head.mtl').write_text('newmtl Appearance\nKd 1 1 1\nillum 0\nmap_Kd texture.png\n')
        lines=obj.read_text().splitlines()
        (folder/'head.obj').write_text('\n'.join('mtllib head.mtl' if l.startswith('mtllib ') else 'usemtl Appearance' if l.startswith('usemtl ') else l for l in lines)+'\n')
    return ref,pure,mix
