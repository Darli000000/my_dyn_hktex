#!/usr/bin/env python3
"""Extend a verified single-frame dataset using local, pinned face parsing + first-hit geometry."""
import argparse, hashlib, io, json, os, shutil, traceback, urllib.request, zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFilter
from common import SCHEMA,REGIONS,write_json,sha
MODEL='jonathandinu/face-parsing'
REVISION='758b82e15a0178c9db39c1ff666a8b56e3a550c8'
LABELS=['background','skin','nose','eye_g','l_eye','r_eye','l_brow','r_brow','l_ear','r_ear','mouth','u_lip','l_lip','hair','hat','ear_r','neck_l','neck','cloth']
PALETTE=np.array([[40,40,40],[210,165,135],[215,190,50],[60,140,255],[220,80,200],[220,80,75],[50,190,170],[130,100,225],[100,190,90]],np.uint8)

class Source:
    def __init__(self,path):
        self.path=Path(path).resolve();self.z=zipfile.ZipFile(path) if self.path.is_file() else None
        if self.z:
            names=[n for n in self.z.namelist() if not n.startswith('__MACOSX/') and (n=='manifest.json' or n.endswith('/manifest.json'))]
            if len(names)!=1:raise ValueError('ZIP must contain exactly one dataset manifest')
            self.prefix=names[0][:-len('manifest.json')]
        self.manifest=json.loads(self.read('manifest.json'))
        if self.manifest['schema']!='hktex-vhap-static-eyes-v1':raise ValueError('Source must be the previous eye-inclusive dataset')
        for rel,digest in self.manifest['sha256'].items():
            if hashlib.sha256(self.read(rel)).hexdigest()!=digest:raise ValueError('Source hash mismatch '+rel)
    def read(self,rel):
        if Path(rel).is_absolute() or '..' in Path(rel).parts:raise ValueError('Unsafe source asset path')
        return self.z.read(self.prefix+rel) if self.z else (self.path/rel).read_bytes()
    def image(self,rel):return Image.open(io.BytesIO(self.read(rel)))
    def arrays(self,rel):
        with np.load(io.BytesIO(self.read(rel)),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def fetch(cache,name):
    p=cache/name;p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        tmp=p.with_suffix(p.suffix+'.part')
        with urllib.request.urlopen(f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}',timeout=120) as r,open(tmp,'wb') as f:shutil.copyfileobj(r,f)
        tmp.replace(p)
    return p

class Parser:
    def __init__(self,cache,threads):
        import onnxruntime as ort
        self.cache=Path(cache)/REVISION
        pp=fetch(self.cache,'preprocessor_config.json');cp=fetch(self.cache,'config.json');mp=fetch(self.cache,'onnx/model_quantized.onnx')
        if sha(mp)!='5bab9bfb3cb979f3098ac3b934b1641dbf87f835e0b03c2ca6d88dcf18c83d27':raise ValueError('Pinned ONNX model hash mismatch')
        self.cfg=json.loads(pp.read_text());config=json.loads(cp.read_text())
        if [config['id2label'][str(i)] for i in range(19)]!=LABELS:raise ValueError('Unexpected semantic model labels')
        if self.cfg['size']!={'height':512,'width':512} or self.cfg['resample']!=2:raise ValueError('Unexpected image processor')
        opt=ort.SessionOptions();opt.intra_op_num_threads=threads;opt.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(mp),sess_options=opt,providers=['CPUExecutionProvider'])
        self.input=self.session.get_inputs()[0].name
        self.record=dict(model=MODEL,revision=REVISION,onnxruntime=ort.__version__,model_sha256=sha(mp),preprocessor_sha256=sha(pp),provider='CPUExecutionProvider',confidence='sum of all allowed semantic class probabilities, not top-class confidence',labels=LABELS)
    def __call__(self,image):
        rgb=np.asarray(image.convert('RGB').resize((512,512),Image.Resampling.BILINEAR),dtype=np.float32)*self.cfg['rescale_factor']
        x=(rgb-np.array(self.cfg['image_mean'],np.float32))/np.array(self.cfg['image_std'],np.float32)
        logits=self.session.run(None,{self.input:np.ascontiguousarray(x.transpose(2,0,1)[None])})[0][0]
        if logits.shape[0]!=19 or not np.isfinite(logits).all():raise ValueError('Invalid parsing logits')
        # Half-pixel bilinear interpolation at original image size; no RGB resizing/cropping in dataset.
        logits=np.stack([np.asarray(Image.fromarray(a).resize(image.size,Image.Resampling.BILINEAR)) for a in logits])
        labels=logits.argmax(0).astype(np.uint8);ex=np.exp(logits-logits.max(0,keepdims=True));confidence=ex[[1,2,4,5,6,7,8,9,11,12,17]].sum(0)/ex.sum(0)
        return labels,confidence.astype(np.float32)

def morph(mask,radius,kind):
    if radius==0:return mask.copy()
    im=Image.fromarray(mask.astype(np.uint8)*255)
    return np.asarray(im.filter((ImageFilter.MinFilter if kind=='erode' else ImageFilter.MaxFilter)(radius*2+1)))>0

def region_image(semantic,face_ids,g):
    safe=np.maximum(face_ids,0);component=g['face_components'][safe];head=int(g['face_components'][g['face_regions']==1][0])
    left=int(g['face_components'][g['face_regions']==3][0]);right=int(g['face_components'][g['face_regions']==4][0])
    labels=np.zeros(face_ids.shape,np.uint8);onhead=(face_ids>=0)&(component==head)
    labels[onhead&np.isin(semantic,[1,4,5,6,7])]=1
    labels[onhead&((g['face_regions'][safe]==2)|np.isin(semantic,[4,5,6,7]))&np.isin(semantic,[1,4,5,6,7])]=2
    for value,sem in [(5,[11,12]),(6,[2]),(7,[8,9]),(8,[17])]:labels[onhead&np.isin(semantic,sem)]=value
    # Pixel left/right labels depend on viewpoint. Geometry component identity is authoritative.
    eyesem=np.isin(semantic,[1,4,5])
    labels[(face_ids>=0)&(component==left)&eyesem]=3;labels[(face_ids>=0)&(component==right)&eyesem]=4
    return labels

def make_mask(semantic,confidence,face,bary,alpha,g,camera,threshold=.8,min_cosine=.25):
    labels=region_image(semantic,face,g);safe=np.maximum(face,0)
    xyz=(g['vertices'][g['faces'][safe]]*bary[...,None]).sum(-2)
    d=np.asarray(camera['transform_matrix'])[:3,3]-xyz
    cos=(g['face_normals'][safe]*d).sum(-1)/np.maximum(np.linalg.norm(d,axis=-1),1e-30)
    boundary_vertices=np.unique(g['boundary_edges'][:,:2]);boundary=np.isin(g['faces'],boundary_vertices).any(1)
    # Keep topology; exclude supervision at an open edge, not the whole neck.
    occluder=np.isin(semantic,[0,3,13,14,15,16,18]);occluder=morph(occluder,1,'dilate')
    valid=(face>=0)&(labels>0)&(alpha>=.99)&(cos>=min_cosine)&~boundary[safe]&~occluder&(confidence>=threshold)
    # Erode the external valid boundary, not internal skin/nose/lip label boundaries.
    general=morph(valid,1,'erode')
    thin=np.isin(labels,[3,4,5])&(confidence>=max(.7,threshold))
    mask=valid&(general|thin)
    return mask,np.where(mask,labels,0).astype(np.uint8),dict(hit=int((face>=0).sum()),semantic_candidate=int((labels>0).sum()),geometry_and_semantic_valid=int(valid.sum()),final=int(mask.sum()))

def overlay(rgb,old,new,labels,path):
    a=np.asarray(rgb.convert('RGB'));w,h=rgb.size
    oldvis=a.copy();oldvis[old]=(a[old]*.5+[30,110,40]).clip(0,255)
    newvis=a.copy();newvis[new]=(a[new]*.5+[20,120,45]).clip(0,255);newvis[old&~new]=(a[old&~new]*.5+[120,20,10]).clip(0,255)
    grid=Image.new('RGB',(w*4,h+30),'white');draw=ImageDraw.Draw(grid)
    for i,(im,title) in enumerate([(rgb,'RGB'),(Image.fromarray(oldvis),'Old ROI'),(Image.fromarray(newvis),'New: green; removed old: red'),(Image.fromarray(PALETTE[labels]),'Skin / eyes / lips / nose / ears / neck')]):
        grid.paste(im,(i*w,30));draw.text((i*w+5,8),title,fill='black')
    grid.save(path)

def prepare(a,out,report):
    src=Source(a.source);m=src.manifest;g=src.arrays(m['mesh']);parser=Parser(a.model_cache,a.threads)
    report.update(source=str(src.path),source_manifest_sha256=hashlib.sha256(src.read('manifest.json')).hexdigest(),parser=parser.record,views=[])
    rows=[];hashes={};thumbs=[]
    for row in m['views']:
        key=row['id'];print('[MASK]',key,flush=True);folder=out/key;folder.mkdir()
        rgb=src.image(row['preview']).convert('RGB');semantic,confidence=parser(rgb)
        if a.semantic_overrides:
            override=Path(a.semantic_overrides)/(key+'.png')
            if override.exists():
                semantic=np.asarray(Image.open(override))
                if semantic.shape!=(row['height'],row['width']) or semantic.max()>18:raise ValueError('Override must be an original-size single-channel label PNG, values 0..18')
                confidence=np.ones(semantic.shape,np.float32)
                report.setdefault('overrides',{})[key]=sha(override)
        Image.fromarray(semantic).save(folder/'semantic_labels.png');np.save(folder/'semantic_confidence.npy',confidence)
        geo=src.arrays(row['geometry']);old=np.asarray(src.image(row['mask']))>0;alpha=np.asarray(src.image(row['alpha']))/255.
        new,labels,counts=make_mask(semantic,confidence,geo['face_ids'],geo['barycentric'],alpha,g,row['camera'],a.confidence,a.min_cosine)
        if new.sum()<32:raise ValueError('Too little valid ROI '+key)
        for field in ['rgb','alpha','geometry','preview']:
            rel=row[field];target=out/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(src.read(rel))
        legacy_mask=f'{key}/legacy_mask.png';legacy_regions=f'{key}/legacy_regions.png'
        (out/legacy_mask).write_bytes(src.read(row['mask']));(out/legacy_regions).write_bytes(src.read(row['regions']))
        Image.fromarray(new.astype(np.uint8)*255).save(out/row['mask']);Image.fromarray(labels).save(out/row['regions'])
        overlay(rgb,old,new,labels,folder/'overlay.png')
        im=Image.open(folder/'overlay.png');im.thumbnail((880,360));thumbs.append(im.copy())
        counts.update(id=key,split=row['split'],old=int(old.sum()),added=int((new&~old).sum()),removed=int((old&~new).sum()),region_pixels={name:int((labels==i).sum()) for i,name in REGIONS.items()})
        report['views'].append(counts);write_json(out/'report.json',report)
        rows.append(dict(row,legacy_mask=legacy_mask,legacy_regions=legacy_regions,valid_pixels=int(new.sum()),semantic_labels=f'{key}/semantic_labels.png'))
        for field in ['rgb','alpha','geometry','preview','mask','regions','legacy_mask','legacy_regions','semantic_labels']:hashes[rows[-1][field]]=sha(out/rows[-1][field])
    # Permit all head/eye faces geometrically; per-view image labels and masks provide supervision.
    head=int(g['face_components'][g['face_regions']==1][0]);lf=int(g['face_components'][g['face_regions']==3][0]);rf=int(g['face_components'][g['face_regions']==4][0])
    newfaces=g['face_regions'].copy();newfaces[g['face_components']==head]=1
    newfaces[g['face_regions']==2]=2;newfaces[g['face_components']==lf]=3;newfaces[g['face_components']==rf]=4
    g['face_regions']=newfaces;g['allowed_faces']=newfaces>0
    np.savez_compressed(out/m['mesh'],**g);hashes[m['mesh']]=sha(out/m['mesh'])
    manifest=dict(m,schema=SCHEMA,views=rows,sha256=hashes,region_names=REGIONS,
        training_permitted=True,roi_semantics='semantic parsing intersected with first-hit geometry; automatic proposal, not manual ground truth',
        checks=dict(m['checks'],min_cosine=a.min_cosine,erosion_pixels=1,thin_region_erosion_pixels=0,semantic_confidence=a.confidence),
        provenance=dict(m['provenance'],source_manifest_sha256=report['source_manifest_sha256'],parser=parser.record))
    write_json(out/'manifest.json',manifest)
    h=sum(t.height+22 for t in thumbs);sheet=Image.new('RGB',(880,h),'white');draw=ImageDraw.Draw(sheet);y=0
    for row,im in zip(rows,thumbs):draw.text((5,y+3),row['id'],fill='black');sheet.paste(im,(0,y+22));y+=im.height+22
    sheet.save(out/'coverage_contact_sheet.jpg',quality=93)
    totalold=sum(r['old'] for r in report['views']);totalnew=sum(r['final'] for r in report['views'])
    report.update(status='prepared',old_pixels=totalold,new_pixels=totalnew,coverage_ratio=totalnew/totalold,
        train_region_pixels={name:sum(r['region_pixels'][name] for r in report['views'] if r['split']=='train') for name in REGIONS.values()})
    if totalnew<=totalold:raise ValueError('Masks did not expand overall: inspect semantic labels/contact sheet')
    if report['train_region_pixels']['lips']<32 or report['train_region_pixels']['nose']<32:raise ValueError('Missing new nose/lip supervision; inspect masks')
    write_json(out/'report.json',report)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model-cache',default=str(Path.home()/'.cache/hktex-face-parser'));p.add_argument('--threads',type=int,default=4)
    p.add_argument('--confidence',type=float,default=.8);p.add_argument('--min-cosine',type=float,default=.25);p.add_argument('--semantic-overrides')
    a=p.parse_args();out=Path(a.output).resolve()
    if out.exists():p.error('Use a NEW output dataset directory')
    if not 0<=a.confidence<=1 or not 0<=a.min_cosine<1 or a.threads<1:p.error('Invalid thresholds/threads')
    out.mkdir(parents=True);report=dict(status='running',args=vars(a))
    try:prepare(a,out,report)
    except BaseException as e:report.update(status='failed',error=str(e),traceback=traceback.format_exc());write_json(out/'report.json',report);raise
    print('[PREPARED]',out,flush=True)
if __name__=='__main__':main()
