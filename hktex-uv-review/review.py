import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

def stat(x,y,mask):
    if not mask.any():return {'pixels':0,'mse':None,'psnr':None}
    err=np.mean(np.square(x[mask].astype(float)-y[mask]))
    return {'pixels':int(mask.sum()),'mse':float(err),'psnr':float(-10*np.log10(max(err,1e-15)))}

def grid(path, panels):
    w,h=panels[0][1].size;out=Image.new('RGB',(len(panels)*w,h+24),'white');draw=ImageDraw.Draw(out)
    for i,(title,img) in enumerate(panels):
        draw.text((i*w+4,5),title,fill='black');out.paste(img,(i*w,24))
    out.save(path)

def evaluate(out,folder,dataset,row,tri,ref,pure,mix,counts,v,pred,old):
    from atlas import decode,encode,sample
    ids=v['ids'];active=v['active'];legacy=v['legacy'];gt=v['rgb'];h,w=row['height'],row['width']
    with np.load(dataset/row['geometry']) as z:
        f=z['face_ids'].ravel();bc=z['barycentric'].reshape(-1,3)
    vis=np.flatnonzero(f>=0);uv=(tri[f[vis]]*bc[vis,:,None]).sum(1)
    loc=np.searchsorted(vis,ids)
    if not np.array_equal(vis[loc],ids):raise ValueError('Cached rays differ from dataset')
    vals=[sample(decode(tex),uv) for tex in (ref,pure,mix)]
    full_support=sample((counts>0)[:,:,None].astype('float32'),uv[loc])[:,0]>=1-1e-7
    inside=active&full_support
    def paint(indices,color):
        im=np.full((h*w,3),255,np.uint8);im[indices]=encode(color);return Image.fromarray(im.reshape(h,w,3))
    def roi(color):return paint(ids[active],color[active])
    photo=Image.fromarray(encode(np.load(dataset/row['rgb'])))
    grid(folder/'roi.png',[('Photo / ROI',roi(gt)),('VHAP texture',roi(vals[0][loc])),('Direct HKTex',roi(pred)),('UV supported / gray fallback',roi(vals[1][loc])),('Hybrid',roi(vals[2][loc]))])
    grid(folder/'mesh.png',[('Photo',photo),('VHAP / mesh',paint(vis,vals[0])),('HKTex / supported',paint(vis,vals[1])),('HKTex + VHAP / mesh',paint(vis,vals[2]))])
    rep={'expanded_pixels':int(active.sum()),'bake_comparable_pixels':int(inside.sum()),'bake_vs_direct_supported':stat(vals[1][loc],pred,inside)}
    for group,mask in [('expanded',active),('bake_comparable',inside),('old_intersection_new',legacy&active),('old_only',legacy&~active)]:
        rep[group]={k:stat(x,gt,mask) for k,x in [('vhap',vals[0][loc]),('direct',pred),('hybrid',vals[2][loc])]}
        if old is not None and group.startswith('old_'):
            oi=np.searchsorted(old['ids'],ids[mask]);assert np.array_equal(old['ids'][oi],ids[mask])
            if not np.allclose(old['gt'][oi],gt[mask],rtol=0,atol=1e-7):raise ValueError('Old GT differs')
            rep[group]['previous_direct']=stat(old['pred'][oi],gt[mask],np.ones(mask.sum(),bool))
    return rep

def main():
    ap=argparse.ArgumentParser()
    for k in ['repo','pair','out']:ap.add_argument('--'+k,required=True)
    ap.add_argument('--obj');ap.add_argument('--old',action='store_true');ap.add_argument('--gpu',default='0')
    a=ap.parse_args();out=Path(a.out);repo=Path(a.repo)
    sys.path.insert(0,str(repo/('hktex-real-face-pair' if a.old else 'hktex-fullface-pair')))
    from experiment import environment,context,make
    from common import sha
    from runtime import predict
    from atlas import read_layout,raster,support,assets
    environment(a);pair,c,dataset,m,g=context(a)
    checkpoint=pair/'baseline'/f'step_{c["steps"]:05d}';state=checkpoint/'state.pt'
    meta=json.loads((checkpoint/'checkpoint.json').read_text())
    if meta['state_sha256']!=sha(state) or meta['manifest_sha256']!=sha(dataset/'manifest.json'):raise ValueError('Checkpoint mismatch')
    folder=out/('old_model' if a.old else 'model');folder.mkdir()
    model,_,_=make(a,pair,c,g,folder,state)
    if not a.old:
        obj=Path(a.obj);tri,tex,translation=read_layout(obj,g);print('[BAKE] rasterizing',flush=True)
        owner,bary,conflict=raster(tri,2048);counts,points,faces=support(dataset,m,g,owner,bary,conflict)
        if not len(faces):raise ValueError('No train-supported texels')
        print('[BAKE] querying',len(faces),'supported texels',flush=True)
        values=np.empty((len(points),3),np.float32)
        for start in range(0,len(points),65536):
            end=min(start+65536,len(points))
            values[start:end]=predict(model,points[start:end],faces[start:end])
            print(f'[BAKE] {end}/{len(points)}',flush=True)
        ref,pure,mix=assets(out,obj,tex,owner,conflict,counts,values)
        report={'status':'evaluating','checkpoint_sha256':sha(state),'training_minus_obj_translation':translation.tolist(),'supported_texels':len(faces),'ambiguous_uv_texels':int(conflict.sum()),'views':{},'display':'unlit sRGB surface appearance; linear RGB MSE','coverage':'train cameras only, same-face first hit with active mask; comparison requires all contributing bilinear taps supported'}
    index=json.loads((pair/'ray_cache/index.json').read_text())
    saved=json.loads((checkpoint/'metrics.json').read_text())
    for row in m['views']:
        if row['split'] not in ('val','test'):continue
        name=row['id'];cache=pair/'ray_cache'/(name+'.npz')
        if sha(cache)!=index[name]:raise ValueError('Ray cache mismatch')
        with np.load(cache) as z:v=dict(z)
        pred=predict(model,v['points'],v['faces']);active=v.get('active',np.ones(len(pred),bool))
        check=stat(pred,v['rgb'],active)['mse']
        if not np.isclose(check,saved[row['split']]['all']['mse'],rtol=1e-4,atol=1e-7):raise ValueError(f'{name}: direct query does not reproduce saved MSE: {check}')
        previous=out/('previous_'+name+'.npz')
        if a.old:
            np.savez_compressed(previous,ids=v['ids'],gt=v['rgb'],pred=pred);continue
        result=out/name;result.mkdir()
        old=None
        if previous.exists():
            with np.load(previous) as z:old=dict(z)
        report['views'][name]=evaluate(out,result,dataset,row,tri,ref,pure,mix,counts,v,pred,old)
        print('[EVAL]',name,flush=True)
    if not a.old:
        report['status']='completed';(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
if __name__=='__main__':main()
