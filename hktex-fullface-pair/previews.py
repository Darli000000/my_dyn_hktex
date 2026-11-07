from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from common import linear_to_srgb

def preview(view,pred,out):
    row=view['row'];h,w=row['height'],row['width'];ids=view['ids']
    def canvas(values):
        image=np.full((h*w,3),255,np.uint8);image[ids]=values;return Image.fromarray(image.reshape(h,w,3))
    gt=canvas(np.rint(linear_to_srgb(view['rgb'])*255).astype(np.uint8))
    prediction=canvas(np.rint(linear_to_srgb(pred)*255).astype(np.uint8))
    e=np.clip(np.abs(pred-view['rgb']).mean(1)*10,0,1)
    err=canvas(np.c_[e*255,np.zeros(len(e)),np.zeros(len(e))].astype(np.uint8))
    grid=Image.new('RGB',(w*3,h+30),'white');draw=ImageDraw.Draw(grid)
    for i,(im,label) in enumerate([(gt,'GT in supervised ROI'),(prediction,'HKTex surface RGB'),(err,'linear RGB MAE x10, red')]):
        grid.paste(im,(i*w,30));draw.text((i*w+5,8),label,fill='black')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);grid.save(out/(row['id']+'.png'))
    eyeids=ids[np.isin(view['regions'],[2,3,4])]
    if len(eyeids):
        y,x=np.divmod(eyeids,w);box=(max(0,int(x.min())-12),max(0,int(y.min())-12),min(w,int(x.max())+13),min(h,int(y.max())+13))
        crops=[im.crop(box) for im in [gt,prediction,err]];cw,ch=crops[0].size
        zoom=Image.new('RGB',(cw*3,ch),'white')
        for i,im in enumerate(crops):zoom.paste(im,(i*cw,0))
        zoom.resize((cw*3*3,ch*3),Image.Resampling.NEAREST).save(out/(row['id']+'_eyes.png'))
    return grid
