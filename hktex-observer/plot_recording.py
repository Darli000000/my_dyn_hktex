"""CPU-only plots and a self-contained local HTML index of recorded/replayed states."""
import argparse
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams.update({'figure.constrained_layout.use': True, 'font.size': 9,
                     'axes.titlesize': 10, 'figure.titlesize': 11})


def project(ax, pos, values, back=False, **kw):
    depth = -pos[:,2] if back else pos[:,2]
    order = np.argsort(depth)
    x = -pos[:,0] if back else pos[:,0]
    out = ax.scatter(x[order],pos[order,1],c=np.asarray(values)[order],s=2,rasterized=True,**kw)
    ax.set_aspect('equal'); ax.set_axis_off()
    return out


def finish(fig, path):
    fig.savefig(path,dpi=130,bbox_inches='tight'); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',required=True)
    p.add_argument('--analysis',help='Default: recording/analysis')
    p.add_argument('--output',help='Default: recording/viewer')
    p.add_argument('--probe',type=int,default=0)
    a=p.parse_args()
    root=Path(a.recording).resolve()
    analysis=Path(a.analysis).resolve() if a.analysis else root/'analysis'
    out=Path(a.output).resolve() if a.output else root/'viewer'
    out.mkdir(parents=True,exist_ok=True)
    metrics=json.loads((analysis/'metrics.json').read_text())
    surface=dict(np.load(root/'surface.npz',allow_pickle=False)); pos=surface['positions']
    limits={k:1e-8 for k in ['raw','normalized','weighted','alpha']}; error_max=1e-8
    for m in metrics:
        name=Path(m['file']).stem
        with np.load(analysis/(name+'_fields.npz'),allow_pickle=False) as d:
            for k in limits:
                if k in d and d[k].size:limits[k]=max(limits[k],float(np.abs(d[k]).max()))
            error_max=max(error_max,float(d['error_mse_rgb_mean'].max()))
    gallery=[]
    for m in metrics:
        name=Path(m['file']).stem
        d=dict(np.load(analysis/(name+'_fields.npz'),allow_pickle=False))
        q=dict(np.load(analysis/(name+'_probes.npz'),allow_pickle=False))
        state=dict(np.load(root/'snapshots'/m['file'],allow_pickle=False))
        title=f'Updates {m["completed_updates"]} | {m["phase"]} | {m["kernels"]} kernels'
        images={}
        fig,axes=plt.subplots(2,4,figsize=(12,8))
        for row,back in enumerate([False,True]):
            project(axes[row,0],pos,surface['gt'],back)
            project(axes[row,1],pos,np.clip(d['predicted'],0,1),back)
            sc=project(axes[row,2],pos,d['error_mse_rgb_mean'],back,cmap='magma',vmin=0,vmax=error_max)
            fig.colorbar(sc,ax=axes[row,2],shrink=.6)
            project(axes[row,3],pos,np.full((len(pos),3),.85),back)
            k=state['_kernel_locations']; axes[row,3].scatter((-k[:,0] if back else k[:,0]),k[:,1],s=.4,c='#005f99',alpha=.5)
            for ax,label in zip(axes[row],['GT','Prediction','RGB mean squared error','All kernel centres (incl. occluded)']):ax.set_title(label,fontsize=9)
        fig.suptitle(title+'\nSurface point projections: front / back; no lighting')
        images['overview']=name+'_overview.png'; finish(fig,out/images['overview'])
        if 'raw' in d:
            n=d['raw'].shape[1]
            fig,axes=plt.subplots(n,5,figsize=(12,max(3,2.5*n)),squeeze=False)
            for j in range(n):
                for col,key in enumerate(['raw','normalized','weighted','alpha']):
                    sc=project(axes[j,col],pos,d[key][:,j],cmap='coolwarm',vmin=-limits[key],vmax=limits[key])
                    axes[j,col].set_title(f'{key} | ID {d["representative_ids"][j]}',fontsize=8)
                    fig.colorbar(sc,ax=axes[j,col],shrink=.55)
                gate=d['candidate'][:,j].astype(int)+d['selected'][:,j].astype(int)
                project(axes[j,4],pos,gate,cmap='viridis',vmin=0,vmax=2)
                axes[j,4].set_title(f'root {d["root_ids"][j]}: gate 0/1/2',fontsize=8)
            fig.suptitle(title+'\nDirect kernel fields; gate: 0 absent, 1 candidate, 2 selected.\nFixed scales across snapshots.')
            images['kernels']=name+'_kernels.png';finish(fig,out/images['kernels'])
        probe=a.probe
        if not 0<=probe<len(q['candidate_ids']):raise ValueError('probe index outside saved probes')
        strongest=int(q['selected_local'][probe,0]); kid=int(q['candidate_ids'][probe,strongest])
        label=f'Fixed probe {probe}; selected kernel with largest alpha in this snapshot: ID {kid}'
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        for ax,key in zip(axes.flat,['evals','decay','phi_source','phi_query','spectral_terms']):
            ax.plot(q[key][probe,strongest],lw=.8);ax.set_title(key);ax.set_xlabel('aligned mode index\n(not frequency sorted)');ax.grid(alpha=.2)
        terms=q['spectral_terms'][probe,strongest]
        axes[1,2].plot(np.cumsum(terms,dtype=np.float64));axes[1,2].axhline(q['raw'][probe,strongest],color='r',ls='--',label='actual raw response');axes[1,2].legend();axes[1,2].set_title('Cumulative spectral sum')
        fig.suptitle(title+'\n'+label)
        images['spectrum']=name+'_spectrum.png';finish(fig,out/images['spectrum'])
        fig,axes2=plt.subplots(2,3,figsize=(14,8));axes=axes2[0]
        axes[0].plot(q['biharmonic_terms'][probe,strongest]);axes[0].set_title('Per-mode biharmonic squared distance')
        axes[1].plot(np.cumsum(q['biharmonic_terms'][probe,strongest],dtype=np.float64));axes[1].axhline(q['distance_squared'][probe,strongest],ls='--',c='r');axes[1].set_title('Cumulative sum vs KNN distance squared')
        dsq=q['distance_squared'][probe];order=np.argsort(dsq)
        axes[2].plot(np.sqrt(np.maximum(dsq[order],0)),q['distance_weight'][probe,order],'.-');axes[2].set_xlabel('Biharmonic distance');axes[2].set_ylabel('Gaussian weight');axes[2].set_title('All candidate kernels at the same point')
        if 'euclidean' in d:
            for ax,key,caption in zip(axes2[1],['euclidean','distance_squared','distance_weight'],['Euclidean distance','Biharmonic distance','Distance weight']):
                values=d[key][:,0]
                if key=='distance_squared':values=np.sqrt(np.maximum(values,0))
                sc=project(ax,pos,values,cmap='viridis');fig.colorbar(sc,ax=ax,shrink=.6)
                ax.set_title(f'{caption}: representative ID {d["representative_ids"][0]}',fontsize=9)
        else:
            for ax in axes2[1]:ax.axis('off')
        fig.suptitle(label+'\nBottom maps use first tracked representative; independent per-map distance scales.')
        images['distance']=name+'_distance.png';finish(fig,out/images['distance'])
        fig,axes=plt.subplots(1,2,figsize=(9,5))
        loc=state['_kernel_locations'];probe_pos=surface['positions'][probe]
        selected_ids=set(q['candidate_ids'][probe][q['selected_local'][probe]].tolist())
        candidate_ids=set(q['candidate_ids'][probe].tolist())
        nearest=np.argsort(np.linalg.norm(loc-probe_pos,axis=1))[:len(candidate_ids)]
        for ax,title2 in zip(axes,['Biharmonic candidates + selected','Same-count Euclidean neighbours']):
            project(ax,pos,np.full((len(pos),3),.9))
            ax.scatter(loc[:,0],loc[:,1],s=1,c='gray',alpha=.3)
            ax.scatter(probe_pos[0],probe_pos[1],s=100,c='red',marker='*',zorder=5)
            ax.set_title(title2,fontsize=10)
        ci=np.flatnonzero(np.isin(state['stable_ids'],list(candidate_ids)));si=np.flatnonzero(np.isin(state['stable_ids'],list(selected_ids)))
        axes[0].scatter(loc[ci,0],loc[ci,1],s=12,c='blue');axes[0].scatter(loc[si,0],loc[si,1],s=18,c='orange')
        axes[1].scatter(loc[nearest,0],loc[nearest,1],s=12,c='blue')
        fig.suptitle(f'Fixed probe {probe}: all centres projected, including occluded points')
        images['neighbours']=name+'_neighbours.png';finish(fig,out/images['neighbours'])
        fig,axes=plt.subplots(2,2,figsize=(12,7))
        cs=q['colour_contributions'][probe];ids=q['candidate_ids'][probe][q['selected_local'][probe]]
        for ax,c,labelc in zip(axes.flat[:3],range(3),['R','G','B']):
            ax.bar(np.arange(len(ids)),cs[:,c]);ax.axhline(0,c='k',lw=.5);ax.set_title(f'{labelc}: signed contributions after denominator');ax.set_xlabel('selected rank')
        axes[1,1].axis('off')
        axes[1,1].text(0,1,'Mean RGB: '+np.array2string(q['mean_rgb'].ravel(),precision=5)+'\n\nPre-clamp RGB: '+np.array2string(q['preclamp'][probe],precision=5)+'\n\nFinal RGB: '+np.array2string(q['final_rgb'][probe],precision=5)+'\n\nGT RGB: '+np.array2string(surface['gt'][probe],precision=5)+'\n\nDenominator: max(sum(selected alpha), 1)\n\nNumerical decomposition matches original forward.',va='top',fontsize=10)
        fig.suptitle(title+'\n'+label);images['colour']=name+'_colour.png';finish(fig,out/images['colour'])
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        angle=q['grid_angle_axis'];scale=q['grid_anisotropy_axis'];xx,yy=np.meshgrid(angle,np.log1p(scale),indexing='ij')
        axes[0].scatter(xx.ravel(),yy.ravel(),c='lightgray',s=15)
        gi=q['grid_indices'][probe,strongest];gw=q['grid_weights'][probe,strongest]
        axes[0].scatter(xx.ravel()[gi],yy.ravel()[gi],s=40+250*gw,c=gw,cmap='viridis',vmin=0,vmax=1)
        axes[0].scatter(q['angles'][probe,strongest],np.log1p(q['anisotropies'][probe,strongest]),marker='x',c='r',s=70)
        axes[0].set(xlabel='angle (rad)',ylabel='log(1 + anisotropy)',title='Actual four selected grid nodes / query')
        axes[1].bar(range(len(gw)),gw);axes[1].set_xticks(range(len(gi)),[str(x) for x in gi]);axes[1].set(xlabel='flat grid index',ylabel='interpolation weight',title=f'Weight sum: {gw.sum():.8f}')
        fig.suptitle(label);images['grid']=name+'_grid.png';finish(fig,out/images['grid'])
        fig,axes=plt.subplots(2,2,figsize=(10,6))
        for ax,key,span in zip(axes.flat,['_angles','_anisotropies','_thresholds','_sharpnesses'],[(0,np.pi),(1,200),(.3,1),(10,200)]):
            ax.hist(state[key].ravel(),bins=np.linspace(*span,31));ax.set_title(key);ax.set_xlim(*span)
        fig.suptitle(title);images['parameters']=name+'_parameters.png';finish(fig,out/images['parameters'])
        gallery.append(dict(label=title,file=m['file'],images=images))
        print('[PLOT]',name,flush=True)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    regular=[m for m in metrics if m['phase'] in ['initial','post_update','final']]
    axes[0].plot([m['completed_updates'] for m in regular],[m['mse_rgb_mean'] for m in regular],'.-');axes[0].set(title='Fixed uniform evaluation: per-channel MSE',xlabel='completed updates')
    steps=[json.loads(l) for l in (root/'steps.jsonl').read_text().splitlines()]
    axes[1].plot([s['completed_updates'] for s in steps],[s['training_loss_rgb_sum'] for s in steps],lw=.6);axes[1].set(title='Training batch loss: RGB-summed',xlabel='completed updates')
    axes[2].plot([s['completed_updates'] for s in steps],[s['kernels'] for s in steps]);axes[2].set(title='Kernel count',xlabel='completed updates')
    finish(fig,out/'curves.png')
    # Lightweight GIF: fixed point projections, not a new renderer or a training-time task.
    gif_frames=[]
    for item in gallery:
        if any(m['file']==item['file'] and m['phase'] in ['initial','post_update','final'] for m in metrics):
            with Image.open(out/item['images']['overview']) as im:
                frame=im.convert('RGB');frame.thumbnail((900,650));gif_frames.append(frame.copy())
    if gif_frames:gif_frames[0].save(out/'training.gif',save_all=True,append_images=gif_frames[1:],duration=600,loop=0)
    events=json.loads((root/'events.json').read_text()) if (root/'events.json').exists() else []
    tracked=json.loads((analysis/'replay.json').read_text())['roots']
    first=dict(np.load(root/'snapshots'/json.loads((root/'snapshots.json').read_text())[0]['file'],allow_pickle=False))
    roots_of={int(i):int(i) for i in first['stable_ids']};birth={i:0 for i in roots_of};death={};links=[]
    for e in events:
        if e['status']!='completed':continue
        step=e['loss_evaluation']-1
        for child,parent in zip(e.get('child_ids',[]),e.get('parent_ids',[])):
            roots_of[child]=roots_of[parent];birth[child]=step;links.append((parent,child,step))
        if e['kind'] in ['prune','split']:
            for i in e['selected_ids']:death[i]=step
    fig,axes=plt.subplots(len(tracked),1,figsize=(12,max(3,len(tracked)*2)),squeeze=False)
    final_step=max(s['completed_updates'] for s in steps)
    for ax,rid in zip(axes[:,0],tracked):
        family=sorted(i for i,r in roots_of.items() if r==rid);rank={i:j for j,i in enumerate(family)}
        for i in family:
            ax.plot([birth[i],death.get(i,final_step)],[rank[i],rank[i]],lw=1.3)
            if i in death:ax.scatter(death[i],rank[i],c='red',marker='x',s=12)
        for parent,child,step in links:
            if child in rank:ax.plot([step,step],[rank[parent],rank[child]],c='gray',lw=.7)
        ax.set_title(f'Root {rid}: {len(family)} historical members (red x = removed/replaced)',fontsize=9)
        if len(family)<=25:ax.set_yticks(list(rank.values()),list(rank.keys()),fontsize=6)
        else:ax.set_ylabel('family member rank')
        ax.set_xlim(0,max(final_step,1));ax.set_xlabel('completed updates at event')
    finish(fig,out/'lineage.png')
    decisions=[json.loads(l) for l in (root/'decisions.jsonl').read_text().splitlines()] if (root/'decisions.jsonl').exists() else []
    event_html=''.join('<details><summary>'+html.escape(f"{e['kind']} | loss {e['loss_evaluation']} | {len(e['selected_ids'])} selected | {e['status']}")+'</summary><pre>'+html.escape(json.dumps(e,indent=2))+'</pre></details>' for e in events)
    decision_html=html.escape(json.dumps(decisions,indent=2))
    payload=json.dumps(gallery).replace('<','\\u003c')
    page='''<!doctype html><meta charset="utf-8"><title>HKTex training observation</title>
<style>body{font:16px system-ui;max-width:1300px;margin:24px auto;padding:0 20px;background:#fafafa;color:#222}img{max-width:100%;background:white}select{padding:8px;margin:8px}pre{white-space:pre-wrap;font-size:12px}details{margin:8px 0}p{line-height:1.6}</style>
<h1>HKTex 训练回放</h1><p>固定评估点的投影图，不是 PBR 渲染。核场使用跨快照固定色标；代表核沿稳定 root ID 跟踪，分裂后显示最小存活后代 ID。数学面板固定目标点，但默认展示该快照 alpha 最大的选中核，标题显示实际 ID；这不等于 RGB 贡献绝对值最大。所有数值在 analysis/*.npz 中保留。横轴 updates 是优化迭代数；当前配置的扩散时间 t 固定，动画不是随物理时间推进的热扩散模拟。</p>
<select id="frame"></select><select id="mode"></select><p id="label"></p><img id="view"><h2>训练与固定点评估</h2><img src="curves.png"><p>训练 loss 对 RGB 求和，评估 MSE 对 RGB 求平均；两者采样分布也不同，请勿直接比较纵轴。</p>ANIMATION<h2>代表核的家族变化</h2><img src="lineage.png"><h2>ADC 事件与父子 ID</h2>EVENTS<h2>控制器决策及容量</h2><pre>DECISIONS</pre>
<script>const data=PAYLOAD;const frame=document.getElementById('frame'),mode=document.getElementById('mode');data.forEach((d,i)=>frame.add(new Option(d.label,i)));const labels={overview:'总览',kernels:'热核逐阶段响应',spectrum:'谱项与累加',distance:'双调和距离',neighbours:'两种距离选邻居',colour:'颜色贡献',grid:'参数网格插值',parameters:'参数分布'};Object.entries(labels).forEach(([k,v])=>mode.add(new Option(v,k)));function show(){const d=data[frame.value];document.getElementById('label').textContent=d.file;const v=document.getElementById('view');if(d.images[mode.value]){v.src=d.images[mode.value];v.style.display='block';}else{v.style.display='none';document.getElementById('label').textContent+=' — 当前没有存活代表核';}}frame.onchange=show;mode.onchange=show;show();</script>'''
    animation = '<h2>演变动画</h2><img src="training.gif">' if gif_frames else ''
    (out/'index.html').write_text(page.replace('ANIMATION',animation).replace('EVENTS',event_html).replace('DECISIONS',decision_html).replace('PAYLOAD',payload))
    print('[PLOT] Open:',out/'index.html')


if __name__=='__main__':main()
