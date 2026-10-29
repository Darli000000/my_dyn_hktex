"""Compare only evaluations made with the exact same common rendering protocol."""
import argparse
import json
from pathlib import Path
from protocol import write_json, digest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--evaluations',nargs='+',required=True,help='label=/absolute/evaluation-directory')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    from PIL import Image,ImageDraw
    out=Path(a.output).resolve()
    if out.exists(): p.error('--output must be new')
    out.mkdir(parents=True)
    records=[]; shared=None; gt_hashes=None
    common_keys=('version','dataset_manifest_sha256','views','size','spp','tile_size','seed',
                 'point_batch','precision','allow_tf32','denoise','metric','renderer','gt')
    for item in a.evaluations:
        label,path=item.split('=',1);root=Path(path)
        protocol=json.loads((root/'common_protocol.json').read_text())
        common={k:protocol[k] for k in common_keys}
        if shared is not None and common!=shared: raise ValueError('Evaluation settings differ: '+label)
        shared=common
        completed=json.loads((root/'completed.json').read_text())
        if len(completed)!=1: raise ValueError('Expected one final checkpoint render')
        folder=root/completed[0]['directory']
        hashes=[digest(folder/f'gt_{i:03d}.npz') for i in range(len(common['views']))]
        # NPZ timestamps may differ. Compare underlying values instead of zip bytes.
        import numpy as np
        gt_values=[np.load(folder/f'gt_{i:03d}.npz',allow_pickle=False)['linear_rgb'] for i in range(len(hashes))]
        if gt_hashes is not None:
            for x,y in zip(gt_hashes,gt_values): np.testing.assert_array_equal(x,y)
        gt_hashes=gt_values
        errors=json.loads((folder/'image_errors.json').read_text())
        if [r['view'] for r in errors]!=common['views']: raise ValueError('View order differs')
        if any(r['raw_mse'] is None or r['valid_pixels']<=0 for r in errors): raise ValueError('Empty foreground')
        run=Path(protocol['run'])
        observer=run/'observer'
        counts=None
        if (observer/'snapshots.json').is_file():
            snaps=json.loads((observer/'snapshots.json').read_text())
            counts=dict(first_recorded=snaps[0]['kernels'],first_recorded_update=snaps[0]['completed_updates'],
                        final=snaps[-1]['kernels'],peak_in_this_recording=max(s['kernels'] for s in snaps))
            if snaps[0]['phase']=='initial': counts['initial']=snaps[0]['kernels']
            else:
                counts['note']='Resumed observer: event totals cover this segment only, not the whole training'
                if (run/'settings.json').is_file():
                    counts['configured_initial']=json.loads((run/'settings.json').read_text()).get('sources')
            events=json.loads((observer/'events.json').read_text()) if (observer/'events.json').is_file() else []
            counts['events']={kind:dict(calls=sum(e['kind']==kind and e['status']=='completed' for e in events),
                selected=sum(len(e['selected_ids']) for e in events if e['kind']==kind and e['status']=='completed'))
                for kind in ['clone','split','prune']}
        row=dict(label=label,evaluation=str(root),field_adapter=protocol['field_adapter'],
            foreground_mse=sum(r['raw_mse']*r['valid_pixels'] for r in errors)/sum(r['valid_pixels'] for r in errors),
            mean_view_mse=sum(r['raw_mse'] for r in errors)/len(errors),views=errors,kernels=counts)
        records.append((row,folder))
    size=shared['size'];height=size+40
    sheet=Image.new('RGB',(size*(len(records)+1),height*len(shared['views'])),'white')
    draw=ImageDraw.Draw(sheet)
    for i,view in enumerate(shared['views']):
        with Image.open(records[0][1]/f'gt_{i:03d}_raw.png') as im: sheet.paste(im,(0,i*height+40))
        draw.text((8,i*height+8),'GT '+view,fill='black')
        for j,(row,folder) in enumerate(records,1):
            with Image.open(folder/f'prediction_{i:03d}_raw.png') as im: sheet.paste(im,(j*size,i*height+40))
            draw.text((j*size+8,i*height+8),f"{row['label']} | MSE {row['views'][i]['raw_mse']:.6g}",fill='black')
    sheet.save(out/'comparison_raw.png')
    write_json(out/'comparison.json',dict(protocol=shared,results=[r for r,_ in records],
        warning='Route comparison, not loss-only ablation. UV/online have source texture access; selected-camera metrics do not equal old 2048px sampled MSE.'))
    print('[COMPARISON]',out,flush=True)


if __name__=='__main__':main()
