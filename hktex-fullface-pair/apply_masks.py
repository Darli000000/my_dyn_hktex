#!/usr/bin/env python3
"""Apply precomputed masks to the exact old dataset; no segmentation dependency needed."""
import argparse,hashlib,json,zipfile
from pathlib import Path
from prepare import Source
from common import verify_dataset,inside,write_json

def apply(source,bundle,output):
    out=Path(output).resolve()
    if out.exists():raise ValueError('Use a new output directory')
    src=Source(source)
    with zipfile.ZipFile(bundle) as z:
        manifest=json.loads(z.read('manifest.json'))
        expected=manifest['provenance']['source_manifest_sha256']
        if hashlib.sha256(src.read('manifest.json')).hexdigest()!=expected:
            raise ValueError('Prepared masks belong to another source manifest. Use prepare.py to create masks for your dataset.')
        names=[n for n in z.namelist() if not n.endswith('/')]
        for n in names:inside(out,n)
        out.mkdir(parents=True)
        for row in src.manifest['views']:
            for key in ['rgb','alpha','geometry','preview']:
                target=inside(out,row[key]);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(src.read(row[key]))
            for newkey,oldkey in [('legacy_mask','mask'),('legacy_regions','regions')]:
                newrow=next(r for r in manifest['views'] if r['id']==row['id']);target=inside(out,newrow[newkey]);target.write_bytes(src.read(row[oldkey]))
        for name in names:
            p=inside(out,name);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
    verify_dataset(out)
    print('[DATASET VERIFIED]',out,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--bundle',default=str(Path(__file__).resolve().parent/'prepared_masks.zip'));a=p.parse_args()
    apply(a.source,a.bundle,a.output)
if __name__=='__main__':main()
