"""Guarded, reversible patch for stale split geometry after same-step pruning."""
import argparse
from datetime import datetime
import difflib
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

OLD = '''        kernel_barys = kernel_info["barycentric_coords"][sel]
        kernel_vert_idx = kernel_info["vert_idx"][sel]
        kernel_faces = self._model._kernel_face_ids[sel]
        kernel_thresholds = self._model.thresholds[sel]
        kernel_locations = self._model.kernel_locations[sel]
        kernel_normals = self._mesh.fnorms[kernel_faces]
'''
NEW = '''        # HKTEX_FIX_CURRENT_SPLIT_GEOMETRY_V1
        # A previous controller may have pruned/reordered kernels this step.
        # Reconstruct geometry from the current model, not pre-pruning kernel_info.
        kernel_faces = self._model._kernel_face_ids[sel]
        kernel_thresholds = self._model.thresholds[sel]
        kernel_locations = self._model.kernel_locations[sel]
        kernel_vert_idx = self._mesh.get_face_vertices(kernel_faces)
        kernel_barys = self._mesh.cartesian_to_barycentric(
            kernel_locations, kernel_vert_idx
        )
        kernel_normals = self._mesh.fnorms[kernel_faces]
'''

def patched_text(text):
    if NEW in text:
        return text
    if text.count(OLD) != 1:
        raise ValueError('Expected source block not found exactly once; no files changed. Send the current density_controllers/base.py for review.')
    new = text.replace(OLD, NEW, 1)
    compile(new, 'hktex/density_controllers/base.py', 'exec')
    return new

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',default='/root/hktex-work/hktex')
    p.add_argument('--apply',action='store_true',help='Apply after making a timestamped backup; otherwise show diff only')
    a=p.parse_args()
    target=Path(a.repo).resolve()/'hktex/density_controllers/base.py'
    original=target.read_text()
    try: new=patched_text(original)
    except ValueError as e: p.error(str(e))
    if new==original:
        print('ALREADY PATCHED:',target);return
    print(''.join(difflib.unified_diff(original.splitlines(True),new.splitlines(True),fromfile=str(target),tofile=str(target)+' (patched)')))
    if not a.apply:
        print('DRY RUN ONLY. Add --apply to install this patch.');return
    backup=target.with_name(target.name+'.before-split-fix-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.bak')
    shutil.copy2(target,backup)
    # Refuse to overwrite a concurrent change.
    if target.read_text()!=original:
        raise RuntimeError('Source changed during patch preparation; source was not overwritten.')
    fd,temp=tempfile.mkstemp(prefix=target.name+'.',dir=target.parent)
    try:
        with os.fdopen(fd,'w') as f:f.write(new)
        os.chmod(temp,target.stat().st_mode)
        os.replace(temp,target)
    finally:
        if os.path.exists(temp):os.unlink(temp)
    print('PATCHED:',target)
    print('BACKUP:',backup)
    print('SHA256:',hashlib.sha256(target.read_bytes()).hexdigest())

if __name__=='__main__':main()
