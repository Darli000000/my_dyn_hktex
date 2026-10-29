#!/usr/bin/env python3
"""Resume adapter for paired training; preserve the original run_arm/source hashes."""
import hashlib
import json
from pathlib import Path

VERSION = 'geodesic-face-reference-v1'


def rebind_geodesic_faces(trainer):
    import torch
    model = trainer.model.model
    location = model._kernel_locations
    faces = model._kernel_face_ids
    if faces.device != location.device or faces.device != trainer.mesh.verts.device:
        raise RuntimeError('Model locations, face IDs and mesh must share a device')
    records = []
    pending = []
    for oi, opt in enumerate(trainer.optimizers):
        for gi, group in enumerate(opt.param_groups):
            if 'face_ids' not in group:
                continue
            params, loaded = group['params'], group['face_ids']
            if len(params) != 1 or params[0] is not location or len(loaded) != 1:
                raise RuntimeError('Unexpected GeodesicOpt parameter group; refusing automatic rebinding')
            old = loaded[0]
            if not torch.is_tensor(old) or old.shape != faces.shape or old.dtype != faces.dtype:
                raise RuntimeError('Optimizer/model face IDs have incompatible shape or dtype')
            if not torch.equal(old.detach().cpu(), faces.detach().cpu()):
                raise RuntimeError('Optimizer/model face IDs differ in checkpoint; investigate before resuming')
            records.append(dict(optimizer=oi, group=gi, old_device=str(old.device),
                                new_device=str(faces.device), values_equal=True,
                                old_was_model_buffer=old is faces))
            pending.append(group)
    if not pending:
        raise RuntimeError('Expected a GeodesicOpt face_ids group, found none')
    # Rebind to the live buffer, NOT a device copy. trace_ mutates this buffer.
    for group in pending:
        group['face_ids'] = [faces]
        assert group['face_ids'][0] is model._kernel_face_ids
    return records


def main():
    import run_arm
    original_install = run_arm.install

    def install(module, trainer_module, args, code_hash):
        original_install(module, trainer_module, args, code_hash)
        cls = trainer_module.ExternalImageTrainer
        original_restore = cls.restore

        def restore(self, state):
            original_restore(self, state)
            rows = rebind_geodesic_faces(self)
            record = dict(version=VERSION, restored_update=self.update, groups=rows,
                          adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            self.settings['resume_face_reference_fix'] = record
            (self.out/'resume_repair.json').write_text(json.dumps(record, indent=2)+'\n')
            (self.out/'settings.json').write_text(json.dumps(self.settings, indent=2)+'\n')
            for row in rows:
                print(f"[RESUME FIX] update={self.update} face_ids "
                      f"{row['old_device']} -> live model buffer on {row['new_device']}; "
                      "checkpoint values verified", flush=True)
        cls.restore = restore

    run_arm.install = install
    run_arm.main()


if __name__ == '__main__':
    main()
