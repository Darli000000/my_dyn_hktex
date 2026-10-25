"""External RGB supervision using the original HKTex network and PRB ray integrator."""
from pathlib import Path
import json
import math
import time
import numpy as np
import torch
import mitsuba as mi
import drjit as dr
from PIL import Image, ImageDraw
import hktex.utils as utils
import hktex.rendering as rendering
from hktex.trainers.mitsuba_ray_trainer import MitsubaRayTrainer
from image_dataset import pixel_samples
from offline_common import sensor_dict, project, write_json, linear_to_srgb


def color3(rgb):
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    return mi.Color3f(mi.Float(rgb[:, 0]), mi.Float(rgb[:, 1]), mi.Float(rgb[:, 2]))


def cpu_state(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: cpu_state(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [cpu_state(v) for v in obj]
    return obj


class ExternalImageTrainer(MitsubaRayTrainer):
    def data_dependent_initialisation(self, **kwargs):
        raise RuntimeError('UV initialization is forbidden in image training')

    def _render_gt_rays(self, *args, **kwargs):
        raise RuntimeError('Online GT rendering is forbidden: read dataset RGB instead')

    def render_gt(self, *args, **kwargs):
        raise RuntimeError('Read saved reference images instead')

    def setup_images(self, dataset, out, settings):
        self.dataset, self.out, self.settings = dataset, Path(out), settings
        self.rng = np.random.default_rng(settings['seed'])
        self.events = []
        self.update = 0
        self.sensors = {}
        self.shader_calls = 0
        self.gradient_checks = 0
        self.first_update_delta = None
        self.started = time.monotonic()
        # Fixed near-neutral initialization is independent of all GT textures,
        # and avoids a fully black clamped output with vanishing colour gradients.
        with torch.no_grad():
            self.model.model._mean_colour.fill_(.35)
            self.model.model._kernel_colours.normal_(0., .003)
        mi_mesh, self.texture = self.renderer.mesh_to_mitsuba(
            self.datamodule.mesh, self.model,
            bsdf_additional_settings=dict(specular=0., metallic=0., roughness=1.))
        mesh_params = mi.traverse(mi_mesh)
        mesh_params['vertex_normals'] = np.asarray(self.datamodule.saved_normals, np.float32).ravel()
        mesh_params.update()
        # Custom rays determine projection. This tiny dummy film is only used by
        # RBIntegrator.prepare; allocating a 2048 film per minibatch is unnecessary.
        tiny = dict(dataset.views[0], width=1, height=1)
        spec = self.renderer.configure_scene()
        spec['camera'] = sensor_dict(mi, tiny, 1)
        spec['mesh'] = mi_mesh
        self.scene = mi.load_dict(spec)
        self.grad_params = mi.traverse(utils.TraversableDict(
            {'texture': [self.texture, mi.ParamFlags.Differentiable]}))
        dr.enable_grad(self.grad_params['texture.grad_activator'])
        self._instrument_density()

    def _instrument_density(self):
        for dc in self.density_controllers:
            for method in ['clone', 'split', '_remove_kernels']:
                original = getattr(dc, method)
                def wrapped(mask, *args, _fn=original, _name=method, _dc=dc, **kwargs):
                    n0 = self.model.model.N_sources
                    selected = int(mask.sum().item())
                    result = _fn(mask, *args, **kwargs)
                    event = dict(update=self.update+1, controller=type(_dc).__name__,
                                 operation=_name, selected=selected, before=n0,
                                 after=self.model.model.N_sources)
                    self.events.append(event)
                    with open(self.out/'density_events.jsonl', 'a') as f:
                        f.write(json.dumps(event)+'\n')
                    return result
                setattr(dc, method, wrapped)

    def sensor(self, record):
        key = (record['id'], record['width'], record['height'])
        if key not in self.sensors:
            self.sensors[key] = mi.load_dict(sensor_dict(mi, record, 1))
        return self.sensors[key]

    def rays(self, record, pixel_ids, aa, transport, rng):
        positions = pixel_samples(record, pixel_ids, aa, transport, rng)
        pos = mi.Point2f(mi.Float(positions[:, 0]), mi.Float(positions[:, 1]))
        # Positional args work on the user's CUDA Mitsuba bindings.
        ray, weight = self.sensor(record).sample_ray(0., .5, pos, mi.Point2f(.5))
        if not bool(dr.all(dr.abs(weight-mi.Color3f(1.)) < 1e-6, axis=None)):
            raise RuntimeError('Unexpected sensor weights; only the A1 pinhole sensor is supported')
        return mi.RayDifferential3f(ray)

    def geometry_check(self):
        result = []
        # Check the TRAINING scene against saved masks/depth before any optimization.
        for r in self.dataset.views:
            ids = self.dataset.fixed_pixels(r, 32, 131)
            x = (ids % r['width']).astype(np.float32)+.5
            y = (ids // r['width']).astype(np.float32)+.5
            pos = mi.Point2f(mi.Float(x/r['width']), mi.Float(y/r['height']))
            ray, _ = self.sensor(r).sample_ray(0., .5, pos, mi.Point2f(.5))
            hit = self.scene.ray_intersect(ray)
            if not bool(dr.all(hit.is_valid())):
                raise RuntimeError('Saved foreground pixels miss training mesh: '+r['id'])
            xyz = np.asarray(hit.p).T
            uv, z = project(xyz, r)
            error = float(np.max(np.abs(uv-np.stack([x, y], -1))))
            saved = np.load(self.dataset.root/r['geometry'], allow_pickle=False)['depth_cv'].ravel()[ids]
            depth_error = float(np.max(np.abs(z-saved)))
            fids = np.asarray(hit.prim_index)
            if error > .02 or depth_error > 2e-4 or fids.max() >= self.mesh.N_faces:
                raise RuntimeError(f'Training mesh/camera mismatch: {r["id"]}, px={error}, depth={depth_error}')
            result.append(dict(id=r['id'], projection_max_px=error, depth_max_abs=depth_error))
        write_json(self.out/'geometry_check.json', dict(passed=True, views=result))

    def prediction(self, record, ids, aa, transport, seed, with_grad):
        ray = self.rays(record, ids, aa, transport, np.random.default_rng(seed))
        if not bool(dr.any(self.scene.ray_test(ray))):
            if with_grad:
                raise RuntimeError('Training rays miss the mesh despite the foreground mask')
            # hide_emitters=True and no ground: fully empty preview batches are black.
            return mi.Color3f(dr.zeros(mi.Float, len(ids)))
        # render_ray returns one radiance value per supplied ray; reduce in the
        # same [pixel,subpixel,transport] order used to construct those rays.
        self.model.reset()
        self.model.inference_mode(False)
        L = rendering.render_ray(self.scene, params=self.grad_params if with_grad else None,
                                 ray=ray, integrator=self.integrator, spp=aa*transport, seed=seed+1)
        pred = rendering.integrate_ray_samples(L, aa*transport)
        self.shader_calls += 1
        return pred

    def _check_gradients(self):
        norms = {}
        for name, p in self.model.model.named_parameters():
            if p.grad is not None:
                if not bool(torch.isfinite(p.grad).all()):
                    raise FloatingPointError('Nonfinite gradient: '+name)
                norms[name] = float(p.grad.norm().item())
        if self.update == 0 and norms.get('_kernel_colours', 0.) <= 0:
            raise RuntimeError('No nonzero kernel-colour gradient: stop and inspect the Dr.Jit/PyTorch bridge')
        self.gradient_checks += 1
        return norms

    def step_images(self, r, ids, target, seed):
        s = self.settings
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)
        before = self.model.model._kernel_colours.detach().clone() if self.update == 0 else None
        pred = self.prediction(r, ids, s['aa'], s['transport'], seed, True)
        target_dr = color3(target)
        loss = dr.mean(dr.dot(dr.abs(pred-target_dr), mi.Color3f(1/3)))
        # Linear RGB, no log/gamma transform and no post-render clipping.
        if not bool(dr.all(dr.isfinite(loss))):
            raise FloatingPointError('Nonfinite rendered loss')
        colours = np.asarray(pred).T.copy()
        kernel_info = None
        if self.has_density_controllers:
            # Attribute pixel error to its centre hit, as in the author's first-hit
            # ADC approximation. It is not exact attribution of indirect paths.
            x = (ids % r['width']).astype(np.float32)+.5
            y = (ids // r['width']).astype(np.float32)+.5
            ray, _ = self.sensor(r).sample_ray(0., .5,
                mi.Point2f(mi.Float(x/r['width']), mi.Float(y/r['height'])), mi.Point2f(.5))
            hit = self.scene.ray_intersect(ray)
            if not bool(dr.all(hit.is_valid())):
                raise RuntimeError('ADC centre hit invalid')
            pts = hit.p.torch().T.contiguous()
            fids = hit.prim_index.torch().to(torch.int32)
            contributions, idx, weights, kernel_info = self._ray_topk_from_first_hits(pts, fids)
            for dc in self.density_controllers:
                dc.pre_backward_step(step=self.update,
                    rendered_colours=torch.as_tensor(colours, device=self.device),
                    gt_colours=torch.as_tensor(target, device=self.device),
                    kernel_contributions=contributions, topk_kernel_idxs=idx, topk_kernel_contribs=weights)
        dr.backward(loss)
        grad_norms = self._check_gradients()
        if self.has_density_controllers:
            for dc in self.density_controllers:
                dc.refresh_state()
                with torch.no_grad():
                    # Refresh after earlier controllers may have changed kernel IDs/count.
                    self.model.reset()
                    kernel_info = self.model.prepare_kernels()
                    dc.post_backward_step(step=self.update, eigalbo_interp=self.model.eigalbo_interp,
                                          kernel_info=kernel_info, tracer=self.model.tracer)
        # The forward pass caches barycentric coordinates used by GeodesicOpt.
        # Recompute them after possible ADC changes, before advancing positions.
        with torch.no_grad():
            self.model.reset()
            self.model.prepare_kernels()
        for opt in self.optimizers:
            opt.step()
        self._post_optimizer_step()
        for scheduler in self.schedulers:
            scheduler.step()
        if before is not None:
            self.first_update_delta = float((before-self.model.model._kernel_colours).abs().max().item())
            if self.first_update_delta <= 0:
                raise RuntimeError('Optimizer did not update kernel colours')
        for name, p in self.model.model.named_parameters():
            if not bool(torch.isfinite(p).all()):
                raise FloatingPointError('Nonfinite parameter after update: '+name)
        self.model.reset()
        return dict(update=self.update+1, view=r['id'], split=r['split'],
                    loss_l1=float(loss[0]), batch_mse=float(np.mean((colours-target)**2)),
                    kernels=self.model.model.N_sources, gradients=grad_norms,
                    peak_cuda_gb=torch.cuda.max_memory_allocated()/1024**3)

    @torch.no_grad()
    def predict_array(self, r, ids, aa, transport, seed):
        with dr.suspend_grad():
            pred = self.prediction(r, ids, aa, transport, seed, False)
            values = np.asarray(pred).T.copy()
        self.model.reset()
        if not np.isfinite(values).all():
            raise FloatingPointError('Nonfinite evaluation predictions')
        return values

    def evaluate(self, final=False):
        rows = []
        splits = ['train', 'val', 'test'] if final else ['train', 'val']
        for split in splits:
            views = self.dataset.by_split[split]
            # All held-out cameras at final evaluation; cheap fixed subset during training.
            if not final:
                views = views[:2]
            for r in views:
                ids = self.dataset.fixed_pixels(r, self.settings['eval_pixels'], 20261025)
                out = []
                for start in range(0, len(ids), self.settings['batch_pixels']):
                    out.append(self.predict_array(r, ids[start:start+self.settings['batch_pixels']],
                                                  4, 4, 800000+start))
                pred = np.concatenate(out)
                target = self.dataset.target(r, ids)
                mse = float(np.mean((pred-target)**2))
                rows.append(dict(view=r['id'], split=split, pixels=len(ids), mse=mse,
                                 psnr_peak1=-10*math.log10(max(mse, 1e-20)),
                                 mae=float(np.mean(np.abs(pred-target)))))
        result = dict(update=self.update, final=final, views=rows,
                      scope='Fixed masked pixels at original resolution; Monte Carlo estimate, not full-image PSNR',
                      mse={s: float(np.mean([v['mse'] for v in rows if v['split'] == s])) for s in splits})
        write_json(self.out/f'eval_{self.update:06d}{"_final" if final else ""}.json', result)
        print('[EVAL]', result['update'], result['mse'], flush=True)
        return result

    def save_checkpoint(self):
        # Keep float32 positions and complete optimizer/controller state. The
        # author's compressed export is inappropriate for exact restart state.
        state = dict(version=1, update=self.update, settings=self.settings,
                     texture=cpu_state(self.model.model.state_dict()),
                     optimizers=[cpu_state(o.state_dict()) for o in self.optimizers],
                     schedulers=[cpu_state(s.state_dict()) for s in self.schedulers],
                     numpy_rng=self.rng.bit_generator.state, sampler=self.dataset.sampler_state(),
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                     first_update_delta=self.first_update_delta, gradient_checks=self.gradient_checks,
                     density_events=self.events)
        tmp = self.out/'checkpoint.tmp.pt'
        torch.save(state, tmp)
        tmp.replace(self.out/'checkpoint.pt')

    def restore(self, state):
        self.model.model.load_state_dict(state['texture'], strict=True)
        for opt, sd in zip(self.optimizers, state['optimizers'], strict=True):
            opt.load_state_dict(sd)
        for scheduler, sd in zip(self.schedulers, state['schedulers'], strict=True):
            scheduler.load_state_dict(sd)
        self.rng.bit_generator.state = state['numpy_rng']
        self.dataset.load_sampler_state(state['sampler'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        self.update = state['update']
        self.first_update_delta = state['first_update_delta']
        self.gradient_checks = state['gradient_checks']
        self.events = state['density_events']
        self.model.reset()

    def preview(self):
        size = self.settings['preview_size']
        if not size:
            return
        # Render whole low-resolution images only for diagnostics. Training never
        # uses downsampled RGB. Reference is an exact box average when size divides W.
        sheets = []
        for r0 in [self.dataset.by_split['train'][0], self.dataset.by_split['val'][0]]:
            if r0['width'] % size:
                raise ValueError('preview_size must divide original image size')
            r = dict(r0, width=size, height=size)
            img = np.empty((size*size, 3), np.float32)
            for start in range(0, size*size, self.settings['batch_pixels']):
                ids = np.arange(start, min(start+self.settings['batch_pixels'], size*size))
                img[start:start+len(ids)] = self.predict_array(r, ids, 4, 2, 900000+start)
            img = img.reshape(size, size, 3)
            factor = r0['width']//size
            target = np.asarray(self.dataset.image(r0)).reshape(size, factor, size, factor, 3).mean((1, 3))
            np.savez_compressed(self.out/f'preview_{r0["id"]}.npz', prediction=img, target=target)
            row = Image.new('RGB', (size*2, size+24), 'white')
            d = ImageDraw.Draw(row)
            d.text((4, 4), f'{r0["split"]} target', fill='black')
            d.text((size+4, 4), 'prediction', fill='black')
            for j, value in enumerate([target, img]):
                rgb = np.round(np.clip(linear_to_srgb(value), 0, 1)*255).astype('uint8')
                row.paste(Image.fromarray(rgb), (j*size, 24))
            sheets.append(row)
        sheet = Image.new('RGB', (size*2, (size+24)*len(sheets)), 'white')
        for i, row in enumerate(sheets):
            sheet.paste(row, (0, i*(size+24)))
        sheet.save(self.out/'preview.png')

    def train_images(self):
        s = self.settings
        self.geometry_check()
        initial_eval = self.evaluate()
        report = dict(status='running', start_update=self.update, total_steps=s['steps'],
                      initial_eval=initial_eval, settings=s)
        write_json(self.out/'report.json', report)
        self.save_checkpoint()
        try:
            while self.update < s['steps']:
                r, ids, target = self.dataset.train_batch(self.rng, s['batch_pixels'])
                seed = int(self.rng.integers(1, 2**30))
                row = self.step_images(r, ids, target, seed)
                self.update += 1
                with open(self.out/'metrics.jsonl', 'a') as f:
                    f.write(json.dumps(row)+'\n')
                if self.update == 1 or self.update % 10 == 0:
                    print(f'[TRAIN] {self.update}/{s["steps"]} {r["id"]} '
                          f'L1={row["loss_l1"]:.6f} kernels={row["kernels"]}', flush=True)
                if self.update % s['save_every'] == 0:
                    self.save_checkpoint()
                if self.update % s['eval_every'] == 0:
                    self.evaluate()
            self.save_checkpoint()
            final_eval = self.evaluate(final=True)
            # Preserve a full-precision, positions-based parameter file for offline inspection.
            torch.save(cpu_state(self.model.model.state_dict()), self.out/'texture_float32.pt')
            self.preview()
            report.update(status='complete', final_eval=final_eval)
        except BaseException as exc:
            # Do NOT overwrite the last known-good checkpoint with partially updated state.
            report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            report.update(completed_updates=self.update, elapsed_seconds=time.monotonic()-self.started,
                          first_update_colour_max_delta=self.first_update_delta,
                          gradient_checks=self.gradient_checks, shader_calls=self.shader_calls,
                          density_events=self.events, dataset_reads=sorted(self.dataset.read_paths))
            write_json(self.out/'report.json', report)
