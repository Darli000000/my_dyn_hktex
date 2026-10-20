#!/usr/bin/env python3
"""Run HKTex's original entry point with observational checks (pinned study version).

Upload to the server and pass --repo /root/hktex-work/hktex.
No repository source edits, no forced density events, no initializer workaround.
CUDA library preloading must be configured by the launching shell.
"""
import argparse
import csv
from datetime import datetime
from functools import wraps
import json
import os
from pathlib import Path
import sys
import uuid


SCRIPT_VERSION = "3.0.0"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--version", action="version", version=f"HKTex checked runner {SCRIPT_VERSION}")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config")
    parser.add_argument("--rendering_config")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--check-tracer", action="store_true",
                        help="Synchronize CUDA and validate geodesic inputs; diagnostic runs only")
    parser.add_argument("--check-cli", action="store_true",
                        help="Validate arguments and repository path, then exit without importing GPU libraries")
    args, extras = parser.parse_known_args(argv)
    for item in extras:
        if item.startswith("-") or "=" not in item or not item.split("=", 1)[0]:
            parser.error(f"Unrecognized argument {item!r}; configuration overrides must be key=value")
    repo = Path(args.repo).resolve()
    if not (repo / "optimisation.py").is_file():
        parser.error("--repo must contain optimisation.py")
    args.config = args.config or str(repo / "configs/texture_hktex_knn.yaml")
    args.rendering_config = args.rendering_config or str(repo / "configs/render.yaml")
    args.repo = str(repo)
    return args, extras


def main():
    args, extras = parse_args()
    print(f"[CHECK] Runner version: {SCRIPT_VERSION}", flush=True)
    print(f"[CHECK] Script: {Path(__file__).resolve()}", flush=True)
    print(f"[CHECK] Tracer checks enabled: {args.check_tracer}", flush=True)
    if args.check_cli:
        print("[CHECK] CLI CHECK: PASS", flush=True)
        print("[CHECK] Config overrides:", json.dumps(extras), flush=True)
        return
    repo = Path(args.repo)
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    os.environ.setdefault("MPLBACKEND", "Agg")

    import torch
    import optimisation
    from hktex.trainers.base import BaseTrainer
    from hktex.trainers.uv_texture import UvTextureTrainer
    from hktex.data.importance_sampling_strategy import ImportanceSamplerDataset
    from hktex.density_controllers.base import BaseDensityController
    from hktex.density_controllers.error_densification import ErrorDensificationController
    from hktex.density_controllers.importance_pruning import ImportancePruningController
    from hktex.modules.tracer import GPUGeodesicTracer

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    report_dir = repo / "outputs/training-checks" / stamp
    report_dir.mkdir(parents=True)
    stats = {
        "status": "running",
        "script_version": SCRIPT_VERSION,
        "script_path": str(Path(__file__).resolve()),
        "check_tracer": args.check_tracer,
        "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
        "config": args.config,
        "overrides": extras,
        "initialisation": [],
        "density_events": [],
        "sampler_batches": {"uniform": 0, "importance": 0},
        "loss_evaluations": 0,
        "tracer_checks": [],
    }
    losses = []
    print(f"[CHECK] Report directory: {report_dir}", flush=True)

    if args.check_tracer:
        original_trace = GPUGeodesicTracer.trace

        @wraps(original_trace)
        def checked_trace(self, coords, face_ids, tangent_vector, **kw):
            event = {"call": len(stats["tracer_checks"]) + 1,
                     "loss_evaluation": len(losses), "stage": "sync_before_trace"}
            stats["tracer_checks"].append(event)
            print(f"[CHECK] TRACE {event['call']}: synchronizing before tracer", flush=True)
            # If this fails, a prior operation failed before the DiGeo call.
            torch.cuda.synchronize(coords.device)
            event["stage"] = "validate_inputs"
            count = coords.shape[0]
            if tuple(coords.shape) != (count, 3) or tuple(tangent_vector.shape) != (count, 3):
                raise ValueError("Unexpected tracer coordinate/direction shape")
            if tuple(face_ids.shape) != (count,):
                raise ValueError("Unexpected tracer face_ids shape")
            if face_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("Tracer face IDs must be integers")
            bad_faces = (face_ids < 0) | (face_ids >= self.mesh.faces.shape[0])
            event["invalid_face_count"] = int(bad_faces.sum().item())
            if event["invalid_face_count"]:
                raise ValueError(f"Out-of-range tracer face IDs: {event}")
            tensors = {"coords": coords, "tangent_vector": tangent_vector}
            if kw.get("transport_vector") is not None:
                tensors["transport_vector"] = kw["transport_vector"]
            for name, tensor in tensors.items():
                if not torch.isfinite(tensor).all().item():
                    event["nonfinite_input"] = name
                    raise FloatingPointError(f"Tracer input {name} contains NaN/Inf")
            tri_ids = self.mesh.faces[face_ids.long()].long()
            bary = kw.get("bary_coords")
            if bary is None:
                bary = self.mesh.cartesian_to_barycentric(coords, tri_ids)
            if tuple(bary.shape) != (count, 3) or not torch.isfinite(bary).all().item():
                raise ValueError("Tracer barycentric coordinates have invalid shape or NaN/Inf")
            event["bary_min"] = float(bary.min().item())
            event["bary_max"] = float(bary.max().item())
            event["bary_sum_max_error"] = float((bary.sum(-1) - 1).abs().max().item())
            reconstructed = (self.mesh.verts[tri_ids] * bary[..., None]).sum(1)
            event["position_bary_max_error"] = float((reconstructed - coords).norm(dim=-1).max().item())
            event["max_step_length"] = float(tangent_vector.norm(dim=-1).max().item())
            event["stage"] = "inputs_checked"
            print("[CHECK] TRACE inputs:", json.dumps(event), flush=True)
            # Absolute tolerance is for HKTex's normalized mesh. These are diagnostic
            # checks, not a repair or a change to the original tracing inputs.
            if (event["bary_min"] < -1e-3 or event["bary_max"] > 1.001
                    or event["bary_sum_max_error"] > 1e-3
                    or event["position_bary_max_error"] > 1e-3):
                raise ValueError("Tracer point/face/barycentric consistency check failed")
            event["stage"] = "inside_original_trace"
            result = original_trace(self, coords, face_ids, tangent_vector, **kw)
            event["stage"] = "sync_after_trace"
            torch.cuda.synchronize(coords.device)
            event["stage"] = "completed"
            return result

        GPUGeodesicTracer.trace = checked_trace

    original_loss = BaseTrainer.compute_loss

    @wraps(original_loss)
    def checked_loss(self, *a, **kw):
        loss, per_point = original_loss(self, *a, **kw)
        if not torch.isfinite(loss).item() or not torch.isfinite(per_point).all().item():
            raise FloatingPointError("Non-finite loss detected before backward")
        value = loss.detach().item()
        losses.append((len(losses), value, int(self.model.N_sources)))
        stats["loss_evaluations"] = len(losses)
        if len(losses) == 1 or len(losses) % 100 == 0:
            print(f"[CHECK] loss_eval={len(losses)} loss={value:.7g} kernels={self.model.N_sources}", flush=True)
        return loss, per_point

    BaseTrainer.compute_loss = checked_loss
    original_init = UvTextureTrainer.data_dependent_initialisation

    @wraps(original_init)
    def checked_init(self, *a, **kw):
        dataset = self.datamodule.train_dataloader().dataset
        event = {"dataset_type": type(dataset).__name__, "has_tex_img": hasattr(dataset, "tex_img")}
        stats["initialisation"].append(event)
        if not event["has_tex_img"]:
            event["result"] = "original_implementation_skips_missing_tex_img"
            print("[CHECK] INIT: original implementation skips this wrapped dataset (no tex_img)", flush=True)
            return original_init(self, *a, **kw)
        locations = self.model._kernel_locations.detach().clone()
        mean_colour = self.model._mean_colour.detach().clone()
        kernel_colours = self.model._kernel_colours.detach().clone()
        result = original_init(self, *a, **kw)
        event.update({
            "location_max_change": float((self.model._kernel_locations.detach() - locations).abs().max().item()),
            "mean_colour_max_change": float((self.model._mean_colour.detach() - mean_colour).abs().max().item()),
            "kernel_colour_max_change": float((self.model._kernel_colours.detach() - kernel_colours).abs().max().item()),
            "result": "returned_normally",
        })
        print("[CHECK] INIT:", json.dumps(event), flush=True)
        return result

    UvTextureTrainer.data_dependent_initialisation = checked_init

    def instrument_density(cls, method_name, label):
        original = getattr(cls, method_name)

        @wraps(original)
        def wrapped(self, *a, **kw):
            before = int(self._model.N_sources)
            result = original(self, *a, **kw)
            after = int(self._model.N_sources)
            event = {"kind": label, "loss_evaluation": len(losses), "before": before, "after": after}
            stats["density_events"].append(event)
            print(f"[CHECK] {label}: kernels {before} -> {after}", flush=True)
            return result

        setattr(cls, method_name, wrapped)

    instrument_density(ErrorDensificationController, "_densify", "densify_check")
    instrument_density(ImportancePruningController, "prune_redundant_kernels", "prune_check")
    instrument_density(BaseDensityController, "clone", "clone")
    instrument_density(BaseDensityController, "split", "split")

    original_iter = ImportanceSamplerDataset.__iter__

    @wraps(original_iter)
    def checked_iter(self):
        for batch in original_iter(self):
            mode = self.current_sampling_method
            stats["sampler_batches"][mode] = stats["sampler_batches"].get(mode, 0) + 1
            if stats["sampler_batches"][mode] == 1:
                print(f"[CHECK] sampler yielded first {mode} batch", flush=True)
            yield batch

    ImportanceSamplerDataset.__iter__ = checked_iter

    try:
        result = optimisation.main(args, extras)
        trainer = result["optimisation"]
        for name, param in trainer.model.named_parameters():
            if not torch.isfinite(param).all().item():
                raise FloatingPointError(f"Non-finite final parameter: {name}")
        stats["final_kernels"] = int(trainer.model.N_sources)
        stats["status"] = "completed"
    except BaseException as exc:
        stats["status"] = "failed"
        stats["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if losses:
            stats["first_loss"] = losses[0][1]
            stats["last_loss"] = losses[-1][1]
        stats["event_counts"] = {
            kind: sum(event["kind"] == kind for event in stats["density_events"])
            for kind in ("densify_check", "prune_check", "clone", "split")
        }
        (report_dir / "report.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        with (report_dir / "loss.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["loss_evaluation_index", "loss", "kernels_before_density_update"])
            writer.writerows(losses)
        print("[CHECK] Status:", stats["status"], flush=True)
        print("[CHECK] Density calls:", stats["event_counts"], flush=True)
        print("[CHECK] Sampler batches:", stats["sampler_batches"], flush=True)
        print("[CHECK] Report:", report_dir / "report.json", flush=True)


if __name__ == "__main__":
    main()
