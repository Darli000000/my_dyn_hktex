#!/usr/bin/env python3
"""Read-only Linux diagnostics. Does not stop training or import HKTex.
Usage: python collect_eigen_diagnostics.py PID [REPO]
"""
import argparse
import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    try:
        return Path(path).read_text(errors="replace").strip()
    except OSError as exc:
        return f"unavailable: {exc}"


def run(args, timeout=15):
    print("\nCOMMAND:", " ".join(map(str, args)), flush=True)
    try:
        result = subprocess.run(args, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=timeout)
        print(result.stdout, flush=True)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(type(exc).__name__, str(exc), flush=True)


def cpu_ticks(proc):
    # Fields after the parenthesized comm start at field 3 (state).
    raw = (proc / "stat").read_text()
    fields = raw[raw.rfind(")") + 2:].split()
    return int(fields[11]) + int(fields[12])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pid", type=int)
    parser.add_argument("repo", nargs="?", default="/root/hktex-work/hktex")
    args = parser.parse_args()
    proc = Path(f"/proc/{args.pid}")
    if not proc.exists():
        raise SystemExit("PID does not exist in this container; select the training Python PID.")
    print("Diagnostic interpreter:", sys.executable)
    for entry in ("exe", "cwd"):
        try:
            print("Target", entry, os.readlink(proc / entry))
        except OSError as exc:
            print(entry, exc)
    print("Target command:", read(proc / "cmdline").replace("\0", " "))
    print("\nTarget status:")
    keep = ("Name:", "State:", "VmPeak:", "VmSize:", "VmRSS:", "VmSwap:",
            "Threads:", "Cpus_allowed_list:", "Mems_allowed_list:")
    print("\n".join(line for line in read(proc / "status").splitlines()
                    if line.startswith(keep)))
    print("\nTarget thread environment (selected keys only):")
    keys = {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_DYNAMIC",
            "MKL_DYNAMIC", "LD_PRELOAD", "CONDA_PREFIX"}
    for item in read(proc / "environ").split("\0"):
        if item.partition("=")[0] in keys:
            print(item)
    print("\nLoaded math libraries:")
    libs = set()
    for line in read(proc / "maps").splitlines():
        if any(key in line.lower() for key in ("openblas", "mkl", "libgomp", "libiomp", "libstdc++")):
            libs.add(line.split()[-1])
    print("\n".join(sorted(libs)) or "No matching libraries visible")
    print("\nCPU:")
    for line in read("/proc/cpuinfo").splitlines():
        if line.startswith("model name"):
            print(line)
            break
    print("os.cpu_count:", os.cpu_count())
    try:
        print("Target affinity:", sorted(os.sched_getaffinity(args.pid)))
    except (OSError, AttributeError) as exc:
        print(exc)
    print("\nContainer cgroups:", read(proc / "cgroup"))
    # Inspect visible v2 paths and ancestors. Host-only ancestor limits may be hidden.
    paths = {Path("/sys/fs/cgroup")}
    for line in read(proc / "cgroup").splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            candidate = Path("/sys/fs/cgroup") / parts[2].lstrip("/")
            if candidate.is_dir():
                while str(candidate).startswith("/sys/fs/cgroup"):
                    paths.add(candidate)
                    candidate = candidate.parent
    stat_files = []
    for base in sorted(paths):
        for name in ("cpu.max", "cpu.stat", "cpuset.cpus.effective", "memory.max",
                     "memory.current", "memory.swap.current", "memory.events",
                     "cpu.pressure", "memory.pressure", "io.pressure"):
            path = base / name
            if path.exists():
                print(f"\n{path}:\n{read(path)}")
                if name == "cpu.stat":
                    stat_files.append(path)
    for path in ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
                 "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes",
                 "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        if Path(path).exists():
            print(path, read(path))
    run(["free", "-h"])
    print("\nCPU time over 5 seconds (100% = one logical CPU):", flush=True)
    try:
        before = cpu_ticks(proc)
        started = time.monotonic()
        time.sleep(5)
        elapsed = time.monotonic() - started
        percent = 100 * (cpu_ticks(proc) - before) / os.sysconf("SC_CLK_TCK") / elapsed
        print(f"Target CPU usage: {percent:.1f}%")
    except (OSError, ValueError) as exc:
        print("Target sampling failed:", exc)
    for path in stat_files:
        print(f"\n{path} AFTER sample:\n{read(path)}")
    run(["ps", "-L", "-p", str(args.pid), "-o", "pid,tid,stat,pcpu,time,wchan:28", "--sort=-pcpu"])
    run(["vmstat", "1", "3"])
    print("\nPackage metadata (diagnostic interpreter):")
    for name in ("numpy", "scipy", "torch", "trimesh", "robust-laplacian", "threadpoolctl"):
        try:
            print(name, md.version(name))
        except md.PackageNotFoundError:
            print(name, "not found")
    # Fresh process: identifies default backend/threads, not the live target's settings.
    run([sys.executable, "-c", "import json,numpy,scipy.linalg; from threadpoolctl import threadpool_info; print(json.dumps(threadpool_info(),indent=2))"], timeout=20)
    run(["git", "-C", args.repo, "rev-parse", "HEAD"])
    run(["git", "-C", args.repo, "status", "--short"])
    print("\nDone. cgroup counters are group-wide; compare deltas, not just totals.")


if __name__ == "__main__":
    main()
