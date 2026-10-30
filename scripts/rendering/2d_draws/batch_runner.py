from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import resource
import sys
import time
from pathlib import Path

from hlr_worker import VIEWS, process_part

STEP_EXTS = {".stp", ".step", ".STP", ".STEP", ".stl", ".STL"}


def _worker_entry(step_path: str, out_dir: str, max_mem_gb: float,
                   q: mp.Queue) -> None:
    
    if max_mem_gb and max_mem_gb > 0:
        try:
            _, hard = resource.getrlimit(resource.RLIMIT_AS)
            cap_bytes = int(max_mem_gb * 1024**3)
            if hard != resource.RLIM_INFINITY:
                cap_bytes = min(cap_bytes, hard)
            resource.setrlimit(resource.RLIMIT_AS, (cap_bytes, hard))
        except (ValueError, OSError) as e:
            # Some platforms/containers restrict RLIMIT_AS -- don't let a
            # failure to set the cap block real work, just proceed uncapped.
            print(f"WARNING: could not set memory limit ({e}); "
                  f"proceeding without a cap for this worker", file=sys.stderr)
    try:
        rep = process_part(step_path, out_dir)
        q.put(("ok", rep))
    except BaseException as e:  # noqa: BLE001 — report anything, then die
        q.put(("error", f"{type(e).__name__}: {e}"))


def outputs_exist(out_dir: Path, prefix: str) -> bool:
    return all((out_dir / f"{prefix}_{v}.dxf").is_file() for v in VIEWS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 4),
                    help="parallel worker processes (default: min(cores, 4) "
                         "-- kept conservative since each worker loads a "
                         "full part into memory; combine with --max-mem-gb "
                         "rather than raising this blindly)")
    ap.add_argument("--max-mem-gb", type=float, default=4.0,
                    help="hard memory cap PER WORKER PROCESS in GB (0 = no "
                         "cap). A part that exceeds this fails cleanly with "
                         "a MemoryError logged to failures.csv instead of "
                         "risking a system-wide OOM that can take down the "
                         "whole instance. Size this so "
                         "--workers * --max-mem-gb stays comfortably under "
                         "total system RAM.")
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="hard kill per part, seconds (all 6 views)")
    ap.add_argument("--limit", type=int, default=0,
                    help="process at most N parts (0 = all); handy for pilots")
    ap.add_argument("--filelist", type=Path, default=None,
                    help="text file with one STEP path per line; overrides "
                         "directory scan")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.filelist:
        parts = [Path(line.strip()) for line in
                 args.filelist.read_text().splitlines() if line.strip()]
        missing = [p for p in parts if not p.is_file()]
        if missing:
            print(f"WARNING: {len(missing)} paths in filelist don't exist "
                  f"(first: {missing[0]})")
            parts = [p for p in parts if p.is_file()]
    else:
        parts = sorted(p for p in args.input_dir.iterdir()
                       if p.suffix in STEP_EXTS)
    if args.limit:
        parts = parts[: args.limit]

    todo = [p for p in parts
            if not outputs_exist(args.output_dir, p.stem)]
    print(f"{len(parts)} parts found, {len(parts) - len(todo)} already done, "
          f"{len(todo)} to process, {args.workers} workers, "
          f"{args.max_mem_gb}GB/worker cap, {args.timeout:.0f}s timeout")

    ctx = mp.get_context("spawn")  # clean interpreter per part; no fork baggage
    failures_path = args.output_dir / "failures.csv"
    fail_f = open(failures_path, "a", newline="")
    fail_w = csv.writer(fail_f)
    if fail_f.tell() == 0:
        fail_w.writerow(["part", "reason"])

    running: dict[str, tuple[mp.Process, mp.Queue, float]] = {}
    queue = list(todo)
    done = failed = 0
    t0 = time.time()

    def launch(p: Path) -> None:
        q = ctx.Queue(maxsize=1)
        proc = ctx.Process(target=_worker_entry,
                           args=(str(p), str(args.output_dir),
                                 args.max_mem_gb, q),
                           daemon=True)
        proc.start()
        running[p.stem] = (proc, q, time.time())

    def reap() -> None:
        nonlocal done, failed
        for stem in list(running):
            proc, q, started = running[stem]
            timed_out = (time.time() - started) > args.timeout
            finished = not proc.is_alive()
            if not (finished or timed_out):
                continue
            if timed_out and not finished:
                proc.kill()
                proc.join()
                fail_w.writerow([stem, "timeout"])
                fail_f.flush()
                failed += 1
            else:
                status, payload = ("error", "worker died without report")
                try:
                    status, payload = q.get_nowait()
                except Exception:
                    pass
                if status == "ok":
                    done += 1
                    empty = [v for v, n in payload["views"].items() if n == 0]
                    if empty:
                        fail_w.writerow([stem, f"empty views: {empty}"])
                        fail_f.flush()
                else:
                    fail_w.writerow([stem, payload])
                    fail_f.flush()
                    failed += 1
                proc.join()
            del running[stem]

    while queue or running:
        while queue and len(running) < args.workers:
            launch(queue.pop(0))
        reap()
        time.sleep(0.05)
        n = done + failed
        if n and n % 25 == 0:
            rate = n / (time.time() - t0) * 60
            sys.stdout.write(
                f"\r{n}/{len(todo)} done ({failed} failed), "
                f"{rate:.1f} parts/min "
            )
            sys.stdout.flush()

    fail_f.close()
    dt = time.time() - t0
    print(f"\nFinished: {done} ok, {failed} failed in {dt/60:.1f} min "
          f"({(done+failed)/max(dt,1)*60:.1f} parts/min). "
          f"Failures logged to {failures_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())