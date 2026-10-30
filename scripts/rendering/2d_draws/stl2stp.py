from __future__ import annotations
import argparse
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeSolid,
    BRepBuilderAPI_Sewing,
)
from OCC.Core.gp import gp_Pnt
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.RWStl import rwstl
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCC.Core.TopAbs import TopAbs_SHELL
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods

# Reuse the exact same triangulation-parsing shim already proven to work
# in this environment, rather than re-deriving OCCT-version compatibility
# logic from scratch.
from hlr_worker import _tri_to_arrays  # noqa: E402


def load_stl_arrays(path: str, max_triangles: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Returns (verts[N,3], tris[M,3]) for one STL file.

    If max_triangles > 0 and the mesh exceeds it, decimates first. This
    matters a lot for memory: the B-Rep sewing step below builds one
    TopoDS_Face (with its own wire/edges/vertices) PER TRIANGLE, all held
    in memory at once before sewing. A high-poly scan-derived STL can
    easily need several GB just for topology objects -- decimating to a
    few tens of thousands of triangles is more than enough resolution for
    2D outline/HLR drawings and avoids that blowup entirely.
    """
    tri = rwstl.ReadFile(path)
    if tri is None or tri.NbTriangles() == 0:
        raise RuntimeError(f"STL read failed / empty: {path}")
    verts, tris = _tri_to_arrays(tri)
    verts = np.asarray(verts, dtype=float)
    tris = np.asarray(tris, dtype=np.int64)

    if max_triangles and len(tris) > max_triangles:
        mesh = trimesh.Trimesh(vertices=verts, faces=tris, process=False)
        mesh = mesh.simplify_quadric_decimation(face_count=max_triangles)
        verts = np.asarray(mesh.vertices, dtype=float)
        tris = np.asarray(mesh.faces, dtype=np.int64)

    return verts, tris


def build_sewn_shape(verts: np.ndarray, tris: np.ndarray, sewing_tol: float):
    """Build one B-Rep triangular face per mesh triangle and sew them into
    a shape. Degenerate (near-zero-area) triangles are skipped -- they
    can't form a valid face and would otherwise raise inside OCC."""
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) / 2.0
    area_floor = max(sewing_tol**2, 1e-12)

    sewing = BRepBuilderAPI_Sewing(sewing_tol)
    n_added = n_skipped_degenerate = n_skipped_face_fail = 0

    for idx in range(len(tris)):
        if areas[idx] <= area_floor:
            n_skipped_degenerate += 1
            continue
        p1 = gp_Pnt(*v0[idx])
        p2 = gp_Pnt(*v1[idx])
        p3 = gp_Pnt(*v2[idx])
        poly = BRepBuilderAPI_MakePolygon(p1, p2, p3, True)
        if not poly.IsDone():
            n_skipped_face_fail += 1
            continue
        face_maker = BRepBuilderAPI_MakeFace(poly.Wire())
        if not face_maker.IsDone():
            n_skipped_face_fail += 1
            continue
        sewing.Add(face_maker.Face())
        n_added += 1

    if n_added == 0:
        raise RuntimeError(
            f"no valid triangles to sew (degenerate={n_skipped_degenerate}, "
            f"face_build_failed={n_skipped_face_fail}, total={len(tris)})"
        )

    sewing.Perform()
    shape = sewing.SewedShape()
    if shape is None or shape.IsNull():
        raise RuntimeError("sewing produced an empty/null shape")

    stats = {
        "n_triangles": int(len(tris)),
        "n_faces_added": n_added,
        "n_skipped_degenerate": n_skipped_degenerate,
        "n_skipped_face_fail": n_skipped_face_fail,
    }
    return shape, stats


def try_make_solid(shape):
    """If the sewn result is a single closed shell, upgrade it to a solid.
    Otherwise return the shape unchanged (an open shell/compound is still
    valid STEP and still usable by HLRBRep_Algo)."""
    exp = TopExp_Explorer(shape, TopAbs_SHELL)
    shells = []
    while exp.More():
        shells.append(topods.Shell(exp.Current()))
        exp.Next()
    if len(shells) != 1:
        return shape, False
    try:
        solid_maker = BRepBuilderAPI_MakeSolid(shells[0])
        if solid_maker.IsDone():
            return solid_maker.Solid(), True
    except Exception:  # noqa: BLE001
        pass
    return shape, False


def export_step(shape, out_path: str) -> None:
    """Writes atomically: a killed/timed-out/OOM'd process must never leave
    a partial file sitting at out_path, since --skip-existing and
    downstream readers trust that its mere presence means it's valid."""
    tmp_path = out_path + ".tmp"
    writer = STEPControl_Writer()
    status = writer.Transfer(shape, STEPControl_AsIs)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP transfer failed (status={status})")
    status = writer.Write(tmp_path)
    if status != IFSelect_RetDone:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"STEP write failed (status={status})")
    os.replace(tmp_path, out_path)  # atomic on the same filesystem


def convert_one(stl_path: str, step_path: str, sewing_tol_frac: float,
                 max_triangles: int = 0) -> dict:
    verts, tris = load_stl_arrays(stl_path, max_triangles=max_triangles)
    bbox_min, bbox_max = verts.min(axis=0), verts.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    if diag <= 0:
        raise RuntimeError(f"degenerate bounding box: {stl_path}")
    sewing_tol = max(diag * sewing_tol_frac, 1e-6)

    shape, stats = build_sewn_shape(verts, tris, sewing_tol)
    shape, made_solid = try_make_solid(shape)
    stats["made_solid"] = made_solid
    stats["sewing_tol"] = sewing_tol

    os.makedirs(os.path.dirname(step_path), exist_ok=True)
    export_step(shape, step_path)
    return stats


def _worker_entry(stl_path: str, step_path: str, sewing_tol_frac: float,
                   max_triangles: int, q: mp.Queue) -> None:
    try:
        stats = convert_one(stl_path, step_path, sewing_tol_frac, max_triangles)
        q.put(("ok", stats))
    except BaseException as e:  # noqa: BLE001
        q.put(("error", f"{type(e).__name__}: {e}"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--recursive", action="store_true",
                     help="convert STLs in subfolders too, mirroring "
                          "directory structure in output_dir")
    ap.add_argument("--workers", type=int, default=1,
                     help="parallel processes (default: 1 -- large meshes "
                          "can each use several GB while sewing; raise "
                          "this only once you know your files fit in RAM "
                          "concurrently)")
    ap.add_argument("--timeout", type=float, default=300.0,
                     help="hard kill per file, seconds")
    ap.add_argument("--limit", type=int, default=0,
                     help="convert at most N files (0 = all); use this "
                          "for a first test run")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--sewing-tol-frac", type=float, default=1e-4,
                     help="sewing tolerance as a fraction of bbox diagonal")
    ap.add_argument("--max-triangles", type=int, default=50000,
                     help="decimate meshes above this triangle count before "
                          "building B-Rep faces (0 = never decimate). This "
                          "is the main lever for OOM: each triangle becomes "
                          "its own TopoDS_Face held in memory until sewing "
                          "completes, so high-poly scans can use several GB "
                          "each. 2D outline/HLR drawings don't need scan-"
                          "level density, so decimating is cheap here.")
    args = ap.parse_args()

    pattern = "**/*.stl" if args.recursive else "*.stl"
    stls = sorted(args.input_dir.glob(pattern))
    if not stls:
        print(f"No .stl files found in {args.input_dir} "
              f"(recursive={args.recursive})")
        return 1

    jobs = []
    n_skip = 0
    for stl_path in stls:
        rel = stl_path.relative_to(args.input_dir)
        step_path = args.output_dir / rel.with_suffix(".step")
        if args.skip_existing and step_path.is_file():
            n_skip += 1
            continue
        jobs.append((str(stl_path), str(step_path)))

    if args.limit:
        jobs = jobs[: args.limit]

    print(f"{len(stls)} STL files ({n_skip} skipped) -> converting "
          f"{len(jobs)}, {args.workers} workers, sewing_tol_frac="
          f"{args.sewing_tol_frac}, max_triangles={args.max_triangles or 'unlimited'}")

    ctx = mp.get_context("spawn")
    running: dict[str, tuple] = {}
    queue = list(jobs)
    n_ok = n_fail = 0
    fail_log = []

    def launch(stl_path: str, step_path: str) -> None:
        q = ctx.Queue(maxsize=1)
        proc = ctx.Process(target=_worker_entry,
                            args=(stl_path, step_path, args.sewing_tol_frac,
                                  args.max_triangles, q),
                            daemon=True)
        proc.start()
        running[stl_path] = (proc, q, time.time())

    def reap() -> None:
        nonlocal n_ok, n_fail
        for key in list(running):
            proc, q, started = running[key]
            timed_out = (time.time() - started) > args.timeout
            finished = not proc.is_alive()
            if not (finished or timed_out):
                continue
            if timed_out and not finished:
                proc.kill()
                proc.join()
                fail_log.append((key, "timeout"))
                n_fail += 1
            else:
                status, payload = ("error", "worker died without report")
                try:
                    status, payload = q.get_nowait()
                except Exception:
                    pass
                if status == "ok":
                    n_ok += 1
                else:
                    fail_log.append((key, payload))
                    n_fail += 1
                proc.join()
            del running[key]

    t0 = time.time()
    while queue or running:
        while queue and len(running) < args.workers:
            launch(*queue.pop(0))
        reap()
        time.sleep(0.05)

    for stl_path, reason in fail_log:
        print(f"FAIL {Path(stl_path).name}: {reason}")

    dt = time.time() - t0
    print(f"\nDone: {n_ok} ok, {n_fail} failed, {n_skip} skipped "
          f"in {dt/60:.1f} min")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())