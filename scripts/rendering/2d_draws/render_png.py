from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import re
import sys
from pathlib import Path

import ezdxf
from ezdxf import bbox
from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing.config import (
    BackgroundPolicy,
    ColorPolicy,
    Configuration,
)
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MARGIN = 0.08
DPI = 100
BASE_ID_RE = re.compile(r"^(.*)_(?:front|back|left|right|top|bottom)$")


def base_id(stem: str) -> str:
    m = BASE_ID_RE.match(stem)
    return m.group(1) if m else stem


def _is_finite_box(box) -> bool:
    """True if `box` has data and all extmin/extmax coords are finite."""
    if box.extmin is None or box.extmax is None:
        return False
    coords = (box.extmin.x, box.extmin.y, box.extmax.x, box.extmax.y)
    return all(math.isfinite(c) for c in coords)


def safe_extents(msp):
    """bbox.extents(msp), but robust to individual entities whose bbox
    calculation degenerates to inf/nan (seen with some malformed HATCH,
    SPLINE, and DIMENSION/MTEXT entities from lossy STL->DXF conversion).

    Falls back to computing the bbox from only the entities that produce
    a finite bbox on their own, so one bad entity doesn't poison the whole
    drawing's extents. Raises a diagnostic ValueError naming the offending
    entity types if nothing usable remains.
    """
    full = bbox.extents(msp)
    if _is_finite_box(full):
        return full

    good_entities = []
    bad_types: set[str] = set()
    for e in msp:
        try:
            b = bbox.extents([e])
        except Exception:  # noqa: BLE001
            bad_types.add(e.dxftype())
            continue
        if _is_finite_box(b):
            good_entities.append(e)
        else:
            bad_types.add(e.dxftype())

    if not good_entities:
        if not bad_types:
            # The fallback loop found literally nothing to iterate: the
            # modelspace has zero entities. This is a different failure
            # mode from "entity produced bad bbox" — there's nothing here
            # to render, and no per-entity fix can help.
            doc = msp.doc
            layout_counts = {
                name: sum(1 for _ in doc.layouts.get(name))
                for name in doc.layouts.names()
            }
            block_counts = {
                b.name: sum(1 for _ in b)
                for b in doc.blocks
                if not b.name.startswith("*") and sum(1 for _ in b) > 0
            }
            hint = ""
            if any(c > 0 for n, c in layout_counts.items() if n != msp.name):
                hint = f" Other layouts have entities: {layout_counts}."
            elif block_counts:
                hint = (f" Unreferenced/undrawn block defs have entities: "
                        f"{block_counts}. Check for missing INSERTs in "
                        f"modelspace.")
            else:
                hint = " No entities found anywhere in the document."
            raise ValueError(
                f"empty modelspace: 0 entities in '{msp.name}'.{hint}"
            )
        raise ValueError(
            "no entity produced a finite bounding box; "
            f"offending entity types: {sorted(bad_types)}"
        )

    result = bbox.extents(good_entities)
    if bad_types:
        print(f"  (excluded non-finite-bbox entities: {sorted(bad_types)})",
              file=sys.stderr)
    return result


def add_auto_dimensions(doc) -> None:
    """Overall width/height + one diameter dim per distinct circle radius.

    Deterministic and style-identical across parts, so 'dims present' is the
    only varying factor in the ablation.
    """
    msp = doc.modelspace()
    extents = safe_extents(msp)
    xmin, ymin = extents.extmin.x, extents.extmin.y
    xmax, ymax = extents.extmax.x, extents.extmax.y
    w, h = xmax - xmin, ymax - ymin
    ref = max(w, h)
    offset = ref * 0.12

    if "AUTO" not in doc.dimstyles:
        style = doc.dimstyles.new("AUTO")
        style.dxf.dimtxt = ref * 0.03
        style.dxf.dimasz = ref * 0.02
        style.dxf.dimexo = ref * 0.01
        style.dxf.dimdec = 2
        style.dxf.dimdsep = ord(".")

    msp.add_linear_dim(
        base=(xmin, ymin - offset), p1=(xmin, ymin), p2=(xmax, ymin),
        dimstyle="AUTO",
    ).render()
    msp.add_linear_dim(
        base=(xmax + offset, ymin), p1=(xmax, ymin), p2=(xmax, ymax),
        angle=90, dimstyle="AUTO",
    ).render()

    seen: set[float] = set()
    for circle in msp.query("CIRCLE"):
        r = round(circle.dxf.radius, 2)
        if r in seen:
            continue
        seen.add(r)
        c = circle.dxf.center
        msp.add_diameter_dim(
            center=(c.x, c.y), radius=circle.dxf.radius, angle=45,
            dimstyle="AUTO",
        ).render()


def render(doc, out_path: Path, size_px: int, show_hidden: bool) -> None:
    msp = doc.modelspace()

    if not show_hidden and "HIDDEN" in doc.layers:
        doc.layers.get("HIDDEN").off()

    extents = safe_extents(msp)  # after dims, so nothing gets clipped
    cx = (extents.extmin.x + extents.extmax.x) / 2
    cy = (extents.extmin.y + extents.extmax.y) / 2
    half = max(extents.size.x, extents.size.y) / 2 * (1 + MARGIN)
    if half <= 0 or not math.isfinite(half):
        raise ValueError(f"invalid drawing extents (half={half!r})")

    config = Configuration(
        background_policy=BackgroundPolicy.WHITE,
        color_policy=ColorPolicy.BLACK,
    )

    fig = plt.figure(figsize=(size_px / DPI, size_px / DPI))
    ax = fig.add_axes([0, 0, 1, 1])
    Frontend(RenderContext(doc), MatplotlibBackend(ax), config=config
             ).draw_layout(msp, finalize=True)

    # ezdxf's finalize resizes the figure to the content aspect ratio —
    # this was the root cause of the recurring non-square outputs.
    # Force the square canvas back, then apply framing LAST.
    fig.set_size_inches(size_px / DPI, size_px / DPI, forward=False)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    fig.savefig(out_path, dpi=DPI)  # no bbox_inches: keeps exact square size
    plt.close(fig)


def process_one(job) -> tuple[str, str | None]:
    """Render one DXF (and its dim variant). Runs in a worker process."""
    path_s, png_dir_s, size, with_dims, no_hidden = job
    path = Path(path_s)
    png_dir = Path(png_dir_s)
    try:
        obj_dir = png_dir / base_id(path.stem)  # group views per sample
        out_png = obj_dir / f"{path.stem}.png"
        dim_png = obj_dir / f"{path.stem}_dim.png"
        obj_dir.mkdir(parents=True, exist_ok=True)
        doc = ezdxf.readfile(path)
        render(doc, out_png, size, show_hidden=not no_hidden)
        if with_dims:
            doc_d = ezdxf.readfile(path)  # fresh copy: base stays pristine
            add_auto_dimensions(doc_d)
            render(doc_d, dim_png, size, show_hidden=not no_hidden)
        return (path.name, None)
    except Exception as e:  # noqa: BLE001
        return (path.name, f"{type(e).__name__}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dxf_dir", type=Path)
    ap.add_argument("png_dir", type=Path)
    ap.add_argument("--size", type=int, default=1024,
                    help="output PNG side in px (default 1024)")
    ap.add_argument("--with-dims", action="store_true",
                    help="also emit *_dim.png dimensioned variant")
    ap.add_argument("--no-hidden", action="store_true",
                    help="omit hidden (dashed) lines")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip DXFs whose output PNGs already exist")
    ap.add_argument("--workers", type=int, default=max((mp.cpu_count() or 4) - 1, 1),
                    help="parallel render processes (default: cores - 1)")
    args = ap.parse_args()

    args.png_dir.mkdir(parents=True, exist_ok=True)
    dxfs = sorted(args.dxf_dir.glob("*.dxf"))

    jobs, n_skip = [], 0
    for path in dxfs:
        obj_dir = args.png_dir / base_id(path.stem)
        out_png = obj_dir / f"{path.stem}.png"
        dim_png = obj_dir / f"{path.stem}_dim.png"
        if (args.skip_existing and out_png.exists()
                and (not args.with_dims or dim_png.exists())):
            n_skip += 1
            continue
        jobs.append((str(path), str(args.png_dir), args.size,
                     args.with_dims, args.no_hidden))

    print(f"{len(dxfs)} DXF files ({n_skip} skipped) -> {args.png_dir} "
          f"at {args.size}px, {args.workers} workers")

    n_ok = n_fail = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers, maxtasksperchild=100) as pool:
        for i, (name, err) in enumerate(pool.imap_unordered(process_one, jobs), 1):
            if err is None:
                n_ok += 1
            else:
                n_fail += 1
                print(f"FAIL {name}: {err}")
            if i % 50 == 0:
                print(f"{i}/{len(jobs)} rendered ({n_fail} failed)")

    print(f"done: {n_ok} ok, {n_fail} failed, {n_skip} skipped")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())