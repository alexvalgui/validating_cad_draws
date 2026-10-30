from __future__ import annotations

import math
import os
import sys

import ezdxf
import numpy as np

from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.GCPnts import GCPnts_QuasiUniformDeflection
from OCC.Core.HLRAlgo import HLRAlgo_Projector
from OCC.Core.HLRBRep import (HLRBRep_Algo, HLRBRep_HLRToShape,
                              HLRBRep_PolyAlgo, HLRBRep_PolyHLRToShape)
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.RWStl import rwstl
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.StlAPI import StlAPI_Reader
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Shape, TopoDS_Vertex, topods
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt

# ----------------------------------------------------------------------------
# View conventions — identical direction/xdir pairs to the FreeCAD pipeline,
# so drawings stay consistent with the Blender render axes.
# (direction = line of sight / projection plane normal, xdir = image +X)
# ----------------------------------------------------------------------------
VIEWS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "front":  ((0, -1, 0), (1, 0, 0)),
    "back":   ((0, 1, 0),  (-1, 0, 0)),
    "left":   ((-1, 0, 0), (0, -1, 0)),
    "right":  ((1, 0, 0),  (0, 1, 0)),
    "top":    ((0, 0, 1),  (1, 0, 0)),
    "bottom": ((0, 0, -1), (1, 0, 0)),
}

# Edge sets pulled from HLRBRep_HLRToShape / HLRBRep_PolyHLRToShape (same
# method names on both), mapped to DXF layers.
# (method name, layer, include_by_default)
EDGE_SETS = [
    ("VCompound",         "VISIBLE", True),   # visible sharp edges
    ("OutLineVCompound",  "VISIBLE", True),   # visible silhouettes (curved faces)
    ("Rg1LineVCompound",  "VISIBLE", False),  # visible smooth (tangent) edges — off: matches TechDraw default
    ("HCompound",         "HIDDEN",  True),   # hidden sharp edges
    ("OutLineHCompound",  "HIDDEN",  True),   # hidden silhouettes
]
# Sets that get the dihedral-angle post-filter on mesh input (silhouettes don't).
SHARP_SETS = {"VCompound", "HCompound", "Rg1LineVCompound"}

MESH_EXTS = {".stl"}

# Dihedral angle (degrees) above which a mesh edge counts as "sharp" and is
# drawn. Lower -> more tessellation lines on curved faces; higher -> lost
# feature edges. Only used for STL input.
DEFAULT_SHARP_DEG = 30.0


def is_mesh_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in MESH_EXTS


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_step(path: str):
    """Read a STEP file and return a single (possibly compound) TopoDS_Shape."""
    reader = STEPControl_Reader()
    status = reader.ReadFile(path)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP read failed (status={status}): {path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape is None or shape.IsNull():
        raise RuntimeError(f"STEP produced empty shape: {path}")
    return shape


def load_stl(path: str):
    """Read an STL (ASCII or binary). Returns (TopoDS_Face, Poly_Triangulation).

    Uses RWStl directly so we keep a handle on the triangulation; StlAPI_Reader
    does the same internally but BRep_Tool.Triangulation does not reliably
    hand it back through the Python wrapper.
    """
    tri = rwstl.ReadFile(path)
    if tri is None or tri.NbTriangles() == 0:
        raise RuntimeError(f"STL read failed / empty: {path}")
    face = TopoDS_Face()
    BRep_Builder().MakeFace(face, tri)
    if face.IsNull():
        raise RuntimeError(f"could not build face from STL mesh: {path}")
    return face, tri


def load_shape(path: str):
    """Dispatch on extension: STL -> (mesh face, triangulation), else (STEP shape, None)."""
    if is_mesh_file(path):
        return load_stl(path)
    return load_step(path), None


def shape_diagonal(shape) -> float:
    """Bounding-box diagonal, used to scale the discretization tolerance."""
    box = Bnd_Box()
    brepbndlib.Add(shape, box)  # uses triangulation when no geometry exists
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return math.dist((xmin, ymin, zmin), (xmax, ymax, zmax))


# ----------------------------------------------------------------------------
# Mesh sharp-edge classification (replaces HLRBRep_PolyAlgo.Angle, which the
# pythonocc wrapper does not expose on all versions)
# ----------------------------------------------------------------------------
def _tri_to_arrays(tri, trsf=None, base: int = 0):
    new_api = hasattr(tri, "Node")  # OCCT >= 7.6
    nodes = None if new_api else tri.Nodes()
    verts = []
    for i in range(1, tri.NbNodes() + 1):
        p = tri.Node(i) if new_api else nodes.Value(i)
        if trsf is not None:
            p = p.Transformed(trsf)
        verts.append((p.X(), p.Y(), p.Z()))
    triangles = None if hasattr(tri, "Triangle") else tri.Triangles()
    tris = []
    for i in range(1, tri.NbTriangles() + 1):
        t = tri.Triangle(i) if triangles is None else triangles.Value(i)
        a, b, c = t.Get()
        tris.append((base + a - 1, base + b - 1, base + c - 1))
    return verts, tris


def triangulation_arrays(shape, tri=None) -> tuple[np.ndarray, np.ndarray]:
    """Return (verts[N,3], tris[M,3]). Uses `tri` directly when given (STL
    path); otherwise walks the shape's faces."""
    if tri is not None:
        verts, tris = _tri_to_arrays(tri)
    else:
        verts, tris = [], []
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            face = topods.Face(exp.Current())
            loc = TopLoc_Location()
            t = BRep_Tool.Triangulation(face, loc)
            if t is not None:
                v, f = _tri_to_arrays(t, loc.Transformation(), len(verts))
                verts.extend(v)
                tris.extend(f)
            exp.Next()
    if not tris:
        raise RuntimeError("shape carries no triangulation")
    return np.asarray(verts, dtype=float), np.asarray(tris, dtype=np.int64)


def mesh_sharp_edges(verts: np.ndarray, tris: np.ndarray,
                     sharp_deg: float) -> np.ndarray:
    """Return [K,2] vertex-index pairs for edges to draw: dihedral angle >
    sharp_deg, plus boundary / non-manifold edges (count != 2)."""
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    nrm = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(nrm, axis=1)
    ln[ln == 0] = 1.0
    nrm /= ln[:, None]

    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e.sort(axis=1)
    tri_id = np.tile(np.arange(len(tris)), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, tri_id = e[order], tri_id[order]
    uniq, start, counts = np.unique(e, axis=0, return_index=True,
                                    return_counts=True)

    keep = counts != 2  # boundary or non-manifold: always draw
    two = np.nonzero(counts == 2)[0]
    if len(two):
        t1 = tri_id[start[two]]
        t2 = tri_id[start[two] + 1]
        dots = np.einsum("ij,ij->i", nrm[t1], nrm[t2])
        keep[two] = dots < math.cos(math.radians(sharp_deg))
    return uniq[keep]


def projector_matrix(projector: HLRAlgo_Projector) -> np.ndarray:
    """3x4 world->view matrix matching the frame HLR output points live in."""
    t = projector.Transformation()
    return np.array([[t.Value(r, c) for c in range(1, 5)] for r in range(1, 4)])


class SegmentFilter:
    """Answers 'does this 2D segment lie on one of the reference segments?'
    using a hash grid keyed by the segment's supporting line (theta, rho)."""

    def __init__(self, segs: np.ndarray, tol: float):
        # segs: [K,2,2] array of 2D endpoints
        self.tol = float(tol)
        self.ang_q = 2e-3
        self.segs: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self.cells: dict[tuple[int, int], list[int]] = {}
        for a, b in segs:
            a, b = (float(a[0]), float(a[1])), (float(b[0]), float(b[1]))
            for key in self._keys(a, b):
                self.cells.setdefault(key, []).append(len(self.segs))
            if self._line(a, b) is not None:
                self.segs.append((a, b))

    def _line(self, a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        if L < self.tol:
            return None
        dx, dy = dx / L, dy / L
        if dx < 0 or (dx == 0 and dy < 0):
            dx, dy = -dx, -dy
        theta = math.atan2(dy, dx)          # (-pi/2, pi/2]
        rho = -dy * a[0] + dx * a[1]        # signed distance of line to origin
        return theta, rho

    def _keys(self, a, b):
        ln = self._line(a, b)
        if ln is None:
            return []
        theta, rho = ln
        keys = [(round(theta / self.ang_q), round(rho / self.tol))]
        # near-vertical lines flip sign at theta = +-pi/2; index both.
        if abs(theta) > math.pi / 2 - 2 * self.ang_q:
            th2 = theta - math.pi if theta > 0 else theta + math.pi
            keys.append((round(th2 / self.ang_q), round(-rho / self.tol)))
        return keys

    def _on(self, a, b, p, q) -> bool:
        ux, uy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(ux, uy)
        ux, uy = ux / L, uy / L
        for r in (p, q):
            rx, ry = r[0] - a[0], r[1] - a[1]
            perp = abs(-uy * rx + ux * ry)
            along = ux * rx + uy * ry
            if perp > self.tol or along < -self.tol or along > L + self.tol:
                return False
        return True

    def contains(self, p, q) -> bool:
        seen = set()
        for kt, kr in self._keys(p, q):
            for dt in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    for i in self.cells.get((kt + dt, kr + dr), ()):
                        if i in seen:
                            continue
                        seen.add(i)
                        a, b = self.segs[i]
                        if self._on(a, b, p, q):
                            return True
        return False


def view_sharp_filter(verts: np.ndarray, sharp_edges: np.ndarray,
                      projector: HLRAlgo_Projector, tol: float) -> SegmentFilter:
    M = projector_matrix(projector)
    pts = verts @ M[:, :3].T + M[:, 3]
    segs = pts[sharp_edges][:, :, :2]  # [K,2,2]
    return SegmentFilter(segs, tol)


# ----------------------------------------------------------------------------
# HLR
# ----------------------------------------------------------------------------
def _edge_endpoints(edge) -> list[tuple[float, float]]:
    v0, v1 = TopoDS_Vertex(), TopoDS_Vertex()
    topexp.Vertices(edge, v0, v1)
    pts = []
    for v in (v0, v1):
        if not v.IsNull():
            p = BRep_Tool.Pnt(v)
            pts.append((p.X(), p.Y()))
    return pts


def discretize_edge(edge, deflection: float) -> list[tuple[float, float]]:
    """Approximate one (already projected, planar) edge as a 2D polyline."""
    try:
        curve = BRepAdaptor_Curve(edge)
        disc = GCPnts_QuasiUniformDeflection(
            curve, deflection, curve.FirstParameter(), curve.LastParameter()
        )
    except Exception:  # no usable curve (can happen on mesh-derived edges)
        return _edge_endpoints(edge)
    if not disc.IsDone() or disc.NbPoints() < 2:
        return _edge_endpoints(edge)
    pts = []
    for i in range(1, disc.NbPoints() + 1):
        p = disc.Value(i)
        pts.append((p.X(), p.Y()))  # projection plane = XY; Z ~ 0
    return pts


def make_projector(direction, xdir) -> HLRAlgo_Projector:
    origin = gp_Pnt(0.0, 0.0, 0.0)
    ax2 = gp_Ax2(origin, gp_Dir(*direction), gp_Dir(*xdir))
    return HLRAlgo_Projector(ax2)


def project_view(shape, projector: HLRAlgo_Projector):
    """Exact HLR on B-rep geometry (STEP); returns HLRBRep_HLRToShape."""
    algo = HLRBRep_Algo()
    algo.Add(shape)
    algo.Projector(projector)
    algo.Update()
    algo.Hide()
    return HLRBRep_HLRToShape(algo)


def project_view_mesh(shape, projector: HLRAlgo_Projector):
    """Polygonal HLR on triangulated shapes (STL); returns HLRBRep_PolyHLRToShape.

    Sharp-vs-smooth classification is done afterwards by SegmentFilter, since
    the wrapper may not expose PolyAlgo's angle setter.
    """
    algo = HLRBRep_PolyAlgo()
    algo.Load(shape)
    algo.Projector(projector)
    algo.Update()
    extractor = HLRBRep_PolyHLRToShape()
    extractor.Update(algo)
    return extractor


def write_view_dxf(extractor, out_path: str, deflection: float,
                   include_smooth: bool = False,
                   sharp_filter: SegmentFilter | None = None) -> int:
    """Write one projected view to a DXF with VISIBLE/HIDDEN layers.

    Returns the number of polylines written (0 => empty view, caller may warn).
    """
    doc = ezdxf.new("R2010", setup=True)  # setup=True loads DASHED linetype
    msp = doc.modelspace()
    doc.layers.add("VISIBLE", color=7)
    doc.layers.add("HIDDEN", color=8, linetype="DASHED")

    n_written = 0
    for method, layer, default_on in EDGE_SETS:
        if not default_on and not include_smooth:
            continue
        compound = getattr(extractor, method)()
        if compound is None or compound.IsNull():
            continue
        filt = sharp_filter if (sharp_filter and method in SHARP_SETS) else None
        exp = TopExp_Explorer(compound, TopAbs_EDGE)
        while exp.More():
            edge = topods.Edge(exp.Current())
            pts = discretize_edge(edge, deflection)
            if len(pts) >= 2 and (filt is None or filt.contains(pts[0], pts[-1])):
                msp.add_lwpolyline(pts, dxfattribs={"layer": layer})
                n_written += 1
            exp.Next()

    doc.saveas(out_path)
    return n_written


def process_part(step_path: str, out_dir: str, prefix: str | None = None,
                 hidden_lines: bool = True,
                 sharp_deg: float = DEFAULT_SHARP_DEG) -> dict:
    """Full pipeline for one STEP or STL file: 6 views -> 6 DXFs.

    Returns a small report dict (used by batch_runner for logging).
    Raises on unreadable/empty input; individual empty views only warn.
    """
    os.makedirs(out_dir, exist_ok=True)
    if prefix is None:
        prefix = os.path.splitext(os.path.basename(step_path))[0]

    mesh = is_mesh_file(step_path)
    shape, tri = load_shape(step_path)
    diag = shape_diagonal(shape)
    if diag <= 0:
        raise RuntimeError(f"Degenerate bounding box: {step_path}")
    # Chord deflection for curve tessellation: 0.05% of part diagonal, floored.
    deflection = max(diag * 5e-4, 1e-4)

    verts = sharp = None
    if mesh:
        verts, tris = triangulation_arrays(shape, tri)
        sharp = mesh_sharp_edges(verts, tris, sharp_deg)

    report = {"part": prefix, "diag": round(diag, 3), "mesh": mesh,
              "views": {}}
    if mesh:
        report["sharp_edges"] = int(len(sharp))
    for name, (direction, xdir) in VIEWS.items():
        projector = make_projector(direction, xdir)
        sharp_filter = None
        if mesh:
            extractor = project_view_mesh(shape, projector)
            sharp_filter = view_sharp_filter(verts, sharp, projector,
                                             tol=max(diag * 2e-4, 1e-5))
        else:
            extractor = project_view(shape, projector)
        out_path = os.path.join(out_dir, f"{prefix}_{name}.dxf")
        n = write_view_dxf(extractor, out_path, deflection,
                           sharp_filter=sharp_filter)
        if not hidden_lines:
            # Cheap post-filter alternative exists, but preferring explicit
            # generation keeps clean/dimmed variants byte-identical; hidden
            # lines are simply skipped at render time instead (see render_png).
            pass
        report["views"][name] = n
    return report


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("usage: python hlr_worker.py <part.stp|part.stl> <out_dir> [sharp_deg]")
        sys.exit(2)
    sharp = float(sys.argv[3]) if len(sys.argv) == 4 else DEFAULT_SHARP_DEG
    rep = process_part(sys.argv[1], sys.argv[2], sharp_deg=sharp)
    print(rep)