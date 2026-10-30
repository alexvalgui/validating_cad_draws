import sys
import json
import shutil
import os
import subprocess
import tempfile
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np

# --- Environment ---

CONDA_PREFIX = "/home/ec2-user/miniconda3/envs/rendering"
SITE_PACKAGES = f"{CONDA_PREFIX}/lib/python3.10/site-packages"
ADD_CONDA_SITE_PACKAGES = False
if ADD_CONDA_SITE_PACKAGES and SITE_PACKAGES not in sys.path:
    sys.path.append(SITE_PACKAGES)

MESH_DIR = Path("/home/ec2-user/cad-2d-vlm-accv/data/medical/meshes/to_render")
RENDER_DIR = Path("/home/ec2-user/cad-2d-vlm-accv/scripts/rendering/blender")
OUTPUT_DIR = Path("/home/ec2-user/cad-2d-vlm-accv/data/medical/renders")
MESH_EXTS = (".stl", ".obj")

# Resolved once in the manager process. Order: env var, PATH, known install.
BLENDER_BIN = (os.environ.get("BLENDER_BIN")
               or shutil.which("blender")
               or "/home/ec2-user/blender-4.2.0-linux-x64/blender")

# --- Render config ---
RESOLUTION = 512
CAM_DIS = 1.5           # fallback only; the real distance is fit per mesh
FOV = np.radians(45.0)
FRAME_FILL = 0.90       # fraction of the frame the part should span

FRONT_AXIS = '+x'
_AZIMUTH_OFFSETS = {'+x': 0.0, '+y': np.pi / 2, '-x': np.pi, '-y': -np.pi / 2}
AZIMUTH_OFFSET = _AZIMUTH_OFFSETS[FRONT_AXIS]

USE_PCA_ALIGNMENT = False
FORCE_RERENDER = False
RECALC_NORMALS = True

# ----------------------------------------------------------------------------
# Lighting
# ----------------------------------------------------------------------------
# Each sun's direction is given in the CAMERA basis: (image +X, image up,
# toward the viewer). The rig is rotated into world space per view, so the
# shading of a given surface orientation is identical across all 6 views.
#
#   key  : upper-right, slightly behind the camera -> primary form
#   fill : opposite side, weaker, flattens the shadow side
#   rim  : below/behind -> separates the silhouette, catches bottom faces
#
# Energies are in W/m^2 (irradiance), independent of distance.
LIGHT_RIG = {
    "Key":  ((0.45,  0.55,  1.00), 2.0),
    "Fill": ((-0.80, 0.15,  0.70), 0.7),
    "Rim":  ((0.10, -0.70, -0.50), 0.5),
}
LIGHT_NAMES = tuple(LIGHT_RIG.keys())
SUN_ANGLE_DEG = 12.0     # softer shadow terminator than a point sun
WORLD_STRENGTH = 0.30    # ambient floor; nothing renders pure black

# --- Surface ---
OVERRIDE_MATERIAL = True          # STL has no materials; set False for OBJ+MTL
BASE_COLOR = (0.72, 0.72, 0.75, 1.0)
ROUGHNESS = 0.65
SPECULAR = 0.25

# --- Diagnostics ---
# CHECK_EXPOSURE: tone stats of the first rendered view per mesh.
# CHECK_ORIENTATION: hard-asserts the camera basis matches hlr_worker per view
#   and prints the normalized bbox so it can be compared to the STEP Bnd_Box.
#   Cheap; leave on.
CHECK_EXPOSURE = False
CHECK_ORIENTATION = True

# --- Speed knobs ---
ENGINE = 'CYCLES'
ENGINE_FALLBACKS = ['BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE', 'CYCLES']

CYCLES_SAMPLES = 16
CYCLES_DENOISE = True
EEVEE_SAMPLES = 32

USE_GPU = True
CHUNK_SIZE = 50
NUM_WORKERS = 2 if USE_GPU else max(1, cpu_count() // 8)
CPU_THREADS_PER_WORKER = max(1, cpu_count() // NUM_WORKERS)
PER_MESH_TIMEOUT = 30


VIEWS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "front":  ((0, -1, 0), (1, 0, 0)),
    "back":   ((0, 1, 0),  (-1, 0, 0)),
    "left":   ((-1, 0, 0), (0, -1, 0)),
    "right":  ((1, 0, 0),  (0, 1, 0)),
    "top":    ((0, 0, 1),  (1, 0, 0)),
    "bottom": ((0, 0, -1), (1, 0, 0)),
}


def _normalize(vec):
    v = np.asarray(vec, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _rotate_z(vec, angle):
    x, y, z = vec
    c, s = np.cos(angle), np.sin(angle)
    return np.array([x * c - y * s, x * s + y * c, z], dtype=np.float64)


def view_basis(direction, xdir):
    """Return (d, x, up) as normalized numpy vectors for a view, azimuth applied."""
    d = _normalize(_rotate_z(direction, AZIMUTH_OFFSET))
    x = _normalize(_rotate_z(xdir, AZIMUTH_OFFSET))
    up = _normalize(np.cross(d, x))
    return d, x, up


def orient_camera(cam, d, x, up, cam_dis):
    """Set the camera to match an hlr_worker view exactly.

    Blender cameras look along local -Z with local +Y up, so the world matrix
    columns are (image +X, image up, toward viewer). This only holds if the
    camera has NO constraints and NO parent -- see create_camera().
    """
    from mathutils import Matrix

    loc = d * cam_dis
    cam.matrix_world = Matrix((
        (x[0], up[0], d[0], loc[0]),
        (x[1], up[1], d[1], loc[1]),
        (x[2], up[2], d[2], loc[2]),
        (0.0,  0.0,   0.0,  1.0),
    ))


def check_camera_basis(cam, name, d, x, up):
    """Fail loudly if anything (constraint, parent, driver) altered the pose."""
    mw = np.array(cam.matrix_world, dtype=np.float64)
    got_x, got_up, got_d = mw[:3, 0], mw[:3, 1], mw[:3, 2]
    for label, got, want in (("X", got_x, x), ("up", got_up, up), ("Z", got_d, d)):
        if not np.allclose(got, want, atol=1e-4):
            raise RuntimeError(
                f"view '{name}': camera {label} axis is {np.round(got, 4).tolist()}, "
                f"expected {np.round(want, 4).tolist()}. The camera pose was "
                f"overridden (constraints={[c.type for c in cam.constraints]}, "
                f"parent={cam.parent})."
            )


# ----------------------------------------------------------------------------
# Bounds, centering, camera fit
# ----------------------------------------------------------------------------

def get_scene_bounds():
    import bpy
    from mathutils import Vector

    mins = Vector((np.inf, np.inf, np.inf))
    maxs = Vector((-np.inf, -np.inf, -np.inf))
    found = False

    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        found = True
        for corner in obj.bound_box:
            wc = obj.matrix_world @ Vector(corner)
            mins.x = min(mins.x, wc.x); mins.y = min(mins.y, wc.y); mins.z = min(mins.z, wc.z)
            maxs.x = max(maxs.x, wc.x); maxs.y = max(maxs.y, wc.y); maxs.z = max(maxs.z, wc.z)

    if not found:
        return None, None, None

    size = maxs - mins
    return mins, maxs, max(size.x, size.y, size.z)


def mesh_roots():
    """Top-level parents of every mesh object in the scene."""
    import bpy

    roots = {}
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        root = obj
        while root.parent is not None:
            root = root.parent
        roots[root.name] = root
    return list(roots.values())


def normalize_scene_local():
    """Uniform scale so the longest bbox side == 1, then center on the origin.

    Replaces render.normalize_scene(). This version is guaranteed to apply NO
    rotation, so the mesh stays in the STEP/STL frame that hlr_worker sees.
    Returns the pre-normalization (mins, maxs) for diagnostics.
    """
    import bpy
    from mathutils import Matrix

    mins, maxs, max_dim = get_scene_bounds()
    if mins is None or max_dim <= 0:
        return None, None

    raw = (mins.copy(), maxs.copy())
    s = 1.0 / max_dim
    S = Matrix.Scale(s, 4)
    for root in mesh_roots():
        root.matrix_world = S @ root.matrix_world
    bpy.context.view_layer.update()

    mins, maxs, _ = get_scene_bounds()
    center = (mins + maxs) / 2.0
    for root in mesh_roots():
        root.matrix_world.translation -= center
    bpy.context.view_layer.update()
    return raw


def compute_fit_cam_dis():
    """Smallest camera distance that keeps the bbox inside FRAME_FILL of frame.

    Solved exactly per bbox corner per view. A single distance is used for all
    6 views so the renders stay scale-consistent with the HLR DXFs.
    """
    mins, maxs, _ = get_scene_bounds()
    if mins is None:
        return CAM_DIS

    corners = np.array([[x, y, z]
                        for x in (mins.x, maxs.x)
                        for y in (mins.y, maxs.y)
                        for z in (mins.z, maxs.z)], dtype=np.float64)

    half_tan = np.tan(FOV / 2.0) * FRAME_FILL
    if half_tan <= 0:
        return CAM_DIS

    required = CAM_DIS
    for direction, xdir in VIEWS.values():
        d, x, up = view_basis(direction, xdir)
        u = corners @ x
        v = corners @ up
        depth_off = corners @ d
        proj = np.maximum(np.abs(u), np.abs(v))
        required = max(required, float(np.max(proj / half_tan + depth_off)))

    return required * 1.02


def pca_align_meshes():
    import bpy
    from mathutils import Matrix

    pts = []
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        mw = np.array(obj.matrix_world)
        n = len(obj.data.vertices)
        co = np.empty(n * 3, dtype=np.float64)
        obj.data.vertices.foreach_get('co', co)
        co = co.reshape(n, 3)
        co = co @ mw[:3, :3].T + mw[:3, 3]
        pts.append(co)

    if not pts:
        return
    pts = np.concatenate(pts, axis=0)
    centered = pts - pts.mean(axis=0)

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    R = eigvecs[:, order]
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1

    R_align = Matrix(R.T.tolist()).to_4x4()
    for root in mesh_roots():
        root.matrix_world = R_align @ root.matrix_world

    bpy.context.view_layer.update()


# ----------------------------------------------------------------------------
# Lighting, world, material
# ----------------------------------------------------------------------------

def create_world(strength=WORLD_STRENGTH):
    """Ambient floor. film_transparent keeps the PNG background at alpha = 0,
    but the world still contributes light to the part."""
    import bpy

    world = bpy.data.worlds.get("RenderWorld") or bpy.data.worlds.new("RenderWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
        bg.inputs[1].default_value = strength
    return world


def create_studio_lights():
    """Create the 3 suns once per chunk. Orientation is set per view."""
    import bpy

    lights = {}
    for name, (_, energy) in LIGHT_RIG.items():
        obj = bpy.data.objects.get(name)
        if obj is None:
            data = bpy.data.lights.new(name=name, type='SUN')
            obj = bpy.data.objects.new(name=name, object_data=data)
            bpy.context.collection.objects.link(obj)
        obj.data.type = 'SUN'
        obj.data.energy = energy
        obj.data.angle = np.radians(SUN_ANGLE_DEG)
        lights[name] = obj
    return lights


def orient_lights_to_view(lights, d, x, up):
    """Rotate the rig into the current camera frame.

    A sun emits along its local -Z, so aiming +Z at `world_dir` (the direction
    from the origin toward the lamp) makes it shine back at the part.
    """
    from mathutils import Vector

    for name, obj in lights.items():
        cx, cy, cz = LIGHT_RIG[name][0]
        world_dir = _normalize(x * cx + up * cy + d * cz)
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion = Vector(world_dir.tolist()).to_track_quat('Z', 'Y')
        obj.location = tuple(world_dir * 10.0)


def get_cad_material():
    import bpy

    mat = bpy.data.materials.get("CADMatte")
    if mat is not None:
        return mat

    mat = bpy.data.materials.new("CADMatte")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = BASE_COLOR
        bsdf.inputs["Roughness"].default_value = ROUGHNESS
        bsdf.inputs["Metallic"].default_value = 0.0
        for spec in ("Specular IOR Level", "Specular"):   # renamed in 4.x
            if spec in bsdf.inputs:
                bsdf.inputs[spec].default_value = SPECULAR
                break
    return mat


def assign_material_to_meshes():
    import bpy

    mat = get_cad_material()
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        obj.data.materials.clear()
        obj.data.materials.append(mat)


def recalc_normals():
    """CAD STL exports are frequently inconsistently wound, which shades
    backfaces black in Cycles and compounds any lighting problem."""
    import bpy
    import bmesh

    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH' or not obj.data.polygons:
            continue
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(obj.data)
        bm.free()
        obj.data.update()


# ----------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------

def create_camera():
    """Create a bare camera: no TRACK_TO constraint, no parent, no drivers.

    Objaverse-style init_camera() helpers attach a TRACK_TO constraint aimed at
    an empty at the origin. That constraint re-derives the orientation on every
    depsgraph update and silently overrides matrix_world. For the four side
    views the constraint's world-Z up-hint coincides with our `up`, so they
    look right; for top/bottom the up-hint is degenerate and Blender picks an
    arbitrary roll -- which is exactly how a "top" render ends up rotated
    relative to the HLR DXF.
    """
    import bpy

    # Remove any camera that init_scene()/factory settings left behind.
    for obj in list(bpy.context.scene.objects):
        if obj.type == 'CAMERA' or (obj.type == 'EMPTY' and obj.name.lower().startswith("empty")):
            bpy.data.objects.remove(obj, do_unlink=True)

    cam_data = bpy.data.cameras.new("ViewCam")
    cam = bpy.data.objects.new("ViewCam", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    cam.data.type = 'PERSP'
    cam.data.sensor_fit = 'HORIZONTAL'
    cam.data.sensor_width = 32.0
    cam.data.lens = 16 / np.tan(FOV / 2)
    cam.data.clip_start = 0.01
    cam.data.clip_end = 100.0
    return cam


def get_transform_matrix(cam):
    """4x4 camera-to-world as nested lists (same layout render.py used)."""
    return [list(row) for row in cam.matrix_world]


# ----------------------------------------------------------------------------
# Engine setup
# ----------------------------------------------------------------------------

def enable_gpu_cycles():
    """Point Cycles at the fastest GPU backend. Returns the backend name or None.

    NOT cached: read_factory_settings() resets preferences (including
    compute_device_type).
    """
    import bpy

    prefs = bpy.context.preferences.addons['cycles'].preferences
    for backend in ('OPTIX', 'CUDA', 'HIP', 'METAL'):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        prefs.get_devices()
        gpu_devices = [d for d in prefs.devices if d.type != 'CPU']
        if gpu_devices:
            for d in prefs.devices:
                d.use = (d.type != 'CPU')
            bpy.context.scene.cycles.device = 'GPU'
            print(f"[GPU] using {backend}: {[d.name for d in gpu_devices]}", flush=True)
            return backend

    print("[GPU] no compute device found, falling back to CPU", flush=True)
    bpy.context.scene.cycles.device = 'CPU'
    return None


def select_engine():
    import bpy

    scene = bpy.context.scene
    for eng in [ENGINE] + ENGINE_FALLBACKS:
        try:
            scene.render.engine = eng
            return eng
        except TypeError:
            continue
    raise RuntimeError("No usable render engine")


def setup_color_management():
    import bpy

    vs = bpy.context.scene.view_settings
    try:
        vs.view_transform = 'Standard'
    except TypeError:
        pass
    try:
        vs.look = 'None'
    except TypeError:
        pass
    vs.exposure = 0.0
    vs.gamma = 1.0


def setup_scene_once():
    """Build scene, camera, lights, world and engine config. Once per chunk."""
    import bpy
    sys.path.insert(0, str(RENDER_DIR))
    from render import init_render, init_scene

    bpy.ops.wm.read_factory_settings(use_empty=True)
    # geo_mode=True clamps samples to 1 and zeroes light bounces -- fine for
    # depth/normal passes, wrong for RGB.
    init_render(engine='CYCLES', resolution=RESOLUTION, geo_mode=False)
    engine = select_engine()

    scene = bpy.context.scene
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.use_persistent_data = True

    if engine == 'CYCLES':
        scene.cycles.use_adaptive_sampling = False
        scene.cycles.samples = CYCLES_SAMPLES
        scene.cycles.use_denoising = CYCLES_DENOISE
        backend = enable_gpu_cycles() if USE_GPU else None
        if backend is None:
            scene.cycles.device = 'CPU'
            scene.render.threads_mode = 'FIXED'
            scene.render.threads = CPU_THREADS_PER_WORKER
            scene.cycles.denoiser = 'OPENIMAGEDENOISE'
        else:
            scene.cycles.denoiser = 'OPTIX' if backend == 'OPTIX' else 'OPENIMAGEDENOISE'
    elif engine.startswith('BLENDER_EEVEE'):
        scene.eevee.taa_render_samples = EEVEE_SAMPLES
    print(f"[ENGINE] {engine}", flush=True)

    init_scene()
    create_world()
    setup_color_management()

    cam = create_camera()
    lights = create_studio_lights()
    return cam, lights


def clear_mesh_objects(keep):
    """Remove everything from the previous mesh (objects + orphan data)."""
    import bpy

    keep_names = {o.name for o in keep}
    for obj in list(bpy.context.scene.objects):
        if obj.name in keep_names:
            continue
        bpy.data.objects.remove(obj, do_unlink=True)

    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for mat in list(bpy.data.materials):
        if mat.users == 0 and mat.name != "CADMatte":
            bpy.data.materials.remove(mat)


def import_mesh(mesh_path, load_object):
    """Import with identity axes so the mesh stays in the STL/STEP frame.

    Blender's OBJ importer defaults to forward=-Z / up=Y, which rotates the
    part 90 deg about X and breaks agreement with hlr_worker. Both importers
    are pinned to forward=Y / up=Z (identity) here.
    """
    import bpy

    ext = mesh_path.suffix.lower()
    if ext == ".stl":
        try:
            bpy.ops.wm.stl_import(filepath=str(mesh_path),
                                  forward_axis='Y', up_axis='Z')
        except (AttributeError, RuntimeError, TypeError):
            bpy.ops.import_mesh.stl(filepath=str(mesh_path))
    elif ext == ".obj":
        try:
            bpy.ops.wm.obj_import(filepath=str(mesh_path),
                                  forward_axis='Y', up_axis='Z')
        except (AttributeError, RuntimeError, TypeError):
            load_object(str(mesh_path))
    else:
        load_object(str(mesh_path))


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------

def report_exposure(png_path, label):
    import bpy

    try:
        img = bpy.data.images.load(str(png_path))
        px = np.array(img.pixels[:], dtype=np.float32).reshape(-1, 4)
        bpy.data.images.remove(img)
    except Exception as e:
        print(f"[EXPOSURE] {label}: unreadable ({e})", flush=True)
        return

    fg = px[px[:, 3] > 0.5][:, :3]
    if fg.size == 0:
        print(f"[EXPOSURE] {label}: empty alpha -- nothing rendered", flush=True)
        return

    clipped_hi = float((fg > 0.99).mean())
    clipped_lo = float((fg < 0.01).mean())
    print(
        f"[EXPOSURE] {label}: min={fg.min():.3f} mean={fg.mean():.3f} "
        f"max={fg.max():.3f} white={clipped_hi:.1%} black={clipped_lo:.1%} "
        f"coverage={(px[:, 3] > 0.5).mean():.1%}",
        flush=True,
    )


def report_bbox(uid, raw):
    """Print raw STL extents so they can be compared against the STEP Bnd_Box
    printed by hlr_worker (shape_diagonal). If the long axis differs, the STL
    and STEP are not in the same frame and the mismatch is upstream."""
    if raw is None or raw[0] is None:
        return
    mins, maxs = raw
    size = [maxs.x - mins.x, maxs.y - mins.y, maxs.z - mins.z]
    print(f"[BBOX] {uid}: size xyz={np.round(size, 4).tolist()} "
          f"long_axis={'xyz'[int(np.argmax(size))]}", flush=True)


# ----------------------------------------------------------------------------
# Per-mesh render
# ----------------------------------------------------------------------------

def render_rgb(obj_path, render_dir, cam, lights, cam_dis):
    import bpy

    uid = obj_path.stem
    to_export = {"frames": []}

    for i, (name, (direction, xdir)) in enumerate(VIEWS.items()):
        d, x, up = view_basis(direction, xdir)
        orient_camera(cam, d, x, up, cam_dis)
        orient_lights_to_view(lights, d, x, up)
        bpy.context.view_layer.update()

        if CHECK_ORIENTATION:
            check_camera_basis(cam, name, d, x, up)

        frame_filename = f"{uid}_{name}.png"
        output_path = (render_dir / frame_filename).resolve()

        bpy.context.scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)

        if not output_path.exists():
            raise RuntimeError(f"Render failed for view {name}: {output_path}")

        if CHECK_EXPOSURE and i == 0:
            report_exposure(output_path, f"{uid}/{name}")

        to_export["frames"].append({
            "file_path": frame_filename,
            "view_name": name,
            "view_idx": i,
            "camera_angle_x": FOV,
            "proj_type": 0,
            "direction": d.tolist(),
            "xdir": x.tolist(),
            "up": up.tolist(),
            "cam_dis": cam_dis,
            "transform_matrix": get_transform_matrix(cam),
        })

    with (render_dir / "transforms.json").open('w') as f:
        json.dump(to_export, f, indent=4)


def render_mesh(mesh_path, render_dir, cam, lights):
    from render import load_object

    clear_mesh_objects(keep=[cam] + list(lights.values()))

    import_mesh(mesh_path, load_object)
    if RECALC_NORMALS:
        recalc_normals()

    raw = normalize_scene_local()      # scale + center only, never rotates
    if CHECK_ORIENTATION:
        report_bbox(mesh_path.stem, raw)

    if OVERRIDE_MATERIAL:
        assign_material_to_meshes()
    if USE_PCA_ALIGNMENT:
        pca_align_meshes()

    cam_dis = compute_fit_cam_dis()
    render_rgb(mesh_path, render_dir, cam, lights, cam_dis)


def resolve_item(item_path):
    """Return (uid, mesh_path, output_dir) for a flat file or a uid folder."""
    if item_path.is_dir():
        uid = item_path.name
        meshes = (list(item_path.rglob("*.obj")) + list(item_path.rglob("*.stl"))
                  + list(item_path.rglob("*.STL")))
        mesh = meshes[0] if meshes else None
        return uid, mesh, item_path / "render_cond"
    return item_path.stem, item_path, OUTPUT_DIR / item_path.stem


def worker_main_chunk(chunk_file_str, force=False):
    """Render every path listed in chunk_file inside ONE Blender process."""
    paths = [line.strip() for line in Path(chunk_file_str).read_text().splitlines()
             if line.strip()]

    cam, lights = setup_scene_once()

    for p in paths:
        uid, mesh_path, out_dir = resolve_item(Path(p))

        n_imgs = len(list(out_dir.glob("*.png"))) if out_dir.exists() else 0
        if n_imgs == len(VIEWS) and not force:
            print(f"[SKIP] {uid}: already complete.", flush=True)
            continue
        if mesh_path is None or not mesh_path.exists():
            print(f"[SKIP] {uid}: mesh not found", flush=True)
            continue

        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            render_mesh(mesh_path, out_dir, cam, lights)
            print(f"[OK] {uid}", flush=True)
        except Exception as e:
            print(f"[ERROR] {uid}: {e}", flush=True)


def process_chunk_manager(chunk_paths):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write('\n'.join(chunk_paths))
        chunk_file = f.name

    cmd = [BLENDER_BIN, "-b", "--factory-startup", "-P", __file__,
           "--", "--worker-chunk", chunk_file]
    if FORCE_RERENDER:
        cmd.append("--force")

    def per_item_errors(reason):
        # One line per path so the [n/total] counter stays accurate.
        return [f"[ERROR] {Path(p).stem}: {reason}" for p in chunk_paths]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=PER_MESH_TIMEOUT * len(chunk_paths) + 60)
        prefixes = ("[OK]", "[SKIP]", "[ERROR]", "[GPU]", "[ENGINE]", "[EXPOSURE]", "[BBOX]")
        lines = [line for line in result.stdout.split('\n') if line.startswith(prefixes)]
        reported = {l.split(']', 1)[1].split(':', 1)[0].strip()
                    for l in lines if l.startswith(("[OK]", "[SKIP]", "[ERROR]"))}
        missing = [p for p in chunk_paths if Path(p).stem not in reported]
        if missing:
            tail = result.stderr.strip()[-500:] or result.stdout.strip()[-500:]
            lines += [f"[ERROR] {Path(p).stem}: chunk crashed before this item: {tail}"
                      for p in missing]
        return lines
    except subprocess.TimeoutExpired:
        return per_item_errors(f"chunk timeout ({len(chunk_paths)} items; "
                               f"finished meshes are kept; re-run to resume)")
    except Exception as e:
        return per_item_errors(f"chunk failed: {e}")
    finally:
        try:
            os.unlink(chunk_file)
        except OSError:
            pass


# === MAIN EXECUTION ===
if __name__ == "__main__":
    if "--worker-chunk" in sys.argv:
        idx = sys.argv.index("--worker-chunk")
        worker_main_chunk(sys.argv[idx + 1], force="--force" in sys.argv)
        sys.exit(0)

    if not BLENDER_BIN or not Path(BLENDER_BIN).exists():
        sys.exit(f"Blender not found (looked for BLENDER_BIN env, PATH, and "
                 f"{BLENDER_BIN!r}). Set BLENDER_BIN=/path/to/blender or add "
                 f"it to PATH.")

    if "--force" in sys.argv:
        FORCE_RERENDER = True

    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    items = list(MESH_DIR.iterdir())
    files = [p for p in items if p.is_file() and p.suffix.lower() in MESH_EXTS]
    folders = [d for d in items if d.is_dir()]

    if files:
        print("--- AUTO-DETECTED: FLAT FILE MODE ---")
        all_items = sorted(files)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    else:
        print("--- AUTO-DETECTED: FOLDER MODE ---")
        all_items = sorted(folders)

    pending = []
    for p in all_items:
        _, _, out_dir = resolve_item(p)
        n_imgs = len(list(out_dir.glob("*.png"))) if out_dir.exists() else 0
        if FORCE_RERENDER or n_imgs != len(VIEWS):
            pending.append(str(p))

    if limit is not None:
        pending = pending[:limit]
        print(f"--- LIMIT: rendering {len(pending)} item(s) ---")

    chunks = [pending[i:i + CHUNK_SIZE] for i in range(0, len(pending), CHUNK_SIZE)]

    print(f"--- Parallel Subprocess Renderer (Blender Headless Mode) ---")
    print(f"Blender:          {BLENDER_BIN}")
    print(f"Mesh dir:         {MESH_DIR}")
    print(f"Output dir:       {OUTPUT_DIR}")
    print(f"Total items:      {len(all_items)}  (pending: {len(pending)})")
    print(f"Chunk size:       {CHUNK_SIZE} (meshes per blender launch)")
    print(f"Workers:          {NUM_WORKERS}")
    print(f"Engine:           {ENGINE}  GPU requested: {USE_GPU}")
    print(f"Lighting:         camera-relative sun rig + world {WORLD_STRENGTH}")
    print(f"Total chunks:     {len(chunks)}")
    print(f"------------------------------------------------------------")

    done = 0
    with Pool(processes=NUM_WORKERS) as pool:
        for chunk_lines in pool.imap_unordered(process_chunk_manager, chunks):
            for line in chunk_lines:
                if line.startswith(("[OK]", "[SKIP]", "[ERROR]")):
                    done += 1
                    print(f"[{done}/{len(pending)}] {line}", flush=True)
                else:
                    print(line, flush=True)

    print("All chunks done.")