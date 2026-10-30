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
if SITE_PACKAGES not in sys.path:
    sys.path.append(SITE_PACKAGES)

MESH_DIR = Path("/home/ec2-user/cad-2d-vlm-accv/data/medical/meshes")
RENDER_DIR = Path("/home/ec2-user/cad-2d-vlm-accv/scripts/rendering/blender")
OUTPUT_DIR = Path("/home/ec2-user/cad-2d-vlm-accv/data/medical/renders")
MESH_EXTS = (".stl", ".obj")

# --- Render config ---
RESOLUTION = 512
CAM_DIS = 1.5
FOV = np.radians(45.0)
FRAME_FILL = 0.85

FRONT_AXIS = '+x'
_AZIMUTH_OFFSETS = {'+x': 0.0, '+y': np.pi / 2, '-x': np.pi, '-y': -np.pi / 2}
AZIMUTH_OFFSET = _AZIMUTH_OFFSETS[FRONT_AXIS]

USE_PCA_ALIGNMENT = False
FORCE_RERENDER = False

# --- Speed knobs ---
# ENGINE: 'CYCLES' (path traced, slowest), 'BLENDER_EEVEE_NEXT' (Blender 4.2+,
# 10-30x faster, near-identical look for CAD), or 'BLENDER_WORKBENCH'
# (fastest, studio shading, ignores scene lights). If the requested engine
# isn't available in this Blender build we fall back down the list.
ENGINE = 'CYCLES'
ENGINE_FALLBACKS = ['BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE', 'CYCLES']

CYCLES_SAMPLES = 16
CYCLES_DENOISE = True
EEVEE_SAMPLES = 32

USE_GPU = True
# One Blender process per mesh pays full startup ~10000 times; instead one
# persistent process works through CHUNK_SIZE meshes with the scene, lights,
# camera and GPU set up exactly once.
CHUNK_SIZE = 50
# A single GPU is saturated by 1-2 Cycles processes; more just contend.
# For CPU rendering, give each worker an even share of the cores.
NUM_WORKERS = 2 if USE_GPU else max(1, cpu_count() // 8)
CPU_THREADS_PER_WORKER = max(1, cpu_count() // NUM_WORKERS)
# Seconds allowed per mesh before the whole chunk is killed. Already-finished
# meshes are skipped on re-run, so a killed chunk only loses the mesh that hung.
PER_MESH_TIMEOUT = 30

# ----------------------------------------------------------------------------
# Must stay byte-identical to hlr_worker.VIEWS so the Blender RGB renders and
# the OCC/HLR line-drawing DXFs are the same 6 views of the same part.
# (direction = vector from the origin to the viewer, xdir = image +X axis)
# ----------------------------------------------------------------------------
VIEWS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "front":  ((0, -1, 0), (1, 0, 0)),
    "back":   ((0, 1, 0),  (-1, 0, 0)),
    "left":   ((-1, 0, 0), (0, -1, 0)),
    "right":  ((1, 0, 0),  (0, 1, 0)),
    "top":    ((0, 0, 1),  (1, 0, 0)),
    "bottom": ((0, 0, -1), (1, 0, 0)),
}

LIGHT_NAMES = ("Light_Front", "Light_Back", "Light_Left", "Light_Right", "Light_Top")
TARGET_NAME = "_render_target"


def _normalize(vec):
    v = np.asarray(vec, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _rotate_z(vec, angle):
    x, y, z = vec
    c, s = np.cos(angle), np.sin(angle)
    return np.array([x * c - y * s, x * s + y * c, z], dtype=np.float64)


def orient_camera(cam, direction, xdir, cam_dis):
    """Set the camera to match an hlr_worker view exactly (up = direction x xdir)."""
    from mathutils import Matrix

    d = _normalize(direction)
    x = _normalize(xdir)
    up = _normalize(np.cross(d, x))
    loc = d * cam_dis

    cam.matrix_world = Matrix((
        (x[0], up[0], d[0], loc[0]),
        (x[1], up[1], d[1], loc[1]),
        (x[2], up[2], d[2], loc[2]),
        (0.0,  0.0,   0.0,  1.0),
    ))


# --- Mesh bounds, centering, camera fit ---

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


def center_scene_meshes():
    import bpy

    mins, maxs, _ = get_scene_bounds()
    if mins is None:
        return

    center = (mins + maxs) / 2.0
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and obj.parent is None:
            obj.location.x -= center.x
            obj.location.y -= center.y
            obj.location.z -= center.z

    bpy.context.view_layer.update()


def compute_fit_cam_dis():
    mins, maxs, _ = get_scene_bounds()
    if mins is None:
        return CAM_DIS
    radius = (maxs - mins).length / 2.0
    if radius <= 0:
        return CAM_DIS
    return radius / (np.sin(FOV / 2.0) * FRAME_FILL)


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
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and obj.parent is None:
            obj.matrix_world = R_align @ obj.matrix_world

    bpy.context.view_layer.update()


# --- One-time scene setup (per chunk) ---

def get_render_target():
    import bpy
    if TARGET_NAME in bpy.data.objects:
        target = bpy.data.objects[TARGET_NAME]
    else:
        target = bpy.data.objects.new(TARGET_NAME, None)
        bpy.context.scene.collection.objects.link(target)
    target.location = (0.0, 0.0, 0.0)
    return target


def create_studio_lights(target_obj):
    """Create the 5 area lights once. Position/energy are updated per mesh."""
    import bpy

    lights = {}
    for name in LIGHT_NAMES:
        if name in bpy.data.objects:
            lights[name] = bpy.data.objects[name]
            continue
        light_data = bpy.data.lights.new(name=name, type='AREA')
        light_obj = bpy.data.objects.new(name=name, object_data=light_data)
        bpy.context.collection.objects.link(light_obj)
        constraint = light_obj.constraints.new(type='TRACK_TO')
        constraint.target = target_obj
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'
        lights[name] = light_obj
    return lights


def update_studio_lights(lights, light_dist, energy):
    size = 3.0 * (light_dist / 2.5)
    positions = {
        "Light_Front": (0.0, -light_dist, 0.0),
        "Light_Back":  (0.0, light_dist, 0.0),
        "Light_Left":  (light_dist, 0.0, 0.0),
        "Light_Right": (-light_dist, 0.0, 0.0),
        "Light_Top":   (0.0, 0.0, light_dist),
    }
    for name, obj in lights.items():
        obj.location = positions[name]
        obj.data.energy = energy
        obj.data.size = size


def enable_gpu_cycles():
    """Point Cycles at the fastest GPU backend. Returns the backend name or None.

    NOT cached: bpy.ops.wm.read_factory_settings() resets preferences (including
    compute_device_type), which is why the old cached version silently fell back
    to CPU for every mesh after the first one in a chunk.
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


def setup_scene_once():
    """Build the scene, camera, lights and engine config. Called once per chunk."""
    import bpy
    sys.path.insert(0, str(RENDER_DIR))
    from render import init_render, init_scene, init_camera

    bpy.ops.wm.read_factory_settings(use_empty=True)
    init_render(engine='CYCLES', resolution=RESOLUTION, geo_mode=True)
    engine = select_engine()

    scene = bpy.context.scene
    scene.render.film_transparent = True
    # Keep BVH / scene sync across the 6 views instead of rebuilding per render call.
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
    cam = init_camera()
    cam.data.type = 'PERSP'
    cam.data.sensor_fit = 'HORIZONTAL'
    cam.data.sensor_width = 32.0
    cam.data.lens = 16 / np.tan(FOV / 2)

    target = get_render_target()
    lights = create_studio_lights(target)
    return cam, lights


def clear_mesh_objects(keep):
    """Remove everything from the previous mesh (objects + orphan data)."""
    import bpy

    keep_names = {o.name for o in keep}
    for obj in list(bpy.context.scene.objects):
        if obj.name in keep_names:
            continue
        bpy.data.objects.remove(obj, do_unlink=True)

    # Free orphaned mesh/material data so memory doesn't grow across the chunk.
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for mat in list(bpy.data.materials):
        if mat.users == 0:
            bpy.data.materials.remove(mat)


def import_mesh(mesh_path, load_object):
    import bpy

    ext = mesh_path.suffix.lower()
    if ext == ".stl":
        try:
            bpy.ops.wm.stl_import(filepath=str(mesh_path))
        except (AttributeError, RuntimeError):
            bpy.ops.import_mesh.stl(filepath=str(mesh_path))
    else:
        load_object(str(mesh_path))


# --- Per-mesh render ---

def render_rgb(obj_path, render_dir, cam, get_transform_matrix, cam_dis):
    import bpy

    uid = obj_path.stem
    to_export = {"frames": []}

    for i, (name, (direction, xdir)) in enumerate(VIEWS.items()):
        d = _rotate_z(direction, AZIMUTH_OFFSET)
        x = _rotate_z(xdir, AZIMUTH_OFFSET)
        orient_camera(cam, d, x, cam_dis)
        bpy.context.view_layer.update()

        frame_filename = f"{uid}_{name}.png"
        output_path = (render_dir / frame_filename).resolve()

        bpy.context.scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)

        if not output_path.exists():
            raise RuntimeError(f"Render failed for view {name}: {output_path}")

        to_export["frames"].append({
            "file_path": frame_filename,
            "view_name": name,
            "view_idx": i,
            "camera_angle_x": FOV,
            "proj_type": 0,
            "direction": d.tolist(),
            "xdir": x.tolist(),
            "cam_dis": cam_dis,
            "transform_matrix": get_transform_matrix(cam),
        })

    with (render_dir / "transforms.json").open('w') as f:
        json.dump(to_export, f, indent=4)


def render_mesh(mesh_path, render_dir, cam, lights):
    import bpy
    from render import load_object, normalize_scene, get_transform_matrix

    target = get_render_target()
    clear_mesh_objects(keep=[cam, target] + list(lights.values()))

    import_mesh(mesh_path, load_object)
    normalize_scene()

    if USE_PCA_ALIGNMENT:
        pca_align_meshes()
    center_scene_meshes()

    cam_dis = compute_fit_cam_dis()
    light_dist = cam_dis * 1.6
    light_energy = 500.0 * (light_dist / 2.5) ** 2
    update_studio_lights(lights, light_dist, light_energy)

    render_rgb(mesh_path, render_dir, cam, get_transform_matrix, cam_dis)


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
    paths = [line.strip() for line in Path(chunk_file_str).read_text().splitlines() if line.strip()]

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

    cmd = ["blender", "-b", "--factory-startup", "-P", __file__,
           "--", "--worker-chunk", chunk_file]
    if FORCE_RERENDER:
        cmd.append("--force")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=PER_MESH_TIMEOUT * len(chunk_paths) + 60)
        lines = [line for line in result.stdout.split('\n')
                 if line.startswith(("[OK]", "[SKIP]", "[ERROR]", "[GPU]", "[ENGINE]"))]
        if result.returncode != 0 and not any(l.startswith(("[OK]", "[SKIP]")) for l in lines):
            lines.append(f"[ERROR] chunk crashed: {result.stderr.strip()[-500:]}")
        return lines
    except subprocess.TimeoutExpired:
        return [f"[TIMEOUT] chunk of {len(chunk_paths)} items (finished meshes are kept; re-run to resume)"]
    except Exception as e:
        return [f"[ERROR] chunk failed: {e}"]
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

    # Skip already-complete items up front so chunks are full of real work.
    pending = []
    for p in all_items:
        _, _, out_dir = resolve_item(p)
        n_imgs = len(list(out_dir.glob("*.png"))) if out_dir.exists() else 0
        if FORCE_RERENDER or n_imgs != len(VIEWS):
            pending.append(str(p))

    chunks = [pending[i:i + CHUNK_SIZE] for i in range(0, len(pending), CHUNK_SIZE)]

    print(f"--- Parallel Subprocess Renderer (Blender Headless Mode) ---")
    print(f"Mesh dir:         {MESH_DIR}")
    print(f"Output dir:       {OUTPUT_DIR}")
    print(f"Total items:      {len(all_items)}  (pending: {len(pending)})")
    print(f"Chunk size:       {CHUNK_SIZE} (meshes per blender launch)")
    print(f"Workers:          {NUM_WORKERS}")
    print(f"Engine:           {ENGINE}  GPU requested: {USE_GPU}")
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