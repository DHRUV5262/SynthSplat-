# SynthSplat

Synthetic multi-view data generation (custom OpenGL PBR renderer) plus **3D Gaussian Splatting** training (PyTorch + [gsplat](https://github.com/nerfstudio-project/gsplat)). No COLMAP, no real photos: camera intrinsics and extrinsics are **exact** from the renderer.

---

## Architecture

```
┌─────────────────────┐     frame_*.png      ┌──────────────────────────┐
│  C++ Renderer       │ ──────────────────►  │  Python trainer          │
│  (OpenGL 4.6 Core)  │     cameras.json     │  train.py / Colab nb     │
│  glTF/GLB → PNGs    │                      │  GaussianModel + gsplat   │
└─────────────────────┘                      └────────────┬─────────────┘
                                                        │
                                                        ▼
                                              checkpoints/*.pt, gaussians.ply
```

- **Renderer** (`renderer/`): Loads `.glb`/`.gltf` via TinyGLTF, PBR shading, MSAA, exports PNGs + `cameras.json` (view/projection matrices, resolution, FOV, scene AABB center/radius).
- **Trainer** (`trainer/`): Reads `renderer/output/`, initializes Gaussians inside the JSON scene bounds, optimizes with differentiable rasterization, densifies/prunes, exports a standard 3DGS-compatible binary PLY.

---

## Renderer (technical)

### Stack

| Component | Role |
|-----------|------|
| **C++17** | Host code |
| **OpenGL 4.6 Core** | Rasterization (GLAD loader) |
| **GLFW 3.3** | Context + window (hidden for batch; visible for interactive city preview) |
| **GLM** | `vec3`/`mat4`, `lookAt`, `perspective` |
| **TinyGLTF** | Scene graph, meshes, materials, textures |
| **stb_image_write** | PNG export |
| **nlohmann/json** | `cameras.json` |

Dependencies are pulled with **CMake FetchContent** (`renderer/CMakeLists.txt`). **GLAD** must be present under `renderer/src/` (see CMake error message if missing).

### Output

- The executable writes to **`../output` relative to the current working directory** (see `OUTPUT_DIR` in `main.cpp`). Typical Visual Studio runs use something like `renderer/build/bin/Debug/`, so frames often land in **`renderer/build/output/`**, not automatically in `renderer/output/`.
- **Training** (`dataset.py`) loads **`renderer/output/`** (i.e. `<repo>/renderer/output/cameras.json` and `frame_*.png`). After rendering, **copy or move** `cameras.json` and all `frame_*.png` into `renderer/output/`, or symlink that folder to your build output path.
- **Frames**: `frame_0000.png` … `frame_0299.png` (300 views by default).
- **`cameras.json`**: array of per-frame records: `frame_id`, `filename`, `position`, `view_matrix`, `projection_matrix`, `width`, `height`, `fov_degrees`, `scene_center`, `scene_radius`.

### Camera sampling

- **Orbit mode** (`Camera::generateOrbitPath`): 8 elevation rings (5°–75°), azimuth stepped per ring; per-ring counts weighted by `cos(elevation)` so lower rings get more samples. Total count = 300.
- **Immersive / city mode** (option 2 in `main.cpp`): interactive FPS preview, then `Camera::generateImmersivePath` from a locked eye position (see `Camera.cpp`).

### Conventions

- Clear color / sky must stay consistent with training: background compositing in gsplat uses **`RENDERER_BG_RGB = (0.15, 0.35, 0.72)`** in `train.py` — match the OpenGL clear / visible sky in `Renderer.cpp`.
- Intrinsics in Python are derived from the stored projection matrix: `fx = |P[0][0]| * W/2`, `fy = |P[1][1]| * H/2`, principal point at image center (`dataset.py`).

### Scene modes (`main.cpp`)

1. **Cafe + optional character (GLB)** — outside orbit, batch render only.
2. **City / large environment** — fly-through (WASD, mouse, Enter to lock pose, then 300 immersive views).

Place assets under `renderer/assets/`; CMake copies `assets/` and `shaders/` into the build directory on each build.

---

## Training (technical)

### Stack

- **Python 3.10+**, **PyTorch**, **gsplat** (CUDA differentiable Gaussian rasterization).
- See `trainer/requirements.txt` (pinned PyTorch CUDA wheels + `gsplat==1.4.0`).

### Training data layout

`SynthSplatDataset` expects:

```
<repo>/
  renderer/
    output/
      cameras.json
      frame_0000.png
      ...
```

`train.py` resolves `renderer` as a sibling of `trainer` under the repo root and reads `renderer/output/`.

### Gaussian representation

- **SH degree 3** → \((d+1)^2 = 16\) coefficients per channel → stored split as `sh_dc` \([N,1,3]\) and `sh_rest` \([N,15,3]\).
- **Parameters**: 3D means, log-scales (3), quaternions (4), logit opacities (1).
- **Init**: `INIT_NUM_POINTS` (e.g. 30k) points uniform in a sphere, shifted by `scene_center` and scaled by `scene_radius` from `cameras.json`.

### Rasterization

- `gsplat.rendering.rasterization` with **`absgrad=True`** (screen-space gradients for densification).
- **`backgrounds`**: shape **`(3,)`** float RGB matching the renderer (packed mode asserts this — not `(1,3)`).
- **`eps2d`**: `RASTER_EPS2D` (minimum projected extent; affects sharpness vs stability).

### Pose / coordinates

- JSON stores OpenGL-style **world-to-camera** `view_matrix`. Training builds **camera-to-world** `c2w = inverse(view)` and applies an OpenGL→OpenCV fix: multiply by `diag(1,-1,-1,1)` so gsplat’s camera convention matches the projection (`dataset.py`).

### Densification & pruning (high level)

- **Gradient accumulation**: With `absgrad=True`, read **`means2d.absgrad`**, not `.grad`. Normalize to pixel space: scale x by `W/2 * n_cameras`, y by `H/2 * n_cameras` (see `accumulate_means2d_grads_for_densify` in `train.py`).
- **Clone / split**: threshold on averaged gradient vs `GRAD_THRESH`; split vs clone by `CLONE_SPLIT_SCALE` on max axis scale.
- **Prune**: low opacity and excessive world scale; **`MAX_SCALE_THRESH`** must be large enough for scene extent (e.g. buildings), or large Gaussians are culled and the reconstruction blurs.
- **Min Gaussian floor**: e.g. `max(INIT_NUM_POINTS, 75_000)` so pruning cannot collapse the model.
- **Prune start iteration**: delayed (e.g. ≥3000) so new Gaussians can learn opacity.
- **Opacity reset**: periodic reset of logit opacities during densification window (forces re-learning, similar in spirit to original 3DGS practice).

### Loss

- `0.9 * L1 + 0.1 * (1 - SSIM)` — heavier L1 tends to preserve thin structure vs patch-based SSIM.

### Scripts

| File | Purpose |
|------|---------|
| `train.py` | Main optimization loop, checkpoints under `trainer/checkpoints/` |
| `evaluate.py` | PSNR / SSIM / LPIPS vs dataset, saves side-by-side PNGs |
| `export_ply.py` | Checkpoint → binary PLY (3DGS viewer compatible) |
| `SynthSplat_Colab.ipynb` | Colab-oriented copy of the training workflow |

---

## Debugging notes (what actually broke in practice)

1. **Blurry output / collapse to few Gaussians** -> `MAX_SCALE_THRESH` too small (e.g. 0.5) prunes anything “large” in world units; scenes need a threshold consistent with **meters-scale** geometry (e.g. 12.0).
2. **Densification idle / flat `num_gaussians`** -> wrong gradient source (`absgrad` path) and missing **screen-space scaling** of `means2d` gradients before comparing to `GRAD_THRESH`.
3. **Sky / background artifacts** -> rasterizer background RGB must match renderer clear/sky; wrong tensor shape for `backgrounds` fails gsplat assertions.
4. **PNG size vs JSON `width`/`height`** -> dataset resizes images to JSON dimensions to avoid silent misalignment.

---

## Build & run (quick reference)

### Renderer (Windows / CMake)

```text
cd renderer
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Run `SynthSplat` from the directory where `../output` and asset resolution match your intent (see `main.cpp` and `resolveDataPath`). Install **GLAD** into `renderer/src/` per `CMakeLists.txt` if configure fails.

### Trainer

```text
cd trainer
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
python train.py
python evaluate.py
python export_ply.py --ckpt checkpoints/ckpt_final.pt --out gaussians.ply
```

Requires **NVIDIA GPU** with CUDA for gsplat training; CPU is not practical for full runs.


## References

- Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering* (SIGGRAPH 2023).
- [gsplat](https://github.com/nerfstudio-project/gsplat) — differentiable Gaussian rasterization used here.

---
