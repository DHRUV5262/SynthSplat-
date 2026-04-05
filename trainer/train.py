import sys
import os

# Force Python to find the CUDA 13.1 toolkit
os.environ["CUDA_HOME"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
os.environ["PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;" + os.environ.get("PATH", "")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import math
from typing import Optional
from dataset import SynthSplatDataset
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

try:
    from gsplat.rendering import rasterization
except ImportError:
    try:
        from gsplat import rasterization
    except ImportError:
        print("Could not import gsplat.rasterization. Please install gsplat.")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SH_DEGREE = 3
NUM_SH_COEFFS = (SH_DEGREE + 1) ** 2  # 16
INIT_NUM_POINTS = 30_000
NUM_ITERATIONS = 30_000

DENSIFY_START = 500
DENSIFY_END = 25_000
DENSIFY_INTERVAL = 100
GRAD_THRESH = 0.0001
MIN_OPACITY = 0.002
# World-space max axis scale before prune. 0.5 was far too small for building-sized scenes → mass prune → blur.
MAX_SCALE_THRESH = 12.0
MAX_GAUSSIANS = 800_000

# OpenGL clear / sky — must match renderer or empty pixels train against wrong composite.
RENDERER_BG_RGB = (0.15, 0.35, 0.72)
# gsplat adds eps2d to projected covariance; 0.3 ≈ ~3px minimum splat (caps sharpness / text).
RASTER_EPS2D = 0.1
# World-space: clone if max axis scale <= this, else split (slightly lower → more splits, sharper detail).
CLONE_SPLIT_SCALE = 0.008

LR_MEANS = 1.6e-4
LR_SCALES = 5e-3
LR_QUATS = 1e-3
LR_OPACITIES = 5e-2
LR_SH_DC = 2.5e-3
LR_SH_REST = 1.25e-4


def get_projection_matrix(fx, fy, cx, cy, W, H, near=0.1, far=100.0):
    P = torch.zeros(4, 4)
    P[0, 0] = 2 * fx / W
    P[1, 1] = -2 * fy / H
    P[0, 2] = -(W - 2 * cx) / W
    P[1, 2] = -(H - 2 * cy) / H
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -(2 * far * near) / (far - near)
    P[3, 2] = -1.0
    return P


# ---------------------------------------------------------------------------
# GaussianModel — uses raw tensors so we can resize during densification
# ---------------------------------------------------------------------------
class GaussianModel:
    def __init__(
        self,
        num_points: int = INIT_NUM_POINTS,
        device="cuda",
        init_center: Optional[torch.Tensor] = None,
        init_radius: Optional[float] = None,
    ):
        self.device = device

        # Match renderer scene bounds (NOT a unit sphere at origin — that caused blurry blobs).
        if init_center is not None and init_radius is not None:
            c = init_center.to(device).reshape(1, 3)
            unit = self._sample_in_sphere(num_points).to(device)
            means = c + float(init_radius) * unit
        else:
            means = self._sample_in_sphere(num_points).to(device)
        self.means = means.requires_grad_(True)
        self.scales = (torch.ones(num_points, 3, device=device) * -5.0).requires_grad_(True)
        self.quats = torch.rand(num_points, 4, device=device).requires_grad_(True)
        self.opacities = torch.zeros(num_points, 1, device=device).requires_grad_(True)

        sh = torch.zeros(num_points, NUM_SH_COEFFS, 3, device=device)
        sh[:, 0, :] = torch.rand(num_points, 3, device=device)
        self.sh_dc = sh[:, :1, :].contiguous().clone().to(device).requires_grad_(True)      # [N,1,3]
        self.sh_rest = sh[:, 1:, :].contiguous().clone().to(device).requires_grad_(True)     # [N,K-1,3]

        self.means2d_grad_accum = torch.zeros(num_points, device=device)
        self.denom = torch.zeros(num_points, device=device)

    @staticmethod
    def _sample_in_sphere(n: int) -> torch.Tensor:
        pts = (torch.rand(n * 2, 3) - 0.5) * 2.0
        pts = pts[pts.norm(dim=1) < 1.0][:n]
        while pts.shape[0] < n:
            extra = (torch.rand(n, 3) - 0.5) * 2.0
            extra = extra[extra.norm(dim=1) < 1.0]
            pts = torch.cat([pts, extra], 0)[:n]
        return pts

    @property
    def num_gaussians(self) -> int:
        return self.means.shape[0]

    def all_params(self):
        return [self.means, self.scales, self.quats, self.opacities, self.sh_dc, self.sh_rest]

    def sh_coeffs(self) -> torch.Tensor:
        return torch.cat([self.sh_dc, self.sh_rest], dim=1)  # [N, K, 3]

    # ------------------------------------------------------------------
    # Optimizer helpers: replace a parameter tensor in an Adam optimizer
    # ------------------------------------------------------------------
    @staticmethod
    def _replace_tensor_in_optimizer(optimizer, old_tensor, new_tensor):
        for group in optimizer.param_groups:
            for i, p in enumerate(group["params"]):
                if p is old_tensor:
                    stored = optimizer.state.get(p, {})
                    del optimizer.state[p]
                    group["params"][i] = new_tensor
                    if stored:
                        for k, v in stored.items():
                            if isinstance(v, torch.Tensor) and v.shape == old_tensor.shape:
                                stored[k] = torch.zeros_like(new_tensor)
                        optimizer.state[new_tensor] = stored
                    return

    def _cat_tensors_to_optimizer(self, optimizer, extension_dict):
        """Extend every parameter with new rows and update optimizer state."""
        for attr_name, ext in extension_dict.items():
            old = getattr(self, attr_name)
            new = torch.cat([old, ext], dim=0).requires_grad_(True)
            self._replace_tensor_in_optimizer(optimizer, old, new)
            setattr(self, attr_name, new)

    def _prune_tensors_in_optimizer(self, optimizer, mask):
        """Keep only rows where mask is True."""
        for attr_name in ["means", "scales", "quats", "opacities", "sh_dc", "sh_rest"]:
            old = getattr(self, attr_name)
            new = old[mask].clone().detach().requires_grad_(True)
            self._replace_tensor_in_optimizer(optimizer, old, new)
            setattr(self, attr_name, new)
        self.means2d_grad_accum = self.means2d_grad_accum[mask]
        self.denom = self.denom[mask]

    # ------------------------------------------------------------------
    # Densification
    # ------------------------------------------------------------------
    def densify_and_prune(self, optimizer, iter_count):
        N = self.num_gaussians
        avg_grad = self.means2d_grad_accum / (self.denom + 1e-8)
        active_scales = torch.exp(self.scales)
        max_scale = active_scales.max(dim=1).values

        # --- Clone small Gaussians with large gradients ---
        clone_mask = (avg_grad >= GRAD_THRESH) & (max_scale <= CLONE_SPLIT_SCALE)
        if clone_mask.sum() > 0 and (N + clone_mask.sum()) <= MAX_GAUSSIANS:
            ext = {
                "means": self.means[clone_mask].clone().detach(),
                "scales": self.scales[clone_mask].clone().detach(),
                "quats": self.quats[clone_mask].clone().detach(),
                "opacities": self.opacities[clone_mask].clone().detach(),
                "sh_dc": self.sh_dc[clone_mask].clone().detach(),
                "sh_rest": self.sh_rest[clone_mask].clone().detach(),
            }
            self._cat_tensors_to_optimizer(optimizer, ext)
            new_n = clone_mask.sum().item()
            self.means2d_grad_accum = torch.cat([self.means2d_grad_accum, torch.zeros(new_n, device=self.device)])
            self.denom = torch.cat([self.denom, torch.zeros(new_n, device=self.device)])

        N = self.num_gaussians
        avg_grad = self.means2d_grad_accum / (self.denom + 1e-8)
        active_scales = torch.exp(self.scales)
        max_scale = active_scales.max(dim=1).values

        # --- Split large Gaussians with large gradients ---
        split_mask = (avg_grad >= GRAD_THRESH) & (max_scale > CLONE_SPLIT_SCALE)
        if split_mask.sum() > 0 and (N + split_mask.sum()) <= MAX_GAUSSIANS:
            stds = active_scales[split_mask]
            sample_means = self.means[split_mask].detach().repeat(2, 1)
            noise = torch.randn_like(sample_means) * stds.repeat(2, 1)
            sample_means = sample_means + noise

            new_scales = (active_scales[split_mask] / 1.6).log().repeat(2, 1)
            new_quats = self.quats[split_mask].detach().repeat(2, 1)
            new_opacities = self.opacities[split_mask].detach().repeat(2, 1)
            new_sh_dc = self.sh_dc[split_mask].detach().repeat(2, 1, 1)
            new_sh_rest = self.sh_rest[split_mask].detach().repeat(2, 1, 1)

            ext = {
                "means": sample_means,
                "scales": new_scales,
                "quats": new_quats,
                "opacities": new_opacities,
                "sh_dc": new_sh_dc,
                "sh_rest": new_sh_rest,
            }
            self._cat_tensors_to_optimizer(optimizer, ext)
            added = sample_means.shape[0]
            self.means2d_grad_accum = torch.cat([self.means2d_grad_accum, torch.zeros(added, device=self.device)])
            self.denom = torch.cat([self.denom, torch.zeros(added, device=self.device)])

            # Remove the originals that were split
            prune_split = torch.ones(self.num_gaussians, dtype=torch.bool, device=self.device)
            # The split originals are at the same indices as before extension
            split_indices = split_mask.nonzero(as_tuple=True)[0]
            prune_split[split_indices] = False
            self._prune_tensors_in_optimizer(optimizer, prune_split)

        # --- Prune low-opacity or oversized Gaussians ---
        # Delay until iter 3000 so newly-densified splats can learn opacity first.
        if iter_count >= 3000:
            active_opacities = torch.sigmoid(self.opacities.squeeze(-1))
            active_scales_now = torch.exp(self.scales)
            max_scale_now = active_scales_now.max(dim=1).values
            keep_mask = (active_opacities > MIN_OPACITY) & (max_scale_now < MAX_SCALE_THRESH)
            min_gaussians = max(INIT_NUM_POINTS, 75_000)
            if keep_mask.sum() < min_gaussians:
                _, topk = active_opacities.topk(min(min_gaussians, self.num_gaussians))
                keep_mask[topk] = True
            if (~keep_mask).sum() > 0:
                self._prune_tensors_in_optimizer(optimizer, keep_mask)

        # Opacity reset every 3000 iters during densification — forces all splats to re-earn opacity
        if iter_count % 3000 == 0 and iter_count <= DENSIFY_END:
            new_op = torch.full_like(self.opacities, -2.0)  # sigmoid(-2) ≈ 0.12
            old_op = self.opacities
            self._replace_tensor_in_optimizer(optimizer, old_op, new_op.requires_grad_(True))
            self.opacities = new_op.requires_grad_(True)

        # Reset accumulators
        self.means2d_grad_accum = torch.zeros(self.num_gaussians, device=self.device)
        self.denom = torch.zeros(self.num_gaussians, device=self.device)

    # ------------------------------------------------------------------
    # Checkpoint save/load
    # ------------------------------------------------------------------
    def state_dict(self):
        return {
            "means": self.means.detach().cpu(),
            "scales": self.scales.detach().cpu(),
            "quats": self.quats.detach().cpu(),
            "opacities": self.opacities.detach().cpu(),
            "sh_dc": self.sh_dc.detach().cpu(),
            "sh_rest": self.sh_rest.detach().cpu(),
        }

    def load_state_dict(self, sd):
        self.means = sd["means"].to(self.device).requires_grad_(True)
        self.scales = sd["scales"].to(self.device).requires_grad_(True)
        self.quats = sd["quats"].to(self.device).requires_grad_(True)
        self.opacities = sd["opacities"].to(self.device).requires_grad_(True)
        if "sh_dc" in sd:
            self.sh_dc = sd["sh_dc"].to(self.device).requires_grad_(True)
            self.sh_rest = sd["sh_rest"].to(self.device).requires_grad_(True)
        else:
            # Legacy checkpoint with single sh_coeffs tensor
            sh = sd["sh_coeffs"].to(self.device)
            self.sh_dc = sh[:, :1, :].contiguous().requires_grad_(True)
            self.sh_rest = sh[:, 1:, :].contiguous().requires_grad_(True)
        N = self.means.shape[0]
        self.means2d_grad_accum = torch.zeros(N, device=self.device)
        self.denom = torch.zeros(N, device=self.device)


def accumulate_means2d_grads_for_densify(model: "GaussianModel", info: dict) -> None:
    """Accumulate 2D positional gradients for densification, matching gsplat DefaultStrategy's normalization."""
    if "means2d" not in info:
        return
    m2d = info["means2d"]
    g = getattr(m2d, "absgrad", None)
    if g is None and m2d.grad is not None:
        g = m2d.grad
    if g is None:
        return
    g = g.clone().detach()
    W = float(info.get("width", 1))
    H = float(info.get("height", 1))
    n_cam = float(info.get("n_cameras", 1))
    g[..., 0] *= W / 2.0 * n_cam
    g[..., 1] *= H / 2.0 * n_cam

    gids = info.get("gaussian_ids")
    if gids is not None and gids.numel() > 0 and g.dim() <= 2:
        gn = g.norm(dim=-1)
        gids_long = gids.long()
        if int(gids_long.max().item()) >= model.num_gaussians:
            return
        ones = torch.ones_like(gn, dtype=model.denom.dtype)
        model.means2d_grad_accum.index_add_(0, gids_long, gn)
        model.denom.index_add_(0, gids_long, ones)
        return
    if g.dim() == 3:
        g = g[0]
    gn = g.norm(dim=-1)
    if gn.shape[0] == model.num_gaussians:
        model.means2d_grad_accum += gn
        model.denom += 1


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    renderer_dir = os.path.join(root_dir, "renderer")
    trainer_dir = os.path.join(root_dir, "trainer")
    checkpoint_dir = os.path.join(trainer_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    dataset = SynthSplatDataset(renderer_dir, split="all")
    print(
        f"Scene init: center={dataset.scene_center.numpy()}, radius={dataset.scene_radius:.4f}"
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = GaussianModel(
        INIT_NUM_POINTS,
        device=device,
        init_center=dataset.scene_center,
        init_radius=dataset.scene_radius,
    )

    optimizer = optim.Adam(
        [
            {"params": [model.means], "lr": LR_MEANS, "name": "means"},
            {"params": [model.scales], "lr": LR_SCALES, "name": "scales"},
            {"params": [model.quats], "lr": LR_QUATS, "name": "quats"},
            {"params": [model.opacities], "lr": LR_OPACITIES, "name": "opacities"},
            {"params": [model.sh_dc], "lr": LR_SH_DC, "name": "sh_dc"},
            {"params": [model.sh_rest], "lr": LR_SH_REST, "name": "sh_rest"},
        ],
        eps=1e-15,
    )

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

    iter_count = 0
    pbar = tqdm(total=NUM_ITERATIONS)

    log_file = open(os.path.join(trainer_dir, "training_log.csv"), "w")
    log_file.write("iteration,loss,psnr,ssim,num_gaussians\n")

    while iter_count < NUM_ITERATIONS:
        for batch in dataloader:
            if iter_count >= NUM_ITERATIONS:
                break

            gt_image = batch["image"].to(device)
            c2w = batch["c2w"].to(device)
            intrinsics = batch["intrinsics"].to(device)
            H, W = batch["height"].item(), batch["width"].item()

            viewmat = torch.inverse(c2w[0])

            fx, fy, cx, cy = intrinsics[0]
            K = torch.zeros((3, 3), device=device)
            K[0, 0] = fx
            K[0, 2] = cx
            K[1, 1] = fy
            K[1, 2] = cy
            K[2, 2] = 1.0

            # Activate parameters
            active_scales = torch.exp(model.scales)
            active_opacities = torch.sigmoid(model.opacities).squeeze(-1)
            active_quats = model.quats / (model.quats.norm(dim=1, keepdim=True) + 1e-8)
            sh_coeffs = model.sh_coeffs()

            # Packed gsplat rasterizer expects shape (3,), not (1, 3).
            bg = torch.tensor(RENDERER_BG_RGB, device=device, dtype=torch.float32)
            render_colors, render_alphas, info = rasterization(
                model.means,
                active_quats,
                active_scales,
                active_opacities,
                sh_coeffs,
                viewmat.unsqueeze(0),
                K.unsqueeze(0),
                W,
                H,
                near_plane=0.01,
                far_plane=1e10,
                radius_clip=0.0,
                eps2d=RASTER_EPS2D,
                sh_degree=SH_DEGREE,
                absgrad=True,
                backgrounds=bg,
            )

            rendered_image = render_colors.permute(0, 3, 1, 2)

            l1_loss = nn.functional.l1_loss(rendered_image, gt_image)
            ssim_loss = 1.0 - ssim_metric(rendered_image, gt_image)
            # More L1 helps thin edges and small text vs SSIM which prefers smooth patches.
            loss = 0.9 * l1_loss + 0.1 * ssim_loss

            optimizer.zero_grad()

            # Non-absgrad path needs retain_grad on means2d for .grad after backward.
            if "means2d" in info:
                info["means2d"].retain_grad()

            loss.backward()
            accumulate_means2d_grads_for_densify(model, info)

            optimizer.step()

            iter_count += 1
            pbar.update(1)
            pbar.set_description(
                f"Loss: {loss.item():.4f} | N: {model.num_gaussians}"
            )

            # Densification / Pruning
            if DENSIFY_START <= iter_count <= DENSIFY_END and iter_count % DENSIFY_INTERVAL == 0:
                if iter_count == DENSIFY_START:
                    ag = model.means2d_grad_accum
                    print(f"\n[densify debug] it={iter_count} accum_max={ag.max():.6f} "
                          f"accum_nonzero={int((ag>0).sum())} denom_max={model.denom.max():.0f}")
                model.densify_and_prune(optimizer, iter_count)
                if iter_count == DENSIFY_START:
                    print(f"[densify debug] after prune: num_gaussians={model.num_gaussians}")
                pbar.set_postfix(gaussians=model.num_gaussians)

            # Logging
            if iter_count % 100 == 0:
                with torch.no_grad():
                    psnr_val = psnr_metric(rendered_image, gt_image)
                    ssim_val = ssim_metric(rendered_image, gt_image)
                    log_file.write(
                        f"{iter_count},{loss.item()},{psnr_val.item()},{ssim_val.item()},{model.num_gaussians}\n"
                    )
                    log_file.flush()

            # Free VRAM fragmentation
            if iter_count % 1000 == 0:
                torch.cuda.empty_cache()

            # Checkpoint
            if iter_count % 2000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(checkpoint_dir, f"ckpt_{iter_count}.pt"),
                )

    log_file.close()
    pbar.close()

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "ckpt_final.pt"))
    print(f"Training complete. Final Gaussians: {model.num_gaussians}")


if __name__ == "__main__":
    train()
