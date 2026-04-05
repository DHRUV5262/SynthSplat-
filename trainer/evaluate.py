import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from dataset import SynthSplatDataset
from train import GaussianModel, RASTER_EPS2D, RENDERER_BG_RGB, SH_DEGREE
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

try:
    from gsplat.rendering import rasterization
except ImportError:
    try:
        from gsplat import rasterization
    except ImportError:
        print("Could not import gsplat.rasterization. Please install gsplat.")
        sys.exit(1)


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    renderer_dir = os.path.join(root_dir, "renderer")
    trainer_dir = os.path.join(root_dir, "trainer")
    checkpoint_dir = os.path.join(trainer_dir, "checkpoints")
    eval_dir = os.path.join(trainer_dir, "eval_results")
    os.makedirs(eval_dir, exist_ok=True)

    dataset = SynthSplatDataset(renderer_dir, split="all")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load model (variable size — checkpoint determines N)
    model = GaussianModel(num_points=1, device=device)
    ckpt_path = os.path.join(checkpoint_dir, "ckpt_final.pt")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        return

    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd)
    print(f"Loaded {model.num_gaussians} Gaussians from checkpoint.")

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)

    psnr_scores = []
    ssim_scores = []
    lpips_scores = []

    print("Starting evaluation...")

    with torch.no_grad():
        active_scales = torch.exp(model.scales)
        active_opacities = torch.sigmoid(model.opacities).squeeze(-1)
        active_quats = model.quats / (model.quats.norm(dim=1, keepdim=True) + 1e-8)
        means = model.means
        sh_coeffs = model.sh_coeffs()

        save_indices = set(np.linspace(0, len(dataset) - 1, 10, dtype=int).tolist())
        to_pil = transforms.ToPILImage()

        for i, batch in enumerate(tqdm(dataloader)):
            gt_image = batch["image"].to(device)
            c2w = batch["c2w"].to(device)
            intrinsics = batch["intrinsics"].to(device)
            H, W = batch["height"].item(), batch["width"].item()
            frame_id = batch["frame_id"].item()

            viewmat = torch.inverse(c2w[0])
            fx, fy, cx, cy = intrinsics[0]
            K = torch.zeros((3, 3), device=device)
            K[0, 0] = fx
            K[0, 2] = cx
            K[1, 1] = fy
            K[1, 2] = cy
            K[2, 2] = 1.0

            bg = torch.tensor(RENDERER_BG_RGB, device=device, dtype=torch.float32)
            render_colors, render_alphas, info = rasterization(
                means,
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
                backgrounds=bg,
            )

            rendered_image = render_colors.permute(0, 3, 1, 2)

            psnr = psnr_metric(rendered_image, gt_image)
            ssim = ssim_metric(rendered_image, gt_image)
            lpips = lpips_metric(rendered_image, gt_image)

            psnr_scores.append(psnr.item())
            ssim_scores.append(ssim.item())
            lpips_scores.append(lpips.item())

            if i in save_indices:
                gt_pil = to_pil(gt_image[0].cpu())
                render_pil = to_pil(rendered_image[0].cpu().clamp(0, 1))

                combined = Image.new("RGB", (W * 2, H))
                combined.paste(gt_pil, (0, 0))
                combined.paste(render_pil, (W, 0))
                combined.save(os.path.join(eval_dir, f"eval_view_{frame_id:04d}.png"))

    print(f"\nResults ({model.num_gaussians} Gaussians, SH degree {SH_DEGREE}):")
    print(f"  Mean PSNR:  {np.mean(psnr_scores):.4f}")
    print(f"  Mean SSIM:  {np.mean(ssim_scores):.4f}")
    print(f"  Mean LPIPS: {np.mean(lpips_scores):.4f}")


if __name__ == "__main__":
    evaluate()
