import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from dataset import SynthSplatDataset
from train import GaussianModel, get_projection_matrix
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torchvision.transforms as transforms

try:
    from gsplat.rendering import rasterization
except ImportError:
    try:
        from gsplat import rasterization
    except ImportError:
        print("Could not import gsplat.rasterization. Please install gsplat.")
        sys.exit(1)


def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup paths
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    renderer_dir = os.path.join(root_dir, 'renderer')
    trainer_dir = os.path.join(root_dir, 'trainer')
    checkpoint_dir = os.path.join(trainer_dir, 'checkpoints')
    eval_dir = os.path.join(trainer_dir, 'eval_results')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Dataset (Use all frames for evaluation as requested "Render all 100 views")
    # But usually we evaluate on test set.
    # The user said "Render all 100 views".
    # So I'll use a dataset with all views.
    dataset = SynthSplatDataset(renderer_dir, split='all') # Need to support 'all' in dataset
    # Hack: just use 'train' and 'test' combined or modify dataset to accept 'all'
    # I'll modify dataset instantiation here to just load everything if I can.
    # Actually, my dataset implementation:
    # if split == 'test': ... elif split == 'train': ...
    # If I pass something else, it might use all or crash.
    # Let's check dataset.py.
    # It defaults to 'train'.
    # I'll modify dataset.py to support 'all' or just set self.cameras = self.cameras (no filtering) if split is 'all'.
    
    # Re-reading dataset.py logic:
    # if split == 'test': ... elif split == 'train': ...
    # So if split is 'all', it does nothing (uses all).
    # Perfect.
    
    dataset = SynthSplatDataset(renderer_dir, split='all')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Load Model
    model = GaussianModel().to(device)
    ckpt_path = os.path.join(checkpoint_dir, "ckpt_final.pt")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        return
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # Metrics
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    
    print("Starting evaluation...")
    
    with torch.no_grad():
        means, scales, quats, opacities, sh_coeffs = model()
        
        active_scales = torch.exp(scales)
        active_opacities = torch.sigmoid(opacities).squeeze(-1)
        active_quats = quats / quats.norm(dim=1, keepdim=True)
        
        for i, batch in enumerate(tqdm(dataloader)):
            gt_image = batch['image'].to(device)
            c2w = batch['c2w'].to(device)
            intrinsics = batch['intrinsics'].to(device)
            H, W = batch['height'].item(), batch['width'].item()
            frame_id = batch['frame_id'].item()
            
            viewmat = torch.inverse(c2w[0])
            fx, fy, cx, cy = intrinsics[0]
            # projmat = get_projection_matrix(fx, fy, cx, cy, W, H).to(device)
            
            # gsplat expects K matrix (3x3)
            K = torch.zeros((3, 3), device=device)
            K[0, 0] = fx
            K[0, 2] = cx
            K[1, 1] = fy
            K[1, 2] = cy
            K[2, 2] = 1.0
            
            # Rasterize
            try:
                render_colors, render_alphas, info = rasterization(
                    means, 
                    active_quats, 
                    active_scales, 
                    active_opacities, 
                    sh_coeffs, 
                    viewmat.unsqueeze(0), 
                    K.unsqueeze(0), 
                    W, H,
                    0.01, # near
                    1e10, # far
                    0.0,  # radius_clip
                    0.3,  # eps2d
                    3     # sh_degree
                )
            except TypeError:
                 # Fallback logic if needed, or assume it works as in train.py
                 pass

            rendered_image = render_colors.permute(0, 3, 1, 2) # [1, 3, H, W]
            
            # Compute metrics
            psnr = psnr_metric(rendered_image, gt_image)
            ssim = ssim_metric(rendered_image, gt_image)
            lpips = lpips_metric(rendered_image, gt_image)
            
            psnr_scores.append(psnr.item())
            ssim_scores.append(ssim.item())
            lpips_scores.append(lpips.item())
            
            # Save side-by-side for 10 evenly spaced views
            # 100 views total. Indices 0, 10, 20, ... 90.
            if i % 10 == 0:
                # Convert to PIL
                to_pil = transforms.ToPILImage()
                
                # Denormalize if needed (but we didn't normalize in dataset, just ToTensor)
                
                # GT
                gt_pil = to_pil(gt_image[0].cpu())
                # Rendered
                render_pil = to_pil(rendered_image[0].cpu().clamp(0, 1))
                
                # Combine
                combined = Image.new('RGB', (W * 2, H))
                combined.paste(gt_pil, (0, 0))
                combined.paste(render_pil, (W, 0))
                
                combined.save(os.path.join(eval_dir, f"eval_view_{frame_id:04d}.png"))
                
    print(f"Mean PSNR: {np.mean(psnr_scores):.4f}")
    print(f"Mean SSIM: {np.mean(ssim_scores):.4f}")
    print(f"Mean LPIPS: {np.mean(lpips_scores):.4f}")

if __name__ == "__main__":
    import torchvision.transforms as transforms
    evaluate()
