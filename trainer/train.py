import sys
import os

# Add current directory to path so we can import dataset
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import math
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


# Helper to create projection matrix from intrinsics
def get_projection_matrix(fx, fy, cx, cy, W, H, near=0.1, far=100.0):
    # OpenCV (Y down) to OpenGL NDC (Y up)
    # P[0][0] = 2 * fx / W
    # P[1][1] = -2 * fy / H  (Flip Y)
    # P[0][2] = -(W - 2 * cx) / W
    # P[1][2] = -(H - 2 * cy) / H
    # P[2][2] = -(far + near) / (far - near)
    # P[2][3] = -(2 * far * near) / (far - near)
    # P[3][2] = -1
    # P[3][3] = 0
    
    P = torch.zeros(4, 4)
    P[0, 0] = 2 * fx / W
    P[1, 1] = -2 * fy / H
    P[0, 2] = -(W - 2 * cx) / W
    P[1, 2] = -(H - 2 * cy) / H
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -(2 * far * near) / (far - near)
    P[3, 2] = -1.0
    P[3, 3] = 0.0
    return P

class GaussianModel(nn.Module):
    def __init__(self, num_points=20000):
        super().__init__()
        self.num_points = num_points
        
        # Initialize points in unit sphere
        # Random points
        means = (torch.rand(num_points, 3) - 0.5) * 2.0
        # Filter to be inside sphere
        mask = means.norm(dim=1) < 1.0
        means = means[mask]
        # Fill the rest
        while means.shape[0] < num_points:
            extra = (torch.rand(num_points - means.shape[0], 3) - 0.5) * 2.0
            mask = extra.norm(dim=1) < 1.0
            means = torch.cat([means, extra[mask]], dim=0)
        means = means[:num_points]
        
        self.means = nn.Parameter(means)
        self.scales = nn.Parameter(torch.ones(num_points, 3) * -5.0) # Log space, start small
        self.quats = nn.Parameter(torch.rand(num_points, 4)) # Random rotations
        self.opacities = nn.Parameter(torch.zeros(num_points, 1)) # Logit space (sigmoid(0) = 0.5)
        self.sh_coeffs = nn.Parameter(torch.zeros(num_points, 16, 3)) # SH degree 3
        
        # Initialize colors (SH DC term) to random colors
        self.sh_coeffs.data[:, 0, :] = torch.rand(num_points, 3)

    def forward(self):
        return self.means, self.scales, self.quats, self.opacities, self.sh_coeffs

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup paths
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Project root
    renderer_dir = os.path.join(root_dir, 'renderer')
    trainer_dir = os.path.join(root_dir, 'trainer')
    checkpoint_dir = os.path.join(trainer_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Dataset
    dataset = SynthSplatDataset(renderer_dir, split='train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Model
    model = GaussianModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Metrics
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    
    # Training Loop
    num_iterations = 15000
    iter_count = 0
    
    pbar = tqdm(total=num_iterations)
    
    # Log file
    log_file = open(os.path.join(trainer_dir, 'training_log.csv'), 'w')
    log_file.write('iteration,loss,psnr,ssim\n')
    
    while iter_count < num_iterations:
        for batch in dataloader:
            if iter_count >= num_iterations:
                break
                
            gt_image = batch['image'].to(device) # [1, 3, H, W]
            c2w = batch['c2w'].to(device) # [1, 4, 4]
            intrinsics = batch['intrinsics'].to(device) # [1, 4]
            H, W = batch['height'].item(), batch['width'].item()
            
            # Prepare matrices
            # gsplat expects View Matrix (W2C)
            viewmat = torch.inverse(c2w[0]) # [4, 4]
            
            # Projection Matrix
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
            means, scales, quats, opacities, sh_coeffs = model()
            
            # Handle gsplat API (assuming v1.0.0+ rasterization)
            # rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width, height)
            
            # Note: scales are log space, opacities are logit space
            active_scales = torch.exp(scales)
            active_opacities = torch.sigmoid(opacities).squeeze(-1)
            active_quats = quats / quats.norm(dim=1, keepdim=True)
            
            # Check gsplat version/API
            # Assuming rasterization exists and takes SH
            try:
                # Try new API with positional args to avoid potential keyword issues
                # rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width, height, near, far, radius_clip, eps2d, sh_degree)
                render_colors, render_alphas, info = rasterization(
                    means, 
                    active_quats, 
                    active_scales, 
                    active_opacities, 
                    sh_coeffs, 
                    viewmat.unsqueeze(0), # [1, 4, 4]
                    K.unsqueeze(0), # [1, 3, 3]
                    W, H,
                    0.01, # near_plane
                    1e10, # far_plane
                    0.0,  # radius_clip
                    0.3,  # eps2d
                    3     # sh_degree
                )
            except TypeError:
                # Fallback if arguments don't match (e.g. older version)
                # For now, let's assume the user has the latest gsplat where rasterization is available.
                pass

            # render_colors is [1, H, W, 3]
            rendered_image = render_colors.permute(0, 3, 1, 2) # [1, 3, H, W]
            
            # Loss
            l1_loss = nn.functional.l1_loss(rendered_image, gt_image)
            ssim_loss = 1.0 - ssim_metric(rendered_image, gt_image)
            loss = 0.6 * l1_loss + 0.4 * ssim_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            iter_count += 1
            pbar.update(1)
            pbar.set_description(f"Loss: {loss.item():.4f}")
            
            # Logging
            if iter_count % 100 == 0:
                with torch.no_grad():
                    psnr = psnr_metric(rendered_image, gt_image)
                    ssim = ssim_metric(rendered_image, gt_image)
                    log_file.write(f"{iter_count},{loss.item()},{psnr.item()},{ssim.item()}\n")
                    log_file.flush()
            
            # Checkpoint
            if iter_count % 1000 == 0:
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"ckpt_{iter_count}.pt"))

    log_file.close()
    pbar.close()
    
    # Save final
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "ckpt_final.pt"))
    print("Training complete.")

if __name__ == "__main__":
    train()
