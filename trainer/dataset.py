import json
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SynthSplatDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split
        self.output_dir = os.path.join(root_dir, 'output')
        self.cameras_path = os.path.join(self.output_dir, 'cameras.json')
        
        if not os.path.exists(self.cameras_path):
            raise FileNotFoundError(f"Cameras JSON not found at {self.cameras_path}")
            
        with open(self.cameras_path, 'r') as f:
            self.cameras = json.load(f)
            
        # Filter if needed (e.g. for train/test split)
        # For now, use all for training as requested, or simple split
        if split == 'test':
            self.cameras = self.cameras[::10] # Every 10th frame for test
        elif split == 'train':
            # Exclude test frames
            self.cameras = [c for i, c in enumerate(self.cameras) if i % 10 != 0]
            
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        cam_data = self.cameras[idx]
        
        # Load image
        img_path = os.path.join(self.output_dir, cam_data['filename'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image) # [3, H, W]
        
        # Dimensions
        H, W = cam_data['height'], cam_data['width']
        
        # Matrices from JSON (stored as list of lists)
        view_matrix = torch.tensor(cam_data['view_matrix'], dtype=torch.float32)
        proj_matrix = torch.tensor(cam_data['projection_matrix'], dtype=torch.float32)
        
        # Convert to C2W (Camera to World)
        # view_matrix is World to Camera (OpenGL)
        # c2w = view_matrix.inverse()
        c2w = torch.inverse(view_matrix)
        
        # Convert OpenGL coordinate system to OpenCV (used by gsplat/NeRF)
        # OpenGL: X right, Y up, Z back
        # OpenCV: X right, Y down, Z forward
        # Transformation: Rotate 180 deg around X
        # diag(1, -1, -1, 1)
        flip_mat = torch.tensor([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32)
        
        c2w = c2w @ flip_mat
        
        # Extract intrinsics from OpenGL Projection Matrix
        # P[0][0] = 2*n / (r-l) = 1 / tan(fov/2 * aspect)
        # P[1][1] = 2*n / (t-b) = 1 / tan(fov/2)
        # fx = P[0][0] * W / 2
        # fy = P[1][1] * H / 2
        # cx = W / 2
        # cy = H / 2
        
        fx = proj_matrix[0, 0] * W / 2.0
        fy = proj_matrix[1, 1] * H / 2.0
        cx = W / 2.0
        cy = H / 2.0
        
        intrinsics = torch.tensor([fx, fy, cx, cy], dtype=torch.float32)
        
        return {
            'image': image,
            'c2w': c2w,
            'intrinsics': intrinsics,
            'height': H,
            'width': W,
            'frame_id': cam_data['frame_id']
        }
