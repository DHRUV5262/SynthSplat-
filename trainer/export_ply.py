import argparse
import os
import struct
from typing import Tuple

import numpy as np
import torch


def infer_sh_degree_from_coeffs(sh_coeffs: torch.Tensor) -> int:
    k = int(sh_coeffs.shape[1])
    degree_plus_1 = int(np.sqrt(k))
    if degree_plus_1 * degree_plus_1 != k:
        raise ValueError(f"Cannot infer SH degree from K={k}. Expected K=(d+1)^2.")
    return degree_plus_1 - 1


def normalize_quats(quats: torch.Tensor) -> torch.Tensor:
    return quats / (quats.norm(dim=1, keepdim=True) + 1e-12)


def export_checkpoint_to_ply(
    ckpt_path: str,
    out_path: str,
) -> Tuple[str, int, int]:
    """
    Exports a trained GaussianModel checkpoint to a 3DGS-compatible binary .ply.

    The original 3DGS / Mip-Splatting viewers expect RAW parameters:
      - opacity stored as logit (viewer applies sigmoid)
      - scales stored as log-scale (viewer applies exp)

    PLY properties (matching graphdeco-inria/gaussian-splatting):
      x, y, z, nx, ny, nz,
      f_dc_0..2,
      f_rest_0..(3*(K-1)-1),
      opacity,
      scale_0..2,
      rot_0..3
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cpu")
    state_dict = torch.load(ckpt_path, map_location=device)

    means = state_dict["means"].detach().float()
    scales = state_dict["scales"].detach().float()       # log-space (raw)
    quats = state_dict["quats"].detach().float()
    opacities = state_dict["opacities"].detach().float()  # logit-space (raw)

    # Support both new (sh_dc + sh_rest) and legacy (sh_coeffs) checkpoint format
    if "sh_dc" in state_dict:
        sh_dc = state_dict["sh_dc"].detach().float()     # [N, 1, 3]
        sh_rest = state_dict["sh_rest"].detach().float()  # [N, K-1, 3]
        sh_coeffs = torch.cat([sh_dc, sh_rest], dim=1)
    else:
        sh_coeffs = state_dict["sh_coeffs"].detach().float()

    N = int(means.shape[0])
    sh_degree = infer_sh_degree_from_coeffs(sh_coeffs)
    K = int(sh_coeffs.shape[1])  # (d+1)^2
    assert sh_coeffs.shape == (N, K, 3)

    quats = normalize_quats(quats)

    # Raw parameters (viewer applies sigmoid/exp)
    scales_out = scales                          # [N, 3] log-scales
    opacities_out = opacities.squeeze(-1)        # [N] logit

    # SH: DC and rest, same ordering as original 3DGS
    # Original 3DGS iterates: for band in range(K): for channel in range(3):
    f_dc = sh_coeffs[:, 0, :]                    # [N, 3]
    f_rest = sh_coeffs[:, 1:, :].reshape(N, -1)  # [N, (K-1)*3]

    normals = np.zeros((N, 3), dtype=np.float32)

    # Concatenate all columns
    data = np.concatenate(
        [
            means.numpy(),                          # 3: x y z
            normals,                                 # 3: nx ny nz
            f_dc.numpy(),                            # 3: f_dc_0..2
            f_rest.numpy(),                          # (K-1)*3: f_rest_0..
            opacities_out.unsqueeze(-1).numpy(),     # 1: opacity
            scales_out.numpy(),                      # 3: scale_0..2
            quats.numpy(),                           # 4: rot_0..3
        ],
        axis=1,
    ).astype(np.float32)

    num_f_rest = int(f_rest.shape[1])

    # Build PLY header (binary little-endian)
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
    ]
    for i in range(num_f_rest):
        header_lines.append(f"property float f_rest_{i}")

    header_lines += [
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header",
    ]

    expected_cols = 3 + 3 + 3 + num_f_rest + 1 + 3 + 4
    if data.shape[1] != expected_cols:
        raise RuntimeError(f"PLY column mismatch: got {data.shape[1]}, expected {expected_cols}")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Print diagnostics
    print(f"[export_ply] N={N}, sh_degree={sh_degree}")
    print(f"[export_ply] means:   min={means.numpy().min():.4f}, max={means.numpy().max():.4f}")
    print(f"[export_ply] scales (log):  min={scales.numpy().min():.4f}, max={scales.numpy().max():.4f}")
    print(f"[export_ply] opacity (logit): min={opacities_out.numpy().min():.4f}, max={opacities_out.numpy().max():.4f}")
    print(f"[export_ply] f_dc:   min={f_dc.numpy().min():.4f}, max={f_dc.numpy().max():.4f}")

    # Write binary PLY
    header_bytes = ("\n".join(header_lines) + "\n").encode("ascii")
    with open(out_path, "wb") as f:
        f.write(header_bytes)
        f.write(data.tobytes())

    return out_path, N, sh_degree


def main():
    parser = argparse.ArgumentParser(description="Export SynthSplat Gaussian checkpoint to .ply")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=os.path.join("trainer", "checkpoints", "ckpt_final.pt"),
        help="Path to ckpt_final.pt (GaussianModel state_dict).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join("trainer", "gaussians.ply"),
        help="Output ply path.",
    )

    args = parser.parse_args()

    out_path, n_points, sh_degree = export_checkpoint_to_ply(args.ckpt, args.out)
    print(f"Exported {n_points} gaussians (SH degree={sh_degree}) to: {out_path}")


if __name__ == "__main__":
    main()
