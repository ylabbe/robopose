import numpy as np
import torch


def clamp(x, t_min, t_max):
    return torch.max(torch.min(x, t_max), t_min)


def add_noise_joints(q0, std_interval_ratio, q_limits):
    assert q_limits.dim() == 2
    assert q_limits.shape[0] == 2
    assert q0.dim() == 2
    assert q_limits.shape[1] == q0.shape[1]
    bsz = len(q0)

    q = q0.clone()
    std = std_interval_ratio * (q_limits[1] - q_limits[0])
    std = std.unsqueeze(0).view(-1).cpu().numpy()
    noise = np.random.normal(loc=0, scale=std, size=(bsz, len(std)))
    noise = torch.as_tensor(noise).to(q.device).to(q.dtype)
    q = q + noise
    q = clamp(q, q_limits[0] + 1e-4, q_limits[1] - 1e-4)
    return q
