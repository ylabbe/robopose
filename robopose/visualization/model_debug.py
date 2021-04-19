import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from robopose.lib3d.camera_geometry import project_points

def make_rot_lines(TCO, K, length=0.03):
    assert TCO.shape == (4, 4)
    assert K.shape == (3, 3)
    TCO = np.asarray(TCO)
    pts = np.array([
        [0, 0, 0],
        [length, 0,  0],
        [0, length, 0],
        [0, 0, length]
    ])
    uv = project_points(torch.tensor(pts).unsqueeze(0).float(),
                        torch.tensor(K).unsqueeze(0).float(),
                        torch.tensor(TCO).unsqueeze(0).float())[0]
    return uv

def plot_rot_axes(ax, TCO, K, length=0.03):
    uv = make_rot_lines(TCO, K, length=length)
    ax.arrow(uv[0, 0], uv[0, 1], uv[1, 0] - uv[0, 0], uv[1, 1] - uv[0, 1],
             color='red')
    ax.arrow(uv[0, 0], uv[0, 1], uv[2, 0] - uv[0, 0], uv[2, 1] - uv[0, 1],
             color='green')
    ax.arrow(uv[0, 0], uv[0, 1], uv[3, 0] - uv[0, 0], uv[3, 1] - uv[0, 1],
             color='blue')
    return uv

def plot_debug(debug_dict, n=0):
    d = debug_dict
    im = d['images'][n].cpu().permute(1, 2, 0)
    render1 = d['renders'][n, :3].cpu().permute(1, 2, 0)
    render = render1
    render2 = d['renders'][n, 3:].cpu().permute(1, 2, 0)
    im_crop = d['images_crop'][n].cpu().permute(1, 2, 0)
    rend_box = d['boxes_rend'][n].cpu().numpy()
    crop_box = d['boxes_crop'][n].cpu().numpy()
    ref_point_uv = d['ref_point_uv'][n, 0].cpu().numpy()
    origin_uv = d['origin_uv'][n, 0].cpu().numpy()
    origin_uv_crop = d['origin_uv_crop'][n, 0].cpu().numpy()
    # TCO = d['TCO'][n].cpu().numpy()
    # TCR = d['TCR'][n].cpu().numpy()
    K_crop = d['K_crop'][n].cpu().numpy()
    uv = d['uv'][n].cpu().numpy()
    uv_gt_root = d['pts_proj_gt'][n].cpu().numpy() if 'pts_proj_gt' in d else None
    uv_input_root = d['pts_proj_input'][n].cpu().numpy() if 'pts_proj_input' in d else None


    plt.figure()
    fig,ax = plt.subplots(1)
    ax.imshow(im)
    ax.axis('off')

    plt.figure()
    fig,ax = plt.subplots(1)
    ax.imshow(im)
    x1, y1, x2, y2 = rend_box
    rect = patches.Rectangle((x1,y1),(x2-x1),(y2-y1),linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    ax.axis('off')

    x1, y1, x2, y2 = crop_box
    rect = patches.Rectangle((x1,y1),(x2-x1),(y2-y1),linewidth=1,edgecolor='g',facecolor='none')
    ax.add_patch(rect)
    x, y = ref_point_uv
    ax.plot(x, y, '+', color='red')
    x, y = origin_uv
    ax.plot(x, y, '+', color='orange')
    ax.scatter(uv[:, 0], uv[:, 1], marker='+')
    ax.axis('off')

    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(im_crop)
    ax.axis('off')
    if uv_gt_root is not None:
        ax.scatter(uv_gt_root[:, 0], uv_gt_root[:, 1], marker='+')

    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(render1)
    ax.axis('off')
    ax.plot(320/2, 240/2, '+', color='red')
    x, y = origin_uv_crop
    ax.plot(x, y, '+', color='orange')
    # plot_rot_axes(ax, TCO, K_crop, 0.03)

    if render2.numel() > 0:
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(render2)
        ax.axis('off')
        ax.plot(320/2, 240/2, '+', color='red')
        x, y = origin_uv_crop
        ax.plot(x, y, '+', color='orange')

#     if uv_input_root is not None:
#         ax.scatter(uv_input_root[:, 0], uv_input_root[:, 1], marker='+')

    plt.figure()
    plt.imshow(render)
    plt.imshow(im_crop, alpha=0.5)
    plt.axis('off')
