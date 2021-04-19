import torch
import numpy as np
import multiprocessing

from robopose.lib3d.transform_ops import invert_T
from .bullet_scene_renderer import BulletSceneRenderer


def init_renderer(urdf_ds, preload=True):
    renderer = BulletSceneRenderer(urdf_ds=urdf_ds,
                                   preload_cache=preload,
                                   background_color=(0, 0, 0))
    return renderer


def worker_loop(worker_id, in_queue, out_queue, object_set, preload=True):
    renderer = init_renderer(object_set, preload=preload)
    while True:
        kwargs = in_queue.get()
        if kwargs is None:
            return
        obj_infos = kwargs['obj_infos']
        cam_infos = kwargs['cam_infos']
        render_depth = kwargs['render_depth']
        render_mask = kwargs['render_mask']
        is_valid = np.isfinite(obj_infos[0]['TWO']).all() \
            and np.isfinite(cam_infos[0]['TWC']).all() \
            and np.isfinite(cam_infos[0]['K']).all()
        if is_valid:
            cam_obs = renderer.render_scene(cam_infos=cam_infos, obj_infos=obj_infos,
                                            render_depth=render_depth)
            images = np.stack([d['rgb'] for d in cam_obs])
            depth = np.stack([d['depth'] for d in cam_obs]) if render_depth else None
            mask = np.stack([d['mask_int'] for d in cam_obs]) if render_mask else None
        else:
            res = cam_infos[0]['resolution']
            images = np.zeros((1, min(res), max(res), 3), dtype=np.uint8)
            depth = np.zeros((1, min(res), max(res)), dtype=np.float32)
            mask = np.zeros((1, min(res), max(res)), dtype=np.int32)
        out_queue.put((kwargs['data_id'], images, depth, mask))


class BulletBatchRenderer:
    def __init__(self, object_set, n_workers=8, preload_cache=True):
        self.object_set = object_set
        self.n_workers = n_workers
        self.init_plotters(preload_cache)

    def render(self, obj_infos, TCO, K, resolution=(240, 320), render_depth=False, render_mask=False):
        TCO = torch.as_tensor(TCO).detach()
        TOC = invert_T(TCO).cpu().numpy()
        K = torch.as_tensor(K).cpu().numpy()
        bsz = len(TCO)
        assert TCO.shape == (bsz, 4, 4)
        assert K.shape == (bsz, 3, 3)

        for n in np.arange(bsz):
            joints = obj_infos[n].get('joints', None)
            new_joints = dict()
            if joints is not None:
                for k, v in joints.items():
                    new_joints[k] = torch.as_tensor(v).clone().detach().cpu().numpy().item()
            joints = new_joints
            obj_info = dict(
                name=obj_infos[n]['name'],
                joints=joints,
                TWO=np.eye(4)
            )
            cam_info = dict(
                resolution=resolution,
                K=K[n],
                TWC=TOC[n],
            )
            kwargs = dict(cam_infos=[cam_info], obj_infos=[obj_info], render_depth=render_depth,
                          render_mask=render_mask)
            if self.n_workers > 0:
                kwargs['data_id'] = n
                self.in_queue.put(kwargs)
            else:
                cam_obs = self.plotters[0].render_scene(**kwargs)
                images = np.stack([d['rgb'] for d in cam_obs])
                depth = np.stack([d['depth'] for d in cam_obs]) if render_depth else None
                mask = np.stack([d['mask_int'] for d in cam_obs]) if render_mask else None
                self.out_queue.put((n, images, depth, mask))

        images = [None for _ in np.arange(bsz)]
        depths = [None for _ in np.arange(bsz)]
        masks = [None for _ in np.arange(bsz)]
        for n in np.arange(bsz):
            data_id, im, depth, mask = self.out_queue.get()
            images[data_id] = im[0]
            if render_depth:
                depths[data_id] = depth[0]
            if render_mask:
                masks[data_id] = mask[0]
        images = torch.as_tensor(np.stack(images, axis=0)).pin_memory().cuda(non_blocking=True)
        images = images.float().permute(0, 3, 1, 2) / 255

        if render_depth:
            depths = torch.as_tensor(np.stack(depths, axis=0)).pin_memory().cuda(non_blocking=True)
            depths = depths.float()
            return images, depths
        if render_mask:
            masks = torch.as_tensor(np.stack(masks, axis=0)).pin_memory().cuda(non_blocking=True)
            masks = masks.int()
            return images, masks
        else:
            return images

    def init_plotters(self, preload_cache):
        self.plotters = []
        self.in_queue = multiprocessing.Queue()
        self.out_queue = multiprocessing.Queue()

        if self.n_workers > 0:
            for n in range(self.n_workers):
                plotter = multiprocessing.Process(target=worker_loop,
                                                  kwargs=dict(worker_id=n,
                                                              in_queue=self.in_queue,
                                                              out_queue=self.out_queue,
                                                              preload=preload_cache,
                                                              object_set=self.object_set))
                plotter.start()
                self.plotters.append(plotter)
        else:
            self.plotters = [init_renderer(self.object_set, preload_cache)]

    def stop(self):
        if self.n_workers > 0:
            for p in self.plotters:
                self.in_queue.put(None)
            for p in self.plotters:
                p.join()
                p.terminate()
        self.in_queue.close()
        self.out_queue.close()

    def __del__(self):
        self.stop()
