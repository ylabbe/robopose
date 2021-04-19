import os
import json
import numpy as np
from .d3 import CameraPose

joint_name = [
    'Rotation', 'Base', 'Elbow', 'Wrist'
]

vertex_seq = [
    [0],
    [1],
    [2],
    [3,4],
    [5,5,6,6,7,8],
    [5,6,7,7,8,8],
    [9,9,10,10,11,12],
    [9,10,11,11,12,12],
    [13,14],
    [15,16],
    [17,18],
    [19,20],
    [21],
    [22,23]
]


def fov2f(fov, width):
    # fov = 2 * arctan(w / 2f)
    fov = fov * np.pi / 180
    return width / (2 * np.tan(fov / 2))


def make_2d_keypoints(vertex_infos, joints_infos, cam_info):
    loc = cam_info['Location']
    rot = cam_info['Rotation']

    if cam_info.__contains__('Fov'):
        f = fov2f(cam_info['Fov'], cam_info['FilmWidth'])
    else:
        f = fov2f(90, cam_info['FilmWidth'])

    loc = cam_info['Location']
    rot = cam_info['Rotation']
    cam = CameraPose(loc['X'], loc['Y'], loc['Z'],
                     rot['Pitch'], rot['Yaw'], rot['Roll'],
                     cam_info['FilmWidth'], cam_info['FilmHeight'], f)

    actor_name = 'RobotArmActor_3'
    joints = joints_infos
    joint = joints[actor_name]['WorldJoints'] # This usage might change
    joints3d = np.array([[joint[v]['X'], joint[v]['Y'], joint[v]['Z']] for v in joint_name])
    joints2d = cam.project_to_2d(joints3d)
    joints2d = joints2d[1:]  # discard the first joint, as we do not predict it.

    vertex = vertex_infos[actor_name]
    vertex3d = np.array([[v['X'], v['Y'], v['Z']] for v in vertex])
    vertex2d = cam.project_to_2d(vertex3d)

    num_vertex = len(vertex_seq)
    pts = np.zeros((num_vertex, 2))

    for i in range(num_vertex):
        pts[i] = np.average(vertex2d[vertex_seq[i]], axis=0)

    pts = np.concatenate((joints2d, pts), axis=0)
    return pts
