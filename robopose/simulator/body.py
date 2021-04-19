from pathlib import Path
from collections import OrderedDict

import numpy as np
import pybullet as pb
from robopose.lib3d import Transform, parse_pose_args

from .client import BulletClient


class Body:
    def __init__(self, body_id, scale=1.0, client_id=0):
        self._body_id = body_id
        self._client = BulletClient(client_id)
        self._scale = scale

    @property
    def name(self):
        info = self._client.getBodyInfo(self._body_id)
        return info[-1].decode('utf8')

    @property
    def pose(self):
        return self.pose

    @pose.getter
    def pose(self):
        pos, orn = self._client.getBasePositionAndOrientation(self._body_id)
        return Transform(orn, pos).toHomogeneousMatrix()

    @pose.setter
    def pose(self, pose_args):
        pose = parse_pose_args(pose_args)
        pos, orn = pose.translation, pose.quaternion.coeffs()
        self._client.resetBasePositionAndOrientation(self._body_id, pos, orn)

    @property
    def num_joints(self):
        return self._client.getNumJoints(self._body_id)

    @property
    def joint_infos(self):
        return (self._client.getJointInfo(self._body_id, n) for n in range(self.num_joints))

    @property
    def joint_states(self):
        return (self._client.getJointState(self._body_id, n) for n in range(self.num_joints))

    def link_states(self):
        return (self._client.getLinkState(self._body_id, n) for n in range(self.num_joints))

    @property
    def q_limits(self):
        q_limits = OrderedDict([(info[1].decode('utf8'), (info[8], info[9])) for info in self.joint_infos])
        return q_limits

    @property
    def joint_names(self):
        joint_names = [
            info[1].decode('utf8') for info in self.joint_infos
        ]
        return joint_names

    @property
    def joint_name_to_id(self):
        return {name: n for n, name in enumerate(self.joint_names)}

    @property
    def q(self):
        values = np.array([state[0] for state in self.joint_states])
        q = OrderedDict({name: value for name, value in zip(self.joint_names, values)})
        return q

    @q.setter
    def q(self, joints):
        joint_name_to_id = self.joint_name_to_id
        for joint_name, joint_value in joints.items():
            joint_id = joint_name_to_id.get(joint_name, None)
            if joint_id is not None:
                self._client.resetJointState(self._body_id, joint_id, joint_value)

    def get_state(self):
        return dict(TWO=self.pose,
                    q=self.q,
                    joint_names=self.joint_names,
                    name=self.name,
                    scale=self._scale,
                    body_id=self._body_id)


    @property
    def visual_shape_data(self):
        return self._client.getVisualShapeData(self.body_id)

    @property
    def body_id(self):
        return self._body_id

    @property
    def client_id(self):
        return self.client_id

    @staticmethod
    def load(urdf_path, scale=1.0, client_id=0):
        urdf_path = Path(urdf_path)
        assert urdf_path.exists, 'URDF does not exist.'
        body_id = pb.loadURDF(urdf_path.as_posix(), physicsClientId=client_id, globalScaling=scale)
        return Body(body_id, scale=scale, client_id=client_id)
