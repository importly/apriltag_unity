import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional, Dict

import numpy as np

class Translation3D:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def as_vector(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    def __repr__(self) -> str:
        return f"Translation3D(x={self.x}, y={self.y}, z={self.z})"


class Rotation3D:
    def __init__(self, roll: float, pitch: float, yaw: float):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def to_matrix(self) -> np.ndarray:
        cr, sr = math.cos(self.roll), math.sin(self.roll)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)

        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        return rz @ ry @ rx

    @staticmethod
    def from_matrix(matrix: np.ndarray) -> "Rotation3D":
        sy = math.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            roll = math.atan2(matrix[2, 1], matrix[2, 2])
            pitch = math.atan2(-matrix[2, 0], sy)
            yaw = math.atan2(matrix[1, 0], matrix[0, 0])
        else:
            roll = math.atan2(-matrix[1, 2], matrix[1, 1])
            pitch = math.atan2(-matrix[2, 0], sy)
            yaw = 0.0
        return Rotation3D(roll, pitch, yaw)

    def __repr__(self) -> str:
        return (
            f"Rotation3D(roll={self.roll}, pitch={self.pitch}, yaw={self.yaw})"
        )


class Pose3D:
    def __init__(self, translation: Translation3D, rotation: Rotation3D):
        self.t = translation
        self.R = rotation

    def inverse(self) -> "Pose3D":
        rm = self.R.to_matrix().T
        tm = -rm @ self.t.as_vector()
        return Pose3D(Translation3D(*tm), Rotation3D.from_matrix(rm))

    def transform_by(self, other: "Pose3D") -> "Pose3D":
        r1 = self.R.to_matrix()
        r12 = r1 @ other.R.to_matrix()
        t12 = r1 @ other.t.as_vector() + self.t.as_vector()
        return Pose3D(Translation3D(*t12), Rotation3D.from_matrix(r12))

    def __repr__(self) -> str:
        return f"Pose3D(translation={self.t}, rotation={self.R})"


class Pose2D:
    """Lightweight 2D pose representation."""

    def __init__(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0):
        self.x = x
        self.y = y
        self.yaw = yaw

    def __repr__(self) -> str:
        return f"Pose2D(x={self.x:.3f}, y={self.y:.3f}, yaw={self.yaw:.3f})"

    def copy(self) -> "Pose2D":
        return Pose2D(self.x, self.y, self.yaw)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    @staticmethod
    def interpolate(a: "Pose2D", b: "Pose2D", t: float) -> "Pose2D":
        t = max(0.0, min(1.0, t))
        x = a.x + (b.x - a.x) * t
        y = a.y + (b.y - a.y) * t
        yaw_diff = Pose2D._wrap_angle(b.yaw - a.yaw)
        yaw = Pose2D._wrap_angle(a.yaw + yaw_diff * t)
        return Pose2D(x, y, yaw)

    def log(self, other: "Pose2D") -> "Twist2D":
        """Return twist from this pose to ``other``."""
        dx = other.x - self.x
        dy = other.y - self.y
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        dtheta = Pose2D._wrap_angle(other.yaw - self.yaw)
        return Twist2D(local_x, local_y, dtheta)

    def exp(self, twist: "Twist2D") -> "Pose2D":
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        x = self.x + twist.dx * cos_yaw - twist.dy * sin_yaw
        y = self.y + twist.dx * sin_yaw + twist.dy * cos_yaw
        yaw = Pose2D._wrap_angle(self.yaw + twist.dtheta)
        return Pose2D(x, y, yaw)


@dataclass
class Twist2D:
    dx: float
    dy: float
    dtheta: float


class TimeInterpolatableBuffer:
    """Buffer storing timestamped samples with interpolation."""

    def __init__(self, interpolate_func: Callable[[Pose2D, Pose2D, float], Pose2D], history: float):
        self.history = history
        self.interpolate_func = interpolate_func
        self.buffer: "OrderedDict[float, Pose2D]" = OrderedDict()

    def add_sample(self, timestamp: float, sample: Pose2D) -> None:
        self._cleanup(timestamp)
        self.buffer[timestamp] = sample

    def _cleanup(self, current: float) -> None:
        while self.buffer:
            first_ts = next(iter(self.buffer))
            if current - first_ts >= self.history:
                self.buffer.popitem(last=False)
            else:
                break

    def clear(self) -> None:
        self.buffer.clear()

    def get_sample(self, timestamp: float) -> Optional[Pose2D]:
        if not self.buffer:
            return None

        if timestamp in self.buffer:
            return self.buffer[timestamp]

        items = list(self.buffer.items())
        for i, (ts, _) in enumerate(items):
            if ts > timestamp:
                break
        else:
            return items[-1][1]

        if i == 0:
            return items[0][1]

        t0, v0 = items[i - 1]
        t1, v1 = items[i]
        ratio = (timestamp - t0) / (t1 - t0)
        return self.interpolate_func(v0, v1, ratio)

    def internal_buffer(self) -> "OrderedDict[float, Pose2D]":
        return self.buffer


class PoseEstimator:
    """Fuses odometry and vision measurements similar to WPILib's estimator."""

    kBufferDuration = 1.5

    def __init__(
        self,
        state_std_devs: tuple[float, float, float] = (0.02, 0.02, math.radians(1)),
        vision_std_devs: tuple[float, float, float] = (0.1, 0.1, math.radians(3)),
    ) -> None:
        self.q = np.array([s * s for s in state_std_devs])
        self.vision_k = np.zeros(3)
        self.set_vision_measurement_std_devs(vision_std_devs)

        self.pose_estimate = Pose2D()
        self.odometry_buffer = TimeInterpolatableBuffer(Pose2D.interpolate, self.kBufferDuration)
        self.vision_updates: Dict[float, "VisionUpdate"] = OrderedDict()

    def set_vision_measurement_std_devs(self, std_devs: tuple[float, float, float]) -> None:
        r = [d * d for d in std_devs]
        for i in range(3):
            if self.q[i] == 0.0:
                self.vision_k[i] = 0.0
            else:
                self.vision_k[i] = self.q[i] / (self.q[i] + math.sqrt(self.q[i] * r[i]))

    # ------------------------------------------------------------------ reset
    def reset_pose(self, pose: Pose2D) -> None:
        self.pose_estimate = pose.copy()
        self.odometry_buffer.clear()
        self.vision_updates.clear()

    # ------------------------------------------------------------------- state
    def get_estimated_position(self) -> Pose2D:
        return self.pose_estimate

    # ---------------------------------------------------------------- vision
    def _clean_up_vision_updates(self) -> None:
        if not self.odometry_buffer.buffer:
            return

        oldest_ts = next(iter(self.odometry_buffer.buffer))
        if not self.vision_updates or oldest_ts < next(iter(self.vision_updates)):
            return

        newest_needed = max(ts for ts in self.vision_updates if ts <= oldest_ts)
        for ts in list(self.vision_updates.keys()):
            if ts < newest_needed:
                self.vision_updates.pop(ts)

    def sample_at(self, timestamp: float) -> Optional[Pose2D]:
        if not self.odometry_buffer.buffer:
            return None

        oldest = next(iter(self.odometry_buffer.buffer))
        newest = next(reversed(self.odometry_buffer.buffer))
        timestamp = max(min(timestamp, newest), oldest)

        if not self.vision_updates or timestamp < next(iter(self.vision_updates)):
            return self.odometry_buffer.get_sample(timestamp)

        floor_ts = max(ts for ts in self.vision_updates if ts <= timestamp)
        vision_update = self.vision_updates[floor_ts]
        odometry_est = self.odometry_buffer.get_sample(timestamp)
        if odometry_est is None:
            return None
        return vision_update.compensate(odometry_est)

    def add_measurement(
        self,
        vision_pose: Pose2D,
        timestamp: float,
        std_devs: tuple[float, float, float] | None = None,
    ) -> None:
        if std_devs is not None:
            self.set_vision_measurement_std_devs(std_devs)
        self.add_vision_measurement(vision_pose, timestamp)

    def add_vision_measurement(self, vision_pose: Pose2D, timestamp: float) -> None:
        if not self.odometry_buffer.buffer or (
            next(reversed(self.odometry_buffer.buffer)) - self.kBufferDuration > timestamp
        ):
            return

        self._clean_up_vision_updates()

        odom_sample = self.odometry_buffer.get_sample(timestamp)
        if odom_sample is None:
            return

        vision_sample = self.sample_at(timestamp)
        if vision_sample is None:
            return

        twist = vision_sample.log(vision_pose)
        k_times_twist = Twist2D(
            self.vision_k[0] * twist.dx,
            self.vision_k[1] * twist.dy,
            self.vision_k[2] * twist.dtheta,
        )
        vision_update = VisionUpdate(vision_sample.exp(k_times_twist), odom_sample)

        self.vision_updates[timestamp] = vision_update
        # remove later updates
        for ts in list(self.vision_updates.keys()):
            if ts > timestamp:
                self.vision_updates.pop(ts)

        self.pose_estimate = vision_update.compensate(self.pose_estimate)


    def estimate(self) -> Pose2D:
        """Return the latest pose estimate."""
        return self.pose_estimate


class VisionUpdate:
    def __init__(self, vision_pose: Pose2D, odometry_pose: Pose2D):
        self.vision_pose = vision_pose
        self.odometry_pose = odometry_pose

    def compensate(self, pose: Pose2D) -> Pose2D:
        delta = pose.log(self.odometry_pose)
        return self.vision_pose.exp(delta)
