import math
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
    def __init__(self, x: float, y: float, yaw: float):
        self.x = x
        self.y = y
        self.yaw = yaw

    def __repr__(self) -> str:
        return f"Pose2D(x={self.x:.3f}, y={self.y:.3f}, yaw={self.yaw:.3f})"


class PoseEstimator:
    """Very small pose estimator using simple averaging of measurements."""

    def __init__(self) -> None:
        self._measurements = []

    def add_measurement(
        self, pose: Pose2D, timestamp: float, std_devs: tuple[float, float, float]
    ) -> None:
        self._measurements.append((pose, timestamp, std_devs))

    def estimate(self):
        if not self._measurements:
            return None

        xs, ys, sines, coses = [], [], [], []
        wxs, wys, wyaws = [], [], []
        for p, _, (sx, sy, syaw) in self._measurements:
            xs.append(p.x)
            ys.append(p.y)
            coses.append(math.cos(p.yaw))
            sines.append(math.sin(p.yaw))
            wxs.append(1 / sx ** 2)
            wys.append(1 / sy ** 2)
            wyaws.append(1 / syaw ** 2)

        sum_wx = sum(wxs)
        sum_wy = sum(wys)
        sum_wyaw = sum(wyaws)
        x_est = sum(x * w for x, w in zip(xs, wxs)) / sum_wx
        y_est = sum(y * w for y, w in zip(ys, wys)) / sum_wy
        cos_avg = sum(c * w for c, w in zip(coses, wyaws)) / sum_wyaw
        sin_avg = sum(s * w for s, w in zip(sines, wyaws)) / sum_wyaw
        yaw_est = math.atan2(sin_avg, cos_avg)

        return (
            Pose2D(x_est, y_est, yaw_est),
            (
                math.sqrt(1 / sum_wx),
                math.sqrt(1 / sum_wy),
                math.sqrt(1 / sum_wyaw),
            ),
        )
