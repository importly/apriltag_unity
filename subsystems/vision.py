from __future__ import annotations
import cv2
import numpy as np
import socket
import time
import math
from pathlib import Path
from pupil_apriltags import Detector

from .geometry import (
    Pose3D,
    Pose2D,
    Translation3D,
    Rotation3D,
    PoseEstimator,
)
from .scheduler import Subsystem


class VisionSubsystem(Subsystem):
    """Subsystem handling camera capture and AprilTag pose estimation."""

    def __init__(
        self,
        camera_indices: list[int] | None = None,
        tag_size: float = 0.165,
        frame_width: int = 1600,
        frame_height: int = 1200,
        frame_rate: int = 50,
        spu_host: str = "localhost",
        spu_port: int = 5008,
    ) -> None:
        super().__init__()
        if camera_indices is None:
            camera_indices = [1]
        self.camera_indices = camera_indices
        self.tag_size = tag_size
        self.caps: list[cv2.VideoCapture] = []
        self.camera_matrices: list[np.ndarray] = []
        self.dist_coeffs_list: list[np.ndarray] = []
        self.camera_params: list[tuple[float, float, float, float]] = []
        for cam_num, idx in enumerate(camera_indices, start=1):
            cap = cv2.VideoCapture(idx)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
            cap.set(cv2.CAP_PROP_FPS, frame_rate)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera {idx}")
            self.caps.append(cap)
            # try .npy then YAML
            cm_npy = Path(f"camera_matrix{cam_num}.npy")
            dist_npy = Path(f"dist_coeffs{cam_num}.npy")
            if cm_npy.exists() and dist_npy.exists():
                K = np.load(cm_npy)
                dist = np.load(dist_npy).ravel()
                print(f"[Cam{cam_num}] Loaded .npy calibration.")
            else:
                fs = cv2.FileStorage(
                    f"calibration_data{cam_num}.yaml", cv2.FILE_STORAGE_READ
                )
                if not fs.isOpened():
                    raise FileNotFoundError(
                        f"No calibration for camera {cam_num}"
                    )
                K = fs.getNode("K").mat()
                dist = fs.getNode("distCoeffs").mat().ravel()
                fs.release()
                print(f"[Cam{cam_num}] Loaded YAML calibration.")
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            self.camera_matrices.append(K)
            self.dist_coeffs_list.append(dist[:4])
            self.camera_params.append((fx, fy, cx, cy))

        self.detector = Detector(
            families="tag36h11",
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

        self.camera_poses = [
            Pose3D(Translation3D(0.0, 0.0, 0.0), Rotation3D(0.0, 0.0, 0.0))
            for _ in camera_indices
        ]
        self.field_apriltag_poses: dict[int, Pose3D] = {
            1: Pose3D(Translation3D(0.0, 0.0, 0.0), Rotation3D(0.0, 0.0, 0.0)),
            2: Pose3D(Translation3D(0.0, 0.0, 0.0), Rotation3D(0.0, 0.0, 0.0)),
            3: Pose3D(Translation3D(0.0, 0.0, 0.0), Rotation3D(0.0, 0.0, 0.0)),
        }
        self.camera_pose_map = {
            cam_idx: self.camera_poses[i]
            for i, cam_idx in enumerate(camera_indices)
        }

        self.sock = self._connect_to_spu(spu_host, spu_port)
        self.pose_estimator = PoseEstimator()

    # ------------------------------------------------------------------ utils
    def _connect_to_spu(self, host: str, port: int, retry_delay: float = 1.0):
        while True:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                print(f"[TCP] Connecting to SPU on {host}:{port}…")
                sock.connect((host, port))
                print("[TCP] ▶︎ Connected to SPU")
                return sock
            except Exception as e:
                print(
                    f"[TCP] ERROR connecting to SPU: {e}. Retrying in {retry_delay}s…"
                )
                time.sleep(retry_delay)

    # --------------------------------------------------------------- subsystem
    def periodic(self) -> None:  # type: ignore[override]
        detections_by_camera: dict[int, list] = {}
        timestamp = time.time()
        for idx, cap in zip(self.camera_indices, self.caps):
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fx, fy, cx, cy = self.camera_params[self.camera_indices.index(idx)]
            dets = self.detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=(fx, fy, cx, cy),
                tag_size=self.tag_size,
            )
            detections_by_camera[idx] = dets
            for det in dets:
                r_cam = det.pose_R
                t_cam = det.pose_t.reshape(3, 1)
                rvec, _ = cv2.Rodrigues(r_cam)
                cv2.drawFrameAxes(
                    frame,
                    self.camera_matrices[self.camera_indices.index(idx)],
                    self.dist_coeffs_list[self.camera_indices.index(idx)],
                    rvec,
                    t_cam,
                    self.tag_size * 0.5,
                )
            cv2.imshow(f"Cam {idx}", frame)

        self.pose_estimator.update(timestamp, self.pose_estimator.get_estimated_position())
        self._process_vision_measurements(detections_by_camera, timestamp, self.pose_estimator)
        fused = self.pose_estimator.estimate()
        if fused:
            pose2d, stddevs = fused
            print(f"[{timestamp:.3f}] Fused pose → {pose2d}, σ = {stddevs}")
            line = f"{pose2d.x:.4f},{pose2d.y:.4f},{pose2d.yaw:.4f}\r\n"
            try:
                self.sock.sendall(line.encode("utf8"))
            except Exception as e:
                print(f"[TCP] ERROR sending to SPU: {e}. Reconnecting…")
                self.sock.close()
                self.sock = self._connect_to_spu(
                    self.sock.getpeername()[0], self.sock.getpeername()[1]
                )
        else:
            print(f"[{timestamp:.3f}] No valid tag detections.")

    def _process_vision_measurements(self, dets_by_cam, timestamp, estimator):
        for cam_idx, dets in dets_by_cam.items():
            cam_mount = self.camera_pose_map[cam_idx]
            inv_mount = cam_mount.inverse()
            for det in dets:
                tag_pose = self.field_apriltag_poses.get(det.tag_id)
                if tag_pose is None:
                    continue
                cam_in_tag = Pose3D(
                    Translation3D(*det.pose_t.reshape(3)),
                    Rotation3D.from_matrix(det.pose_R),
                )
                cam_global = tag_pose.transform_by(cam_in_tag)
                robot_global = cam_global.transform_by(inv_mount)
                meas2d = Pose2D(
                    robot_global.t.x,
                    robot_global.t.y,
                    robot_global.R.yaw,
                )
                estimator.add_measurement(
                    meas2d, timestamp, (0.05, 0.05, math.radians(3))
                )

    def close(self) -> None:  # type: ignore[override]
        if self.sock:
            self.sock.close()
        for cap in self.caps:
            cap.release()
        cv2.destroyAllWindows()
