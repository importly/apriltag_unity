#!/usr/bin/env python3
import cv2
import numpy as np
import math
import time
import socket
from pupil_apriltags import Detector

# ——— 3D TRANSFORM CLASSES ———

class Translation3D:
    def __init__(self, x: float, y: float, z: float):
        self.x, self.y, self.z = x, y, z

    def as_vector(self):
        return np.array([self.x, self.y, self.z], dtype=float)

    def __repr__(self):
        return f"Translation3D(x={self.x}, y={self.y}, z={self.z})"

class Rotation3D:
    def __init__(self, roll: float, pitch: float, yaw: float):
        self.roll, self.pitch, self.yaw = roll, pitch, yaw

    def to_matrix(self):
        cr, sr = math.cos(self.roll),  math.sin(self.roll)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cy, sy = math.cos(self.yaw),   math.sin(self.yaw)
        Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
        Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
        Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
        return Rz @ Ry @ Rx

    @staticmethod
    def from_matrix(R: np.ndarray):
        sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
        singular = sy < 1e-6
        if not singular:
            roll  = math.atan2(R[2,1],  R[2,2])
            pitch = math.atan2(-R[2,0], sy)
            yaw   = math.atan2(R[1,0],  R[0,0])
        else:
            roll  = math.atan2(-R[1,2], R[1,1])
            pitch = math.atan2(-R[2,0], sy)
            yaw   = 0.0
        return Rotation3D(roll, pitch, yaw)

    def __repr__(self):
        return f"Rotation3D(roll={self.roll}, pitch={self.pitch}, yaw={self.yaw})"

class Pose3D:
    def __init__(self, translation: Translation3D, rotation: Rotation3D):
        self.t = translation
        self.R = rotation

    def inverse(self):
        Rm = self.R.to_matrix().T
        tm = -Rm @ self.t.as_vector()
        return Pose3D(Translation3D(*tm), Rotation3D.from_matrix(Rm))

    def transform_by(self, other: "Pose3D"):
        R1 = self.R.to_matrix()
        R12 = R1 @ other.R.to_matrix()
        t12 = R1 @ other.t.as_vector() + self.t.as_vector()
        return Pose3D(Translation3D(*t12), Rotation3D.from_matrix(R12))

    def __repr__(self):
        return f"Pose3D(translation={self.t}, rotation={self.R})"

# ——— 2D ROBOT POSE & ESTIMATOR ———
# (unchanged)

class Pose2D:
    def __init__(self, x: float, y: float, yaw: float):
        self.x, self.y, self.yaw = x, y, yaw

    def __repr__(self):
        return f"Pose2D(x={self.x:.3f}, y={self.y:.3f}, yaw={self.yaw:.3f})"

class PoseEstimator: # rework pose estimator to be over time intervals
    def __init__(self):
        self._meas = []

    def add_measurement(self, pose: Pose2D, timestamp: float, std_devs: tuple[float,float,float]):
        self._meas.append((pose, timestamp, std_devs))

    def estimate(self):
        if not self._meas:
            return None

        xs, ys, sines, coses = [], [], [], []
        wxs, wys, wyaws = [], [], []
        for p, _, (sx, sy, syaw) in self._meas:
            xs.append(p.x); ys.append(p.y)
            coses.append(math.cos(p.yaw)); sines.append(math.sin(p.yaw))
            wxs.append(1/sx**2); wys.append(1/sy**2); wyaws.append(1/syaw**2)

        sum_wx = sum(wxs); sum_wy = sum(wys); sum_wyaw = sum(wyaws)
        x_est = sum(x*w for x,w in zip(xs, wxs)) / sum_wx
        y_est = sum(y*w for y,w in zip(ys, wys)) / sum_wy
        cos_avg = sum(c*w for c,w in zip(coses, wyaws)) / sum_wyaw
        sin_avg = sum(s*w for s,w in zip(sines, wyaws)) / sum_wyaw
        yaw_est = math.atan2(sin_avg, cos_avg)

        return (
            Pose2D(x_est, y_est, yaw_est),
            (math.sqrt(1/sum_wx), math.sqrt(1/sum_wy), math.sqrt(1/sum_wyaw))
        )

# ——— CONFIG ———

USE_TWO_CAMERAS = False
CAMERA_INDEX_1  = 1
CAMERA_INDEX_2  = 2
FRAME_WIDTH     = 1600
FRAME_HEIGHT    = 1200
FRAME_RATE      = 50
TAG_SIZE        = 0.165   # meters

camera_indices = [CAMERA_INDEX_1] + ([CAMERA_INDEX_2] if USE_TWO_CAMERAS else [])

# ——— OPEN & CALIBRATE CAMERAS ———

caps = []
camera_matrices = []
dist_coeffs_list = []
camera_params = []

for cam_num, idx in enumerate(camera_indices, start=1):
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          FRAME_RATE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {idx}")
    caps.append(cap)

    try:
        K = np.load(f'camera_matrix{cam_num}.npy')
        dist = np.load(f'dist_coeffs{cam_num}.npy').ravel()
        print(f"[Cam{cam_num}] Loaded .npy calibration.")
    except:
        fs = cv2.FileStorage(f'calibration_data{cam_num}.yaml', cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise FileNotFoundError(f"No calibration for camera {cam_num}")
        K = fs.getNode('K').mat()
        dist = fs.getNode('distCoeffs').mat().ravel()
        fs.release()
        print(f"[Cam{cam_num}] Loaded YAML calibration.")

    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    camera_matrices.append(K)
    dist_coeffs_list.append(dist[:4])
    camera_params.append((fx, fy, cx, cy))

# ——— APRILTAG DETECTOR ———
detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

# ——— YOUR ROBOT CONFIG ———
camera_poses = [
    Pose3D(Translation3D(0.0, 0.0, 0.0), Rotation3D(0.0,0.0,0.0)),  # cam0
    # … add more if USE_TWO_CAMERAS=True …
]
field_apriltag_poses = {
    1: Pose3D(Translation3D(0.0, 0.0, 0.0), Rotation3D(0.0,0.0,0.0)),
    2: Pose3D(Translation3D(0.0, 0.0, 0.0), Rotation3D(0.0,0.0,0.0)),
    3: Pose3D(Translation3D(0.0, 0.0, 0.0), Rotation3D(0.0,0.0,0.0)),
}
camera_pose_map = {
    cam_idx: camera_poses[i]
    for i, cam_idx in enumerate(camera_indices)
}

def process_vision_measurements(dets_by_cam, timestamp, estimator):
    for cam_idx, dets in dets_by_cam.items():
        cam_mount = camera_pose_map[cam_idx]
        inv_mount = cam_mount.inverse()
        for det in dets:
            tag_pose = field_apriltag_poses.get(det.tag_id)
            if tag_pose is None:
                continue
            cam_in_tag = Pose3D(
                Translation3D(*det.pose_t.reshape(3)),
                Rotation3D.from_matrix(det.pose_R)
            )
            cam_global = tag_pose.transform_by(cam_in_tag)
            robot_global = cam_global.transform_by(inv_mount)
            meas2d = Pose2D(
                robot_global.t.x,
                robot_global.t.y,
                robot_global.R.yaw
            )
            estimator.add_measurement(meas2d, timestamp, (0.05, 0.05, math.radians(3)))

# ——— RESILIENT SPU CONNECTION ———

SPU_HOST = "localhost"
SPU_PORT = 5008

def connect_to_spu(host, port, retry_delay=1.0):
    """Keep trying to connect until successful."""
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print(f"[TCP] Connecting to SPU on {host}:{port}…")
            sock.connect((host, port))
            print("[TCP] ▶︎ Connected to SPU")
            return sock
        except Exception as e:
            print(f"[TCP] ERROR connecting to SPU: {e}. Retrying in {retry_delay}s…")
            time.sleep(retry_delay)

# ——— MAIN LOOP ———

if __name__ == "__main__":
    sock = connect_to_spu(SPU_HOST, SPU_PORT)

    try:
        while True:
            detections_by_camera = {}
            timestamp = time.time()

            # 1) Grab & detect per camera
            for idx, cap in zip(camera_indices, caps):
                ret, frame = cap.read()
                if not ret:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                fx, fy, cx, cy = camera_params[camera_indices.index(idx)]
                dets = detector.detect(
                    gray,
                    estimate_tag_pose=True,
                    camera_params=(fx, fy, cx, cy),
                    tag_size=TAG_SIZE
                )
                detections_by_camera[idx] = dets

                for det in dets:
                    R_cam = det.pose_R
                    t_cam = det.pose_t.reshape(3,1)
                    rvec, _ = cv2.Rodrigues(R_cam)
                    cv2.drawFrameAxes(
                        frame,
                        camera_matrices[camera_indices.index(idx)],
                        dist_coeffs_list[camera_indices.index(idx)],
                        rvec,
                        t_cam,
                        TAG_SIZE * 0.5
                    )

                cv2.imshow(f"Cam {idx}", frame)

            # 2) Fuse all detections this frame
            estimator = PoseEstimator()
            process_vision_measurements(detections_by_camera, timestamp, estimator)
            fused = estimator.estimate()

            if fused:
                pose2d, stddevs = fused
                print(f"[{timestamp:.3f}] Fused pose → {pose2d}, σ = {stddevs}")
                x, y, yaw = pose2d.x, pose2d.y, pose2d.yaw
                line = f"{x:.4f},{y:.4f},{yaw:.4f}\r\n"

                # Attempt to send; on failure, reconnect and retry next loop
                try:
                    sock.sendall(line.encode("utf8"))
                except Exception as e:
                    print(f"[TCP] ERROR sending to SPU: {e}. Reconnecting…")
                    sock.close()
                    sock = connect_to_spu(SPU_HOST, SPU_PORT)
            else:
                print(f"[{timestamp:.3f}] No valid tag detections.")

            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

    finally:
        sock.close()
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()
