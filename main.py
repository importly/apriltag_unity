import cv2
import numpy as np
import socket
from pupil_apriltags import Detector

cam_index = 1

# ——— USB camera setup ———
cap = cv2.VideoCapture(cam_index)
# set resolution and framerate (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
cap.set(cv2.CAP_PROP_FPS, 50)

# ——— Camera intrinsics ———
# Attempt to load calibration from .npy files
try:
    camera_matrix = np.load('camera_matrix1.npy')
    dist = np.load('dist_coeffs1.npy').ravel()
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    dist_coeffs = dist[:4]
    print("Loaded calibration from .npy files.")
except Exception:
    fs = cv2.FileStorage('calibration_data1.yaml', cv2.FILE_STORAGE_READ)
    if fs.isOpened():
        camera_matrix = fs.getNode('K').mat()
        dist = fs.getNode('distCoeffs').mat().ravel()
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        dist_coeffs = dist[:4]
        fs.release()
        print("Loaded calibration from YAML file.")
    else:
        # Fallback
        ret, tmp_frame = cap.read()
        h, w = tmp_frame.shape[:2]
        fx = fy = w
        cx = w / 2
        cy = h / 2
        dist_coeffs = np.zeros(4)
        print("Using fallback intrinsics.")

camera_params = (fx, fy, cx, cy)

# ——— AprilTag 3 detector ———
detector = Detector(
    families="tag36h11",
    nthreads=24,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)
TAG_SIZE = 0.165

# ——— UDP socket to Unity ———
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
UNITY_IP = "127.0.0.1"
UNITY_PORT = 5065

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=TAG_SIZE
        )

        for det in detections:
            R_cam = det.pose_R
            t_cam = det.pose_t.reshape(3)
            rvec_cam, _ = cv2.Rodrigues(R_cam)
            r_cam = rvec_cam.flatten()
            t_uni = np.array([t_cam[0], -t_cam[1], -t_cam[2]])
            r_uni = np.array([r_cam[0], -r_cam[1], -r_cam[2]])
            msg = (
                f"{det.tag_id},"
                + ",".join(f"{v:.6f}" for v in t_uni)
                + ","
                + ",".join(f"{v:.6f}" for v in r_uni)
            )
            sock.sendto(msg.encode("utf8"), (UNITY_IP, UNITY_PORT))

            cv2.drawFrameAxes(
                frame,
                camera_matrix,
                dist_coeffs,
                rvec_cam,
                t_cam.reshape(3,1),
                0.1
            )

        cv2.imshow("USB Camera + AprilTag3", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    sock.close()

