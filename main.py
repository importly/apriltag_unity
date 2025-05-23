import cv2
import numpy as np
import pyrealsense2 as rs
import socket
from pupil_apriltags import Detector

# ——— RealSense setup ———
pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile  = pipeline.start(config)

# fetch intrinsics
color_profile = profile.get_stream(rs.stream.color) \
                       .as_video_stream_profile()
intr = color_profile.get_intrinsics()
fx, fy = intr.fx, intr.fy
cx, cy = intr.ppx, intr.ppy
# use only the first 4 distortion coeffs (k1,k2,p1,p2)
dist_coeffs = np.array(intr.coeffs[:4])

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

TAG_SIZE = 0.165  # meters

# ——— UDP socket to Unity ———
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
UNITY_IP   = "127.0.0.1"
UNITY_PORT = 5065

try:
    while True:
        # grab a color frame
        frames = pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            continue
        frame = np.asanyarray(color.get_data())
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect + pose
        detections = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=TAG_SIZE
        )

        for det in detections:
            # get rotation matrix and translation vector
            R_cam = det.pose_R           # 3×3
            t_cam = det.pose_t.reshape(3)  # (x, y, z) in meters

            # convert R→rvec
            rvec_cam, _ = cv2.Rodrigues(R_cam)
            r_cam = rvec_cam.flatten()  # (rx, ry, rz), radians

            # —— coordinate‐system flip ——
            # OpenCV camera coords:  X→right, Y→down, Z→forward
            # Unity world coords:     X→right, Y→up,   Z→forward
            # so invert Y & Z axes:
            t_uni = np.array([  t_cam[0],
                               -t_cam[1],
                               -t_cam[2] ])
            r_uni = np.array([  r_cam[0],
                               -r_cam[1],
                               -r_cam[2] ])

            # send: tagID, tx,ty,tz, rx,ry,rz
            msg = (
                f"{det.tag_id},"
                + ",".join(f"{v:.6f}" for v in t_uni)
                + ","
                + ",".join(f"{v:.6f}" for v in r_uni)
            )
            sock.sendto(msg.encode("utf8"), (UNITY_IP, UNITY_PORT))

            # (optional) draw for debugging using original cam pose:
            cv2.drawFrameAxes(
                frame,
                np.array([[fx,0,cx],[0,fy,cy],[0,0,1]]),
                dist_coeffs,
                rvec_cam,
                t_cam.reshape(3,1),
                0.1
            )

        cv2.imshow("D455 + Apriltag3", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    sock.close()
