#!/usr/bin/env python3
"""
Simple AprilTag detection script for testing individual cameras.
Shows live feed with tag detection overlays including ID, pose, and coordinate axes.
"""

import cv2
import numpy as np
from pupil_apriltags import Detector
import sys


def main():
    # Configuration - modify these variables as needed
    camera_index = 3  # Camera index
    tag_size = 0.165  # AprilTag size in meters
    frame_width = 1600  # Frame width
    frame_height = 1200  # Frame height
    fps = 50  # Frame rate

    # Initialize camera
    cap = cv2.VideoCapture(camera_index-2)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_index}")
        sys.exit(1)

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Get actual camera parameters
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Camera {camera_index} initialized:")
    print(f"  Resolution: {actual_width}x{actual_height}")
    print(f"  FPS: {actual_fps}")
    print(f"  Tag size: {tag_size}m")
    print("\nControls:")
    print("  'q' or ESC - Quit")
    print("  'd' - Toggle debug info")
    print("  'c' - Toggle coordinate axes")

    # Load calibration data - try .npy files first, then YAML
    from pathlib import Path

    cm_npy = Path(f"camera_matrix{camera_index-1}.npy")
    dist_npy = Path(f"dist_coeffs{camera_index-1}.npy")

    if cm_npy.exists() and dist_npy.exists():
        # Load .npy calibration files
        camera_matrix = np.load(cm_npy)
        dist_coeffs = np.load(dist_npy).ravel()
        print(f"Loaded .npy calibration files for camera {camera_index}")
    else:
        # Try YAML calibration file
        fs = cv2.FileStorage(f"calibration_data{camera_index-1}.yaml", cv2.FILE_STORAGE_READ)
        if fs.isOpened():
            camera_matrix = fs.getNode("K").mat()
            dist_coeffs = fs.getNode("distCoeffs").mat().ravel()
            fs.release()
            print(f"Loaded YAML calibration file for camera {camera_index}")
        else:
            # Fallback to default camera matrix
            print(f"Warning: No calibration data found for camera {camera_index}")
            print("Using default camera matrix - pose estimation will be inaccurate")
            fx = fy = actual_width * 0.8  # Rough estimate
            cx, cy = actual_width / 2, actual_height / 2
            camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            dist_coeffs = np.zeros((4, 1))

    # Extract camera parameters for AprilTag detector
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    # Initialize AprilTag detector
    detector = Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )

    # Display flags
    show_debug = True
    show_axes = True

    print("\nStarting detection... Press any key in the video window to begin.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect AprilTags
            detections = detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=(fx, fy, cx, cy),
                tag_size=tag_size
            )

            # Draw detections
            for detection in detections:
                # Draw tag outline
                corners = detection.corners.astype(int)
                cv2.polylines(frame, [corners], True, (0, 255, 0), 2)

                # Draw tag ID
                center = corners.mean(axis=0).astype(int)
                cv2.putText(frame, f"ID: {detection.tag_id}",
                            (center[0] - 20, center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Draw coordinate axes if pose estimation succeeded
                if show_axes and hasattr(detection, 'pose_R') and detection.pose_R is not None:
                    try:
                        # Convert rotation matrix to rotation vector
                        rvec, _ = cv2.Rodrigues(detection.pose_R)
                        tvec = detection.pose_t.reshape(3, 1)

                        # Draw coordinate axes
                        axis_length = tag_size * 0.5
                        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs[:4],
                                          rvec, tvec, axis_length)

                        # Display pose information
                        if show_debug:
                            pose_text = f"X:{tvec[0, 0]:.3f} Y:{tvec[1, 0]:.3f} Z:{tvec[2, 0]:.3f}"
                            cv2.putText(frame, pose_text,
                                        (center[0] - 60, center[1] + 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    except Exception as e:
                        print(f"Error drawing axes: {e}")

            # Add status text
            status_text = f"Camera {camera_index} | Tags: {len(detections)}"
            if show_debug:
                status_text += " | Debug: ON"
            if show_axes:
                status_text += " | Axes: ON"

            cv2.putText(frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show frame
            cv2.imshow(f'AprilTag Detection - Camera {camera_index}', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('d'):
                show_debug = not show_debug
                print(f"Debug info: {'ON' if show_debug else 'OFF'}")
            elif key == ord('c'):
                show_axes = not show_axes
                print(f"Coordinate axes: {'ON' if show_axes else 'OFF'}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Cleanup complete")


if __name__ == "__main__":
    main()