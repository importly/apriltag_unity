import os
import cv2
import numpy as np
import glob

aruco = cv2.aruco

squares_x, squares_y = 9, 12
square_size_mm = 20.8 #! TODO fix
marker_length_mm = 15.5 #! TODO fix

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
board = aruco.CharucoBoard((squares_x, squares_y), square_size_mm, marker_length_mm, aruco_dict)

params = aruco.DetectorParameters()
params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
params.cornerRefinementWinSize = 5
params.cornerRefinementMaxIterations = 30
params.cornerRefinementMinAccuracy = 0.1

charuco_corners_all = []
charuco_ids_all = []

images = glob.glob('calibration_images/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"[!] Couldn't load {fname}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=params)
    num_markers = 0 if ids is None else len(ids)
    print(f"{os.path.basename(fname)} → Detected markers: {num_markers}")

    if ids is None or len(ids) < 4:
        continue

    corners, ids, rejected = aruco.refineDetectedMarkers(
        image=gray,
        board=board,
        detectedCorners=corners,
        detectedIds=ids,
        rejectedCorners=rejected
    )[:3]

    ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray,
        board=board
    )
    print(f"    → Charuco corners found: {ret}")

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    aruco.drawDetectedMarkers(vis, corners, ids)
    if ret > 0:
        aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)

    # cv2.imshow('Charuco Calibration', vis)
    # cv2.waitKey(0)

    if ret > 3:
        charuco_corners_all.append(charuco_corners)
        charuco_ids_all.append(charuco_ids)

if charuco_corners_all:
    ret, K, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=charuco_corners_all,
        charucoIds=charuco_ids_all,
        board=board,
        imageSize=gray.shape[::-1],
        cameraMatrix=None,
        distCoeffs=None
    )
    print("\nCalibration successful!")
    print("RMS error:", ret)
    print("Camera matrix (K):\n", K)
    print("Distortion coefficients:\n", distCoeffs.ravel())

    np.save('camera_matrix1.npy', K)
    np.save('dist_coeffs1.npy', distCoeffs)
    fs = cv2.FileStorage('calibration_data1.yaml', cv2.FILE_STORAGE_WRITE)
    fs.write('K', K)
    fs.write('distCoeffs', distCoeffs)
    fs.release()
    print("\nSaved: camera_matrix1.npy, dist_coeffs1.npy, calibration_data1.yaml")
else:
    print("\nNot enough valid Charuco corners detected for calibration.")
