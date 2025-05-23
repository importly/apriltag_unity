import cv2
import numpy as np

# 1. Charuco board specs (the correct 12 columns × 9 rows for your setup)
squares_x       = 9      # number of chessboard squares in X (columns)
squares_y       = 12       # number of chessboard squares in Y (rows)
square_size_mm  = 20.0    # side length of a full square, in millimeters
marker_length_mm= 15.0    # side length of each ArUco marker, in millimeters

# 2. Create the dictionary and board
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
board      = cv2.aruco.CharucoBoard(
    (squares_x, squares_y),
    square_size_mm,
    marker_length_mm,
    aruco_dict
)

# 3. Draw the board to a high-res image (for printing)
#    We convert mm→pixels by choosing, e.g., 10 px per mm → 200 px per square
px_per_mm = 10
img_size = (
    int((squares_x + 1) * square_size_mm * px_per_mm),
    int((squares_y + 1) * square_size_mm * px_per_mm)
)

# Use drawPlanarBoard to render the board:
board_img = cv2.aruco.drawPlanarBoard(
    board,
    img_size,
    marginSize=0,
    borderBits=1
)

cv2.imwrite("calibration_board/charuco_board_12x9.png", board_img)
print("Saved charuco_board_12x9.png — print at 300 dpi for best results")
