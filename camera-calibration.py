import numpy as np
import cv2
import glob

# termination criteria
# whats this?
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, [0, 0, 0], [1, 0, 0], [2, 0, 0],...,[7, 5, 0]
# why should define such a list? why should it be transposed?
# how does transpose and reshape work?
objp = np.zeros((6*8, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:6].transpose().reshape(-1, 2)

# arrays to store object points and image points
objpoints = []
imgpoints = []

images = glob.glob("calibration_wide/GOPR00*.jpg")
for frame in images:
    img = cv2.imread(frame)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find the chess board corners
    # core code of find corners, how does it realize?
    ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

    # if found, add object points, image points
    if ret:
        objpoints.append(objp)

        # refine them? how?
        # corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # draw and display the corners
        # img = cv2.drawChessboardCorners(img, (8, 6), corners2, ret)
        # cv2.imshow("img", img)
        # cv2.waitKey(500)
# cv2.destroyAllWindows()

# where do i define gray?
img = cv2.imread("calibration_wide/test_image.jpg")
img_size = (img.shape[1], img.shape[0])
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

h, w = img.shape[:2]
mtx2, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, mtx)

# crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
cv2.imshow("dst", dst)
cv2.waitKey()
cv2.imwrite("../calibration_wide/test_result.jpg", dst)

