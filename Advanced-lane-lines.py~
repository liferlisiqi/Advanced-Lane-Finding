import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
from moviepy.editor import VideoFileClip


def undistort():
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].transpose().reshape(-1, 2)

    # arrays to store object points and image points
    objpoints = []
    imgpoints = []

    images = glob.glob("camera_cal/cal*.jpg")
    for frame in images:
        img = cv2.imread(frame)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    img = cv2.imread("camera_cal/calibration1.jpg")
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return mtx, dist


# mtx, dist = undistort()


def region_of_interest(img):
    mask = np.zeros_like(img)

    xsize = img.shape[1]
    ysize = img.shape[0]
    left_bottom = (180, ysize - 48)
    left_top = (xsize / 2 - 80, ysize / 2 + 100)
    right_bottom = (xsize - 20, ysize - 48)
    right_top = (xsize / 2 + 80, ysize / 2 + 100)
    vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)

    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def line_vertexs(img, lines):
    xsize = img.shape[1]
    ysize = img.shape[0]
    x_middle = xsize / 2
    left_bottom = [0, 0]
    left_top = [0, ysize]
    right_bottom = [0, 0]
    right_top = [0, ysize]

    for line in lines:
        if abs(line[0][0] - line[0][2]) > 2:
            k = (line[0][3] - line[0][1]) * 1.0 / (line[0][2] - line[0][0])
            if line[0][0] < x_middle and k < 0:
                if line[0][1] > left_bottom[1]:
                    left_bottom = [line[0][0], line[0][1]]
                if line[0][3] < left_top[1]:
                    left_top = [line[0][2], line[0][3]]
            elif line[0][2] > x_middle and k > 0:
                if line[0][1] < right_top[1]:
                    right_top = [line[0][0], line[0][1]]
                if line[0][3] > right_bottom[1]:
                    right_bottom = [line[0][2], line[0][3]]

    if left_top[0] - left_bottom[0] < 0.00001:
        k_left = (left_top[1] - left_bottom[1]) * 1.0 / 0.00001
    else:
        k_left = (left_top[1] - left_bottom[1]) * 1.0 / (left_top[0] - left_bottom[0])

    if right_bottom[0] - right_top[0] < 0.00001:
        k_right = -(right_top[1] - right_bottom[1]) * 1.0 / 0.00001
    else:
        k_right = (right_top[1] - right_bottom[1]) * 1.0 / (right_top[0] - right_bottom[0])

    left_bottom = [int(left_top[0] - (left_top[1] - ysize) / k_left), ysize]
    right_bottom = [int(right_top[0] - (right_top[1] - ysize) / k_right), ysize]

    return left_top, left_bottom, right_top, right_bottom


def divide_lines(img, lines):
    x_middle = img.shape[1] / 2
    all_left_lines = []
    all_right_lines = []
    left_lines = []
    right_lines = []
    for line in lines:
        if abs(line[0][0] - line[0][2]) > 2:
            k = (line[0][3] - line[0][1]) * 1.0 / (line[0][2] - line[0][0])
            if line[0][0] < x_middle and k < -0.5:
                all_left_lines.append(line[0])
            elif line[0][2] > x_middle and k > 0.5:
                all_right_lines.append(line[0])
    all_left_lines.sort(key=lambda x: x[0])
    all_right_lines.sort(key=lambda x: x[0])

    for line in all_left_lines:
        if len(left_lines) != 0:
            if line[0] > left_lines[-1][2] and line[1] < left_lines[-1][3]:
                left_lines.append([left_lines[-1][2], left_lines[-1][3], line[0], line[1]])
                left_lines.append([line[0], line[1], line[2], line[3]])
        else:
            left_lines.append([line[0], line[1], line[2], line[3]])

    for line in all_right_lines:
        if len(right_lines) != 0:
            if line[0] > right_lines[-1][2] and line[1] > right_lines[-1][3]:
                right_lines.append([right_lines[-1][2], right_lines[-1][3], line[0], line[1]])
                right_lines.append([line[0], line[1], line[2], line[3]])
        else:
            right_lines.append([line[0], line[1], line[2], line[3]])

    return left_lines, right_lines


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        cv2.line(img, (line[0], line[1]), (line[2], line[3]), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    left_lines, right_lines = divide_lines(img, lines)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, left_lines, [255, 0, 0], 10)
    draw_lines(line_img, right_lines, [0, 255, 0], 10)

    return line_img


def x_gradient(img):
    x_gra = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    edges = cv2.convertScaleAbs(x_gra)
    return edges


def warp(img):
    xsize = img.shape[1]
    ysize = img.shape[0]
    left_bottom = (180, ysize - 48)
    left_top = (xsize / 2 - 80, ysize / 2 + 100)
    right_bottom = (xsize - 20, ysize - 48)
    right_top = (xsize / 2 + 80, ysize / 2 + 100)
    pts1 = np.float32([left_top, right_top, left_bottom, right_bottom])
    pts2 = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])
    mtx = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, mtx, (500, 500))
    # dst = x_gradient(dst)
    dst = cv2.dilate(dst, (5, 5))
    return dst


def re_warp(img):
    xsize = 320
    ysize = 180
    left_bottom = (45, ysize - 12)
    left_top = (xsize / 2 - 20, ysize / 2 + 25)
    right_bottom = (xsize - 5, ysize - 12)
    right_top = (xsize / 2 + 20, ysize / 2 + 25)
    pts1 = np.float32([left_top, right_top, left_bottom, right_bottom])
    pts2 = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])
    mtx = cv2.getPerspectiveTransform(pts2, pts1)
    dst = cv2.warpPerspective(img, mtx, (xsize, ysize))
    dst = cv2.resize(dst, None, fx=4, fy=4)
    return dst


def historgram(img):
    xsize = img.shape[1]
    ysize = img.shape[0]
    hist = [0 for i in range(xsize)]
    for i in range(50, 150):
        for j in range(ysize):
            hist[i] += img[j][i]
    for i in range(400, 500):
        for j in range(ysize):
            hist[i] += img[j][i]
    return hist


def get_peaks(his):
    left_peak = 0
    right_peak = len(his)
    for i in range(0, len(his) / 2):
        if his[i] > his[left_peak]:
            left_peak = i

    for i in range(len(his) / 2, len(his)):
        if his[i] > his[right_peak]:
            right_peak = i
    return left_peak, right_peak


def sliding_window(img):
    # left_peak, right_peak = get_peaks(img)
    size = img.shape[1]
    l_img = img[:, 50:150]
    r_img = img[:, 380:480]
    lx = []
    ly = []
    rx = []
    ry = []

    for y in range(size):
        n = 0
        s = 0
        for x in range(50, 150):

            if img[y][x] > 0:
                s += x
                n += 1

        if n != 0:
            lx.append(s / n)
            ly.append(y)

        n = 0
        s = 0
        for x in range(380, 480):
            if img[y][x] > 0:
                s += x
                n += 1

        if n != 0:
            rx.append(s / n)
            ry.append(y)

    return lx, ly, rx, ry


def polyfit_2(img):
    lxo, lyo, rxo, ryo = sliding_window(img)
    size = img.shape[0]
    yp = np.arange(1, size, 1)
    zl = np.polyfit(lyo, lxo, 1)
    zr = np.polyfit(ryo, rxo, 1)
    lxp = np.polyval(zl, yp)
    rxp = np.polyval(zr, yp)
    img2 = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size - 2):
        cv2.line(img2, (int(lxp[y]), y), (int(lxp[y + 1]), y + 1), color=[255, 0, 0], thickness=10)
        cv2.line(img2, (int(rxp[y]), y), (int(rxp[y + 1]), y + 1), color=[0, 255, 0], thickness=10)

    return img2


def process_image(image):
    # udst = cv2.undistort(image, mtx, dist, None, mtx)
    roi = region_of_interest(image)
    bird = warp(roi)
    hls = cv2.cvtColor(bird, cv2.COLOR_RGB2HLS)
    sta = hls[:, :, 2]
    light = hls[:, :, 1]
    # blur = cv2.GaussianBlur(sta, (5, 5), 0)
    yellow = cv2.Canny(sta, 50, 200)
    white = cv2.Canny(light, 100, 150)
    canny = cv2.addWeighted(yellow, 1.0, white, 1.0, 0.0)
    edges = x_gradient(canny)
    poly = polyfit_2(edges)
    unbird = re_warp(poly)
    result = cv2.addWeighted(image, 0.8, unbird, 1.0, 0.0)

    # plt.subplot(4, 4, 1), plt.imshow(image)
    # plt.title("origin"), plt.xticks([]), plt.yticks([])
    # plt.subplot(4, 4, 2), plt.imshow(roi, cmap='gray')
    # plt.title("roi"), plt.xticks([]), plt.yticks([])
    # plt.subplot(4, 4, 3), plt.imshow(bird, cmap='gray')
    # plt.title("bird"), plt.xticks([]), plt.yticks([])
    # plt.subplot(4, 4, 4), plt.imshow(sta, cmap='gray')
    # plt.title("S"), plt.xticks([]), plt.yticks([])
    # plt.subplot(4, 4, 5), plt.imshow(yellow, cmap='gray')
    # plt.title("yellow"), plt.xticks([]), plt.yticks([])
    # plt.subplot(4, 4, 6), plt.imshow(light, cmap='gray')
    # plt.title("L"), plt.xticks([]), plt.yticks([])
    # plt.subplot(4, 4, 7), plt.imshow(white, cmap='gray')
    # plt.title("white"), plt.xticks([]), plt.yticks([])
    # plt.subplot(4, 4, 8), plt.imshow(canny, cmap='gray')
    # plt.title("canny"), plt.xticks([]), plt.yticks([])
    # plt.subplot(4, 4, 9), plt.imshow(edges, cmap='gray')
    # plt.title("edges"), plt.xticks([]), plt.yticks([])
    # plt.subplot(4, 4, 10), plt.imshow(poly, cmap='gray')
    # plt.title("poly"), plt.xticks([]), plt.yticks([])
    # plt.subplot(4, 4, 11), plt.imshow(result)
    # plt.title("result"), plt.xticks([]), plt.yticks([])
    # plt.show()
    return result


# images = glob.glob("test_images/*.jpg")
#
# for frame in images:
#     image = cv2.imread(frame)
#     process_image(image)

clip = VideoFileClip("project_video.mp4")
output = "project_result.mp4"
line_clip = clip.fl_image(process_image)
line_clip.write_videofile(output, audio=False)
