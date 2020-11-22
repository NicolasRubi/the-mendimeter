import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import yaml
import random as rng

rng.seed(12345)


def crop_spectrum(mag):
    mag_rows, mag_cols = mag.shape
    # crop the spectrum, if it has an odd number of rows or columns
    magR = mag[0 : (mag_rows & -2), 0 : (mag_cols & -2)]
    cx = int(mag_rows / 2)
    cy = int(mag_cols / 2)
    q0 = mag[0:cx, 0:cy]  # Top-Left - Create a ROI per quadrant
    q1 = mag[cx : cx + cx, 0:cy]  # Top-Right
    q2 = mag[0:cx, cy : cy + cy]  # Bottom-Left
    q3 = mag[cx : cx + cx, cy : cy + cy]  # Bottom-Right
    tmp = np.copy(q0)  # swap quadrants (Top-Left with Bottom-Right)
    mag[0:cx, 0:cy] = q3
    mag[cx : cx + cx, cy : cy + cy] = tmp
    tmp = np.copy(q1)  # swap quadrant (Top-Right with Bottom-Left)
    mag[cx : cx + cx, 0:cy] = q2
    mag[0:cx, cy : cy + cy] = tmp
    return mag


def calibrate(source, calibration_file):
    print(
        "Please hold up the checkerboard pattern until enough datapoiints are gathered and press c"
    )
    pattern_size = (9, 6)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    # pattern_points *= square_size

    obj_points = []
    img_points = []
    h, w = 0, 0
    i = -1
    frame_step = 20

    while True:
        i += 1
        if isinstance(source, list):
            # glob
            if i == len(source):
                break
            img = cv.imread(source[i])
        else:
            # cv2.VideoCapture
            retval, img = source.read()
            # if not retval:
            #     break
            if i % frame_step != 0:
                continue

        print("Searching for chessboard in frame " + str(i) + "..."),
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cv.imshow("checker", img)

        if cv.waitKey(1) & 0xFF == ord("c"):
            break

        h, w = img.shape[:2]
        found, corners = cv.findChessboardCorners(
            img, pattern_size, flags=cv.CALIB_CB_FILTER_QUADS
        )
        if found:
            term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
            cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
        if not found:
            print("not found")
            continue
        img_points.append(corners.reshape(1, -1, 2))
        obj_points.append(pattern_points.reshape(1, -1, 3))

        print("ok")
        img_chess = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cv.drawChessboardCorners(img_chess, pattern_size, corners, found)
        cv.imshow("draw on checkerboard", img_chess)

    print("\nPerforming calibration...")
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera(
        obj_points, img_points, (w, h), None, None
    )
    print("RMS: {}".format(rms))
    print("camera matrix:\n{}".format(camera_matrix))
    print("distortion coefficients: {}".format(dist_coefs.ravel()))

    calibration = {
        "rms": rms,
        "camera_matrix": camera_matrix.tolist(),
        "dist_coefs": dist_coefs.tolist(),
    }

    with open(calibration_file, "w") as fw:
        yaml.dump(calibration, fw)
    return (camera_matrix, dist_coefs)


def mendimeter():
    cap = cv.VideoCapture(0)
    # thresh = 100  # initial threshold
    # # Create Window
    # source_window = "Source"
    # cv.namedWindow(source_window)
    calibration_file_name = "camera_calibration.yml"
    calibration = {}
    with open(calibration_file_name, "r") as file:
        calibration = yaml.load(file, Loader=yaml.FullLoader)
        print(calibration)
    if calibration is None:
        (camera_matrix, dist_coefs) = calibrate(cap, calibration_file_name)
    else:
        camera_matrix = calibration.get("camera_matrix")
        dist_coefs = calibration.get("dist_coefs")

    print(camera_matrix, dist_coefs)
    camera_matrix = cv.UMat(np.array(camera_matrix))
    dist_coefs = cv.UMat(np.array(dist_coefs))

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        (cols, rows, channels) = frame.shape

        newcameramtx, roi = cv.getOptimalNewCameraMatrix(
            camera_matrix, dist_coefs, (rows, cols), 1, (rows, cols)
        )

        undistorted_frame = cv.undistort(
            frame, camera_matrix, dist_coefs, None, newcameramtx
        )

        x, y, w, h = roi
        croped_undistorted_frame = undistorted_frame.get()[y : y + h, x : x + w]
        (cols, rows, channels) = croped_undistorted_frame.shape

        # cv.flip(frame, 1, frame)
        f_frame = np.float32(croped_undistorted_frame)
        # upload image to the GPU.
        f_frame = cv.UMat(f_frame)
        cv.flip(f_frame, 1, f_frame)

        # Our operations on the frame come here
        nrows = cv.getOptimalDFTSize(rows)
        ncols = cv.getOptimalDFTSize(cols)
        nimg = cv.resize(f_frame, (nrows, ncols))

        gray = cv.cvtColor(nimg, cv.COLOR_RGBA2GRAY, 0)
        # cv.blur(gray, (2, 2), gray)

        # b, g, r = cv.split(nimg)

        gray_dft = cv.dft(gray, flags=cv.DFT_COMPLEX_OUTPUT)

        # r_dft = cv.dft(
        #     r, flags=cv.DFT_COMPLEX_OUTPUT
        # )  # this way the result may fit in the source matrix
        # g_dft = cv.dft(
        #     g, flags=cv.DFT_COMPLEX_OUTPUT
        # )  # also doing these inplace for performance.
        # b_dft = cv.dft(b, flags=cv.DFT_COMPLEX_OUTPUT)

        # Extract magnitude and phase angle.
        mag_gray, angle_gray = cv.cartToPolar(
            gray_dft.get()[:, :, 0], gray_dft.get()[:, :, 1]
        )
        # magR, angleR = cv.cartToPolar(r_dft.get()[:, :, 0], r_dft.get()[:, :, 1])
        # magG, angleG = cv.cartToPolar(g_dft.get()[:, :, 0], g_dft.get()[:, :, 1])
        # magB, angleB = cv.cartToPolar(b_dft.get()[:, :, 0], b_dft.get()[:, :, 1])

        mag_gray = cv.UMat(mag_gray)
        # magR = cv.UMat(magR)
        # magG = cv.UMat(magG)
        # magB = cv.UMat(magB)

        #  switch to logarithmic scale
        cv.log(mag_gray, mag_gray)
        # cv.log(magR, magR)
        # cv.log(magG, magG)
        # cv.log(magB, magB)

        cv.divide(mag_gray, 30.0, mag_gray)
        # cv.divide(magR, 30.0, magR)
        # cv.multiply(magG, 1.0 / 30.0, magG)
        # cv.multiply(magB, 1.0 / 30.0, magB)

        mag_gray = cv.UMat(crop_spectrum(mag_gray.get()))
        # magR = cv.UMat(crop_spectrum(magR.get()))
        # magG = cv.UMat(crop_spectrum(magG.get()))
        # magB = cv.UMat(crop_spectrum(magB.get()))

        gray_8u = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        # sobelx8u = cv.Sobel(gray_8u, cv.CV_8U, 1, 0, ksize=5)
        # sobelx64f = cv.Sobel(gray_8u, cv.CV_64F, 1, 0, ksize=5)
        # sobel_8u = cv.convertScaleAbs(sobelx64f)

        edges = cv.Canny(gray_8u, 75, 150)

        cv.imshow("Canny edges", edges)
        # Display the resulting frame
        cv.imshow(
            "cropped for dft",
            cv.normalize(f_frame, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U),
        )

        cv.imshow("dft magnitude spectrum", mag_gray)

        white_points = cv.findNonZero(edges)
        white_cpu = white_points.get()

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    mendimeter()