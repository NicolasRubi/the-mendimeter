import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
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


def mendimeter():
    cap = cv.VideoCapture(0)
    thresh = 100  # initial threshold
    # Create Window
    source_window = "Source"
    cv.namedWindow(source_window)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        (cols, rows, channels) = frame.shape
        # cv.flip(frame, 1, frame)
        f_frame = np.float32(frame)
        # upload image to the GPU.
        f_frame = cv.UMat(f_frame)
        cv.flip(f_frame, 1, f_frame)

        # Our operations on the frame come here
        nrows = cv.getOptimalDFTSize(rows)
        ncols = cv.getOptimalDFTSize(cols)
        nimg = cv.resize(f_frame, (nrows, ncols))

        gray = cv.cvtColor(nimg, cv.COLOR_RGBA2GRAY, 0)
        cv.blur(gray, (3, 3), gray)

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
        sobelx8u = cv.Sobel(gray_8u, cv.CV_8U, 1, 0, ksize=5)
        sobelx64f = cv.Sobel(gray_8u, cv.CV_64F, 1, 0, ksize=5)
        sobel_8u = cv.convertScaleAbs(sobelx64f)

        max_thresh = 255

        cv.imshow(source_window, frame)

        def thresh_callback(val, src_gray=gray_8u):
            threshold = val

            canny_output = cv.Canny(gray_8u, threshold, threshold * 2)

            contours, _ = cv.findContours(
                canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
            )

            contours_poly = [None] * len(contours)
            boundRect = [None] * len(contours)
            centers = [None] * len(contours)
            radius = [None] * len(contours)
            for i, c in enumerate(contours):
                contours_poly[i] = cv.approxPolyDP(c, 3, True)
                boundRect[i] = cv.boundingRect(contours_poly[i])
                centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])

            shape = canny_output.get().shape
            drawing = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)

            for i in range(len(contours)):
                color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
                cv.drawContours(drawing, contours_poly, i, color)
                cv.rectangle(
                    drawing,
                    (int(boundRect[i][0]), int(boundRect[i][1])),
                    (
                        int(boundRect[i][0] + boundRect[i][2]),
                        int(boundRect[i][1] + boundRect[i][3]),
                    ),
                    color,
                    2,
                )
                cv.circle(
                    drawing,
                    (int(centers[i][0]), int(centers[i][1])),
                    int(radius[i]),
                    color,
                    2,
                )

            cv.imshow("Contours", drawing)

        cv.createTrackbar(
            "Canny thresh:", source_window, thresh, max_thresh, thresh_callback
        )
        edges = cv.Canny(gray_8u, 75, 150)

        cv.imshow("Canny edges", edges)
        cv.imshow("sobel of grayscale", sobel_8u)
        cv.imshow("worse sobel of grayscale", sobelx8u)
        # Display the resulting frame
        cv.imshow(
            "cropped for dft",
            cv.normalize(f_frame, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U),
        )
        # cv.imshow(
        #     "red channel", cv.normalize(r, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        # )
        # cv.imshow(
        #     "green channel", cv.normalize(g, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        # )
        # cv.imshow(
        #     "blue channel", cv.normalize(b, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        # )
        cv.imshow("dft magnitude spectrum", mag_gray)
        # cv.imshow("red channel dft magnitude spectrum", magR)
        # cv.imshow("green channel dft magnitude spectrum", magG)
        # cv.imshow("blue channel dft magnitude spectrum", magB)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    mendimeter()