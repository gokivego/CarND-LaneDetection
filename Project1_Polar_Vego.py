import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import math


# # Reading the image into the notebook
# image = mpimg.imread('test_images/solidWhiteRight.jpg')
# # Printing out some stats about the image
# print ('The image is of the type ', type(image), 'with dimensions: ', image.shape)
# plt.imshow(image)


def grayscale(img):
    """
    Converts a 3 color channel into grayscale image
    mpimg reads image in RGB format whereas the
    image if read with cv2 is in BGR format so,
    we need to use the appropriate function.
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Since initially we read the image using mpimg
    # return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # if the image is read using cv2


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.

    This is done by creating a polygon with the given vertices and then
    filling the polygon with the color white. When a bitwise_and operator is
    applied on the image and the mask then only that part of the image is
    retained that is desired.

    Example
    Using region_of_interest function
    vertices = np.array(([0,200],[1000,200],[1000,600],[10,600]))
    vertices = vertices.reshape((-1,1,2))
    plt.imshow(region_of_interest(image,[vertices]))

    The region of interest typically lies between the mid region of the image to the bottom of the image.
    So while giving vertices its better to keep this point in view.
    """

    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def equalized_Histogram(img):
    """
    Processes the image for contrast.
    Improves the contrast of the image that is dull.
    It uses a technique called histogram equalization
    """
    return cv2.equalizeHist(img)


def gaussian_blur(img, kernel_size):
    """
    Applies a Gaussian Noise Kernel.
    The kernel_size has to be an odd number
    The output is a blurred image with reduced noise
    Changing the kernel size reduces or increases the noise but there is also a trade off.
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def adaptiveThreshold(img, maxValue, adaptiveMethod, thresholdType, blockSize, C):
    """
    src – Source 8-bit single-channel image.
    maxValue – Non-zero value assigned to the pixels for which the condition is satisfied. See the details below.
    adaptiveMethod – Adaptive thresholding algorithm to use, ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C .
    thresholdType – Thresholding type that must be either THRESH_BINARY or THRESH_BINARY_INV .
    blockSize – Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
    C – Constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may be zero or negative as well.
    """
    return cv2.adaptiveThreshold(img, maxValue, adaptiveMethod, thresholdType, blockSize, C)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def slope(lines):
    """
    Below is a vectorized notation for calculating the slopes and the line lengths for all the lines
    """
    # line length calculation
    pt1, pt2 = np.hsplit(lines.reshape(-1, 2, 2), 2)
    pt2_pt1 = pt2 - pt1  # Subtraction

    # Slope calculation
    s = pt2_pt1.reshape((len(lines), 2, 1))
    slope = s[:, 1] / s[:, 0]
    return slope


def line_eq(pts1, pts2):
    """
    We are going to calculate the line intersection using cramers rule. If we have 2 lines
    (a1x + b1y = c1) and (a2x + b2y = c2) their intersection is given by
    """
    pass


def draw_extrapolated_lines(img, lines, vertices, color=[0, 0, 255], thickness=6):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    y_max = img.shape[0]
    y_min = img.shape[0]

    line_slope = slope(lines)
    left_neg_slope_lines = []
    right_pos_slope_lines = []
    slope_sum_left = 0
    slope_sum_right = 0
    for i in range(len(line_slope)):
        if line_slope[i] > 0.5:
            slope_sum_right += line_slope[i]
            right_pos_slope_lines.append(lines[i])
            y_min = min(min(lines[i,0,1],lines[i,0,3]),y_min)
        elif line_slope[i] < -0.5:
            slope_sum_left += line_slope[i]
            left_neg_slope_lines.append(lines[i])
        else:
            continue

    left_array_length = len(left_neg_slope_lines)
    right_array_length = len(right_pos_slope_lines)
    avg_slope_left = slope_sum_left / left_array_length
    avg_slope_right = slope_sum_right / right_array_length
    left_avg_line = sum(left_neg_slope_lines) // left_array_length
    right_avg_line = sum(right_pos_slope_lines) // right_array_length

    left_angle = -np.arctan(avg_slope_left)
    right_angle = np.arctan(avg_slope_right)
    # print(left_angle, right_angle)

    left_avg_line_mid = (
    (left_avg_line[0][0] + left_avg_line[0][2]) // 2, (left_avg_line[0][1] + left_avg_line[0][3]) // 2)
    right_avg_line_mid = (
    (right_avg_line[0][0] + right_avg_line[0][2]) // 2, (right_avg_line[0][1] + right_avg_line[0][3]) // 2)

    left_line = (
    int(left_avg_line_mid[0] - 250 * np.cos(left_angle)), y_max,
    int(left_avg_line_mid[0] + 150 * np.cos(left_angle)),y_min)

    right_line = (
    int(right_avg_line_mid[0] + 300 * np.cos(right_angle)), y_max,
    int(right_avg_line_mid[0] - 200 * np.cos(right_angle)), y_min)

    # print(left_avg_line, right_avg_line)
    # print(left_line, right_line)

    """
    cv2.line(img, (left_avg_line[0][0], left_avg_line[0][1]), (left_avg_line[0][2], left_avg_line[0][3]), [255, 0, 0],
             thickness)
    cv2.line(img, (right_avg_line[0][0], right_avg_line[0][1]), (right_avg_line[0][2], right_avg_line[0][3]),
             [255, 0, 0], thickness)

    """

    cv2.line(img, (left_line[0], left_line[1]), (left_line[2], left_line[3]), color, thickness)
    cv2.line(img, (right_line[0], right_line[1]), (right_line[2], right_line[3]), color, thickness)

    return img


def draw_lines(img, lines, color=[0, 0, 255], thickness=4):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    """
    The lines files is of the shape (n,1,4) - where 4 represents points [x1,y1,x2,y2].
    Line is drawn from (x1,y1) to (x2,y2)
    """
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    """
    line_img generates a black screen the size of the original image. Now Hough lines are drawn on this
    We need to use the weighted_img function to superimpose this image on the original image
    """
    # draw_lines(line_img, lines)
    return line_img, lines


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

# Preprocessing Steps for the Video

vertices = np.array(([100,540],[400,350],[550,350],[900,540]))
vertices = vertices.reshape((-1,1,2))

# Loading the video using cv2
video = cv2.VideoCapture('solidWhiteRight.mp4')

# Creating an output file for the video
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('Project1_Vego.avi',fourcc,20.0,(540,960))

while(video.isOpened()):
    ret, frame = video.read()
    if frame != None:
        gray = grayscale(frame)
        blur = gaussian_blur(gray,7)
        cannyEdge = canny(blur,120,200)
        roi_cannyEdge = region_of_interest(cannyEdge, [vertices])
        hough_lines_img , lines= hough_lines(img=roi_cannyEdge,rho=1,theta=math.pi/180,threshold = 5,min_line_len=10,max_line_gap=10)
        extrapolated_hough_lines_img = draw_extrapolated_lines(hough_lines_img, lines, vertices)
        extrapolated_final_image = weighted_img(extrapolated_hough_lines_img, frame)


        # Writing to output file
        #out.write(final_image)

        cv2.imshow('final_image',extrapolated_final_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video
video.release()
# out.release()
cv2.destroyAllWindows()
