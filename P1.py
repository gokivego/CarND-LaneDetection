
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import os
#get_ipython().magic(u'matplotlib inline')


def grayscale(img):
    """
    Converts a 3 color channel into grayscale image
    mpimg reads image in RGB format whereas the
    image if read with cv2 is in BGR format so,
    we need to use the appropriate function.
    """
    return cv2.cvtColor(img,cv2.COLOR_RGB2GRAY) # Since initially we read the image using mpimg
    # return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # if the image is read using cv2
        
def region_of_interest(img,vertices):
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
    
    The region of interest typically lies between the mid region of the image to the bottom of the image. For
    a specific camera angle the region of interest is already defined.
    So while giving vertices its better to keep this point in view.
    """

    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color    
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

def gaussian_blur(img,kernel_size):
    """
    Applies a Gaussian Noise Kernel.
    The kernel_size has to be an odd number
    The output is a blurred image with reduced noise
    Changing the kernel size reduces or increases the noise but there is also a trade off. 
    """
    return cv2.GaussianBlur(img,(kernel_size,kernel_size),0)

def adaptiveThreshold(img,maxValue,adaptiveMethod,thresholdType,blockSize,C):
    """
    src – Source 8-bit single-channel image.
    maxValue – Non-zero value assigned to the pixels for which the condition is satisfied. See the details below.
    adaptiveMethod – Adaptive thresholding algorithm to use, ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C .
    thresholdType – Thresholding type that must be either THRESH_BINARY or THRESH_BINARY_INV .
    blockSize – Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
    C – Constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may be zero or negative as well.
    """
    return cv2.adaptiveThreshold(img,maxValue,adaptiveMethod,thresholdType,blockSize,C)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def draw_lines(img, lines, color=[255, 0, 0], thickness=4):
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
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            

def hough_lines(img, rho, theta, threshold,min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho,theta, threshold,np.array([]),
                            minLineLength= min_line_len ,maxLineGap= max_line_gap) 
                        
    """
    The lines files is of the shape (n,1,4) - where 4 represents points [x1,y1,x2,y2]. 
    Line is drawn from (x1,y1) to (x2,y2)
    """
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    """
    line_img generates a black screen the size of the original image. Now Hough lines are drawn on this
    We need to use the weighted_img function to superimpose this image on the original image
    """
    draw_lines(line_img, lines)
    return line_img, lines 
#     return lines

def weighted_img(img, initial_img,weighted_alpha=0.8, weighted_beta=1., weighted_lamda=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(img,initial_img, weighted_alpha,weighted_beta, weighted_lamda)


class find_lanes(object):
 
    def __init__(self,image,kernel_size = 5,
                 color =[255,0,0],
                 thickness = 10,
                 canny_low = 65,
                 canny_high = 195,
                 hough_rho = 1,
                 hough_theta = np.pi/180, 
                 hough_threshold = 5,
                 hough_min_line_len = 10, 
                 hough_max_line_gap = 5,
                 weighted_alpha = 0.8,
                 weighted_beta = 1.,
                 weighted_lamda = 0.):
        
        self.orig_image = image
        self.processed_image = np.copy(self.orig_image)
        self.kernel_size = kernel_size
        self.color = color
        self.thickness = thickness
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_rho = hough_rho
        self.hough_theta = hough_theta
        self.hough_threshold = hough_threshold
        self.hough_min_line_len = hough_min_line_len
        self.hough_max_line_gap = hough_max_line_gap
        self.weighted_alpha = weighted_alpha
        self.weighted_beta = weighted_beta
        self.weighted_lamda = weighted_lamda
        
        
    def isolate_yellow_white(self):
        """
        This function isolates the yellow lines in the image by converting the RGB image to HSV image
        and isolating the yellow lines, it could also be used to isolate white lines in the image as well.
        """
        img_hsv = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2HSV)
        lower_yellow = np.array([20,100,100],dtype=np.uint8)
        upper_yellow = np.array([40,255,255],dtype=np.uint8)
        
#         lower_white = np.array([0,0,0],dtype=np.uint8)
#         upper_white = np.array([255,5,255],dtype=np.uint8)

        mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow) # Isolate yellow
#         mask_white = cv2.inRange(img_hsv, lower_white, upper_white) # Isolate white
        
        mask_yellow = cv2.cvtColor(mask_yellow, cv2.COLOR_GRAY2RGB)
#         mask_white = cv2.cvtColor(mask_white, cv2.COLOR_GRAY2RGB)
#         isolated_image = cv2.bitwise_or(mask_white, self.processed_image)
#         isolated_image = cv2.bitwise_or(mask_yellow, isolated_image)
        return cv2.bitwise_or(mask_yellow, self.processed_image)
#         plt.imshow(isolated_image)
#         return isolated_image

    def pre_process_image(self):
        
#         plt.subplots(nrows =3, ncols = 3)
#         plt.tight_layout()
        
#         plt.subplot(331)
#         plt.title('Original Image')
#         plt.imshow(self.processed_image)
        
        self.processed_image = self.isolate_yellow_white()
#         mpimg.imsave('images_for_writeup/isolated_yellow.jpg', self.processed_image)
#         plt.subplot(332)
#         plt.title('Yellow Isolated Image')
#         plt.imshow(self.processed_image)
        
        self.processed_image = grayscale(self.processed_image)
#         mpimg.imsave('images_for_writeup/grayscale.jpg', self.processed_image, cmap= 'gray')
#         plt.subplot(333)
#         plt.title('Grayscale Image')
#         plt.imshow(self.processed_image, cmap= 'gray')
        
        self.processed_image = gaussian_blur(self.processed_image, self.kernel_size)
#         mpimg.imsave('images_for_writeup/gaussian_blurred.jpg', self.processed_image, cmap = 'gray')

#         plt.subplot(334)
#         plt.title('Gaussian Blurred Image')
#         plt.imshow(self.processed_image, cmap = 'gray')        
        
    def find_line_eq(self,lines):
        """
        This function finds the lanes from a set of points. Since the argument that is passed to 
        this function lines is a bunch of points, all the function does is calculates the least squares
        regression line made by the points
        """
        # linear regression least squares alogrithm
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html
        lines_array = np.asarray(lines)
        x = np.reshape(lines_array[:, [0, 2]], (1, len(lines) * 2))[0]
        y = np.reshape(lines_array[:, [1, 3]], (1, len(lines) * 2))[0]
        A = np.vstack([x, np.ones(len(x))]).T
        m,c = np.linalg.lstsq(A, y)[0]
        x = np.array(x)
        y = np.array(x*m + c)
        return x, y, m, c

        
    def find_vertices(self):
        """
        This functions helps define the vertices for the region of the lines that we are interested in
        """
        height, width,_ = self.orig_image.shape
        lower_left_vertex = [0.20*width,0.9*height]
        upper_left_vertex = [0.43*width,0.6*height]
        upper_right_vertex = [0.57*width,0.6*height]
        lower_right_vertex = [0.90*width,0.9*height]
        self.vertices = np.array([lower_left_vertex,upper_left_vertex,upper_right_vertex,lower_right_vertex], np.int32)
        
    def find_lines(self):
        """
        This function does all the pre-processing steps to feed the processed image into hough_lines
        """
        # Does all the pre-processing steps
        self.pre_process_image()
        
        self.processed_image = canny(self.processed_image, self.canny_low, self.canny_high)
#         mpimg.imsave('images_for_writeup/canny_image.jpg', self.processed_image, cmap = 'gray')

#         plt.subplot(335)
#         plt.title('Canny Image')
#         plt.imshow(self.processed_image, cmap = 'gray')
        
        self.find_vertices()
        self.processed_image = region_of_interest(self.processed_image, [self.vertices])
#         mpimg.imsave('images_for_writeup/region_of_interest.jpg', self.processed_image, cmap = 'gray')

#         plt.subplot(336)
#         plt.title('Region of Interest')
#         plt.imshow(self.processed_image, cmap = 'gray')
        
        
        self.hough_image, self.lines = hough_lines(self.processed_image,
                                        self.hough_rho,
                                        self.hough_theta,
                                        self.hough_threshold,
                                        self.hough_min_line_len,
                                        self.hough_max_line_gap )

#         mpimg.imsave('images_for_writeup/hough_image.jpg',self.hough_image, cmap = 'gray')
        
        
        self.processed_image = cv2.bitwise_or(self.hough_image,
                                            self.orig_image)
        
#         mpimg.imsave('images_for_writeup/superimposed_hough_image.jpg',self.processed_image)

#         self.hough = np.copy(self.hough_image)
#         self.hough = region_of_interest(self.hough, [self.vertices])
#         plt.imshow(self.processed_image)
        
    def find_slopes(self):
        self.slope = []
        for line in self.lines:
            for x1,y1,x2,y2 in line:
                self.slope.append((y2-y1)/(x2-x1))
        
    def draw_extrapolated_lines(self):
        
        self.find_lines()
        
        # Approach 2 - Using The least squares regression line to find extrapolated lines
        
        left_x_y_values = []

        right_x_y_values = []

        self.upper_y = self.processed_image.shape[0]
        self.lower_y = self.processed_image.shape[0]
        
        center_x = int(self.processed_image.shape[1]*0.5)

        for line in self.lines:
            for x1,y1,x2,y2 in line:
                slope = (y2-y1)/(x2-x1)
                self.upper_y = min(self.upper_y, y1, y2)
                
                #slope,intercept = np.polyfit((x1,x2),(y1,y2),1)
                if (slope < -0.4 and slope > -0.8 and x1 < center_x and x2 < center_x):
                    left_x_y_values.append([x1,y1,x2,y2])
                elif (slope > 0.4 and slope <0.8 and x1 > center_x and x2 > center_x):
                    right_x_y_values.append([x1,y1,x2,y2])
        
         
        left_line_x, left_line_y, left_line_slope, left_line_intercept = self.find_line_eq(left_x_y_values)

        right_line_x, right_line_y, right_line_slope, right_line_intercept = self.find_line_eq(right_x_y_values)
        
        #print ("left intercept = ",left_line_intercept, "right line intercept = ", right_line_intercept)
        #right_line_intercept = min(right_line_intercept, 0)
        
        lower_left_point = np.array([(self.lower_y - left_line_intercept)/left_line_slope,self.lower_y], dtype= int)
        lower_right_point = np.array([(self.lower_y - right_line_intercept)/right_line_slope,self.lower_y], dtype= int)
        upper_left_point = np.array([(self.upper_y - left_line_intercept)/left_line_slope,self.upper_y], dtype= int)
        upper_right_point = np.array([(self.upper_y - right_line_intercept)/right_line_slope,self.upper_y], dtype= int)
        
        line_image = np.zeros_like(self.orig_image)
        cv2.line(line_image,(lower_left_point[0],lower_left_point[1]), 
                 (upper_left_point[0],upper_left_point[1]), 
                 self.color, 
                 self.thickness)
        
        cv2.line(line_image,(lower_right_point[0],lower_right_point[1]), 
                 (upper_right_point[0],upper_right_point[1]), 
                 self.color, 
                 self.thickness)
        
#         cv2.line(line_image,(center_x,self.lower_y),(center_x,self.upper_y), self.color, 5)
        
#         plt.subplots(1,2)
#         plt.subplot(121)
#         plt.imshow(line_image)
        
        self.final_image = cv2.bitwise_or(line_image, self.orig_image)
#         mpimg.imsave('images_for_writeup/final_image.jpg', self.final_image)

#        self.final_image = region_of_interest(self.final_image, vertices = [self.vertices])
        
        
#         plt.subplot(122)
#         plt.imshow(self.final_image)
          


# # Creating an Image Processing Pipeline


def process_test_images(inp_dir, out_dir):
    files = os.listdir(inp_dir)
    for file in files:
        print ('Processing image file - ',file)
        image_pipeline(file, out_dir)
        
def image_pipeline(file, out_dir):
    image = mpimg.imread('test_images/'+ file)
    img = find_lanes(image)
    img.draw_extrapolated_lines()
    mpimg.imsave(out_dir + file ,img.final_image)
    


# # Video part using moviepy

def process_image(image):
    lane_find = find_lanes(image)
    lane_find.draw_extrapolated_lines()
    return lane_find.final_image

def process_video_file(inp_video, out_video):
    print ('------------------------Processing Video - ' + inp_video + ' ---------------------------')
    output_file = out_video
    clip1 = VideoFileClip(inp_video)
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(output_file, audio=False)


def main():
    process_test_images('test_images/', 'processed_images/')
    process_video_file("solidWhiteRight.mp4", "white.mp4")
    process_video_file("solidYellowLeft.mp4", "yellow.mp4")
    process_video_file("challenge.mp4", "extra.mp4")

if __name__ == "__main__":
    main()





