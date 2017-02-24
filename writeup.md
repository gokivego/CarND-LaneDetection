#**Finding Lane Lines on the Road** 


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./images_for_writeup/isolated_yellow.jpg "Yellow_lanes_isolated"
[image2]: ./images_for_writeup/grayscale.jpg "Grayscale"
[image3]: ./images_for_writeup/gaussian_blurred.jpg "gaussian_blurred"
[image4]: ./images_for_writeup/canny_image.jpg "Canny Transform Image"
[image5]: ./images_for_writeup/region_of_interest.jpg "Region of Interest"
[image6]: ./images_for_writeup/hough_image.jpg "Hough Transform"
[image7]: ./images_for_writeup/superimposed_hough_image.jpg "Hough Transform Superimposed on Original Image"
[image8]: ./images_for_writeup/final_image.jpg "Final Image with Extrapolated Lines"

---

### Reflection

###1. My pipeline

In order to draw the lines onto lanes I created a find_lanes class. Parameter defaults have been set in the class and if we wish we could pass parameters while creating a class object. This was there is flexibility to pass any values of parameters while creating the class object.

My pipeline consisted of 7 steps:

**Step 1**

First, I began to pre-process the images. To start with, I created a function to isolate the yellow coloured lanes in the image. In order to isolate them I converted the image from RGB space to HSV color space and by doing this I could isolate the yellow coloured objects in the image by specifying a threshold for lower yellow and the upper yellow.

I used the cv2.inRange() function to isolate the lanes and superimposed the lanes onto the original image using the cv2.bitwise_or() function.

I did this step particularly to isolate only yellow because in the challenge video it was becoming difficult to isolate the yellow lanes when the road color was in close contrast to yellow color in the gray scale image. As a result, the pipeline wasn't detecting the yellow images and one of the lanes was not found. 

Below you can see the image that was obtained at the end of the first step.

Yellow Isolated Image
![alt text][image1]

**Step 2**

I converted the image that was obtained from step 1 into grayscale.

Below is an image of the grayscale image.

Grayscale Image
![alt text][image2]

**Step 3**

The idea in this step is to blur the image to remove the unwanted edges from being detected when passed to the canny edge detection algorithm.

I used a gaussian kernel with kernal size of (5,5) to achieve the result.

Below is the image which is a result of gaussian blurring.

Gaussian Blurred Image
![alt text][image3]

**Step 4**

In this step I passed the image from the previous step to get a canny edge image. The canny edge algorithm uses the gradient difference to identify the edges and then thins the edges.

Below is the image of Canny Edge Detection

Canny Image
![alt text][image4]

**Step 5**

In this step we mask the unwanted portion of the image. After applying this step on the canny image we shall notice that only the lanes are visible and the rest of the images is masked as it is nothing more than noise in our analysis of lane detection.

Below is the image of only region of interest.

Region of Interest
![alt text][image5]

**Step 6**

In this step we identify the hough lines which are obtained from the canny image. Once we obtain the hough image we can then superimpose the hough lines onto the original image to get a clearer understanding of what is happening.

Below is the image of hough transform and below that is the image obtained by superimposing the hough transform ont the original image.

Hough Transform
![alt text][image6]

Hough Transform Superimposed on Original Image
![alt text][image7]

**Step 7**

This is the most important step in the whole process. We take the output of the hough transform which is a set of lines and do certain operations to find the best lines to fit the lanes.

To do this I have used the numpy.linalg.lstsq function which takes a set of points as input and gives the slope and intercept of the best fit line. 

Firstly, I identified the lines based on slope('+' slope for right lane and '-' for the left lane) and separated them into two right and left. More precise, I used slopes in the range of (0.4,0.8) and (-0.4,-0.8) as with respect to the camera the lanes are always at an angle which is close to the aformentioned ranges.

After identifying these lines, I passed them to the find_line_eq function to obtain the best fit line. With the slope and intercept I could plot the lines and then superimpose them onto the original image to get the final extrapolated lines.

Below is the final image.
![alt text][image8]


###2. Potential shortcomings with the pipeline

* There is still little bit trouble processing the challenge video. It does run but at a few points the lines jitter and point in arbitrary directions.

* The pipeline is still not robust enough to tackle the changes in road color and shadows that form.

* The pipeline hasn't yet been tested against night driving conditions.

* The pipeline can only produce straight lines which is not the case in reality. It has to be account for the curved roads as well.

* The pipeline scales with the processing power. In my laptop a 27sec video took 29seconds to process which is slower than real time. I would only expect that the processor in the real world car would be faster than mine.


###3. Possible improvements to the pipeline

* There is a lit bit of jitter in the lines that are drawn. I plan on smoothing them out by using running averages.

* There is also a case of solving the challenge video. Although it works, there is still scope for ironing out small errors that could be causing the massive deviation in the lines in some instances.

*The processing speed of the pipeline could be improved slightly by reducing the number of operations that are performed on images.


