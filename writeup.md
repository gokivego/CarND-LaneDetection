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


In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 



###2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


###3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
