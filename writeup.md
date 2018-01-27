## Advanced Lane Finding Project
---
The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/origin.jpg
[image2]: ./output_images/undist.jpg
[image3]: ./output_images/roi.jpg
[image4]: ./output_images/warp.jpg
[image5]: ./output_images/binary.jpg
[image6]: ./output_images/histogram.png
[image7]: ./output_images/window.png
[image8]: ./output_images/polynomial.png
[image9]: ./output_images/result.jpg
[image10]: ./output_images/chessboard.jpg
[image11]: ./output_images/chessboard_undist.jpg
[video1]: ./project_result.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Computed the camera matrix and distortion coefficients. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. 

### 2. An example of a distortion corrected calibration image.

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:   
Before:  
![alt text][image10] 
After:  
![alt text][image11]

### Pipeline (single images)

#### 1. Undistort 

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:  
Before:  
![alt text][image1]
After:  
![alt text][image2]


#### 2. Region of interest
After undistorting, my next step is not binaring the image as suggested, I choose to find the region of interest instead. In my opinion, it is more easy to process a little part of undistorted image and binary is very important in this project. The coordinate of corner points of ROI is as follows:  
| Point         | coordinate    |
|:-------------:|:-------------:|
| left top      | 560, 450      |
| right top     | 740, 450      |
| right bottom  | 1160, 672     |
| left bottom   | 200, 672      |  
Then, the method to get ROI is the same as in the first project: finding lane lines. The result is as follows:
roi
![alt text][image3]


#### 3. Perspective transform

The code for my perspective transform includes a function called `warper()`.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

Firstly, I use function `getPerspectiveTransform( )` to calculte the transform matrix. `src` is the region to be warped, and of course, this region is ROI. `dst` is the position to put the warped image, I choose an image with 1280 * 720 just as the origin image.
```python
M = cv2.getPerspectiveTransform(src, dst)
```
Then, I'll use funtion `cv2.warpPerspective( )` to transform the ROI 
```python
warped = cv2.warpPerspective(img, M, (1280, 720), flags=cv2.INTER_LINEAR)
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.
![alt text][image4]

In this step, I don't think the lane lines in the warped image must be paralled. Cause the warped image is used for find the warped lane lines, and after, you have rewarp it back. So, it doesn't whether the lane lines are paralled or not, only if you can find the lane lines.

#### 5. Binary
In this step, I'll generate a binary image of the warped ROI. The threshold in my method is a combination of color and gradient. The lane lines are almost vertical, thus, I use the gradient in the x direction to detect them. However, when something unexpected appearing in the image, it may be recognized as lane lines by mistake. To solve this problem, I use saturation channel in HLS color space to perform and operation. Moreover, I also use the lightness channel to improved my method.  

The binary image of the warped ROI is as follows:
![alt text][image5]


#### 4. Histogram
Before finding the pixels of lane lines, finding the region where the lane lines most likely are will make thinkg easier. Calculate the histogram is a way to help us as follows. The two peaks are x coordinate of most pixels of lane lines. 
![alt text][image6]

#### 6. Sliding windows
This step suggested in course is very useful, it slides a window automatically to scan the most likely region of lane line pixels.
![alt text][image7]

Once the pixels of lane lines have been found, there is no need to perform this step every time.

#### 7. Fit polynomial
Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:   
![alt text][image8]

#### 8. Radius of curvature and position to center.
The radius of curvature are calculate by following equation:
$$
R = \frac{[1 + (2Ay + B)^2]^{3/2}}{|2A|}  
$$
And the postion to center is the distance between middle of image(640) and the middel of lane lines at the bottom.

#### 9. Rewarp

The final step is to rewarp the above result to the undistorted image. Then, the pipeline to find lane lines is completed and it will be used to process the video.
![alt text][image9]

---

### Pipeline (video)

I don't how to process a video as in the first project: finding lane line, so I use `extract.py`(./extract.py) to extract every frame and `video.py`(./video.py) to combine these frame after processed.

Here's a [link to my video result](./project_result.mp4)

---

### Discussion

#### 1. Suitable ROI
The fist serious problem I've meet in this project is selecting suitable or even perfect ROI. Cause the second step in my pipeline is get ROI, just behind undistorting the image. If the ROI cannot contain enough pixels of lane line, the polynomial will not fit lane line perfectly. And there is not better way but try again and again. Although I've spent a lot of time on it, if a car is ahead of me, I'll be done.

#### 2. Binary thresh
I combine three kinds or parameters:  ((x-sobel | saturation) & lightness) to transform the warped image to binary. Each paremeter should given the min_thresh and max_thresh, so 6 in total. It is very hard to find the best solution for project.mp4 and it cannot be used for other video.

#### 3. Polynomial
I use second polynomial as suggented in the course, and I've tried to make the left line and right line be paralleled by used weight. But this method it doesn't work, maybe my opinion is right, but I cannot realize it.

 
