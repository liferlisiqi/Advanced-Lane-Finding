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

![alt text][image10] ![alt text][image11]

### Pipeline (single images)

#### 1. Undistort 

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text](./output_images/origin.jpg = 400*) ![alt text][image2]


#### 2. Region of interest
roi
![alt text][image3]


#### 3. Perspective transform

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.
![alt text][image4]

#### 5. Binary

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)
![alt text][image5]


#### 4. Histogram
his  
![alt text][image6]

#### 6. Sliding windows

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:  
![alt text][image7]

#### 7. Fit polynomial
poly  
![alt text][image8]

#### 8. Radius of curvature and position to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 9. Rewarp

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:
![alt text][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_result.mp4)

---

### Discussion

#### 1. Suitable ROI
The fist serious problem I've meet in this project is selecting suitable or even perfect ROI. Cause the second step in my pipeline is get ROI, just behind undistorting the image. If the ROI cannot contain enough pixels of lane line, the polynomial will not fit lane line perfectly. And there is not better way but try again and again. Although I've spent a lot of time on it, if a car is ahead of me, I'll be done.

#### 2. Binary thresh
I combine three kinds or parameters:  ((x-sobel | saturation) & lightness) to transform the warped image to binary. Each paremeter should given the min_thresh and max_thresh, so 6 in total. It is very hard to find the best solution for project.mp4 and it cannot be used for other video.

#### 3. Polynomial
I use second polynomial as suggented in the course, and I've tried to make the left line and right line be paralleled by used weight. But this method it doesn't work, maybe my opinion is right, but I cannot realize it.



Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
