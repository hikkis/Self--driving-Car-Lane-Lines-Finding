
## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Four steps to get the camera matrix and distortion coefficients. 
1. change the chessboard picture to gray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY));
2. find corners and draw the chessboardCorners(cv2.findchessboardCornes; cv2.drawchessboardCorners);
3. calculate the undistorted corners positions and distorted corners positions;
4. get the camera matrix and distortion coefficients and some other camera position parameters(cv2.calibrateCamera);



![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
Based on the code from last questions, cv2.undistort() is needed.
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

For this step, I used Sobel and HLS.
For Sobel, there are five steps while there are three steps for HLS.
For Sobel:
1. Conside the gradient in x direction or y direction. Because Lane lines are nomally vertical. So in the code, I only use x direction.(cv2.Sobel(gray, cv2.CV_64F,1,0); absolute(Sobelx);scaled_sobel; use threshold to choose the gradient);
2. use the magnitude(((Sobelx)**2+(Sobely)**2)**(1/2)) and direction (0, np.pi/2);
3. choose kernel oder number(bigger, smoother);
4. combine the first and second steps;

For HLS:
1. get the HLS color space(cv2.cvtColor(img, cv2.COLOR_RGB2HLS));
2. choose the threshold(normally choose s space);

3. combine Sobel and HLS color space;

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Three steps for perspective transform:
1. choose the four position for both original and goal;
2. calculate the transform(cv2.getPerspectivetransform());
3. perform perspective transform(cv2.warpPerspective(image, M, size, flags=cv2.INTER_LINEAR));

```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 456      | 303, 0        | 
| 253, 697      | 303, 697      |
| 1061, 690     | 1011, 690      |
| 700, 456      | 1011, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Three steps:A. histogram B. Windowsearch C. fit the points
A: histogram to find the initial left and right lane lines position;
B: Windowsearch:
1. initial some parameters(window numbers; window height; left/right lane lines postions; empty lane lines points sets);
2. define some requirements(pix number for a window; the length/size of window);
3. Do 'for' (1. find the center of different windows according to the decreasing of y position and pix numbers; 2. add the related pix into lane lines points sets);
C: fit the points(np.polyfit);


![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Two steps:
1. use the radius formula to get the number for picture(code for left lane line: ((1 + (2 * left_fit_cr[0] * y_eval * meters_per_pixel_y_dir + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0]))
2. according to different country requirement for real lane lines length and width(for example: USA/30 meter length while 3.7 meter width)


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

In the same zip document.

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

In the same zip document.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Problems:
1. some big edge or shadow close to lane lines, especially in the initial time. This may cause the mistake of window search.;
2. big curve road;
3. big sun on the window;

What could I do?
1. For the 1&3 issues, I may find whether there are some special filters;
2. For the three issues, change the pix number in the windows may works?
3. Is there any other Sobel or color space method works well? change the direction of Sobel?



```python

```
