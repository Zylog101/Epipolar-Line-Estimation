# Epipolar-Line-Estimation

Implementation of epipolar line estimation using weak calibration and 8 point algorithm where two views of the same object is considered and and estimation of where a point on one view lies on other view is computerd by reducing the search space for the corresponding points on the views to a line.

**Overview:**
* Read two images from command line
* load them and display next to each other
* user clicks on corresponding points in the image through mouse
* normalizing each point
* Fundamental matrix of normalized points
* Ensure rank 2 matrix
* Fundamental matrix calculation
* Epipolar line calculation and drawing
* Epipole calculation

![image](https://github.com/Zylog101/Epipolar-Line-Estimation/blob/master/Image/ELEstim.JPG)
