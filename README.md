# **Finding Lane Lines on the Road** 
---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps.

1. Converted the images to grayscale 
2. Use GaussianBlur to filter noise in image
3. Use Canny to detect the edges of image
4. Use a four sided polygon to mask two lane lines
5. Use Hough to find two lane lines
6. Extend two land lines

In order to draw a single line on the left and right lanes, I add two functions "extend_lines()", "draw_extend_line()"
extend_lines() could find two end points of each line, then feed this information to "draw_extend_line"
draw_extend_line() use end points to calculate slope, and extend original lane lines



### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when detecting dotted line is unstable.

Another shortcoming could be the failed to detecting lane lines when car turn.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to use more points to compose a line.

Another potential improvement could be to use more points to compose a line and use multiple lines to represent a curve.
