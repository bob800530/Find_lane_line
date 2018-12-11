# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 22:57:55 2018

@author: Bob
"""

# Do relevant imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Read in an image
image = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')

gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

# Define a kernel size and apply Gaussian smoothing
kernel_size = 3
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Next we'll create a masked edges image using cv2.fillPoly()
mask = np.zeros_like(edges)   
ignore_mask_color = 255   

# This time we are defining a four sided polygon to mask
imshape = image.shape
vertices = np.array([[(0,imshape[0]),(450, 350), (490, 350), (imshape[1],imshape[0])]], dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 20     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 40 #minimum number of pixels making up a line
max_line_gap = 20    # maximum gap in pixels between connectable line segments
line_image = np.copy(image)*0 # creating a blank to draw lines on
red_line_image = np.copy(image)*0
# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

# Iterate over the output "lines" and draw lines on a blank image
for line in lines:
    for x1,y1,x2,y2 in line:
        """m = -(y2-y1)/(x2-x1)
        if m<-0.1 :
            cv2.line(line_image,(x1,y1),(x1+int(200*m),y1+int(-200)),(255,0,0),10)"""
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
        

gray_line_img = cv2.cvtColor(line_image,cv2.COLOR_RGB2GRAY)

rows,cols=gray_line_img.shape
max_y = 0
min_y = 539
max_x = 0
min_x = 959
line1_topX = 0;
line1_topY = 0;
line1_botX = 0;
line1_botY = 0;

line1_rightX = 0;
line1_rightY = 0;
line1_leftX = 0;
line1_leftY = 0;

for i in range(rows):
    for j in range(int(cols/2)):
        if gray_line_img[i,j]>70:             
             if i<min_y:
                 min_y = i
                 line1_topX = j
                 line1_topY = i
             if i>max_y:
                 max_y = i
                 line1_botX = j
                 line1_botY = i
             if j<min_x:
                 min_x = j
                 line1_leftX = j
                 line1_leftY = i
             if j>max_x:
                 max_x = j
                 line1_rightX = j
                 line1_rightY = i
                 
left_high_pointX = int((line1_topX + line1_rightX)/2)
left_high_pointY = int((line1_topY + line1_rightY)/2)
left_low_pointX = int((line1_botX + line1_leftX)/2)
left_low_pointY = int((line1_botY + line1_leftY)/2)
cv2.line(line_image,(left_high_pointX,left_high_pointY),(left_low_pointX,left_low_pointY),(255,255,0),10)       
           
max_y = 0
min_y = 539
max_x = 0
min_x = 959
line2_topX = 0;
line2_topY = 0;
line2_botX = 0;
line2_botY = 0;

line2_rightX = 0;
line2_rightY = 0;
line2_leftX = 0;
line2_leftY = 0;
for i in range(rows):
    for j in range(int(cols/2),cols):
        if gray_line_img[i,j]>70:             
             if i<min_y:
                 min_y = i
                 line2_topX = j
                 line2_topY = i
             if i>max_y:
                 max_y = i
                 line2_botX = j
                 line2_botY = i
             if j<min_x:
                 min_x = j
                 line2_leftX = j
                 line2_leftY = i
             if j>max_x:
                 max_x = j
                 line2_rightX = j
                 line2_rightY = i

right_high_pointX = int((line2_topX + line2_leftX)/2)
right_high_pointY = int((line2_topY + line2_leftY)/2)
right_low_pointX = int((line2_botX + line2_rightX)/2)
right_low_pointY = int((line2_botY + line2_rightY)/2)
cv2.line(line_image,(right_high_pointX,right_high_pointY),(right_low_pointX,right_low_pointY),(255,255,0),10) 

m =  (right_high_pointY-right_low_pointY) / (right_high_pointX - right_low_pointX)
bot_d = 540 - right_low_pointY
top_d = right_low_pointY - 350
bot_x = right_low_pointX + bot_d/m;
top_x = right_low_pointX - top_d/m;
cv2.line(line_image,(int(bot_x),540),(int(top_x),350),(255,0,255),10) 
#取兩個正方形 然後取平均          
"""
red_lines = cv2.HoughLinesP(gray_line_img, rho*10, theta*5 ,threshold*20, np.array([]),
                        min_line_length*3, max_line_gap-20)
for red_line in red_lines:
    for x1,y1,x2,y2 in red_line:
        cv2.line(red_line_image,(x1,y1),(x2,y2),(255,255,0),10)"""
            
# Create a "color" binary image to combine with line image
color_edges = np.dstack((edges, edges, edges)) 

# Draw the lines on the edge image
#lines_edges = cv2.addWeighted(color_edges, 0.8, red_line_image, 1, 0) 
lines_edges = cv2.addWeighted(color_edges, 1, line_image, 1, 0) 
plt.imshow(lines_edges)