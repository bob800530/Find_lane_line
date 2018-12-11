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
import os 

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
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
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
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

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def draw_extend_line(extend_img,x1, y1, x2, y2):
    m =  (y2-y1) / (x2 - x1)
    bot_d = 540 - y1
    top_d = y1 - 350
    bot_x = x1 + bot_d/m;
    top_x = x1 - top_d/m;
    cv2.line(extend_img,(int(bot_x),540),(int(top_x),350),(255,0,255),10) 
    return extend_img
    
def extend_lines(extend_img, line_img, side):
    gray_line_img = cv2.cvtColor(line_img,cv2.COLOR_RGB2GRAY)
    rows,cols=gray_line_img.shape
    
    # Set the range, according to its side
    if side == 0:
        col_range = range(int(cols/2)) 
    else:
        col_range = range(int(cols/2),cols)
            
    line_top = np.empty(2) 
    line_bot = np.empty(2)
    
    max_y = 0
    min_y = 539
    max_x = 0
    min_x = 959
    top = np.empty(2) 
    bot = np.empty(2)
    left = np.empty(2)
    right = np.empty(2)
    
    for y in range(rows):
        for x in col_range:
            if gray_line_img[y,x]>70:             
                if y<min_y:
                 min_y = y
                 top[0] = x
                 top[1] = y
                if y>max_y:
                 max_y = y
                 bot[0] = x
                 bot[1] = y
                if x<min_x:
                 min_x = x
                 left[0] = x
                 left[1] = y
                if x>max_x:
                 max_x = x
                 right[0] = x
                 right[1] = y
                 
    # Find end points, according to its side
    if side == 0:             
        line_top[0] = int((top[0] + right[0])/2)
        line_top[1] = int((top[1] + right[1])/2)
        line_bot[0] = int((bot[0] + left[0])/2)
        line_bot[1] = int((bot[1] + left[1])/2)
    else:
        line_top[0] = int((top[0] + left[0])/2)
        line_top[1] = int((top[1] + left[1])/2)
        line_bot[0] = int((bot[0] + right[0])/2)
        line_bot[1] = int((bot[1] + right[1])/2)
    
    extend_img = draw_extend_line(extend_img, line_top[0], line_top[1], line_bot[0], line_bot[1])
    return extend_img
    
def process_image(image):       
    # Grayscale image
    gray = grayscale(image)
    
    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 3
    blur_gray = gaussian_blur(gray, kernel_size)
    
    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    
    # Defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(450, 350), (490, 350), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_img = region_of_interest(edges, vertices)
    
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 20     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments
    extend_image = np.copy(image)*0 # creating a blank to draw lines on
    
    # Run Hough on edge detected image
    line_image = hough_lines(masked_img, rho, theta, threshold, min_line_length, max_line_gap)
    
    # Use extend two land lines 
    extend_image = extend_lines(extend_image, line_image, 0) #Extend left line
    extend_image = extend_lines(extend_image, line_image, 1) #Extend right line
                
    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges)) 
    
    # Draw the lines on the edge image
    lines_edges = weighted_img(extend_image, color_edges)
    return lines_edges

images = os.listdir("test_images/")
for img_file in images:
    if img_file[0:10] == 'Proccessed':
        continue
    
    image = mpimg.imread('test_images/'+img_file)   
    proccessed = process_image(image)
    plt.imshow(proccessed)
    mpimg.imsave('test_images/Proccessed-' + img_file, proccessed)