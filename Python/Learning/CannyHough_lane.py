import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


image = cv.imread('C:/Users/rsidhan1/PycharmProjects/Ninja/Udacity_/lane_images/exit-ramp.jpg')
cv.namedWindow('image')
img_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
kernel_size = 3
img_gray_blur = cv.GaussianBlur(img_gray, (kernel_size, kernel_size), 0)
low_threshold =50
high_threshold =150
img_edge = cv.Canny(img_gray_blur, low_threshold, high_threshold)

mask = np.zeros_like(img_edge)
ignore_mask_color = 255
imshape = image.shape
vertices = np.array([[(0, imshape[0]), (450, 290), (490, 290), (imshape[1], imshape[0])]], dtype=np.int32)
cv.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv.bitwise_and(img_edge, mask)

rho =2
theta = np.pi/180
threshold = 15
min_line_length = 40
max_lin_gap =20
line_image = np.copy(image)*0

lines = cv.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_lin_gap)
for line in lines:
    for x1, y1, x2, y2 in line:
        cv.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)

color_edges = np.dstack((img_edge, img_edge, img_edge))
lines_edges = cv.addWeighted(color_edges, 0.8, line_image, 1, 0)


cv.imshow('image', lines_edges)
cv.waitKey(0)
cv.destroyAllWindows()
