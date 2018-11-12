#!/usr/bin/python

# Standard imports
import cv2
import numpy as np;

"""
==========================================================================
                            Preprocess Image
==========================================================================
"""

# Read image
#img = cv2.imread("./test.JPG", cv2.IMREAD_COLOR)
img = cv2.imread("./test.JPG", 0)

# blur image to remove noises from targets
blurred = cv2.GaussianBlur(img, (11, 11), 0)

# applying edge detection
#edged = cv2.Canny(blurred, 30, 150)

# threshold the image by setting all pixel values less than 225
# to 255 (white; foreground) and all pixel values >= 225 to 255
# (black; background), thereby segmenting the image
#thresh = cv2.threshold(edged, 225, 255, cv2.THRESH_BINARY_INV)[1]
thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY_INV)[1]


#cv2.imwrite( "./blurred.jpg", blurred);
#cv2.imwrite( "./edged.jpg", edged);
cv2.imwrite( "./thresh.jpg", thresh);

# set prepared image
prepedImg = thresh
# ------------------------------------------------------------------------


"""
==========================================================================
                               Crop Target
==========================================================================
"""



# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200


# Filter by Area.
params.filterByArea = False
#params.minArea = 200
#params.maxArea = 500


# Filter by Circularity
params.filterByCircularity = False
#params.minCircularity = 0.1
#params.maxCircularity = 3

# Filter by Convexity
params.filterByConvexity = False
#params.minConvexity = 1
#params.maxConvexity = 3

# Filter by Inertia
params.filterByInertia = False
#params.minInertiaRatio = 0.01
#params.maxInertiaRatio = 1



# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else :
	detector = cv2.SimpleBlobDetector_create(params)


# Detect blobs.
keypoints = detector.detect(prepedImg)

print(type(keypoints))
print(keypoints)
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(prepedImg, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# Show blobs
#cv2.imshow("Keypoints", im_with_keypoints)
#cv2.waitKey(0)



target_crop = []

# set range of the crop
r = 50

for i in range(len(keypoints)):
    x = int(keypoints[i].pt[0])
    y = int(keypoints[i].pt[1])
#    x, y, w, h = [ v for v in k ]
    cv2.rectangle(img, (x-r,y-r), (x+r, y+r), (255,0,0), 3)
    # Define the region of interest in the image  
    target_crop.append(img[y-r:y+r, x-r:x+r])

for i in range(len(target_crop)):
    cv2.imwrite('./target-'+str(i)+'.jpg', target_crop[i])


