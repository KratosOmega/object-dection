#!/usr/bin/python

# Standard imports
import cv2
import numpy as np;
import math

class CvTarget(object):
    def __init__(self):
        self.path = "./test.JPG"


    """
    ==========================================================================
                                 Preprocess Image
    ==========================================================================
    """

    def prepImg(self):
        # Read image
        #img = cv2.imread(self.path, cv2.IMREAD_COLOR)
        img = cv2.imread(self.path, 0)

        # blur image to remove noises from targets
        blurred = cv2.GaussianBlur(img, (11, 11), 0)

        # applying edge detection
        edged = cv2.Canny(blurred, 30, 150)

        # threshold the image by setting all pixel values less than 225
        # to 255 (white; foreground) and all pixel values >= 225 to 255
        # (black; background), thereby segmenting the image
        thresh = cv2.threshold(edged, 225, 255, cv2.THRESH_BINARY_INV)[1]
        #thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY_INV)[1]


        #cv2.imwrite( "./blurred.jpg", blurred);
        #cv2.imwrite( "./edged.jpg", edged);
        cv2.imwrite( "./thresh.jpg", thresh);

        # set prepared image
        prepedImg = thresh

        return prepedImg


    """
    ==========================================================================
                                   Crop Target
    ==========================================================================
    """

    def getRawBlobs(self, prepedImg):
        """
        input:
        prepedImg = img without noise, and ready for blob detection

        output:
        keypoints = raw keypoints that multiple blobs can belong to the same target object
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

        print("================================== keypoints")
        print(len(keypoints))

        return keypoints


    def blobCluster(self, keypoints, dist):
        """
        input:
        keypoints: raw keypoints which multiple blobs belong to same target object
        dist: set keypoints distance that classify blobs that belong to the same target object

        output:
        blobs: avg blobs of same target object to get central coords
        """
        blobs = []
        kpCluster = []
        indTrack = [False] * len(keypoints)

        for i in range(len(keypoints)-1):
            if(indTrack[i] == False):
                indTrack[i] = True

                x = keypoints[i].pt[0]
                y = keypoints[i].pt[1]

                cluster = [x, y, 1]

                for j in range(i+1, len(keypoints)):
                    x_n = keypoints[j].pt[0]
                    y_n = keypoints[j].pt[1]

                    if(math.sqrt(abs(x - x_n) ** 2 + abs(y - y_n) ** 2) < dist):
                        indTrack[j] = True
                        cluster[0] += x_n
                        cluster[1] += y_n
                        cluster[2] += 1

                kpCluster.append(cluster)

        print("================================== indTrack")
        print(indTrack)

        for cluster in kpCluster:
            x = int(cluster[0] / cluster[2])
            y = int(cluster[1] / cluster[2])
            blobs.append([x, y])

        print("================================== blobs")
        print(len(blobs))

        return blobs

    def cropTargets(self, blobs, r):
        """
        input:
        blobs = list of blobs with x & y coords for each blob
        r = range of crop selection

        output:
        void: write cropped image to local
        """
        img = cv2.imread(self.path, cv2.IMREAD_COLOR)

        target_crop = []

        for blob in blobs:
            x = blob[0]
            y = blob[1]

            cv2.rectangle(img, (x-r,y-r), (x+r, y+r), (255,0,0), 3)
            # Define the region of interest in the image
            target_crop.append(img[y-r:y+r, x-r:x+r])

        for i in range(len(target_crop)):
            cv2.imwrite('./target-'+str(i)+'.jpg', target_crop[i])


if __name__ == '__main__':
    crop = CvTarget()
    prepedImg = crop.prepImg()
    keypoints = crop.getRawBlobs(prepedImg)
    blobs = crop.blobCluster(keypoints, 100)
    crop.cropTargets(blobs, 40)


