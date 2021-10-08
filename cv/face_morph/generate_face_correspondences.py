# First step
import os
from pathlib import Path

import cv2
import dlib
import numpy as np


def crop_image(image: np.ndarray):
    """
    Given an image, return its blob equivalent
    """
    return cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)))


def generate_face_correspondences(theImage1, theImage2):
    # Detect the points of face.
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("code/utils/shape_predictor_68_face_landmarks.dat")
    corresp = np.zeros((68, 2))

    imgList = [crop_image(theImage1), crop_image(theImage2)]
    list1 = []
    list2 = []
    j = 1

    for img in imgList:

        size = (img.shape[0], img.shape[1])
        if j == 1:
            currList = list1
        else:
            currList = list2

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.

        dets = detector(img, 1)

        try:
            if len(dets) == 0:
                raise NotImplementedError("Lol easy")
        except NotImplementedError:
            print("Sorry, but I couldn't find a face in the image.")

        j = j + 1

        for k, rect in enumerate(dets):

            # Get the landmarks/parts for the face in rect.
            shape = predictor(img, rect)
            # corresp = face_utils.shape_to_np(shape)

            for i in range(0, 68):
                x = shape.part(i).x
                y = shape.part(i).y
                currList.append((x, y))
                corresp[i][0] += x
                corresp[i][1] += y
                # cv2.circle(img, (x, y), 2, (0, 255, 0), 2)

            # Add back the background
            currList.append((1, 1))
            currList.append((size[1] - 1, 1))
            currList.append(((size[1] - 1) // 2, 1))
            currList.append((1, size[0] - 1))
            currList.append((1, (size[0] - 1) // 2))
            currList.append(((size[1] - 1) // 2, size[0] - 1))
            currList.append((size[1] - 1, size[0] - 1))
            currList.append(((size[1] - 1) // 2, (size[0] - 1) // 2))

        cv2.imwrite("test.png", img)

    # Add back the background
    narray = corresp / 2
    narray = np.append(narray, [[1, 1]], axis=0)
    narray = np.append(narray, [[size[1] - 1, 1]], axis=0)
    narray = np.append(narray, [[(size[1] - 1) // 2, 1]], axis=0)
    narray = np.append(narray, [[1, size[0] - 1]], axis=0)
    narray = np.append(narray, [[1, (size[0] - 1) // 2]], axis=0)
    narray = np.append(narray, [[(size[1] - 1) // 2, size[0] - 1]], axis=0)
    narray = np.append(narray, [[size[1] - 1, size[0] - 1]], axis=0)
    narray = np.append(narray, [[(size[1] - 1) // 2, (size[0] - 1) // 2]], axis=0)

    return [size, imgList[0], imgList[1], list1, list2, narray]


# Stuff
cwd = str(Path(__file__).parents[0])

image1 = cv2.imread(os.path.join(cwd, "25.jpeg"))
image2 = cv2.imread(os.path.join(cwd, "26.jpeg"))

generate_face_correspondences(image1, image2)
