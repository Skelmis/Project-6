import os
from pathlib import Path

import cv2
import numpy as np

upper_cwd = str(Path(__file__).parents[1])
cwd = str(Path(__file__).parents[0])

min_face_confidence = 0.5
counter = 0


def image_to_blob(image: np.ndarray):
    """
    Given an image, return its blob equivalent
    """
    return cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)))


net: cv2.dnn_Net = cv2.dnn.readNetFromCaffe(
    os.path.join(upper_cwd, "caffe", "deploy.prototxt"),
    os.path.join(upper_cwd, "caffe", "opencv_face_detector.caffemodel"),
)


def find_face(image: np.ndarray, file_type: str):
    """Entry to this class for predictions"""
    try:
        h, w = image.shape[:2]
    except:
        return False

    blob = image_to_blob(image)
    net.setInput(blob)

    detections = net.forward()
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > min_face_confidence:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            try:
                cropped_image = image[
                    (startY - 100) : (endY + 100), (startX - 100) : (endX + 100)
                ]
                global counter
                cv2.imwrite(
                    os.path.join(cwd, "output", f"{counter}.{file_type}"), cropped_image
                )
                counter += 1
            except:
                continue


def main():
    with os.scandir(os.path.join(cwd, "input")) as scanner:
        for photo_path in scanner:
            path: str = photo_path.path
            print(path)
            if path.endswith("png"):
                file_type = "png"
            else:
                file_type = "jpeg"

            photo = cv2.imread(path)
            find_face(photo, file_type)


if __name__ == "__main__":
    main()
