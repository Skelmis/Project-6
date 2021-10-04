import os

import numpy as np
import cv2


class FacialRecognition:
    def __init__(self, bot, cv):
        self.bot = bot
        self.cv = cv

        self.net: cv2.dnn_Net = cv2.dnn.readNetFromCaffe(
            os.path.join(cv.cwd, "caffe", "deploy.prototxt"),
            os.path.join(cv.cwd, "caffe", "opencv_face_detector.caffemodel"),
        )

    def image_to_blob(self, image: np.ndarray):
        """
        Given an image, return its blob equivalent
        """
        return cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)))

    def predict(self, image: np.ndarray):
        """Entry to this class for predictions"""
        h, w = image.shape[:2]

        blob = self.image_to_blob(image)
        self.net.setInput(blob)

        detections = self.net.forward()

        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the bounding box of the face along with the associated
                # probability
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(
                    image,
                    text,
                    (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    2,
                )
