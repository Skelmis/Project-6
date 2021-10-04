import os
import pickle

import imutils
import numpy as np
import cv2

from cv.facial_recognition.recognize_data import RecognizeReturn


class FacialRecognition:
    def __init__(self, bot, cv):
        self.bot = bot
        self.cv = cv

        self.cwd = os.path.join(cv.cwd, "facial_recognition")

        self.min_face_confidence: float = 0.5

    def image_to_blob(self, image: np.ndarray):
        """
        Given an image, return its blob equivalent
        """
        return cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)))

    def find_face(self, image: np.ndarray):
        """Entry to this class for predictions"""
        h, w = image.shape[:2]

        net: cv2.dnn_Net = cv2.dnn.readNetFromCaffe(
            os.path.join(self.cwd, "caffe", "deploy.prototxt"),
            os.path.join(self.cwd, "caffe", "opencv_face_detector.caffemodel"),
        )

        blob = self.image_to_blob(image)
        net.setInput(blob)

        detections = net.forward()

        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > self.min_face_confidence:
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

    def recognize(self, image: np.ndarray) -> RecognizeReturn:
        """Recognize a face from an image

        Returns
        -------
        np.ndarray
            The modified image
        str
            The name of the recognized person
        """
        args = {
            "embedding_model": os.path.join(self.cwd, "openface.nn4.small2.v1.t7"),
            "recognizer": os.path.join(self.cwd, "output", "recognizer.pickle"),
            "le": os.path.join(self.cwd, "output", "le.pickle"),
        }

        # load our serialized face detector from disk
        print("[INFO] loading face detector...")
        detector: cv2.dnn_Net = cv2.dnn.readNetFromCaffe(
            os.path.join(self.cwd, "caffe", "deploy.prototxt"),
            os.path.join(self.cwd, "caffe", "opencv_face_detector.caffemodel"),
        )
        # load our serialized face embedding model from disk
        print("[INFO] loading face recognizer...")
        embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])
        # load the actual face recognition model along with the label encoder
        recognizer = pickle.loads(open(args["recognizer"], "rb").read())
        le = pickle.loads(open(args["le"], "rb").read())

        # load the image, resize it to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image dimensions
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]
        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
            swapRB=False,
            crop=False,
        )
        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        return_value = RecognizeReturn(image=image)

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections
            if confidence > self.min_face_confidence:
                # compute the (x, y)-coordinates of the bounding box for the
                # face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # extract the face ROI
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(
                    face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False
                )
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                # perform classification to recognize the face
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]

                return_value.name = name
                return_value.top_left_x = startX
                return_value.top_left_y = startY
                return_value.bottom_right_x = endX
                return_value.bottom_right_y = endY

                # draw the bounding box of the face along with the associated
                # probability
                # text = "{}: {:.2f}%".format(name, proba * 100)
                # y = startY - 10 if startY - 10 > 10 else startY + 10
                # cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                # cv2.putText(
                #     image,
                #     text,
                #     (startX, y),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.45,
                #     (0, 0, 255),
                #     2,
                # )

        return return_value
