import imutils as imutils
import numpy as np
import itertools
import logging
import os
from pathlib import Path
from typing import Optional, Union, Tuple
from skimage.metrics import structural_similarity as compare_ssim

import cv2

from bot_base import BotBase
from .box_colours import BoxColors

log = logging.getLogger(__name__)


def inject_cam(f):
    def wrapper(*args):
        """Reset video capture when not in use"""
        self = args[0]
        self.feed = cv2.VideoCapture(0)

        values = f(*args)

        self.release()
        self.feed = None

        return values

    return wrapper


class CV:
    def __init__(self, bot, *, min_width: int = 50, min_height: int = 50):
        self.bot: BotBase = bot
        self.feed = None
        self.cwd = str(Path(__file__).parents[0])

        self.image_suffix = itertools.count().__next__

        self.min_width = min_width
        self.min_height = min_height

    @inject_cam
    def take_picture(self):
        """Takes a picture and saves it

        This runs as a background thread to keep this shit running, idk

        Returns
        -------
        The grabbed image
        """
        ret, frame = self.feed.read()
        if not ret:
            log.warning("No video feed active to capture from.")
            return None

        return frame

    def save_picture(self, frame, image_name: str = None) -> Optional[str]:
        """
        Saves a given picture locally

        Parameters
        ----------
        frame
        image_name

        Returns
        -------

        """
        image_name = image_name or f"image_{self.image_suffix()}"
        image_name += ".png"

        path = os.path.join(self.cwd, "pictures", image_name)
        cv2.imwrite(path, frame)

        return path

    def compare_images(
        self, image_one, image_two
    ) -> (Union[float, np.ndarray], np.ndarray):
        """
        Compares two images and returns the MSSIM or
        mean structural similarity over the image

        Returns
        -------
        Union[float, np.ndarray]
            The mean structural similarity
        np.ndarray
            The full SSIM image for visual purposes

        Notes
        -----
        Comparison is done at a
        grey scale level
        """
        image_one_grey = cv2.cvtColor(image_one, cv2.COLOR_BGR2GRAY)
        image_two_grey = cv2.cvtColor(image_two, cv2.COLOR_BGR2GRAY)

        score, diff = compare_ssim(image_one_grey, image_two_grey, full=True)
        diff: np.ndarray = (diff * 255).astype("uint8")

        # noinspection PyTypeChecker
        return score, diff

    def get_image_differences(self, diff: np.ndarray):
        """
        Visualize the differences between images
        """
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cnts = imutils.grab_contours(cnts)

        return thresh, cnts

    def get_relevant_differences(self, contours):
        """
        Returns the relevant contours based on a set of criteria

        This essentially gets rid of the shitty small changes
        """
        relevant_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w > self.min_width and h > self.min_height:
                relevant_contours.append(contour)

        return relevant_contours

    def drawline(self, img, pt1, pt2, color, thickness=1, style="dotted", gap=15):
        dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
        pts = []
        for i in np.arange(0, dist, gap):
            r = i / dist
            x = int((pt1[0] * (1 - r) + pt2[0] * r) + 0.5)
            y = int((pt1[1] * (1 - r) + pt2[1] * r) + 0.5)
            p = (x, y)
            pts.append(p)

        if style == "dotted":
            for p in pts:
                cv2.circle(img, p, thickness, color, -1)
        else:
            s = pts[0]
            e = pts[0]
            i = 0
            for p in pts:
                s = e
                e = p
                if i % 2 == 1:
                    cv2.line(img, s, e, color, thickness)
                i += 1

    def build_bounding_boxes(self, image, bounding_boxes, color_enum: BoxColors):
        """
        Draws bounding boxes around people
        Parameters
        ----------
        image
        bounding_boxes
        color_enum

        Returns
        -------
        image
            The image with drawn bounding boxes
        """
        for contour in bounding_boxes:
            # compute the bounding box of the contour and then draw the
            # bounding box on both input images to represent where the two
            # images differ
            x, y, w, h = cv2.boundingRect(contour)
            # Draws a solid line smh, see self.drawline calls for dotted
            # cv2.rectangle(image, (x, y), (x + w, y + h), color.get_bgr(), 2)

            # Draw 'bigger' corners
            if color_enum == BoxColors.ANALOG_INTERFACE:
                # Also denoted with yellow points on corner and middle points
                color = BoxColors.get_yellow()
            else:
                color = color_enum.get_bgr()
            # How long to draw points
            size = 4
            x_line_length = int(w / 25)
            y_line_length = int(h / 25)

            # Set corners
            top_left = (x, y)
            bottom_left = (x, y + h)
            top_right = (x + w, y)
            bottom_right = (x + w, y + h)

            # Draw a dotted rectangle
            self.drawline(image, top_left, top_right, color_enum.get_bgr(), size - 2)
            self.drawline(image, top_left, bottom_left, color_enum.get_bgr(), size - 2)
            self.drawline(
                image, top_right, bottom_right, color_enum.get_bgr(), size - 2
            )
            self.drawline(
                image, bottom_left, bottom_right, color_enum.get_bgr(), size - 2
            )

            # Draw corners
            # Top left
            cv2.line(image, top_left, (x, y + y_line_length), color, size)
            cv2.line(image, top_left, (x + x_line_length, y), color, size)

            # Top right
            cv2.line(image, top_right, (top_right[0], y + y_line_length), color, size)
            cv2.line(
                image,
                top_right,
                (top_right[0] - x_line_length, top_right[1]),
                color,
                size,
            )

            # Bottom left
            cv2.line(
                image,
                bottom_left,
                (bottom_left[0], bottom_left[1] - y_line_length),
                color,
                size,
            )
            cv2.line(
                image,
                bottom_left,
                (bottom_left[0] + x_line_length, bottom_left[1]),
                color,
                size,
            )

            # Bottom right
            cv2.line(
                image,
                bottom_right,
                (bottom_right[0], bottom_right[1] - y_line_length),
                color,
                size,
            )
            cv2.line(
                image,
                bottom_right,
                (bottom_right[0] - x_line_length, bottom_right[1]),
                color,
                size,
            )

            # Draw lines in the middle
            middle_x: int = (w // 2) + x
            middle_y: int = (h // 2) + y

            cv2.line(image, (middle_x, y), (middle_x, y + y_line_length), color, size)
            cv2.line(
                image, (middle_x, y + h), (middle_x, y + h - y_line_length), color, size
            )
            cv2.line(image, (x, middle_y), (x + x_line_length, middle_y), color, size)
            cv2.line(
                image, (x + w, middle_y), (x + w - x_line_length, middle_y), color, size
            )

    def release(self) -> None:
        """Cleans up video objects before shutdown"""
        if self.feed:
            self.feed.release()

        # Destroy all the windows
        cv2.destroyAllWindows()
