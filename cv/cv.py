import imutils as imutils
import numpy as np
import itertools
import logging
import os
from pathlib import Path
from typing import Optional, Union
from skimage.metrics import structural_similarity as compare_ssim

import cv2

from bot_base import BotBase

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
    def __init__(self, bot):
        self.bot: BotBase = bot
        self.feed = None
        self.cwd = str(Path(__file__).parents[0])

        self.image_suffix = itertools.count().__next__

        self.current_frame = None

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

    def visualize_image_differences(self, diff: np.ndarray):
        """
        Visualize the differences between images
        """
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cnts = imutils.grab_contours(cnts)

        return thresh, cnts

    def get_biggest_difference(self, contours):
        pass

    def release(self) -> None:
        """Cleans up video objects before shutdown"""
        if self.feed:
            self.feed.release()

        # Destroy all the windows
        cv2.destroyAllWindows()
